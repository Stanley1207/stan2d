# stan2d - 2D Physics Engine Design Document

**Date:** 2026-03-09
**Status:** Approved

## Vision

A pure C++ 2D physics engine designed from the ground up to serve as the foundation for World Model research. The evolution path is: 2D physics engine → 3D physics engine → ML-serving world model.

### ML Integration Goals (phased)

1. **Environment Simulator** — Gym-like `step(action) → (state, reward)` interface for reinforcement learning
2. **Data Generator** — Batch generation of large-scale physics trajectories for training predictive models
3. **Differentiable Physics** — Gradient backpropagation through the physics engine for gradient-based optimization

## Technical Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| C++ Standard | C++20 | `std::span`, concepts, ranges |
| Architecture | Hybrid OOP API + SoA internals | Clean API for users, cache-friendly internals for performance |
| Build System | CMake + Google Test | Moderate engineering, no CI for now |
| Dependencies | vcpkg (glm, nlohmann-json, gtest, SDL2) | Minimal, mature libraries only |
| Rendering | SDL2 debug viz + state export | Debug visualization for development, export for external consumers |
| GPU Strategy | CPU first, GPU interface reserved | Architecture allows future CUDA migration without API changes |
| Python Bindings | After core stabilizes | pybind11, added once API is stable |

## Architecture: Handle + SoA Hybrid

### Design Principle

Users interact through lightweight **Handles** (value types with index + generation). The engine internally stores all data in **Structure of Arrays (SoA)** layout for cache efficiency and ML-friendly memory layout. Users never hold raw pointers to internal data.

### Handle Mechanism

```cpp
struct BodyHandle {
    uint32_t index;      // Sparse set slot index
    uint32_t generation; // Generation counter, invalidates stale handles
};
```

- Creation: allocate sparse slot (from free list or append), increment generation, point to dense tail
- Destruction: Swap-and-Pop in dense arrays, update bidirectional mappings, recycle sparse slot
- Lookup: `sparse[handle.index]` → dense index, validate generation match

### SoA Internal Storage

```cpp
struct BodyStorage {
    // Kinematics
    std::vector<Vec2>        positions;
    std::vector<Vec2>        velocities;
    std::vector<float>       rotations;
    std::vector<float>       angular_velocities;

    // Mass properties
    std::vector<float>       masses;
    std::vector<float>       inverse_masses;
    std::vector<float>       inertias;            // Moment of inertia
    std::vector<float>       inverse_inertias;

    // Force accumulators (written in Apply Forces, cleared after integration)
    std::vector<Vec2>        forces;
    std::vector<float>       torques;

    // Body classification
    std::vector<BodyType>    body_types;           // Static / Dynamic / Kinematic

    // Shape reference
    std::vector<ShapeHandle> shape_ids;  // Reference into ShapeRegistry
};

enum class BodyType : uint8_t {
    Static,     // Infinite mass, never moves (floors, walls)
    Dynamic,    // Fully simulated (default)
    Kinematic   // User-controlled velocity, infinite mass for collision response
};
```

**BodyType semantics:**
- `Static`: `inverse_mass = 0`, `inverse_inertia = 0`, skipped in integration
- `Dynamic`: fully simulated, participates in all pipeline stages
- `Kinematic`: user sets velocity directly, infinite mass in collision response, skipped in force integration

Benefits over AoS:
- Sequential memory access when iterating a single property (cache line utilization)
- Direct mapping to ML tensors (contiguous float arrays)
- GPU-friendly data layout for future migration

### Sparse Set (Handle ↔ Dense Index Mapping)

```cpp
struct SparseSet {
    std::vector<uint32_t> sparse;          // handle.index → dense index
    std::vector<uint32_t> dense_to_sparse; // dense index → handle.index
    std::vector<uint32_t> generations;     // per sparse slot
    std::vector<uint32_t> free_list;       // recycled sparse slots
};
```

Object destruction uses **Swap-and-Pop**: the last dense element is swapped into the deleted slot, maintaining contiguous memory at all times.

### User-Facing OOP API

```cpp
World world(WorldConfig{.max_bodies = 50000, .max_contacts = 100000});
world.set_gravity({0.0f, -9.81f});

// Explicit two-step: create shape first, then reference by handle
// Useful when multiple bodies share the same shape
ShapeHandle circle_shape = world.create_shape(CircleShape{.radius = 0.5f});
BodyHandle ball = world.create_body({
    .position = {0.0f, 10.0f},
    .shape = circle_shape,
    .mass = 1.0f
});

// Convenience one-step: engine internally registers the shape
// Useful for quick prototyping, creates a unique shape per body
BodyHandle ball2 = world.create_body({
    .position = {3.0f, 10.0f},
    .shape_data = CircleShape{.radius = 0.3f},
    .mass = 2.0f
});

world.step(1.0f / 60.0f);
Vec2 pos = world.get_position(ball);
```

### World Class Responsibilities

```
World
├── BodyStorage (SoA storage pool)
├── ConstraintStorage (constraint storage pool)
├── ShapeRegistry (geometry data pool)
├── BroadPhase (spatial partitioning)
├── NarrowPhase (precise collision detection)
├── ConstraintSolver (impulse-based solver)
├── Integrator (time integration)
├── step(dt) — drives the simulation pipeline
└── State view / snapshot / export interfaces
```

## Pre-allocation Strategy: Zero Allocation in step()

```cpp
struct WorldConfig {
    uint32_t max_bodies = 10000;
    uint32_t max_constraints = 5000;
    uint32_t max_contacts = 20000;
    uint32_t max_shapes = 10000;
};
```

- All SoA vectors `reserve(max_xxx)` at World construction
- Temporary buffers (collision pairs, solver workspace) are pre-allocated as member variables
- Broad phase output writes to pre-allocated `contact_buffer`
- Capacity overflow: assertion failure (debug) or silent ignore (release)

## Shape Management: ShapeRegistry

```cpp
using ShapeData = std::variant<CircleShape, PolygonShape, CapsuleShape>;

struct CircleShape {
    float radius;
};

struct PolygonShape {
    uint32_t vertex_count;
    std::array<Vec2, MAX_POLYGON_VERTICES> vertices;  // Fixed upper bound, no heap allocation
    std::array<Vec2, MAX_POLYGON_VERTICES> normals;
};

struct CapsuleShape {
    Vec2 point_a, point_b;
    float radius;
};

struct ShapeRegistry {
    SparseSet handles;
    std::vector<ShapeData> shapes;
    std::vector<AABB>      local_aabbs;  // Pre-computed per shape
};
```

- Body SoA stores only `ShapeHandle`, uniform element size
- Multiple bodies can share a single Shape (memory efficient, common in particle systems)
- Shape lifecycle independent of Body (create templates, then batch-create bodies)
- `MAX_POLYGON_VERTICES = 8`, fixed-size arrays avoid internal heap allocation

## Physics Simulation Pipeline

Each `world.step(dt)` executes the following stages in order:

```
step(dt)
│
├── 1. Apply Forces         — Gravity + user-defined forces on all bodies
├── 2. Integrate Velocities — Symplectic Euler: v += (F/m) * dt
├── 3. Broad Phase          — Spatial partitioning → candidate collision pairs
├── 4. Narrow Phase         — Precise collision → ContactManifolds
├── 5. Solve Constraints    — Sequential Impulse (N iterations)
│   ├── Collision constraints (normal + friction)
│   └── Joint constraints (hinge, spring, distance, etc.)
├── 6. Integrate Positions  — x += v * dt, θ += ω * dt
└── 7. Post Step            — Event callbacks, state snapshot, data export
```

### Integrator: Symplectic Euler (default)

```
v_new = v + (F/m) * dt     // Update velocity first
x_new = x + v_new * dt     // Use new velocity for position
```

- Industry standard for real-time physics (Box2D, Bullet)
- Better energy conservation than explicit Euler
- Extensible to Verlet or RK4 via Integrator interface

### Broad Phase: Dynamic AABB Tree

- Adaptive to varying object counts
- Insert/delete O(log N), query O(N log N) worst case
- Box2D v3's choice, well-suited for dynamic scenes

### Narrow Phase: Shape-Pair Dispatch

```
              Circle    Polygon    Capsule
Circle        CC        CP         CCa
Polygon       CP        PP(GJK)    PCa
Capsule       CCa       PCa        CaCa
```

- Circle vs Circle: distance comparison
- Circle vs Polygon: closest point projection
- Polygon vs Polygon: GJK + EPA (general and robust)

### Contact Manifold

```cpp
struct ContactManifold {
    BodyHandle body_a, body_b;
    Vec2 normal;                          // A→B direction
    uint32_t point_count;                 // 1 or 2
    struct ContactPoint {
        Vec2 position;
        float penetration;
        uint32_t id;                      // For warm starting
    } points[2];
};
```

### Constraint Solver: Sequential Impulse

- Typical iteration count: 8-16
- Warm Starting: use previous frame's impulse as initial guess
- `ContactPoint.id` for cross-frame contact matching
- Coulomb friction model: `|f_t| <= mu * f_n`

## State System: View + Backup Separation

### WorldStateView — True Zero-Copy (O(1))

```cpp
struct WorldStateView {
    float timestamp;
    uint32_t active_body_count;

    std::span<const Vec2>  positions;
    std::span<const Vec2>  velocities;
    std::span<const float> rotations;
    std::span<const float> angular_velocities;
    std::span<const float> masses;
};

WorldStateView world.get_state_view() const;
```

- `std::span` is a non-owning view, zero allocation cost
- Future pybind11 can map directly to NumPy arrays via `py::buffer_info`
- **Lifetime constraint:** view is invalidated after next `step()` or any mutation

### WorldSnapshot — Pre-allocated State Backup

```cpp
struct WorldSnapshot {
    float timestamp;
    uint32_t body_count;

    std::vector<Vec2>     positions;
    std::vector<Vec2>     velocities;
    std::vector<float>    rotations;
    std::vector<float>    angular_velocities;
    std::vector<float>    masses;
    std::vector<float>    inverse_masses;
    std::vector<ShapeHandle> shape_ids;

    // Critical: must save Sparse Set state for Handle validity after restore
    std::vector<uint32_t> sparse_to_dense;
    std::vector<uint32_t> dense_to_sparse;
    std::vector<uint32_t> generations;
    std::vector<uint32_t> free_list;

    // ShapeRegistry state (must be saved to keep ShapeHandle references valid)
    std::vector<ShapeData> shapes;
    std::vector<AABB>      shape_aabbs;
    std::vector<uint32_t>  shape_sparse_to_dense;
    std::vector<uint32_t>  shape_dense_to_sparse;
    std::vector<uint32_t>  shape_generations;
    std::vector<uint32_t>  shape_free_list;
};

// Caller pre-allocates, engine only memcpy's
void world.save_state(WorldSnapshot& out) const;
void world.restore_state(const WorldSnapshot& snapshot);
```

- `save_state` only performs `memcpy` into pre-reserved vectors
- `restore_state` fully restores SoA data + SparseSet mappings
- Debug mode asserts sufficient capacity

### TrajectoryRecorder — Fixed Stride for ML Tensor Alignment

```cpp
struct TrajectoryRecorder {
    TrajectoryRecorder(const World& world, uint32_t max_frames);

    void start();
    void capture();
    void save(const std::string& path, ExportFormat fmt);

private:
    uint32_t max_frames_;
    uint32_t max_bodies_;         // = world_config.max_bodies, fixed stride
    uint32_t current_frame_ = 0;

    // Pre-allocated: max_frames * max_bodies (fixed stride)
    std::vector<Vec2>     all_positions_;       // reshape → [frames, max_bodies, 2]
    std::vector<Vec2>     all_velocities_;      // reshape → [frames, max_bodies, 2]
    std::vector<float>    all_rotations_;       // reshape → [frames, max_bodies]

    std::vector<uint32_t> frame_active_counts_; // reshape → [frames]
};
```

**Fixed Stride Strategy:**
- Regardless of active body count, each frame occupies exactly `max_bodies` slots
- Only active portion is memcpy'd, remaining slots are zero-initialized
- `frame_active_counts_` enables masking in ML pipelines
- Result: perfectly aligned dense tensor, directly `mmap`-able and `reshape`-able

```python
# Future Python consumption (not implemented now)
data = np.memmap("trajectory.bin", dtype=np.float32)
positions = data.reshape(num_frames, max_bodies, 2)
mask = active_counts[:, None] > np.arange(max_bodies)
```

### Serialization Export

```cpp
world.export_state("frame.json", ExportFormat::JSON);    // Debug
world.export_state("frame.bin", ExportFormat::Binary);   // ML training data
```

File I/O operations, not called within `step()`, not subject to zero-allocation constraint.

## Determinism Guarantees

- Fixed timestep `dt` only, no variable stepping
- Fixed floating-point operation order (bit-exact on same platform)
- Seedable PRNG for any randomness
- All `WorldConfig` parameters serializable for exact environment reconstruction
- Compiler constraints:

```cmake
target_compile_options(stan2d PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:
        -fno-fast-math
        -ffp-contract=off
    >
    $<$<CXX_COMPILER_ID:MSVC>:
        /fp:precise
    >
)
```

## Physics Features Roadmap

| Phase | Features |
|-------|----------|
| Phase 1 | Rigid body dynamics, collision detection (circle, polygon, capsule), gravity, friction |
| Phase 2 | Joint constraints (hinge, spring, distance, pulley) |
| Phase 3 | Soft body / particle system |
| Phase 4 | Python bindings (pybind11), Gym-like interface |
| Phase 5 | Batch simulation, GPU acceleration (CUDA) |
| Phase 6 | Differentiable physics support |

## Project Structure

```
stan2d/
├── CMakeLists.txt
├── vcpkg.json
├── src/
│   ├── stan2d/
│   │   ├── core/           # Vec2, Mat2, AABB, SparseSet, memory utilities
│   │   ├── dynamics/       # Rigid body dynamics, forces, integrator
│   │   ├── collision/      # Broad phase (AABB tree), narrow phase (GJK/EPA)
│   │   ├── constraints/    # Joints, springs, constraint solver
│   │   ├── softbody/       # Soft body / particle system (Phase 3)
│   │   ├── world/          # World class, step pipeline, scene management
│   │   └── export/         # State serialization, data export
│   └── debug_renderer/     # SDL2 debug visualization (optional build target)
├── include/
│   └── stan2d/             # Public headers (user-facing OOP API)
├── tests/
│   ├── unit/
│   └── integration/
├── examples/
└── docs/
    └── plans/
```

## Math Types

Core math types are thin wrappers or aliases over `glm`:

```cpp
using Vec2 = glm::vec2;   // glm::vec2 is already a 2-component float vector
using Mat2 = glm::mat2;   // 2x2 rotation/transformation matrix
```

Custom geometry types (not provided by glm):

```cpp
struct AABB {
    Vec2 min;
    Vec2 max;
};
```

## Dependencies (vcpkg)

- `glm` — Vector/matrix math
- `nlohmann-json` — State serialization
- `gtest` — Unit testing
- `SDL2` — Debug rendering (optional)

## Build Targets

- `stan2d` — Static library (core engine)
- `stan2d_debug_renderer` — Optional static library
- `stan2d_tests` — Test executable
- `stan2d_examples` — Example programs
