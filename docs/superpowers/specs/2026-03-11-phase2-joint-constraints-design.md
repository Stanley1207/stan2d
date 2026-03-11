# Stan2D Phase 2 — Joint Constraints Design

**Date:** 2026-03-11
**Status:** Approved
**Scope:** Tasks 13–18

---

## Vision

Phase 2 extends Stan2D's constraint system with four joint types — Hinge, Spring, Distance, and Pulley — targeting ML/RL and world model research. All design decisions preserve Phase 1's three hard constraints:

- **SoA memory layout** — joint data stored in parallel vectors, tensor-compatible
- **Strict determinism** — fixed iteration order, complete warm-start state in Snapshot
- **Zero heap allocation in `step()`** — all buffers pre-reserved at construction

---

## Use Case

Primary: ML/RL environments (robotics control, walking agents, articulated bodies).

RL agents control joints via `set_motor_speed()` / `set_motor_torque()`, observe joint state via `JointStateView` (zero-copy spans), and reset episodes via `save_state()` / `restore_state()` (including warm-start accumulated impulses for MCTS compatibility).

---

## Architecture: Single Union SoA (Approach A)

All joint types share one `JointStorage` with a `JointType` enum tag. Solver dispatches per type in the same iteration loop. Unused fields for a given type are skipped at solve time — no dynamic allocation, no polymorphism.

**Rationale over alternatives:**
- Single `JointHandle` + single `SparseSet` → consistent with `BodyHandle` pattern
- Single `JointSnapshot` → simple save/restore implementation
- Sparse field waste (~32–64 bytes/joint for unused fields) is negligible at `max_joints = 2000`

---

## Module Structure

```
include/stan2d/joints/
├── joint_types.hpp        # JointType enum, JointHandle, JointDef
├── joint_storage.hpp      # JointStorage SoA (all fields)
└── joint_solver.hpp       # prepare / warm_start / solve_joints()

src/stan2d/joints/
└── joint_solver.cpp       # Solver implementation
```

No Phase 1 files are modified except:
- `include/stan2d/world/world.hpp` — new joint API methods
- `include/stan2d/world/world_config.hpp` — `max_joints` field
- `include/stan2d/world/state_view.hpp` — `JointStateView` + `WorldStateView::joints`
- `include/stan2d/world/snapshot.hpp` — `JointSnapshot` + `WorldSnapshot::joints`
- `src/stan2d/world/world.cpp` — `create_joint`, `destroy_joint`, `solve()` extension

---

## Data Types

### JointHandle & JointType

```cpp
struct JointHandle {
    uint32_t index;
    uint32_t generation;
};

enum class JointType : uint8_t {
    Hinge    = 0,
    Spring   = 1,
    Distance = 2,
    Pulley   = 3,
};
```

### JointDef

```cpp
struct JointDef {
    JointType  type       = JointType::Hinge;
    BodyHandle body_a;
    BodyHandle body_b;
    Vec2       anchor_a   = {0.0f, 0.0f};   // body-space
    Vec2       anchor_b   = {0.0f, 0.0f};   // body-space

    // Hinge: limits
    bool  limit_enabled   = false;
    float limit_min       = -glm::pi<float>();
    float limit_max       =  glm::pi<float>();

    // Hinge: motor
    bool  motor_enabled   = false;
    float motor_speed     = 0.0f;    // rad/s
    float motor_torque    = 0.0f;    // N·m max

    // Spring
    float stiffness       = 100.0f;
    float damping         = 1.0f;
    float rest_length     = 0.0f;    // 0 = current distance at creation

    // Distance
    float distance        = 0.0f;    // 0 = current distance at creation

    // Pulley
    Vec2  ground_a        = {0.0f, 0.0f};   // world-space
    Vec2  ground_b        = {0.0f, 0.0f};   // world-space
    float pulley_ratio    = 1.0f;
};
```

### WorldConfig Extension

```cpp
struct WorldConfig {
    uint32_t max_bodies      = 10000;
    uint32_t max_shapes      = 10000;
    uint32_t max_constraints = 5000;
    uint32_t max_contacts    = 20000;
    uint32_t max_joints      = 2000;   // Phase 2
};
```

---

## JointStorage SoA Layout

All vectors reserved to `max_joints` at `World` construction.

```cpp
struct JointStorage {
    // Common (all types)
    std::vector<JointType> types;
    std::vector<uint32_t>  body_a;          // dense index
    std::vector<uint32_t>  body_b;
    std::vector<Vec2>      anchor_a;        // body-space
    std::vector<Vec2>      anchor_b;

    // Hinge: limits (optional, default disabled)
    std::vector<bool>      limit_enabled;
    std::vector<float>     limit_min;
    std::vector<float>     limit_max;
    std::vector<float>     reference_angle;
    std::vector<float>     accumulated_limit_impulse;  // warm-start

    // Hinge/Motor (optional, default disabled)
    std::vector<bool>      motor_enabled;
    std::vector<float>     motor_target_speed;
    std::vector<float>     motor_max_torque;
    std::vector<float>     accumulated_motor_impulse;  // warm-start

    // Spring
    std::vector<float>     spring_stiffness;
    std::vector<float>     spring_damping;
    std::vector<float>     spring_rest_length;

    // Distance
    std::vector<float>     distance_length;

    // Pulley
    std::vector<Vec2>      pulley_ground_a;
    std::vector<Vec2>      pulley_ground_b;
    std::vector<float>     pulley_ratio;
    std::vector<float>     pulley_constant;

    // Common warm-start impulse (linear, all types except Spring)
    std::vector<float>     accumulated_impulse_x;
    std::vector<float>     accumulated_impulse_y;

    uint32_t size = 0;
};
```

---

## World Public API

```cpp
class World {
public:
    // Joint management
    JointHandle create_joint(const JointDef& def);
    void        destroy_joint(JointHandle handle);
    [[nodiscard]] bool is_valid(JointHandle handle) const;

    // Joint queries
    [[nodiscard]] float get_joint_angle(JointHandle handle) const;   // Hinge
    [[nodiscard]] float get_joint_speed(JointHandle handle) const;   // Hinge
    [[nodiscard]] float get_joint_length(JointHandle handle) const;  // Distance/Spring

    // Motor control (RL agent interface)
    void set_motor_speed(JointHandle handle, float speed);
    void set_motor_torque(JointHandle handle, float max_torque);

    // State system
    [[nodiscard]] JointStateView get_joint_state_view() const;
    // save_state / restore_state automatically include joint state
};
```

---

## Solver Integration

### Extended Pipeline

```
World::step(dt)
├── integrate_velocities()
├── broad_phase()
├── narrow_phase()
└── solve()
    ├── prepare_contact_constraints()
    ├── prepare_joint_constraints()     ← new
    ├── warm_start_contacts()
    ├── warm_start_joints()             ← new
    ├── for i in iterations:
    │   ├── solve_contacts()
    │   └── solve_joints()              ← new, same loop
    └── integrate_positions()
```

Contact and joint constraints are solved in the **same iteration loop** for correct interaction (e.g., jointed bodies colliding with each other).

### Per-Type Impulse Strategy

**Hinge:**
- Linear impulse: eliminate relative velocity at anchor point (2D, x/y)
- Limit impulse: when `angle < min` or `angle > max`, angular impulse, one-sided clamp (same as contact normal)
- Motor impulse: `target_speed - current_speed → angular impulse`, clamped by `max_torque * dt`, two-sided

**Distance:**
- Impulse along anchor-to-anchor axis
- Baumgarte bias for position correction (same `slop` + `baumgarte` as contacts)
- Configurable one-sided (cable) or two-sided (rigid rod)

**Spring:**
- `F = -stiffness * (len - rest_len) - damping * v_rel`
- Applied as `impulse = F * dt` each frame, **no warm-start** (soft constraint, recomputed each frame)

**Pulley:**
- Scalar constraint: `len_a + ratio * len_b = constant`
- Constraint velocity: `v_a_proj + ratio * v_b_proj = 0`
- Effective mass: scalar including ratio factor
- Impulse applied along each rope segment direction

### Determinism Guarantees

- Joints solved in dense-index order (fixed, deterministic)
- Spring has no accumulated state (stateless per frame)
- All other warm-start impulses stored in SoA, fully captured in `WorldSnapshot`
- No platform-dependent behavior beyond what Phase 1 already guarantees

---

## State System Extensions

### JointStateView (zero-copy spans)

```cpp
struct JointStateView {
    uint32_t active_joint_count = 0;
    std::span<const uint8_t> types;             // JointType as uint8
    std::span<const float>   angles;            // Hinge current angle (0 for others)
    std::span<const float>   angular_speeds;    // Hinge current angular speed
    std::span<const float>   motor_speeds;      // actual motor speed (0 if disabled)
    std::span<const float>   constraint_forces; // constraint force magnitude per frame
    std::span<const float>   lengths;           // current anchor distance
};

struct WorldStateView {
    // ... existing body fields unchanged ...
    JointStateView joints;   // zero-cost addition
};
```

### JointSnapshot (deep copy for save/restore)

```cpp
struct JointSnapshot {
    // SparseSet state (JointHandle validity across restore)
    std::vector<uint32_t> sparse;
    std::vector<uint32_t> dense_to_sparse;
    std::vector<uint32_t> generations;
    std::vector<uint32_t> free_list;

    // Full SoA copy (all fields from JointStorage)
    // ... mirrors JointStorage field-for-field ...
    uint32_t count = 0;
};

struct WorldSnapshot {
    // ... existing body snapshot unchanged ...
    JointSnapshot joints;
};
```

**Key guarantee:** `restore_state()` fully restores SparseSet state + all warm-start accumulated impulses. Physics is bit-identical after restore — safe for MCTS rollback and episode reset.

---

## Task Breakdown

### Task 13 — Joint Infrastructure
**Deliverables:** `JointHandle`, `JointType`, `JointDef`, `JointStorage` (all fields pre-reserved), `SparseSet` instance for joints, `WorldConfig::max_joints`, `World::create_joint()` / `destroy_joint()` / `is_valid()`.
**Not included:** Any solver logic. `step()` unchanged.
**Tests:** Create/destroy all joint types, handle validity, capacity boundary.

### Task 14 — Hinge Joint Solver
**Deliverables:** `prepare_joint_constraints()` for Hinge (effective mass), `warm_start_joints()` for Hinge, `solve_joints()` for Hinge (point constraint + limit impulse + motor impulse), `get_joint_angle()`, `get_joint_speed()`, `set_motor_speed()`, `set_motor_torque()`. Integrated into `World::solve()`.
**Tests:** Pendulum swing, limit hard stop, motor-driven rotation, warm-start frame consistency.

### Task 15 — Distance Joint Solver
**Deliverables:** Hinge solver extended with Distance case in `solve_joints()`, `get_joint_length()`, Baumgarte bias for Distance.
**Tests:** Fixed-distance pendulum, two-body distance hold, external force resistance.

### Task 16 — Spring Joint Solver
**Deliverables:** Spring case in `solve_joints()` (no warm-start, force = `-k*x - d*v`, applied as impulse), `rest_length = 0` auto-detection at creation.
**Tests:** Oscillation with energy decay, overdamped case, zero-rest-length spring.

### Task 17 — Pulley Joint Solver
**Deliverables:** Pulley case in `solve_joints()` (scalar constraint, ratio, constant computed at creation), `pulley_constant` initialization in `create_joint()`.
**Tests:** Equal-ratio balance, unequal-ratio lift, rope-length conservation precision.

### Task 18 — Joint State System
**Deliverables:** `JointStateView` + `WorldStateView::joints`, `JointSnapshot` + `WorldSnapshot::joints`, `save_state()` / `restore_state()` extension, `TrajectoryRecorder::capture()` extension, `export_state()` JSON/Binary extension (`joints` array).
**Tests:** Save/restore full physics reproduction, warm-start state completeness, trajectory recording with joints, MCTS rollback scenario.

### Task Dependency Graph

```
Task 13 (Infrastructure)
    ├── Task 14 (Hinge)
    ├── Task 15 (Distance)
    ├── Task 16 (Spring)
    └── Task 17 (Pulley)
              └── Task 18 (State System) ← requires 14–17 complete
```

Tasks 14–17 are independent and can be designed in parallel; implementation is sequential (shared solver framework).

---

## Testing Strategy

Each task follows the Phase 1 TDD pattern:
1. Write failing tests (RED)
2. Implement to pass (GREEN)
3. Code review, then commit

Minimum coverage targets:
- Unit tests per joint type: construction, basic constraint behavior, edge cases
- Integration tests: multi-joint scenes (robot arm, pendulum chain)
- State system tests: save/restore determinism, trajectory export round-trip
