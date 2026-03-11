> **Note for Claude**: 牢记 `docs/tasks/00_master_plan.md` 中的四大铁律。

## Task 10: World::step() Pipeline Wiring

**Goal:** Wire the full 7-stage simulation pipeline into `World::step(dt)`. Each call executes: Apply Forces → Integrate Velocities → Broad Phase → Narrow Phase → Solve Constraints → Integrate Positions → Post Step. All temporary buffers (collision pairs, manifolds, constraints) are pre-allocated as World members — zero allocation per step.

**Files:**
- Modify: `include/stan2d/world/world.hpp` — add pipeline members and internal methods
- Modify: `src/stan2d/world/world.cpp` — implement `step()` pipeline
- Create: `tests/integration/test_step_pipeline.cpp`

**Depends on:** Task 5, Task 6, Task 7, Task 8, Task 9

### Step 1: Write failing tests (RED)

**File:** `tests/integration/test_step_pipeline.cpp`

```cpp
#include <gtest/gtest.h>
#include <cmath>
#include <stan2d/world/world.hpp>

using namespace stan2d;

// ── Gravity integration through step() ────────────────────────────

TEST(StepPipeline, SingleBallFallsUnderGravity) {
    World world(WorldConfig{.max_bodies = 100, .max_shapes = 100, .max_contacts = 100});
    world.set_gravity({0.0f, -10.0f});

    ShapeHandle shape = world.create_shape(CircleShape{.radius = 0.5f});
    BodyHandle ball = world.create_body({
        .position = {0.0f, 10.0f},
        .shape    = shape,
        .mass     = 1.0f
    });

    float dt = 1.0f / 60.0f;
    world.step(dt);

    Vec2 pos = world.get_position(ball);
    Vec2 vel = world.get_velocity(ball);

    // After one step: vel.y = 0 + (-10) * dt
    EXPECT_NEAR(vel.y, -10.0f * dt, 1e-5f);
    // pos.y = 10 + vel.y * dt (symplectic Euler)
    EXPECT_NEAR(pos.y, 10.0f + vel.y * dt, 1e-5f);
    // x unchanged
    EXPECT_FLOAT_EQ(pos.x, 0.0f);
}

TEST(StepPipeline, MultipleStepsAccumulateGravity) {
    World world(WorldConfig{.max_bodies = 100, .max_shapes = 100, .max_contacts = 100});
    world.set_gravity({0.0f, -10.0f});

    ShapeHandle shape = world.create_shape(CircleShape{.radius = 0.5f});
    BodyHandle ball = world.create_body({
        .position = {0.0f, 100.0f},
        .shape    = shape,
        .mass     = 1.0f
    });

    float dt = 1.0f / 60.0f;
    for (int i = 0; i < 60; ++i) {
        world.step(dt);
    }

    Vec2 vel = world.get_velocity(ball);
    // After 60 steps: vel.y ≈ -10.0 * 1.0 = -10.0
    EXPECT_NEAR(vel.y, -10.0f, 0.01f);

    Vec2 pos = world.get_position(ball);
    // Ball should have fallen significantly
    EXPECT_LT(pos.y, 100.0f);
}

// ── Static bodies don't move ──────────────────────────────────────

TEST(StepPipeline, StaticBodyUnaffectedByGravity) {
    World world(WorldConfig{.max_bodies = 100, .max_shapes = 100, .max_contacts = 100});
    world.set_gravity({0.0f, -10.0f});

    ShapeHandle shape = world.create_shape(CircleShape{.radius = 1.0f});
    BodyHandle floor = world.create_body({
        .position  = {0.0f, 0.0f},
        .shape     = shape,
        .body_type = BodyType::Static
    });

    float dt = 1.0f / 60.0f;
    for (int i = 0; i < 10; ++i) {
        world.step(dt);
    }

    Vec2 pos = world.get_position(floor);
    EXPECT_FLOAT_EQ(pos.x, 0.0f);
    EXPECT_FLOAT_EQ(pos.y, 0.0f);
}

// ── Kinematic bodies move at set velocity ─────────────────────────

TEST(StepPipeline, KinematicBodyMovesAtSetVelocity) {
    World world(WorldConfig{.max_bodies = 100, .max_shapes = 100, .max_contacts = 100});
    world.set_gravity({0.0f, -10.0f});

    ShapeHandle shape = world.create_shape(CircleShape{.radius = 0.5f});
    BodyHandle platform = world.create_body({
        .position  = {0.0f, 5.0f},
        .velocity  = {2.0f, 0.0f},
        .shape     = shape,
        .body_type = BodyType::Kinematic
    });

    float dt = 1.0f / 60.0f;
    world.step(dt);

    Vec2 vel = world.get_velocity(platform);
    // Kinematic: velocity not affected by gravity
    EXPECT_FLOAT_EQ(vel.x, 2.0f);
    EXPECT_FLOAT_EQ(vel.y, 0.0f);

    Vec2 pos = world.get_position(platform);
    // Position updated from velocity
    EXPECT_NEAR(pos.x, 2.0f * dt, 1e-5f);
    EXPECT_NEAR(pos.y, 5.0f, 1e-5f);
}

// ── Collision: ball onto static floor ─────────────────────────────

TEST(StepPipeline, BallCollidesWithStaticFloor) {
    World world(WorldConfig{.max_bodies = 100, .max_shapes = 100, .max_contacts = 200});
    world.set_gravity({0.0f, -10.0f});

    ShapeHandle circle = world.create_shape(CircleShape{.radius = 0.5f});

    // Ball just above floor
    BodyHandle ball = world.create_body({
        .position = {0.0f, 1.0f},
        .velocity = {0.0f, -5.0f},
        .shape    = circle,
        .mass     = 1.0f
    });

    // Wide floor polygon
    PolygonShape floor_shape;
    floor_shape.vertex_count = 4;
    floor_shape.vertices[0] = {-10.0f, -0.5f};
    floor_shape.vertices[1] = { 10.0f, -0.5f};
    floor_shape.vertices[2] = { 10.0f,  0.0f};
    floor_shape.vertices[3] = {-10.0f,  0.0f};
    floor_shape.normals[0] = { 0.0f, -1.0f};
    floor_shape.normals[1] = { 1.0f,  0.0f};
    floor_shape.normals[2] = { 0.0f,  1.0f};
    floor_shape.normals[3] = {-1.0f,  0.0f};

    BodyHandle floor = world.create_body({
        .position  = {0.0f, 0.0f},
        .shape_data = floor_shape,
        .body_type = BodyType::Static
    });

    float dt = 1.0f / 60.0f;
    for (int i = 0; i < 120; ++i) {
        world.step(dt);
    }

    Vec2 pos = world.get_position(ball);
    // Ball should not have fallen below the floor surface
    EXPECT_GE(pos.y, -0.1f);  // Allow small penetration tolerance
}

// ── Two dynamic circles collide ───────────────────────────────────

TEST(StepPipeline, TwoDynamicCirclesCollide) {
    World world(WorldConfig{.max_bodies = 100, .max_shapes = 100, .max_contacts = 200});
    world.set_gravity({0.0f, 0.0f});  // No gravity

    ShapeHandle circle = world.create_shape(CircleShape{.radius = 0.5f});

    BodyHandle a = world.create_body({
        .position = {0.0f, 0.0f},
        .velocity = {5.0f, 0.0f},
        .shape    = circle,
        .mass     = 1.0f
    });

    BodyHandle b = world.create_body({
        .position = {3.0f, 0.0f},
        .velocity = {-5.0f, 0.0f},
        .shape    = circle,
        .mass     = 1.0f
    });

    float dt = 1.0f / 60.0f;
    for (int i = 0; i < 60; ++i) {
        world.step(dt);
    }

    // After collision, bodies should have separated
    Vec2 pos_a = world.get_position(a);
    Vec2 pos_b = world.get_position(b);
    float dist = std::abs(pos_b.x - pos_a.x);

    // Circles should not be overlapping (sum of radii = 1.0)
    EXPECT_GE(dist, 0.9f);
}

// ── Force accumulators cleared each step ──────────────────────────

TEST(StepPipeline, ForceAccumulatorsClearedAfterStep) {
    World world(WorldConfig{.max_bodies = 100, .max_shapes = 100, .max_contacts = 100});
    world.set_gravity({0.0f, 0.0f});

    ShapeHandle shape = world.create_shape(CircleShape{.radius = 0.5f});
    BodyHandle body = world.create_body({
        .position = {0.0f, 0.0f},
        .shape    = shape,
        .mass     = 1.0f
    });

    // Apply force for one step
    world.apply_force(body, {100.0f, 0.0f});
    world.step(1.0f / 60.0f);

    Vec2 vel_after_first = world.get_velocity(body);
    EXPECT_GT(vel_after_first.x, 0.0f);

    // Second step without force — velocity should NOT continue to accelerate
    world.step(1.0f / 60.0f);
    Vec2 vel_after_second = world.get_velocity(body);

    // Velocity same as after first step (no new force applied)
    EXPECT_NEAR(vel_after_second.x, vel_after_first.x, 1e-5f);
}

// ── Mixed body types in pipeline ──────────────────────────────────

TEST(StepPipeline, MixedBodyTypesPipelineRuns) {
    World world(WorldConfig{.max_bodies = 100, .max_shapes = 100, .max_contacts = 200});
    world.set_gravity({0.0f, -10.0f});

    ShapeHandle circle = world.create_shape(CircleShape{.radius = 0.5f});

    BodyHandle dynamic_body = world.create_body({
        .position = {0.0f, 10.0f},
        .shape    = circle,
        .mass     = 1.0f
    });

    BodyHandle static_body = world.create_body({
        .position  = {5.0f, 0.0f},
        .shape     = circle,
        .body_type = BodyType::Static
    });

    BodyHandle kinematic_body = world.create_body({
        .position  = {10.0f, 0.0f},
        .velocity  = {1.0f, 0.0f},
        .shape     = circle,
        .body_type = BodyType::Kinematic
    });

    float dt = 1.0f / 60.0f;
    for (int i = 0; i < 10; ++i) {
        world.step(dt);
    }

    // Dynamic fell
    EXPECT_LT(world.get_position(dynamic_body).y, 10.0f);

    // Static unchanged
    EXPECT_FLOAT_EQ(world.get_position(static_body).x, 5.0f);
    EXPECT_FLOAT_EQ(world.get_position(static_body).y, 0.0f);

    // Kinematic moved horizontally
    EXPECT_GT(world.get_position(kinematic_body).x, 10.0f);
    EXPECT_FLOAT_EQ(world.get_position(kinematic_body).y, 0.0f);
}

// ── Zero dt is a no-op ────────────────────────────────────────────

TEST(StepPipeline, ZeroDtNoOp) {
    World world(WorldConfig{.max_bodies = 100, .max_shapes = 100, .max_contacts = 100});
    world.set_gravity({0.0f, -10.0f});

    ShapeHandle shape = world.create_shape(CircleShape{.radius = 0.5f});
    BodyHandle body = world.create_body({
        .position = {1.0f, 2.0f},
        .velocity = {3.0f, 4.0f},
        .shape    = shape,
        .mass     = 1.0f
    });

    world.step(0.0f);

    Vec2 pos = world.get_position(body);
    EXPECT_FLOAT_EQ(pos.x, 1.0f);
    EXPECT_FLOAT_EQ(pos.y, 2.0f);
}
```

### Step 2: Run tests to verify they fail

Run: `cmake --build build && ctest --test-dir build --output-on-failure`

Expected: FAIL — tests reference `apply_force()` and pipeline logic not yet implemented

### Step 3: Add pipeline members and API to World

**File:** `include/stan2d/world/world.hpp`

Add includes and new members to the existing World class:

```cpp
#pragma once

#include <optional>
#include <vector>

#include <stan2d/collision/aabb_tree.hpp>
#include <stan2d/collision/contact.hpp>
#include <stan2d/collision/narrow_phase.hpp>
#include <stan2d/constraints/contact_constraint.hpp>
#include <stan2d/constraints/solver.hpp>
#include <stan2d/core/handle.hpp>
#include <stan2d/core/math_types.hpp>
#include <stan2d/core/shape_registry.hpp>
#include <stan2d/core/shapes.hpp>
#include <stan2d/core/sparse_set.hpp>
#include <stan2d/dynamics/body_storage.hpp>
#include <stan2d/dynamics/integrator.hpp>
#include <stan2d/world/world_config.hpp>

namespace stan2d {

struct BodyDef {
    Vec2       position   = {0.0f, 0.0f};
    Vec2       velocity   = {0.0f, 0.0f};
    float      rotation   = 0.0f;
    float      angular_velocity = 0.0f;

    // Two-step: provide a pre-created ShapeHandle
    std::optional<ShapeHandle> shape = std::nullopt;

    // One-step: provide inline ShapeData (engine creates shape internally)
    std::optional<ShapeData>   shape_data = std::nullopt;

    float      mass       = 1.0f;
    float      inertia    = 1.0f;
    BodyType   body_type  = BodyType::Dynamic;
};

class World {
public:
    explicit World(const WorldConfig& config);

    // ── Shape management ──────────────────────────────────────────
    ShapeHandle create_shape(const ShapeData& shape);

    // ── Body management ───────────────────────────────────────────
    BodyHandle create_body(const BodyDef& def);
    void       destroy_body(BodyHandle handle);

    // ── Queries ───────────────────────────────────────────────────
    [[nodiscard]] bool is_valid(BodyHandle handle) const;
    [[nodiscard]] Vec2  get_position(BodyHandle handle) const;
    [[nodiscard]] Vec2  get_velocity(BodyHandle handle) const;
    [[nodiscard]] float get_mass(BodyHandle handle) const;
    [[nodiscard]] float get_inverse_mass(BodyHandle handle) const;
    [[nodiscard]] float get_inverse_inertia(BodyHandle handle) const;
    [[nodiscard]] uint32_t body_count() const;

    // ── Force application ─────────────────────────────────────────
    void apply_force(BodyHandle handle, Vec2 force);
    void apply_torque(BodyHandle handle, float torque);

    // ── Gravity ───────────────────────────────────────────────────
    void set_gravity(Vec2 gravity);
    [[nodiscard]] Vec2 get_gravity() const;

    // ── Solver configuration ──────────────────────────────────────
    void set_solver_config(const SolverConfig& config);
    [[nodiscard]] const SolverConfig& get_solver_config() const;

    // ── Simulation ────────────────────────────────────────────────
    void step(float dt);

private:
    [[nodiscard]] uint32_t dense_index(BodyHandle handle) const;

    // ── Pipeline stages ───────────────────────────────────────────
    void apply_gravity();
    void build_aabb_proxies();
    void update_aabb_proxies();
    void broad_phase();
    void narrow_phase();
    void solve();

    // ── Core data ─────────────────────────────────────────────────
    WorldConfig    config_;
    SparseSet      body_handles_;
    BodyStorage    bodies_;
    ShapeRegistry  shape_registry_;
    Vec2           gravity_{0.0f, 0.0f};
    SolverConfig   solver_config_;
    bool           proxies_built_ = false;

    // ── Pre-allocated pipeline buffers ─────────────────────────────
    // Broad phase: AABB tree + proxy mapping
    AABBTree                      aabb_tree_;
    std::vector<int32_t>          body_proxies_;   // dense index → tree proxy

    // Broad phase output
    std::vector<CollisionPair>    collision_pairs_;

    // Narrow phase output
    std::vector<ContactManifold>  manifolds_;

    // Solver workspace
    std::vector<ContactConstraint> constraints_;
};

} // namespace stan2d
```

### Step 4: Implement step() pipeline (GREEN)

**File:** `src/stan2d/world/world.cpp`

```cpp
#include <stan2d/world/world.hpp>

#include <cassert>
#include <cmath>
#include <variant>

namespace stan2d {

World::World(const WorldConfig& config)
    : config_(config)
{
    body_handles_.reserve(config.max_bodies);
    bodies_.reserve(config.max_bodies);
    shape_registry_.reserve(config.max_shapes);

    // Pre-allocate pipeline buffers
    body_proxies_.reserve(config.max_bodies);
    collision_pairs_.reserve(config.max_contacts);
    manifolds_.reserve(config.max_contacts);
    constraints_.reserve(config.max_contacts * 2);  // 2 contact points max per manifold
}

ShapeHandle World::create_shape(const ShapeData& shape) {
    return shape_registry_.create(shape);
}

BodyHandle World::create_body(const BodyDef& def) {
    // Resolve shape handle
    ShapeHandle shape_handle;
    if (def.shape.has_value()) {
        shape_handle = def.shape.value();
    } else if (def.shape_data.has_value()) {
        shape_handle = shape_registry_.create(def.shape_data.value());
    } else {
        assert(false && "BodyDef must provide either 'shape' or 'shape_data'");
    }

    Handle h = body_handles_.allocate();

    // Compute mass properties based on body type
    float mass         = def.mass;
    float inv_mass     = (def.mass > 0.0f) ? (1.0f / def.mass) : 0.0f;
    float inertia      = def.inertia;
    float inv_inertia  = (def.inertia > 0.0f) ? (1.0f / def.inertia) : 0.0f;

    if (def.body_type == BodyType::Static || def.body_type == BodyType::Kinematic) {
        inv_mass    = 0.0f;
        inv_inertia = 0.0f;
    }

    bodies_.push_back(
        def.position, def.velocity, def.rotation, def.angular_velocity,
        mass, inv_mass, inertia, inv_inertia,
        def.body_type, shape_handle
    );

    // Mark that proxies need rebuilding
    proxies_built_ = false;

    return BodyHandle{h.index, h.generation};
}

void World::destroy_body(BodyHandle handle) {
    Handle h{handle.index, handle.generation};
    auto swap = body_handles_.deallocate(h);

    if (swap.has_value()) {
        bodies_.swap_and_pop(swap->removed_dense, swap->moved_from_dense);
    }
    bodies_.pop_back();

    // Mark that proxies need rebuilding
    proxies_built_ = false;
}

bool World::is_valid(BodyHandle handle) const {
    return body_handles_.is_valid(Handle{handle.index, handle.generation});
}

Vec2 World::get_position(BodyHandle handle) const {
    return bodies_.positions[dense_index(handle)];
}

Vec2 World::get_velocity(BodyHandle handle) const {
    return bodies_.velocities[dense_index(handle)];
}

float World::get_mass(BodyHandle handle) const {
    return bodies_.masses[dense_index(handle)];
}

float World::get_inverse_mass(BodyHandle handle) const {
    return bodies_.inverse_masses[dense_index(handle)];
}

float World::get_inverse_inertia(BodyHandle handle) const {
    return bodies_.inverse_inertias[dense_index(handle)];
}

uint32_t World::body_count() const {
    return body_handles_.size();
}

void World::apply_force(BodyHandle handle, Vec2 force) {
    uint32_t idx = dense_index(handle);
    bodies_.forces[idx] = bodies_.forces[idx] + force;
}

void World::apply_torque(BodyHandle handle, float torque) {
    uint32_t idx = dense_index(handle);
    bodies_.torques[idx] += torque;
}

void World::set_gravity(Vec2 gravity) {
    gravity_ = gravity;
}

Vec2 World::get_gravity() const {
    return gravity_;
}

void World::set_solver_config(const SolverConfig& config) {
    solver_config_ = config;
}

const SolverConfig& World::get_solver_config() const {
    return solver_config_;
}

void World::step(float dt) {
    if (dt <= 0.0f) return;

    uint32_t count = bodies_.size();
    if (count == 0) return;

    // Stage 1 & 2: Apply forces (gravity) + Integrate velocities
    integrate_velocities(bodies_, count, gravity_, dt);

    // Stage 3: Broad phase — update AABBs and find collision pairs
    broad_phase();

    // Stage 4: Narrow phase — precise collision detection
    narrow_phase();

    // Stage 5: Solve constraints
    solve();

    // Stage 6: Integrate positions
    integrate_positions(bodies_, count, dt);
}

uint32_t World::dense_index(BodyHandle handle) const {
    Handle h{handle.index, handle.generation};
    return body_handles_.dense_index(h);
}

// ── Pipeline stage implementations ────────────────────────────────

void World::build_aabb_proxies() {
    aabb_tree_ = AABBTree{};  // Reset tree
    body_proxies_.clear();
    body_proxies_.resize(bodies_.size(), AABBTree::NULL_NODE);

    for (uint32_t i = 0; i < bodies_.size(); ++i) {
        ShapeHandle sh = bodies_.shape_ids[i];
        AABB local_aabb = shape_registry_.get_local_aabb(sh);

        // Transform AABB to world space (translation only for now)
        Vec2 pos = bodies_.positions[i];
        AABB world_aabb{
            {local_aabb.min.x + pos.x, local_aabb.min.y + pos.y},
            {local_aabb.max.x + pos.x, local_aabb.max.y + pos.y}
        };

        body_proxies_[i] = aabb_tree_.insert(world_aabb, i);
    }

    proxies_built_ = true;
}

void World::update_aabb_proxies() {
    for (uint32_t i = 0; i < bodies_.size(); ++i) {
        if (bodies_.body_types[i] == BodyType::Static) continue;

        ShapeHandle sh = bodies_.shape_ids[i];
        AABB local_aabb = shape_registry_.get_local_aabb(sh);

        Vec2 pos = bodies_.positions[i];
        AABB world_aabb{
            {local_aabb.min.x + pos.x, local_aabb.min.y + pos.y},
            {local_aabb.max.x + pos.x, local_aabb.max.y + pos.y}
        };

        aabb_tree_.update(body_proxies_[i], world_aabb);
    }
}

void World::broad_phase() {
    if (!proxies_built_) {
        build_aabb_proxies();
    } else {
        update_aabb_proxies();
    }

    collision_pairs_.clear();
    aabb_tree_.query_pairs(collision_pairs_);
}

void World::narrow_phase() {
    manifolds_.clear();

    for (const auto& pair : collision_pairs_) {
        uint32_t dense_a = pair.user_data_a;
        uint32_t dense_b = pair.user_data_b;

        // Skip static-static pairs
        if (bodies_.body_types[dense_a] == BodyType::Static &&
            bodies_.body_types[dense_b] == BodyType::Static) {
            continue;
        }

        ShapeHandle shape_a = bodies_.shape_ids[dense_a];
        ShapeHandle shape_b = bodies_.shape_ids[dense_b];

        const ShapeData& data_a = shape_registry_.get_shape(shape_a);
        const ShapeData& data_b = shape_registry_.get_shape(shape_b);

        ContactManifold manifold;
        bool colliding = detect_collision(
            data_a, bodies_.positions[dense_a], bodies_.rotations[dense_a],
            data_b, bodies_.positions[dense_b], bodies_.rotations[dense_b],
            manifold
        );

        if (colliding && manifold.point_count > 0) {
            manifold.body_a = BodyHandle{};  // Not used by solver (uses dense indices)
            manifold.body_b = BodyHandle{};
            // Store dense indices in a way the solver can use
            // We pass them directly to prepare_contact_constraints
            manifolds_.push_back(manifold);

            // Immediately prepare constraints for this manifold
            prepare_contact_constraints(manifold, dense_a, dense_b, bodies_, constraints_);
        }
    }
}

void World::solve() {
    constraints_.clear();

    // Re-generate constraints from manifolds
    for (size_t m = 0; m < manifolds_.size(); ++m) {
        uint32_t dense_a = collision_pairs_[m].user_data_a;
        uint32_t dense_b = collision_pairs_[m].user_data_b;

        // Filter static-static (redundant with narrow_phase, but safe)
        if (bodies_.body_types[dense_a] == BodyType::Static &&
            bodies_.body_types[dense_b] == BodyType::Static) {
            continue;
        }

        prepare_contact_constraints(manifolds_[m], dense_a, dense_b, bodies_, constraints_);
    }

    if (constraints_.empty()) return;

    warm_start(constraints_, bodies_);
    solve_constraints(constraints_, bodies_, solver_config_);
}

} // namespace stan2d
```

### Step 5: Add `apply_force`, `apply_torque`, `get_local_aabb`, and `get_shape` if missing

The following accessors are needed on `ShapeRegistry` (add if not present):

**File:** `include/stan2d/core/shape_registry.hpp` — ensure these methods exist:

```cpp
[[nodiscard]] const AABB& get_local_aabb(ShapeHandle handle) const {
    uint32_t dense = handles.dense_index(Handle{handle.index, handle.generation});
    return local_aabbs[dense];
}

[[nodiscard]] const ShapeData& get_shape(ShapeHandle handle) const {
    uint32_t dense = handles.dense_index(Handle{handle.index, handle.generation});
    return shapes[dense];
}
```

### Step 6: Run tests to verify they pass

Run: `cmake --build build && ctest --test-dir build --output-on-failure`

Expected: PASS — all pipeline integration tests green

### Step 7: Commit

```bash
git add include/stan2d/world/world.hpp \
        src/stan2d/world/world.cpp \
        include/stan2d/core/shape_registry.hpp \
        tests/integration/test_step_pipeline.cpp
git commit -m "feat: World::step() full 7-stage pipeline with pre-allocated buffers"
```

---