> **Note for Claude**: 牢记 `docs/tasks/00_master_plan.md` 中的四大铁律。

## Task 5: BodyStorage + World Basics

**Goal:** SoA body storage, World class with create_body/destroy_body/get_position/get_velocity/set_gravity. Supports two-step (shared ShapeHandle) and one-step (inline ShapeData) body creation.

**Files:**
- Create: `include/stan2d/dynamics/body_storage.hpp`
- Create: `include/stan2d/world/world_config.hpp`
- Create: `include/stan2d/world/world.hpp`
- Create: `src/stan2d/world/world.cpp`
- Create: `tests/unit/test_world_basics.cpp`

**Depends on:** Task 3, Task 4

### Step 1: Write failing tests (RED)

**File:** `tests/unit/test_world_basics.cpp`

```cpp
#include <gtest/gtest.h>
#include <stan2d/world/world.hpp>

using namespace stan2d;

// ── World construction ────────────────────────────────────────────

TEST(World, ConstructWithConfig) {
    World world(WorldConfig{.max_bodies = 100, .max_shapes = 100});
    EXPECT_EQ(world.body_count(), 0u);
}

// ── Body creation (two-step with shared shape) ────────────────────

TEST(World, CreateBodyTwoStep) {
    World world(WorldConfig{.max_bodies = 100, .max_shapes = 100});

    ShapeHandle shape = world.create_shape(CircleShape{.radius = 0.5f});
    BodyHandle body = world.create_body({
        .position = {1.0f, 2.0f},
        .shape    = shape,
        .mass     = 1.0f
    });

    EXPECT_TRUE(world.is_valid(body));
    EXPECT_EQ(world.body_count(), 1u);
}

TEST(World, CreateBodyPosition) {
    World world(WorldConfig{.max_bodies = 100, .max_shapes = 100});

    ShapeHandle shape = world.create_shape(CircleShape{.radius = 1.0f});
    BodyHandle body = world.create_body({
        .position = {3.0f, 7.0f},
        .shape    = shape,
        .mass     = 2.0f
    });

    Vec2 pos = world.get_position(body);
    EXPECT_FLOAT_EQ(pos.x, 3.0f);
    EXPECT_FLOAT_EQ(pos.y, 7.0f);
}

TEST(World, CreateBodyVelocity) {
    World world(WorldConfig{.max_bodies = 100, .max_shapes = 100});

    ShapeHandle shape = world.create_shape(CircleShape{.radius = 1.0f});
    BodyHandle body = world.create_body({
        .position = {0.0f, 0.0f},
        .velocity = {5.0f, -3.0f},
        .shape    = shape,
        .mass     = 1.0f
    });

    Vec2 vel = world.get_velocity(body);
    EXPECT_FLOAT_EQ(vel.x, 5.0f);
    EXPECT_FLOAT_EQ(vel.y, -3.0f);
}

// ── Body creation (one-step with inline shape) ────────────────────

TEST(World, CreateBodyOneStep) {
    World world(WorldConfig{.max_bodies = 100, .max_shapes = 100});

    BodyHandle body = world.create_body({
        .position   = {0.0f, 10.0f},
        .shape_data = CircleShape{.radius = 0.3f},
        .mass       = 2.0f
    });

    EXPECT_TRUE(world.is_valid(body));
    Vec2 pos = world.get_position(body);
    EXPECT_FLOAT_EQ(pos.x, 0.0f);
    EXPECT_FLOAT_EQ(pos.y, 10.0f);
}

// ── Mass properties ───────────────────────────────────────────────

TEST(World, DynamicBodyMassProperties) {
    World world(WorldConfig{.max_bodies = 100, .max_shapes = 100});

    ShapeHandle shape = world.create_shape(CircleShape{.radius = 1.0f});
    BodyHandle body = world.create_body({
        .position = {0.0f, 0.0f},
        .shape    = shape,
        .mass     = 4.0f
    });

    EXPECT_FLOAT_EQ(world.get_mass(body), 4.0f);
    EXPECT_FLOAT_EQ(world.get_inverse_mass(body), 0.25f);
}

TEST(World, StaticBodyHasZeroInverseMass) {
    World world(WorldConfig{.max_bodies = 100, .max_shapes = 100});

    ShapeHandle shape = world.create_shape(CircleShape{.radius = 1.0f});
    BodyHandle body = world.create_body({
        .position  = {0.0f, 0.0f},
        .shape     = shape,
        .body_type = BodyType::Static
    });

    EXPECT_FLOAT_EQ(world.get_inverse_mass(body), 0.0f);
    EXPECT_FLOAT_EQ(world.get_inverse_inertia(body), 0.0f);
}

TEST(World, KinematicBodyHasZeroInverseMass) {
    World world(WorldConfig{.max_bodies = 100, .max_shapes = 100});

    ShapeHandle shape = world.create_shape(CircleShape{.radius = 1.0f});
    BodyHandle body = world.create_body({
        .position  = {0.0f, 0.0f},
        .velocity  = {1.0f, 0.0f},
        .shape     = shape,
        .body_type = BodyType::Kinematic
    });

    EXPECT_FLOAT_EQ(world.get_inverse_mass(body), 0.0f);
}

// ── Body destruction ──────────────────────────────────────────────

TEST(World, DestroyBody) {
    World world(WorldConfig{.max_bodies = 100, .max_shapes = 100});

    ShapeHandle shape = world.create_shape(CircleShape{.radius = 1.0f});
    BodyHandle body = world.create_body({
        .position = {0.0f, 0.0f},
        .shape    = shape,
        .mass     = 1.0f
    });

    world.destroy_body(body);
    EXPECT_FALSE(world.is_valid(body));
    EXPECT_EQ(world.body_count(), 0u);
}

TEST(World, DestroyMiddleBodyMaintainsOthers) {
    World world(WorldConfig{.max_bodies = 100, .max_shapes = 100});

    ShapeHandle shape = world.create_shape(CircleShape{.radius = 1.0f});
    BodyHandle a = world.create_body({.position = {1.0f, 0.0f}, .shape = shape, .mass = 1.0f});
    BodyHandle b = world.create_body({.position = {2.0f, 0.0f}, .shape = shape, .mass = 1.0f});
    BodyHandle c = world.create_body({.position = {3.0f, 0.0f}, .shape = shape, .mass = 1.0f});

    world.destroy_body(b);

    EXPECT_TRUE(world.is_valid(a));
    EXPECT_FALSE(world.is_valid(b));
    EXPECT_TRUE(world.is_valid(c));

    EXPECT_FLOAT_EQ(world.get_position(a).x, 1.0f);
    EXPECT_FLOAT_EQ(world.get_position(c).x, 3.0f);
    EXPECT_EQ(world.body_count(), 2u);
}

// ── Gravity ───────────────────────────────────────────────────────

TEST(World, SetAndGetGravity) {
    World world(WorldConfig{.max_bodies = 100, .max_shapes = 100});

    world.set_gravity({0.0f, -9.81f});
    Vec2 g = world.get_gravity();
    EXPECT_FLOAT_EQ(g.x, 0.0f);
    EXPECT_FLOAT_EQ(g.y, -9.81f);
}

// ── Stale handle ──────────────────────────────────────────────────

TEST(World, StaleHandleIsInvalid) {
    World world(WorldConfig{.max_bodies = 100, .max_shapes = 100});

    ShapeHandle shape = world.create_shape(CircleShape{.radius = 1.0f});
    BodyHandle body = world.create_body({.position = {0.0f, 0.0f}, .shape = shape, .mass = 1.0f});
    world.destroy_body(body);

    // Create new body that reuses the slot
    BodyHandle new_body = world.create_body({.position = {5.0f, 5.0f}, .shape = shape, .mass = 1.0f});

    EXPECT_FALSE(world.is_valid(body));       // stale
    EXPECT_TRUE(world.is_valid(new_body));    // current
}
```

### Step 2: Run tests to verify they fail

Run: `cmake --build build && ctest --test-dir build --output-on-failure`

Expected: FAIL — `fatal error: 'stan2d/world/world.hpp' file not found`

### Step 3: Implement BodyStorage

**File:** `include/stan2d/dynamics/body_storage.hpp`

```cpp
#pragma once

#include <cstdint>
#include <vector>

#include <stan2d/core/handle.hpp>
#include <stan2d/core/math_types.hpp>
#include <stan2d/core/sparse_set.hpp>

namespace stan2d {

enum class BodyType : uint8_t {
    Static,
    Dynamic,
    Kinematic
};

struct BodyStorage {
    // Kinematics
    std::vector<Vec2>        positions;
    std::vector<Vec2>        velocities;
    std::vector<float>       rotations;
    std::vector<float>       angular_velocities;

    // Mass properties
    std::vector<float>       masses;
    std::vector<float>       inverse_masses;
    std::vector<float>       inertias;
    std::vector<float>       inverse_inertias;

    // Force accumulators
    std::vector<Vec2>        forces;
    std::vector<float>       torques;

    // Body classification
    std::vector<BodyType>    body_types;

    // Shape reference
    std::vector<ShapeHandle> shape_ids;

    void reserve(uint32_t capacity) {
        positions.reserve(capacity);
        velocities.reserve(capacity);
        rotations.reserve(capacity);
        angular_velocities.reserve(capacity);
        masses.reserve(capacity);
        inverse_masses.reserve(capacity);
        inertias.reserve(capacity);
        inverse_inertias.reserve(capacity);
        forces.reserve(capacity);
        torques.reserve(capacity);
        body_types.reserve(capacity);
        shape_ids.reserve(capacity);
    }

    void push_back(Vec2 pos, Vec2 vel, float rot, float ang_vel,
                   float mass, float inv_mass, float inertia, float inv_inertia,
                   BodyType type, ShapeHandle shape) {
        positions.push_back(pos);
        velocities.push_back(vel);
        rotations.push_back(rot);
        angular_velocities.push_back(ang_vel);
        masses.push_back(mass);
        inverse_masses.push_back(inv_mass);
        inertias.push_back(inertia);
        inverse_inertias.push_back(inv_inertia);
        forces.push_back({0.0f, 0.0f});
        torques.push_back(0.0f);
        body_types.push_back(type);
        shape_ids.push_back(shape);
    }

    void swap_and_pop(uint32_t dst, uint32_t src) {
        positions[dst]          = positions[src];
        velocities[dst]         = velocities[src];
        rotations[dst]          = rotations[src];
        angular_velocities[dst] = angular_velocities[src];
        masses[dst]             = masses[src];
        inverse_masses[dst]     = inverse_masses[src];
        inertias[dst]           = inertias[src];
        inverse_inertias[dst]   = inverse_inertias[src];
        forces[dst]             = forces[src];
        torques[dst]            = torques[src];
        body_types[dst]         = body_types[src];
        shape_ids[dst]          = shape_ids[src];
    }

    void pop_back() {
        positions.pop_back();
        velocities.pop_back();
        rotations.pop_back();
        angular_velocities.pop_back();
        masses.pop_back();
        inverse_masses.pop_back();
        inertias.pop_back();
        inverse_inertias.pop_back();
        forces.pop_back();
        torques.pop_back();
        body_types.pop_back();
        shape_ids.pop_back();
    }

    [[nodiscard]] uint32_t size() const {
        return static_cast<uint32_t>(positions.size());
    }
};

} // namespace stan2d
```

### Step 4: Implement WorldConfig

**File:** `include/stan2d/world/world_config.hpp`

```cpp
#pragma once

#include <cstdint>

namespace stan2d {

struct WorldConfig {
    uint32_t max_bodies      = 10000;
    uint32_t max_constraints  = 5000;
    uint32_t max_contacts     = 20000;
    uint32_t max_shapes       = 10000;
};

} // namespace stan2d
```

### Step 5: Implement World class (GREEN)

**File:** `include/stan2d/world/world.hpp`

```cpp
#pragma once

#include <optional>

#include <stan2d/core/handle.hpp>
#include <stan2d/core/math_types.hpp>
#include <stan2d/core/shape_registry.hpp>
#include <stan2d/core/shapes.hpp>
#include <stan2d/core/sparse_set.hpp>
#include <stan2d/dynamics/body_storage.hpp>
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

    // ── Gravity ───────────────────────────────────────────────────
    void set_gravity(Vec2 gravity);
    [[nodiscard]] Vec2 get_gravity() const;

    // ── Simulation (placeholder for now) ──────────────────────────
    void step(float dt);

private:
    [[nodiscard]] uint32_t dense_index(BodyHandle handle) const;

    WorldConfig    config_;
    SparseSet      body_handles_;
    BodyStorage    bodies_;
    ShapeRegistry  shape_registry_;
    Vec2           gravity_{0.0f, 0.0f};
};

} // namespace stan2d
```

**File:** `src/stan2d/world/world.cpp`

```cpp
#include <stan2d/world/world.hpp>

#include <cassert>

namespace stan2d {

World::World(const WorldConfig& config)
    : config_(config)
{
    body_handles_.reserve(config.max_bodies);
    bodies_.reserve(config.max_bodies);
    shape_registry_.reserve(config.max_shapes);
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

    return BodyHandle{h.index, h.generation};
}

void World::destroy_body(BodyHandle handle) {
    Handle h{handle.index, handle.generation};
    auto swap = body_handles_.deallocate(h);

    if (swap.has_value()) {
        bodies_.swap_and_pop(swap->removed_dense, swap->moved_from_dense);
    }
    bodies_.pop_back();
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

void World::set_gravity(Vec2 gravity) {
    gravity_ = gravity;
}

Vec2 World::get_gravity() const {
    return gravity_;
}

void World::step(float /*dt*/) {
    // Placeholder — will be implemented in Task 10
}

uint32_t World::dense_index(BodyHandle handle) const {
    Handle h{handle.index, handle.generation};
    return body_handles_.dense_index(h);
}

} // namespace stan2d
```

### Step 6: Run tests to verify they pass

Run: `cmake --build build && ctest --test-dir build --output-on-failure`

Expected: PASS — all World basics tests green

### Step 7: Commit

```bash
git add include/stan2d/dynamics/body_storage.hpp \
        include/stan2d/world/world_config.hpp \
        include/stan2d/world/world.hpp \
        src/stan2d/world/world.cpp \
        tests/unit/test_world_basics.cpp
git commit -m "feat: BodyStorage (SoA), WorldConfig, and World with create/destroy/query"
```

---