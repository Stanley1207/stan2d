> **Note for Claude**: 牢记 `docs/tasks/00_master_plan.md` 中的四大铁律。

## Task 11: State System (View + Snapshot)

**Goal:** WorldStateView (zero-copy `std::span` views into SoA data), WorldSnapshot (deep-copy state backup), and save_state/restore_state for deterministic replay. Snapshot includes full SparseSet state so Handles remain valid after restore.

**Files:**
- Create: `include/stan2d/world/state_view.hpp`
- Create: `include/stan2d/world/snapshot.hpp`
- Modify: `include/stan2d/world/world.hpp` — add state API
- Modify: `src/stan2d/world/world.cpp` — implement state methods
- Create: `tests/unit/test_state_system.cpp`

**Depends on:** Task 10

### Step 1: Write failing tests (RED)

**File:** `tests/unit/test_state_system.cpp`

```cpp
#include <gtest/gtest.h>
#include <stan2d/world/world.hpp>
#include <stan2d/world/state_view.hpp>
#include <stan2d/world/snapshot.hpp>

using namespace stan2d;

// ── WorldStateView ────────────────────────────────────────────────

TEST(StateView, ViewReflectsCurrentState) {
    World world(WorldConfig{.max_bodies = 100, .max_shapes = 100, .max_contacts = 100});

    ShapeHandle shape = world.create_shape(CircleShape{.radius = 0.5f});
    world.create_body({.position = {1.0f, 2.0f}, .shape = shape, .mass = 1.0f});
    world.create_body({.position = {3.0f, 4.0f}, .shape = shape, .mass = 2.0f});

    WorldStateView view = world.get_state_view();

    EXPECT_EQ(view.active_body_count, 2u);
    EXPECT_EQ(view.positions.size(), 2u);
    EXPECT_EQ(view.velocities.size(), 2u);
    EXPECT_EQ(view.rotations.size(), 2u);
    EXPECT_EQ(view.masses.size(), 2u);
}

TEST(StateView, ViewDataMatchesBodies) {
    World world(WorldConfig{.max_bodies = 100, .max_shapes = 100, .max_contacts = 100});

    ShapeHandle shape = world.create_shape(CircleShape{.radius = 0.5f});
    BodyHandle a = world.create_body({
        .position = {1.0f, 2.0f},
        .velocity = {3.0f, 4.0f},
        .shape    = shape,
        .mass     = 5.0f
    });

    WorldStateView view = world.get_state_view();

    EXPECT_FLOAT_EQ(view.positions[0].x, 1.0f);
    EXPECT_FLOAT_EQ(view.positions[0].y, 2.0f);
    EXPECT_FLOAT_EQ(view.velocities[0].x, 3.0f);
    EXPECT_FLOAT_EQ(view.velocities[0].y, 4.0f);
    EXPECT_FLOAT_EQ(view.masses[0], 5.0f);
}

TEST(StateView, EmptyWorldView) {
    World world(WorldConfig{.max_bodies = 100, .max_shapes = 100, .max_contacts = 100});

    WorldStateView view = world.get_state_view();

    EXPECT_EQ(view.active_body_count, 0u);
    EXPECT_TRUE(view.positions.empty());
}

// ── WorldSnapshot: save and restore ───────────────────────────────

TEST(Snapshot, SaveAndRestorePositions) {
    World world(WorldConfig{.max_bodies = 100, .max_shapes = 100, .max_contacts = 100});
    world.set_gravity({0.0f, -10.0f});

    ShapeHandle shape = world.create_shape(CircleShape{.radius = 0.5f});
    BodyHandle ball = world.create_body({
        .position = {0.0f, 10.0f},
        .shape    = shape,
        .mass     = 1.0f
    });

    // Save state before simulation
    WorldSnapshot snapshot;
    world.save_state(snapshot);

    // Simulate several steps
    float dt = 1.0f / 60.0f;
    for (int i = 0; i < 60; ++i) {
        world.step(dt);
    }

    // Ball should have fallen
    EXPECT_LT(world.get_position(ball).y, 10.0f);

    // Restore state
    world.restore_state(snapshot);

    // Ball should be back at original position
    Vec2 pos = world.get_position(ball);
    EXPECT_FLOAT_EQ(pos.x, 0.0f);
    EXPECT_FLOAT_EQ(pos.y, 10.0f);
}

TEST(Snapshot, RestorePreservesHandleValidity) {
    World world(WorldConfig{.max_bodies = 100, .max_shapes = 100, .max_contacts = 100});

    ShapeHandle shape = world.create_shape(CircleShape{.radius = 0.5f});
    BodyHandle a = world.create_body({.position = {1.0f, 0.0f}, .shape = shape, .mass = 1.0f});
    BodyHandle b = world.create_body({.position = {2.0f, 0.0f}, .shape = shape, .mass = 1.0f});

    WorldSnapshot snapshot;
    world.save_state(snapshot);

    // Destroy body b and create c
    world.destroy_body(b);
    BodyHandle c = world.create_body({.position = {3.0f, 0.0f}, .shape = shape, .mass = 1.0f});

    // Restore — b should be valid again, c should NOT be
    world.restore_state(snapshot);

    EXPECT_TRUE(world.is_valid(a));
    EXPECT_TRUE(world.is_valid(b));
    EXPECT_FALSE(world.is_valid(c));

    EXPECT_FLOAT_EQ(world.get_position(a).x, 1.0f);
    EXPECT_FLOAT_EQ(world.get_position(b).x, 2.0f);
}

TEST(Snapshot, DeterministicReplay) {
    auto run_simulation = [](World& world, BodyHandle ball, int steps, float dt) {
        for (int i = 0; i < steps; ++i) {
            world.step(dt);
        }
        return world.get_position(ball);
    };

    World world(WorldConfig{.max_bodies = 100, .max_shapes = 100, .max_contacts = 100});
    world.set_gravity({0.0f, -10.0f});

    ShapeHandle shape = world.create_shape(CircleShape{.radius = 0.5f});
    BodyHandle ball = world.create_body({
        .position = {0.0f, 10.0f},
        .shape    = shape,
        .mass     = 1.0f
    });

    WorldSnapshot snapshot;
    world.save_state(snapshot);

    float dt = 1.0f / 60.0f;
    Vec2 pos1 = run_simulation(world, ball, 120, dt);

    // Restore and replay
    world.restore_state(snapshot);
    Vec2 pos2 = run_simulation(world, ball, 120, dt);

    // Must be bit-exact
    EXPECT_FLOAT_EQ(pos1.x, pos2.x);
    EXPECT_FLOAT_EQ(pos1.y, pos2.y);
}

TEST(Snapshot, SnapshotPreservesVelocities) {
    World world(WorldConfig{.max_bodies = 100, .max_shapes = 100, .max_contacts = 100});

    ShapeHandle shape = world.create_shape(CircleShape{.radius = 0.5f});
    BodyHandle body = world.create_body({
        .position = {0.0f, 0.0f},
        .velocity = {5.0f, -3.0f},
        .shape    = shape,
        .mass     = 1.0f
    });

    WorldSnapshot snapshot;
    world.save_state(snapshot);

    // Modify velocity via simulation
    world.set_gravity({0.0f, -10.0f});
    world.step(1.0f / 60.0f);

    EXPECT_NE(world.get_velocity(body).y, -3.0f);

    world.restore_state(snapshot);

    Vec2 vel = world.get_velocity(body);
    EXPECT_FLOAT_EQ(vel.x, 5.0f);
    EXPECT_FLOAT_EQ(vel.y, -3.0f);
}

TEST(Snapshot, MultipleSnapshotsIndependent) {
    World world(WorldConfig{.max_bodies = 100, .max_shapes = 100, .max_contacts = 100});
    world.set_gravity({0.0f, -10.0f});

    ShapeHandle shape = world.create_shape(CircleShape{.radius = 0.5f});
    BodyHandle ball = world.create_body({
        .position = {0.0f, 10.0f},
        .shape    = shape,
        .mass     = 1.0f
    });

    // Snapshot at t=0
    WorldSnapshot snap0;
    world.save_state(snap0);

    // Simulate 30 steps
    float dt = 1.0f / 60.0f;
    for (int i = 0; i < 30; ++i) {
        world.step(dt);
    }

    // Snapshot at t=0.5s
    WorldSnapshot snap1;
    world.save_state(snap1);
    Vec2 pos_at_snap1 = world.get_position(ball);

    // Simulate more
    for (int i = 0; i < 30; ++i) {
        world.step(dt);
    }

    // Restore to t=0.5s
    world.restore_state(snap1);
    EXPECT_FLOAT_EQ(world.get_position(ball).y, pos_at_snap1.y);

    // Restore to t=0
    world.restore_state(snap0);
    EXPECT_FLOAT_EQ(world.get_position(ball).y, 10.0f);
}
```

### Step 2: Run tests to verify they fail

Run: `cmake --build build && ctest --test-dir build --output-on-failure`

Expected: FAIL — `fatal error: 'stan2d/world/state_view.hpp' file not found`

### Step 3: Implement WorldStateView

**File:** `include/stan2d/world/state_view.hpp`

```cpp
#pragma once

#include <cstdint>
#include <span>

#include <stan2d/core/math_types.hpp>

namespace stan2d {

struct WorldStateView {
    float    timestamp = 0.0f;
    uint32_t active_body_count = 0;

    std::span<const Vec2>  positions;
    std::span<const Vec2>  velocities;
    std::span<const float> rotations;
    std::span<const float> angular_velocities;
    std::span<const float> masses;
};

} // namespace stan2d
```

### Step 4: Implement WorldSnapshot

**File:** `include/stan2d/world/snapshot.hpp`

```cpp
#pragma once

#include <cstdint>
#include <vector>

#include <stan2d/core/handle.hpp>
#include <stan2d/core/math_types.hpp>
#include <stan2d/core/shapes.hpp>

namespace stan2d {

struct WorldSnapshot {
    float    timestamp = 0.0f;
    uint32_t body_count = 0;

    // Body SoA data
    std::vector<Vec2>        positions;
    std::vector<Vec2>        velocities;
    std::vector<float>       rotations;
    std::vector<float>       angular_velocities;
    std::vector<float>       masses;
    std::vector<float>       inverse_masses;
    std::vector<float>       inertias;
    std::vector<float>       inverse_inertias;
    std::vector<Vec2>        forces;
    std::vector<float>       torques;
    std::vector<uint8_t>     body_types;  // BodyType stored as uint8_t
    std::vector<ShapeHandle> shape_ids;

    // Body SparseSet state (critical for Handle validity after restore)
    std::vector<uint32_t>    body_sparse_to_dense;
    std::vector<uint32_t>    body_dense_to_sparse;
    std::vector<uint32_t>    body_generations;
    std::vector<uint32_t>    body_free_list;

    // ShapeRegistry state
    std::vector<ShapeData>   shapes;
    std::vector<AABB>        shape_aabbs;
    std::vector<uint32_t>    shape_sparse_to_dense;
    std::vector<uint32_t>    shape_dense_to_sparse;
    std::vector<uint32_t>    shape_generations;
    std::vector<uint32_t>    shape_free_list;

    // Gravity
    Vec2 gravity{0.0f, 0.0f};
};

} // namespace stan2d
```

### Step 5: Add state API to World

**File:** `include/stan2d/world/world.hpp` — add to public section:

```cpp
    // ── State system ──────────────────────────────────────────────
    [[nodiscard]] WorldStateView get_state_view() const;
    void save_state(WorldSnapshot& out) const;
    void restore_state(const WorldSnapshot& snapshot);
```

### Step 6: Implement state methods (GREEN)

**File:** `src/stan2d/world/world.cpp` — add implementations:

```cpp
WorldStateView World::get_state_view() const {
    uint32_t count = bodies_.size();
    WorldStateView view;
    view.timestamp = 0.0f;
    view.active_body_count = count;
    view.positions         = std::span<const Vec2>(bodies_.positions.data(), count);
    view.velocities        = std::span<const Vec2>(bodies_.velocities.data(), count);
    view.rotations         = std::span<const float>(bodies_.rotations.data(), count);
    view.angular_velocities = std::span<const float>(bodies_.angular_velocities.data(), count);
    view.masses            = std::span<const float>(bodies_.masses.data(), count);
    return view;
}

void World::save_state(WorldSnapshot& out) const {
    uint32_t count = bodies_.size();
    out.timestamp  = 0.0f;
    out.body_count = count;
    out.gravity    = gravity_;

    // Copy body SoA data
    out.positions.assign(bodies_.positions.begin(), bodies_.positions.begin() + count);
    out.velocities.assign(bodies_.velocities.begin(), bodies_.velocities.begin() + count);
    out.rotations.assign(bodies_.rotations.begin(), bodies_.rotations.begin() + count);
    out.angular_velocities.assign(bodies_.angular_velocities.begin(),
                                  bodies_.angular_velocities.begin() + count);
    out.masses.assign(bodies_.masses.begin(), bodies_.masses.begin() + count);
    out.inverse_masses.assign(bodies_.inverse_masses.begin(),
                              bodies_.inverse_masses.begin() + count);
    out.inertias.assign(bodies_.inertias.begin(), bodies_.inertias.begin() + count);
    out.inverse_inertias.assign(bodies_.inverse_inertias.begin(),
                                bodies_.inverse_inertias.begin() + count);
    out.forces.assign(bodies_.forces.begin(), bodies_.forces.begin() + count);
    out.torques.assign(bodies_.torques.begin(), bodies_.torques.begin() + count);

    // BodyType → uint8_t
    out.body_types.resize(count);
    for (uint32_t i = 0; i < count; ++i) {
        out.body_types[i] = static_cast<uint8_t>(bodies_.body_types[i]);
    }

    out.shape_ids.assign(bodies_.shape_ids.begin(), bodies_.shape_ids.begin() + count);

    // Save body SparseSet state
    body_handles_.save_state(out.body_sparse_to_dense,
                             out.body_dense_to_sparse,
                             out.body_generations,
                             out.body_free_list);

    // Save ShapeRegistry state
    shape_registry_.save_state(out.shapes, out.shape_aabbs,
                               out.shape_sparse_to_dense,
                               out.shape_dense_to_sparse,
                               out.shape_generations,
                               out.shape_free_list);
}

void World::restore_state(const WorldSnapshot& snapshot) {
    uint32_t count = snapshot.body_count;
    gravity_ = snapshot.gravity;

    // Restore body SoA data
    bodies_.positions.assign(snapshot.positions.begin(), snapshot.positions.end());
    bodies_.velocities.assign(snapshot.velocities.begin(), snapshot.velocities.end());
    bodies_.rotations.assign(snapshot.rotations.begin(), snapshot.rotations.end());
    bodies_.angular_velocities.assign(snapshot.angular_velocities.begin(),
                                      snapshot.angular_velocities.end());
    bodies_.masses.assign(snapshot.masses.begin(), snapshot.masses.end());
    bodies_.inverse_masses.assign(snapshot.inverse_masses.begin(),
                                  snapshot.inverse_masses.end());
    bodies_.inertias.assign(snapshot.inertias.begin(), snapshot.inertias.end());
    bodies_.inverse_inertias.assign(snapshot.inverse_inertias.begin(),
                                    snapshot.inverse_inertias.end());
    bodies_.forces.assign(snapshot.forces.begin(), snapshot.forces.end());
    bodies_.torques.assign(snapshot.torques.begin(), snapshot.torques.end());

    // uint8_t → BodyType
    bodies_.body_types.resize(count);
    for (uint32_t i = 0; i < count; ++i) {
        bodies_.body_types[i] = static_cast<BodyType>(snapshot.body_types[i]);
    }

    bodies_.shape_ids.assign(snapshot.shape_ids.begin(), snapshot.shape_ids.end());

    // Restore body SparseSet state
    body_handles_.restore_state(snapshot.body_sparse_to_dense,
                                snapshot.body_dense_to_sparse,
                                snapshot.body_generations,
                                snapshot.body_free_list);

    // Restore ShapeRegistry state
    shape_registry_.restore_state(snapshot.shapes, snapshot.shape_aabbs,
                                  snapshot.shape_sparse_to_dense,
                                  snapshot.shape_dense_to_sparse,
                                  snapshot.shape_generations,
                                  snapshot.shape_free_list);

    // Force proxy rebuild on next step
    proxies_built_ = false;
}
```

### Step 7: Add save_state/restore_state to SparseSet and ShapeRegistry

**File:** `include/stan2d/core/sparse_set.hpp` — add methods:

```cpp
void save_state(std::vector<uint32_t>& out_sparse,
                std::vector<uint32_t>& out_dense_to_sparse,
                std::vector<uint32_t>& out_generations,
                std::vector<uint32_t>& out_free_list) const {
    out_sparse.assign(sparse.begin(), sparse.end());
    out_dense_to_sparse.assign(dense_to_sparse.begin(), dense_to_sparse.end());
    out_generations.assign(generations.begin(), generations.end());
    out_free_list.assign(free_list.begin(), free_list.end());
}

void restore_state(const std::vector<uint32_t>& in_sparse,
                   const std::vector<uint32_t>& in_dense_to_sparse,
                   const std::vector<uint32_t>& in_generations,
                   const std::vector<uint32_t>& in_free_list) {
    sparse.assign(in_sparse.begin(), in_sparse.end());
    dense_to_sparse.assign(in_dense_to_sparse.begin(), in_dense_to_sparse.end());
    generations.assign(in_generations.begin(), in_generations.end());
    free_list.assign(in_free_list.begin(), in_free_list.end());
}
```

**File:** `include/stan2d/core/shape_registry.hpp` — add methods:

```cpp
void save_state(std::vector<ShapeData>& out_shapes,
                std::vector<AABB>& out_aabbs,
                std::vector<uint32_t>& out_sparse,
                std::vector<uint32_t>& out_dense_to_sparse,
                std::vector<uint32_t>& out_generations,
                std::vector<uint32_t>& out_free_list) const {
    out_shapes.assign(shapes.begin(), shapes.end());
    out_aabbs.assign(local_aabbs.begin(), local_aabbs.end());
    handles.save_state(out_sparse, out_dense_to_sparse, out_generations, out_free_list);
}

void restore_state(const std::vector<ShapeData>& in_shapes,
                   const std::vector<AABB>& in_aabbs,
                   const std::vector<uint32_t>& in_sparse,
                   const std::vector<uint32_t>& in_dense_to_sparse,
                   const std::vector<uint32_t>& in_generations,
                   const std::vector<uint32_t>& in_free_list) {
    shapes.assign(in_shapes.begin(), in_shapes.end());
    local_aabbs.assign(in_aabbs.begin(), in_aabbs.end());
    handles.restore_state(in_sparse, in_dense_to_sparse, in_generations, in_free_list);
}
```

### Step 8: Run tests to verify they pass

Run: `cmake --build build && ctest --test-dir build --output-on-failure`

Expected: PASS — all state system tests green

### Step 9: Commit

```bash
git add include/stan2d/world/state_view.hpp \
        include/stan2d/world/snapshot.hpp \
        include/stan2d/world/world.hpp \
        src/stan2d/world/world.cpp \
        include/stan2d/core/sparse_set.hpp \
        include/stan2d/core/shape_registry.hpp \
        tests/unit/test_state_system.cpp
git commit -m "feat: State system with zero-copy WorldStateView and restorable WorldSnapshot"
```

---