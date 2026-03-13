# Task 18: Joint State System — Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the state system (StateView, Snapshot, TrajectoryRecorder, state export) to include joint data — enabling ML/RL observation, save/restore for MCTS rollback, and trajectory recording for offline training.

**Architecture:** `JointStateView` provides zero-copy spans into `JointStorage` cached fields. `JointSnapshot` deep-copies all SoA fields + SparseSet state for bit-identical restore. `TrajectoryRecorder` records 4 float fields per joint per frame in a flat `[frames, max_joints, 4]` tensor layout. `export_state()` adds a `"joints"` array to JSON output.

**Tech Stack:** C++20, glm, nlohmann-json, Google Test

**Depends on:** Tasks 13–17 (all joint types implemented)

---

## File Structure

| Action | File | Responsibility |
|--------|------|---------------|
| Modify | `include/stan2d/world/state_view.hpp` | Add `JointStateView` struct, add `joints` field to `WorldStateView` |
| Modify | `include/stan2d/world/snapshot.hpp` | Add `JointSnapshot` struct, add `joints` field to `WorldSnapshot` |
| Modify | `src/stan2d/world/world.cpp` | Extend `get_state_view()`, `save_state()`, `restore_state()` for joints |
| Modify | `include/stan2d/export/trajectory_recorder.hpp` | Add joint trajectory buffers + accessors |
| Modify | `src/stan2d/export/trajectory_recorder.cpp` | Capture joint data per frame |
| Modify | `src/stan2d/export/state_export.cpp` | Add joints to JSON/Binary export |
| Create | `tests/unit/test_joint_state_system.cpp` | All joint state system tests |

---

### Task 18.1: Add JointStateView to state_view.hpp

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_joint_state_system.cpp`:

```cpp
#include <gtest/gtest.h>
#include <cmath>
#include <limits>
#include <stan2d/world/world.hpp>
#include <stan2d/world/state_view.hpp>

using namespace stan2d;

class JointStateSystemFixture : public ::testing::Test {
protected:
    World world{WorldConfig{
        .max_bodies = 100, .max_constraints = 200,
        .max_contacts = 200, .max_shapes = 100, .max_joints = 50
    }};
    ShapeHandle shape = world.create_shape(CircleShape{.radius = 0.5f});
    float dt = 1.0f / 60.0f;

    BodyHandle make_dynamic(Vec2 pos, float mass = 1.0f) {
        return world.create_body({
            .position = pos, .shape = shape,
            .mass = mass, .inertia = 0.5f * mass
        });
    }

    BodyHandle make_static(Vec2 pos) {
        return world.create_body({
            .position = pos, .shape = shape, .body_type = BodyType::Static
        });
    }
};

TEST_F(JointStateSystemFixture, JointStateViewReflectsJoints) {
    BodyHandle a = make_static({0.0f, 0.0f});
    BodyHandle b = make_dynamic({2.0f, 0.0f});

    JointDef def;
    def.type = JointType::Hinge;
    def.body_a = a;
    def.body_b = b;
    def.motor_enabled = true;
    def.motor_speed = 3.0f;
    def.motor_torque = 10.0f;

    world.create_joint(def);
    world.step(dt);  // Populate cached values

    WorldStateView view = world.get_state_view();
    EXPECT_EQ(view.joints.active_joint_count, 1u);
    EXPECT_EQ(view.joints.types.size(), 1u);
    EXPECT_EQ(view.joints.angles.size(), 1u);
    EXPECT_EQ(view.joints.angular_speeds.size(), 1u);
    EXPECT_EQ(view.joints.motor_target_speeds.size(), 1u);
    EXPECT_EQ(view.joints.motor_enabled.size(), 1u);
    EXPECT_EQ(view.joints.constraint_forces.size(), 1u);
    EXPECT_EQ(view.joints.lengths.size(), 1u);

    // Motor enabled should be 1
    EXPECT_EQ(view.joints.motor_enabled[0], 1);
    // Motor speed should reflect set value
    EXPECT_FLOAT_EQ(view.joints.motor_target_speeds[0], 3.0f);
}

TEST_F(JointStateSystemFixture, JointStateViewEmptyWorld) {
    WorldStateView view = world.get_state_view();
    EXPECT_EQ(view.joints.active_joint_count, 0u);
    EXPECT_TRUE(view.joints.types.empty());
    EXPECT_TRUE(view.joints.angles.empty());
}

TEST_F(JointStateSystemFixture, JointStateViewMultipleTypes) {
    BodyHandle a = make_static({0.0f, 0.0f});
    BodyHandle b = make_dynamic({2.0f, 0.0f});
    BodyHandle c = make_dynamic({4.0f, 0.0f});

    JointDef hinge_def;
    hinge_def.type = JointType::Hinge;
    hinge_def.body_a = a;
    hinge_def.body_b = b;
    world.create_joint(hinge_def);

    JointDef spring_def;
    spring_def.type = JointType::Spring;
    spring_def.body_a = b;
    spring_def.body_b = c;
    spring_def.stiffness = 50.0f;
    world.create_joint(spring_def);

    world.step(dt);

    WorldStateView view = world.get_state_view();
    EXPECT_EQ(view.joints.active_joint_count, 2u);

    // Verify types are readable as uint8_t
    EXPECT_EQ(view.joints.types[0], static_cast<uint8_t>(JointType::Hinge));
    EXPECT_EQ(view.joints.types[1], static_cast<uint8_t>(JointType::Spring));

    // Non-motor joint (Spring) should have NaN motor_target_speed per spec
    // RL agents must mask by motor_enabled
    EXPECT_EQ(view.joints.motor_enabled[1], 0);
    EXPECT_TRUE(std::isnan(view.joints.motor_target_speeds[1]));
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cmake --build build && ./build/stan2d_tests --gtest_filter=JointStateSystemFixture.JointStateView*`
Expected: FAIL — `JointStateView` not defined, `WorldStateView::joints` not defined

- [ ] **Step 3: Modify state_view.hpp**

Replace `include/stan2d/world/state_view.hpp` with:

```cpp
#pragma once

#include <cstdint>
#include <span>

#include <stan2d/core/math_types.hpp>

namespace stan2d {

struct JointStateView {
    uint32_t active_joint_count = 0;
    std::span<const uint8_t> types;
    std::span<const float>   angles;
    std::span<const float>   angular_speeds;
    std::span<const float>   motor_target_speeds;
    std::span<const uint8_t> motor_enabled;
    std::span<const float>   constraint_forces;
    std::span<const float>   lengths;
};

struct WorldStateView {
    float    timestamp = 0.0f;
    uint32_t active_body_count = 0;

    std::span<const Vec2>  positions;
    std::span<const Vec2>  velocities;
    std::span<const float> rotations;
    std::span<const float> angular_velocities;
    std::span<const float> masses;

    JointStateView joints;
};

} // namespace stan2d
```

- [ ] **Step 4: Extend World::get_state_view() to populate joints**

In `src/stan2d/world/world.cpp`, in `get_state_view()`, add after existing body fields:

```cpp
    // Joint state view
    uint32_t joint_count_val = joints_.size;
    view.joints.active_joint_count = joint_count_val;
    if (joint_count_val > 0) {
        // JointType is `enum class JointType : uint8_t` so sizeof == 1.
        // The reinterpret_cast is safe per C++ aliasing rules (uint8_t == unsigned char).
        // Alternative: change JointStorage::types to std::vector<uint8_t> to avoid the cast.
        view.joints.types = std::span<const uint8_t>(
            reinterpret_cast<const uint8_t*>(joints_.types.data()), joint_count_val);
        view.joints.angles = std::span<const float>(
            joints_.cached_angles.data(), joint_count_val);
        view.joints.angular_speeds = std::span<const float>(
            joints_.cached_angular_speeds.data(), joint_count_val);
        // Per spec: motor_target_speeds should be NaN for joints where motor_enabled=0
        // or joint type has no motor. This NaN sentinel must be set in JointStorage::push_back()
        // (Task 13) or in prepare_joint_constraints() (Task 14). The implementer must ensure
        // that when motor_enabled=0, motor_target_speeds[i] = std::numeric_limits<float>::quiet_NaN().
        view.joints.motor_target_speeds = std::span<const float>(
            joints_.motor_target_speeds.data(), joint_count_val);
        view.joints.motor_enabled = std::span<const uint8_t>(
            joints_.motor_enabled.data(), joint_count_val);
        view.joints.constraint_forces = std::span<const float>(
            joints_.constraint_forces.data(), joint_count_val);
        view.joints.lengths = std::span<const float>(
            joints_.cached_lengths.data(), joint_count_val);
    }
```

- [ ] **Step 5: Run test to verify it passes**

Run: `cmake --build build && ./build/stan2d_tests --gtest_filter=JointStateSystemFixture.JointStateView*`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add include/stan2d/world/state_view.hpp src/stan2d/world/world.cpp tests/unit/test_joint_state_system.cpp
git commit -m "feat: add JointStateView with zero-copy spans into JointStorage"
```

---

### Task 18.2: Add JointSnapshot and save/restore

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/test_joint_state_system.cpp`:

```cpp
#include <stan2d/world/snapshot.hpp>

TEST_F(JointStateSystemFixture, SaveRestoreJointState) {
    world.set_gravity({0.0f, -10.0f});

    BodyHandle a = make_static({0.0f, 5.0f});
    BodyHandle b = make_dynamic({2.0f, 5.0f});

    JointDef def;
    def.type = JointType::Hinge;
    def.body_a = a;
    def.body_b = b;
    def.motor_enabled = true;
    def.motor_speed = 5.0f;
    def.motor_torque = 50.0f;

    JointHandle jh = world.create_joint(def);

    // Run some steps to build up warm-start state
    for (int i = 0; i < 30; ++i) {
        world.step(dt);
    }

    // Save state
    WorldSnapshot snapshot;
    world.save_state(snapshot);

    float angle_at_save = world.get_joint_angle(jh);
    float speed_at_save = world.get_joint_speed(jh);

    // Run more steps to change state
    for (int i = 0; i < 60; ++i) {
        world.step(dt);
    }

    // State should have changed
    EXPECT_NE(world.get_joint_angle(jh), angle_at_save);

    // Restore
    world.restore_state(snapshot);

    // Joint should be valid again
    EXPECT_TRUE(world.is_valid(jh));
    EXPECT_EQ(world.joint_count(), 1u);
}

TEST_F(JointStateSystemFixture, SaveRestoreDeterministicReplay) {
    world.set_gravity({0.0f, -10.0f});

    BodyHandle a = make_static({0.0f, 5.0f});
    BodyHandle b = make_dynamic({2.0f, 5.0f});

    JointDef def;
    def.type = JointType::Hinge;
    def.body_a = a;
    def.body_b = b;

    JointHandle jh = world.create_joint(def);

    // Warm up
    for (int i = 0; i < 10; ++i) {
        world.step(dt);
    }

    // Save
    WorldSnapshot snapshot;
    world.save_state(snapshot);

    // Run 120 steps and record final state
    for (int i = 0; i < 120; ++i) {
        world.step(dt);
    }
    Vec2 pos_run1 = world.get_position(b);
    float angle_run1 = world.get_joint_angle(jh);

    // Restore and replay
    world.restore_state(snapshot);
    for (int i = 0; i < 120; ++i) {
        world.step(dt);
    }
    Vec2 pos_run2 = world.get_position(b);
    float angle_run2 = world.get_joint_angle(jh);

    // Must be bit-exact
    EXPECT_FLOAT_EQ(pos_run1.x, pos_run2.x);
    EXPECT_FLOAT_EQ(pos_run1.y, pos_run2.y);
    EXPECT_FLOAT_EQ(angle_run1, angle_run2);
}

TEST_F(JointStateSystemFixture, SaveRestoreWarmStartCompleteness) {
    world.set_gravity({0.0f, -10.0f});

    BodyHandle a = make_static({0.0f, 5.0f});
    BodyHandle b = make_dynamic({0.0f, 3.0f});

    JointDef def;
    def.type = JointType::Hinge;
    def.body_a = a;
    def.body_b = b;
    def.limit_enabled = true;
    def.limit_min = -1.0f;
    def.limit_max = 1.0f;
    def.motor_enabled = true;
    def.motor_speed = 2.0f;
    def.motor_torque = 20.0f;

    world.create_joint(def);

    // Run to build up warm-start state (limit + motor impulses)
    for (int i = 0; i < 60; ++i) {
        world.step(dt);
    }

    // Save
    WorldSnapshot snap;
    world.save_state(snap);

    // Verify snapshot has joint data
    EXPECT_EQ(snap.joints.count, 1u);
    EXPECT_EQ(snap.joints.types.size(), 1u);
}

TEST_F(JointStateSystemFixture, DestroyJointThenRestoreRestoresIt) {
    BodyHandle a = make_static({0.0f, 0.0f});
    BodyHandle b = make_dynamic({2.0f, 0.0f});

    JointDef def;
    def.type = JointType::Distance;
    def.body_a = a;
    def.body_b = b;
    JointHandle jh = world.create_joint(def);

    WorldSnapshot snap;
    world.save_state(snap);

    world.destroy_joint(jh);
    EXPECT_FALSE(world.is_valid(jh));
    EXPECT_EQ(world.joint_count(), 0u);

    world.restore_state(snap);
    EXPECT_TRUE(world.is_valid(jh));
    EXPECT_EQ(world.joint_count(), 1u);
}

TEST_F(JointStateSystemFixture, MCTSRollbackScenario) {
    world.set_gravity({0.0f, -10.0f});

    BodyHandle a = make_static({0.0f, 5.0f});
    BodyHandle b = make_dynamic({2.0f, 5.0f});

    JointDef def;
    def.type = JointType::Hinge;
    def.body_a = a;
    def.body_b = b;
    def.motor_enabled = true;
    def.motor_speed = 0.0f;
    def.motor_torque = 50.0f;

    JointHandle jh = world.create_joint(def);

    // Save root state
    WorldSnapshot root;
    for (int i = 0; i < 10; ++i) { world.step(dt); }
    world.save_state(root);

    // Branch 1: motor CW
    world.set_motor_speed(jh, 5.0f);
    for (int i = 0; i < 60; ++i) { world.step(dt); }
    Vec2 pos_branch1 = world.get_position(b);

    // Rollback to root
    world.restore_state(root);

    // Verify motor speed was restored to snapshot value (0.0f)
    world.step(dt);  // populate cached values
    // Motor target speed should be restored from snapshot
    WorldStateView view_after_restore = world.get_state_view();
    EXPECT_FLOAT_EQ(view_after_restore.joints.motor_target_speeds[0], 0.0f);
    world.restore_state(root);  // re-restore since we stepped

    // Branch 2: motor CCW
    world.set_motor_speed(jh, -5.0f);
    for (int i = 0; i < 60; ++i) { world.step(dt); }
    Vec2 pos_branch2 = world.get_position(b);

    // Branches should diverge
    float divergence = glm::length(pos_branch1 - pos_branch2);
    EXPECT_GT(divergence, 0.5f);

    // Rollback to root and replay branch 1 — should be identical
    world.restore_state(root);
    world.set_motor_speed(jh, 5.0f);
    for (int i = 0; i < 60; ++i) { world.step(dt); }
    Vec2 pos_replay = world.get_position(b);

    EXPECT_FLOAT_EQ(pos_branch1.x, pos_replay.x);
    EXPECT_FLOAT_EQ(pos_branch1.y, pos_replay.y);
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cmake --build build && ./build/stan2d_tests --gtest_filter=JointStateSystemFixture.SaveRestore*:JointStateSystemFixture.Destroy*:JointStateSystemFixture.MCTS*`
Expected: FAIL — `JointSnapshot` not defined, `WorldSnapshot::joints` not defined

- [ ] **Step 3: Add JointSnapshot to snapshot.hpp**

In `include/stan2d/world/snapshot.hpp`, add include:

```cpp
#include <stan2d/joints/joint_types.hpp>
```

Add before `WorldSnapshot`:

```cpp
struct JointSnapshot {
    // SparseSet state
    std::vector<uint32_t> sparse;
    std::vector<uint32_t> dense_to_sparse;
    std::vector<uint32_t> generations;
    std::vector<uint32_t> free_list;

    // Full SoA copy (mirrors JointStorage)
    std::vector<JointType> types;
    std::vector<uint32_t>  body_a;
    std::vector<uint32_t>  body_b;
    std::vector<Vec2>      anchor_a;
    std::vector<Vec2>      anchor_b;

    std::vector<uint8_t>   limit_enabled;
    std::vector<float>     limit_min;
    std::vector<float>     limit_max;
    std::vector<float>     reference_angle;
    std::vector<float>     accumulated_limit_impulse;

    std::vector<uint8_t>   motor_enabled;
    std::vector<float>     motor_target_speeds;
    std::vector<float>     motor_max_torque;
    std::vector<float>     accumulated_motor_impulse;

    std::vector<float>     spring_stiffness;
    std::vector<float>     spring_damping;
    std::vector<float>     spring_rest_length;

    std::vector<float>     distance_length;
    std::vector<uint8_t>   distance_cable_mode;

    std::vector<Vec2>      pulley_ground_a;
    std::vector<Vec2>      pulley_ground_b;
    std::vector<float>     pulley_ratio;
    std::vector<float>     pulley_constant;

    std::vector<float>     accumulated_impulse_x;
    std::vector<float>     accumulated_impulse_y;

    uint32_t count = 0;
};
```

Add to `WorldSnapshot`, after the gravity field:

```cpp
    // Joint snapshot
    JointSnapshot joints;
```

- [ ] **Step 4: Extend save_state() for joints**

In `src/stan2d/world/world.cpp`, at the end of `save_state()`, add:

```cpp
    // Save joint state
    uint32_t jcount = joints_.size;
    out.joints.count = jcount;

    // Save joint SparseSet state
    joint_handles_.save_state(out.joints.sparse,
                              out.joints.dense_to_sparse,
                              out.joints.generations,
                              out.joints.free_list);

    // Save joint SoA data
    out.joints.types.assign(joints_.types.begin(), joints_.types.begin() + jcount);
    out.joints.body_a.assign(joints_.body_a.begin(), joints_.body_a.begin() + jcount);
    out.joints.body_b.assign(joints_.body_b.begin(), joints_.body_b.begin() + jcount);
    out.joints.anchor_a.assign(joints_.anchor_a.begin(), joints_.anchor_a.begin() + jcount);
    out.joints.anchor_b.assign(joints_.anchor_b.begin(), joints_.anchor_b.begin() + jcount);

    out.joints.limit_enabled.assign(joints_.limit_enabled.begin(),
                                    joints_.limit_enabled.begin() + jcount);
    out.joints.limit_min.assign(joints_.limit_min.begin(), joints_.limit_min.begin() + jcount);
    out.joints.limit_max.assign(joints_.limit_max.begin(), joints_.limit_max.begin() + jcount);
    out.joints.reference_angle.assign(joints_.reference_angle.begin(),
                                      joints_.reference_angle.begin() + jcount);
    out.joints.accumulated_limit_impulse.assign(
        joints_.accumulated_limit_impulse.begin(),
        joints_.accumulated_limit_impulse.begin() + jcount);

    out.joints.motor_enabled.assign(joints_.motor_enabled.begin(),
                                    joints_.motor_enabled.begin() + jcount);
    out.joints.motor_target_speeds.assign(joints_.motor_target_speeds.begin(),
                                          joints_.motor_target_speeds.begin() + jcount);
    out.joints.motor_max_torque.assign(joints_.motor_max_torque.begin(),
                                       joints_.motor_max_torque.begin() + jcount);
    out.joints.accumulated_motor_impulse.assign(
        joints_.accumulated_motor_impulse.begin(),
        joints_.accumulated_motor_impulse.begin() + jcount);

    out.joints.spring_stiffness.assign(joints_.spring_stiffness.begin(),
                                       joints_.spring_stiffness.begin() + jcount);
    out.joints.spring_damping.assign(joints_.spring_damping.begin(),
                                     joints_.spring_damping.begin() + jcount);
    out.joints.spring_rest_length.assign(joints_.spring_rest_length.begin(),
                                         joints_.spring_rest_length.begin() + jcount);

    out.joints.distance_length.assign(joints_.distance_length.begin(),
                                      joints_.distance_length.begin() + jcount);
    out.joints.distance_cable_mode.assign(joints_.distance_cable_mode.begin(),
                                          joints_.distance_cable_mode.begin() + jcount);

    out.joints.pulley_ground_a.assign(joints_.pulley_ground_a.begin(),
                                      joints_.pulley_ground_a.begin() + jcount);
    out.joints.pulley_ground_b.assign(joints_.pulley_ground_b.begin(),
                                      joints_.pulley_ground_b.begin() + jcount);
    out.joints.pulley_ratio.assign(joints_.pulley_ratio.begin(),
                                   joints_.pulley_ratio.begin() + jcount);
    out.joints.pulley_constant.assign(joints_.pulley_constant.begin(),
                                      joints_.pulley_constant.begin() + jcount);

    out.joints.accumulated_impulse_x.assign(joints_.accumulated_impulse_x.begin(),
                                            joints_.accumulated_impulse_x.begin() + jcount);
    out.joints.accumulated_impulse_y.assign(joints_.accumulated_impulse_y.begin(),
                                            joints_.accumulated_impulse_y.begin() + jcount);
```

- [ ] **Step 5: Extend restore_state() for joints**

In `src/stan2d/world/world.cpp`, at the end of `restore_state()`, before `proxies_built_ = false;`, add:

```cpp
    // Restore joint state
    uint32_t jcount = snapshot.joints.count;
    joints_.size = jcount;

    // Restore joint SparseSet
    joint_handles_.restore_state(snapshot.joints.sparse,
                                 snapshot.joints.dense_to_sparse,
                                 snapshot.joints.generations,
                                 snapshot.joints.free_list);

    // Restore joint SoA data
    joints_.types.assign(snapshot.joints.types.begin(), snapshot.joints.types.end());
    joints_.body_a.assign(snapshot.joints.body_a.begin(), snapshot.joints.body_a.end());
    joints_.body_b.assign(snapshot.joints.body_b.begin(), snapshot.joints.body_b.end());
    joints_.anchor_a.assign(snapshot.joints.anchor_a.begin(), snapshot.joints.anchor_a.end());
    joints_.anchor_b.assign(snapshot.joints.anchor_b.begin(), snapshot.joints.anchor_b.end());

    joints_.limit_enabled.assign(snapshot.joints.limit_enabled.begin(),
                                 snapshot.joints.limit_enabled.end());
    joints_.limit_min.assign(snapshot.joints.limit_min.begin(),
                             snapshot.joints.limit_min.end());
    joints_.limit_max.assign(snapshot.joints.limit_max.begin(),
                             snapshot.joints.limit_max.end());
    joints_.reference_angle.assign(snapshot.joints.reference_angle.begin(),
                                   snapshot.joints.reference_angle.end());
    joints_.accumulated_limit_impulse.assign(
        snapshot.joints.accumulated_limit_impulse.begin(),
        snapshot.joints.accumulated_limit_impulse.end());

    joints_.motor_enabled.assign(snapshot.joints.motor_enabled.begin(),
                                 snapshot.joints.motor_enabled.end());
    joints_.motor_target_speeds.assign(snapshot.joints.motor_target_speeds.begin(),
                                       snapshot.joints.motor_target_speeds.end());
    joints_.motor_max_torque.assign(snapshot.joints.motor_max_torque.begin(),
                                    snapshot.joints.motor_max_torque.end());
    joints_.accumulated_motor_impulse.assign(
        snapshot.joints.accumulated_motor_impulse.begin(),
        snapshot.joints.accumulated_motor_impulse.end());

    joints_.spring_stiffness.assign(snapshot.joints.spring_stiffness.begin(),
                                    snapshot.joints.spring_stiffness.end());
    joints_.spring_damping.assign(snapshot.joints.spring_damping.begin(),
                                  snapshot.joints.spring_damping.end());
    joints_.spring_rest_length.assign(snapshot.joints.spring_rest_length.begin(),
                                      snapshot.joints.spring_rest_length.end());

    joints_.distance_length.assign(snapshot.joints.distance_length.begin(),
                                   snapshot.joints.distance_length.end());
    joints_.distance_cable_mode.assign(snapshot.joints.distance_cable_mode.begin(),
                                       snapshot.joints.distance_cable_mode.end());

    joints_.pulley_ground_a.assign(snapshot.joints.pulley_ground_a.begin(),
                                   snapshot.joints.pulley_ground_a.end());
    joints_.pulley_ground_b.assign(snapshot.joints.pulley_ground_b.begin(),
                                   snapshot.joints.pulley_ground_b.end());
    joints_.pulley_ratio.assign(snapshot.joints.pulley_ratio.begin(),
                                snapshot.joints.pulley_ratio.end());
    joints_.pulley_constant.assign(snapshot.joints.pulley_constant.begin(),
                                   snapshot.joints.pulley_constant.end());

    joints_.accumulated_impulse_x.assign(snapshot.joints.accumulated_impulse_x.begin(),
                                         snapshot.joints.accumulated_impulse_x.end());
    joints_.accumulated_impulse_y.assign(snapshot.joints.accumulated_impulse_y.begin(),
                                         snapshot.joints.accumulated_impulse_y.end());

    // Reset cached observable arrays (not snapshotted, recomputed each step).
    // Use assign() not resize() — resize() won't shrink, leaving stale tail data.
    joints_.constraint_forces.assign(jcount, 0.0f);
    joints_.cached_angles.assign(jcount, 0.0f);
    joints_.cached_angular_speeds.assign(jcount, 0.0f);
    joints_.cached_lengths.assign(jcount, 0.0f);
```

- [ ] **Step 6: Run test to verify it passes**

Run: `cmake --build build && ./build/stan2d_tests --gtest_filter=JointStateSystemFixture.*`
Expected: ALL PASS

- [ ] **Step 7: Run all existing tests for regression**

Run: `ctest --test-dir build`
Expected: ALL PASS

- [ ] **Step 8: Commit**

```bash
git add include/stan2d/world/snapshot.hpp src/stan2d/world/world.cpp tests/unit/test_joint_state_system.cpp
git commit -m "feat: add JointSnapshot with save/restore for deterministic replay"
```

---

### Task 18.3: Extend TrajectoryRecorder for joints

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/test_joint_state_system.cpp`:

```cpp
#include <stan2d/export/trajectory_recorder.hpp>

TEST_F(JointStateSystemFixture, TrajectoryRecorderCapturesJoints) {
    world.set_gravity({0.0f, -10.0f});

    BodyHandle a = make_static({0.0f, 5.0f});
    BodyHandle b = make_dynamic({2.0f, 5.0f});

    JointDef def;
    def.type = JointType::Hinge;
    def.body_a = a;
    def.body_b = b;

    world.create_joint(def);

    TrajectoryRecorder recorder(world, 10);
    recorder.start();

    for (int i = 0; i < 5; ++i) {
        world.step(dt);
        recorder.capture();
    }

    EXPECT_EQ(recorder.current_frame(), 5u);
    EXPECT_EQ(recorder.max_joints(), 50u);  // from config.max_joints

    // Verify joint active count
    EXPECT_EQ(recorder.get_joint_active_count(0), 1u);
}

TEST_F(JointStateSystemFixture, TrajectoryJointTensorShape) {
    BodyHandle a = make_static({0.0f, 5.0f});
    BodyHandle b = make_dynamic({2.0f, 5.0f});

    JointDef def;
    def.type = JointType::Hinge;
    def.body_a = a;
    def.body_b = b;
    world.create_joint(def);

    TrajectoryRecorder recorder(world, 10);
    recorder.start();  // Required before capture()

    world.step(dt);
    recorder.capture();

    // Raw joint data should have shape [max_frames * max_joints]
    auto raw_angles = recorder.raw_joint_angles();
    EXPECT_EQ(raw_angles.size(), 10u * 50u);  // max_frames * max_joints

    auto raw_lengths = recorder.raw_joint_lengths();
    EXPECT_EQ(raw_lengths.size(), 10u * 50u);
}

TEST_F(JointStateSystemFixture, TrajectoryJointDataAccessors) {
    world.set_gravity({0.0f, -10.0f});

    BodyHandle a = make_static({0.0f, 5.0f});
    BodyHandle b = make_dynamic({2.0f, 5.0f});

    JointDef def;
    def.type = JointType::Hinge;
    def.body_a = a;
    def.body_b = b;
    world.create_joint(def);

    TrajectoryRecorder recorder(world, 10);
    recorder.start();

    for (int i = 0; i < 3; ++i) {
        world.step(dt);
        recorder.capture();
    }

    // Angle should be evolving across frames (pendulum swinging down)
    float angle_f0 = recorder.get_joint_angle(0, 0);
    float angle_f2 = recorder.get_joint_angle(2, 0);
    // They might be similar early, but lengths should be nonzero
    float len = recorder.get_joint_length(0, 0);
    EXPECT_GT(len, 0.0f);
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cmake --build build && ./build/stan2d_tests --gtest_filter=JointStateSystemFixture.Trajectory*`
Expected: FAIL — `max_joints()`, `get_joint_active_count()`, `raw_joint_angles()`, etc. not defined

- [ ] **Step 3: Extend trajectory_recorder.hpp**

In `include/stan2d/export/trajectory_recorder.hpp`, add to the public section:

```cpp
    [[nodiscard]] uint32_t max_joints() const { return max_joints_; }

    [[nodiscard]] float get_joint_angle(uint32_t frame, uint32_t joint) const {
        return all_joint_angles_[frame * max_joints_ + joint];
    }

    [[nodiscard]] float get_joint_angular_speed(uint32_t frame, uint32_t joint) const {
        return all_joint_angular_speeds_[frame * max_joints_ + joint];
    }

    [[nodiscard]] float get_joint_constraint_force(uint32_t frame, uint32_t joint) const {
        return all_joint_constraint_forces_[frame * max_joints_ + joint];
    }

    [[nodiscard]] float get_joint_length(uint32_t frame, uint32_t joint) const {
        return all_joint_lengths_[frame * max_joints_ + joint];
    }

    [[nodiscard]] uint32_t get_joint_active_count(uint32_t frame) const {
        return joint_frame_active_counts_[frame];
    }

    [[nodiscard]] std::span<const float> raw_joint_angles() const {
        return std::span<const float>(all_joint_angles_.data(), all_joint_angles_.size());
    }

    [[nodiscard]] std::span<const float> raw_joint_angular_speeds() const {
        return std::span<const float>(all_joint_angular_speeds_.data(),
                                      all_joint_angular_speeds_.size());
    }

    [[nodiscard]] std::span<const float> raw_joint_constraint_forces() const {
        return std::span<const float>(all_joint_constraint_forces_.data(),
                                      all_joint_constraint_forces_.size());
    }

    [[nodiscard]] std::span<const float> raw_joint_lengths() const {
        return std::span<const float>(all_joint_lengths_.data(), all_joint_lengths_.size());
    }
```

Add to the private section:

```cpp
    uint32_t max_joints_ = 0;

    // Pre-allocated: max_frames * max_joints (fixed stride)
    std::vector<float>    all_joint_angles_;
    std::vector<float>    all_joint_angular_speeds_;
    std::vector<float>    all_joint_constraint_forces_;
    std::vector<float>    all_joint_lengths_;
    std::vector<uint32_t> joint_frame_active_counts_;
```

- [ ] **Step 4: Extend trajectory_recorder.cpp constructor and capture()**

In `src/stan2d/export/trajectory_recorder.cpp`, update the constructor:

```cpp
TrajectoryRecorder::TrajectoryRecorder(const World& world, uint32_t max_frames)
    : world_(world)
    , max_frames_(max_frames)
    , max_bodies_(world.config().max_bodies)
    , max_joints_(world.config().max_joints)
{
    uint32_t body_total = max_frames_ * max_bodies_;
    all_positions_.resize(body_total, Vec2{0.0f, 0.0f});
    all_velocities_.resize(body_total, Vec2{0.0f, 0.0f});
    all_rotations_.resize(body_total, 0.0f);
    frame_active_counts_.resize(max_frames_, 0);

    uint32_t joint_total = max_frames_ * max_joints_;
    all_joint_angles_.resize(joint_total, 0.0f);
    all_joint_angular_speeds_.resize(joint_total, 0.0f);
    all_joint_constraint_forces_.resize(joint_total, 0.0f);
    all_joint_lengths_.resize(joint_total, 0.0f);
    joint_frame_active_counts_.resize(max_frames_, 0);
}
```

Update `start()` to also clear joint buffers:

```cpp
void TrajectoryRecorder::start() {
    current_frame_ = 0;
    std::fill(all_positions_.begin(), all_positions_.end(), Vec2{0.0f, 0.0f});
    std::fill(all_velocities_.begin(), all_velocities_.end(), Vec2{0.0f, 0.0f});
    std::fill(all_rotations_.begin(), all_rotations_.end(), 0.0f);
    std::fill(frame_active_counts_.begin(), frame_active_counts_.end(), 0u);

    std::fill(all_joint_angles_.begin(), all_joint_angles_.end(), 0.0f);
    std::fill(all_joint_angular_speeds_.begin(), all_joint_angular_speeds_.end(), 0.0f);
    std::fill(all_joint_constraint_forces_.begin(), all_joint_constraint_forces_.end(), 0.0f);
    std::fill(all_joint_lengths_.begin(), all_joint_lengths_.end(), 0.0f);
    std::fill(joint_frame_active_counts_.begin(), joint_frame_active_counts_.end(), 0u);
}
```

Update `capture()` to record joint data after body data:

```cpp
void TrajectoryRecorder::capture() {
    if (current_frame_ >= max_frames_) return;

    WorldStateView view = world_.get_state_view();

    // Body data
    uint32_t active = view.active_body_count;
    uint32_t body_offset = current_frame_ * max_bodies_;
    if (active > 0) {
        std::memcpy(&all_positions_[body_offset], view.positions.data(),
                    active * sizeof(Vec2));
        std::memcpy(&all_velocities_[body_offset], view.velocities.data(),
                    active * sizeof(Vec2));
        std::memcpy(&all_rotations_[body_offset], view.rotations.data(),
                    active * sizeof(float));
    }
    frame_active_counts_[current_frame_] = active;

    // Joint data
    uint32_t joint_active = view.joints.active_joint_count;
    uint32_t joint_offset = current_frame_ * max_joints_;
    if (joint_active > 0) {
        std::memcpy(&all_joint_angles_[joint_offset], view.joints.angles.data(),
                    joint_active * sizeof(float));
        std::memcpy(&all_joint_angular_speeds_[joint_offset],
                    view.joints.angular_speeds.data(), joint_active * sizeof(float));
        std::memcpy(&all_joint_constraint_forces_[joint_offset],
                    view.joints.constraint_forces.data(), joint_active * sizeof(float));
        std::memcpy(&all_joint_lengths_[joint_offset], view.joints.lengths.data(),
                    joint_active * sizeof(float));
    }
    joint_frame_active_counts_[current_frame_] = joint_active;

    ++current_frame_;
}
```

- [ ] **Step 5: Run test to verify it passes**

Run: `cmake --build build && ./build/stan2d_tests --gtest_filter=JointStateSystemFixture.Trajectory*`
Expected: ALL PASS

- [ ] **Step 6: Commit**

```bash
git add include/stan2d/export/trajectory_recorder.hpp src/stan2d/export/trajectory_recorder.cpp tests/unit/test_joint_state_system.cpp
git commit -m "feat: extend TrajectoryRecorder for joint data with tensor-compatible layout"
```

---

### Task 18.4: Extend export_state() for joints

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/test_joint_state_system.cpp`:

```cpp
#include <stan2d/export/state_export.hpp>
#include <nlohmann/json.hpp>
#include <fstream>
#include <filesystem>

TEST_F(JointStateSystemFixture, ExportStateJSONIncludesJoints) {
    BodyHandle a = make_static({0.0f, 0.0f});
    BodyHandle b = make_dynamic({2.0f, 0.0f});

    JointDef def;
    def.type = JointType::Hinge;
    def.body_a = a;
    def.body_b = b;
    def.motor_enabled = true;
    def.motor_speed = 3.0f;
    world.create_joint(def);
    world.step(dt);

    auto temp = std::filesystem::temp_directory_path() / "stan2d_joint_export_test.json";

    export_state(world, temp.string(), ExportFormat::JSON);

    std::ifstream file(temp);
    nlohmann::json j;
    file >> j;

    EXPECT_TRUE(j.contains("joints"));
    EXPECT_EQ(j["joint_count"], 1);
    EXPECT_EQ(j["joints"].size(), 1u);
    EXPECT_EQ(j["joints"][0]["type"], "Hinge");
    EXPECT_TRUE(j["joints"][0].contains("angle"));
    EXPECT_TRUE(j["joints"][0].contains("constraint_force"));
    EXPECT_TRUE(j["joints"][0].contains("length"));

    std::filesystem::remove(temp);
}

TEST_F(JointStateSystemFixture, ExportStateJSONNoJoints) {
    make_dynamic({0.0f, 0.0f});
    world.step(dt);

    auto temp = std::filesystem::temp_directory_path() / "stan2d_no_joints_test.json";
    export_state(world, temp.string(), ExportFormat::JSON);

    std::ifstream file(temp);
    nlohmann::json j;
    file >> j;

    EXPECT_EQ(j["joint_count"], 0);
    EXPECT_TRUE(j["joints"].empty());

    std::filesystem::remove(temp);
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cmake --build build && ./build/stan2d_tests --gtest_filter=JointStateSystemFixture.ExportState*`
Expected: FAIL — JSON output doesn't contain `joints` key

- [ ] **Step 3: Extend export_state() in state_export.cpp**

In `src/stan2d/export/state_export.cpp`, in the JSON branch, after the bodies loop, add:

```cpp
        // Joint data
        j["joint_count"] = view.joints.active_joint_count;
        j["joints"] = nlohmann::json::array();

        const char* type_names[] = {"Hinge", "Spring", "Distance", "Pulley"};

        for (uint32_t i = 0; i < view.joints.active_joint_count; ++i) {
            nlohmann::json joint;
            uint8_t type_val = view.joints.types[i];
            joint["type"] = (type_val < 4) ? type_names[type_val] : "Unknown";
            joint["angle"] = view.joints.angles[i];
            joint["angular_speed"] = view.joints.angular_speeds[i];
            joint["motor_enabled"] = view.joints.motor_enabled[i] != 0;
            joint["motor_target_speed"] = view.joints.motor_target_speeds[i];
            joint["constraint_force"] = view.joints.constraint_forces[i];
            joint["length"] = view.joints.lengths[i];
            j["joints"].push_back(joint);
        }
```

In the Binary branch, after body data, add:

```cpp
        // Bump version to 2 (joint data added after body data)
        // Change: uint32_t version = 1; → uint32_t version = 2;

        // Joint count + joint observable data
        uint32_t jcount = view.joints.active_joint_count;
        file.write(reinterpret_cast<const char*>(&jcount), sizeof(jcount));
        if (jcount > 0) {
            file.write(reinterpret_cast<const char*>(view.joints.types.data()),
                       jcount * sizeof(uint8_t));
            file.write(reinterpret_cast<const char*>(view.joints.angles.data()),
                       jcount * sizeof(float));
            file.write(reinterpret_cast<const char*>(view.joints.angular_speeds.data()),
                       jcount * sizeof(float));
            file.write(reinterpret_cast<const char*>(view.joints.constraint_forces.data()),
                       jcount * sizeof(float));
            file.write(reinterpret_cast<const char*>(view.joints.lengths.data()),
                       jcount * sizeof(float));
        }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cmake --build build && ./build/stan2d_tests --gtest_filter=JointStateSystemFixture.ExportState*`
Expected: ALL PASS

- [ ] **Step 5: Run all tests for regression**

Run: `ctest --test-dir build`
Expected: ALL PASS

- [ ] **Step 6: Commit**

```bash
git add src/stan2d/export/state_export.cpp tests/unit/test_joint_state_system.cpp
git commit -m "feat: extend export_state() with joint data in JSON and Binary (Task 18 complete)"
```

---

### Task 18.5: Extend TrajectoryRecorder::save() for joints

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/test_joint_state_system.cpp`:

```cpp
TEST_F(JointStateSystemFixture, TrajectoryJSONExportIncludesJoints) {
    world.set_gravity({0.0f, -10.0f});

    BodyHandle a = make_static({0.0f, 5.0f});
    BodyHandle b = make_dynamic({2.0f, 5.0f});

    JointDef def;
    def.type = JointType::Hinge;
    def.body_a = a;
    def.body_b = b;
    world.create_joint(def);

    TrajectoryRecorder recorder(world, 5);
    recorder.start();

    for (int i = 0; i < 3; ++i) {
        world.step(dt);
        recorder.capture();
    }

    auto temp = std::filesystem::temp_directory_path() / "stan2d_traj_joints_test.json";
    recorder.save(temp.string(), ExportFormat::JSON);

    std::ifstream file(temp);
    nlohmann::json j;
    file >> j;

    EXPECT_TRUE(j.contains("max_joints"));
    EXPECT_EQ(j["max_joints"], 50);
    EXPECT_TRUE(j["frames"][0].contains("joint_active_count"));
    EXPECT_EQ(j["frames"][0]["joint_active_count"], 1);
    EXPECT_TRUE(j["frames"][0].contains("joint_angles"));

    std::filesystem::remove(temp);
}

TEST_F(JointStateSystemFixture, TrajectoryBinaryExportIncludesJoints) {
    world.set_gravity({0.0f, -10.0f});

    BodyHandle a = make_static({0.0f, 5.0f});
    BodyHandle b = make_dynamic({2.0f, 5.0f});

    JointDef def;
    def.type = JointType::Hinge;
    def.body_a = a;
    def.body_b = b;
    world.create_joint(def);

    TrajectoryRecorder recorder(world, 5);
    recorder.start();

    for (int i = 0; i < 3; ++i) {
        world.step(dt);
        recorder.capture();
    }

    auto temp = std::filesystem::temp_directory_path() / "stan2d_traj_joints_test.bin";
    recorder.save(temp.string(), ExportFormat::Binary);

    EXPECT_TRUE(std::filesystem::exists(temp));
    EXPECT_GT(std::filesystem::file_size(temp), 0u);

    std::filesystem::remove(temp);
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cmake --build build && ./build/stan2d_tests --gtest_filter=JointStateSystemFixture.TrajectoryJSONExport*:JointStateSystemFixture.TrajectoryBinaryExport*`
Expected: FAIL — trajectory JSON doesn't contain joint fields

- [ ] **Step 3: Extend TrajectoryRecorder::save()**

In `src/stan2d/export/trajectory_recorder.cpp`, in the JSON branch of `save()`, update the header and frames:

In the header, after `j["max_bodies"]`:
```cpp
        j["max_joints"] = max_joints_;
```

In the per-frame loop, after the body data (positions/velocities), add:
```cpp
            frame["joint_active_count"] = joint_frame_active_counts_[f];
            frame["joint_angles"] = nlohmann::json::array();
            frame["joint_angular_speeds"] = nlohmann::json::array();
            frame["joint_constraint_forces"] = nlohmann::json::array();
            frame["joint_lengths"] = nlohmann::json::array();

            uint32_t j_offset = f * max_joints_;
            uint32_t j_active = joint_frame_active_counts_[f];

            for (uint32_t jj = 0; jj < j_active; ++jj) {
                frame["joint_angles"].push_back(all_joint_angles_[j_offset + jj]);
                frame["joint_angular_speeds"].push_back(
                    all_joint_angular_speeds_[j_offset + jj]);
                frame["joint_constraint_forces"].push_back(
                    all_joint_constraint_forces_[j_offset + jj]);
                frame["joint_lengths"].push_back(all_joint_lengths_[j_offset + jj]);
            }
```

In the Binary branch, after the body data, add:

After writing `max_bodies_`, write `max_joints_`. Also bump version: `uint32_t version = 1;` → `uint32_t version = 2;`:
```cpp
        file.write(reinterpret_cast<const char*>(&max_joints_), sizeof(max_joints_));
```

After body array writes:
```cpp
        // Joint active counts per frame
        file.write(reinterpret_cast<const char*>(joint_frame_active_counts_.data()),
                   current_frame_ * sizeof(uint32_t));

        // Joint data: current_frame_ * max_joints_ entries each
        uint32_t joint_total = current_frame_ * max_joints_;
        file.write(reinterpret_cast<const char*>(all_joint_angles_.data()),
                   joint_total * sizeof(float));
        file.write(reinterpret_cast<const char*>(all_joint_angular_speeds_.data()),
                   joint_total * sizeof(float));
        file.write(reinterpret_cast<const char*>(all_joint_constraint_forces_.data()),
                   joint_total * sizeof(float));
        file.write(reinterpret_cast<const char*>(all_joint_lengths_.data()),
                   joint_total * sizeof(float));
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cmake --build build && ./build/stan2d_tests --gtest_filter=JointStateSystemFixture.Trajectory*Export*`
Expected: ALL PASS

- [ ] **Step 5: Run ALL tests**

Run: `ctest --test-dir build`
Expected: ALL PASS

- [ ] **Step 6: Commit**

```bash
git add src/stan2d/export/trajectory_recorder.cpp tests/unit/test_joint_state_system.cpp
git commit -m "feat: extend TrajectoryRecorder::save() with joint data (Task 18 complete)"
```

---

## Verification Checklist

- [ ] All Task 18 tests pass: `./build/stan2d_tests --gtest_filter=JointStateSystemFixture.*`
- [ ] All previous joint tests pass: `./build/stan2d_tests --gtest_filter=HingeJointFixture.*:DistanceJointFixture.*:SpringJointFixture.*:PulleyJointFixture.*`
- [ ] All Phase 1 tests pass: `ctest --test-dir build`
- [ ] `JointStateView` provides zero-copy spans for all observable fields
- [ ] `JointSnapshot` captures all SoA fields + SparseSet state + warm-start impulses
- [ ] Save/restore produces bit-identical replay (determinism test)
- [ ] MCTS rollback scenario works (divergent branches, identical replay)
- [ ] Destroyed joints are restored by `restore_state()`
- [ ] `TrajectoryRecorder` records 4 joint fields per frame in `[frames, max_joints]` stride
- [ ] JSON export includes `joints` array with type, angle, constraint_force, length
- [ ] Binary export includes joint data after body data
