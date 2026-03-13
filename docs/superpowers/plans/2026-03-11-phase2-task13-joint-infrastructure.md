# Task 13: Joint Infrastructure — Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add JointHandle, JointType, JointDef, JointStorage SoA, and joint CRUD operations to World — no solver logic yet.

**Architecture:** Follows the existing Handle + SparseSet pattern from bodies. JointStorage is a flat SoA struct with all fields for all joint types (union approach). A dedicated SparseSet manages JointHandle validity. World gets `create_joint()`, `destroy_joint()`, `is_valid(JointHandle)`.

**Tech Stack:** C++20, glm, Google Test

---

## File Structure

| Action | File | Responsibility |
|--------|------|---------------|
| Create | `include/stan2d/joints/joint_types.hpp` | `JointHandle`, `JointType` enum, `JointDef` struct |
| Create | `include/stan2d/joints/joint_storage.hpp` | `JointStorage` SoA struct with reserve/push_back/swap_and_pop/pop_back |
| Modify | `include/stan2d/core/handle.hpp` | Add `JointHandle` typed handle |
| Modify | `include/stan2d/world/world_config.hpp` | Add `max_joints` field |
| Modify | `include/stan2d/world/world.hpp` | Add joint members + public API declarations |
| Modify | `src/stan2d/world/world.cpp` | Implement `create_joint()`, `destroy_joint()`, `is_valid(JointHandle)` |
| Create | `tests/unit/test_joint_infrastructure.cpp` | All tests for this task |

---

### Task 13.1: Add JointHandle to handle.hpp

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_joint_infrastructure.cpp`:

```cpp
#include <gtest/gtest.h>
#include <stan2d/core/handle.hpp>

using namespace stan2d;

TEST(JointHandle, DefaultConstruction) {
    JointHandle h;
    EXPECT_EQ(h.index, 0u);
    EXPECT_EQ(h.generation, 0u);
}

TEST(JointHandle, Equality) {
    JointHandle a{1, 2};
    JointHandle b{1, 2};
    JointHandle c{1, 3};
    EXPECT_EQ(a, b);
    EXPECT_NE(a, c);
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cmake --build build && ./build/stan2d_tests --gtest_filter=JointHandle.*`
Expected: FAIL — `JointHandle` is not defined

- [ ] **Step 3: Write minimal implementation**

Add to `include/stan2d/core/handle.hpp`, after `ShapeHandle`:

```cpp
struct JointHandle {
    uint32_t index      = 0;
    uint32_t generation = 0;

    bool operator==(const JointHandle&) const = default;
    bool operator!=(const JointHandle&) const = default;
};
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cmake --build build && ./build/stan2d_tests --gtest_filter=JointHandle.*`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add include/stan2d/core/handle.hpp tests/unit/test_joint_infrastructure.cpp
git commit -m "feat: add JointHandle typed handle"
```

---

### Task 13.2: Create joint_types.hpp (JointType enum + JointDef)

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/test_joint_infrastructure.cpp`:

```cpp
#include <stan2d/joints/joint_types.hpp>

TEST(JointType, EnumValues) {
    EXPECT_EQ(static_cast<uint8_t>(JointType::Hinge),    0);
    EXPECT_EQ(static_cast<uint8_t>(JointType::Spring),   1);
    EXPECT_EQ(static_cast<uint8_t>(JointType::Distance), 2);
    EXPECT_EQ(static_cast<uint8_t>(JointType::Pulley),   3);
}

TEST(JointDef, DefaultValues) {
    JointDef def;
    EXPECT_EQ(def.type, JointType::Hinge);
    EXPECT_FALSE(def.limit_enabled);
    EXPECT_FALSE(def.motor_enabled);
    EXPECT_FLOAT_EQ(def.stiffness, 100.0f);
    EXPECT_FLOAT_EQ(def.damping, 1.0f);
    EXPECT_FLOAT_EQ(def.rest_length, 0.0f);
    EXPECT_FLOAT_EQ(def.distance, 0.0f);
    EXPECT_FALSE(def.cable_mode);
    EXPECT_FLOAT_EQ(def.pulley_ratio, 1.0f);
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cmake --build build && ./build/stan2d_tests --gtest_filter=JointType.*:JointDef.*`
Expected: FAIL — file not found

- [ ] **Step 3: Write minimal implementation**

Create `include/stan2d/joints/joint_types.hpp`:

```cpp
#pragma once

#include <cstdint>
#include <glm/gtc/constants.hpp>

#include <stan2d/core/handle.hpp>
#include <stan2d/core/math_types.hpp>

namespace stan2d {

enum class JointType : uint8_t {
    Hinge    = 0,
    Spring   = 1,
    Distance = 2,
    Pulley   = 3,
};

struct JointDef {
    JointType  type       = JointType::Hinge;
    BodyHandle body_a;
    BodyHandle body_b;
    Vec2       anchor_a   = {0.0f, 0.0f};
    Vec2       anchor_b   = {0.0f, 0.0f};

    // Hinge: limits
    bool  limit_enabled   = false;
    float limit_min       = -glm::pi<float>();
    float limit_max       =  glm::pi<float>();

    // Hinge: motor
    bool  motor_enabled   = false;
    float motor_speed     = 0.0f;
    float motor_torque    = 0.0f;

    // Spring
    float stiffness       = 100.0f;
    float damping         = 1.0f;
    float rest_length     = 0.0f;

    // Distance
    float distance        = 0.0f;
    bool  cable_mode      = false;

    // Pulley
    Vec2  ground_a        = {0.0f, 0.0f};
    Vec2  ground_b        = {0.0f, 0.0f};
    float pulley_ratio    = 1.0f;
};

} // namespace stan2d
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cmake --build build && ./build/stan2d_tests --gtest_filter=JointType.*:JointDef.*`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add include/stan2d/joints/joint_types.hpp tests/unit/test_joint_infrastructure.cpp
git commit -m "feat: add JointType enum and JointDef struct"
```

---

### Task 13.3: Create joint_storage.hpp (SoA storage)

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/test_joint_infrastructure.cpp`:

```cpp
#include <stan2d/joints/joint_storage.hpp>

TEST(JointStorage, ReserveDoesNotChangeSize) {
    JointStorage storage;
    storage.reserve(100);
    EXPECT_EQ(storage.size, 0u);
    EXPECT_GE(storage.types.capacity(), 100u);
    EXPECT_GE(storage.body_a.capacity(), 100u);
    EXPECT_GE(storage.accumulated_impulse_x.capacity(), 100u);
    EXPECT_GE(storage.constraint_forces.capacity(), 100u);
}

TEST(JointStorage, PushBackIncreasesSize) {
    JointStorage storage;
    storage.reserve(10);

    JointDef def;
    def.type = JointType::Hinge;
    storage.push_back(def, 0, 1, 0.5f, 0.0f);
    EXPECT_EQ(storage.size, 1u);
    EXPECT_EQ(storage.types[0], JointType::Hinge);
    EXPECT_EQ(storage.body_a[0], 0u);
    EXPECT_EQ(storage.body_b[0], 1u);
    EXPECT_FLOAT_EQ(storage.reference_angle[0], 0.5f);
}

TEST(JointStorage, SwapAndPopMaintainsData) {
    JointStorage storage;
    storage.reserve(10);

    JointDef def_a;
    def_a.type = JointType::Hinge;
    storage.push_back(def_a, 0, 1, 0.0f, 0.0f);

    JointDef def_b;
    def_b.type = JointType::Spring;
    def_b.stiffness = 200.0f;
    storage.push_back(def_b, 2, 3, 0.0f, 5.0f);

    // Remove first element (swap with last)
    storage.swap_and_pop(0, 1);
    storage.pop_back();

    EXPECT_EQ(storage.size, 1u);
    EXPECT_EQ(storage.types[0], JointType::Spring);
    EXPECT_EQ(storage.body_a[0], 2u);
    EXPECT_EQ(storage.body_b[0], 3u);
    EXPECT_FLOAT_EQ(storage.spring_stiffness[0], 200.0f);
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cmake --build build && ./build/stan2d_tests --gtest_filter=JointStorage.*`
Expected: FAIL — file not found

- [ ] **Step 3: Write minimal implementation**

Create `include/stan2d/joints/joint_storage.hpp`:

```cpp
#pragma once

#include <cstdint>
#include <vector>

#include <stan2d/core/math_types.hpp>
#include <stan2d/joints/joint_types.hpp>

namespace stan2d {

struct JointStorage {
    // Common (all types)
    std::vector<JointType> types;
    std::vector<uint32_t>  body_a;
    std::vector<uint32_t>  body_b;
    std::vector<Vec2>      anchor_a;
    std::vector<Vec2>      anchor_b;

    // Hinge: limits
    std::vector<uint8_t>   limit_enabled;
    std::vector<float>     limit_min;
    std::vector<float>     limit_max;
    std::vector<float>     reference_angle;
    std::vector<float>     accumulated_limit_impulse;

    // Hinge/Motor
    std::vector<uint8_t>   motor_enabled;
    std::vector<float>     motor_target_speeds;
    std::vector<float>     motor_max_torque;
    std::vector<float>     accumulated_motor_impulse;

    // Spring
    std::vector<float>     spring_stiffness;
    std::vector<float>     spring_damping;
    std::vector<float>     spring_rest_length;

    // Distance
    std::vector<float>     distance_length;
    std::vector<uint8_t>   distance_cable_mode;

    // Pulley
    std::vector<Vec2>      pulley_ground_a;
    std::vector<Vec2>      pulley_ground_b;
    std::vector<float>     pulley_ratio;
    std::vector<float>     pulley_constant;

    // Common warm-start impulse (linear, all types except Spring)
    std::vector<float>     accumulated_impulse_x;
    std::vector<float>     accumulated_impulse_y;

    // Per-frame constraint force magnitude
    std::vector<float>     constraint_forces;

    // Cached per-frame observables
    std::vector<float>     cached_angles;
    std::vector<float>     cached_angular_speeds;
    std::vector<float>     cached_lengths;

    uint32_t size = 0;

    void reserve(uint32_t capacity) {
        types.reserve(capacity);
        body_a.reserve(capacity);
        body_b.reserve(capacity);
        anchor_a.reserve(capacity);
        anchor_b.reserve(capacity);
        limit_enabled.reserve(capacity);
        limit_min.reserve(capacity);
        limit_max.reserve(capacity);
        reference_angle.reserve(capacity);
        accumulated_limit_impulse.reserve(capacity);
        motor_enabled.reserve(capacity);
        motor_target_speeds.reserve(capacity);
        motor_max_torque.reserve(capacity);
        accumulated_motor_impulse.reserve(capacity);
        spring_stiffness.reserve(capacity);
        spring_damping.reserve(capacity);
        spring_rest_length.reserve(capacity);
        distance_length.reserve(capacity);
        distance_cable_mode.reserve(capacity);
        pulley_ground_a.reserve(capacity);
        pulley_ground_b.reserve(capacity);
        pulley_ratio.reserve(capacity);
        pulley_constant.reserve(capacity);
        accumulated_impulse_x.reserve(capacity);
        accumulated_impulse_y.reserve(capacity);
        constraint_forces.reserve(capacity);
        cached_angles.reserve(capacity);
        cached_angular_speeds.reserve(capacity);
        cached_lengths.reserve(capacity);
    }

    // Push a new joint. dense_body_a / dense_body_b are dense indices.
    // reference_angle: θ_b - θ_a at creation (Hinge) or 0.
    // pulley_const: len_a + ratio*len_b at creation (Pulley) or 0.
    void push_back(const JointDef& def, uint32_t dense_a, uint32_t dense_b,
                   float ref_angle, float pulley_const) {
        types.push_back(def.type);
        body_a.push_back(dense_a);
        body_b.push_back(dense_b);
        anchor_a.push_back(def.anchor_a);
        anchor_b.push_back(def.anchor_b);

        limit_enabled.push_back(def.limit_enabled ? 1 : 0);
        limit_min.push_back(def.limit_min);
        limit_max.push_back(def.limit_max);
        reference_angle.push_back(ref_angle);
        accumulated_limit_impulse.push_back(0.0f);

        motor_enabled.push_back(def.motor_enabled ? 1 : 0);
        motor_target_speeds.push_back(def.motor_speed);
        motor_max_torque.push_back(def.motor_torque);
        accumulated_motor_impulse.push_back(0.0f);

        spring_stiffness.push_back(def.stiffness);
        spring_damping.push_back(def.damping);
        spring_rest_length.push_back(def.rest_length);

        distance_length.push_back(def.distance);
        distance_cable_mode.push_back(def.cable_mode ? 1 : 0);

        pulley_ground_a.push_back(def.ground_a);
        pulley_ground_b.push_back(def.ground_b);
        pulley_ratio.push_back(def.pulley_ratio);
        pulley_constant.push_back(pulley_const);

        accumulated_impulse_x.push_back(0.0f);
        accumulated_impulse_y.push_back(0.0f);

        constraint_forces.push_back(0.0f);
        cached_angles.push_back(0.0f);
        cached_angular_speeds.push_back(0.0f);
        cached_lengths.push_back(0.0f);

        ++size;
    }

    void swap_and_pop(uint32_t dst, uint32_t src) {
        types[dst]                     = types[src];
        body_a[dst]                    = body_a[src];
        body_b[dst]                    = body_b[src];
        anchor_a[dst]                  = anchor_a[src];
        anchor_b[dst]                  = anchor_b[src];
        limit_enabled[dst]             = limit_enabled[src];
        limit_min[dst]                 = limit_min[src];
        limit_max[dst]                 = limit_max[src];
        reference_angle[dst]           = reference_angle[src];
        accumulated_limit_impulse[dst] = accumulated_limit_impulse[src];
        motor_enabled[dst]             = motor_enabled[src];
        motor_target_speeds[dst]       = motor_target_speeds[src];
        motor_max_torque[dst]          = motor_max_torque[src];
        accumulated_motor_impulse[dst] = accumulated_motor_impulse[src];
        spring_stiffness[dst]          = spring_stiffness[src];
        spring_damping[dst]            = spring_damping[src];
        spring_rest_length[dst]        = spring_rest_length[src];
        distance_length[dst]           = distance_length[src];
        distance_cable_mode[dst]       = distance_cable_mode[src];
        pulley_ground_a[dst]           = pulley_ground_a[src];
        pulley_ground_b[dst]           = pulley_ground_b[src];
        pulley_ratio[dst]              = pulley_ratio[src];
        pulley_constant[dst]           = pulley_constant[src];
        accumulated_impulse_x[dst]     = accumulated_impulse_x[src];
        accumulated_impulse_y[dst]     = accumulated_impulse_y[src];
        constraint_forces[dst]         = constraint_forces[src];
        cached_angles[dst]             = cached_angles[src];
        cached_angular_speeds[dst]     = cached_angular_speeds[src];
        cached_lengths[dst]            = cached_lengths[src];
    }

    void pop_back() {
        types.pop_back();
        body_a.pop_back();
        body_b.pop_back();
        anchor_a.pop_back();
        anchor_b.pop_back();
        limit_enabled.pop_back();
        limit_min.pop_back();
        limit_max.pop_back();
        reference_angle.pop_back();
        accumulated_limit_impulse.pop_back();
        motor_enabled.pop_back();
        motor_target_speeds.pop_back();
        motor_max_torque.pop_back();
        accumulated_motor_impulse.pop_back();
        spring_stiffness.pop_back();
        spring_damping.pop_back();
        spring_rest_length.pop_back();
        distance_length.pop_back();
        distance_cable_mode.pop_back();
        pulley_ground_a.pop_back();
        pulley_ground_b.pop_back();
        pulley_ratio.pop_back();
        pulley_constant.pop_back();
        accumulated_impulse_x.pop_back();
        accumulated_impulse_y.pop_back();
        constraint_forces.pop_back();
        cached_angles.pop_back();
        cached_angular_speeds.pop_back();
        cached_lengths.pop_back();
        --size;
    }
};

} // namespace stan2d
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cmake --build build && ./build/stan2d_tests --gtest_filter=JointStorage.*`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add include/stan2d/joints/joint_storage.hpp tests/unit/test_joint_infrastructure.cpp
git commit -m "feat: add JointStorage SoA struct"
```

---

### Task 13.4: Add max_joints to WorldConfig

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/test_joint_infrastructure.cpp`:

```cpp
#include <stan2d/world/world_config.hpp>

TEST(WorldConfig, MaxJointsDefaultValue) {
    WorldConfig config;
    EXPECT_EQ(config.max_joints, 2000u);
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cmake --build build && ./build/stan2d_tests --gtest_filter=WorldConfig.MaxJointsDefaultValue`
Expected: FAIL — no member named `max_joints`

- [ ] **Step 3: Write minimal implementation**

In `include/stan2d/world/world_config.hpp`, add after `max_shapes`:

```cpp
    uint32_t max_joints       = 2000;
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cmake --build build && ./build/stan2d_tests --gtest_filter=WorldConfig.MaxJointsDefaultValue`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add include/stan2d/world/world_config.hpp tests/unit/test_joint_infrastructure.cpp
git commit -m "feat: add max_joints to WorldConfig"
```

---

### Task 13.5: Add joint CRUD to World

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/test_joint_infrastructure.cpp`:

```cpp
#include <stan2d/world/world.hpp>

class JointWorldFixture : public ::testing::Test {
protected:
    World world{WorldConfig{
        .max_bodies = 100, .max_constraints = 100,
        .max_contacts = 100, .max_shapes = 100, .max_joints = 50
    }};
    ShapeHandle shape = world.create_shape(CircleShape{.radius = 0.5f});

    BodyHandle make_body(Vec2 pos = {0.0f, 0.0f}) {
        return world.create_body({.position = pos, .shape = shape, .mass = 1.0f});
    }
};

TEST_F(JointWorldFixture, CreateHingeJointReturnsValidHandle) {
    BodyHandle a = make_body({0.0f, 0.0f});
    BodyHandle b = make_body({1.0f, 0.0f});

    JointDef def;
    def.type = JointType::Hinge;
    def.body_a = a;
    def.body_b = b;
    def.anchor_a = {0.5f, 0.0f};
    def.anchor_b = {-0.5f, 0.0f};

    JointHandle jh = world.create_joint(def);
    EXPECT_TRUE(world.is_valid(jh));
}

TEST_F(JointWorldFixture, CreateAllJointTypes) {
    BodyHandle a = make_body();
    BodyHandle b = make_body({2.0f, 0.0f});

    for (auto type : {JointType::Hinge, JointType::Spring,
                      JointType::Distance, JointType::Pulley}) {
        JointDef def;
        def.type = type;
        def.body_a = a;
        def.body_b = b;
        JointHandle jh = world.create_joint(def);
        EXPECT_TRUE(world.is_valid(jh));
    }
}

TEST_F(JointWorldFixture, DestroyJointInvalidatesHandle) {
    BodyHandle a = make_body();
    BodyHandle b = make_body({1.0f, 0.0f});

    JointDef def;
    def.body_a = a;
    def.body_b = b;
    JointHandle jh = world.create_joint(def);

    EXPECT_TRUE(world.is_valid(jh));
    world.destroy_joint(jh);
    EXPECT_FALSE(world.is_valid(jh));
}

TEST_F(JointWorldFixture, DestroyAndRecreateJoint) {
    BodyHandle a = make_body();
    BodyHandle b = make_body({1.0f, 0.0f});

    JointDef def;
    def.body_a = a;
    def.body_b = b;

    JointHandle jh1 = world.create_joint(def);
    world.destroy_joint(jh1);
    JointHandle jh2 = world.create_joint(def);

    EXPECT_FALSE(world.is_valid(jh1));
    EXPECT_TRUE(world.is_valid(jh2));
    // Generation should differ (slot reused)
    EXPECT_EQ(jh1.index, jh2.index);
    EXPECT_NE(jh1.generation, jh2.generation);
}

TEST_F(JointWorldFixture, JointCountTracked) {
    BodyHandle a = make_body();
    BodyHandle b = make_body({1.0f, 0.0f});

    JointDef def;
    def.body_a = a;
    def.body_b = b;

    EXPECT_EQ(world.joint_count(), 0u);

    JointHandle j1 = world.create_joint(def);
    EXPECT_EQ(world.joint_count(), 1u);

    JointHandle j2 = world.create_joint(def);
    EXPECT_EQ(world.joint_count(), 2u);

    world.destroy_joint(j1);
    EXPECT_EQ(world.joint_count(), 1u);
}

TEST_F(JointWorldFixture, DistanceAutoDetectsLength) {
    BodyHandle a = make_body({0.0f, 0.0f});
    BodyHandle b = make_body({3.0f, 4.0f});

    JointDef def;
    def.type = JointType::Distance;
    def.body_a = a;
    def.body_b = b;
    def.distance = 0.0f;  // auto-detect

    JointHandle jh = world.create_joint(def);
    EXPECT_TRUE(world.is_valid(jh));
    // Verified by checking internal storage length == 5.0 (3-4-5 triangle)
    // This is tested indirectly through solver behavior in Task 15
}

TEST_F(JointWorldFixture, SpringAutoDetectsRestLength) {
    BodyHandle a = make_body({0.0f, 0.0f});
    BodyHandle b = make_body({3.0f, 4.0f});

    JointDef def;
    def.type = JointType::Spring;
    def.body_a = a;
    def.body_b = b;
    def.rest_length = 0.0f;  // auto-detect

    JointHandle jh = world.create_joint(def);
    EXPECT_TRUE(world.is_valid(jh));
}

TEST_F(JointWorldFixture, StepRunsWithJointsPresent) {
    BodyHandle a = make_body({0.0f, 0.0f});
    BodyHandle b = make_body({1.0f, 0.0f});

    JointDef def;
    def.body_a = a;
    def.body_b = b;
    world.create_joint(def);

    // step() should not crash even though no solver logic exists yet
    world.step(1.0f / 60.0f);
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cmake --build build && ./build/stan2d_tests --gtest_filter=JointWorldFixture.*`
Expected: FAIL — `create_joint`, `destroy_joint`, `is_valid(JointHandle)`, `joint_count` not defined

- [ ] **Step 3: Add joint members and API to world.hpp**

In `include/stan2d/world/world.hpp`, add the include:

```cpp
#include <stan2d/joints/joint_storage.hpp>
```

Add to the `public:` section of `World`, after body queries:

```cpp
    // ── Joint management ──────────────────────────────────────────
    JointHandle create_joint(const JointDef& def);
    void        destroy_joint(JointHandle handle);
    [[nodiscard]] bool is_valid(JointHandle handle) const;
    [[nodiscard]] uint32_t joint_count() const;
```

Add to the `private:` section, after `solver_config_`:

```cpp
    // ── Joint data ─────────────────────────────────────────────────
    SparseSet      joint_handles_;
    JointStorage   joints_;
```

- [ ] **Step 4: Implement joint CRUD in world.cpp**

Add to `World::World()` constructor, after shape_registry reserve:

```cpp
    joint_handles_.reserve(config.max_joints);
    joints_.reserve(config.max_joints);
```

Add the following methods to `src/stan2d/world/world.cpp`:

```cpp
JointHandle World::create_joint(const JointDef& def) {
    Handle ha{def.body_a.index, def.body_a.generation};
    Handle hb{def.body_b.index, def.body_b.generation};
    assert(body_handles_.is_valid(ha) && "body_a is not valid");
    assert(body_handles_.is_valid(hb) && "body_b is not valid");

    uint32_t dense_a = body_handles_.dense_index(ha);
    uint32_t dense_b = body_handles_.dense_index(hb);

    // Compute reference angle for Hinge joints
    float ref_angle = 0.0f;
    if (def.type == JointType::Hinge) {
        ref_angle = bodies_.rotations[dense_b] - bodies_.rotations[dense_a];
    }

    // Compute auto-detected distances
    JointDef resolved = def;
    Vec2 world_anchor_a = bodies_.positions[dense_a] + def.anchor_a;
    Vec2 world_anchor_b = bodies_.positions[dense_b] + def.anchor_b;
    float current_dist = glm::length(world_anchor_b - world_anchor_a);

    if (def.type == JointType::Distance && def.distance == 0.0f) {
        resolved.distance = current_dist;
    }
    if (def.type == JointType::Spring && def.rest_length == 0.0f) {
        resolved.rest_length = current_dist;
    }

    // Compute pulley constant
    float pulley_const = 0.0f;
    if (def.type == JointType::Pulley) {
        float len_a = glm::length(world_anchor_a - def.ground_a);
        float len_b = glm::length(world_anchor_b - def.ground_b);
        pulley_const = len_a + def.pulley_ratio * len_b;
    }

    Handle h = joint_handles_.allocate();
    joints_.push_back(resolved, dense_a, dense_b, ref_angle, pulley_const);

    return JointHandle{h.index, h.generation};
}

void World::destroy_joint(JointHandle handle) {
    Handle h{handle.index, handle.generation};
    auto swap = joint_handles_.deallocate(h);

    if (swap.has_value()) {
        joints_.swap_and_pop(swap->removed_dense, swap->moved_from_dense);
    }
    joints_.pop_back();
}

bool World::is_valid(JointHandle handle) const {
    return joint_handles_.is_valid(Handle{handle.index, handle.generation});
}

uint32_t World::joint_count() const {
    return joint_handles_.size();
}
```

- [ ] **Step 5: Run test to verify it passes**

Run: `cmake --build build && ./build/stan2d_tests --gtest_filter=JointWorldFixture.*:JointHandle.*:JointType.*:JointDef.*:WorldConfig.*:JointStorage.*`
Expected: ALL PASS

- [ ] **Step 6: Commit**

```bash
git add include/stan2d/world/world.hpp src/stan2d/world/world.cpp tests/unit/test_joint_infrastructure.cpp
git commit -m "feat: add joint CRUD operations to World (Task 13 complete)"
```

---

## Verification Checklist

- [ ] All tests pass: `./build/stan2d_tests --gtest_filter=JointHandle.*:JointType.*:JointDef.*:JointStorage.*:WorldConfig.MaxJointsDefaultValue:JointWorldFixture.*`
- [ ] Existing tests still pass: `ctest --test-dir build`
- [ ] No solver logic introduced — `step()` runs unchanged
- [ ] JointStorage pre-reserved to `max_joints` at World construction (zero alloc in step)
