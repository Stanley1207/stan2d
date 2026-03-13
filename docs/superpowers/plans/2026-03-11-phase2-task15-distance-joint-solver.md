# Task 15: Distance Joint Solver — Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Distance joint constraint to the joint solver — a 1D scalar constraint along the anchor-to-anchor axis with Baumgarte position correction and cable mode.

**Architecture:** Distance joint is a scalar impulse along the line connecting the two world-space anchor points. Uses the same accumulated-impulse clamping pattern as contacts. `cable_mode=false` (rigid rod) = two-sided constraint (no clamp). `cable_mode=true` (cable) = one-sided, pull-only (clamp accumulated impulse ≥ 0). Baumgarte bias corrects position drift.

**Tech Stack:** C++20, glm, Google Test

**Depends on:** Task 13 (infrastructure), Task 14 (solver integration)

---

## File Structure

| Action | File | Responsibility |
|--------|------|---------------|
| Modify | `src/stan2d/joints/joint_solver.cpp` | Add Distance case to `solve_joints()` and warm-start |
| Create | `tests/unit/test_distance_joint.cpp` | All Distance joint tests |

---

### Task 15.1: Distance joint basic constraint

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_distance_joint.cpp`:

```cpp
#include <gtest/gtest.h>
#include <cmath>
#include <stan2d/world/world.hpp>

using namespace stan2d;

class DistanceJointFixture : public ::testing::Test {
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

TEST_F(DistanceJointFixture, RigidRodMaintainsDistance) {
    world.set_gravity({0.0f, -10.0f});

    BodyHandle anchor = make_static({0.0f, 10.0f});
    BodyHandle bob = make_dynamic({3.0f, 10.0f});

    JointDef def;
    def.type = JointType::Distance;
    def.body_a = anchor;
    def.body_b = bob;
    def.distance = 0.0f;  // auto-detect: 3.0

    JointHandle jh = world.create_joint(def);

    for (int i = 0; i < 300; ++i) {
        world.step(dt);
    }

    // Bob should maintain ~3.0 distance from anchor
    float len = world.get_joint_length(jh);
    EXPECT_NEAR(len, 3.0f, 0.5f);  // iterative solver tolerance
}

TEST_F(DistanceJointFixture, TwoBodyDistanceHold) {
    world.set_gravity({0.0f, -10.0f});

    BodyHandle a = make_dynamic({0.0f, 10.0f}, 2.0f);
    BodyHandle b = make_dynamic({4.0f, 10.0f}, 1.0f);

    JointDef def;
    def.type = JointType::Distance;
    def.body_a = a;
    def.body_b = b;
    def.distance = 4.0f;

    JointHandle jh = world.create_joint(def);

    for (int i = 0; i < 120; ++i) {
        world.step(dt);
    }

    float len = world.get_joint_length(jh);
    EXPECT_NEAR(len, 4.0f, 0.6f);
}

TEST_F(DistanceJointFixture, ExternalForceResistance) {
    world.set_gravity({0.0f, 0.0f});

    BodyHandle a = make_static({0.0f, 0.0f});
    BodyHandle b = make_dynamic({5.0f, 0.0f});

    JointDef def;
    def.type = JointType::Distance;
    def.body_a = a;
    def.body_b = b;
    def.distance = 5.0f;

    JointHandle jh = world.create_joint(def);

    // Apply force pulling bob away
    for (int i = 0; i < 60; ++i) {
        world.apply_force(b, {100.0f, 0.0f});
        world.step(dt);
    }

    // Distance should still be approximately maintained
    float len = world.get_joint_length(jh);
    EXPECT_NEAR(len, 5.0f, 1.0f);
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cmake --build build && ./build/stan2d_tests --gtest_filter=DistanceJointFixture.*`
Expected: FAIL — Distance case not implemented in solve_joints

- [ ] **Step 3: Add Distance case to solve_joints()**

In `src/stan2d/joints/joint_solver.cpp`, inside `solve_joints()`, after the Hinge block's closing brace, add the Distance case:

```cpp
        if (joints.types[i] == JointType::Distance) {
            uint32_t a = joints.body_a[i];
            uint32_t b = joints.body_b[i];

            float inv_ma = bodies.inverse_masses[a];
            float inv_mb = bodies.inverse_masses[b];
            float inv_ia = bodies.inverse_inertias[a];
            float inv_ib = bodies.inverse_inertias[b];

            float cos_a = std::cos(bodies.rotations[a]);
            float sin_a = std::sin(bodies.rotations[a]);
            float cos_b = std::cos(bodies.rotations[b]);
            float sin_b = std::sin(bodies.rotations[b]);

            Vec2 ra = {cos_a * joints.anchor_a[i].x - sin_a * joints.anchor_a[i].y,
                       sin_a * joints.anchor_a[i].x + cos_a * joints.anchor_a[i].y};
            Vec2 rb = {cos_b * joints.anchor_b[i].x - sin_b * joints.anchor_b[i].y,
                       sin_b * joints.anchor_b[i].x + cos_b * joints.anchor_b[i].y};

            Vec2 world_a = bodies.positions[a] + ra;
            Vec2 world_b = bodies.positions[b] + rb;
            Vec2 delta = world_b - world_a;
            float current_len = glm::length(delta);

            if (current_len < 1e-6f) continue;

            Vec2 axis = delta / current_len;

            // Effective mass along axis
            float rn_a = cross_vec_vec(ra, axis);
            float rn_b = cross_vec_vec(rb, axis);
            float k = inv_ma + inv_mb + inv_ia * rn_a * rn_a + inv_ib * rn_b * rn_b;
            if (k < 1e-12f) continue;
            float eff_mass = 1.0f / k;

            // Relative velocity along axis
            Vec2 v_rel = (bodies.velocities[b]
                         + cross_scalar_vec(bodies.angular_velocities[b], rb))
                       - (bodies.velocities[a]
                        + cross_scalar_vec(bodies.angular_velocities[a], ra));
            float v_along = glm::dot(v_rel, axis);

            // Baumgarte position correction
            float c_val = current_len - joints.distance_length[i];
            float bias = config.baumgarte / dt * c_val;

            float lambda = eff_mass * (-v_along - bias);

            // Cable mode: one-sided (pull only, clamp accumulated >= 0)
            // Rigid rod: two-sided (no clamp)
            if (joints.distance_cable_mode[i]) {
                float old_acc_x = joints.accumulated_impulse_x[i];
                // Project accumulated impulse onto axis (scalar)
                float old_acc = old_acc_x;  // We'll track scalar in x
                float new_acc = glm::max(old_acc + lambda, 0.0f);
                lambda = new_acc - old_acc;
                joints.accumulated_impulse_x[i] = new_acc;
                joints.accumulated_impulse_y[i] = 0.0f;
            } else {
                joints.accumulated_impulse_x[i] += lambda;
                joints.accumulated_impulse_y[i] = 0.0f;
            }

            Vec2 impulse = axis * lambda;

            bodies.velocities[a] = bodies.velocities[a] - impulse * inv_ma;
            bodies.angular_velocities[a] -= inv_ia * cross_vec_vec(ra, impulse);

            bodies.velocities[b] = bodies.velocities[b] + impulse * inv_mb;
            bodies.angular_velocities[b] += inv_ib * cross_vec_vec(rb, impulse);

            joints.constraint_forces[i] += std::abs(lambda);
        }
```

Also update `warm_start_joints()` to handle Distance joints. The existing warm-start code already applies `accumulated_impulse_x/y` for all non-Spring types. However, for Distance joints the accumulated impulse is stored as a scalar in `accumulated_impulse_x` and must be projected onto the current axis. Update the warm-start to handle this properly:

In `warm_start_joints()`, the existing generic warm-start code already applies the 2D impulse vector. For Distance joints, we store the scalar in `accumulated_impulse_x` and zero in `y`. The warm-start needs to project this along the current axis. Replace the warm-start body with:

```cpp
void warm_start_joints(const JointStorage& joints, BodyStorage& bodies) {
    for (uint32_t i = 0; i < joints.size; ++i) {
        if (joints.types[i] == JointType::Spring) continue;

        uint32_t a = joints.body_a[i];
        uint32_t b = joints.body_b[i];

        float cos_a = std::cos(bodies.rotations[a]);
        float sin_a = std::sin(bodies.rotations[a]);
        float cos_b = std::cos(bodies.rotations[b]);
        float sin_b = std::sin(bodies.rotations[b]);

        Vec2 ra = {cos_a * joints.anchor_a[i].x - sin_a * joints.anchor_a[i].y,
                   sin_a * joints.anchor_a[i].x + cos_a * joints.anchor_a[i].y};
        Vec2 rb = {cos_b * joints.anchor_b[i].x - sin_b * joints.anchor_b[i].y,
                   sin_b * joints.anchor_b[i].x + cos_b * joints.anchor_b[i].y};

        Vec2 impulse;
        if (joints.types[i] == JointType::Distance ||
            joints.types[i] == JointType::Pulley) {
            // Scalar impulse stored in accumulated_impulse_x — project onto current axis
            Vec2 world_a = bodies.positions[a] + ra;
            Vec2 world_b = bodies.positions[b] + rb;
            Vec2 delta = world_b - world_a;
            float len = glm::length(delta);
            Vec2 axis = (len > 1e-6f) ? delta / len : Vec2{1.0f, 0.0f};
            impulse = axis * joints.accumulated_impulse_x[i];
        } else {
            // Hinge: 2D impulse stored directly
            impulse = {joints.accumulated_impulse_x[i], joints.accumulated_impulse_y[i]};
        }

        bodies.velocities[a] = bodies.velocities[a]
                              - impulse * bodies.inverse_masses[a];
        bodies.angular_velocities[a] -= bodies.inverse_inertias[a]
                                       * cross_vec_vec(ra, impulse);

        bodies.velocities[b] = bodies.velocities[b]
                              + impulse * bodies.inverse_masses[b];
        bodies.angular_velocities[b] += bodies.inverse_inertias[b]
                                       * cross_vec_vec(rb, impulse);

        // Warm-start angular impulses (Hinge only)
        if (joints.types[i] == JointType::Hinge) {
            float limit_imp = joints.accumulated_limit_impulse[i];
            float motor_imp = joints.accumulated_motor_impulse[i];
            float total_angular = limit_imp + motor_imp;

            bodies.angular_velocities[a] -= bodies.inverse_inertias[a] * total_angular;
            bodies.angular_velocities[b] += bodies.inverse_inertias[b] * total_angular;
        }
    }
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cmake --build build && ./build/stan2d_tests --gtest_filter=DistanceJointFixture.*`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/stan2d/joints/joint_solver.cpp tests/unit/test_distance_joint.cpp
git commit -m "feat: implement Distance joint solver with Baumgarte bias"
```

---

### Task 15.2: Cable mode tests

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/test_distance_joint.cpp`:

```cpp
TEST_F(DistanceJointFixture, CableModeAllowsCompression) {
    world.set_gravity({0.0f, 0.0f});

    BodyHandle a = make_static({0.0f, 0.0f});
    BodyHandle b = make_dynamic({5.0f, 0.0f});

    JointDef def;
    def.type = JointType::Distance;
    def.body_a = a;
    def.body_b = b;
    def.distance = 5.0f;
    def.cable_mode = true;  // pull only

    world.create_joint(def);

    // Push bob TOWARD anchor (compression) — cable should allow this
    for (int i = 0; i < 60; ++i) {
        world.apply_force(b, {-50.0f, 0.0f});
        world.step(dt);
    }

    Vec2 bob_pos = world.get_position(b);
    // Bob should have moved closer than 5.0 (cable allows compression)
    EXPECT_LT(bob_pos.x, 4.5f);
}

TEST_F(DistanceJointFixture, CableModePreventsExtension) {
    world.set_gravity({0.0f, 0.0f});

    BodyHandle a = make_static({0.0f, 0.0f});
    BodyHandle b = make_dynamic({5.0f, 0.0f});

    JointDef def;
    def.type = JointType::Distance;
    def.body_a = a;
    def.body_b = b;
    def.distance = 5.0f;
    def.cable_mode = true;

    JointHandle jh = world.create_joint(def);

    // Push bob AWAY from anchor — cable should resist
    for (int i = 0; i < 120; ++i) {
        world.apply_force(b, {100.0f, 0.0f});
        world.step(dt);
    }

    float len = world.get_joint_length(jh);
    // Cable should prevent bob from going much beyond 5.0
    EXPECT_NEAR(len, 5.0f, 1.0f);
}

TEST_F(DistanceJointFixture, RigidRodResistsBothDirections) {
    world.set_gravity({0.0f, 0.0f});

    BodyHandle a = make_static({0.0f, 0.0f});
    BodyHandle b = make_dynamic({5.0f, 0.0f});

    JointDef def;
    def.type = JointType::Distance;
    def.body_a = a;
    def.body_b = b;
    def.distance = 5.0f;
    def.cable_mode = false;  // rigid rod

    JointHandle jh = world.create_joint(def);

    // Push bob toward anchor — rod should resist compression
    for (int i = 0; i < 60; ++i) {
        world.apply_force(b, {-50.0f, 0.0f});
        world.step(dt);
    }

    float len = world.get_joint_length(jh);
    EXPECT_NEAR(len, 5.0f, 0.8f);
}
```

- [ ] **Step 2: Run test to verify they pass**

Run: `cmake --build build && ./build/stan2d_tests --gtest_filter=DistanceJointFixture.*`
Expected: ALL PASS

- [ ] **Step 3: Run all tests for regression**

Run: `ctest --test-dir build`
Expected: ALL PASS

- [ ] **Step 4: Commit**

```bash
git add tests/unit/test_distance_joint.cpp
git commit -m "test: add cable mode and rigid rod tests for Distance joint (Task 15 complete)"
```

---

## Verification Checklist

- [ ] All Distance tests pass: `./build/stan2d_tests --gtest_filter=DistanceJointFixture.*`
- [ ] All Hinge tests still pass: `./build/stan2d_tests --gtest_filter=HingeJointFixture.*`
- [ ] All existing tests pass: `ctest --test-dir build`
- [ ] Rigid rod maintains distance against external forces
- [ ] Cable mode allows compression but prevents extension
- [ ] `get_joint_length()` returns accurate anchor-to-anchor distance
- [ ] Baumgarte bias corrects position drift over time
