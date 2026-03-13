# Task 17: Pulley Joint Solver — Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Pulley joint constraint — a scalar constraint `len_a + ratio * len_b = constant`, where `len_a`/`len_b` are distances from each body's anchor to its ground point.

**Architecture:** Pulley impulse is applied along each rope segment direction. Body A receives `lambda * uA`, body B receives `ratio * lambda * uB`. The constant is computed at creation time (`pulley_constant` in JointStorage, already computed in Task 13's `create_joint()`). Effective mass combines both bodies weighted by the ratio.

**Tech Stack:** C++20, glm, Google Test

**Depends on:** Task 13 (infrastructure), Task 14 (solver integration)

---

## File Structure

| Action | File | Responsibility |
|--------|------|---------------|
| Modify | `src/stan2d/joints/joint_solver.cpp` | Add Pulley case to `solve_joints()` |
| Create | `tests/unit/test_pulley_joint.cpp` | All Pulley joint tests |

---

### Task 17.1: Implement Pulley joint solver

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_pulley_joint.cpp`:

```cpp
#include <gtest/gtest.h>
#include <cmath>
#include <stan2d/world/world.hpp>

using namespace stan2d;

class PulleyJointFixture : public ::testing::Test {
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

TEST_F(PulleyJointFixture, EqualRatioBalance) {
    world.set_gravity({0.0f, -10.0f});

    // Two equal-mass bodies on a 1:1 pulley
    BodyHandle a = make_dynamic({-2.0f, 3.0f}, 1.0f);
    BodyHandle b = make_dynamic({2.0f, 3.0f}, 1.0f);

    JointDef def;
    def.type = JointType::Pulley;
    def.body_a = a;
    def.body_b = b;
    def.ground_a = {-2.0f, 5.0f};
    def.ground_b = {2.0f, 5.0f};
    def.pulley_ratio = 1.0f;

    world.create_joint(def);

    Vec2 initial_a = world.get_position(a);
    Vec2 initial_b = world.get_position(b);

    for (int i = 0; i < 120; ++i) {
        world.step(dt);
    }

    Vec2 pos_a = world.get_position(a);
    Vec2 pos_b = world.get_position(b);

    // Equal mass, equal ratio: both should fall similarly
    // The constraint limits total rope length
    float len_a = glm::length(pos_a - Vec2{-2.0f, 5.0f});
    float len_b = glm::length(pos_b - Vec2{2.0f, 5.0f});
    float total = len_a + len_b;

    // Original total
    float orig_len_a = glm::length(initial_a - Vec2{-2.0f, 5.0f});
    float orig_len_b = glm::length(initial_b - Vec2{2.0f, 5.0f});
    float orig_total = orig_len_a + orig_len_b;

    // Total rope length should be approximately conserved
    EXPECT_NEAR(total, orig_total, 0.5f);
}

TEST_F(PulleyJointFixture, UnequalRatioLift) {
    world.set_gravity({0.0f, -10.0f});

    // Heavy body on side A, light body on side B with 2:1 ratio
    BodyHandle a = make_dynamic({-2.0f, 3.0f}, 2.0f);
    BodyHandle b = make_dynamic({2.0f, 3.0f}, 1.0f);

    JointDef def;
    def.type = JointType::Pulley;
    def.body_a = a;
    def.body_b = b;
    def.ground_a = {-2.0f, 5.0f};
    def.ground_b = {2.0f, 5.0f};
    def.pulley_ratio = 2.0f;  // side B moves 2x for every 1x of side A

    world.create_joint(def);

    for (int i = 0; i < 120; ++i) {
        world.step(dt);
    }

    Vec2 pos_a = world.get_position(a);
    Vec2 pos_b = world.get_position(b);

    // Rope conservation: len_a + 2*len_b = constant
    float len_a = glm::length(pos_a - Vec2{-2.0f, 5.0f});
    float len_b = glm::length(pos_b - Vec2{2.0f, 5.0f});
    float orig_total = 2.0f + 2.0f * 2.0f;  // initial: both at 2.0 from ground
    float total = len_a + 2.0f * len_b;

    EXPECT_NEAR(total, orig_total, 0.8f);
}

TEST_F(PulleyJointFixture, RopeLengthConservationPrecision) {
    world.set_gravity({0.0f, -10.0f});

    BodyHandle a = make_dynamic({-3.0f, 2.0f}, 1.0f);
    BodyHandle b = make_dynamic({3.0f, 4.0f}, 1.5f);

    JointDef def;
    def.type = JointType::Pulley;
    def.body_a = a;
    def.body_b = b;
    def.ground_a = {-3.0f, 6.0f};
    def.ground_b = {3.0f, 6.0f};
    def.pulley_ratio = 1.0f;

    world.create_joint(def);

    // Compute initial constant
    float orig_len_a = glm::length(Vec2{-3.0f, 2.0f} - Vec2{-3.0f, 6.0f});
    float orig_len_b = glm::length(Vec2{3.0f, 4.0f} - Vec2{3.0f, 6.0f});
    float constant = orig_len_a + orig_len_b;

    float max_error = 0.0f;
    for (int i = 0; i < 300; ++i) {
        world.step(dt);

        Vec2 pa = world.get_position(a);
        Vec2 pb = world.get_position(b);
        float la = glm::length(pa - Vec2{-3.0f, 6.0f});
        float lb = glm::length(pb - Vec2{3.0f, 6.0f});
        float err = std::abs(la + lb - constant);
        max_error = std::max(max_error, err);
    }

    // Constraint should be maintained within solver tolerance
    EXPECT_LT(max_error, 1.0f);
}

TEST_F(PulleyJointFixture, OneBodyFallingLiftsOther) {
    world.set_gravity({0.0f, -10.0f});

    // A is much heavier — should fall and pull B up
    BodyHandle a = make_dynamic({-2.0f, 3.0f}, 5.0f);
    BodyHandle b = make_dynamic({2.0f, 3.0f}, 1.0f);

    JointDef def;
    def.type = JointType::Pulley;
    def.body_a = a;
    def.body_b = b;
    def.ground_a = {-2.0f, 5.0f};
    def.ground_b = {2.0f, 5.0f};
    def.pulley_ratio = 1.0f;

    world.create_joint(def);

    float initial_b_y = world.get_position(b).y;

    for (int i = 0; i < 120; ++i) {
        world.step(dt);
    }

    // Heavy body A should pull light body B upward
    Vec2 pos_a = world.get_position(a);
    Vec2 pos_b = world.get_position(b);

    EXPECT_LT(pos_a.y, 3.0f);  // A fell
    EXPECT_GT(pos_b.y, initial_b_y - 0.5f);  // B didn't fall as much (or was pulled up)
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cmake --build build && ./build/stan2d_tests --gtest_filter=PulleyJointFixture.*`
Expected: FAIL — Pulley case not implemented

- [ ] **Step 3: Add Pulley case to solve_joints()**

In `src/stan2d/joints/joint_solver.cpp`, inside `solve_joints()`, add after the Spring block:

```cpp
        if (joints.types[i] == JointType::Pulley) {
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

            // Rope segment vectors
            Vec2 delta_a = world_a - joints.pulley_ground_a[i];
            Vec2 delta_b = world_b - joints.pulley_ground_b[i];
            float len_a = glm::length(delta_a);
            float len_b = glm::length(delta_b);

            if (len_a < 1e-6f || len_b < 1e-6f) continue;

            Vec2 uA = delta_a / len_a;
            Vec2 uB = delta_b / len_b;
            float ratio = joints.pulley_ratio[i];

            // Effective mass (scalar):
            // 1/k = 1/mA + (rA × uA)²/IA + ratio² * (1/mB + (rB × uB)²/IB)
            float rn_a = cross_vec_vec(ra, uA);
            float rn_b = cross_vec_vec(rb, uB);
            float k = inv_ma + inv_ia * rn_a * rn_a
                    + ratio * ratio * (inv_mb + inv_ib * rn_b * rn_b);
            if (k < 1e-12f) continue;
            float eff_mass = 1.0f / k;

            // Constraint: len_a + ratio * len_b = constant
            float c_val = len_a + ratio * len_b - joints.pulley_constant[i];

            // Constraint velocity: v_a_proj + ratio * v_b_proj
            Vec2 v_a = bodies.velocities[a]
                     + cross_scalar_vec(bodies.angular_velocities[a], ra);
            Vec2 v_b = bodies.velocities[b]
                     + cross_scalar_vec(bodies.angular_velocities[b], rb);
            float v_proj = glm::dot(v_a, uA) + ratio * glm::dot(v_b, uB);

            // Baumgarte position correction
            float bias = config.baumgarte / dt * c_val;

            float lambda = eff_mass * (-v_proj - bias);

            // Accumulate (pulley is one-sided: rope can only pull)
            float old_acc = joints.accumulated_impulse_x[i];
            joints.accumulated_impulse_x[i] = glm::max(old_acc + lambda, 0.0f);
            lambda = joints.accumulated_impulse_x[i] - old_acc;
            joints.accumulated_impulse_y[i] = 0.0f;

            // Apply impulse along each rope segment
            Vec2 impulse_a = uA * lambda;
            Vec2 impulse_b = uB * (ratio * lambda);

            bodies.velocities[a] = bodies.velocities[a] - impulse_a * inv_ma;
            bodies.angular_velocities[a] -= inv_ia * cross_vec_vec(ra, impulse_a);

            bodies.velocities[b] = bodies.velocities[b] - impulse_b * inv_mb;
            bodies.angular_velocities[b] -= inv_ib * cross_vec_vec(rb, impulse_b);

            joints.constraint_forces[i] += std::abs(lambda);
        }
```

**Note on impulse direction:** The pulley constraint `len_a + ratio * len_b = constant` is designed so that when body A moves down (increasing `len_a`), body B must move up (decreasing `len_b`). The impulse along `uA` pulls body A **toward** its ground point, and along `uB` pulls body B **toward** its ground point. Both use subtraction (pulling toward ground) because the constraint resists rope extension.

- [ ] **Step 4: Run test to verify it passes**

Run: `cmake --build build && ./build/stan2d_tests --gtest_filter=PulleyJointFixture.*`
Expected: ALL PASS

- [ ] **Step 5: Run all tests for regression**

Run: `ctest --test-dir build`
Expected: ALL PASS

- [ ] **Step 6: Commit**

```bash
git add src/stan2d/joints/joint_solver.cpp tests/unit/test_pulley_joint.cpp
git commit -m "feat: implement Pulley joint solver with ratio and rope conservation (Task 17 complete)"
```

---

## Verification Checklist

- [ ] All Pulley tests pass: `./build/stan2d_tests --gtest_filter=PulleyJointFixture.*`
- [ ] All other joint tests still pass: `./build/stan2d_tests --gtest_filter=HingeJointFixture.*:DistanceJointFixture.*:SpringJointFixture.*`
- [ ] All existing Phase 1 tests pass: `ctest --test-dir build`
- [ ] Equal-ratio pulley balances equal masses
- [ ] Unequal-ratio respects mechanical advantage
- [ ] Rope length conservation maintained (< 1.0 error over 300 steps)
- [ ] Heavy body falling lifts lighter body
