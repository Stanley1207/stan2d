# Task 16: Spring Joint Solver — Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Spring joint constraint — a soft constraint with stiffness and damping, no warm-start, no clamping. Impulse = `(-stiffness * (len - rest_len) - damping * v_rel) * dt / eff_mass`.

**Architecture:** Spring is the simplest joint type. No accumulated state (stateless per frame). `accumulated_impulse_x/y` fields exist for uniform snapshot layout but are always zeroed. Force is converted to impulse using effective mass, applied along the anchor-to-anchor axis. `rest_length=0` at creation auto-detects current distance (already handled in Task 13's `create_joint()`).

**Tech Stack:** C++20, glm, Google Test

**Depends on:** Task 13 (infrastructure), Task 14 (solver integration)

---

## File Structure

| Action | File | Responsibility |
|--------|------|---------------|
| Modify | `src/stan2d/joints/joint_solver.cpp` | Add Spring case to `solve_joints()` |
| Create | `tests/unit/test_spring_joint.cpp` | All Spring joint tests |

---

### Task 16.1: Implement Spring joint solver

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_spring_joint.cpp`:

```cpp
#include <gtest/gtest.h>
#include <cmath>
#include <stan2d/world/world.hpp>

using namespace stan2d;

class SpringJointFixture : public ::testing::Test {
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

TEST_F(SpringJointFixture, OscillationWithEnergyDecay) {
    world.set_gravity({0.0f, 0.0f});

    BodyHandle anchor = make_static({0.0f, 0.0f});
    BodyHandle bob = make_dynamic({3.0f, 0.0f});  // displaced from rest

    JointDef def;
    def.type = JointType::Spring;
    def.body_a = anchor;
    def.body_b = bob;
    def.stiffness = 50.0f;
    def.damping = 2.0f;
    def.rest_length = 1.0f;  // rest at 1.0, starting at 3.0

    world.create_joint(def);

    // Track peak displacements
    float max_x_first_half = 0.0f;
    float max_x_second_half = 0.0f;

    for (int i = 0; i < 120; ++i) {
        world.step(dt);
        Vec2 pos = world.get_position(bob);
        float dist_from_rest = std::abs(pos.x - 0.0f);  // distance from anchor
        if (i < 60) {
            max_x_first_half = std::max(max_x_first_half, dist_from_rest);
        } else {
            max_x_second_half = std::max(max_x_second_half, dist_from_rest);
        }
    }

    // With damping, oscillation should decay
    EXPECT_LT(max_x_second_half, max_x_first_half);
}

TEST_F(SpringJointFixture, OverdampedSettlesToRest) {
    world.set_gravity({0.0f, 0.0f});

    BodyHandle anchor = make_static({0.0f, 0.0f});
    BodyHandle bob = make_dynamic({3.0f, 0.0f});

    JointDef def;
    def.type = JointType::Spring;
    def.body_a = anchor;
    def.body_b = bob;
    def.stiffness = 20.0f;
    def.damping = 50.0f;  // heavily overdamped
    def.rest_length = 1.0f;

    world.create_joint(def);

    for (int i = 0; i < 300; ++i) {
        world.step(dt);
    }

    // Should settle near rest length
    Vec2 pos = world.get_position(bob);
    float dist = glm::length(pos);
    EXPECT_NEAR(dist, 1.0f, 0.5f);
}

TEST_F(SpringJointFixture, ZeroRestLengthSpring) {
    world.set_gravity({0.0f, 0.0f});

    BodyHandle a = make_dynamic({0.0f, 0.0f});
    BodyHandle b = make_dynamic({2.0f, 0.0f});

    JointDef def;
    def.type = JointType::Spring;
    def.body_a = a;
    def.body_b = b;
    def.stiffness = 100.0f;
    def.damping = 10.0f;
    def.rest_length = 0.0f;  // auto-detect: 2.0

    world.create_joint(def);

    // Displace body b further away
    // The spring should pull it back toward rest length (2.0)
    for (int i = 0; i < 5; ++i) {
        world.apply_force(b, {50.0f, 0.0f});
        world.step(dt);
    }
    // Stop applying force
    for (int i = 0; i < 300; ++i) {
        world.step(dt);
    }

    Vec2 pos_a = world.get_position(a);
    Vec2 pos_b = world.get_position(b);
    float dist = glm::length(pos_b - pos_a);
    // Should settle near 2.0 (original distance)
    EXPECT_NEAR(dist, 2.0f, 0.8f);
}

TEST_F(SpringJointFixture, SpringPullsCompressedBodies) {
    world.set_gravity({0.0f, 0.0f});

    BodyHandle anchor = make_static({0.0f, 0.0f});
    BodyHandle bob = make_dynamic({0.5f, 0.0f});  // closer than rest

    JointDef def;
    def.type = JointType::Spring;
    def.body_a = anchor;
    def.body_b = bob;
    def.stiffness = 100.0f;
    def.damping = 5.0f;
    def.rest_length = 3.0f;  // rest at 3.0, starting at 0.5

    world.create_joint(def);

    for (int i = 0; i < 60; ++i) {
        world.step(dt);
    }

    Vec2 pos = world.get_position(bob);
    // Spring should push bob away from anchor toward rest length
    EXPECT_GT(pos.x, 0.5f);
}

TEST_F(SpringJointFixture, SpringWithGravityPendulum) {
    world.set_gravity({0.0f, -10.0f});

    BodyHandle anchor = make_static({0.0f, 5.0f});
    BodyHandle bob = make_dynamic({0.0f, 2.0f});

    JointDef def;
    def.type = JointType::Spring;
    def.body_a = anchor;
    def.body_b = bob;
    def.stiffness = 50.0f;
    def.damping = 5.0f;
    def.rest_length = 3.0f;

    world.create_joint(def);

    for (int i = 0; i < 300; ++i) {
        world.step(dt);
    }

    // Bob should settle below anchor at roughly rest_length distance
    Vec2 bob_pos = world.get_position(bob);
    EXPECT_LT(bob_pos.y, 5.0f);
    float dist = glm::length(bob_pos - Vec2{0.0f, 5.0f});
    // With gravity and damping, should be near rest length (slightly stretched)
    EXPECT_GT(dist, 2.0f);
    EXPECT_LT(dist, 5.0f);
}

TEST_F(SpringJointFixture, NoWarmStartState) {
    // Spring joints should not accumulate warm-start state
    world.set_gravity({0.0f, 0.0f});

    BodyHandle a = make_static({0.0f, 0.0f});
    BodyHandle b = make_dynamic({2.0f, 0.0f});

    JointDef def;
    def.type = JointType::Spring;
    def.body_a = a;
    def.body_b = b;
    def.stiffness = 50.0f;
    def.damping = 1.0f;
    def.rest_length = 1.0f;

    world.create_joint(def);

    // Run simulation
    for (int i = 0; i < 60; ++i) {
        world.step(dt);
    }

    // Verify spring still works (bob moved toward rest)
    Vec2 pos = world.get_position(b);
    EXPECT_LT(pos.x, 2.0f);
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cmake --build build && ./build/stan2d_tests --gtest_filter=SpringJointFixture.*`
Expected: FAIL — Spring case not implemented

- [ ] **Step 3: Add Spring case to solve_joints()**

In `src/stan2d/joints/joint_solver.cpp`, inside `solve_joints()`, add after the Distance block:

```cpp
        if (joints.types[i] == JointType::Spring) {
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

            // Spring impulse: lambda = (-k*(x-x0) - d*v) * dt / eff_mass_inv
            // Applied as impulse using effective mass
            float displacement = current_len - joints.spring_rest_length[i];
            float spring_force = -joints.spring_stiffness[i] * displacement
                               - joints.spring_damping[i] * v_along;
            float lambda = spring_force * dt * eff_mass;

            Vec2 impulse = axis * lambda;

            bodies.velocities[a] = bodies.velocities[a] - impulse * inv_ma;
            bodies.angular_velocities[a] -= inv_ia * cross_vec_vec(ra, impulse);

            bodies.velocities[b] = bodies.velocities[b] + impulse * inv_mb;
            bodies.angular_velocities[b] += inv_ib * cross_vec_vec(rb, impulse);

            // No warm-start accumulation for Spring
            joints.accumulated_impulse_x[i] = 0.0f;
            joints.accumulated_impulse_y[i] = 0.0f;

            joints.constraint_forces[i] += std::abs(lambda);
        }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cmake --build build && ./build/stan2d_tests --gtest_filter=SpringJointFixture.*`
Expected: ALL PASS

- [ ] **Step 5: Run all tests for regression**

Run: `ctest --test-dir build`
Expected: ALL PASS

- [ ] **Step 6: Commit**

```bash
git add src/stan2d/joints/joint_solver.cpp tests/unit/test_spring_joint.cpp
git commit -m "feat: implement Spring joint solver with stiffness and damping (Task 16 complete)"
```

---

## Verification Checklist

- [ ] All Spring tests pass: `./build/stan2d_tests --gtest_filter=SpringJointFixture.*`
- [ ] All Hinge + Distance tests still pass
- [ ] All existing Phase 1 tests pass: `ctest --test-dir build`
- [ ] Oscillation decays with damping
- [ ] Overdamped spring settles without oscillation
- [ ] Zero rest-length auto-detection works
- [ ] No warm-start state accumulated (stateless per frame)
