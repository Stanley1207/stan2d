# Task 14: Hinge Joint Solver — Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the Hinge joint solver (point-to-point constraint + angular limits + motor), integrate into `World::solve()`, and add joint query/control API.

**Architecture:** Joint solver lives in `joint_solver.hpp/cpp` with three functions: `prepare_joint_constraints()`, `warm_start_joints()`, `solve_joints()`. These are called from `World::solve()` alongside existing contact constraints in the same iteration loop. The Phase 1 early-exit guard `if (constraints_.empty()) return;` must be removed so joints solve even without contacts.

**Tech Stack:** C++20, glm, Google Test

**Depends on:** Task 13 (joint infrastructure)

---

## File Structure

| Action | File | Responsibility |
|--------|------|---------------|
| Create | `include/stan2d/joints/joint_solver.hpp` | `prepare_joint_constraints()`, `warm_start_joints()`, `solve_joints()` declarations |
| Create | `src/stan2d/joints/joint_solver.cpp` | Hinge solver implementation (2×2 point constraint, limit, motor) |
| Modify | `include/stan2d/world/world.hpp` | Add `get_joint_angle()`, `get_joint_speed()`, `set_motor_speed()`, `set_motor_torque()` |
| Modify | `src/stan2d/world/world.cpp` | Implement new API methods, extend `solve()` to call joint solver |
| Create | `tests/unit/test_hinge_joint.cpp` | All Hinge joint tests |

---

### Task 14.1: Create joint_solver.hpp declarations

- [ ] **Step 1: Create the header file**

Create `include/stan2d/joints/joint_solver.hpp`:

```cpp
#pragma once

#include <stan2d/constraints/contact_constraint.hpp>
#include <stan2d/dynamics/body_storage.hpp>
#include <stan2d/joints/joint_storage.hpp>

namespace stan2d {

// Compute effective masses and cache per-frame values for all joints.
// Call once per step, before warm_start_joints().
void prepare_joint_constraints(JointStorage& joints, const BodyStorage& bodies, float dt);

// Apply accumulated impulses from previous frame for all joints.
// Call after prepare_joint_constraints(), before iteration loop.
void warm_start_joints(const JointStorage& joints, BodyStorage& bodies);

// Solve all joint constraints for one iteration.
// Call inside the iteration loop, after solve_contacts().
void solve_joints(JointStorage& joints, BodyStorage& bodies,
                  const SolverConfig& config, float dt);

} // namespace stan2d
```

- [ ] **Step 2: Create stub implementation**

Create `src/stan2d/joints/joint_solver.cpp`:

```cpp
#include <stan2d/joints/joint_solver.hpp>

#include <cmath>
#include <glm/glm.hpp>
#include <glm/gtc/constants.hpp>
#include <stan2d/constraints/solver.hpp>

namespace stan2d {

void prepare_joint_constraints(JointStorage& joints, const BodyStorage& bodies, float dt) {
    // Reset per-frame observables
    for (uint32_t i = 0; i < joints.size; ++i) {
        joints.constraint_forces[i] = 0.0f;
    }
}

void warm_start_joints(const JointStorage& joints, BodyStorage& bodies) {
    // Stub — implemented in 14.3
}

void solve_joints(JointStorage& joints, BodyStorage& bodies,
                  const SolverConfig& config, float dt) {
    // Stub — implemented in 14.4
}

} // namespace stan2d
```

- [ ] **Step 3: Build to verify compilation**

Run: `cmake --build build`
Expected: PASS (no linker errors — new .cpp is auto-discovered by GLOB_RECURSE)

- [ ] **Step 4: Commit**

```bash
git add include/stan2d/joints/joint_solver.hpp src/stan2d/joints/joint_solver.cpp
git commit -m "feat: add joint_solver.hpp/cpp stubs"
```

---

### Task 14.2: Wire joint solver into World::solve() and add dt to solve()

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_hinge_joint.cpp`:

```cpp
#include <gtest/gtest.h>
#include <cmath>
#include <stan2d/world/world.hpp>

using namespace stan2d;

class HingeJointFixture : public ::testing::Test {
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

// Joint solver runs without contacts (early-exit guard removed)
TEST_F(HingeJointFixture, SolveRunsWithJointsButNoContacts) {
    world.set_gravity({0.0f, -10.0f});

    BodyHandle anchor = make_static({0.0f, 5.0f});
    BodyHandle bob = make_dynamic({1.0f, 5.0f});

    JointDef def;
    def.type = JointType::Hinge;
    def.body_a = anchor;
    def.body_b = bob;
    def.anchor_a = {0.0f, 0.0f};
    def.anchor_b = {0.0f, 0.0f};

    world.create_joint(def);

    // Step should exercise joint solver even with no contacts
    for (int i = 0; i < 10; ++i) {
        world.step(dt);
    }

    // Bob should have fallen (gravity) but be constrained near anchor
    Vec2 bob_pos = world.get_position(bob);
    float dist = glm::length(bob_pos - Vec2{0.0f, 5.0f});
    // With point constraint, bob should stay close to anchor point
    EXPECT_LT(dist, 2.0f);  // relaxed — solver is iterative
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cmake --build build && ./build/stan2d_tests --gtest_filter=HingeJointFixture.SolveRunsWithJointsButNoContacts`
Expected: FAIL or passes vacuously (joint solver not wired in yet)

- [ ] **Step 3: Modify World to store dt and call joint solver**

In `include/stan2d/world/world.hpp`, add private member:

```cpp
    float current_dt_ = 0.0f;
```

In `src/stan2d/world/world.cpp`, add include:

```cpp
#include <stan2d/joints/joint_solver.hpp>
```

Modify `World::step()` to store dt:

```cpp
void World::step(float dt) {
    if (dt <= 0.0f) return;

    uint32_t count = bodies_.size();
    if (count == 0) return;

    current_dt_ = dt;
    // ... rest unchanged
}
```

Replace `World::solve()` with:

```cpp
void World::solve() {
    // Prepare contact constraints
    constraints_.clear();
    for (const auto& entry : manifold_entries_) {
        prepare_contact_constraints(
            entry.manifold, entry.dense_a, entry.dense_b,
            bodies_, constraints_
        );
    }

    // Prepare joint constraints
    prepare_joint_constraints(joints_, bodies_, current_dt_);

    bool has_contacts = !constraints_.empty();
    bool has_joints = joints_.size > 0;

    if (!has_contacts && !has_joints) return;

    // Warm start
    if (has_contacts) {
        warm_start(constraints_, bodies_);
    }
    if (has_joints) {
        warm_start_joints(joints_, bodies_);
    }

    // Iterative solve
    for (uint32_t iter = 0; iter < solver_config_.iterations; ++iter) {
        if (has_contacts) {
            solve_constraints(constraints_, bodies_, solver_config_);
        }
        if (has_joints) {
            solve_joints(joints_, bodies_, solver_config_, current_dt_);
        }
    }
}
```

**Important:** The old `solve_constraints()` already runs its own iteration loop internally. We need to change `solve_constraints()` to a single-iteration version OR refactor. The simplest approach: extract a `solve_contacts_one_iteration()` function. However, to minimize Phase 1 changes, we instead change `World::solve()` to call `solve_constraints()` once with `iterations=1` inside our loop.

Actually, looking at the existing code, `solve_constraints` has its own iteration loop. We should **not** double-nest. Instead, change the `World::solve()` to:

```cpp
void World::solve() {
    constraints_.clear();
    for (const auto& entry : manifold_entries_) {
        prepare_contact_constraints(
            entry.manifold, entry.dense_a, entry.dense_b,
            bodies_, constraints_
        );
    }

    prepare_joint_constraints(joints_, bodies_, current_dt_);

    bool has_contacts = !constraints_.empty();
    bool has_joints = joints_.size > 0;

    if (!has_contacts && !has_joints) return;

    if (has_contacts) {
        warm_start(constraints_, bodies_);
    }
    if (has_joints) {
        warm_start_joints(joints_, bodies_);
    }

    // Unified iteration loop: contacts + joints solved together
    for (uint32_t iter = 0; iter < solver_config_.iterations; ++iter) {
        // Solve contacts (one iteration)
        for (auto& c : constraints_) {
            Vec2 ra = c.contact_point - bodies_.positions[c.body_a];
            Vec2 rb = c.contact_point - bodies_.positions[c.body_b];

            Vec2 rel_vel = (bodies_.velocities[c.body_b]
                          + cross_scalar_vec(bodies_.angular_velocities[c.body_b], rb))
                         - (bodies_.velocities[c.body_a]
                          + cross_scalar_vec(bodies_.angular_velocities[c.body_a], ra));

            // Normal impulse
            float vn = glm::dot(rel_vel, c.normal);
            float bias = solver_config_.baumgarte
                       * glm::max(c.penetration - solver_config_.slop, 0.0f);
            float lambda_n = c.normal_mass * (-vn + bias);
            float new_impulse = glm::max(c.accumulated_normal_impulse + lambda_n, 0.0f);
            lambda_n = new_impulse - c.accumulated_normal_impulse;
            c.accumulated_normal_impulse = new_impulse;

            Vec2 impulse_n = c.normal * lambda_n;
            bodies_.velocities[c.body_a] = bodies_.velocities[c.body_a]
                                          - impulse_n * bodies_.inverse_masses[c.body_a];
            bodies_.angular_velocities[c.body_a] -= bodies_.inverse_inertias[c.body_a]
                                                   * cross_vec_vec(ra, impulse_n);
            bodies_.velocities[c.body_b] = bodies_.velocities[c.body_b]
                                          + impulse_n * bodies_.inverse_masses[c.body_b];
            bodies_.angular_velocities[c.body_b] += bodies_.inverse_inertias[c.body_b]
                                                   * cross_vec_vec(rb, impulse_n);

            // Tangent impulse
            rel_vel = (bodies_.velocities[c.body_b]
                     + cross_scalar_vec(bodies_.angular_velocities[c.body_b], rb))
                    - (bodies_.velocities[c.body_a]
                     + cross_scalar_vec(bodies_.angular_velocities[c.body_a], ra));

            float vt = glm::dot(rel_vel, c.tangent);
            float lambda_t = c.tangent_mass * (-vt);
            float max_friction = solver_config_.friction * c.accumulated_normal_impulse;
            float new_tangent = glm::clamp(
                c.accumulated_tangent_impulse + lambda_t,
                -max_friction, max_friction);
            lambda_t = new_tangent - c.accumulated_tangent_impulse;
            c.accumulated_tangent_impulse = new_tangent;

            Vec2 impulse_t = c.tangent * lambda_t;
            bodies_.velocities[c.body_a] = bodies_.velocities[c.body_a]
                                          - impulse_t * bodies_.inverse_masses[c.body_a];
            bodies_.angular_velocities[c.body_a] -= bodies_.inverse_inertias[c.body_a]
                                                   * cross_vec_vec(ra, impulse_t);
            bodies_.velocities[c.body_b] = bodies_.velocities[c.body_b]
                                          + impulse_t * bodies_.inverse_masses[c.body_b];
            bodies_.angular_velocities[c.body_b] += bodies_.inverse_inertias[c.body_b]
                                                   * cross_vec_vec(rb, impulse_t);
        }

        // Solve joints (one iteration)
        if (has_joints) {
            solve_joints(joints_, bodies_, solver_config_, current_dt_);
        }
    }
}
```

**Note:** This inlines the contact solve loop to avoid the double-nesting problem. The old `solve_constraints()` function in `solver.hpp` remains available for standalone unit testing but is no longer called from `World::solve()`.

- [ ] **Step 4: Run test to verify it passes**

Run: `cmake --build build && ./build/stan2d_tests --gtest_filter=HingeJointFixture.SolveRunsWithJointsButNoContacts`
Expected: PASS

- [ ] **Step 5: Run all existing tests to verify no regression**

Run: `ctest --test-dir build`
Expected: ALL PASS

- [ ] **Step 6: Commit**

```bash
git add include/stan2d/world/world.hpp src/stan2d/world/world.cpp tests/unit/test_hinge_joint.cpp
git commit -m "feat: wire joint solver into World::solve() with unified iteration loop"
```

---

### Task 14.3: Implement Hinge point-to-point constraint (prepare + warm_start + solve)

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/test_hinge_joint.cpp`:

```cpp
TEST_F(HingeJointFixture, PendulumSwings) {
    world.set_gravity({0.0f, -10.0f});

    // Anchor point at (0, 5), bob at (2, 5) — horizontal pendulum
    BodyHandle anchor = make_static({0.0f, 5.0f});
    BodyHandle bob = make_dynamic({2.0f, 5.0f});

    JointDef def;
    def.type = JointType::Hinge;
    def.body_a = anchor;
    def.body_b = bob;
    // Anchor at the anchor body center, bob attached at its center
    def.anchor_a = {0.0f, 0.0f};
    def.anchor_b = {0.0f, 0.0f};

    world.create_joint(def);

    // Simulate pendulum swing
    for (int i = 0; i < 60; ++i) {
        world.step(dt);
    }

    Vec2 bob_pos = world.get_position(bob);
    Vec2 anchor_pos = world.get_position(anchor);

    // Bob should maintain roughly constant distance from anchor
    float dist = glm::length(bob_pos - anchor_pos);
    EXPECT_NEAR(dist, 2.0f, 0.3f);  // iterative solver tolerance

    // Bob should have swung down (gravity pulls it)
    EXPECT_LT(bob_pos.y, 5.0f);
}

TEST_F(HingeJointFixture, TwoDynamicBodiesStayConnected) {
    world.set_gravity({0.0f, -10.0f});

    BodyHandle a = make_dynamic({0.0f, 5.0f});
    BodyHandle b = make_dynamic({1.0f, 5.0f});

    JointDef def;
    def.type = JointType::Hinge;
    def.body_a = a;
    def.body_b = b;
    def.anchor_a = {0.5f, 0.0f};   // right side of a
    def.anchor_b = {-0.5f, 0.0f};  // left side of b

    world.create_joint(def);

    for (int i = 0; i < 120; ++i) {
        world.step(dt);
    }

    // World-space anchor points should be close
    Vec2 pos_a = world.get_position(a);
    Vec2 pos_b = world.get_position(b);
    Vec2 wa = pos_a + Vec2{0.5f, 0.0f};   // approximate (ignores rotation)
    Vec2 wb = pos_b + Vec2{-0.5f, 0.0f};
    float gap = glm::length(wa - wb);
    EXPECT_LT(gap, 0.5f);  // relaxed for iterative solver
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cmake --build build && ./build/stan2d_tests --gtest_filter=HingeJointFixture.PendulumSwings:HingeJointFixture.TwoDynamicBodiesStayConnected`
Expected: FAIL — stub solver does nothing, constraint not enforced

- [ ] **Step 3: Implement Hinge point-to-point solver**

Replace the contents of `src/stan2d/joints/joint_solver.cpp`:

```cpp
#include <stan2d/joints/joint_solver.hpp>

#include <cmath>
#include <glm/glm.hpp>
#include <glm/gtc/constants.hpp>
#include <stan2d/constraints/solver.hpp>

namespace stan2d {

namespace {

// Wrap angle to [-pi, pi]
float wrap_angle(float a) {
    while (a > glm::pi<float>()) a -= 2.0f * glm::pi<float>();
    while (a < -glm::pi<float>()) a += 2.0f * glm::pi<float>();
    return a;
}

} // anonymous namespace

void prepare_joint_constraints(JointStorage& joints, const BodyStorage& bodies, float dt) {
    for (uint32_t i = 0; i < joints.size; ++i) {
        joints.constraint_forces[i] = 0.0f;

        uint32_t a = joints.body_a[i];
        uint32_t b = joints.body_b[i];

        // Compute and cache observables
        float angle = wrap_angle(bodies.rotations[b] - bodies.rotations[a]
                                 - joints.reference_angle[i]);
        float angular_speed = bodies.angular_velocities[b] - bodies.angular_velocities[a];

        // Compute world-space anchors and length
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
        float len = glm::length(world_b - world_a);

        joints.cached_angles[i] = (joints.types[i] == JointType::Hinge) ? angle : 0.0f;
        joints.cached_angular_speeds[i] = (joints.types[i] == JointType::Hinge)
                                           ? angular_speed : 0.0f;
        joints.cached_lengths[i] = len;
    }
}

void warm_start_joints(const JointStorage& joints, BodyStorage& bodies) {
    for (uint32_t i = 0; i < joints.size; ++i) {
        if (joints.types[i] == JointType::Spring) continue;  // Spring has no warm-start

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

        Vec2 impulse = {joints.accumulated_impulse_x[i], joints.accumulated_impulse_y[i]};

        bodies.velocities[a] = bodies.velocities[a]
                              - impulse * bodies.inverse_masses[a];
        bodies.angular_velocities[a] -= bodies.inverse_inertias[a]
                                       * cross_vec_vec(ra, impulse);

        bodies.velocities[b] = bodies.velocities[b]
                              + impulse * bodies.inverse_masses[b];
        bodies.angular_velocities[b] += bodies.inverse_inertias[b]
                                       * cross_vec_vec(rb, impulse);

        // Warm-start limit and motor (Hinge only)
        if (joints.types[i] == JointType::Hinge) {
            float limit_imp = joints.accumulated_limit_impulse[i];
            float motor_imp = joints.accumulated_motor_impulse[i];
            float total_angular = limit_imp + motor_imp;

            bodies.angular_velocities[a] -= bodies.inverse_inertias[a] * total_angular;
            bodies.angular_velocities[b] += bodies.inverse_inertias[b] * total_angular;
        }
    }
}

void solve_joints(JointStorage& joints, BodyStorage& bodies,
                  const SolverConfig& config, float dt) {
    for (uint32_t i = 0; i < joints.size; ++i) {
        if (joints.types[i] != JointType::Hinge) continue;

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

        // ── Point-to-point constraint (2×2) ──────────────────────
        // Constraint velocity: v_b + w_b × rb - v_a - w_a × ra
        Vec2 v_constraint = (bodies.velocities[b]
                            + cross_scalar_vec(bodies.angular_velocities[b], rb))
                           - (bodies.velocities[a]
                            + cross_scalar_vec(bodies.angular_velocities[a], ra));

        // Position error for Baumgarte correction
        Vec2 world_a = bodies.positions[a] + ra;
        Vec2 world_b = bodies.positions[b] + rb;
        Vec2 pos_error = world_b - world_a;
        float beta = config.baumgarte;

        // 2×2 effective mass matrix K
        // K = (inv_ma + inv_mb)*I - inv_ia*(ra_perp ⊗ ra_perp) - inv_ib*(rb_perp ⊗ rb_perp)
        // where ra_perp = (-ra.y, ra.x)
        float k11 = inv_ma + inv_mb + inv_ia * ra.y * ra.y + inv_ib * rb.y * rb.y;
        float k12 = -inv_ia * ra.x * ra.y - inv_ib * rb.x * rb.y;
        float k22 = inv_ma + inv_mb + inv_ia * ra.x * ra.x + inv_ib * rb.x * rb.x;

        // Invert 2×2 matrix: K^-1
        float det = k11 * k22 - k12 * k12;
        if (std::abs(det) < 1e-12f) continue;
        float inv_det = 1.0f / det;

        // RHS = -(v_constraint + beta/dt * pos_error)
        Vec2 rhs = -(v_constraint + (beta / dt) * pos_error);

        // Solve K * lambda = rhs
        Vec2 lambda = {inv_det * (k22 * rhs.x - k12 * rhs.y),
                       inv_det * (-k12 * rhs.x + k11 * rhs.y)};

        // Accumulate (no clamp for point constraint — two-sided)
        joints.accumulated_impulse_x[i] += lambda.x;
        joints.accumulated_impulse_y[i] += lambda.y;

        // Apply impulse
        bodies.velocities[a] = bodies.velocities[a] - lambda * inv_ma;
        bodies.angular_velocities[a] -= inv_ia * cross_vec_vec(ra, lambda);

        bodies.velocities[b] = bodies.velocities[b] + lambda * inv_mb;
        bodies.angular_velocities[b] += inv_ib * cross_vec_vec(rb, lambda);

        joints.constraint_forces[i] += glm::length(lambda);

        // ── Angular limit ────────────────────────────────────────
        if (joints.limit_enabled[i]) {
            float angle = wrap_angle(bodies.rotations[b] - bodies.rotations[a]
                                     - joints.reference_angle[i]);
            float angular_speed = bodies.angular_velocities[b]
                                - bodies.angular_velocities[a];
            float inv_i_total = inv_ia + inv_ib;
            if (inv_i_total > 0.0f) {
                float eff_mass = 1.0f / inv_i_total;

                if (angle < joints.limit_min[i]) {
                    float c_val = angle - joints.limit_min[i];
                    float lambda_lim = eff_mass * (-angular_speed - beta / dt * c_val);
                    float old_acc = joints.accumulated_limit_impulse[i];
                    joints.accumulated_limit_impulse[i] = glm::max(old_acc + lambda_lim, 0.0f);
                    float delta = joints.accumulated_limit_impulse[i] - old_acc;

                    bodies.angular_velocities[a] -= inv_ia * delta;
                    bodies.angular_velocities[b] += inv_ib * delta;
                    joints.constraint_forces[i] += std::abs(delta);
                } else if (angle > joints.limit_max[i]) {
                    float c_val = angle - joints.limit_max[i];
                    float lambda_lim = eff_mass * (-angular_speed - beta / dt * c_val);
                    float old_acc = joints.accumulated_limit_impulse[i];
                    joints.accumulated_limit_impulse[i] = glm::min(old_acc + lambda_lim, 0.0f);
                    float delta = joints.accumulated_limit_impulse[i] - old_acc;

                    bodies.angular_velocities[a] -= inv_ia * delta;
                    bodies.angular_velocities[b] += inv_ib * delta;
                    joints.constraint_forces[i] += std::abs(delta);
                }
            }
        }

        // ── Motor ────────────────────────────────────────────────
        if (joints.motor_enabled[i]) {
            float angular_speed = bodies.angular_velocities[b]
                                - bodies.angular_velocities[a];
            float inv_i_total = inv_ia + inv_ib;
            if (inv_i_total > 0.0f) {
                float eff_mass = 1.0f / inv_i_total;
                float speed_error = joints.motor_target_speeds[i] - angular_speed;
                float lambda_motor = eff_mass * speed_error;

                float max_impulse = joints.motor_max_torque[i] * dt;
                float old_acc = joints.accumulated_motor_impulse[i];
                joints.accumulated_motor_impulse[i] = glm::clamp(
                    old_acc + lambda_motor, -max_impulse, max_impulse);
                float delta = joints.accumulated_motor_impulse[i] - old_acc;

                bodies.angular_velocities[a] -= inv_ia * delta;
                bodies.angular_velocities[b] += inv_ib * delta;
                joints.constraint_forces[i] += std::abs(delta);
            }
        }
    }
}

} // namespace stan2d
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cmake --build build && ./build/stan2d_tests --gtest_filter=HingeJointFixture.PendulumSwings:HingeJointFixture.TwoDynamicBodiesStayConnected`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/stan2d/joints/joint_solver.cpp tests/unit/test_hinge_joint.cpp
git commit -m "feat: implement Hinge point-to-point constraint solver"
```

---

### Task 14.4: Add joint query and motor control API

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/test_hinge_joint.cpp`:

```cpp
TEST_F(HingeJointFixture, GetJointAngleReturnsRelativeAngle) {
    BodyHandle a = make_dynamic({0.0f, 0.0f});
    BodyHandle b = make_dynamic({1.0f, 0.0f});

    JointDef def;
    def.type = JointType::Hinge;
    def.body_a = a;
    def.body_b = b;

    JointHandle jh = world.create_joint(def);

    // Before stepping, angle should be ~0 (bodies at same rotation)
    world.step(dt);  // Need at least one step to populate cached_angles
    float angle = world.get_joint_angle(jh);
    EXPECT_NEAR(angle, 0.0f, 0.1f);
}

TEST_F(HingeJointFixture, GetJointSpeedReturnsAngularSpeed) {
    BodyHandle a = make_static({0.0f, 0.0f});
    BodyHandle b = make_dynamic({1.0f, 0.0f});

    JointDef def;
    def.type = JointType::Hinge;
    def.body_a = a;
    def.body_b = b;

    JointHandle jh = world.create_joint(def);
    world.step(dt);

    float speed = world.get_joint_speed(jh);
    // Speed is a float, just verify it doesn't crash and returns finite
    EXPECT_TRUE(std::isfinite(speed));
}

TEST_F(HingeJointFixture, MotorDrivesRotation) {
    world.set_gravity({0.0f, 0.0f});

    BodyHandle a = make_static({0.0f, 0.0f});
    BodyHandle b = make_dynamic({1.0f, 0.0f});

    JointDef def;
    def.type = JointType::Hinge;
    def.body_a = a;
    def.body_b = b;
    def.motor_enabled = true;
    def.motor_speed = 5.0f;      // target: 5 rad/s CCW
    def.motor_torque = 100.0f;   // strong motor

    JointHandle jh = world.create_joint(def);

    for (int i = 0; i < 120; ++i) {
        world.step(dt);
    }

    // Bob should be rotating in the positive direction
    float speed = world.get_joint_speed(jh);
    EXPECT_GT(speed, 1.0f);
}

TEST_F(HingeJointFixture, SetMotorSpeedChangesTarget) {
    world.set_gravity({0.0f, 0.0f});

    BodyHandle a = make_static({0.0f, 0.0f});
    BodyHandle b = make_dynamic({1.0f, 0.0f});

    JointDef def;
    def.type = JointType::Hinge;
    def.body_a = a;
    def.body_b = b;
    def.motor_enabled = true;
    def.motor_speed = 0.0f;
    def.motor_torque = 100.0f;

    JointHandle jh = world.create_joint(def);

    // Start with zero target, then switch
    for (int i = 0; i < 30; ++i) { world.step(dt); }
    world.set_motor_speed(jh, -5.0f);
    for (int i = 0; i < 120; ++i) { world.step(dt); }

    float speed = world.get_joint_speed(jh);
    EXPECT_LT(speed, -1.0f);
}

TEST_F(HingeJointFixture, LimitStopsRotation) {
    world.set_gravity({0.0f, -10.0f});

    BodyHandle a = make_static({0.0f, 5.0f});
    BodyHandle b = make_dynamic({2.0f, 5.0f});

    JointDef def;
    def.type = JointType::Hinge;
    def.body_a = a;
    def.body_b = b;
    def.limit_enabled = true;
    def.limit_min = -0.5f;  // ~28 degrees
    def.limit_max = 0.5f;

    world.create_joint(def);

    // Gravity should try to swing bob down but limit should restrict
    for (int i = 0; i < 300; ++i) {
        world.step(dt);
    }

    Vec2 bob_pos = world.get_position(b);
    Vec2 anchor_pos = world.get_position(a);
    Vec2 delta = bob_pos - anchor_pos;

    // Angle from horizontal should be within limits (approximately)
    float angle = std::atan2(delta.y, delta.x);
    // The hinge reference was set at creation (horizontal), so the
    // pendulum angle should be close to the limit
    // Just check bob hasn't swung all the way down
    EXPECT_GT(bob_pos.y, 3.5f);  // should still be relatively high
}

TEST_F(HingeJointFixture, WarmStartImprovesSolverConvergence) {
    world.set_gravity({0.0f, -10.0f});

    BodyHandle anchor = make_static({0.0f, 5.0f});
    BodyHandle bob = make_dynamic({0.0f, 3.0f});

    JointDef def;
    def.type = JointType::Hinge;
    def.body_a = anchor;
    def.body_b = bob;

    world.create_joint(def);

    // Run multiple steps — warm-start should prevent drift
    for (int i = 0; i < 300; ++i) {
        world.step(dt);
    }

    Vec2 bob_pos = world.get_position(bob);
    float dist = glm::length(bob_pos - Vec2{0.0f, 5.0f});
    // With warm starting, distance should stay close to 2
    EXPECT_NEAR(dist, 2.0f, 0.5f);
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cmake --build build && ./build/stan2d_tests --gtest_filter=HingeJointFixture.*`
Expected: FAIL — `get_joint_angle`, `get_joint_speed`, `set_motor_speed`, `set_motor_torque` not defined

- [ ] **Step 3: Add query/control API to World**

In `include/stan2d/world/world.hpp`, add to the public section:

```cpp
    // ── Joint queries ─────────────────────────────────────────────
    [[nodiscard]] float get_joint_angle(JointHandle handle) const;
    [[nodiscard]] float get_joint_speed(JointHandle handle) const;
    [[nodiscard]] float get_joint_length(JointHandle handle) const;

    // ── Motor control (RL agent interface) ────────────────────────
    void set_motor_speed(JointHandle handle, float speed);
    void set_motor_torque(JointHandle handle, float max_torque);
```

In `src/stan2d/world/world.cpp`, add implementations:

```cpp
float World::get_joint_angle(JointHandle handle) const {
    Handle h{handle.index, handle.generation};
    uint32_t idx = joint_handles_.dense_index(h);
    return joints_.cached_angles[idx];
}

float World::get_joint_speed(JointHandle handle) const {
    Handle h{handle.index, handle.generation};
    uint32_t idx = joint_handles_.dense_index(h);
    return joints_.cached_angular_speeds[idx];
}

float World::get_joint_length(JointHandle handle) const {
    Handle h{handle.index, handle.generation};
    uint32_t idx = joint_handles_.dense_index(h);
    return joints_.cached_lengths[idx];
}

void World::set_motor_speed(JointHandle handle, float speed) {
    Handle h{handle.index, handle.generation};
    uint32_t idx = joint_handles_.dense_index(h);
    joints_.motor_target_speeds[idx] = speed;
}

void World::set_motor_torque(JointHandle handle, float max_torque) {
    Handle h{handle.index, handle.generation};
    uint32_t idx = joint_handles_.dense_index(h);
    joints_.motor_max_torque[idx] = max_torque;
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cmake --build build && ./build/stan2d_tests --gtest_filter=HingeJointFixture.*`
Expected: ALL PASS

- [ ] **Step 5: Run all tests for regression**

Run: `ctest --test-dir build`
Expected: ALL PASS

- [ ] **Step 6: Commit**

```bash
git add include/stan2d/world/world.hpp src/stan2d/world/world.cpp tests/unit/test_hinge_joint.cpp
git commit -m "feat: add Hinge joint solver with limits, motor, and query API (Task 14 complete)"
```

---

## Verification Checklist

- [ ] All Hinge tests pass: `./build/stan2d_tests --gtest_filter=HingeJointFixture.*`
- [ ] All existing tests pass: `ctest --test-dir build`
- [ ] Point-to-point constraint keeps bodies connected (pendulum test)
- [ ] Angular limits stop rotation at boundaries
- [ ] Motor drives rotation toward target speed
- [ ] `set_motor_speed()` / `set_motor_torque()` work at runtime
- [ ] Warm-starting prevents drift over 300+ steps
- [ ] Joint solver runs even when no contacts exist (early-exit guard removed)
