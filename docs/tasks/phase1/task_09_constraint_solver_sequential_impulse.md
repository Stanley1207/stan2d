> **Note for Claude**: 牢记 `docs/tasks/00_master_plan.md` 中的四大铁律。

## Task 9: Constraint Solver (Sequential Impulse)

**Goal:** Impulse-based constraint solver. Resolves collision contacts by applying normal impulses (non-penetration) and tangent impulses (Coulomb friction). Supports warm starting via cached impulses.

**Files:**
- Create: `include/stan2d/constraints/contact_constraint.hpp`
- Create: `include/stan2d/constraints/solver.hpp`
- Create: `tests/unit/test_solver.cpp`

**Depends on:** Task 8

### Step 1: Write failing tests (RED)

**File:** `tests/unit/test_solver.cpp`

```cpp
#include <gtest/gtest.h>
#include <cmath>
#include <stan2d/constraints/solver.hpp>
#include <stan2d/dynamics/body_storage.hpp>

using namespace stan2d;

// Helper: create two-body storage for collision tests
struct SolverFixture : public ::testing::Test {
    BodyStorage bodies;
    uint32_t count = 0;

    // Returns dense index
    uint32_t add_dynamic(Vec2 pos, Vec2 vel, float mass) {
        float inv_mass = 1.0f / mass;
        float inertia = 0.5f * mass;  // simple
        float inv_inertia = 1.0f / inertia;
        bodies.push_back(pos, vel, 0.0f, 0.0f,
                         mass, inv_mass, inertia, inv_inertia,
                         BodyType::Dynamic, ShapeHandle{});
        return count++;
    }

    uint32_t add_static(Vec2 pos) {
        bodies.push_back(pos, {0.0f, 0.0f}, 0.0f, 0.0f,
                         0.0f, 0.0f, 0.0f, 0.0f,
                         BodyType::Static, ShapeHandle{});
        return count++;
    }
};

// ── Normal impulse: ball falling onto static floor ────────────────

TEST_F(SolverFixture, NormalImpulseStopsFallingBall) {
    uint32_t ball  = add_dynamic({0.0f, 1.0f}, {0.0f, -5.0f}, 1.0f);
    uint32_t floor = add_static({0.0f, 0.0f});

    ContactManifold manifold;
    manifold.normal = {0.0f, 1.0f};  // Floor pushes ball up
    manifold.point_count = 1;
    manifold.points[0].position = {0.0f, 0.5f};
    manifold.points[0].penetration = 0.1f;
    manifold.points[0].id = 0;

    std::vector<ContactConstraint> constraints;
    prepare_contact_constraints(manifold, ball, floor, bodies, constraints);

    SolverConfig config{.iterations = 10, .friction = 0.3f};
    solve_constraints(constraints, bodies, config);

    // Ball should no longer be moving downward
    EXPECT_GE(bodies.velocities[ball].y, -0.01f);
}

// ── Two dynamic bodies: head-on collision ─────────────────────────

TEST_F(SolverFixture, TwoDynamicBodiesHeadOn) {
    uint32_t a = add_dynamic({0.0f, 0.0f}, { 5.0f, 0.0f}, 1.0f);
    uint32_t b = add_dynamic({2.0f, 0.0f}, {-5.0f, 0.0f}, 1.0f);

    ContactManifold manifold;
    manifold.normal = {1.0f, 0.0f};  // A→B
    manifold.point_count = 1;
    manifold.points[0].position = {1.0f, 0.0f};
    manifold.points[0].penetration = 0.1f;
    manifold.points[0].id = 0;

    std::vector<ContactConstraint> constraints;
    prepare_contact_constraints(manifold, a, b, bodies, constraints);

    SolverConfig config{.iterations = 10, .friction = 0.0f};
    solve_constraints(constraints, bodies, config);

    // Equal mass, head-on: velocities should swap (perfectly elastic by default)
    // With sequential impulse (restitution=1 by default), they swap or at least separate
    EXPECT_LE(bodies.velocities[a].x, bodies.velocities[b].x);
}

// ── Friction: ball sliding on surface ─────────────────────────────

TEST_F(SolverFixture, FrictionSlowsLateralVelocity) {
    uint32_t ball  = add_dynamic({0.0f, 0.5f}, {10.0f, -1.0f}, 1.0f);
    uint32_t floor = add_static({0.0f, 0.0f});

    ContactManifold manifold;
    manifold.normal = {0.0f, 1.0f};
    manifold.point_count = 1;
    manifold.points[0].position = {0.0f, 0.0f};
    manifold.points[0].penetration = 0.05f;
    manifold.points[0].id = 0;

    std::vector<ContactConstraint> constraints;
    prepare_contact_constraints(manifold, ball, floor, bodies, constraints);

    SolverConfig config{.iterations = 10, .friction = 0.5f};
    solve_constraints(constraints, bodies, config);

    // Friction should have reduced horizontal velocity
    EXPECT_LT(std::abs(bodies.velocities[ball].x), 10.0f);
}

// ── Warm starting: cached impulse applied ─────────────────────────

TEST_F(SolverFixture, WarmStartingAppliesCachedImpulse) {
    uint32_t ball  = add_dynamic({0.0f, 0.5f}, {0.0f, -2.0f}, 1.0f);
    uint32_t floor = add_static({0.0f, 0.0f});

    ContactManifold manifold;
    manifold.normal = {0.0f, 1.0f};
    manifold.point_count = 1;
    manifold.points[0].position = {0.0f, 0.0f};
    manifold.points[0].penetration = 0.05f;
    manifold.points[0].id = 0;

    std::vector<ContactConstraint> constraints;
    prepare_contact_constraints(manifold, ball, floor, bodies, constraints);

    // Simulate warm start by pre-setting accumulated impulse
    constraints[0].accumulated_normal_impulse = 1.5f;

    SolverConfig config{.iterations = 10, .friction = 0.3f};
    warm_start(constraints, bodies);
    solve_constraints(constraints, bodies, config);

    // Ball should be pushed upward by warm start + solver
    EXPECT_GE(bodies.velocities[ball].y, -0.01f);
}

// ── Multiple contacts ─────────────────────────────────────────────

TEST_F(SolverFixture, MultipleContactPointsResolved) {
    uint32_t box   = add_dynamic({0.0f, 1.0f}, {0.0f, -3.0f}, 2.0f);
    uint32_t floor = add_static({0.0f, 0.0f});

    ContactManifold manifold;
    manifold.normal = {0.0f, 1.0f};
    manifold.point_count = 2;
    manifold.points[0].position = {-0.5f, 0.0f};
    manifold.points[0].penetration = 0.1f;
    manifold.points[0].id = 0;
    manifold.points[1].position = { 0.5f, 0.0f};
    manifold.points[1].penetration = 0.1f;
    manifold.points[1].id = 1;

    std::vector<ContactConstraint> constraints;
    prepare_contact_constraints(manifold, box, floor, bodies, constraints);
    EXPECT_EQ(constraints.size(), 2u);

    SolverConfig config{.iterations = 10, .friction = 0.3f};
    solve_constraints(constraints, bodies, config);

    EXPECT_GE(bodies.velocities[box].y, -0.01f);
}
```

### Step 2: Run tests to verify they fail

Run: `cmake --build build && ctest --test-dir build --output-on-failure`

Expected: FAIL — `fatal error: 'stan2d/constraints/solver.hpp' file not found`

### Step 3: Implement ContactConstraint and SolverConfig

**File:** `include/stan2d/constraints/contact_constraint.hpp`

```cpp
#pragma once

#include <cstdint>
#include <stan2d/core/math_types.hpp>

namespace stan2d {

struct ContactConstraint {
    uint32_t body_a;         // Dense index
    uint32_t body_b;         // Dense index

    Vec2  normal;            // A → B
    Vec2  tangent;           // Perpendicular to normal
    Vec2  contact_point;
    float penetration;

    // Effective mass along normal and tangent
    float normal_mass;
    float tangent_mass;

    // Accumulated impulses (for clamping and warm starting)
    float accumulated_normal_impulse  = 0.0f;
    float accumulated_tangent_impulse = 0.0f;

    // Bias for position correction (Baumgarte stabilization)
    float bias = 0.0f;

    // Contact point ID for warm starting across frames
    uint32_t id = 0;
};

struct SolverConfig {
    uint32_t iterations = 8;
    float    friction   = 0.3f;
    float    restitution = 0.0f;       // 0 = perfectly inelastic
    float    baumgarte  = 0.2f;        // Position correction factor
    float    slop       = 0.005f;      // Penetration allowance
};

} // namespace stan2d
```

### Step 4: Implement Solver (GREEN)

**File:** `include/stan2d/constraints/solver.hpp`

```cpp
#pragma once

#include <algorithm>
#include <cmath>
#include <vector>

#include <glm/glm.hpp>
#include <stan2d/collision/contact.hpp>
#include <stan2d/constraints/contact_constraint.hpp>
#include <stan2d/dynamics/body_storage.hpp>

namespace stan2d {

inline Vec2 cross_scalar_vec(float s, Vec2 v) {
    return {-s * v.y, s * v.x};
}

inline float cross_vec_vec(Vec2 a, Vec2 b) {
    return a.x * b.y - a.y * b.x;
}

inline void prepare_contact_constraints(
    const ContactManifold& manifold,
    uint32_t dense_a, uint32_t dense_b,
    const BodyStorage& bodies,
    std::vector<ContactConstraint>& out)
{
    float inv_mass_a    = bodies.inverse_masses[dense_a];
    float inv_mass_b    = bodies.inverse_masses[dense_b];
    float inv_inertia_a = bodies.inverse_inertias[dense_a];
    float inv_inertia_b = bodies.inverse_inertias[dense_b];

    Vec2 normal = manifold.normal;
    Vec2 tangent{-normal.y, normal.x};

    for (uint32_t i = 0; i < manifold.point_count; ++i) {
        ContactConstraint c;
        c.body_a = dense_a;
        c.body_b = dense_b;
        c.normal = normal;
        c.tangent = tangent;
        c.contact_point = manifold.points[i].position;
        c.penetration = manifold.points[i].penetration;
        c.id = manifold.points[i].id;

        Vec2 ra = c.contact_point - bodies.positions[dense_a];
        Vec2 rb = c.contact_point - bodies.positions[dense_b];

        // Effective mass along normal
        float rn_a = cross_vec_vec(ra, normal);
        float rn_b = cross_vec_vec(rb, normal);
        float k_normal = inv_mass_a + inv_mass_b
                       + inv_inertia_a * rn_a * rn_a
                       + inv_inertia_b * rn_b * rn_b;
        c.normal_mass = (k_normal > 0.0f) ? 1.0f / k_normal : 0.0f;

        // Effective mass along tangent
        float rt_a = cross_vec_vec(ra, tangent);
        float rt_b = cross_vec_vec(rb, tangent);
        float k_tangent = inv_mass_a + inv_mass_b
                        + inv_inertia_a * rt_a * rt_a
                        + inv_inertia_b * rt_b * rt_b;
        c.tangent_mass = (k_tangent > 0.0f) ? 1.0f / k_tangent : 0.0f;

        out.push_back(c);
    }
}

inline void warm_start(
    const std::vector<ContactConstraint>& constraints,
    BodyStorage& bodies)
{
    for (const auto& c : constraints) {
        Vec2 impulse = c.normal * c.accumulated_normal_impulse
                     + c.tangent * c.accumulated_tangent_impulse;

        Vec2 ra = c.contact_point - bodies.positions[c.body_a];
        Vec2 rb = c.contact_point - bodies.positions[c.body_b];

        bodies.velocities[c.body_a] = bodies.velocities[c.body_a]
                                    - impulse * bodies.inverse_masses[c.body_a];
        bodies.angular_velocities[c.body_a] -= bodies.inverse_inertias[c.body_a]
                                             * cross_vec_vec(ra, impulse);

        bodies.velocities[c.body_b] = bodies.velocities[c.body_b]
                                    + impulse * bodies.inverse_masses[c.body_b];
        bodies.angular_velocities[c.body_b] += bodies.inverse_inertias[c.body_b]
                                             * cross_vec_vec(rb, impulse);
    }
}

inline void solve_constraints(
    std::vector<ContactConstraint>& constraints,
    BodyStorage& bodies,
    const SolverConfig& config)
{
    for (uint32_t iter = 0; iter < config.iterations; ++iter) {
        for (auto& c : constraints) {
            Vec2 ra = c.contact_point - bodies.positions[c.body_a];
            Vec2 rb = c.contact_point - bodies.positions[c.body_b];

            // Relative velocity at contact point
            Vec2 rel_vel = (bodies.velocities[c.body_b]
                          + cross_scalar_vec(bodies.angular_velocities[c.body_b], rb))
                         - (bodies.velocities[c.body_a]
                          + cross_scalar_vec(bodies.angular_velocities[c.body_a], ra));

            // ── Normal impulse ────────────────────────────────
            float vn = glm::dot(rel_vel, c.normal);

            // Baumgarte bias for position correction
            float bias = config.baumgarte
                       * glm::max(c.penetration - config.slop, 0.0f);

            float lambda_n = c.normal_mass * (-vn + bias);

            // Clamp: accumulated impulse must be >= 0 (no pull)
            float new_impulse = glm::max(c.accumulated_normal_impulse + lambda_n, 0.0f);
            lambda_n = new_impulse - c.accumulated_normal_impulse;
            c.accumulated_normal_impulse = new_impulse;

            Vec2 impulse_n = c.normal * lambda_n;

            bodies.velocities[c.body_a] = bodies.velocities[c.body_a]
                                        - impulse_n * bodies.inverse_masses[c.body_a];
            bodies.angular_velocities[c.body_a] -= bodies.inverse_inertias[c.body_a]
                                                 * cross_vec_vec(ra, impulse_n);
            bodies.velocities[c.body_b] = bodies.velocities[c.body_b]
                                        + impulse_n * bodies.inverse_masses[c.body_b];
            bodies.angular_velocities[c.body_b] += bodies.inverse_inertias[c.body_b]
                                                 * cross_vec_vec(rb, impulse_n);

            // ── Tangent impulse (friction) ────────────────────
            // Recompute relative velocity after normal impulse
            rel_vel = (bodies.velocities[c.body_b]
                     + cross_scalar_vec(bodies.angular_velocities[c.body_b], rb))
                    - (bodies.velocities[c.body_a]
                     + cross_scalar_vec(bodies.angular_velocities[c.body_a], ra));

            float vt = glm::dot(rel_vel, c.tangent);
            float lambda_t = c.tangent_mass * (-vt);

            // Coulomb friction clamp: |f_t| <= mu * f_n
            float max_friction = config.friction * c.accumulated_normal_impulse;
            float new_tangent = glm::clamp(
                c.accumulated_tangent_impulse + lambda_t,
                -max_friction, max_friction);
            lambda_t = new_tangent - c.accumulated_tangent_impulse;
            c.accumulated_tangent_impulse = new_tangent;

            Vec2 impulse_t = c.tangent * lambda_t;

            bodies.velocities[c.body_a] = bodies.velocities[c.body_a]
                                        - impulse_t * bodies.inverse_masses[c.body_a];
            bodies.angular_velocities[c.body_a] -= bodies.inverse_inertias[c.body_a]
                                                 * cross_vec_vec(ra, impulse_t);
            bodies.velocities[c.body_b] = bodies.velocities[c.body_b]
                                        + impulse_t * bodies.inverse_masses[c.body_b];
            bodies.angular_velocities[c.body_b] += bodies.inverse_inertias[c.body_b]
                                                 * cross_vec_vec(rb, impulse_t);
        }
    }
}

} // namespace stan2d
```

### Step 5: Run tests to verify they pass

Run: `cmake --build build && ctest --test-dir build --output-on-failure`

Expected: PASS — all solver tests green

### Step 6: Commit

```bash
git add include/stan2d/constraints/contact_constraint.hpp \
        include/stan2d/constraints/solver.hpp \
        tests/unit/test_solver.cpp
git commit -m "feat: Sequential Impulse constraint solver with friction and warm starting"
```

---