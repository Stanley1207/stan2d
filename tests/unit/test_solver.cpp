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
    manifold.normal = {0.0f, -1.0f};  // A→B (ball down to floor)
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

    // Equal mass, head-on: velocities should swap or at least separate
    EXPECT_LE(bodies.velocities[a].x, bodies.velocities[b].x);
}

// ── Friction: ball sliding on surface ─────────────────────────────

TEST_F(SolverFixture, FrictionSlowsLateralVelocity) {
    uint32_t ball  = add_dynamic({0.0f, 0.5f}, {10.0f, -1.0f}, 1.0f);
    uint32_t floor = add_static({0.0f, 0.0f});

    ContactManifold manifold;
    manifold.normal = {0.0f, -1.0f};  // A→B (ball down to floor)
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
    manifold.normal = {0.0f, -1.0f};  // A→B (ball down to floor)
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
    manifold.normal = {0.0f, -1.0f};  // A→B (box down to floor)
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
