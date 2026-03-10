#include <gtest/gtest.h>
#include <stan2d/dynamics/integrator.hpp>
#include <stan2d/dynamics/body_storage.hpp>
#include <stan2d/core/handle.hpp>

using namespace stan2d;

// Helper: create a minimal body storage with one dynamic body
struct IntegratorFixture : public ::testing::Test {
    BodyStorage bodies;
    uint32_t count = 0;

    void add_body(Vec2 pos, Vec2 vel, float mass, BodyType type) {
        float inv_mass    = (type == BodyType::Dynamic && mass > 0.0f) ? 1.0f / mass : 0.0f;
        float inertia     = 1.0f;
        float inv_inertia = (type == BodyType::Dynamic) ? 1.0f : 0.0f;
        bodies.push_back(pos, vel, 0.0f, 0.0f,
                         mass, inv_mass, inertia, inv_inertia,
                         type, ShapeHandle{});
        ++count;
    }
};

// ── Basic integration ─────────────────────────────────────────────

TEST_F(IntegratorFixture, VelocityUnchangedWithoutForces) {
    add_body({0.0f, 0.0f}, {5.0f, 0.0f}, 1.0f, BodyType::Dynamic);

    Vec2 gravity{0.0f, 0.0f};
    float dt = 1.0f / 60.0f;

    integrate_velocities(bodies, count, gravity, dt);
    integrate_positions(bodies, count, dt);

    EXPECT_FLOAT_EQ(bodies.velocities[0].x, 5.0f);
    EXPECT_FLOAT_EQ(bodies.velocities[0].y, 0.0f);

    // position = old_pos + v * dt = 0 + 5 * (1/60)
    EXPECT_NEAR(bodies.positions[0].x, 5.0f / 60.0f, 1e-5f);
    EXPECT_NEAR(bodies.positions[0].y, 0.0f, 1e-5f);
}

TEST_F(IntegratorFixture, GravityAcceleratesBody) {
    add_body({0.0f, 10.0f}, {0.0f, 0.0f}, 1.0f, BodyType::Dynamic);

    Vec2 gravity{0.0f, -9.81f};
    float dt = 1.0f / 60.0f;

    integrate_velocities(bodies, count, gravity, dt);

    // v_new = v + (F/m + gravity) * dt = 0 + (-9.81) * (1/60)
    EXPECT_NEAR(bodies.velocities[0].y, -9.81f / 60.0f, 1e-5f);

    integrate_positions(bodies, count, dt);

    // Symplectic: x_new = x + v_new * dt
    float expected_vy = -9.81f / 60.0f;
    EXPECT_NEAR(bodies.positions[0].y, 10.0f + expected_vy / 60.0f, 1e-5f);
}

TEST_F(IntegratorFixture, ExternalForceApplied) {
    add_body({0.0f, 0.0f}, {0.0f, 0.0f}, 2.0f, BodyType::Dynamic);

    // Apply external force
    bodies.forces[0] = {10.0f, 0.0f};

    Vec2 gravity{0.0f, 0.0f};
    float dt = 1.0f / 60.0f;

    integrate_velocities(bodies, count, gravity, dt);

    // v = 0 + (F/m) * dt = (10/2) * (1/60) = 5/60
    EXPECT_NEAR(bodies.velocities[0].x, 5.0f / 60.0f, 1e-5f);

    // Forces should be cleared after integration
    EXPECT_FLOAT_EQ(bodies.forces[0].x, 0.0f);
    EXPECT_FLOAT_EQ(bodies.forces[0].y, 0.0f);
}

TEST_F(IntegratorFixture, AngularIntegration) {
    add_body({0.0f, 0.0f}, {0.0f, 0.0f}, 1.0f, BodyType::Dynamic);
    bodies.angular_velocities[0] = 3.14f;
    bodies.torques[0] = 2.0f;

    Vec2 gravity{0.0f, 0.0f};
    float dt = 1.0f / 60.0f;

    integrate_velocities(bodies, count, gravity, dt);

    // angular_vel += (torque / inertia) * dt = 3.14 + (2/1) * (1/60)
    float expected_av = 3.14f + 2.0f / 60.0f;
    EXPECT_NEAR(bodies.angular_velocities[0], expected_av, 1e-5f);

    // Torque should be cleared
    EXPECT_FLOAT_EQ(bodies.torques[0], 0.0f);

    integrate_positions(bodies, count, dt);

    // rotation += angular_vel * dt
    EXPECT_NEAR(bodies.rotations[0], expected_av / 60.0f, 1e-5f);
}

// ── Body type filtering ───────────────────────────────────────────

TEST_F(IntegratorFixture, StaticBodyNotIntegrated) {
    add_body({5.0f, 5.0f}, {0.0f, 0.0f}, 0.0f, BodyType::Static);

    Vec2 gravity{0.0f, -9.81f};
    float dt = 1.0f / 60.0f;

    integrate_velocities(bodies, count, gravity, dt);
    integrate_positions(bodies, count, dt);

    EXPECT_FLOAT_EQ(bodies.positions[0].x, 5.0f);
    EXPECT_FLOAT_EQ(bodies.positions[0].y, 5.0f);
    EXPECT_FLOAT_EQ(bodies.velocities[0].x, 0.0f);
    EXPECT_FLOAT_EQ(bodies.velocities[0].y, 0.0f);
}

TEST_F(IntegratorFixture, KinematicBodyPositionIntegratedButNotForce) {
    add_body({0.0f, 0.0f}, {2.0f, 0.0f}, 0.0f, BodyType::Kinematic);

    Vec2 gravity{0.0f, -9.81f};
    float dt = 1.0f / 60.0f;

    integrate_velocities(bodies, count, gravity, dt);

    // Kinematic: velocity NOT affected by gravity or forces
    EXPECT_FLOAT_EQ(bodies.velocities[0].x, 2.0f);
    EXPECT_FLOAT_EQ(bodies.velocities[0].y, 0.0f);

    integrate_positions(bodies, count, dt);

    // But position IS updated from velocity
    EXPECT_NEAR(bodies.positions[0].x, 2.0f / 60.0f, 1e-5f);
}

// ── Multiple bodies ───────────────────────────────────────────────

TEST_F(IntegratorFixture, MixedBodyTypes) {
    add_body({0.0f, 0.0f}, {1.0f, 0.0f}, 1.0f, BodyType::Dynamic);
    add_body({5.0f, 5.0f}, {0.0f, 0.0f}, 0.0f, BodyType::Static);
    add_body({10.0f, 0.0f}, {3.0f, 0.0f}, 0.0f, BodyType::Kinematic);

    Vec2 gravity{0.0f, -10.0f};
    float dt = 0.1f;

    integrate_velocities(bodies, count, gravity, dt);
    integrate_positions(bodies, count, dt);

    // Dynamic: vel.y = 0 + (-10)*0.1 = -1.0, pos = {0+1*0.1, 0+(-1)*0.1}
    EXPECT_NEAR(bodies.velocities[0].y, -1.0f, 1e-5f);
    EXPECT_NEAR(bodies.positions[0].x, 0.1f, 1e-5f);
    EXPECT_NEAR(bodies.positions[0].y, -0.1f, 1e-5f);

    // Static: unchanged
    EXPECT_FLOAT_EQ(bodies.positions[1].x, 5.0f);
    EXPECT_FLOAT_EQ(bodies.positions[1].y, 5.0f);

    // Kinematic: vel unchanged, pos = 10 + 3*0.1
    EXPECT_FLOAT_EQ(bodies.velocities[2].x, 3.0f);
    EXPECT_NEAR(bodies.positions[2].x, 10.3f, 1e-5f);
}
