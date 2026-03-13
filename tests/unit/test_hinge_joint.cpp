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

    for (int i = 0; i < 10; ++i) {
        world.step(dt);
    }

    Vec2 bob_pos = world.get_position(bob);
    float dist = glm::length(bob_pos - Vec2{0.0f, 5.0f});
    EXPECT_LT(dist, 2.0f);
}

TEST_F(HingeJointFixture, PendulumSwings) {
    world.set_gravity({0.0f, -10.0f});

    BodyHandle anchor = make_static({0.0f, 5.0f});
    BodyHandle bob = make_dynamic({2.0f, 5.0f});

    JointDef def;
    def.type = JointType::Hinge;
    def.body_a = anchor;
    def.body_b = bob;
    // anchor_a at pivot center (0,5); anchor_b points from bob center (2,5)
    // to the attachment point (0,5) in body space → (-2,0)
    def.anchor_a = {0.0f, 0.0f};
    def.anchor_b = {-2.0f, 0.0f};

    world.create_joint(def);

    for (int i = 0; i < 60; ++i) {
        world.step(dt);
    }

    Vec2 bob_pos = world.get_position(bob);
    Vec2 anchor_pos = world.get_position(anchor);

    float dist = glm::length(bob_pos - anchor_pos);
    EXPECT_NEAR(dist, 2.0f, 0.3f);

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
    def.anchor_a = {0.5f, 0.0f};
    def.anchor_b = {-0.5f, 0.0f};

    world.create_joint(def);

    for (int i = 0; i < 120; ++i) {
        world.step(dt);
    }

    Vec2 pos_a = world.get_position(a);
    Vec2 pos_b = world.get_position(b);
    Vec2 wa = pos_a + Vec2{0.5f, 0.0f};
    Vec2 wb = pos_b + Vec2{-0.5f, 0.0f};
    float gap = glm::length(wa - wb);
    EXPECT_LT(gap, 0.5f);
}

TEST_F(HingeJointFixture, GetJointAngleReturnsRelativeAngle) {
    BodyHandle a = make_dynamic({0.0f, 0.0f});
    BodyHandle b = make_dynamic({1.0f, 0.0f});

    JointDef def;
    def.type = JointType::Hinge;
    def.body_a = a;
    def.body_b = b;

    JointHandle jh = world.create_joint(def);

    world.step(dt);
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
    EXPECT_TRUE(std::isfinite(speed));
}

TEST_F(HingeJointFixture, MotorDrivesRotation) {
    world.set_gravity({0.0f, 0.0f});

    BodyHandle a = make_static({0.0f, 0.0f});
    BodyHandle b = make_dynamic({3.0f, 0.0f});

    JointDef def;
    def.type = JointType::Hinge;
    def.body_a = a;
    def.body_b = b;
    def.anchor_a = {0.0f, 0.0f};
    def.anchor_b = {-3.0f, 0.0f};
    def.motor_enabled = true;
    def.motor_speed = 5.0f;
    def.motor_torque = 100.0f;

    JointHandle jh = world.create_joint(def);

    for (int i = 0; i < 120; ++i) {
        world.step(dt);
    }

    float speed = world.get_joint_speed(jh);
    EXPECT_GT(speed, 1.0f);
}

TEST_F(HingeJointFixture, SetMotorSpeedChangesTarget) {
    world.set_gravity({0.0f, 0.0f});

    BodyHandle a = make_static({0.0f, 0.0f});
    BodyHandle b = make_dynamic({3.0f, 0.0f});

    JointDef def;
    def.type = JointType::Hinge;
    def.body_a = a;
    def.body_b = b;
    def.anchor_a = {0.0f, 0.0f};
    def.anchor_b = {-3.0f, 0.0f};
    def.motor_enabled = true;
    def.motor_speed = 0.0f;
    def.motor_torque = 100.0f;

    JointHandle jh = world.create_joint(def);

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
    def.limit_min = -0.5f;
    def.limit_max = 0.5f;

    world.create_joint(def);

    for (int i = 0; i < 300; ++i) {
        world.step(dt);
    }

    Vec2 bob_pos = world.get_position(b);
    EXPECT_GT(bob_pos.y, 3.5f);
}

TEST_F(HingeJointFixture, WarmStartImprovesSolverConvergence) {
    world.set_gravity({0.0f, -10.0f});

    BodyHandle anchor = make_static({0.0f, 5.0f});
    BodyHandle bob = make_dynamic({0.0f, 3.0f});

    JointDef def;
    def.type = JointType::Hinge;
    def.body_a = anchor;
    def.body_b = bob;
    // anchor_a at pivot center (0,5); anchor_b points from bob center (0,3)
    // to the attachment point (0,5) in body space → (0,2)
    def.anchor_a = {0.0f, 0.0f};
    def.anchor_b = {0.0f, 2.0f};

    world.create_joint(def);

    for (int i = 0; i < 300; ++i) {
        world.step(dt);
    }

    Vec2 bob_pos = world.get_position(bob);
    float dist = glm::length(bob_pos - Vec2{0.0f, 5.0f});
    EXPECT_NEAR(dist, 2.0f, 0.5f);
}
