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

    float len = world.get_joint_length(jh);
    EXPECT_NEAR(len, 3.0f, 0.5f);
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

    for (int i = 0; i < 60; ++i) {
        world.apply_force(b, {100.0f, 0.0f});
        world.step(dt);
    }

    float len = world.get_joint_length(jh);
    EXPECT_NEAR(len, 5.0f, 1.0f);
}

TEST_F(DistanceJointFixture, CableModeAllowsCompression) {
    world.set_gravity({0.0f, 0.0f});

    BodyHandle a = make_static({0.0f, 0.0f});
    BodyHandle b = make_dynamic({5.0f, 0.0f});

    JointDef def;
    def.type = JointType::Distance;
    def.body_a = a;
    def.body_b = b;
    def.distance = 5.0f;
    def.cable_mode = true;

    world.create_joint(def);

    for (int i = 0; i < 60; ++i) {
        world.apply_force(b, {-50.0f, 0.0f});
        world.step(dt);
    }

    Vec2 bob_pos = world.get_position(b);
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

    for (int i = 0; i < 120; ++i) {
        world.apply_force(b, {100.0f, 0.0f});
        world.step(dt);
    }

    float len = world.get_joint_length(jh);
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
    def.cable_mode = false;

    JointHandle jh = world.create_joint(def);

    for (int i = 0; i < 60; ++i) {
        world.apply_force(b, {-50.0f, 0.0f});
        world.step(dt);
    }

    float len = world.get_joint_length(jh);
    EXPECT_NEAR(len, 5.0f, 0.8f);
}
