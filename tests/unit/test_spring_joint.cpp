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
    BodyHandle bob = make_dynamic({3.0f, 0.0f});

    JointDef def;
    def.type = JointType::Spring;
    def.body_a = anchor;
    def.body_b = bob;
    def.stiffness = 50.0f;
    def.damping = 2.0f;
    def.rest_length = 1.0f;

    world.create_joint(def);

    float max_x_first_half = 0.0f;
    float max_x_second_half = 0.0f;

    for (int i = 0; i < 120; ++i) {
        world.step(dt);
        Vec2 pos = world.get_position(bob);
        float dist_from_rest = std::abs(pos.x - 0.0f);
        if (i < 60) {
            max_x_first_half = std::max(max_x_first_half, dist_from_rest);
        } else {
            max_x_second_half = std::max(max_x_second_half, dist_from_rest);
        }
    }

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
    def.damping = 50.0f;
    def.rest_length = 1.0f;

    world.create_joint(def);

    for (int i = 0; i < 300; ++i) {
        world.step(dt);
    }

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

    for (int i = 0; i < 5; ++i) {
        world.apply_force(b, {50.0f, 0.0f});
        world.step(dt);
    }
    for (int i = 0; i < 300; ++i) {
        world.step(dt);
    }

    Vec2 pos_a = world.get_position(a);
    Vec2 pos_b = world.get_position(b);
    float dist = glm::length(pos_b - pos_a);
    EXPECT_NEAR(dist, 2.0f, 0.8f);
}

TEST_F(SpringJointFixture, SpringPullsCompressedBodies) {
    world.set_gravity({0.0f, 0.0f});

    BodyHandle anchor = make_static({0.0f, 0.0f});
    BodyHandle bob = make_dynamic({0.5f, 0.0f});

    JointDef def;
    def.type = JointType::Spring;
    def.body_a = anchor;
    def.body_b = bob;
    def.stiffness = 100.0f;
    def.damping = 5.0f;
    def.rest_length = 3.0f;

    world.create_joint(def);

    for (int i = 0; i < 60; ++i) {
        world.step(dt);
    }

    Vec2 pos = world.get_position(bob);
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

    Vec2 bob_pos = world.get_position(bob);
    EXPECT_LT(bob_pos.y, 5.0f);
    float dist = glm::length(bob_pos - Vec2{0.0f, 5.0f});
    EXPECT_GT(dist, 2.0f);
    EXPECT_LT(dist, 5.0f);
}

TEST_F(SpringJointFixture, NoWarmStartState) {
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

    for (int i = 0; i < 60; ++i) {
        world.step(dt);
    }

    Vec2 pos = world.get_position(b);
    EXPECT_LT(pos.x, 2.0f);
}
