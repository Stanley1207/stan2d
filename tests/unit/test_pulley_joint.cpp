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

    float len_a = glm::length(pos_a - Vec2{-2.0f, 5.0f});
    float len_b = glm::length(pos_b - Vec2{2.0f, 5.0f});
    float total = len_a + len_b;

    float orig_len_a = glm::length(initial_a - Vec2{-2.0f, 5.0f});
    float orig_len_b = glm::length(initial_b - Vec2{2.0f, 5.0f});
    float orig_total = orig_len_a + orig_len_b;

    EXPECT_NEAR(total, orig_total, 0.5f);
}

TEST_F(PulleyJointFixture, UnequalRatioLift) {
    world.set_gravity({0.0f, -10.0f});

    BodyHandle a = make_dynamic({-2.0f, 3.0f}, 2.0f);
    BodyHandle b = make_dynamic({2.0f, 3.0f}, 1.0f);

    JointDef def;
    def.type = JointType::Pulley;
    def.body_a = a;
    def.body_b = b;
    def.ground_a = {-2.0f, 5.0f};
    def.ground_b = {2.0f, 5.0f};
    def.pulley_ratio = 2.0f;

    world.create_joint(def);

    for (int i = 0; i < 120; ++i) {
        world.step(dt);
    }

    Vec2 pos_a = world.get_position(a);
    Vec2 pos_b = world.get_position(b);

    float len_a = glm::length(pos_a - Vec2{-2.0f, 5.0f});
    float len_b = glm::length(pos_b - Vec2{2.0f, 5.0f});
    float orig_total = 2.0f + 2.0f * 2.0f;
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

    EXPECT_LT(max_error, 1.0f);
}

TEST_F(PulleyJointFixture, OneBodyFallingLiftsOther) {
    world.set_gravity({0.0f, -10.0f});

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

    Vec2 pos_a = world.get_position(a);
    Vec2 pos_b = world.get_position(b);

    EXPECT_LT(pos_a.y, 3.0f);
    EXPECT_GT(pos_b.y, initial_b_y - 0.5f);
}
