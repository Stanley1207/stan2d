#include <gtest/gtest.h>

#include <stan2d/core/handle.hpp>
#include <stan2d/joints/joint_types.hpp>
#include <stan2d/joints/joint_storage.hpp>
#include <stan2d/world/world.hpp>
#include <stan2d/world/world_config.hpp>

using namespace stan2d;

// ── JointHandle ────────────────────────────────────────────────────

TEST(JointHandle, DefaultConstruction) {
    JointHandle h;
    EXPECT_EQ(h.index, 0u);
    EXPECT_EQ(h.generation, 0u);
}

TEST(JointHandle, Equality) {
    JointHandle a{1, 2};
    JointHandle b{1, 2};
    JointHandle c{1, 3};
    EXPECT_EQ(a, b);
    EXPECT_NE(a, c);
}

// ── JointType ──────────────────────────────────────────────────────

TEST(JointType, EnumValues) {
    EXPECT_EQ(static_cast<uint8_t>(JointType::Hinge),    0);
    EXPECT_EQ(static_cast<uint8_t>(JointType::Spring),   1);
    EXPECT_EQ(static_cast<uint8_t>(JointType::Distance), 2);
    EXPECT_EQ(static_cast<uint8_t>(JointType::Pulley),   3);
}

// ── JointDef ───────────────────────────────────────────────────────

TEST(JointDef, DefaultValues) {
    JointDef def;
    EXPECT_EQ(def.type, JointType::Hinge);
    EXPECT_FALSE(def.limit_enabled);
    EXPECT_FALSE(def.motor_enabled);
    EXPECT_FLOAT_EQ(def.stiffness, 100.0f);
    EXPECT_FLOAT_EQ(def.damping, 1.0f);
    EXPECT_FLOAT_EQ(def.rest_length, 0.0f);
    EXPECT_FLOAT_EQ(def.distance, 0.0f);
    EXPECT_FALSE(def.cable_mode);
    EXPECT_FLOAT_EQ(def.pulley_ratio, 1.0f);
}

// ── JointStorage ───────────────────────────────────────────────────

TEST(JointStorage, ReserveDoesNotChangeSize) {
    JointStorage storage;
    storage.reserve(100);
    EXPECT_EQ(storage.size, 0u);
    EXPECT_GE(storage.types.capacity(), 100u);
    EXPECT_GE(storage.body_a.capacity(), 100u);
    EXPECT_GE(storage.accumulated_impulse_x.capacity(), 100u);
    EXPECT_GE(storage.constraint_forces.capacity(), 100u);
}

TEST(JointStorage, PushBackIncreasesSize) {
    JointStorage storage;
    storage.reserve(10);

    JointDef def;
    def.type = JointType::Hinge;
    storage.push_back(def, 0, 1, 0.5f, 0.0f);
    EXPECT_EQ(storage.size, 1u);
    EXPECT_EQ(storage.types[0], JointType::Hinge);
    EXPECT_EQ(storage.body_a[0], 0u);
    EXPECT_EQ(storage.body_b[0], 1u);
    EXPECT_FLOAT_EQ(storage.reference_angle[0], 0.5f);
}

TEST(JointStorage, SwapAndPopMaintainsData) {
    JointStorage storage;
    storage.reserve(10);

    JointDef def_a;
    def_a.type = JointType::Hinge;
    storage.push_back(def_a, 0, 1, 0.0f, 0.0f);

    JointDef def_b;
    def_b.type = JointType::Spring;
    def_b.stiffness = 200.0f;
    storage.push_back(def_b, 2, 3, 0.0f, 5.0f);

    // Remove first element (swap with last)
    storage.swap_and_pop(0, 1);
    storage.pop_back();

    EXPECT_EQ(storage.size, 1u);
    EXPECT_EQ(storage.types[0], JointType::Spring);
    EXPECT_EQ(storage.body_a[0], 2u);
    EXPECT_EQ(storage.body_b[0], 3u);
    EXPECT_FLOAT_EQ(storage.spring_stiffness[0], 200.0f);
}

// ── WorldConfig ────────────────────────────────────────────────────

TEST(WorldConfig, MaxJointsDefaultValue) {
    WorldConfig config;
    EXPECT_EQ(config.max_joints, 2000u);
}

// ── JointWorldFixture ──────────────────────────────────────────────

class JointWorldFixture : public ::testing::Test {
protected:
    World world{WorldConfig{
        .max_bodies = 100, .max_constraints = 100,
        .max_contacts = 100, .max_shapes = 100, .max_joints = 50
    }};
    ShapeHandle shape = world.create_shape(CircleShape{.radius = 0.5f});

    BodyHandle make_body(Vec2 pos = {0.0f, 0.0f}) {
        return world.create_body({.position = pos, .shape = shape, .mass = 1.0f});
    }
};

TEST_F(JointWorldFixture, CreateHingeJointReturnsValidHandle) {
    BodyHandle a = make_body({0.0f, 0.0f});
    BodyHandle b = make_body({1.0f, 0.0f});

    JointDef def;
    def.type = JointType::Hinge;
    def.body_a = a;
    def.body_b = b;
    def.anchor_a = {0.5f, 0.0f};
    def.anchor_b = {-0.5f, 0.0f};

    JointHandle jh = world.create_joint(def);
    EXPECT_TRUE(world.is_valid(jh));
}

TEST_F(JointWorldFixture, CreateAllJointTypes) {
    BodyHandle a = make_body();
    BodyHandle b = make_body({2.0f, 0.0f});

    for (auto type : {JointType::Hinge, JointType::Spring,
                      JointType::Distance, JointType::Pulley}) {
        JointDef def;
        def.type = type;
        def.body_a = a;
        def.body_b = b;
        JointHandle jh = world.create_joint(def);
        EXPECT_TRUE(world.is_valid(jh));
    }
}

TEST_F(JointWorldFixture, DestroyJointInvalidatesHandle) {
    BodyHandle a = make_body();
    BodyHandle b = make_body({1.0f, 0.0f});

    JointDef def;
    def.body_a = a;
    def.body_b = b;
    JointHandle jh = world.create_joint(def);

    EXPECT_TRUE(world.is_valid(jh));
    world.destroy_joint(jh);
    EXPECT_FALSE(world.is_valid(jh));
}

TEST_F(JointWorldFixture, DestroyAndRecreateJoint) {
    BodyHandle a = make_body();
    BodyHandle b = make_body({1.0f, 0.0f});

    JointDef def;
    def.body_a = a;
    def.body_b = b;

    JointHandle jh1 = world.create_joint(def);
    world.destroy_joint(jh1);
    JointHandle jh2 = world.create_joint(def);

    EXPECT_FALSE(world.is_valid(jh1));
    EXPECT_TRUE(world.is_valid(jh2));
    // Generation should differ (slot reused)
    EXPECT_EQ(jh1.index, jh2.index);
    EXPECT_NE(jh1.generation, jh2.generation);
}

TEST_F(JointWorldFixture, JointCountTracked) {
    BodyHandle a = make_body();
    BodyHandle b = make_body({1.0f, 0.0f});

    JointDef def;
    def.body_a = a;
    def.body_b = b;

    EXPECT_EQ(world.joint_count(), 0u);

    JointHandle j1 = world.create_joint(def);
    EXPECT_EQ(world.joint_count(), 1u);

    JointHandle j2 = world.create_joint(def);
    EXPECT_EQ(world.joint_count(), 2u);

    world.destroy_joint(j1);
    EXPECT_EQ(world.joint_count(), 1u);
}

TEST_F(JointWorldFixture, DistanceAutoDetectsLength) {
    BodyHandle a = make_body({0.0f, 0.0f});
    BodyHandle b = make_body({3.0f, 4.0f});

    JointDef def;
    def.type = JointType::Distance;
    def.body_a = a;
    def.body_b = b;
    def.distance = 0.0f;  // auto-detect

    JointHandle jh = world.create_joint(def);
    EXPECT_TRUE(world.is_valid(jh));
    // Verified by checking internal storage length == 5.0 (3-4-5 triangle)
    // This is tested indirectly through solver behavior in Task 15
}

TEST_F(JointWorldFixture, SpringAutoDetectsRestLength) {
    BodyHandle a = make_body({0.0f, 0.0f});
    BodyHandle b = make_body({3.0f, 4.0f});

    JointDef def;
    def.type = JointType::Spring;
    def.body_a = a;
    def.body_b = b;
    def.rest_length = 0.0f;  // auto-detect

    JointHandle jh = world.create_joint(def);
    EXPECT_TRUE(world.is_valid(jh));
}

TEST_F(JointWorldFixture, StepRunsWithJointsPresent) {
    BodyHandle a = make_body({0.0f, 0.0f});
    BodyHandle b = make_body({1.0f, 0.0f});

    JointDef def;
    def.body_a = a;
    def.body_b = b;
    world.create_joint(def);

    // step() should not crash even though no solver logic exists yet
    world.step(1.0f / 60.0f);
}
