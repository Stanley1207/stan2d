#include <gtest/gtest.h>
#include <cmath>
#include <limits>
#include <fstream>
#include <filesystem>

#include <stan2d/world/world.hpp>
#include <stan2d/world/state_view.hpp>
#include <stan2d/world/snapshot.hpp>
#include <stan2d/export/trajectory_recorder.hpp>
#include <stan2d/export/state_export.hpp>
#include <nlohmann/json.hpp>

using namespace stan2d;

class JointStateSystemFixture : public ::testing::Test {
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

// ── Task 18.1: JointStateView ─────────────────────────────────────

TEST_F(JointStateSystemFixture, JointStateViewReflectsJoints) {
    BodyHandle a = make_static({0.0f, 0.0f});
    BodyHandle b = make_dynamic({2.0f, 0.0f});

    JointDef def;
    def.type = JointType::Hinge;
    def.body_a = a;
    def.body_b = b;
    def.motor_enabled = true;
    def.motor_speed = 3.0f;
    def.motor_torque = 10.0f;

    world.create_joint(def);
    world.step(dt);  // Populate cached values

    WorldStateView view = world.get_state_view();
    EXPECT_EQ(view.joints.active_joint_count, 1u);
    EXPECT_EQ(view.joints.types.size(), 1u);
    EXPECT_EQ(view.joints.angles.size(), 1u);
    EXPECT_EQ(view.joints.angular_speeds.size(), 1u);
    EXPECT_EQ(view.joints.motor_target_speeds.size(), 1u);
    EXPECT_EQ(view.joints.motor_enabled.size(), 1u);
    EXPECT_EQ(view.joints.constraint_forces.size(), 1u);
    EXPECT_EQ(view.joints.lengths.size(), 1u);

    // Motor enabled should be 1
    EXPECT_EQ(view.joints.motor_enabled[0], 1);
    // Motor speed should reflect set value
    EXPECT_FLOAT_EQ(view.joints.motor_target_speeds[0], 3.0f);
}

TEST_F(JointStateSystemFixture, JointStateViewEmptyWorld) {
    WorldStateView view = world.get_state_view();
    EXPECT_EQ(view.joints.active_joint_count, 0u);
    EXPECT_TRUE(view.joints.types.empty());
    EXPECT_TRUE(view.joints.angles.empty());
}

TEST_F(JointStateSystemFixture, JointStateViewMultipleTypes) {
    BodyHandle a = make_static({0.0f, 0.0f});
    BodyHandle b = make_dynamic({2.0f, 0.0f});
    BodyHandle c = make_dynamic({4.0f, 0.0f});

    JointDef hinge_def;
    hinge_def.type = JointType::Hinge;
    hinge_def.body_a = a;
    hinge_def.body_b = b;
    world.create_joint(hinge_def);

    JointDef spring_def;
    spring_def.type = JointType::Spring;
    spring_def.body_a = b;
    spring_def.body_b = c;
    spring_def.stiffness = 50.0f;
    world.create_joint(spring_def);

    world.step(dt);

    WorldStateView view = world.get_state_view();
    EXPECT_EQ(view.joints.active_joint_count, 2u);

    // Verify types are readable as uint8_t
    EXPECT_EQ(view.joints.types[0], static_cast<uint8_t>(JointType::Hinge));
    EXPECT_EQ(view.joints.types[1], static_cast<uint8_t>(JointType::Spring));

    // Non-motor joint (Spring) should have NaN motor_target_speed per spec
    // RL agents must mask by motor_enabled
    EXPECT_EQ(view.joints.motor_enabled[1], 0);
    EXPECT_TRUE(std::isnan(view.joints.motor_target_speeds[1]));
}

// ── Task 18.2: JointSnapshot save/restore ────────────────────────

TEST_F(JointStateSystemFixture, SaveRestoreJointState) {
    world.set_gravity({0.0f, -10.0f});

    BodyHandle a = make_static({0.0f, 5.0f});
    BodyHandle b = make_dynamic({2.0f, 5.0f});

    JointDef def;
    def.type = JointType::Hinge;
    def.body_a = a;
    def.body_b = b;
    def.motor_enabled = true;
    def.motor_speed = 5.0f;
    def.motor_torque = 50.0f;

    JointHandle jh = world.create_joint(def);

    // Run some steps to build up warm-start state
    for (int i = 0; i < 30; ++i) {
        world.step(dt);
    }

    // Save state
    WorldSnapshot snapshot;
    world.save_state(snapshot);

    float angle_at_save = world.get_joint_angle(jh);

    // Run more steps to change state
    for (int i = 0; i < 60; ++i) {
        world.step(dt);
    }

    // State should have changed
    EXPECT_NE(world.get_joint_angle(jh), angle_at_save);

    // Restore
    world.restore_state(snapshot);

    // Joint should be valid again
    EXPECT_TRUE(world.is_valid(jh));
    EXPECT_EQ(world.joint_count(), 1u);
}

TEST_F(JointStateSystemFixture, SaveRestoreDeterministicReplay) {
    world.set_gravity({0.0f, -10.0f});

    BodyHandle a = make_static({0.0f, 5.0f});
    BodyHandle b = make_dynamic({2.0f, 5.0f});

    JointDef def;
    def.type = JointType::Hinge;
    def.body_a = a;
    def.body_b = b;

    JointHandle jh = world.create_joint(def);

    // Warm up
    for (int i = 0; i < 10; ++i) {
        world.step(dt);
    }

    // Save
    WorldSnapshot snapshot;
    world.save_state(snapshot);

    // Run 120 steps and record final state
    for (int i = 0; i < 120; ++i) {
        world.step(dt);
    }
    Vec2 pos_run1 = world.get_position(b);
    float angle_run1 = world.get_joint_angle(jh);

    // Restore and replay
    world.restore_state(snapshot);
    for (int i = 0; i < 120; ++i) {
        world.step(dt);
    }
    Vec2 pos_run2 = world.get_position(b);
    float angle_run2 = world.get_joint_angle(jh);

    // Must be bit-exact
    EXPECT_FLOAT_EQ(pos_run1.x, pos_run2.x);
    EXPECT_FLOAT_EQ(pos_run1.y, pos_run2.y);
    EXPECT_FLOAT_EQ(angle_run1, angle_run2);
}

TEST_F(JointStateSystemFixture, SaveRestoreWarmStartCompleteness) {
    world.set_gravity({0.0f, -10.0f});

    BodyHandle a = make_static({0.0f, 5.0f});
    BodyHandle b = make_dynamic({0.0f, 3.0f});

    JointDef def;
    def.type = JointType::Hinge;
    def.body_a = a;
    def.body_b = b;
    def.limit_enabled = true;
    def.limit_min = -1.0f;
    def.limit_max = 1.0f;
    def.motor_enabled = true;
    def.motor_speed = 2.0f;
    def.motor_torque = 20.0f;

    world.create_joint(def);

    // Run to build up warm-start state (limit + motor impulses)
    for (int i = 0; i < 60; ++i) {
        world.step(dt);
    }

    // Save
    WorldSnapshot snap;
    world.save_state(snap);

    // Verify snapshot has joint data
    EXPECT_EQ(snap.joints.count, 1u);
    EXPECT_EQ(snap.joints.types.size(), 1u);
}

TEST_F(JointStateSystemFixture, DestroyJointThenRestoreRestoresIt) {
    BodyHandle a = make_static({0.0f, 0.0f});
    BodyHandle b = make_dynamic({2.0f, 0.0f});

    JointDef def;
    def.type = JointType::Distance;
    def.body_a = a;
    def.body_b = b;
    JointHandle jh = world.create_joint(def);

    WorldSnapshot snap;
    world.save_state(snap);

    world.destroy_joint(jh);
    EXPECT_FALSE(world.is_valid(jh));
    EXPECT_EQ(world.joint_count(), 0u);

    world.restore_state(snap);
    EXPECT_TRUE(world.is_valid(jh));
    EXPECT_EQ(world.joint_count(), 1u);
}

TEST_F(JointStateSystemFixture, MCTSRollbackScenario) {
    world.set_gravity({0.0f, -10.0f});

    BodyHandle a = make_static({0.0f, 5.0f});
    BodyHandle b = make_dynamic({2.0f, 5.0f});

    JointDef def;
    def.type = JointType::Hinge;
    def.body_a = a;
    def.body_b = b;
    def.anchor_b = {-2.0f, 0.0f};  // arm of 2 units from hinge to body b center
    def.motor_enabled = true;
    def.motor_speed = 0.0f;
    def.motor_torque = 50.0f;

    JointHandle jh = world.create_joint(def);

    // Save root state
    WorldSnapshot root;
    for (int i = 0; i < 10; ++i) { world.step(dt); }
    world.save_state(root);

    // Branch 1: motor CW
    world.set_motor_speed(jh, 5.0f);
    for (int i = 0; i < 60; ++i) { world.step(dt); }
    Vec2 pos_branch1 = world.get_position(b);

    // Rollback to root
    world.restore_state(root);

    // Verify motor speed was restored to snapshot value (0.0f)
    world.step(dt);  // populate cached values
    WorldStateView view_after_restore = world.get_state_view();
    EXPECT_FLOAT_EQ(view_after_restore.joints.motor_target_speeds[0], 0.0f);
    world.restore_state(root);  // re-restore since we stepped

    // Branch 2: motor CCW
    world.set_motor_speed(jh, -5.0f);
    for (int i = 0; i < 60; ++i) { world.step(dt); }
    Vec2 pos_branch2 = world.get_position(b);

    // Branches should diverge
    float divergence = glm::length(pos_branch1 - pos_branch2);
    EXPECT_GT(divergence, 0.5f);

    // Rollback to root and replay branch 1 — should be identical
    world.restore_state(root);
    world.set_motor_speed(jh, 5.0f);
    for (int i = 0; i < 60; ++i) { world.step(dt); }
    Vec2 pos_replay = world.get_position(b);

    EXPECT_FLOAT_EQ(pos_branch1.x, pos_replay.x);
    EXPECT_FLOAT_EQ(pos_branch1.y, pos_replay.y);
}

// ── Task 18.3: TrajectoryRecorder joint support ───────────────────

TEST_F(JointStateSystemFixture, TrajectoryRecorderCapturesJoints) {
    world.set_gravity({0.0f, -10.0f});

    BodyHandle a = make_static({0.0f, 5.0f});
    BodyHandle b = make_dynamic({2.0f, 5.0f});

    JointDef def;
    def.type = JointType::Hinge;
    def.body_a = a;
    def.body_b = b;

    world.create_joint(def);

    TrajectoryRecorder recorder(world, 10);
    recorder.start();

    for (int i = 0; i < 5; ++i) {
        world.step(dt);
        recorder.capture();
    }

    EXPECT_EQ(recorder.current_frame(), 5u);
    EXPECT_EQ(recorder.max_joints(), 50u);  // from config.max_joints

    // Verify joint active count
    EXPECT_EQ(recorder.get_joint_active_count(0), 1u);
}

TEST_F(JointStateSystemFixture, TrajectoryJointTensorShape) {
    BodyHandle a = make_static({0.0f, 5.0f});
    BodyHandle b = make_dynamic({2.0f, 5.0f});

    JointDef def;
    def.type = JointType::Hinge;
    def.body_a = a;
    def.body_b = b;
    world.create_joint(def);

    TrajectoryRecorder recorder(world, 10);
    recorder.start();  // Required before capture()

    world.step(dt);
    recorder.capture();

    // Raw joint data should have shape [max_frames * max_joints]
    auto raw_angles = recorder.raw_joint_angles();
    EXPECT_EQ(raw_angles.size(), 10u * 50u);  // max_frames * max_joints

    auto raw_lengths = recorder.raw_joint_lengths();
    EXPECT_EQ(raw_lengths.size(), 10u * 50u);
}

TEST_F(JointStateSystemFixture, TrajectoryJointDataAccessors) {
    world.set_gravity({0.0f, -10.0f});

    BodyHandle a = make_static({0.0f, 5.0f});
    BodyHandle b = make_dynamic({2.0f, 5.0f});

    JointDef def;
    def.type = JointType::Hinge;
    def.body_a = a;
    def.body_b = b;
    world.create_joint(def);

    TrajectoryRecorder recorder(world, 10);
    recorder.start();

    for (int i = 0; i < 3; ++i) {
        world.step(dt);
        recorder.capture();
    }

    // Length should be nonzero (2-unit arm)
    float len = recorder.get_joint_length(0, 0);
    EXPECT_GT(len, 0.0f);
}

// ── Task 18.4: export_state() joint support ───────────────────────

TEST_F(JointStateSystemFixture, ExportStateJSONIncludesJoints) {
    BodyHandle a = make_static({0.0f, 0.0f});
    BodyHandle b = make_dynamic({2.0f, 0.0f});

    JointDef def;
    def.type = JointType::Hinge;
    def.body_a = a;
    def.body_b = b;
    def.motor_enabled = true;
    def.motor_speed = 3.0f;
    world.create_joint(def);
    world.step(dt);

    auto temp = std::filesystem::temp_directory_path() / "stan2d_joint_export_test.json";

    export_state(world, temp.string(), ExportFormat::JSON);

    std::ifstream file(temp);
    nlohmann::json j;
    file >> j;

    EXPECT_TRUE(j.contains("joints"));
    EXPECT_EQ(j["joint_count"], 1);
    EXPECT_EQ(j["joints"].size(), 1u);
    EXPECT_EQ(j["joints"][0]["type"], "Hinge");
    EXPECT_TRUE(j["joints"][0].contains("angle"));
    EXPECT_TRUE(j["joints"][0].contains("constraint_force"));
    EXPECT_TRUE(j["joints"][0].contains("length"));

    std::filesystem::remove(temp);
}

TEST_F(JointStateSystemFixture, ExportStateJSONNoJoints) {
    make_dynamic({0.0f, 0.0f});
    world.step(dt);

    auto temp = std::filesystem::temp_directory_path() / "stan2d_no_joints_test.json";
    export_state(world, temp.string(), ExportFormat::JSON);

    std::ifstream file(temp);
    nlohmann::json j;
    file >> j;

    EXPECT_EQ(j["joint_count"], 0);
    EXPECT_TRUE(j["joints"].empty());

    std::filesystem::remove(temp);
}

// ── Task 18.5: TrajectoryRecorder::save() joint support ──────────

TEST_F(JointStateSystemFixture, TrajectoryJSONExportIncludesJoints) {
    world.set_gravity({0.0f, -10.0f});

    BodyHandle a = make_static({0.0f, 5.0f});
    BodyHandle b = make_dynamic({2.0f, 5.0f});

    JointDef def;
    def.type = JointType::Hinge;
    def.body_a = a;
    def.body_b = b;
    world.create_joint(def);

    TrajectoryRecorder recorder(world, 5);
    recorder.start();

    for (int i = 0; i < 3; ++i) {
        world.step(dt);
        recorder.capture();
    }

    auto temp = std::filesystem::temp_directory_path() / "stan2d_traj_joints_test.json";
    recorder.save(temp.string(), ExportFormat::JSON);

    std::ifstream file(temp);
    nlohmann::json j;
    file >> j;

    EXPECT_TRUE(j.contains("max_joints"));
    EXPECT_EQ(j["max_joints"], 50);
    EXPECT_TRUE(j["frames"][0].contains("joint_active_count"));
    EXPECT_EQ(j["frames"][0]["joint_active_count"], 1);
    EXPECT_TRUE(j["frames"][0].contains("joint_angles"));

    std::filesystem::remove(temp);
}
