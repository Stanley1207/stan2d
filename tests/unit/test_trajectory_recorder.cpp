#include <gtest/gtest.h>
#include <stan2d/export/trajectory_recorder.hpp>
#include <stan2d/world/world.hpp>

using namespace stan2d;

// ── Basic recording ───────────────────────────────────────────────

TEST(TrajectoryRecorder, ConstructWithWorld) {
    World world(WorldConfig{.max_bodies = 100, .max_shapes = 100, .max_contacts = 100});
    TrajectoryRecorder recorder(world, 60);

    EXPECT_EQ(recorder.max_frames(), 60u);
    EXPECT_EQ(recorder.current_frame(), 0u);
}

TEST(TrajectoryRecorder, CaptureIncreasesFrameCount) {
    World world(WorldConfig{.max_bodies = 100, .max_shapes = 100, .max_contacts = 100});

    ShapeHandle shape = world.create_shape(CircleShape{.radius = 0.5f});
    world.create_body({.position = {0.0f, 0.0f}, .shape = shape, .mass = 1.0f});

    TrajectoryRecorder recorder(world, 60);
    recorder.start();
    recorder.capture();

    EXPECT_EQ(recorder.current_frame(), 1u);
}

TEST(TrajectoryRecorder, CaptureRecordsPositions) {
    World world(WorldConfig{.max_bodies = 10, .max_shapes = 10, .max_contacts = 10});

    ShapeHandle shape = world.create_shape(CircleShape{.radius = 0.5f});
    world.create_body({.position = {1.0f, 2.0f}, .shape = shape, .mass = 1.0f});
    world.create_body({.position = {3.0f, 4.0f}, .shape = shape, .mass = 1.0f});

    TrajectoryRecorder recorder(world, 10);
    recorder.start();
    recorder.capture();

    // Frame 0, body 0
    Vec2 pos = recorder.get_position(0, 0);
    EXPECT_FLOAT_EQ(pos.x, 1.0f);
    EXPECT_FLOAT_EQ(pos.y, 2.0f);

    // Frame 0, body 1
    Vec2 pos2 = recorder.get_position(0, 1);
    EXPECT_FLOAT_EQ(pos2.x, 3.0f);
    EXPECT_FLOAT_EQ(pos2.y, 4.0f);
}

TEST(TrajectoryRecorder, MultipleFramesTrackMotion) {
    World world(WorldConfig{.max_bodies = 10, .max_shapes = 10, .max_contacts = 10});
    world.set_gravity({0.0f, -10.0f});

    ShapeHandle shape = world.create_shape(CircleShape{.radius = 0.5f});
    world.create_body({.position = {0.0f, 10.0f}, .shape = shape, .mass = 1.0f});

    TrajectoryRecorder recorder(world, 100);
    recorder.start();

    float dt = 1.0f / 60.0f;
    for (int i = 0; i < 10; ++i) {
        recorder.capture();
        world.step(dt);
    }

    EXPECT_EQ(recorder.current_frame(), 10u);

    // Each frame should show the ball lower than the previous
    for (uint32_t f = 1; f < 10; ++f) {
        float y_prev = recorder.get_position(f - 1, 0).y;
        float y_curr = recorder.get_position(f, 0).y;
        EXPECT_LT(y_curr, y_prev);
    }
}

// ── Fixed stride layout ───────────────────────────────────────────

TEST(TrajectoryRecorder, FixedStrideLayout) {
    World world(WorldConfig{.max_bodies = 10, .max_shapes = 10, .max_contacts = 10});

    ShapeHandle shape = world.create_shape(CircleShape{.radius = 0.5f});
    world.create_body({.position = {1.0f, 2.0f}, .shape = shape, .mass = 1.0f});

    TrajectoryRecorder recorder(world, 5);
    recorder.start();
    recorder.capture();

    // Get raw position buffer: should have max_bodies slots per frame
    std::span<const Vec2> raw_positions = recorder.raw_positions();
    // Total size = max_frames * max_bodies = 5 * 10 = 50
    EXPECT_EQ(raw_positions.size(), 50u);
}

TEST(TrajectoryRecorder, ActiveCountsTracked) {
    World world(WorldConfig{.max_bodies = 10, .max_shapes = 10, .max_contacts = 10});

    ShapeHandle shape = world.create_shape(CircleShape{.radius = 0.5f});
    world.create_body({.position = {1.0f, 2.0f}, .shape = shape, .mass = 1.0f});

    TrajectoryRecorder recorder(world, 10);
    recorder.start();
    recorder.capture();

    EXPECT_EQ(recorder.get_active_count(0), 1u);
}

// ── Capacity limit ────────────────────────────────────────────────

TEST(TrajectoryRecorder, StopsAtMaxFrames) {
    World world(WorldConfig{.max_bodies = 10, .max_shapes = 10, .max_contacts = 10});

    ShapeHandle shape = world.create_shape(CircleShape{.radius = 0.5f});
    world.create_body({.position = {0.0f, 0.0f}, .shape = shape, .mass = 1.0f});

    TrajectoryRecorder recorder(world, 3);
    recorder.start();

    recorder.capture();
    recorder.capture();
    recorder.capture();
    recorder.capture();  // Should be ignored (at capacity)

    EXPECT_EQ(recorder.current_frame(), 3u);
}

// ── Reset ─────────────────────────────────────────────────────────

TEST(TrajectoryRecorder, ResetClearsFrames) {
    World world(WorldConfig{.max_bodies = 10, .max_shapes = 10, .max_contacts = 10});

    ShapeHandle shape = world.create_shape(CircleShape{.radius = 0.5f});
    world.create_body({.position = {0.0f, 0.0f}, .shape = shape, .mass = 1.0f});

    TrajectoryRecorder recorder(world, 10);
    recorder.start();
    recorder.capture();
    recorder.capture();

    EXPECT_EQ(recorder.current_frame(), 2u);

    recorder.start();  // Reset
    EXPECT_EQ(recorder.current_frame(), 0u);
}
