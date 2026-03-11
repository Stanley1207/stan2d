#include <gtest/gtest.h>
#include <stan2d/world/world.hpp>
#include <stan2d/world/state_view.hpp>
#include <stan2d/world/snapshot.hpp>

using namespace stan2d;

// ── WorldStateView ────────────────────────────────────────────────

TEST(StateView, ViewReflectsCurrentState) {
    World world(WorldConfig{.max_bodies = 100, .max_shapes = 100, .max_contacts = 100});

    ShapeHandle shape = world.create_shape(CircleShape{.radius = 0.5f});
    world.create_body({.position = {1.0f, 2.0f}, .shape = shape, .mass = 1.0f});
    world.create_body({.position = {3.0f, 4.0f}, .shape = shape, .mass = 2.0f});

    WorldStateView view = world.get_state_view();

    EXPECT_EQ(view.active_body_count, 2u);
    EXPECT_EQ(view.positions.size(), 2u);
    EXPECT_EQ(view.velocities.size(), 2u);
    EXPECT_EQ(view.rotations.size(), 2u);
    EXPECT_EQ(view.masses.size(), 2u);
}

TEST(StateView, ViewDataMatchesBodies) {
    World world(WorldConfig{.max_bodies = 100, .max_shapes = 100, .max_contacts = 100});

    ShapeHandle shape = world.create_shape(CircleShape{.radius = 0.5f});
    BodyHandle a = world.create_body({
        .position = {1.0f, 2.0f},
        .velocity = {3.0f, 4.0f},
        .shape    = shape,
        .mass     = 5.0f
    });

    WorldStateView view = world.get_state_view();

    EXPECT_FLOAT_EQ(view.positions[0].x, 1.0f);
    EXPECT_FLOAT_EQ(view.positions[0].y, 2.0f);
    EXPECT_FLOAT_EQ(view.velocities[0].x, 3.0f);
    EXPECT_FLOAT_EQ(view.velocities[0].y, 4.0f);
    EXPECT_FLOAT_EQ(view.masses[0], 5.0f);
}

TEST(StateView, EmptyWorldView) {
    World world(WorldConfig{.max_bodies = 100, .max_shapes = 100, .max_contacts = 100});

    WorldStateView view = world.get_state_view();

    EXPECT_EQ(view.active_body_count, 0u);
    EXPECT_TRUE(view.positions.empty());
}

// ── WorldSnapshot: save and restore ───────────────────────────────

TEST(Snapshot, SaveAndRestorePositions) {
    World world(WorldConfig{.max_bodies = 100, .max_shapes = 100, .max_contacts = 100});
    world.set_gravity({0.0f, -10.0f});

    ShapeHandle shape = world.create_shape(CircleShape{.radius = 0.5f});
    BodyHandle ball = world.create_body({
        .position = {0.0f, 10.0f},
        .shape    = shape,
        .mass     = 1.0f
    });

    // Save state before simulation
    WorldSnapshot snapshot;
    world.save_state(snapshot);

    // Simulate several steps
    float dt = 1.0f / 60.0f;
    for (int i = 0; i < 60; ++i) {
        world.step(dt);
    }

    // Ball should have fallen
    EXPECT_LT(world.get_position(ball).y, 10.0f);

    // Restore state
    world.restore_state(snapshot);

    // Ball should be back at original position
    Vec2 pos = world.get_position(ball);
    EXPECT_FLOAT_EQ(pos.x, 0.0f);
    EXPECT_FLOAT_EQ(pos.y, 10.0f);
}

TEST(Snapshot, RestorePreservesHandleValidity) {
    World world(WorldConfig{.max_bodies = 100, .max_shapes = 100, .max_contacts = 100});

    ShapeHandle shape = world.create_shape(CircleShape{.radius = 0.5f});
    BodyHandle a = world.create_body({.position = {1.0f, 0.0f}, .shape = shape, .mass = 1.0f});
    BodyHandle b = world.create_body({.position = {2.0f, 0.0f}, .shape = shape, .mass = 1.0f});

    WorldSnapshot snapshot;
    world.save_state(snapshot);

    // Destroy body b and create c
    world.destroy_body(b);
    BodyHandle c = world.create_body({.position = {3.0f, 0.0f}, .shape = shape, .mass = 1.0f});

    // Restore — b should be valid again, c should NOT be
    world.restore_state(snapshot);

    EXPECT_TRUE(world.is_valid(a));
    EXPECT_TRUE(world.is_valid(b));
    EXPECT_FALSE(world.is_valid(c));

    EXPECT_FLOAT_EQ(world.get_position(a).x, 1.0f);
    EXPECT_FLOAT_EQ(world.get_position(b).x, 2.0f);
}

TEST(Snapshot, DeterministicReplay) {
    auto run_simulation = [](World& world, BodyHandle ball, int steps, float dt) {
        for (int i = 0; i < steps; ++i) {
            world.step(dt);
        }
        return world.get_position(ball);
    };

    World world(WorldConfig{.max_bodies = 100, .max_shapes = 100, .max_contacts = 100});
    world.set_gravity({0.0f, -10.0f});

    ShapeHandle shape = world.create_shape(CircleShape{.radius = 0.5f});
    BodyHandle ball = world.create_body({
        .position = {0.0f, 10.0f},
        .shape    = shape,
        .mass     = 1.0f
    });

    WorldSnapshot snapshot;
    world.save_state(snapshot);

    float dt = 1.0f / 60.0f;
    Vec2 pos1 = run_simulation(world, ball, 120, dt);

    // Restore and replay
    world.restore_state(snapshot);
    Vec2 pos2 = run_simulation(world, ball, 120, dt);

    // Must be bit-exact
    EXPECT_FLOAT_EQ(pos1.x, pos2.x);
    EXPECT_FLOAT_EQ(pos1.y, pos2.y);
}

TEST(Snapshot, SnapshotPreservesVelocities) {
    World world(WorldConfig{.max_bodies = 100, .max_shapes = 100, .max_contacts = 100});

    ShapeHandle shape = world.create_shape(CircleShape{.radius = 0.5f});
    BodyHandle body = world.create_body({
        .position = {0.0f, 0.0f},
        .velocity = {5.0f, -3.0f},
        .shape    = shape,
        .mass     = 1.0f
    });

    WorldSnapshot snapshot;
    world.save_state(snapshot);

    // Modify velocity via simulation
    world.set_gravity({0.0f, -10.0f});
    world.step(1.0f / 60.0f);

    EXPECT_NE(world.get_velocity(body).y, -3.0f);

    world.restore_state(snapshot);

    Vec2 vel = world.get_velocity(body);
    EXPECT_FLOAT_EQ(vel.x, 5.0f);
    EXPECT_FLOAT_EQ(vel.y, -3.0f);
}

TEST(Snapshot, MultipleSnapshotsIndependent) {
    World world(WorldConfig{.max_bodies = 100, .max_shapes = 100, .max_contacts = 100});
    world.set_gravity({0.0f, -10.0f});

    ShapeHandle shape = world.create_shape(CircleShape{.radius = 0.5f});
    BodyHandle ball = world.create_body({
        .position = {0.0f, 10.0f},
        .shape    = shape,
        .mass     = 1.0f
    });

    // Snapshot at t=0
    WorldSnapshot snap0;
    world.save_state(snap0);

    // Simulate 30 steps
    float dt = 1.0f / 60.0f;
    for (int i = 0; i < 30; ++i) {
        world.step(dt);
    }

    // Snapshot at t=0.5s
    WorldSnapshot snap1;
    world.save_state(snap1);
    Vec2 pos_at_snap1 = world.get_position(ball);

    // Simulate more
    for (int i = 0; i < 30; ++i) {
        world.step(dt);
    }

    // Restore to t=0.5s
    world.restore_state(snap1);
    EXPECT_FLOAT_EQ(world.get_position(ball).y, pos_at_snap1.y);

    // Restore to t=0
    world.restore_state(snap0);
    EXPECT_FLOAT_EQ(world.get_position(ball).y, 10.0f);
}
