#include <gtest/gtest.h>
#include <cmath>
#include <stan2d/world/world.hpp>

using namespace stan2d;

// ── Gravity integration through step() ────────────────────────────

TEST(StepPipeline, SingleBallFallsUnderGravity) {
    World world(WorldConfig{.max_bodies = 100, .max_shapes = 100, .max_contacts = 100});
    world.set_gravity({0.0f, -10.0f});

    ShapeHandle shape = world.create_shape(CircleShape{.radius = 0.5f});
    BodyHandle ball = world.create_body({
        .position = {0.0f, 10.0f},
        .shape    = shape,
        .mass     = 1.0f
    });

    float dt = 1.0f / 60.0f;
    world.step(dt);

    Vec2 pos = world.get_position(ball);
    Vec2 vel = world.get_velocity(ball);

    // After one step: vel.y = 0 + (-10) * dt
    EXPECT_NEAR(vel.y, -10.0f * dt, 1e-5f);
    // pos.y = 10 + vel.y * dt (symplectic Euler)
    EXPECT_NEAR(pos.y, 10.0f + vel.y * dt, 1e-5f);
    // x unchanged
    EXPECT_FLOAT_EQ(pos.x, 0.0f);
}

TEST(StepPipeline, MultipleStepsAccumulateGravity) {
    World world(WorldConfig{.max_bodies = 100, .max_shapes = 100, .max_contacts = 100});
    world.set_gravity({0.0f, -10.0f});

    ShapeHandle shape = world.create_shape(CircleShape{.radius = 0.5f});
    BodyHandle ball = world.create_body({
        .position = {0.0f, 100.0f},
        .shape    = shape,
        .mass     = 1.0f
    });

    float dt = 1.0f / 60.0f;
    for (int i = 0; i < 60; ++i) {
        world.step(dt);
    }

    Vec2 vel = world.get_velocity(ball);
    // After 60 steps: vel.y ≈ -10.0 * 1.0 = -10.0
    EXPECT_NEAR(vel.y, -10.0f, 0.01f);

    Vec2 pos = world.get_position(ball);
    // Ball should have fallen significantly
    EXPECT_LT(pos.y, 100.0f);
}

// ── Static bodies don't move ──────────────────────────────────────

TEST(StepPipeline, StaticBodyUnaffectedByGravity) {
    World world(WorldConfig{.max_bodies = 100, .max_shapes = 100, .max_contacts = 100});
    world.set_gravity({0.0f, -10.0f});

    ShapeHandle shape = world.create_shape(CircleShape{.radius = 1.0f});
    BodyHandle floor = world.create_body({
        .position  = {0.0f, 0.0f},
        .shape     = shape,
        .body_type = BodyType::Static
    });

    float dt = 1.0f / 60.0f;
    for (int i = 0; i < 10; ++i) {
        world.step(dt);
    }

    Vec2 pos = world.get_position(floor);
    EXPECT_FLOAT_EQ(pos.x, 0.0f);
    EXPECT_FLOAT_EQ(pos.y, 0.0f);
}

// ── Kinematic bodies move at set velocity ─────────────────────────

TEST(StepPipeline, KinematicBodyMovesAtSetVelocity) {
    World world(WorldConfig{.max_bodies = 100, .max_shapes = 100, .max_contacts = 100});
    world.set_gravity({0.0f, -10.0f});

    ShapeHandle shape = world.create_shape(CircleShape{.radius = 0.5f});
    BodyHandle platform = world.create_body({
        .position  = {0.0f, 5.0f},
        .velocity  = {2.0f, 0.0f},
        .shape     = shape,
        .body_type = BodyType::Kinematic
    });

    float dt = 1.0f / 60.0f;
    world.step(dt);

    Vec2 vel = world.get_velocity(platform);
    // Kinematic: velocity not affected by gravity
    EXPECT_FLOAT_EQ(vel.x, 2.0f);
    EXPECT_FLOAT_EQ(vel.y, 0.0f);

    Vec2 pos = world.get_position(platform);
    // Position updated from velocity
    EXPECT_NEAR(pos.x, 2.0f * dt, 1e-5f);
    EXPECT_NEAR(pos.y, 5.0f, 1e-5f);
}

// ── Collision: ball onto static floor ─────────────────────────────

TEST(StepPipeline, BallCollidesWithStaticFloor) {
    World world(WorldConfig{.max_bodies = 100, .max_shapes = 100, .max_contacts = 200});
    world.set_gravity({0.0f, -10.0f});

    ShapeHandle circle = world.create_shape(CircleShape{.radius = 0.5f});

    // Ball just above floor
    BodyHandle ball = world.create_body({
        .position = {0.0f, 1.0f},
        .velocity = {0.0f, -5.0f},
        .shape    = circle,
        .mass     = 1.0f
    });

    // Wide floor polygon
    PolygonShape floor_shape;
    floor_shape.vertex_count = 4;
    floor_shape.vertices[0] = {-10.0f, -0.5f};
    floor_shape.vertices[1] = { 10.0f, -0.5f};
    floor_shape.vertices[2] = { 10.0f,  0.0f};
    floor_shape.vertices[3] = {-10.0f,  0.0f};
    floor_shape.normals[0] = { 0.0f, -1.0f};
    floor_shape.normals[1] = { 1.0f,  0.0f};
    floor_shape.normals[2] = { 0.0f,  1.0f};
    floor_shape.normals[3] = {-1.0f,  0.0f};

    BodyHandle floor = world.create_body({
        .position  = {0.0f, 0.0f},
        .shape_data = floor_shape,
        .body_type = BodyType::Static
    });

    float dt = 1.0f / 60.0f;
    for (int i = 0; i < 120; ++i) {
        world.step(dt);
    }

    Vec2 pos = world.get_position(ball);
    // Ball should not have fallen below the floor surface.
    // Iterative solvers allow some residual penetration.
    EXPECT_GE(pos.y, -0.5f);
}

// ── Two dynamic circles collide ───────────────────────────────────

TEST(StepPipeline, TwoDynamicCirclesCollide) {
    World world(WorldConfig{.max_bodies = 100, .max_shapes = 100, .max_contacts = 200});
    world.set_gravity({0.0f, 0.0f});  // No gravity

    ShapeHandle circle = world.create_shape(CircleShape{.radius = 0.5f});

    BodyHandle a = world.create_body({
        .position = {0.0f, 0.0f},
        .velocity = {5.0f, 0.0f},
        .shape    = circle,
        .mass     = 1.0f
    });

    BodyHandle b = world.create_body({
        .position = {3.0f, 0.0f},
        .velocity = {-5.0f, 0.0f},
        .shape    = circle,
        .mass     = 1.0f
    });

    float dt = 1.0f / 60.0f;
    for (int i = 0; i < 60; ++i) {
        world.step(dt);
    }

    // After collision, bodies should have separated
    Vec2 pos_a = world.get_position(a);
    Vec2 pos_b = world.get_position(b);
    float dist = std::abs(pos_b.x - pos_a.x);

    // Circles should not be heavily overlapping (sum of radii = 1.0).
    // Allow some solver slop.
    EXPECT_GE(dist, 0.8f);
}

// ── Force accumulators cleared each step ──────────────────────────

TEST(StepPipeline, ForceAccumulatorsClearedAfterStep) {
    World world(WorldConfig{.max_bodies = 100, .max_shapes = 100, .max_contacts = 100});
    world.set_gravity({0.0f, 0.0f});

    ShapeHandle shape = world.create_shape(CircleShape{.radius = 0.5f});
    BodyHandle body = world.create_body({
        .position = {0.0f, 0.0f},
        .shape    = shape,
        .mass     = 1.0f
    });

    // Apply force for one step
    world.apply_force(body, {100.0f, 0.0f});
    world.step(1.0f / 60.0f);

    Vec2 vel_after_first = world.get_velocity(body);
    EXPECT_GT(vel_after_first.x, 0.0f);

    // Second step without force — velocity should NOT continue to accelerate
    world.step(1.0f / 60.0f);
    Vec2 vel_after_second = world.get_velocity(body);

    // Velocity same as after first step (no new force applied)
    EXPECT_NEAR(vel_after_second.x, vel_after_first.x, 1e-5f);
}

// ── Mixed body types in pipeline ──────────────────────────────────

TEST(StepPipeline, MixedBodyTypesPipelineRuns) {
    World world(WorldConfig{.max_bodies = 100, .max_shapes = 100, .max_contacts = 200});
    world.set_gravity({0.0f, -10.0f});

    ShapeHandle circle = world.create_shape(CircleShape{.radius = 0.5f});

    BodyHandle dynamic_body = world.create_body({
        .position = {0.0f, 10.0f},
        .shape    = circle,
        .mass     = 1.0f
    });

    BodyHandle static_body = world.create_body({
        .position  = {5.0f, 0.0f},
        .shape     = circle,
        .body_type = BodyType::Static
    });

    BodyHandle kinematic_body = world.create_body({
        .position  = {10.0f, 0.0f},
        .velocity  = {1.0f, 0.0f},
        .shape     = circle,
        .body_type = BodyType::Kinematic
    });

    float dt = 1.0f / 60.0f;
    for (int i = 0; i < 10; ++i) {
        world.step(dt);
    }

    // Dynamic fell
    EXPECT_LT(world.get_position(dynamic_body).y, 10.0f);

    // Static unchanged
    EXPECT_FLOAT_EQ(world.get_position(static_body).x, 5.0f);
    EXPECT_FLOAT_EQ(world.get_position(static_body).y, 0.0f);

    // Kinematic moved horizontally
    EXPECT_GT(world.get_position(kinematic_body).x, 10.0f);
    EXPECT_FLOAT_EQ(world.get_position(kinematic_body).y, 0.0f);
}

// ── Zero dt is a no-op ────────────────────────────────────────────

TEST(StepPipeline, ZeroDtNoOp) {
    World world(WorldConfig{.max_bodies = 100, .max_shapes = 100, .max_contacts = 100});
    world.set_gravity({0.0f, -10.0f});

    ShapeHandle shape = world.create_shape(CircleShape{.radius = 0.5f});
    BodyHandle body = world.create_body({
        .position = {1.0f, 2.0f},
        .velocity = {3.0f, 4.0f},
        .shape    = shape,
        .mass     = 1.0f
    });

    world.step(0.0f);

    Vec2 pos = world.get_position(body);
    EXPECT_FLOAT_EQ(pos.x, 1.0f);
    EXPECT_FLOAT_EQ(pos.y, 2.0f);
}
