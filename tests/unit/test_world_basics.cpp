#include <gtest/gtest.h>
#include <stan2d/world/world.hpp>

using namespace stan2d;

// ── World construction ────────────────────────────────────────────

TEST(World, ConstructWithConfig) {
    World world(WorldConfig{.max_bodies = 100, .max_shapes = 100});
    EXPECT_EQ(world.body_count(), 0u);
}

// ── Body creation (two-step with shared shape) ────────────────────

TEST(World, CreateBodyTwoStep) {
    World world(WorldConfig{.max_bodies = 100, .max_shapes = 100});

    ShapeHandle shape = world.create_shape(CircleShape{.radius = 0.5f});
    BodyHandle body = world.create_body({
        .position = {1.0f, 2.0f},
        .shape    = shape,
        .mass     = 1.0f
    });

    EXPECT_TRUE(world.is_valid(body));
    EXPECT_EQ(world.body_count(), 1u);
}

TEST(World, CreateBodyPosition) {
    World world(WorldConfig{.max_bodies = 100, .max_shapes = 100});

    ShapeHandle shape = world.create_shape(CircleShape{.radius = 1.0f});
    BodyHandle body = world.create_body({
        .position = {3.0f, 7.0f},
        .shape    = shape,
        .mass     = 2.0f
    });

    Vec2 pos = world.get_position(body);
    EXPECT_FLOAT_EQ(pos.x, 3.0f);
    EXPECT_FLOAT_EQ(pos.y, 7.0f);
}

TEST(World, CreateBodyVelocity) {
    World world(WorldConfig{.max_bodies = 100, .max_shapes = 100});

    ShapeHandle shape = world.create_shape(CircleShape{.radius = 1.0f});
    BodyHandle body = world.create_body({
        .position = {0.0f, 0.0f},
        .velocity = {5.0f, -3.0f},
        .shape    = shape,
        .mass     = 1.0f
    });

    Vec2 vel = world.get_velocity(body);
    EXPECT_FLOAT_EQ(vel.x, 5.0f);
    EXPECT_FLOAT_EQ(vel.y, -3.0f);
}

// ── Body creation (one-step with inline shape) ────────────────────

TEST(World, CreateBodyOneStep) {
    World world(WorldConfig{.max_bodies = 100, .max_shapes = 100});

    BodyHandle body = world.create_body({
        .position   = {0.0f, 10.0f},
        .shape_data = CircleShape{.radius = 0.3f},
        .mass       = 2.0f
    });

    EXPECT_TRUE(world.is_valid(body));
    Vec2 pos = world.get_position(body);
    EXPECT_FLOAT_EQ(pos.x, 0.0f);
    EXPECT_FLOAT_EQ(pos.y, 10.0f);
}

// ── Mass properties ───────────────────────────────────────────────

TEST(World, DynamicBodyMassProperties) {
    World world(WorldConfig{.max_bodies = 100, .max_shapes = 100});

    ShapeHandle shape = world.create_shape(CircleShape{.radius = 1.0f});
    BodyHandle body = world.create_body({
        .position = {0.0f, 0.0f},
        .shape    = shape,
        .mass     = 4.0f
    });

    EXPECT_FLOAT_EQ(world.get_mass(body), 4.0f);
    EXPECT_FLOAT_EQ(world.get_inverse_mass(body), 0.25f);
}

TEST(World, StaticBodyHasZeroInverseMass) {
    World world(WorldConfig{.max_bodies = 100, .max_shapes = 100});

    ShapeHandle shape = world.create_shape(CircleShape{.radius = 1.0f});
    BodyHandle body = world.create_body({
        .position  = {0.0f, 0.0f},
        .shape     = shape,
        .body_type = BodyType::Static
    });

    EXPECT_FLOAT_EQ(world.get_inverse_mass(body), 0.0f);
    EXPECT_FLOAT_EQ(world.get_inverse_inertia(body), 0.0f);
}

TEST(World, KinematicBodyHasZeroInverseMass) {
    World world(WorldConfig{.max_bodies = 100, .max_shapes = 100});

    ShapeHandle shape = world.create_shape(CircleShape{.radius = 1.0f});
    BodyHandle body = world.create_body({
        .position  = {0.0f, 0.0f},
        .velocity  = {1.0f, 0.0f},
        .shape     = shape,
        .body_type = BodyType::Kinematic
    });

    EXPECT_FLOAT_EQ(world.get_inverse_mass(body), 0.0f);
}

// ── Body destruction ──────────────────────────────────────────────

TEST(World, DestroyBody) {
    World world(WorldConfig{.max_bodies = 100, .max_shapes = 100});

    ShapeHandle shape = world.create_shape(CircleShape{.radius = 1.0f});
    BodyHandle body = world.create_body({
        .position = {0.0f, 0.0f},
        .shape    = shape,
        .mass     = 1.0f
    });

    world.destroy_body(body);
    EXPECT_FALSE(world.is_valid(body));
    EXPECT_EQ(world.body_count(), 0u);
}

TEST(World, DestroyMiddleBodyMaintainsOthers) {
    World world(WorldConfig{.max_bodies = 100, .max_shapes = 100});

    ShapeHandle shape = world.create_shape(CircleShape{.radius = 1.0f});
    BodyHandle a = world.create_body({.position = {1.0f, 0.0f}, .shape = shape, .mass = 1.0f});
    BodyHandle b = world.create_body({.position = {2.0f, 0.0f}, .shape = shape, .mass = 1.0f});
    BodyHandle c = world.create_body({.position = {3.0f, 0.0f}, .shape = shape, .mass = 1.0f});

    world.destroy_body(b);

    EXPECT_TRUE(world.is_valid(a));
    EXPECT_FALSE(world.is_valid(b));
    EXPECT_TRUE(world.is_valid(c));

    EXPECT_FLOAT_EQ(world.get_position(a).x, 1.0f);
    EXPECT_FLOAT_EQ(world.get_position(c).x, 3.0f);
    EXPECT_EQ(world.body_count(), 2u);
}

// ── Gravity ───────────────────────────────────────────────────────

TEST(World, SetAndGetGravity) {
    World world(WorldConfig{.max_bodies = 100, .max_shapes = 100});

    world.set_gravity({0.0f, -9.81f});
    Vec2 g = world.get_gravity();
    EXPECT_FLOAT_EQ(g.x, 0.0f);
    EXPECT_FLOAT_EQ(g.y, -9.81f);
}

// ── Stale handle ──────────────────────────────────────────────────

TEST(World, StaleHandleIsInvalid) {
    World world(WorldConfig{.max_bodies = 100, .max_shapes = 100});

    ShapeHandle shape = world.create_shape(CircleShape{.radius = 1.0f});
    BodyHandle body = world.create_body({.position = {0.0f, 0.0f}, .shape = shape, .mass = 1.0f});
    world.destroy_body(body);

    // Create new body that reuses the slot
    BodyHandle new_body = world.create_body({.position = {5.0f, 5.0f}, .shape = shape, .mass = 1.0f});

    EXPECT_FALSE(world.is_valid(body));       // stale
    EXPECT_TRUE(world.is_valid(new_body));    // current
}
