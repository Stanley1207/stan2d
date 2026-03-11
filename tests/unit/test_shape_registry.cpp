#include <gtest/gtest.h>
#include <stan2d/core/shape_registry.hpp>

using namespace stan2d;

// ── Shape creation ────────────────────────────────────────────────

TEST(ShapeRegistry, CreateCircle) {
    ShapeRegistry registry;
    registry.reserve(100);

    ShapeHandle h = registry.create(CircleShape{.radius = 0.5f});
    EXPECT_TRUE(registry.is_valid(h));
    EXPECT_EQ(registry.size(), 1u);
}

TEST(ShapeRegistry, CreatePolygon) {
    ShapeRegistry registry;
    registry.reserve(100);

    PolygonShape box;
    box.vertex_count = 4;
    box.vertices[0] = {-1.0f, -1.0f};
    box.vertices[1] = { 1.0f, -1.0f};
    box.vertices[2] = { 1.0f,  1.0f};
    box.vertices[3] = {-1.0f,  1.0f};
    box.normals[0] = { 0.0f, -1.0f};
    box.normals[1] = { 1.0f,  0.0f};
    box.normals[2] = { 0.0f,  1.0f};
    box.normals[3] = {-1.0f,  0.0f};

    ShapeHandle h = registry.create(box);
    EXPECT_TRUE(registry.is_valid(h));
}

TEST(ShapeRegistry, CreateCapsule) {
    ShapeRegistry registry;
    registry.reserve(100);

    CapsuleShape cap{
        .point_a = {0.0f, -0.5f},
        .point_b = {0.0f,  0.5f},
        .radius  = 0.2f
    };

    ShapeHandle h = registry.create(cap);
    EXPECT_TRUE(registry.is_valid(h));
}

// ── Shape retrieval ───────────────────────────────────────────────

TEST(ShapeRegistry, GetShapeData) {
    ShapeRegistry registry;
    registry.reserve(100);

    ShapeHandle h = registry.create(CircleShape{.radius = 1.5f});
    const ShapeData& data = registry.get(h);

    ASSERT_TRUE(std::holds_alternative<CircleShape>(data));
    EXPECT_FLOAT_EQ(std::get<CircleShape>(data).radius, 1.5f);
}

// ── Local AABB computation ────────────────────────────────────────

TEST(ShapeRegistry, CircleLocalAABB) {
    ShapeRegistry registry;
    registry.reserve(100);

    ShapeHandle h = registry.create(CircleShape{.radius = 2.0f});
    const AABB& aabb = registry.get_local_aabb(h);

    EXPECT_FLOAT_EQ(aabb.min.x, -2.0f);
    EXPECT_FLOAT_EQ(aabb.min.y, -2.0f);
    EXPECT_FLOAT_EQ(aabb.max.x,  2.0f);
    EXPECT_FLOAT_EQ(aabb.max.y,  2.0f);
}

TEST(ShapeRegistry, PolygonLocalAABB) {
    ShapeRegistry registry;
    registry.reserve(100);

    PolygonShape tri;
    tri.vertex_count = 3;
    tri.vertices[0] = {0.0f, 0.0f};
    tri.vertices[1] = {4.0f, 0.0f};
    tri.vertices[2] = {2.0f, 3.0f};

    ShapeHandle h = registry.create(tri);
    const AABB& aabb = registry.get_local_aabb(h);

    EXPECT_FLOAT_EQ(aabb.min.x, 0.0f);
    EXPECT_FLOAT_EQ(aabb.min.y, 0.0f);
    EXPECT_FLOAT_EQ(aabb.max.x, 4.0f);
    EXPECT_FLOAT_EQ(aabb.max.y, 3.0f);
}

TEST(ShapeRegistry, CapsuleLocalAABB) {
    ShapeRegistry registry;
    registry.reserve(100);

    CapsuleShape cap{
        .point_a = {0.0f, -1.0f},
        .point_b = {0.0f,  1.0f},
        .radius  = 0.5f
    };

    ShapeHandle h = registry.create(cap);
    const AABB& aabb = registry.get_local_aabb(h);

    EXPECT_FLOAT_EQ(aabb.min.x, -0.5f);
    EXPECT_FLOAT_EQ(aabb.min.y, -1.5f);
    EXPECT_FLOAT_EQ(aabb.max.x,  0.5f);
    EXPECT_FLOAT_EQ(aabb.max.y,  1.5f);
}

// ── Destruction ───────────────────────────────────────────────────

TEST(ShapeRegistry, DestroyInvalidatesHandle) {
    ShapeRegistry registry;
    registry.reserve(100);

    ShapeHandle h = registry.create(CircleShape{.radius = 1.0f});
    registry.destroy(h);

    EXPECT_FALSE(registry.is_valid(h));
    EXPECT_EQ(registry.size(), 0u);
}

TEST(ShapeRegistry, DestroyMiddleSwapsLast) {
    ShapeRegistry registry;
    registry.reserve(100);

    ShapeHandle a = registry.create(CircleShape{.radius = 1.0f});
    ShapeHandle b = registry.create(CircleShape{.radius = 2.0f});
    ShapeHandle c = registry.create(CircleShape{.radius = 3.0f});

    registry.destroy(b);

    EXPECT_TRUE(registry.is_valid(a));
    EXPECT_FALSE(registry.is_valid(b));
    EXPECT_TRUE(registry.is_valid(c));
    EXPECT_EQ(registry.size(), 2u);

    // c's data should still be accessible and correct
    const auto& data = std::get<CircleShape>(registry.get(c));
    EXPECT_FLOAT_EQ(data.radius, 3.0f);
}

// ── Shared shapes ─────────────────────────────────────────────────

TEST(ShapeRegistry, MultipleHandlesShareShape) {
    ShapeRegistry registry;
    registry.reserve(100);

    ShapeHandle shared = registry.create(CircleShape{.radius = 0.5f});

    // Multiple bodies can hold the same ShapeHandle — no copy needed
    ShapeHandle copy1 = shared;
    ShapeHandle copy2 = shared;

    EXPECT_EQ(copy1, copy2);
    EXPECT_TRUE(registry.is_valid(copy1));

    const auto& d1 = std::get<CircleShape>(registry.get(copy1));
    const auto& d2 = std::get<CircleShape>(registry.get(copy2));
    EXPECT_FLOAT_EQ(d1.radius, d2.radius);
}
