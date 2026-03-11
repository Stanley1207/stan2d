#include <gtest/gtest.h>
#include <cmath>
#include <stan2d/collision/narrow_phase.hpp>

using namespace stan2d;

// ── Helper: make a box polygon ────────────────────────────────────

static PolygonShape make_box(float half_w, float half_h) {
    PolygonShape box;
    box.vertex_count = 4;
    box.vertices[0] = {-half_w, -half_h};
    box.vertices[1] = { half_w, -half_h};
    box.vertices[2] = { half_w,  half_h};
    box.vertices[3] = {-half_w,  half_h};
    box.normals[0] = { 0.0f, -1.0f};
    box.normals[1] = { 1.0f,  0.0f};
    box.normals[2] = { 0.0f,  1.0f};
    box.normals[3] = {-1.0f,  0.0f};
    return box;
}

// ── Circle vs Circle ──────────────────────────────────────────────

TEST(NarrowPhase, CircleCircleOverlap) {
    CircleShape a{.radius = 1.0f};
    CircleShape b{.radius = 1.0f};
    Vec2 pos_a{0.0f, 0.0f};
    Vec2 pos_b{1.5f, 0.0f};

    ContactManifold m;
    bool hit = collide_circle_circle(a, pos_a, b, pos_b, m);

    EXPECT_TRUE(hit);
    EXPECT_EQ(m.point_count, 1u);
    EXPECT_NEAR(m.normal.x, 1.0f, 1e-5f);  // A→B direction
    EXPECT_NEAR(m.normal.y, 0.0f, 1e-5f);
    EXPECT_NEAR(m.points[0].penetration, 0.5f, 1e-5f); // 1+1 - 1.5
}

TEST(NarrowPhase, CircleCircleNoOverlap) {
    CircleShape a{.radius = 1.0f};
    CircleShape b{.radius = 1.0f};
    Vec2 pos_a{0.0f, 0.0f};
    Vec2 pos_b{3.0f, 0.0f};

    ContactManifold m;
    bool hit = collide_circle_circle(a, pos_a, b, pos_b, m);

    EXPECT_FALSE(hit);
}

TEST(NarrowPhase, CircleCircleTouching) {
    CircleShape a{.radius = 1.0f};
    CircleShape b{.radius = 1.0f};
    Vec2 pos_a{0.0f, 0.0f};
    Vec2 pos_b{2.0f, 0.0f};

    ContactManifold m;
    bool hit = collide_circle_circle(a, pos_a, b, pos_b, m);

    // Touching: penetration = 0, still counts as collision
    EXPECT_TRUE(hit);
    EXPECT_NEAR(m.points[0].penetration, 0.0f, 1e-5f);
}

TEST(NarrowPhase, CircleCircleDiagonal) {
    CircleShape a{.radius = 1.0f};
    CircleShape b{.radius = 1.0f};
    Vec2 pos_a{0.0f, 0.0f};
    Vec2 pos_b{1.0f, 1.0f};

    ContactManifold m;
    bool hit = collide_circle_circle(a, pos_a, b, pos_b, m);

    EXPECT_TRUE(hit);
    float dist = std::sqrt(2.0f);
    EXPECT_NEAR(m.points[0].penetration, 2.0f - dist, 1e-4f);
    // Normal should point from A to B
    EXPECT_NEAR(m.normal.x, 1.0f / dist, 1e-4f);
    EXPECT_NEAR(m.normal.y, 1.0f / dist, 1e-4f);
}

// ── Circle vs Polygon ─────────────────────────────────────────────

TEST(NarrowPhase, CirclePolygonOverlap) {
    CircleShape circle{.radius = 0.5f};
    PolygonShape box = make_box(1.0f, 1.0f);

    Vec2 circle_pos{1.2f, 0.0f};  // Circle center near right edge of box
    Vec2 box_pos{0.0f, 0.0f};

    ContactManifold m;
    bool hit = collide_circle_polygon(circle, circle_pos, box, box_pos, 0.0f, m);

    EXPECT_TRUE(hit);
    EXPECT_EQ(m.point_count, 1u);
    // Normal should push circle out to the right
    EXPECT_NEAR(m.normal.x, 1.0f, 1e-4f);
    EXPECT_NEAR(m.normal.y, 0.0f, 1e-4f);
    // Penetration: circle edge at 1.2-0.5=0.7, box edge at 1.0 → pen = 0.5 - 0.2 = 0.3
    EXPECT_NEAR(m.points[0].penetration, 0.3f, 1e-4f);
}

TEST(NarrowPhase, CirclePolygonNoOverlap) {
    CircleShape circle{.radius = 0.5f};
    PolygonShape box = make_box(1.0f, 1.0f);

    Vec2 circle_pos{2.0f, 0.0f};
    Vec2 box_pos{0.0f, 0.0f};

    ContactManifold m;
    bool hit = collide_circle_polygon(circle, circle_pos, box, box_pos, 0.0f, m);

    EXPECT_FALSE(hit);
}

TEST(NarrowPhase, CircleInsidePolygon) {
    CircleShape circle{.radius = 0.3f};
    PolygonShape box = make_box(2.0f, 2.0f);

    Vec2 circle_pos{0.0f, 0.0f};  // Circle fully inside
    Vec2 box_pos{0.0f, 0.0f};

    ContactManifold m;
    bool hit = collide_circle_polygon(circle, circle_pos, box, box_pos, 0.0f, m);

    EXPECT_TRUE(hit);
    EXPECT_EQ(m.point_count, 1u);
    EXPECT_GT(m.points[0].penetration, 0.0f);
}

// ── Polygon vs Polygon (SAT) ─────────────────────────────────────

TEST(NarrowPhase, PolygonPolygonOverlap) {
    PolygonShape a = make_box(1.0f, 1.0f);
    PolygonShape b = make_box(1.0f, 1.0f);

    Vec2 pos_a{0.0f, 0.0f};
    Vec2 pos_b{1.5f, 0.0f};

    ContactManifold m;
    bool hit = collide_polygon_polygon(a, pos_a, 0.0f, b, pos_b, 0.0f, m);

    EXPECT_TRUE(hit);
    EXPECT_GT(m.point_count, 0u);
    // Minimum penetration axis should be x (0.5 overlap)
    EXPECT_NEAR(std::abs(m.normal.x), 1.0f, 1e-4f);
    EXPECT_NEAR(m.points[0].penetration, 0.5f, 1e-4f);
}

TEST(NarrowPhase, PolygonPolygonNoOverlap) {
    PolygonShape a = make_box(1.0f, 1.0f);
    PolygonShape b = make_box(1.0f, 1.0f);

    Vec2 pos_a{0.0f, 0.0f};
    Vec2 pos_b{3.0f, 0.0f};

    ContactManifold m;
    bool hit = collide_polygon_polygon(a, pos_a, 0.0f, b, pos_b, 0.0f, m);

    EXPECT_FALSE(hit);
}

TEST(NarrowPhase, PolygonPolygonVerticalOverlap) {
    PolygonShape a = make_box(1.0f, 1.0f);
    PolygonShape b = make_box(1.0f, 1.0f);

    Vec2 pos_a{0.0f, 0.0f};
    Vec2 pos_b{0.0f, 1.2f};

    ContactManifold m;
    bool hit = collide_polygon_polygon(a, pos_a, 0.0f, b, pos_b, 0.0f, m);

    EXPECT_TRUE(hit);
    EXPECT_NEAR(std::abs(m.normal.y), 1.0f, 1e-4f);
    EXPECT_NEAR(m.points[0].penetration, 0.8f, 1e-4f);
}

// ── Dispatch function ─────────────────────────────────────────────

TEST(NarrowPhase, DispatchCircleCircle) {
    ShapeData sa = CircleShape{.radius = 1.0f};
    ShapeData sb = CircleShape{.radius = 1.0f};

    ContactManifold m;
    bool hit = collide_shapes(sa, {0.0f, 0.0f}, 0.0f, sb, {1.5f, 0.0f}, 0.0f, m);

    EXPECT_TRUE(hit);
}

TEST(NarrowPhase, DispatchCirclePolygon) {
    ShapeData sa = CircleShape{.radius = 0.5f};
    ShapeData sb = make_box(1.0f, 1.0f);

    ContactManifold m;
    bool hit = collide_shapes(sa, {0.8f, 0.0f}, 0.0f, sb, {0.0f, 0.0f}, 0.0f, m);

    EXPECT_TRUE(hit);
}

TEST(NarrowPhase, DispatchPolygonPolygon) {
    ShapeData sa = ShapeData{make_box(1.0f, 1.0f)};
    ShapeData sb = ShapeData{make_box(1.0f, 1.0f)};

    ContactManifold m;
    bool hit = collide_shapes(sa, {0.0f, 0.0f}, 0.0f, sb, {1.5f, 0.0f}, 0.0f, m);

    EXPECT_TRUE(hit);
}
