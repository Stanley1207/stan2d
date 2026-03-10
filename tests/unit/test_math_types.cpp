#include <gtest/gtest.h>
#include <cmath>
#include <stan2d/core/math_types.hpp>

using namespace stan2d;

// ── Vec2 alias ────────────────────────────────────────────────────

TEST(MathTypes, Vec2DefaultConstruction) {
    Vec2 v{};
    EXPECT_FLOAT_EQ(v.x, 0.0f);
    EXPECT_FLOAT_EQ(v.y, 0.0f);
}

TEST(MathTypes, Vec2Arithmetic) {
    Vec2 a{1.0f, 2.0f};
    Vec2 b{3.0f, 4.0f};
    Vec2 sum = a + b;
    EXPECT_FLOAT_EQ(sum.x, 4.0f);
    EXPECT_FLOAT_EQ(sum.y, 6.0f);
}

TEST(MathTypes, Vec2ScalarMultiply) {
    Vec2 v{2.0f, 3.0f};
    Vec2 result = v * 2.0f;
    EXPECT_FLOAT_EQ(result.x, 4.0f);
    EXPECT_FLOAT_EQ(result.y, 6.0f);
}

TEST(MathTypes, Vec2Subtraction) {
    Vec2 a{5.0f, 7.0f};
    Vec2 b{2.0f, 3.0f};
    Vec2 diff = a - b;
    EXPECT_FLOAT_EQ(diff.x, 3.0f);
    EXPECT_FLOAT_EQ(diff.y, 4.0f);
}

TEST(MathTypes, Vec2Negation) {
    Vec2 v{3.0f, -4.0f};
    Vec2 neg = -v;
    EXPECT_FLOAT_EQ(neg.x, -3.0f);
    EXPECT_FLOAT_EQ(neg.y, 4.0f);
}

TEST(MathTypes, Vec2ScalarDivision) {
    Vec2 v{6.0f, 8.0f};
    Vec2 result = v / 2.0f;
    EXPECT_FLOAT_EQ(result.x, 3.0f);
    EXPECT_FLOAT_EQ(result.y, 4.0f);
}

TEST(MathTypes, Vec2Dot) {
    Vec2 a{1.0f, 0.0f};
    Vec2 b{0.0f, 1.0f};
    EXPECT_FLOAT_EQ(glm::dot(a, b), 0.0f);

    Vec2 c{3.0f, 4.0f};
    EXPECT_FLOAT_EQ(glm::dot(c, c), 25.0f);
}

TEST(MathTypes, Vec2DotAntiParallel) {
    Vec2 a{1.0f, 0.0f};
    Vec2 b{-1.0f, 0.0f};
    EXPECT_FLOAT_EQ(glm::dot(a, b), -1.0f);
}

TEST(MathTypes, Vec2Length) {
    Vec2 v{3.0f, 4.0f};
    EXPECT_FLOAT_EQ(glm::length(v), 5.0f);

    Vec2 unit_x{1.0f, 0.0f};
    EXPECT_FLOAT_EQ(glm::length(unit_x), 1.0f);

    Vec2 zero{0.0f, 0.0f};
    EXPECT_FLOAT_EQ(glm::length(zero), 0.0f);
}

TEST(MathTypes, Vec2LengthSquared) {
    // length squared avoids sqrt — used heavily in physics for comparisons
    Vec2 v{3.0f, 4.0f};
    float len_sq = glm::dot(v, v);
    EXPECT_FLOAT_EQ(len_sq, 25.0f);
}

TEST(MathTypes, Vec2Normalize) {
    Vec2 v{3.0f, 4.0f};
    Vec2 n = glm::normalize(v);
    EXPECT_FLOAT_EQ(n.x, 3.0f / 5.0f);
    EXPECT_FLOAT_EQ(n.y, 4.0f / 5.0f);
    EXPECT_NEAR(glm::length(n), 1.0f, 1e-6f);
}

TEST(MathTypes, Vec2NormalizeCardinalDirections) {
    Vec2 up = glm::normalize(Vec2{0.0f, 1.0f});
    EXPECT_FLOAT_EQ(up.x, 0.0f);
    EXPECT_FLOAT_EQ(up.y, 1.0f);

    Vec2 right = glm::normalize(Vec2{1.0f, 0.0f});
    EXPECT_FLOAT_EQ(right.x, 1.0f);
    EXPECT_FLOAT_EQ(right.y, 0.0f);
}

TEST(MathTypes, Vec2Cross2D) {
    // 2D cross product: a.x*b.y - a.y*b.x (returns scalar, used for torque)
    Vec2 a{1.0f, 0.0f};
    Vec2 b{0.0f, 1.0f};
    float cross = a.x * b.y - a.y * b.x;
    EXPECT_FLOAT_EQ(cross, 1.0f);

    // Reversed order gives negative (anti-clockwise vs clockwise)
    float cross_rev = b.x * a.y - b.y * a.x;
    EXPECT_FLOAT_EQ(cross_rev, -1.0f);

    // Parallel vectors have zero cross product
    Vec2 c{2.0f, 0.0f};
    float cross_parallel = a.x * c.y - a.y * c.x;
    EXPECT_FLOAT_EQ(cross_parallel, 0.0f);
}

TEST(MathTypes, Vec2Perpendicular) {
    // Perpendicular vector: (-y, x) — used for collision normals
    Vec2 v{3.0f, 4.0f};
    Vec2 perp{-v.y, v.x};
    EXPECT_FLOAT_EQ(glm::dot(v, perp), 0.0f);  // must be orthogonal
}

// ── Mat2 ──────────────────────────────────────────────────────────

TEST(MathTypes, Mat2Identity) {
    Mat2 I{1.0f};
    Vec2 v{3.0f, 4.0f};
    Vec2 result = I * v;
    EXPECT_FLOAT_EQ(result.x, 3.0f);
    EXPECT_FLOAT_EQ(result.y, 4.0f);
}

TEST(MathTypes, Mat2Rotation90) {
    // 90 degree rotation: cos=0, sin=1
    float angle = glm::half_pi<float>();
    float c = std::cos(angle);
    float s = std::sin(angle);
    // glm is column-major: Mat2(col0, col1)
    Mat2 rot{Vec2{c, s}, Vec2{-s, c}};

    Vec2 v{1.0f, 0.0f};
    Vec2 rotated = rot * v;
    EXPECT_NEAR(rotated.x, 0.0f, 1e-6f);
    EXPECT_NEAR(rotated.y, 1.0f, 1e-6f);
}

TEST(MathTypes, Mat2Rotation180) {
    float angle = glm::pi<float>();
    float c = std::cos(angle);
    float s = std::sin(angle);
    Mat2 rot{Vec2{c, s}, Vec2{-s, c}};

    Vec2 v{1.0f, 0.0f};
    Vec2 rotated = rot * v;
    EXPECT_NEAR(rotated.x, -1.0f, 1e-6f);
    EXPECT_NEAR(rotated.y, 0.0f, 1e-6f);
}

TEST(MathTypes, Mat2RotationPreservesLength) {
    float angle = 0.7853f;  // ~45 degrees
    float c = std::cos(angle);
    float s = std::sin(angle);
    Mat2 rot{Vec2{c, s}, Vec2{-s, c}};

    Vec2 v{3.0f, 4.0f};
    Vec2 rotated = rot * v;
    EXPECT_NEAR(glm::length(rotated), glm::length(v), 1e-5f);
}

TEST(MathTypes, Mat2MultiplyAssociative) {
    // (A * B) * v == A * (B * v)
    float a1 = 0.5f, a2 = 1.2f;
    Mat2 A{Vec2{std::cos(a1), std::sin(a1)}, Vec2{-std::sin(a1), std::cos(a1)}};
    Mat2 B{Vec2{std::cos(a2), std::sin(a2)}, Vec2{-std::sin(a2), std::cos(a2)}};

    Vec2 v{2.0f, 3.0f};
    Vec2 r1 = (A * B) * v;
    Vec2 r2 = A * (B * v);
    EXPECT_NEAR(r1.x, r2.x, 1e-5f);
    EXPECT_NEAR(r1.y, r2.y, 1e-5f);
}

// ── AABB (additional) ────────────────────────────────────────────

TEST(AABB, ContainsAllCorners) {
    AABB box{{-5.0f, -5.0f}, {5.0f, 5.0f}};
    EXPECT_TRUE(aabb_contains(box, {-5.0f, -5.0f}));  // min corner
    EXPECT_TRUE(aabb_contains(box, {5.0f, 5.0f}));    // max corner
    EXPECT_TRUE(aabb_contains(box, {-5.0f, 5.0f}));   // top-left
    EXPECT_TRUE(aabb_contains(box, {5.0f, -5.0f}));   // bottom-right
}

// ── AABB ──────────────────────────────────────────────────────────

TEST(AABB, ContainsPoint) {
    AABB box{{0.0f, 0.0f}, {10.0f, 10.0f}};
    EXPECT_TRUE(aabb_contains(box, {5.0f, 5.0f}));
    EXPECT_TRUE(aabb_contains(box, {0.0f, 0.0f}));   // boundary inclusive
    EXPECT_FALSE(aabb_contains(box, {-1.0f, 5.0f}));
    EXPECT_FALSE(aabb_contains(box, {5.0f, 11.0f}));
}

TEST(AABB, OverlapsTrue) {
    AABB a{{0.0f, 0.0f}, {5.0f, 5.0f}};
    AABB b{{3.0f, 3.0f}, {8.0f, 8.0f}};
    EXPECT_TRUE(aabb_overlaps(a, b));
    EXPECT_TRUE(aabb_overlaps(b, a));  // commutative
}

TEST(AABB, OverlapsFalse) {
    AABB a{{0.0f, 0.0f}, {2.0f, 2.0f}};
    AABB b{{5.0f, 5.0f}, {8.0f, 8.0f}};
    EXPECT_FALSE(aabb_overlaps(a, b));
}

TEST(AABB, OverlapsTouching) {
    AABB a{{0.0f, 0.0f}, {5.0f, 5.0f}};
    AABB b{{5.0f, 0.0f}, {10.0f, 5.0f}};
    // Touching edges: overlaps should return true (<=)
    EXPECT_TRUE(aabb_overlaps(a, b));
}

TEST(AABB, Merge) {
    AABB a{{1.0f, 2.0f}, {5.0f, 6.0f}};
    AABB b{{3.0f, 0.0f}, {8.0f, 4.0f}};
    AABB merged = aabb_merge(a, b);
    EXPECT_FLOAT_EQ(merged.min.x, 1.0f);
    EXPECT_FLOAT_EQ(merged.min.y, 0.0f);
    EXPECT_FLOAT_EQ(merged.max.x, 8.0f);
    EXPECT_FLOAT_EQ(merged.max.y, 6.0f);
}

TEST(AABB, SurfaceArea) {
    // "Surface area" in 2D = perimeter
    AABB box{{0.0f, 0.0f}, {3.0f, 4.0f}};
    EXPECT_FLOAT_EQ(aabb_perimeter(box), 14.0f);  // 2*(3+4)
}

TEST(AABB, ExpandByMargin) {
    AABB box{{1.0f, 1.0f}, {5.0f, 5.0f}};
    AABB expanded = aabb_expand(box, 0.5f);
    EXPECT_FLOAT_EQ(expanded.min.x, 0.5f);
    EXPECT_FLOAT_EQ(expanded.min.y, 0.5f);
    EXPECT_FLOAT_EQ(expanded.max.x, 5.5f);
    EXPECT_FLOAT_EQ(expanded.max.y, 5.5f);
}

TEST(AABB, NegativeCoordinates) {
    AABB a{{-10.0f, -10.0f}, {-5.0f, -5.0f}};
    AABB b{{-7.0f, -7.0f}, {-3.0f, -3.0f}};
    EXPECT_TRUE(aabb_overlaps(a, b));

    AABB merged = aabb_merge(a, b);
    EXPECT_FLOAT_EQ(merged.min.x, -10.0f);
    EXPECT_FLOAT_EQ(merged.min.y, -10.0f);
    EXPECT_FLOAT_EQ(merged.max.x, -3.0f);
    EXPECT_FLOAT_EQ(merged.max.y, -3.0f);
}

TEST(AABB, ZeroSizeBox) {
    AABB point{{3.0f, 4.0f}, {3.0f, 4.0f}};
    EXPECT_FLOAT_EQ(aabb_perimeter(point), 0.0f);
    EXPECT_TRUE(aabb_contains(point, {3.0f, 4.0f}));
    EXPECT_FALSE(aabb_contains(point, {3.1f, 4.0f}));
}

TEST(AABB, MergeWithSelf) {
    AABB box{{1.0f, 2.0f}, {5.0f, 6.0f}};
    AABB merged = aabb_merge(box, box);
    EXPECT_FLOAT_EQ(merged.min.x, box.min.x);
    EXPECT_FLOAT_EQ(merged.min.y, box.min.y);
    EXPECT_FLOAT_EQ(merged.max.x, box.max.x);
    EXPECT_FLOAT_EQ(merged.max.y, box.max.y);
}

TEST(AABB, ExpandPreservesCenter) {
    AABB box{{2.0f, 2.0f}, {8.0f, 6.0f}};
    AABB expanded = aabb_expand(box, 1.0f);
    Vec2 center_before = (box.min + box.max) * 0.5f;
    Vec2 center_after = (expanded.min + expanded.max) * 0.5f;
    EXPECT_FLOAT_EQ(center_before.x, center_after.x);
    EXPECT_FLOAT_EQ(center_before.y, center_after.y);
}

TEST(AABB, OverlapsContainedBox) {
    AABB outer{{0.0f, 0.0f}, {10.0f, 10.0f}};
    AABB inner{{2.0f, 2.0f}, {5.0f, 5.0f}};
    EXPECT_TRUE(aabb_overlaps(outer, inner));
    EXPECT_TRUE(aabb_overlaps(inner, outer));
}
