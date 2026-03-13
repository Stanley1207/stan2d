> **Note for Claude**: 牢记 `docs/tasks/00_master_plan.md` 中的四大铁律。

## Task 8: Narrow Phase (Shape-Pair Collision Detection)

**Goal:** Precise collision detection between shape pairs. Circle-Circle (distance check), Circle-Polygon (closest point), Polygon-Polygon (GJK + EPA). Returns ContactManifold with normal, penetration depth, and contact points.

**Files:**
- Create: `include/stan2d/collision/contact.hpp`
- Create: `include/stan2d/collision/narrow_phase.hpp`
- Create: `tests/unit/test_narrow_phase.cpp`

**Depends on:** Task 2, Task 4

### Step 1: Write failing tests (RED)

**File:** `tests/unit/test_narrow_phase.cpp`

```cpp
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
    // Penetration: circle edge at 1.2-0.5=0.7, box edge at 1.0 → pen = 1.0 - 0.7 = 0.3
    //   Actually: pen = radius - distance_from_edge = ...
    //   Distance from circle center to edge = |1.2 - 1.0| = 0.2, pen = 0.5 - 0.2 = 0.3
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

// ── Polygon vs Polygon (SAT / GJK+EPA) ───────────────────────────

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
```

### Step 2: Run tests to verify they fail

Run: `cmake --build build && ctest --test-dir build --output-on-failure`

Expected: FAIL — `fatal error: 'stan2d/collision/narrow_phase.hpp' file not found`

### Step 3: Implement ContactManifold

**File:** `include/stan2d/collision/contact.hpp`

```cpp
#pragma once

#include <cstdint>
#include <stan2d/core/handle.hpp>
#include <stan2d/core/math_types.hpp>

namespace stan2d {

struct ContactPoint {
    Vec2     position    = {0.0f, 0.0f};
    float    penetration = 0.0f;
    uint32_t id          = 0;       // For warm starting across frames
};

struct ContactManifold {
    BodyHandle body_a{};
    BodyHandle body_b{};
    Vec2       normal{0.0f, 0.0f};   // A → B direction
    uint32_t   point_count = 0;
    ContactPoint points[2]{};
};

} // namespace stan2d
```

### Step 4: Implement narrow phase collision functions (GREEN)

**File:** `include/stan2d/collision/narrow_phase.hpp`

```cpp
#pragma once

#include <cmath>
#include <algorithm>
#include <variant>

#include <glm/glm.hpp>
#include <stan2d/collision/contact.hpp>
#include <stan2d/core/math_types.hpp>
#include <stan2d/core/shapes.hpp>

namespace stan2d {

// ── Circle vs Circle ──────────────────────────────────────────────

inline bool collide_circle_circle(
    const CircleShape& a, Vec2 pos_a,
    const CircleShape& b, Vec2 pos_b,
    ContactManifold& out)
{
    Vec2 d = pos_b - pos_a;
    float dist_sq = glm::dot(d, d);
    float radius_sum = a.radius + b.radius;

    if (dist_sq > radius_sum * radius_sum) {
        return false;
    }

    float dist = std::sqrt(dist_sq);

    out.point_count = 1;

    if (dist > 1e-7f) {
        out.normal = d / dist;
    } else {
        out.normal = {1.0f, 0.0f}; // arbitrary for coincident centers
    }

    out.points[0].penetration = radius_sum - dist;
    out.points[0].position = pos_a + out.normal * a.radius;
    out.points[0].id = 0;

    return true;
}

// ── Circle vs Polygon ─────────────────────────────────────────────

inline Vec2 rotate_vec2(Vec2 v, float angle) {
    float c = std::cos(angle);
    float s = std::sin(angle);
    return {c * v.x - s * v.y, s * v.x + c * v.y};
}

inline bool collide_circle_polygon(
    const CircleShape& circle, Vec2 circle_pos,
    const PolygonShape& poly, Vec2 poly_pos, float poly_rot,
    ContactManifold& out)
{
    // Transform circle center into polygon local space
    Vec2 local_center = rotate_vec2(circle_pos - poly_pos, -poly_rot);

    // Find the edge with minimum separation
    float max_sep = -1e30f;
    uint32_t best_edge = 0;

    for (uint32_t i = 0; i < poly.vertex_count; ++i) {
        Vec2 n = poly.normals[i];
        float sep = glm::dot(n, local_center - poly.vertices[i]);
        if (sep > circle.radius) {
            return false; // Separating axis found
        }
        if (sep > max_sep) {
            max_sep = sep;
            best_edge = i;
        }
    }

    // Check if center is inside polygon (all separations negative)
    if (max_sep < 0.0f) {
        // Circle center inside polygon
        out.point_count = 1;
        Vec2 local_normal = poly.normals[best_edge];
        out.normal = rotate_vec2(local_normal, poly_rot);
        out.points[0].penetration = circle.radius - max_sep;
        out.points[0].position = circle_pos + out.normal * (max_sep);
        out.points[0].id = best_edge;
        return true;
    }

    // Check vertex regions
    uint32_t v1_idx = best_edge;
    uint32_t v2_idx = (best_edge + 1) % poly.vertex_count;
    Vec2 v1 = poly.vertices[v1_idx];
    Vec2 v2 = poly.vertices[v2_idx];

    Vec2 edge = v2 - v1;
    float t = glm::dot(local_center - v1, edge) / glm::dot(edge, edge);
    t = glm::clamp(t, 0.0f, 1.0f);

    Vec2 closest = v1 + edge * t;
    Vec2 diff = local_center - closest;
    float dist_sq = glm::dot(diff, diff);

    if (dist_sq > circle.radius * circle.radius) {
        return false;
    }

    float dist = std::sqrt(dist_sq);

    out.point_count = 1;
    Vec2 local_normal = (dist > 1e-7f) ? diff / dist : poly.normals[best_edge];
    out.normal = rotate_vec2(local_normal, poly_rot);
    out.points[0].penetration = circle.radius - dist;
    out.points[0].position = circle_pos - out.normal * dist;
    out.points[0].id = best_edge;

    return true;
}

// ── Polygon vs Polygon (SAT) ─────────────────────────────────────

namespace detail {

struct SATResult {
    float    min_penetration = 1e30f;
    Vec2     min_normal      = {0.0f, 0.0f};
    uint32_t min_edge        = 0;
};

// Transform polygon vertices to world space
inline void transform_vertices(
    const PolygonShape& poly, Vec2 pos, float rot,
    Vec2* out_verts, Vec2* out_normals)
{
    for (uint32_t i = 0; i < poly.vertex_count; ++i) {
        out_verts[i]   = pos + rotate_vec2(poly.vertices[i], rot);
        out_normals[i] = rotate_vec2(poly.normals[i], rot);
    }
}

// Project polygon onto axis, return [min, max]
inline void project_polygon(
    const Vec2* verts, uint32_t count, Vec2 axis,
    float& out_min, float& out_max)
{
    out_min = out_max = glm::dot(verts[0], axis);
    for (uint32_t i = 1; i < count; ++i) {
        float p = glm::dot(verts[i], axis);
        out_min = glm::min(out_min, p);
        out_max = glm::max(out_max, p);
    }
}

} // namespace detail

inline bool collide_polygon_polygon(
    const PolygonShape& a, Vec2 pos_a, float rot_a,
    const PolygonShape& b, Vec2 pos_b, float rot_b,
    ContactManifold& out)
{
    Vec2 verts_a[MAX_POLYGON_VERTICES], normals_a[MAX_POLYGON_VERTICES];
    Vec2 verts_b[MAX_POLYGON_VERTICES], normals_b[MAX_POLYGON_VERTICES];

    detail::transform_vertices(a, pos_a, rot_a, verts_a, normals_a);
    detail::transform_vertices(b, pos_b, rot_b, verts_b, normals_b);

    float min_pen = 1e30f;
    Vec2  min_normal{0.0f, 0.0f};

    // Test axes from polygon A
    for (uint32_t i = 0; i < a.vertex_count; ++i) {
        Vec2 axis = normals_a[i];
        float min_a, max_a, min_b, max_b;
        detail::project_polygon(verts_a, a.vertex_count, axis, min_a, max_a);
        detail::project_polygon(verts_b, b.vertex_count, axis, min_b, max_b);

        float overlap = glm::min(max_a - min_b, max_b - min_a);
        if (overlap < 0.0f) return false; // Separating axis

        if (overlap < min_pen) {
            min_pen = overlap;
            min_normal = axis;
        }
    }

    // Test axes from polygon B
    for (uint32_t i = 0; i < b.vertex_count; ++i) {
        Vec2 axis = normals_b[i];
        float min_a, max_a, min_b, max_b;
        detail::project_polygon(verts_a, a.vertex_count, axis, min_a, max_a);
        detail::project_polygon(verts_b, b.vertex_count, axis, min_b, max_b);

        float overlap = glm::min(max_a - min_b, max_b - min_a);
        if (overlap < 0.0f) return false;

        if (overlap < min_pen) {
            min_pen = overlap;
            min_normal = axis;
        }
    }

    // Ensure normal points from A to B
    Vec2 center_diff = pos_b - pos_a;
    if (glm::dot(min_normal, center_diff) < 0.0f) {
        min_normal = -min_normal;
    }

    out.normal = min_normal;
    out.point_count = 1;
    out.points[0].penetration = min_pen;

    // Contact point: deepest vertex of B along -normal
    float best_dot = glm::dot(verts_b[0], -min_normal);
    Vec2 best_point = verts_b[0];
    for (uint32_t i = 1; i < b.vertex_count; ++i) {
        float d = glm::dot(verts_b[i], -min_normal);
        if (d < best_dot) {
            best_dot = d;
            best_point = verts_b[i];
        }
    }
    out.points[0].position = best_point;
    out.points[0].id = 0;

    return true;
}

// ── Shape dispatch ────────────────────────────────────────────────

inline bool collide_shapes(
    const ShapeData& a, Vec2 pos_a, float rot_a,
    const ShapeData& b, Vec2 pos_b, float rot_b,
    ContactManifold& out)
{
    // Circle vs Circle
    if (auto* ca = std::get_if<CircleShape>(&a)) {
        if (auto* cb = std::get_if<CircleShape>(&b)) {
            return collide_circle_circle(*ca, pos_a, *cb, pos_b, out);
        }
        if (auto* pb = std::get_if<PolygonShape>(&b)) {
            return collide_circle_polygon(*ca, pos_a, *pb, pos_b, rot_b, out);
        }
    }

    // Polygon vs Circle (swap and flip normal)
    if (auto* pa = std::get_if<PolygonShape>(&a)) {
        if (auto* cb = std::get_if<CircleShape>(&b)) {
            bool hit = collide_circle_polygon(*cb, pos_b, *pa, pos_a, rot_a, out);
            if (hit) {
                out.normal = -out.normal; // Flip: was B→A, need A→B
            }
            return hit;
        }
        if (auto* pb = std::get_if<PolygonShape>(&b)) {
            return collide_polygon_polygon(*pa, pos_a, rot_a, *pb, pos_b, rot_b, out);
        }
    }

    // CapsuleShape collisions — placeholder for future
    return false;
}

} // namespace stan2d
```

### Step 5: Run tests to verify they pass

Run: `cmake --build build && ctest --test-dir build --output-on-failure`

Expected: PASS — all narrow phase tests green

### Step 6: Commit

```bash
git add include/stan2d/collision/contact.hpp include/stan2d/collision/narrow_phase.hpp tests/unit/test_narrow_phase.cpp
git commit -m "feat: narrow phase collision — Circle-Circle, Circle-Polygon, Polygon-Polygon (SAT)"
```

---