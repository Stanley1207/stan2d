> **Note for Claude**: 牢记 `docs/tasks/00_master_plan.md` 中的四大铁律。

## Task 2: Core Math Types

**Goal:** Define Vec2/Mat2 aliases over glm, AABB struct with helper functions.

**Files:**
- Create: `include/stan2d/core/math_types.hpp`
- Create: `tests/unit/test_math_types.cpp`

**Depends on:** Task 1

### Step 1: Write failing tests (RED)

**File:** `tests/unit/test_math_types.cpp`

```cpp
#include <gtest/gtest.h>
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

TEST(MathTypes, Vec2Dot) {
    Vec2 a{1.0f, 0.0f};
    Vec2 b{0.0f, 1.0f};
    EXPECT_FLOAT_EQ(glm::dot(a, b), 0.0f);

    Vec2 c{3.0f, 4.0f};
    EXPECT_FLOAT_EQ(glm::dot(c, c), 25.0f);
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
```

### Step 2: Run tests to verify they fail

Run: `cmake --build build && ctest --test-dir build --output-on-failure`

Expected: FAIL — `fatal error: 'stan2d/core/math_types.hpp' file not found`

### Step 3: Implement (GREEN)

**File:** `include/stan2d/core/math_types.hpp`

```cpp
#pragma once

#include <glm/glm.hpp>
#include <glm/gtc/constants.hpp>

namespace stan2d {

// ── Vector / Matrix aliases ───────────────────────────────────────

using Vec2 = glm::vec2;
using Mat2 = glm::mat2;

// ── AABB ──────────────────────────────────────────────────────────

struct AABB {
    Vec2 min;
    Vec2 max;
};

[[nodiscard]] inline bool aabb_contains(const AABB& box, Vec2 point) {
    return point.x >= box.min.x && point.x <= box.max.x
        && point.y >= box.min.y && point.y <= box.max.y;
}

[[nodiscard]] inline bool aabb_overlaps(const AABB& a, const AABB& b) {
    return a.min.x <= b.max.x && a.max.x >= b.min.x
        && a.min.y <= b.max.y && a.max.y >= b.min.y;
}

[[nodiscard]] inline AABB aabb_merge(const AABB& a, const AABB& b) {
    return AABB{
        .min = {glm::min(a.min.x, b.min.x), glm::min(a.min.y, b.min.y)},
        .max = {glm::max(a.max.x, b.max.x), glm::max(a.max.y, b.max.y)}
    };
}

[[nodiscard]] inline float aabb_perimeter(const AABB& box) {
    Vec2 d = box.max - box.min;
    return 2.0f * (d.x + d.y);
}

[[nodiscard]] inline AABB aabb_expand(const AABB& box, float margin) {
    return AABB{
        .min = box.min - Vec2{margin, margin},
        .max = box.max + Vec2{margin, margin}
    };
}

} // namespace stan2d
```

### Step 4: Run tests to verify they pass

Run: `cmake --build build && ctest --test-dir build --output-on-failure`

Expected: PASS — all AABB and Vec2 tests green

### Step 5: Commit

```bash
git add include/stan2d/core/math_types.hpp tests/unit/test_math_types.cpp
git commit -m "feat: core math types — Vec2/Mat2 aliases and AABB utilities"
```

---