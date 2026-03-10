> **Note for Claude**: 牢记 `docs/tasks/00_master_plan.md` 中的四大铁律。

## Task 4: ShapeRegistry

**Goal:** Shape storage pool using SparseSet + std::variant, with pre-computed local AABBs. Supports CircleShape, PolygonShape, CapsuleShape.

**Files:**
- Create: `include/stan2d/core/shapes.hpp`
- Create: `include/stan2d/core/shape_registry.hpp`
- Create: `tests/unit/test_shape_registry.cpp`

**Depends on:** Task 2, Task 3

### Step 1: Write failing tests (RED)

**File:** `tests/unit/test_shape_registry.cpp`

```cpp
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
```

### Step 2: Run tests to verify they fail

Run: `cmake --build build && ctest --test-dir build --output-on-failure`

Expected: FAIL — `fatal error: 'stan2d/core/shape_registry.hpp' file not found`

### Step 3: Implement shape types

**File:** `include/stan2d/core/shapes.hpp`

```cpp
#pragma once

#include <array>
#include <cstdint>
#include <variant>

#include <stan2d/core/math_types.hpp>

namespace stan2d {

inline constexpr uint32_t MAX_POLYGON_VERTICES = 8;

struct CircleShape {
    float radius = 1.0f;
};

struct PolygonShape {
    uint32_t vertex_count = 0;
    std::array<Vec2, MAX_POLYGON_VERTICES> vertices{};
    std::array<Vec2, MAX_POLYGON_VERTICES> normals{};
};

struct CapsuleShape {
    Vec2  point_a{0.0f, -0.5f};
    Vec2  point_b{0.0f,  0.5f};
    float radius = 0.1f;
};

using ShapeData = std::variant<CircleShape, PolygonShape, CapsuleShape>;

// ── Local AABB computation (shape-space, centered at origin) ──────

[[nodiscard]] inline AABB compute_local_aabb(const ShapeData& shape) {
    struct Visitor {
        AABB operator()(const CircleShape& c) const {
            return AABB{
                .min = {-c.radius, -c.radius},
                .max = { c.radius,  c.radius}
            };
        }
        AABB operator()(const PolygonShape& p) const {
            Vec2 lo = p.vertices[0];
            Vec2 hi = p.vertices[0];
            for (uint32_t i = 1; i < p.vertex_count; ++i) {
                lo.x = glm::min(lo.x, p.vertices[i].x);
                lo.y = glm::min(lo.y, p.vertices[i].y);
                hi.x = glm::max(hi.x, p.vertices[i].x);
                hi.y = glm::max(hi.y, p.vertices[i].y);
            }
            return AABB{.min = lo, .max = hi};
        }
        AABB operator()(const CapsuleShape& c) const {
            Vec2 lo{
                glm::min(c.point_a.x, c.point_b.x) - c.radius,
                glm::min(c.point_a.y, c.point_b.y) - c.radius
            };
            Vec2 hi{
                glm::max(c.point_a.x, c.point_b.x) + c.radius,
                glm::max(c.point_a.y, c.point_b.y) + c.radius
            };
            return AABB{.min = lo, .max = hi};
        }
    };
    return std::visit(Visitor{}, shape);
}

} // namespace stan2d
```

### Step 4: Implement ShapeRegistry (GREEN)

**File:** `include/stan2d/core/shape_registry.hpp`

```cpp
#pragma once

#include <cassert>
#include <vector>

#include <stan2d/core/handle.hpp>
#include <stan2d/core/math_types.hpp>
#include <stan2d/core/shapes.hpp>
#include <stan2d/core/sparse_set.hpp>

namespace stan2d {

class ShapeRegistry {
public:
    void reserve(uint32_t capacity) {
        handles_.reserve(capacity);
        shapes_.reserve(capacity);
        local_aabbs_.reserve(capacity);
    }

    [[nodiscard]] ShapeHandle create(const ShapeData& shape) {
        Handle h = handles_.allocate();

        shapes_.push_back(shape);
        local_aabbs_.push_back(compute_local_aabb(shape));

        return ShapeHandle{h.index, h.generation};
    }

    void destroy(ShapeHandle handle) {
        Handle h{handle.index, handle.generation};
        auto swap = handles_.deallocate(h);

        if (swap.has_value()) {
            uint32_t dst = swap->removed_dense;
            uint32_t src = swap->moved_from_dense;
            shapes_[dst]      = shapes_[src];
            local_aabbs_[dst] = local_aabbs_[src];
        }

        shapes_.pop_back();
        local_aabbs_.pop_back();
    }

    [[nodiscard]] bool is_valid(ShapeHandle handle) const {
        return handles_.is_valid(Handle{handle.index, handle.generation});
    }

    [[nodiscard]] const ShapeData& get(ShapeHandle handle) const {
        Handle h{handle.index, handle.generation};
        return shapes_[handles_.dense_index(h)];
    }

    [[nodiscard]] const AABB& get_local_aabb(ShapeHandle handle) const {
        Handle h{handle.index, handle.generation};
        return local_aabbs_[handles_.dense_index(h)];
    }

    [[nodiscard]] uint32_t size() const { return handles_.size(); }

    // Accessors for state backup/restore
    [[nodiscard]] const SparseSet&             handles()     const { return handles_; }
    [[nodiscard]] const std::vector<ShapeData>& shapes()     const { return shapes_; }
    [[nodiscard]] const std::vector<AABB>&      local_aabbs() const { return local_aabbs_; }

private:
    SparseSet              handles_;
    std::vector<ShapeData> shapes_;
    std::vector<AABB>      local_aabbs_;
};

} // namespace stan2d
```

### Step 5: Run tests to verify they pass

Run: `cmake --build build && ctest --test-dir build --output-on-failure`

Expected: PASS — all ShapeRegistry tests green

### Step 6: Commit

```bash
git add include/stan2d/core/shapes.hpp include/stan2d/core/shape_registry.hpp tests/unit/test_shape_registry.cpp
git commit -m "feat: ShapeRegistry with Circle/Polygon/Capsule and local AABB computation"
```

---