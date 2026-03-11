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
