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
