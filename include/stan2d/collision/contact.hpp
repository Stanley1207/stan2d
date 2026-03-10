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
