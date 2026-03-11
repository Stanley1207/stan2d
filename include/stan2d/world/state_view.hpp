#pragma once

#include <cstdint>
#include <span>

#include <stan2d/core/math_types.hpp>

namespace stan2d {

struct WorldStateView {
    float    timestamp = 0.0f;
    uint32_t active_body_count = 0;

    std::span<const Vec2>  positions;
    std::span<const Vec2>  velocities;
    std::span<const float> rotations;
    std::span<const float> angular_velocities;
    std::span<const float> masses;
};

} // namespace stan2d
