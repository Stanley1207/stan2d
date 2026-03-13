#pragma once

#include <cstdint>
#include <span>

#include <stan2d/core/math_types.hpp>

namespace stan2d {

struct JointStateView {
    uint32_t active_joint_count = 0;
    std::span<const uint8_t> types;
    std::span<const float>   angles;
    std::span<const float>   angular_speeds;
    std::span<const float>   motor_target_speeds;
    std::span<const uint8_t> motor_enabled;
    std::span<const float>   constraint_forces;
    std::span<const float>   lengths;
};

struct WorldStateView {
    float    timestamp = 0.0f;
    uint32_t active_body_count = 0;

    std::span<const Vec2>  positions;
    std::span<const Vec2>  velocities;
    std::span<const float> rotations;
    std::span<const float> angular_velocities;
    std::span<const float> masses;

    JointStateView joints;
};

} // namespace stan2d
