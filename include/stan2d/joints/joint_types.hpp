#pragma once

#include <cstdint>

#include <glm/gtc/constants.hpp>

#include <stan2d/core/handle.hpp>
#include <stan2d/core/math_types.hpp>

namespace stan2d {

enum class JointType : uint8_t {
    Hinge    = 0,
    Spring   = 1,
    Distance = 2,
    Pulley   = 3,
};

struct JointDef {
    JointType  type       = JointType::Hinge;
    BodyHandle body_a;
    BodyHandle body_b;
    Vec2       anchor_a   = {0.0f, 0.0f};   // body-space
    Vec2       anchor_b   = {0.0f, 0.0f};   // body-space

    // Hinge: limits
    bool  limit_enabled   = false;
    float limit_min       = -glm::pi<float>();
    float limit_max       =  glm::pi<float>();

    // Hinge: motor
    bool  motor_enabled   = false;
    float motor_speed     = 0.0f;    // rad/s
    float motor_torque    = 0.0f;    // N·m max

    // Spring
    float stiffness       = 100.0f;
    float damping         = 1.0f;
    float rest_length     = 0.0f;    // 0 = current distance at creation

    // Distance
    float distance        = 0.0f;    // 0 = current distance at creation
    bool  cable_mode      = false;   // false = rigid rod, true = cable (pull only)

    // Pulley
    Vec2  ground_a        = {0.0f, 0.0f};   // world-space
    Vec2  ground_b        = {0.0f, 0.0f};   // world-space
    float pulley_ratio    = 1.0f;
};

} // namespace stan2d
