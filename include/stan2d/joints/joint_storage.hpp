#pragma once

#include <cstdint>
#include <limits>
#include <vector>

#include <stan2d/core/math_types.hpp>
#include <stan2d/joints/joint_types.hpp>

namespace stan2d {

struct JointStorage {
    // Common (all types)
    std::vector<JointType> types;
    std::vector<uint32_t>  body_a;          // dense index
    std::vector<uint32_t>  body_b;
    std::vector<Vec2>      anchor_a;        // body-space
    std::vector<Vec2>      anchor_b;

    // Hinge: limits (uint8_t avoids std::vector<bool> specialization)
    std::vector<uint8_t>   limit_enabled;   // 0 = disabled, 1 = enabled
    std::vector<float>     limit_min;
    std::vector<float>     limit_max;
    std::vector<float>     reference_angle;
    std::vector<float>     accumulated_limit_impulse;  // warm-start

    // Hinge/Motor
    std::vector<uint8_t>   motor_enabled;        // 0 = disabled, 1 = enabled
    std::vector<float>     motor_target_speeds;
    std::vector<float>     motor_max_torque;
    std::vector<float>     accumulated_motor_impulse;  // warm-start

    // Spring
    std::vector<float>     spring_stiffness;
    std::vector<float>     spring_damping;
    std::vector<float>     spring_rest_length;

    // Distance
    std::vector<float>     distance_length;
    std::vector<uint8_t>   distance_cable_mode;  // 0 = rigid rod, 1 = cable

    // Pulley
    std::vector<Vec2>      pulley_ground_a;
    std::vector<Vec2>      pulley_ground_b;
    std::vector<float>     pulley_ratio;
    std::vector<float>     pulley_constant;

    // Common warm-start impulse (linear, all types except Spring)
    std::vector<float>     accumulated_impulse_x;
    std::vector<float>     accumulated_impulse_y;

    // Per-frame constraint force magnitude (written each solve(), reset each frame)
    std::vector<float>     constraint_forces;

    // Cached per-frame observables for JointStateView (recomputed each step)
    std::vector<float>     cached_angles;
    std::vector<float>     cached_angular_speeds;
    std::vector<float>     cached_lengths;

    uint32_t size = 0;

    void reserve(uint32_t capacity) {
        types.reserve(capacity);
        body_a.reserve(capacity);
        body_b.reserve(capacity);
        anchor_a.reserve(capacity);
        anchor_b.reserve(capacity);
        limit_enabled.reserve(capacity);
        limit_min.reserve(capacity);
        limit_max.reserve(capacity);
        reference_angle.reserve(capacity);
        accumulated_limit_impulse.reserve(capacity);
        motor_enabled.reserve(capacity);
        motor_target_speeds.reserve(capacity);
        motor_max_torque.reserve(capacity);
        accumulated_motor_impulse.reserve(capacity);
        spring_stiffness.reserve(capacity);
        spring_damping.reserve(capacity);
        spring_rest_length.reserve(capacity);
        distance_length.reserve(capacity);
        distance_cable_mode.reserve(capacity);
        pulley_ground_a.reserve(capacity);
        pulley_ground_b.reserve(capacity);
        pulley_ratio.reserve(capacity);
        pulley_constant.reserve(capacity);
        accumulated_impulse_x.reserve(capacity);
        accumulated_impulse_y.reserve(capacity);
        constraint_forces.reserve(capacity);
        cached_angles.reserve(capacity);
        cached_angular_speeds.reserve(capacity);
        cached_lengths.reserve(capacity);
    }

    // Push a new joint. dense_body_a/b are dense indices.
    // ref_angle: θ_b - θ_a at creation (Hinge), 0 otherwise.
    // pulley_const: len_a + ratio*len_b at creation (Pulley), 0 otherwise.
    void push_back(const JointDef& def, uint32_t dense_a, uint32_t dense_b,
                   float ref_angle, float pulley_const) {
        types.push_back(def.type);
        body_a.push_back(dense_a);
        body_b.push_back(dense_b);
        anchor_a.push_back(def.anchor_a);
        anchor_b.push_back(def.anchor_b);

        limit_enabled.push_back(def.limit_enabled ? 1u : 0u);
        limit_min.push_back(def.limit_min);
        limit_max.push_back(def.limit_max);
        reference_angle.push_back(ref_angle);
        accumulated_limit_impulse.push_back(0.0f);

        motor_enabled.push_back(def.motor_enabled ? 1u : 0u);
        // NaN sentinel for non-motor joints — RL agents must mask by motor_enabled
        motor_target_speeds.push_back(
            def.motor_enabled ? def.motor_speed
                              : std::numeric_limits<float>::quiet_NaN());
        motor_max_torque.push_back(def.motor_torque);
        accumulated_motor_impulse.push_back(0.0f);

        spring_stiffness.push_back(def.stiffness);
        spring_damping.push_back(def.damping);
        spring_rest_length.push_back(def.rest_length);

        distance_length.push_back(def.distance);
        distance_cable_mode.push_back(def.cable_mode ? 1u : 0u);

        pulley_ground_a.push_back(def.ground_a);
        pulley_ground_b.push_back(def.ground_b);
        pulley_ratio.push_back(def.pulley_ratio);
        pulley_constant.push_back(pulley_const);

        accumulated_impulse_x.push_back(0.0f);
        accumulated_impulse_y.push_back(0.0f);

        constraint_forces.push_back(0.0f);
        cached_angles.push_back(0.0f);
        cached_angular_speeds.push_back(0.0f);
        cached_lengths.push_back(0.0f);

        ++size;
    }

    void swap_and_pop(uint32_t dst, uint32_t src) {
        types[dst]                     = types[src];
        body_a[dst]                    = body_a[src];
        body_b[dst]                    = body_b[src];
        anchor_a[dst]                  = anchor_a[src];
        anchor_b[dst]                  = anchor_b[src];
        limit_enabled[dst]             = limit_enabled[src];
        limit_min[dst]                 = limit_min[src];
        limit_max[dst]                 = limit_max[src];
        reference_angle[dst]           = reference_angle[src];
        accumulated_limit_impulse[dst] = accumulated_limit_impulse[src];
        motor_enabled[dst]             = motor_enabled[src];
        motor_target_speeds[dst]       = motor_target_speeds[src];
        motor_max_torque[dst]          = motor_max_torque[src];
        accumulated_motor_impulse[dst] = accumulated_motor_impulse[src];
        spring_stiffness[dst]          = spring_stiffness[src];
        spring_damping[dst]            = spring_damping[src];
        spring_rest_length[dst]        = spring_rest_length[src];
        distance_length[dst]           = distance_length[src];
        distance_cable_mode[dst]       = distance_cable_mode[src];
        pulley_ground_a[dst]           = pulley_ground_a[src];
        pulley_ground_b[dst]           = pulley_ground_b[src];
        pulley_ratio[dst]              = pulley_ratio[src];
        pulley_constant[dst]           = pulley_constant[src];
        accumulated_impulse_x[dst]     = accumulated_impulse_x[src];
        accumulated_impulse_y[dst]     = accumulated_impulse_y[src];
        constraint_forces[dst]         = constraint_forces[src];
        cached_angles[dst]             = cached_angles[src];
        cached_angular_speeds[dst]     = cached_angular_speeds[src];
        cached_lengths[dst]            = cached_lengths[src];
    }

    void pop_back() {
        types.pop_back();
        body_a.pop_back();
        body_b.pop_back();
        anchor_a.pop_back();
        anchor_b.pop_back();
        limit_enabled.pop_back();
        limit_min.pop_back();
        limit_max.pop_back();
        reference_angle.pop_back();
        accumulated_limit_impulse.pop_back();
        motor_enabled.pop_back();
        motor_target_speeds.pop_back();
        motor_max_torque.pop_back();
        accumulated_motor_impulse.pop_back();
        spring_stiffness.pop_back();
        spring_damping.pop_back();
        spring_rest_length.pop_back();
        distance_length.pop_back();
        distance_cable_mode.pop_back();
        pulley_ground_a.pop_back();
        pulley_ground_b.pop_back();
        pulley_ratio.pop_back();
        pulley_constant.pop_back();
        accumulated_impulse_x.pop_back();
        accumulated_impulse_y.pop_back();
        constraint_forces.pop_back();
        cached_angles.pop_back();
        cached_angular_speeds.pop_back();
        cached_lengths.pop_back();
        --size;
    }
};

} // namespace stan2d
