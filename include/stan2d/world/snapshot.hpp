#pragma once

#include <cstdint>
#include <vector>

#include <stan2d/core/handle.hpp>
#include <stan2d/core/math_types.hpp>
#include <stan2d/core/shapes.hpp>
#include <stan2d/joints/joint_types.hpp>

namespace stan2d {

struct JointSnapshot {
    // SparseSet state
    std::vector<uint32_t> sparse;
    std::vector<uint32_t> dense_to_sparse;
    std::vector<uint32_t> generations;
    std::vector<uint32_t> free_list;

    // Full SoA copy (mirrors JointStorage)
    std::vector<JointType> types;
    std::vector<uint32_t>  body_a;
    std::vector<uint32_t>  body_b;
    std::vector<Vec2>      anchor_a;
    std::vector<Vec2>      anchor_b;

    std::vector<uint8_t>   limit_enabled;
    std::vector<float>     limit_min;
    std::vector<float>     limit_max;
    std::vector<float>     reference_angle;
    std::vector<float>     accumulated_limit_impulse;

    std::vector<uint8_t>   motor_enabled;
    std::vector<float>     motor_target_speeds;
    std::vector<float>     motor_max_torque;
    std::vector<float>     accumulated_motor_impulse;

    std::vector<float>     spring_stiffness;
    std::vector<float>     spring_damping;
    std::vector<float>     spring_rest_length;

    std::vector<float>     distance_length;
    std::vector<uint8_t>   distance_cable_mode;

    std::vector<Vec2>      pulley_ground_a;
    std::vector<Vec2>      pulley_ground_b;
    std::vector<float>     pulley_ratio;
    std::vector<float>     pulley_constant;

    std::vector<float>     accumulated_impulse_x;
    std::vector<float>     accumulated_impulse_y;

    uint32_t count = 0;
};

struct WorldSnapshot {
    float    timestamp = 0.0f;
    uint32_t body_count = 0;

    // Body SoA data
    std::vector<Vec2>        positions;
    std::vector<Vec2>        velocities;
    std::vector<float>       rotations;
    std::vector<float>       angular_velocities;
    std::vector<float>       masses;
    std::vector<float>       inverse_masses;
    std::vector<float>       inertias;
    std::vector<float>       inverse_inertias;
    std::vector<Vec2>        forces;
    std::vector<float>       torques;
    std::vector<uint8_t>     body_types;  // BodyType stored as uint8_t
    std::vector<ShapeHandle> shape_ids;

    // Body SparseSet state (critical for Handle validity after restore)
    std::vector<uint32_t>    body_sparse;
    std::vector<uint32_t>    body_dense_to_sparse;
    std::vector<uint32_t>    body_generations;
    std::vector<uint32_t>    body_free_list;

    // ShapeRegistry state
    std::vector<ShapeData>   shapes;
    std::vector<AABB>        shape_aabbs;
    std::vector<uint32_t>    shape_sparse;
    std::vector<uint32_t>    shape_dense_to_sparse;
    std::vector<uint32_t>    shape_generations;
    std::vector<uint32_t>    shape_free_list;

    // Gravity
    Vec2 gravity{0.0f, 0.0f};

    // Joint snapshot
    JointSnapshot joints;
};

} // namespace stan2d
