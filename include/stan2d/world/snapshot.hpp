#pragma once

#include <cstdint>
#include <vector>

#include <stan2d/core/handle.hpp>
#include <stan2d/core/math_types.hpp>
#include <stan2d/core/shapes.hpp>

namespace stan2d {

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
};

} // namespace stan2d
