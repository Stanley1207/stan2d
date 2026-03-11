#pragma once

#include <cstdint>
#include <vector>

#include <stan2d/core/handle.hpp>
#include <stan2d/core/math_types.hpp>
#include <stan2d/core/sparse_set.hpp>

namespace stan2d {

enum class BodyType : uint8_t {
    Static,
    Dynamic,
    Kinematic
};

struct BodyStorage {
    // Kinematics
    std::vector<Vec2>        positions;
    std::vector<Vec2>        velocities;
    std::vector<float>       rotations;
    std::vector<float>       angular_velocities;

    // Mass properties
    std::vector<float>       masses;
    std::vector<float>       inverse_masses;
    std::vector<float>       inertias;
    std::vector<float>       inverse_inertias;

    // Force accumulators
    std::vector<Vec2>        forces;
    std::vector<float>       torques;

    // Body classification
    std::vector<BodyType>    body_types;

    // Shape reference
    std::vector<ShapeHandle> shape_ids;

    void reserve(uint32_t capacity) {
        positions.reserve(capacity);
        velocities.reserve(capacity);
        rotations.reserve(capacity);
        angular_velocities.reserve(capacity);
        masses.reserve(capacity);
        inverse_masses.reserve(capacity);
        inertias.reserve(capacity);
        inverse_inertias.reserve(capacity);
        forces.reserve(capacity);
        torques.reserve(capacity);
        body_types.reserve(capacity);
        shape_ids.reserve(capacity);
    }

    void push_back(Vec2 pos, Vec2 vel, float rot, float ang_vel,
                   float mass, float inv_mass, float inertia, float inv_inertia,
                   BodyType type, ShapeHandle shape) {
        positions.push_back(pos);
        velocities.push_back(vel);
        rotations.push_back(rot);
        angular_velocities.push_back(ang_vel);
        masses.push_back(mass);
        inverse_masses.push_back(inv_mass);
        inertias.push_back(inertia);
        inverse_inertias.push_back(inv_inertia);
        forces.push_back({0.0f, 0.0f});
        torques.push_back(0.0f);
        body_types.push_back(type);
        shape_ids.push_back(shape);
    }

    void swap_and_pop(uint32_t dst, uint32_t src) {
        positions[dst]          = positions[src];
        velocities[dst]         = velocities[src];
        rotations[dst]          = rotations[src];
        angular_velocities[dst] = angular_velocities[src];
        masses[dst]             = masses[src];
        inverse_masses[dst]     = inverse_masses[src];
        inertias[dst]           = inertias[src];
        inverse_inertias[dst]   = inverse_inertias[src];
        forces[dst]             = forces[src];
        torques[dst]            = torques[src];
        body_types[dst]         = body_types[src];
        shape_ids[dst]          = shape_ids[src];
    }

    void pop_back() {
        positions.pop_back();
        velocities.pop_back();
        rotations.pop_back();
        angular_velocities.pop_back();
        masses.pop_back();
        inverse_masses.pop_back();
        inertias.pop_back();
        inverse_inertias.pop_back();
        forces.pop_back();
        torques.pop_back();
        body_types.pop_back();
        shape_ids.pop_back();
    }

    [[nodiscard]] uint32_t size() const {
        return static_cast<uint32_t>(positions.size());
    }
};

} // namespace stan2d
