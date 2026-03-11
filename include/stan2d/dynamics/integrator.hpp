#pragma once

#include <cstdint>

#include <stan2d/core/math_types.hpp>
#include <stan2d/dynamics/body_storage.hpp>

namespace stan2d {

// Integrate velocities: v += (F/m + gravity) * dt for Dynamic bodies only
// Clears force/torque accumulators after integration
inline void integrate_velocities(BodyStorage& bodies, uint32_t count,
                                 Vec2 gravity, float dt) {
    for (uint32_t i = 0; i < count; ++i) {
        if (bodies.body_types[i] == BodyType::Static ||
            bodies.body_types[i] == BodyType::Kinematic) {
            // Clear forces even for non-dynamic bodies
            bodies.forces[i] = {0.0f, 0.0f};
            bodies.torques[i] = 0.0f;
            continue;
        }

        // Linear: v += (F * inv_mass + gravity) * dt
        Vec2 acceleration = bodies.forces[i] * bodies.inverse_masses[i] + gravity;
        bodies.velocities[i] = bodies.velocities[i] + acceleration * dt;

        // Angular: ω += (τ * inv_inertia) * dt
        float angular_accel = bodies.torques[i] * bodies.inverse_inertias[i];
        bodies.angular_velocities[i] += angular_accel * dt;

        // Clear accumulators
        bodies.forces[i] = {0.0f, 0.0f};
        bodies.torques[i] = 0.0f;
    }
}

// Integrate positions: x += v * dt for Dynamic and Kinematic bodies
// Static bodies are skipped entirely
inline void integrate_positions(BodyStorage& bodies, uint32_t count, float dt) {
    for (uint32_t i = 0; i < count; ++i) {
        if (bodies.body_types[i] == BodyType::Static) {
            continue;
        }

        // Linear: x += v * dt (uses updated velocity — symplectic Euler)
        bodies.positions[i] = bodies.positions[i] + bodies.velocities[i] * dt;

        // Angular: θ += ω * dt
        bodies.rotations[i] += bodies.angular_velocities[i] * dt;
    }
}

} // namespace stan2d
