#pragma once

#include <cstdint>
#include <stan2d/core/math_types.hpp>

namespace stan2d {

struct ContactConstraint {
    uint32_t body_a;         // Dense index
    uint32_t body_b;         // Dense index

    Vec2  normal;            // A → B
    Vec2  tangent;           // Perpendicular to normal
    Vec2  contact_point;
    float penetration;

    // Effective mass along normal and tangent
    float normal_mass;
    float tangent_mass;

    // Accumulated impulses (for clamping and warm starting)
    float accumulated_normal_impulse  = 0.0f;
    float accumulated_tangent_impulse = 0.0f;

    // Bias for position correction (Baumgarte stabilization)
    float bias = 0.0f;

    // Contact point ID for warm starting across frames
    uint32_t id = 0;
};

struct SolverConfig {
    uint32_t iterations = 8;
    float    friction   = 0.3f;
    float    restitution = 0.0f;       // 0 = perfectly inelastic
    float    baumgarte  = 0.2f;        // Position correction factor
    float    slop       = 0.005f;      // Penetration allowance
};

} // namespace stan2d
