#pragma once

#include <algorithm>
#include <cmath>
#include <vector>

#include <glm/glm.hpp>
#include <stan2d/collision/contact.hpp>
#include <stan2d/constraints/contact_constraint.hpp>
#include <stan2d/dynamics/body_storage.hpp>

namespace stan2d {

inline Vec2 cross_scalar_vec(float s, Vec2 v) {
    return {-s * v.y, s * v.x};
}

inline float cross_vec_vec(Vec2 a, Vec2 b) {
    return a.x * b.y - a.y * b.x;
}

inline void prepare_contact_constraints(
    const ContactManifold& manifold,
    uint32_t dense_a, uint32_t dense_b,
    const BodyStorage& bodies,
    std::vector<ContactConstraint>& out)
{
    float inv_mass_a    = bodies.inverse_masses[dense_a];
    float inv_mass_b    = bodies.inverse_masses[dense_b];
    float inv_inertia_a = bodies.inverse_inertias[dense_a];
    float inv_inertia_b = bodies.inverse_inertias[dense_b];

    Vec2 normal = manifold.normal;
    Vec2 tangent{-normal.y, normal.x};

    for (uint32_t i = 0; i < manifold.point_count; ++i) {
        ContactConstraint c;
        c.body_a = dense_a;
        c.body_b = dense_b;
        c.normal = normal;
        c.tangent = tangent;
        c.contact_point = manifold.points[i].position;
        c.penetration = manifold.points[i].penetration;
        c.id = manifold.points[i].id;

        Vec2 ra = c.contact_point - bodies.positions[dense_a];
        Vec2 rb = c.contact_point - bodies.positions[dense_b];

        // Effective mass along normal
        float rn_a = cross_vec_vec(ra, normal);
        float rn_b = cross_vec_vec(rb, normal);
        float k_normal = inv_mass_a + inv_mass_b
                       + inv_inertia_a * rn_a * rn_a
                       + inv_inertia_b * rn_b * rn_b;
        c.normal_mass = (k_normal > 0.0f) ? 1.0f / k_normal : 0.0f;

        // Effective mass along tangent
        float rt_a = cross_vec_vec(ra, tangent);
        float rt_b = cross_vec_vec(rb, tangent);
        float k_tangent = inv_mass_a + inv_mass_b
                        + inv_inertia_a * rt_a * rt_a
                        + inv_inertia_b * rt_b * rt_b;
        c.tangent_mass = (k_tangent > 0.0f) ? 1.0f / k_tangent : 0.0f;

        out.push_back(c);
    }
}

inline void warm_start(
    const std::vector<ContactConstraint>& constraints,
    BodyStorage& bodies)
{
    for (const auto& c : constraints) {
        Vec2 impulse = c.normal * c.accumulated_normal_impulse
                     + c.tangent * c.accumulated_tangent_impulse;

        Vec2 ra = c.contact_point - bodies.positions[c.body_a];
        Vec2 rb = c.contact_point - bodies.positions[c.body_b];

        bodies.velocities[c.body_a] = bodies.velocities[c.body_a]
                                    - impulse * bodies.inverse_masses[c.body_a];
        bodies.angular_velocities[c.body_a] -= bodies.inverse_inertias[c.body_a]
                                             * cross_vec_vec(ra, impulse);

        bodies.velocities[c.body_b] = bodies.velocities[c.body_b]
                                    + impulse * bodies.inverse_masses[c.body_b];
        bodies.angular_velocities[c.body_b] += bodies.inverse_inertias[c.body_b]
                                             * cross_vec_vec(rb, impulse);
    }
}

inline void solve_constraints(
    std::vector<ContactConstraint>& constraints,
    BodyStorage& bodies,
    const SolverConfig& config)
{
    for (uint32_t iter = 0; iter < config.iterations; ++iter) {
        for (auto& c : constraints) {
            Vec2 ra = c.contact_point - bodies.positions[c.body_a];
            Vec2 rb = c.contact_point - bodies.positions[c.body_b];

            // Relative velocity at contact point
            Vec2 rel_vel = (bodies.velocities[c.body_b]
                          + cross_scalar_vec(bodies.angular_velocities[c.body_b], rb))
                         - (bodies.velocities[c.body_a]
                          + cross_scalar_vec(bodies.angular_velocities[c.body_a], ra));

            // ── Normal impulse ────────────────────────────────
            float vn = glm::dot(rel_vel, c.normal);

            // Baumgarte bias for position correction
            float bias = config.baumgarte
                       * glm::max(c.penetration - config.slop, 0.0f);

            float lambda_n = c.normal_mass * (-vn + bias);

            // Clamp: accumulated impulse must be >= 0 (no pull)
            float new_impulse = glm::max(c.accumulated_normal_impulse + lambda_n, 0.0f);
            lambda_n = new_impulse - c.accumulated_normal_impulse;
            c.accumulated_normal_impulse = new_impulse;

            Vec2 impulse_n = c.normal * lambda_n;

            bodies.velocities[c.body_a] = bodies.velocities[c.body_a]
                                        - impulse_n * bodies.inverse_masses[c.body_a];
            bodies.angular_velocities[c.body_a] -= bodies.inverse_inertias[c.body_a]
                                                 * cross_vec_vec(ra, impulse_n);
            bodies.velocities[c.body_b] = bodies.velocities[c.body_b]
                                        + impulse_n * bodies.inverse_masses[c.body_b];
            bodies.angular_velocities[c.body_b] += bodies.inverse_inertias[c.body_b]
                                                 * cross_vec_vec(rb, impulse_n);

            // ── Tangent impulse (friction) ────────────────────
            // Recompute relative velocity after normal impulse
            rel_vel = (bodies.velocities[c.body_b]
                     + cross_scalar_vec(bodies.angular_velocities[c.body_b], rb))
                    - (bodies.velocities[c.body_a]
                     + cross_scalar_vec(bodies.angular_velocities[c.body_a], ra));

            float vt = glm::dot(rel_vel, c.tangent);
            float lambda_t = c.tangent_mass * (-vt);

            // Coulomb friction clamp: |f_t| <= mu * f_n
            float max_friction = config.friction * c.accumulated_normal_impulse;
            float new_tangent = glm::clamp(
                c.accumulated_tangent_impulse + lambda_t,
                -max_friction, max_friction);
            lambda_t = new_tangent - c.accumulated_tangent_impulse;
            c.accumulated_tangent_impulse = new_tangent;

            Vec2 impulse_t = c.tangent * lambda_t;

            bodies.velocities[c.body_a] = bodies.velocities[c.body_a]
                                        - impulse_t * bodies.inverse_masses[c.body_a];
            bodies.angular_velocities[c.body_a] -= bodies.inverse_inertias[c.body_a]
                                                 * cross_vec_vec(ra, impulse_t);
            bodies.velocities[c.body_b] = bodies.velocities[c.body_b]
                                        + impulse_t * bodies.inverse_masses[c.body_b];
            bodies.angular_velocities[c.body_b] += bodies.inverse_inertias[c.body_b]
                                                 * cross_vec_vec(rb, impulse_t);
        }
    }
}

} // namespace stan2d
