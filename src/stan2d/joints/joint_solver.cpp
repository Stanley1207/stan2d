#include <stan2d/joints/joint_solver.hpp>

#include <cmath>
#include <glm/glm.hpp>
#include <glm/gtc/constants.hpp>

namespace stan2d {

namespace {

// Wrap angle to [-pi, pi]
float wrap_angle(float a) {
    while (a > glm::pi<float>())  a -= 2.0f * glm::pi<float>();
    while (a < -glm::pi<float>()) a += 2.0f * glm::pi<float>();
    return a;
}

// Rotate a body-space vector by angle (cos_t, sin_t)
inline Vec2 rotate(Vec2 v, float cos_t, float sin_t) {
    return {cos_t * v.x - sin_t * v.y,
            sin_t * v.x + cos_t * v.y};
}

} // anonymous namespace

void prepare_joint_constraints(JointStorage& joints, const BodyStorage& bodies, float /*dt*/) {
    for (uint32_t i = 0; i < joints.size; ++i) {
        joints.constraint_forces[i] = 0.0f;

        uint32_t a = joints.body_a[i];
        uint32_t b = joints.body_b[i];

        float cos_a = std::cos(bodies.rotations[a]);
        float sin_a = std::sin(bodies.rotations[a]);
        float cos_b = std::cos(bodies.rotations[b]);
        float sin_b = std::sin(bodies.rotations[b]);

        Vec2 ra = rotate(joints.anchor_a[i], cos_a, sin_a);
        Vec2 rb = rotate(joints.anchor_b[i], cos_b, sin_b);

        Vec2 world_a = bodies.positions[a] + ra;
        Vec2 world_b = bodies.positions[b] + rb;
        float len = glm::length(world_b - world_a);

        float angle = wrap_angle(bodies.rotations[b] - bodies.rotations[a]
                                 - joints.reference_angle[i]);
        float angular_speed = bodies.angular_velocities[b] - bodies.angular_velocities[a];

        joints.cached_angles[i]         = (joints.types[i] == JointType::Hinge) ? angle : 0.0f;
        joints.cached_angular_speeds[i] = (joints.types[i] == JointType::Hinge) ? angular_speed : 0.0f;
        joints.cached_lengths[i]        = len;
    }
}

void warm_start_joints(const JointStorage& joints, BodyStorage& bodies) {
    for (uint32_t i = 0; i < joints.size; ++i) {
        if (joints.types[i] == JointType::Spring) continue;  // Spring: no warm-start

        uint32_t a = joints.body_a[i];
        uint32_t b = joints.body_b[i];

        float cos_a = std::cos(bodies.rotations[a]);
        float sin_a = std::sin(bodies.rotations[a]);
        float cos_b = std::cos(bodies.rotations[b]);
        float sin_b = std::sin(bodies.rotations[b]);

        Vec2 ra = rotate(joints.anchor_a[i], cos_a, sin_a);
        Vec2 rb = rotate(joints.anchor_b[i], cos_b, sin_b);

        // Pulley: warm-start each rope segment independently
        if (joints.types[i] == JointType::Pulley) {
            float lambda = joints.accumulated_impulse_x[i];
            Vec2 world_a = bodies.positions[a] + ra;
            Vec2 world_b = bodies.positions[b] + rb;
            Vec2 delta_a = world_a - joints.pulley_ground_a[i];
            Vec2 delta_b = world_b - joints.pulley_ground_b[i];
            float len_a = glm::length(delta_a);
            float len_b = glm::length(delta_b);
            if (len_a < 1e-6f || len_b < 1e-6f) continue;
            Vec2 uA = delta_a / len_a;
            Vec2 uB = delta_b / len_b;
            float ratio = joints.pulley_ratio[i];
            Vec2 imp_a = uA * lambda;
            Vec2 imp_b = uB * (ratio * lambda);
            bodies.velocities[a] = bodies.velocities[a] - imp_a * bodies.inverse_masses[a];
            bodies.angular_velocities[a] -= bodies.inverse_inertias[a] * cross_vec_vec(ra, imp_a);
            bodies.velocities[b] = bodies.velocities[b] - imp_b * bodies.inverse_masses[b];
            bodies.angular_velocities[b] -= bodies.inverse_inertias[b] * cross_vec_vec(rb, imp_b);
            continue;
        }

        Vec2 impulse;
        if (joints.types[i] == JointType::Distance) {
            // Scalar impulse stored in accumulated_impulse_x — project onto current axis
            Vec2 world_a = bodies.positions[a] + ra;
            Vec2 world_b = bodies.positions[b] + rb;
            Vec2 delta = world_b - world_a;
            float len = glm::length(delta);
            Vec2 axis = (len > 1e-6f) ? delta / len : Vec2{1.0f, 0.0f};
            impulse = axis * joints.accumulated_impulse_x[i];
        } else {
            // Hinge: 2D impulse stored directly
            impulse = {joints.accumulated_impulse_x[i], joints.accumulated_impulse_y[i]};
        }

        bodies.velocities[a] = bodies.velocities[a] - impulse * bodies.inverse_masses[a];
        bodies.angular_velocities[a] -= bodies.inverse_inertias[a] * cross_vec_vec(ra, impulse);

        bodies.velocities[b] = bodies.velocities[b] + impulse * bodies.inverse_masses[b];
        bodies.angular_velocities[b] += bodies.inverse_inertias[b] * cross_vec_vec(rb, impulse);

        // Warm-start limit and motor (Hinge only)
        if (joints.types[i] == JointType::Hinge) {
            float total_angular = joints.accumulated_limit_impulse[i]
                                + joints.accumulated_motor_impulse[i];
            bodies.angular_velocities[a] -= bodies.inverse_inertias[a] * total_angular;
            bodies.angular_velocities[b] += bodies.inverse_inertias[b] * total_angular;
        }
    }
}

void solve_joints(JointStorage& joints, BodyStorage& bodies,
                  const SolverConfig& config, float dt) {
    for (uint32_t i = 0; i < joints.size; ++i) {
        // ── Distance joint ────────────────────────────────────────────
        if (joints.types[i] == JointType::Distance) {
            uint32_t a = joints.body_a[i];
            uint32_t b = joints.body_b[i];

            float inv_ma = bodies.inverse_masses[a];
            float inv_mb = bodies.inverse_masses[b];
            float inv_ia = bodies.inverse_inertias[a];
            float inv_ib = bodies.inverse_inertias[b];

            float cos_a = std::cos(bodies.rotations[a]);
            float sin_a = std::sin(bodies.rotations[a]);
            float cos_b = std::cos(bodies.rotations[b]);
            float sin_b = std::sin(bodies.rotations[b]);

            Vec2 ra = rotate(joints.anchor_a[i], cos_a, sin_a);
            Vec2 rb = rotate(joints.anchor_b[i], cos_b, sin_b);

            Vec2 world_a = bodies.positions[a] + ra;
            Vec2 world_b = bodies.positions[b] + rb;
            Vec2 delta = world_b - world_a;
            float current_len = glm::length(delta);

            if (current_len < 1e-6f) continue;

            Vec2 axis = delta / current_len;

            float rn_a = cross_vec_vec(ra, axis);
            float rn_b = cross_vec_vec(rb, axis);
            float k = inv_ma + inv_mb + inv_ia * rn_a * rn_a + inv_ib * rn_b * rn_b;
            if (k < 1e-12f) continue;
            float eff_mass = 1.0f / k;

            Vec2 v_rel = (bodies.velocities[b]
                         + cross_scalar_vec(bodies.angular_velocities[b], rb))
                        - (bodies.velocities[a]
                         + cross_scalar_vec(bodies.angular_velocities[a], ra));
            float v_along = glm::dot(v_rel, axis);

            float c_val = current_len - joints.distance_length[i];
            float bias = config.baumgarte / dt * c_val;

            float lambda = eff_mass * (-v_along - bias);

            float old_acc = joints.accumulated_impulse_x[i];
            float new_acc;
            if (joints.distance_cable_mode[i]) {
                // Cable: pull-only (λ ≤ 0 pulls bodies together, no pushing)
                new_acc = glm::min(old_acc + lambda, 0.0f);
            } else {
                // Rigid rod: two-sided, no clamp
                new_acc = old_acc + lambda;
            }
            lambda = new_acc - old_acc;
            joints.accumulated_impulse_x[i] = new_acc;
            joints.accumulated_impulse_y[i] = 0.0f;

            Vec2 impulse = axis * lambda;

            bodies.velocities[a] = bodies.velocities[a] - impulse * inv_ma;
            bodies.angular_velocities[a] -= inv_ia * cross_vec_vec(ra, impulse);
            bodies.velocities[b] = bodies.velocities[b] + impulse * inv_mb;
            bodies.angular_velocities[b] += inv_ib * cross_vec_vec(rb, impulse);

            joints.constraint_forces[i] += std::abs(lambda);
            continue;
        }

        // ── Spring joint ──────────────────────────────────────────────
        if (joints.types[i] == JointType::Spring) {
            uint32_t a = joints.body_a[i];
            uint32_t b = joints.body_b[i];

            float inv_ma = bodies.inverse_masses[a];
            float inv_mb = bodies.inverse_masses[b];
            float inv_ia = bodies.inverse_inertias[a];
            float inv_ib = bodies.inverse_inertias[b];

            float cos_a = std::cos(bodies.rotations[a]);
            float sin_a = std::sin(bodies.rotations[a]);
            float cos_b = std::cos(bodies.rotations[b]);
            float sin_b = std::sin(bodies.rotations[b]);

            Vec2 ra = rotate(joints.anchor_a[i], cos_a, sin_a);
            Vec2 rb = rotate(joints.anchor_b[i], cos_b, sin_b);

            Vec2 world_a = bodies.positions[a] + ra;
            Vec2 world_b = bodies.positions[b] + rb;
            Vec2 delta = world_b - world_a;
            float current_len = glm::length(delta);

            if (current_len < 1e-6f) continue;

            Vec2 axis = delta / current_len;

            float rn_a = cross_vec_vec(ra, axis);
            float rn_b = cross_vec_vec(rb, axis);
            float k = inv_ma + inv_mb + inv_ia * rn_a * rn_a + inv_ib * rn_b * rn_b;
            if (k < 1e-12f) continue;
            float eff_mass = 1.0f / k;

            Vec2 v_rel = (bodies.velocities[b]
                         + cross_scalar_vec(bodies.angular_velocities[b], rb))
                        - (bodies.velocities[a]
                         + cross_scalar_vec(bodies.angular_velocities[a], ra));
            float v_along = glm::dot(v_rel, axis);

            float displacement = current_len - joints.spring_rest_length[i];
            float spring_force = -joints.spring_stiffness[i] * displacement
                               - joints.spring_damping[i] * v_along;
            float lambda = spring_force * dt * eff_mass;

            Vec2 impulse = axis * lambda;

            bodies.velocities[a] = bodies.velocities[a] - impulse * inv_ma;
            bodies.angular_velocities[a] -= inv_ia * cross_vec_vec(ra, impulse);
            bodies.velocities[b] = bodies.velocities[b] + impulse * inv_mb;
            bodies.angular_velocities[b] += inv_ib * cross_vec_vec(rb, impulse);

            // Spring: stateless, no warm-start accumulation
            joints.accumulated_impulse_x[i] = 0.0f;
            joints.accumulated_impulse_y[i] = 0.0f;

            joints.constraint_forces[i] += std::abs(lambda);
            continue;
        }

        // ── Pulley joint ──────────────────────────────────────────────
        if (joints.types[i] == JointType::Pulley) {
            uint32_t a = joints.body_a[i];
            uint32_t b = joints.body_b[i];

            float inv_ma = bodies.inverse_masses[a];
            float inv_mb = bodies.inverse_masses[b];
            float inv_ia = bodies.inverse_inertias[a];
            float inv_ib = bodies.inverse_inertias[b];

            float cos_a = std::cos(bodies.rotations[a]);
            float sin_a = std::sin(bodies.rotations[a]);
            float cos_b = std::cos(bodies.rotations[b]);
            float sin_b = std::sin(bodies.rotations[b]);

            Vec2 ra = rotate(joints.anchor_a[i], cos_a, sin_a);
            Vec2 rb = rotate(joints.anchor_b[i], cos_b, sin_b);

            Vec2 world_a = bodies.positions[a] + ra;
            Vec2 world_b = bodies.positions[b] + rb;

            Vec2 delta_a = world_a - joints.pulley_ground_a[i];
            Vec2 delta_b = world_b - joints.pulley_ground_b[i];
            float len_a = glm::length(delta_a);
            float len_b = glm::length(delta_b);

            if (len_a < 1e-6f || len_b < 1e-6f) continue;

            Vec2 uA = delta_a / len_a;
            Vec2 uB = delta_b / len_b;
            float ratio = joints.pulley_ratio[i];

            // Effective mass: 1/k = 1/mA + (rA×uA)²/IA + ratio²*(1/mB + (rB×uB)²/IB)
            float rn_a = cross_vec_vec(ra, uA);
            float rn_b = cross_vec_vec(rb, uB);
            float k = inv_ma + inv_ia * rn_a * rn_a
                    + ratio * ratio * (inv_mb + inv_ib * rn_b * rn_b);
            if (k < 1e-12f) continue;
            float eff_mass = 1.0f / k;

            // Constraint: len_a + ratio * len_b = constant
            float c_val = len_a + ratio * len_b - joints.pulley_constant[i];

            // Constraint velocity: v_a·uA + ratio * v_b·uB
            Vec2 v_a = bodies.velocities[a]
                     + cross_scalar_vec(bodies.angular_velocities[a], ra);
            Vec2 v_b = bodies.velocities[b]
                     + cross_scalar_vec(bodies.angular_velocities[b], rb);
            float v_proj = glm::dot(v_a, uA) + ratio * glm::dot(v_b, uB);

            float bias = config.baumgarte / dt * c_val;
            // λ > 0 pulls bodies toward ground (rope taut); λ = 0 when slack
            float lambda = eff_mass * (v_proj + bias);

            // Pulley is one-sided: rope can only pull (accumulated >= 0)
            float old_acc = joints.accumulated_impulse_x[i];
            joints.accumulated_impulse_x[i] = glm::max(old_acc + lambda, 0.0f);
            lambda = joints.accumulated_impulse_x[i] - old_acc;
            joints.accumulated_impulse_y[i] = 0.0f;

            // Apply impulses: pull each body toward its ground point
            Vec2 impulse_a = uA * lambda;
            Vec2 impulse_b = uB * (ratio * lambda);

            bodies.velocities[a] = bodies.velocities[a] - impulse_a * inv_ma;
            bodies.angular_velocities[a] -= inv_ia * cross_vec_vec(ra, impulse_a);

            bodies.velocities[b] = bodies.velocities[b] - impulse_b * inv_mb;
            bodies.angular_velocities[b] -= inv_ib * cross_vec_vec(rb, impulse_b);

            joints.constraint_forces[i] += std::abs(lambda);
            continue;
        }

        if (joints.types[i] != JointType::Hinge) continue;

        uint32_t a = joints.body_a[i];
        uint32_t b = joints.body_b[i];

        float inv_ma = bodies.inverse_masses[a];
        float inv_mb = bodies.inverse_masses[b];
        float inv_ia = bodies.inverse_inertias[a];
        float inv_ib = bodies.inverse_inertias[b];

        float cos_a = std::cos(bodies.rotations[a]);
        float sin_a = std::sin(bodies.rotations[a]);
        float cos_b = std::cos(bodies.rotations[b]);
        float sin_b = std::sin(bodies.rotations[b]);

        Vec2 ra = rotate(joints.anchor_a[i], cos_a, sin_a);
        Vec2 rb = rotate(joints.anchor_b[i], cos_b, sin_b);

        // ── Point-to-point constraint (2×2 effective mass) ───────────
        Vec2 v_constraint = (bodies.velocities[b]
                            + cross_scalar_vec(bodies.angular_velocities[b], rb))
                           - (bodies.velocities[a]
                            + cross_scalar_vec(bodies.angular_velocities[a], ra));

        Vec2 pos_error = (bodies.positions[b] + rb) - (bodies.positions[a] + ra);
        float beta = config.baumgarte;

        // K = (inv_ma + inv_mb)*I - inv_ia*(ra_perp ⊗ ra_perp) - inv_ib*(rb_perp ⊗ rb_perp)
        float k11 = inv_ma + inv_mb + inv_ia * ra.y * ra.y + inv_ib * rb.y * rb.y;
        float k12 = -inv_ia * ra.x * ra.y - inv_ib * rb.x * rb.y;
        float k22 = inv_ma + inv_mb + inv_ia * ra.x * ra.x + inv_ib * rb.x * rb.x;

        float det = k11 * k22 - k12 * k12;
        if (std::abs(det) < 1e-12f) continue;
        float inv_det = 1.0f / det;

        Vec2 rhs = -(v_constraint + (beta / dt) * pos_error);
        Vec2 lambda{inv_det * (k22 * rhs.x - k12 * rhs.y),
                    inv_det * (-k12 * rhs.x + k11 * rhs.y)};

        joints.accumulated_impulse_x[i] += lambda.x;
        joints.accumulated_impulse_y[i] += lambda.y;

        bodies.velocities[a] = bodies.velocities[a] - lambda * inv_ma;
        bodies.angular_velocities[a] -= inv_ia * cross_vec_vec(ra, lambda);
        bodies.velocities[b] = bodies.velocities[b] + lambda * inv_mb;
        bodies.angular_velocities[b] += inv_ib * cross_vec_vec(rb, lambda);

        joints.constraint_forces[i] += glm::length(lambda);

        // ── Angular limit ─────────────────────────────────────────────
        if (joints.limit_enabled[i]) {
            float angle = wrap_angle(bodies.rotations[b] - bodies.rotations[a]
                                     - joints.reference_angle[i]);
            float angular_speed = bodies.angular_velocities[b] - bodies.angular_velocities[a];
            float inv_i_total = inv_ia + inv_ib;
            if (inv_i_total > 0.0f) {
                float eff_mass = 1.0f / inv_i_total;

                if (angle < joints.limit_min[i]) {
                    float c_val = angle - joints.limit_min[i];
                    float lambda_lim = eff_mass * (-angular_speed - beta / dt * c_val);
                    float old_acc = joints.accumulated_limit_impulse[i];
                    joints.accumulated_limit_impulse[i] = glm::max(old_acc + lambda_lim, 0.0f);
                    float delta = joints.accumulated_limit_impulse[i] - old_acc;
                    bodies.angular_velocities[a] -= inv_ia * delta;
                    bodies.angular_velocities[b] += inv_ib * delta;
                    joints.constraint_forces[i] += std::abs(delta);
                } else if (angle > joints.limit_max[i]) {
                    float c_val = angle - joints.limit_max[i];
                    float lambda_lim = eff_mass * (-angular_speed - beta / dt * c_val);
                    float old_acc = joints.accumulated_limit_impulse[i];
                    joints.accumulated_limit_impulse[i] = glm::min(old_acc + lambda_lim, 0.0f);
                    float delta = joints.accumulated_limit_impulse[i] - old_acc;
                    bodies.angular_velocities[a] -= inv_ia * delta;
                    bodies.angular_velocities[b] += inv_ib * delta;
                    joints.constraint_forces[i] += std::abs(delta);
                }
            }
        }

        // ── Motor ──────────────────────────────────────────────────────
        if (joints.motor_enabled[i]) {
            float angular_speed = bodies.angular_velocities[b] - bodies.angular_velocities[a];
            float inv_i_total = inv_ia + inv_ib;
            if (inv_i_total > 0.0f) {
                float eff_mass = 1.0f / inv_i_total;
                float speed_error = joints.motor_target_speeds[i] - angular_speed;
                float lambda_motor = eff_mass * speed_error;

                float max_impulse = joints.motor_max_torque[i] * dt;
                float old_acc = joints.accumulated_motor_impulse[i];
                joints.accumulated_motor_impulse[i] = glm::clamp(
                    old_acc + lambda_motor, -max_impulse, max_impulse);
                float delta = joints.accumulated_motor_impulse[i] - old_acc;

                bodies.angular_velocities[a] -= inv_ia * delta;
                bodies.angular_velocities[b] += inv_ib * delta;
                joints.constraint_forces[i] += std::abs(delta);
            }
        }
    }
}

} // namespace stan2d
