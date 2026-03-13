#include <stan2d/world/world.hpp>

#include <cassert>

#include <glm/glm.hpp>
#include <stan2d/collision/narrow_phase.hpp>
#include <stan2d/constraints/solver.hpp>
#include <stan2d/dynamics/integrator.hpp>

namespace stan2d {

World::World(const WorldConfig& config)
    : config_(config)
{
    body_handles_.reserve(config.max_bodies);
    bodies_.reserve(config.max_bodies);
    shape_registry_.reserve(config.max_shapes);
    joint_handles_.reserve(config.max_joints);
    joints_.reserve(config.max_joints);

    // Pre-allocate pipeline buffers
    body_proxies_.reserve(config.max_bodies);
    collision_pairs_.reserve(config.max_contacts);
    manifold_entries_.reserve(config.max_contacts);
    constraints_.reserve(config.max_contacts * 2);
}

ShapeHandle World::create_shape(const ShapeData& shape) {
    return shape_registry_.create(shape);
}

BodyHandle World::create_body(const BodyDef& def) {
    // Resolve shape handle
    ShapeHandle shape_handle;
    if (def.shape.has_value()) {
        shape_handle = def.shape.value();
    } else if (def.shape_data.has_value()) {
        shape_handle = shape_registry_.create(def.shape_data.value());
    } else {
        assert(false && "BodyDef must provide either 'shape' or 'shape_data'");
    }

    Handle h = body_handles_.allocate();

    // Compute mass properties based on body type
    float mass         = def.mass;
    float inv_mass     = (def.mass > 0.0f) ? (1.0f / def.mass) : 0.0f;
    float inertia      = def.inertia;
    float inv_inertia  = (def.inertia > 0.0f) ? (1.0f / def.inertia) : 0.0f;

    if (def.body_type == BodyType::Static || def.body_type == BodyType::Kinematic) {
        inv_mass    = 0.0f;
        inv_inertia = 0.0f;
    }

    bodies_.push_back(
        def.position, def.velocity, def.rotation, def.angular_velocity,
        mass, inv_mass, inertia, inv_inertia,
        def.body_type, shape_handle
    );

    // Mark that proxies need rebuilding
    proxies_built_ = false;

    return BodyHandle{h.index, h.generation};
}

void World::destroy_body(BodyHandle handle) {
    Handle h{handle.index, handle.generation};
    auto swap = body_handles_.deallocate(h);

    if (swap.has_value()) {
        bodies_.swap_and_pop(swap->removed_dense, swap->moved_from_dense);
    }
    bodies_.pop_back();

    // Mark that proxies need rebuilding
    proxies_built_ = false;
}

bool World::is_valid(BodyHandle handle) const {
    return body_handles_.is_valid(Handle{handle.index, handle.generation});
}

Vec2 World::get_position(BodyHandle handle) const {
    return bodies_.positions[dense_index(handle)];
}

Vec2 World::get_velocity(BodyHandle handle) const {
    return bodies_.velocities[dense_index(handle)];
}

float World::get_mass(BodyHandle handle) const {
    return bodies_.masses[dense_index(handle)];
}

float World::get_inverse_mass(BodyHandle handle) const {
    return bodies_.inverse_masses[dense_index(handle)];
}

float World::get_inverse_inertia(BodyHandle handle) const {
    return bodies_.inverse_inertias[dense_index(handle)];
}

uint32_t World::body_count() const {
    return body_handles_.size();
}

void World::apply_force(BodyHandle handle, Vec2 force) {
    uint32_t idx = dense_index(handle);
    bodies_.forces[idx] = bodies_.forces[idx] + force;
}

void World::apply_torque(BodyHandle handle, float torque) {
    uint32_t idx = dense_index(handle);
    bodies_.torques[idx] = bodies_.torques[idx] + torque;
}

void World::set_gravity(Vec2 gravity) {
    gravity_ = gravity;
}

Vec2 World::get_gravity() const {
    return gravity_;
}

void World::set_solver_config(const SolverConfig& config) {
    solver_config_ = config;
}

const SolverConfig& World::get_solver_config() const {
    return solver_config_;
}

// ── State system ─────────────────────────────────────────────────

WorldStateView World::get_state_view() const {
    uint32_t count = bodies_.size();
    WorldStateView view;
    view.timestamp          = 0.0f;
    view.active_body_count  = count;
    view.positions          = std::span<const Vec2>(bodies_.positions.data(), count);
    view.velocities         = std::span<const Vec2>(bodies_.velocities.data(), count);
    view.rotations          = std::span<const float>(bodies_.rotations.data(), count);
    view.angular_velocities = std::span<const float>(bodies_.angular_velocities.data(), count);
    view.masses             = std::span<const float>(bodies_.masses.data(), count);
    return view;
}

void World::save_state(WorldSnapshot& out) const {
    uint32_t count = bodies_.size();
    out.timestamp  = 0.0f;
    out.body_count = count;
    out.gravity    = gravity_;

    // Copy body SoA data
    out.positions.assign(bodies_.positions.begin(), bodies_.positions.begin() + count);
    out.velocities.assign(bodies_.velocities.begin(), bodies_.velocities.begin() + count);
    out.rotations.assign(bodies_.rotations.begin(), bodies_.rotations.begin() + count);
    out.angular_velocities.assign(bodies_.angular_velocities.begin(),
                                  bodies_.angular_velocities.begin() + count);
    out.masses.assign(bodies_.masses.begin(), bodies_.masses.begin() + count);
    out.inverse_masses.assign(bodies_.inverse_masses.begin(),
                              bodies_.inverse_masses.begin() + count);
    out.inertias.assign(bodies_.inertias.begin(), bodies_.inertias.begin() + count);
    out.inverse_inertias.assign(bodies_.inverse_inertias.begin(),
                                bodies_.inverse_inertias.begin() + count);
    out.forces.assign(bodies_.forces.begin(), bodies_.forces.begin() + count);
    out.torques.assign(bodies_.torques.begin(), bodies_.torques.begin() + count);

    // BodyType → uint8_t
    out.body_types.resize(count);
    for (uint32_t i = 0; i < count; ++i) {
        out.body_types[i] = static_cast<uint8_t>(bodies_.body_types[i]);
    }

    out.shape_ids.assign(bodies_.shape_ids.begin(), bodies_.shape_ids.begin() + count);

    // Save body SparseSet state
    body_handles_.save_state(out.body_sparse,
                             out.body_dense_to_sparse,
                             out.body_generations,
                             out.body_free_list);

    // Save ShapeRegistry state
    shape_registry_.save_state(out.shapes, out.shape_aabbs,
                               out.shape_sparse,
                               out.shape_dense_to_sparse,
                               out.shape_generations,
                               out.shape_free_list);
}

void World::restore_state(const WorldSnapshot& snapshot) {
    uint32_t count = snapshot.body_count;
    gravity_ = snapshot.gravity;

    // Restore body SoA data
    bodies_.positions.assign(snapshot.positions.begin(), snapshot.positions.end());
    bodies_.velocities.assign(snapshot.velocities.begin(), snapshot.velocities.end());
    bodies_.rotations.assign(snapshot.rotations.begin(), snapshot.rotations.end());
    bodies_.angular_velocities.assign(snapshot.angular_velocities.begin(),
                                      snapshot.angular_velocities.end());
    bodies_.masses.assign(snapshot.masses.begin(), snapshot.masses.end());
    bodies_.inverse_masses.assign(snapshot.inverse_masses.begin(),
                                  snapshot.inverse_masses.end());
    bodies_.inertias.assign(snapshot.inertias.begin(), snapshot.inertias.end());
    bodies_.inverse_inertias.assign(snapshot.inverse_inertias.begin(),
                                    snapshot.inverse_inertias.end());
    bodies_.forces.assign(snapshot.forces.begin(), snapshot.forces.end());
    bodies_.torques.assign(snapshot.torques.begin(), snapshot.torques.end());

    // uint8_t → BodyType
    bodies_.body_types.resize(count);
    for (uint32_t i = 0; i < count; ++i) {
        bodies_.body_types[i] = static_cast<BodyType>(snapshot.body_types[i]);
    }

    bodies_.shape_ids.assign(snapshot.shape_ids.begin(), snapshot.shape_ids.end());

    // Restore body SparseSet state
    body_handles_.restore_state(snapshot.body_sparse,
                                snapshot.body_dense_to_sparse,
                                snapshot.body_generations,
                                snapshot.body_free_list);

    // Restore ShapeRegistry state
    shape_registry_.restore_state(snapshot.shapes, snapshot.shape_aabbs,
                                  snapshot.shape_sparse,
                                  snapshot.shape_dense_to_sparse,
                                  snapshot.shape_generations,
                                  snapshot.shape_free_list);

    // Force proxy rebuild on next step
    proxies_built_ = false;
}

// ── Simulation pipeline ──────────────────────────────────────────

void World::step(float dt) {
    if (dt <= 0.0f) return;

    uint32_t count = bodies_.size();
    if (count == 0) return;

    // Stage 1 & 2: Apply forces (gravity) + Integrate velocities
    integrate_velocities(bodies_, count, gravity_, dt);

    // Stage 3: Broad phase — update AABBs and find collision pairs
    broad_phase();

    // Stage 4: Narrow phase — precise collision detection
    narrow_phase();

    // Stage 5: Solve constraints
    solve();

    // Stage 6: Integrate positions
    integrate_positions(bodies_, count, dt);
}

// ── Joint management ─────────────────────────────────────────────

JointHandle World::create_joint(const JointDef& def) {
    Handle ha{def.body_a.index, def.body_a.generation};
    Handle hb{def.body_b.index, def.body_b.generation};
    assert(body_handles_.is_valid(ha) && "body_a is not valid");
    assert(body_handles_.is_valid(hb) && "body_b is not valid");

    uint32_t dense_a = body_handles_.dense_index(ha);
    uint32_t dense_b = body_handles_.dense_index(hb);

    // Compute reference angle for Hinge joints
    float ref_angle = 0.0f;
    if (def.type == JointType::Hinge) {
        ref_angle = bodies_.rotations[dense_b] - bodies_.rotations[dense_a];
    }

    // Auto-detect distances / rest lengths
    JointDef resolved = def;
    Vec2 world_anchor_a = bodies_.positions[dense_a] + def.anchor_a;
    Vec2 world_anchor_b = bodies_.positions[dense_b] + def.anchor_b;
    float current_dist = glm::length(world_anchor_b - world_anchor_a);

    if (def.type == JointType::Distance && def.distance == 0.0f) {
        resolved.distance = current_dist;
    }
    if (def.type == JointType::Spring && def.rest_length == 0.0f) {
        resolved.rest_length = current_dist;
    }

    // Compute pulley constant (len_a + ratio * len_b at creation)
    float pulley_const = 0.0f;
    if (def.type == JointType::Pulley) {
        float len_a = glm::length(world_anchor_a - def.ground_a);
        float len_b = glm::length(world_anchor_b - def.ground_b);
        pulley_const = len_a + def.pulley_ratio * len_b;
    }

    Handle h = joint_handles_.allocate();
    joints_.push_back(resolved, dense_a, dense_b, ref_angle, pulley_const);

    return JointHandle{h.index, h.generation};
}

void World::destroy_joint(JointHandle handle) {
    Handle h{handle.index, handle.generation};
    auto swap = joint_handles_.deallocate(h);

    if (swap.has_value()) {
        joints_.swap_and_pop(swap->removed_dense, swap->moved_from_dense);
    }
    joints_.pop_back();
}

bool World::is_valid(JointHandle handle) const {
    return joint_handles_.is_valid(Handle{handle.index, handle.generation});
}

uint32_t World::joint_count() const {
    return joint_handles_.size();
}

uint32_t World::dense_index(BodyHandle handle) const {
    Handle h{handle.index, handle.generation};
    return body_handles_.dense_index(h);
}

// ── Pipeline stage implementations ────────────────────────────────

void World::build_aabb_proxies() {
    aabb_tree_ = AABBTree{};
    body_proxies_.clear();
    body_proxies_.resize(bodies_.size(), AABBTree::NULL_NODE);

    for (uint32_t i = 0; i < bodies_.size(); ++i) {
        ShapeHandle sh = bodies_.shape_ids[i];
        AABB local_aabb = shape_registry_.get_local_aabb(sh);

        Vec2 pos = bodies_.positions[i];
        AABB world_aabb{
            {local_aabb.min.x + pos.x, local_aabb.min.y + pos.y},
            {local_aabb.max.x + pos.x, local_aabb.max.y + pos.y}
        };

        body_proxies_[i] = aabb_tree_.insert(world_aabb, i);
    }

    proxies_built_ = true;
}

void World::update_aabb_proxies() {
    for (uint32_t i = 0; i < bodies_.size(); ++i) {
        if (bodies_.body_types[i] == BodyType::Static) continue;

        ShapeHandle sh = bodies_.shape_ids[i];
        AABB local_aabb = shape_registry_.get_local_aabb(sh);

        Vec2 pos = bodies_.positions[i];
        AABB world_aabb{
            {local_aabb.min.x + pos.x, local_aabb.min.y + pos.y},
            {local_aabb.max.x + pos.x, local_aabb.max.y + pos.y}
        };

        aabb_tree_.update(body_proxies_[i], world_aabb);
    }
}

void World::broad_phase() {
    if (!proxies_built_) {
        build_aabb_proxies();
    } else {
        update_aabb_proxies();
    }

    collision_pairs_.clear();
    aabb_tree_.query_pairs(collision_pairs_);
}

void World::narrow_phase() {
    manifold_entries_.clear();

    for (const auto& pair : collision_pairs_) {
        uint32_t dense_a = pair.user_data_a;
        uint32_t dense_b = pair.user_data_b;

        // Skip static-static pairs
        if (bodies_.body_types[dense_a] == BodyType::Static &&
            bodies_.body_types[dense_b] == BodyType::Static) {
            continue;
        }

        ShapeHandle shape_a = bodies_.shape_ids[dense_a];
        ShapeHandle shape_b = bodies_.shape_ids[dense_b];

        const ShapeData& data_a = shape_registry_.get(shape_a);
        const ShapeData& data_b = shape_registry_.get(shape_b);

        ContactManifold manifold;
        bool colliding = collide_shapes(
            data_a, bodies_.positions[dense_a], bodies_.rotations[dense_a],
            data_b, bodies_.positions[dense_b], bodies_.rotations[dense_b],
            manifold
        );

        if (colliding && manifold.point_count > 0) {
            manifold_entries_.push_back({manifold, dense_a, dense_b});
        }
    }
}

void World::solve() {
    constraints_.clear();

    for (const auto& entry : manifold_entries_) {
        prepare_contact_constraints(
            entry.manifold, entry.dense_a, entry.dense_b,
            bodies_, constraints_
        );
    }

    if (constraints_.empty()) return;

    warm_start(constraints_, bodies_);
    solve_constraints(constraints_, bodies_, solver_config_);
}

} // namespace stan2d
