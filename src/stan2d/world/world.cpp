#include <stan2d/world/world.hpp>

#include <cassert>

namespace stan2d {

World::World(const WorldConfig& config)
    : config_(config)
{
    body_handles_.reserve(config.max_bodies);
    bodies_.reserve(config.max_bodies);
    shape_registry_.reserve(config.max_shapes);
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

    return BodyHandle{h.index, h.generation};
}

void World::destroy_body(BodyHandle handle) {
    Handle h{handle.index, handle.generation};
    auto swap = body_handles_.deallocate(h);

    if (swap.has_value()) {
        bodies_.swap_and_pop(swap->removed_dense, swap->moved_from_dense);
    }
    bodies_.pop_back();
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

void World::set_gravity(Vec2 gravity) {
    gravity_ = gravity;
}

Vec2 World::get_gravity() const {
    return gravity_;
}

void World::step(float /*dt*/) {
    // Placeholder — will be implemented in Task 10
}

uint32_t World::dense_index(BodyHandle handle) const {
    Handle h{handle.index, handle.generation};
    return body_handles_.dense_index(h);
}

} // namespace stan2d
