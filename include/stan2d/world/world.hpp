#pragma once

#include <optional>

#include <stan2d/core/handle.hpp>
#include <stan2d/core/math_types.hpp>
#include <stan2d/core/shape_registry.hpp>
#include <stan2d/core/shapes.hpp>
#include <stan2d/core/sparse_set.hpp>
#include <stan2d/dynamics/body_storage.hpp>
#include <stan2d/world/world_config.hpp>

namespace stan2d {

struct BodyDef {
    Vec2       position   = {0.0f, 0.0f};
    Vec2       velocity   = {0.0f, 0.0f};
    float      rotation   = 0.0f;
    float      angular_velocity = 0.0f;

    // Two-step: provide a pre-created ShapeHandle
    std::optional<ShapeHandle> shape = std::nullopt;

    // One-step: provide inline ShapeData (engine creates shape internally)
    std::optional<ShapeData>   shape_data = std::nullopt;

    float      mass       = 1.0f;
    float      inertia    = 1.0f;
    BodyType   body_type  = BodyType::Dynamic;
};

class World {
public:
    explicit World(const WorldConfig& config);

    // ── Shape management ──────────────────────────────────────────
    ShapeHandle create_shape(const ShapeData& shape);

    // ── Body management ───────────────────────────────────────────
    BodyHandle create_body(const BodyDef& def);
    void       destroy_body(BodyHandle handle);

    // ── Queries ───────────────────────────────────────────────────
    [[nodiscard]] bool is_valid(BodyHandle handle) const;
    [[nodiscard]] Vec2  get_position(BodyHandle handle) const;
    [[nodiscard]] Vec2  get_velocity(BodyHandle handle) const;
    [[nodiscard]] float get_mass(BodyHandle handle) const;
    [[nodiscard]] float get_inverse_mass(BodyHandle handle) const;
    [[nodiscard]] float get_inverse_inertia(BodyHandle handle) const;
    [[nodiscard]] uint32_t body_count() const;

    // ── Gravity ───────────────────────────────────────────────────
    void set_gravity(Vec2 gravity);
    [[nodiscard]] Vec2 get_gravity() const;

    // ── Simulation (placeholder for now) ──────────────────────────
    void step(float dt);

private:
    [[nodiscard]] uint32_t dense_index(BodyHandle handle) const;

    WorldConfig    config_;
    SparseSet      body_handles_;
    BodyStorage    bodies_;
    ShapeRegistry  shape_registry_;
    Vec2           gravity_{0.0f, 0.0f};
};

} // namespace stan2d
