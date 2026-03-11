#pragma once

#include <cassert>
#include <vector>

#include <stan2d/core/handle.hpp>
#include <stan2d/core/math_types.hpp>
#include <stan2d/core/shapes.hpp>
#include <stan2d/core/sparse_set.hpp>

namespace stan2d {

class ShapeRegistry {
public:
    void reserve(uint32_t capacity) {
        handles_.reserve(capacity);
        shapes_.reserve(capacity);
        local_aabbs_.reserve(capacity);
    }

    [[nodiscard]] ShapeHandle create(const ShapeData& shape) {
        Handle h = handles_.allocate();

        shapes_.push_back(shape);
        local_aabbs_.push_back(compute_local_aabb(shape));

        return ShapeHandle{h.index, h.generation};
    }

    void destroy(ShapeHandle handle) {
        Handle h{handle.index, handle.generation};
        auto swap = handles_.deallocate(h);

        if (swap.has_value()) {
            uint32_t dst = swap->removed_dense;
            uint32_t src = swap->moved_from_dense;
            shapes_[dst]      = shapes_[src];
            local_aabbs_[dst] = local_aabbs_[src];
        }

        shapes_.pop_back();
        local_aabbs_.pop_back();
    }

    [[nodiscard]] bool is_valid(ShapeHandle handle) const {
        return handles_.is_valid(Handle{handle.index, handle.generation});
    }

    [[nodiscard]] const ShapeData& get(ShapeHandle handle) const {
        Handle h{handle.index, handle.generation};
        return shapes_[handles_.dense_index(h)];
    }

    [[nodiscard]] const AABB& get_local_aabb(ShapeHandle handle) const {
        Handle h{handle.index, handle.generation};
        return local_aabbs_[handles_.dense_index(h)];
    }

    [[nodiscard]] uint32_t size() const { return handles_.size(); }

    // Accessors for state backup/restore
    [[nodiscard]] const SparseSet&             handles()     const { return handles_; }
    [[nodiscard]] const std::vector<ShapeData>& shapes()     const { return shapes_; }
    [[nodiscard]] const std::vector<AABB>&      local_aabbs() const { return local_aabbs_; }

    void save_state(std::vector<ShapeData>& out_shapes,
                    std::vector<AABB>& out_aabbs,
                    std::vector<uint32_t>& out_sparse,
                    std::vector<uint32_t>& out_dense_to_sparse,
                    std::vector<uint32_t>& out_generations,
                    std::vector<uint32_t>& out_free_list) const {
        out_shapes.assign(shapes_.begin(), shapes_.end());
        out_aabbs.assign(local_aabbs_.begin(), local_aabbs_.end());
        handles_.save_state(out_sparse, out_dense_to_sparse, out_generations, out_free_list);
    }

    void restore_state(const std::vector<ShapeData>& in_shapes,
                       const std::vector<AABB>& in_aabbs,
                       const std::vector<uint32_t>& in_sparse,
                       const std::vector<uint32_t>& in_dense_to_sparse,
                       const std::vector<uint32_t>& in_generations,
                       const std::vector<uint32_t>& in_free_list) {
        shapes_.assign(in_shapes.begin(), in_shapes.end());
        local_aabbs_.assign(in_aabbs.begin(), in_aabbs.end());
        handles_.restore_state(in_sparse, in_dense_to_sparse, in_generations, in_free_list);
    }

private:
    SparseSet              handles_;
    std::vector<ShapeData> shapes_;
    std::vector<AABB>      local_aabbs_;
};

} // namespace stan2d
