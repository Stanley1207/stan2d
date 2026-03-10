#pragma once

#include <cassert>
#include <cstdint>
#include <optional>
#include <vector>

#include <stan2d/core/handle.hpp>

namespace stan2d {

struct SwapInfo {
    uint32_t removed_dense;    // dense index that was removed
    uint32_t moved_dense;      // dense index where the moved element now lives (== removed_dense)
    uint32_t moved_from_dense; // dense index the moved element came from
};

class SparseSet {
public:
    void reserve(uint32_t capacity) {
        sparse_.reserve(capacity);
        dense_to_sparse_.reserve(capacity);
        generations_.reserve(capacity);
    }

    [[nodiscard]] Handle allocate() {
        uint32_t sparse_index;

        if (!free_list_.empty()) {
            // Reuse freed slot — generation was already bumped by deallocate
            sparse_index = free_list_.back();
            free_list_.pop_back();
        } else {
            // Fresh slot — start at generation 1
            sparse_index = static_cast<uint32_t>(sparse_.size());
            sparse_.push_back(0);
            generations_.push_back(1);
        }

        uint32_t dense_index = static_cast<uint32_t>(dense_to_sparse_.size());
        sparse_[sparse_index] = dense_index;
        dense_to_sparse_.push_back(sparse_index);

        return Handle{sparse_index, generations_[sparse_index]};
    }

    [[nodiscard]] std::optional<SwapInfo> deallocate(Handle handle) {
        assert(is_valid(handle) && "Attempted to deallocate invalid handle");

        uint32_t dense_idx = sparse_[handle.index];
        uint32_t last_dense = static_cast<uint32_t>(dense_to_sparse_.size()) - 1;

        std::optional<SwapInfo> result;

        if (dense_idx != last_dense) {
            // Swap-and-pop: move last element into the removed slot
            uint32_t last_sparse = dense_to_sparse_[last_dense];

            dense_to_sparse_[dense_idx] = last_sparse;
            sparse_[last_sparse] = dense_idx;

            result = SwapInfo{
                .removed_dense    = dense_idx,
                .moved_dense      = dense_idx,
                .moved_from_dense = last_dense
            };
        }

        dense_to_sparse_.pop_back();

        // Bump generation to invalidate all existing handles to this slot
        generations_[handle.index] += 1;
        free_list_.push_back(handle.index);

        return result;
    }

    [[nodiscard]] bool is_valid(Handle handle) const {
        return handle.index < generations_.size()
            && generations_[handle.index] == handle.generation
            && sparse_[handle.index] < dense_to_sparse_.size();
    }

    [[nodiscard]] uint32_t dense_index(Handle handle) const {
        assert(is_valid(handle) && "Handle is not valid");
        return sparse_[handle.index];
    }

    [[nodiscard]] uint32_t size() const {
        return static_cast<uint32_t>(dense_to_sparse_.size());
    }

    // Accessors for state backup/restore
    [[nodiscard]] const std::vector<uint32_t>& sparse()          const { return sparse_; }
    [[nodiscard]] const std::vector<uint32_t>& dense_to_sparse() const { return dense_to_sparse_; }
    [[nodiscard]] const std::vector<uint32_t>& generations()     const { return generations_; }
    [[nodiscard]] const std::vector<uint32_t>& free_list()       const { return free_list_; }

private:
    std::vector<uint32_t> sparse_;          // sparse_index → dense_index
    std::vector<uint32_t> dense_to_sparse_; // dense_index  → sparse_index
    std::vector<uint32_t> generations_;     // per sparse slot
    std::vector<uint32_t> free_list_;       // recycled sparse slots (LIFO)
};

} // namespace stan2d
