> **Note for Claude**: 牢记 `docs/tasks/00_master_plan.md` 中的四大铁律。

## Task 3: SparseSet

**Goal:** Generic SparseSet data structure with generation-based handle validation, swap-and-pop deletion, and free list recycling.

**Files:**
- Create: `include/stan2d/core/handle.hpp`
- Create: `include/stan2d/core/sparse_set.hpp`
- Create: `tests/unit/test_sparse_set.cpp`

**Depends on:** Task 1

### Step 1: Write failing tests (RED)

**File:** `tests/unit/test_sparse_set.cpp`

```cpp
#include <gtest/gtest.h>
#include <stan2d/core/sparse_set.hpp>

using namespace stan2d;

// ── Allocation ────────────────────────────────────────────────────

TEST(SparseSet, AllocateReturnsValidHandle) {
    SparseSet set;
    set.reserve(100);

    Handle h = set.allocate();
    EXPECT_TRUE(set.is_valid(h));
}

TEST(SparseSet, AllocateSequentialIndices) {
    SparseSet set;
    set.reserve(100);

    Handle h0 = set.allocate();
    Handle h1 = set.allocate();
    Handle h2 = set.allocate();

    EXPECT_EQ(set.dense_index(h0), 0u);
    EXPECT_EQ(set.dense_index(h1), 1u);
    EXPECT_EQ(set.dense_index(h2), 2u);
    EXPECT_EQ(set.size(), 3u);
}

TEST(SparseSet, InitialGenerationIsOne) {
    SparseSet set;
    set.reserve(100);

    Handle h = set.allocate();
    EXPECT_EQ(h.generation, 1u);
}

// ── Deallocation ──────────────────────────────────────────────────

TEST(SparseSet, DeallocateInvalidatesHandle) {
    SparseSet set;
    set.reserve(100);

    Handle h = set.allocate();
    set.deallocate(h);

    EXPECT_FALSE(set.is_valid(h));
    EXPECT_EQ(set.size(), 0u);
}

TEST(SparseSet, SwapAndPopMaintainsContiguity) {
    SparseSet set;
    set.reserve(100);

    Handle h0 = set.allocate(); // dense[0]
    Handle h1 = set.allocate(); // dense[1]
    Handle h2 = set.allocate(); // dense[2]

    // Delete middle element — h2 should swap into dense[1]
    set.deallocate(h1);

    EXPECT_EQ(set.size(), 2u);
    EXPECT_TRUE(set.is_valid(h0));
    EXPECT_FALSE(set.is_valid(h1));
    EXPECT_TRUE(set.is_valid(h2));

    // h0 stays at dense[0], h2 moved to dense[1]
    EXPECT_EQ(set.dense_index(h0), 0u);
    EXPECT_EQ(set.dense_index(h2), 1u);
}

// ── Generation & Free List ────────────────────────────────────────

TEST(SparseSet, ReuseSlotIncrementsGeneration) {
    SparseSet set;
    set.reserve(100);

    Handle h1 = set.allocate();
    uint32_t slot = h1.index;

    set.deallocate(h1);
    Handle h2 = set.allocate(); // should reuse the freed slot

    EXPECT_EQ(h2.index, slot);
    EXPECT_EQ(h2.generation, h1.generation + 1);
}

TEST(SparseSet, StaleHandleRejected) {
    SparseSet set;
    set.reserve(100);

    Handle h_old = set.allocate();
    set.deallocate(h_old);
    Handle h_new = set.allocate(); // reuses same slot

    EXPECT_FALSE(set.is_valid(h_old)); // stale: old generation
    EXPECT_TRUE(set.is_valid(h_new));  // current generation
}

// ── Dense index query ─────────────────────────────────────────────

TEST(SparseSet, DenseIndexAfterMultipleOperations) {
    SparseSet set;
    set.reserve(100);

    Handle a = set.allocate();
    Handle b = set.allocate();
    Handle c = set.allocate();
    Handle d = set.allocate();

    // Remove b (dense[1]) — d swaps into dense[1]
    set.deallocate(b);
    EXPECT_EQ(set.dense_index(a), 0u);
    EXPECT_EQ(set.dense_index(d), 1u);
    EXPECT_EQ(set.dense_index(c), 2u);

    // Remove a (dense[0]) — c swaps into dense[0]
    set.deallocate(a);
    EXPECT_EQ(set.dense_index(c), 0u);
    EXPECT_EQ(set.dense_index(d), 1u);
    EXPECT_EQ(set.size(), 2u);
}

// ── Bulk operations ───────────────────────────────────────────────

TEST(SparseSet, AllocateManyThenDeallocateAll) {
    SparseSet set;
    set.reserve(1000);

    std::vector<Handle> handles;
    for (int i = 0; i < 100; ++i) {
        handles.push_back(set.allocate());
    }
    EXPECT_EQ(set.size(), 100u);

    for (auto& h : handles) {
        set.deallocate(h);
    }
    EXPECT_EQ(set.size(), 0u);

    // Re-allocate — all should reuse freed slots with bumped generation
    for (int i = 0; i < 100; ++i) {
        Handle h = set.allocate();
        EXPECT_EQ(h.generation, 2u);
    }
    EXPECT_EQ(set.size(), 100u);
}

// ── Swap callback ─────────────────────────────────────────────────

TEST(SparseSet, DeallocateReportsSwapIndices) {
    SparseSet set;
    set.reserve(100);

    set.allocate(); // dense[0]
    set.allocate(); // dense[1]
    set.allocate(); // dense[2]

    Handle h0 = set.allocate(); // dense[3]  — but we allocated earlier too
    // Let's redo cleanly:
    SparseSet set2;
    set2.reserve(100);
    Handle a = set2.allocate(); // dense[0]
    Handle b = set2.allocate(); // dense[1]
    Handle c = set2.allocate(); // dense[2]

    // Deallocate b → c swaps from dense[2] to dense[1]
    auto swap_info = set2.deallocate(b);
    EXPECT_TRUE(swap_info.has_value());
    EXPECT_EQ(swap_info->removed_dense, 1u);
    EXPECT_EQ(swap_info->moved_dense, 1u);   // c now lives at dense[1]
    EXPECT_EQ(swap_info->moved_from_dense, 2u);
}

TEST(SparseSet, DeallocateLastElementNoSwap) {
    SparseSet set;
    set.reserve(100);
    Handle a = set.allocate();
    Handle b = set.allocate();

    // Deallocate b (last element) — no swap needed, just pop
    auto swap_info = set.deallocate(b);
    EXPECT_FALSE(swap_info.has_value());
}
```

### Step 2: Run tests to verify they fail

Run: `cmake --build build && ctest --test-dir build --output-on-failure`

Expected: FAIL — `fatal error: 'stan2d/core/sparse_set.hpp' file not found`

### Step 3: Implement Handle type

**File:** `include/stan2d/core/handle.hpp`

```cpp
#pragma once

#include <cstdint>
#include <functional>

namespace stan2d {

struct Handle {
    uint32_t index      = 0;
    uint32_t generation = 0;

    bool operator==(const Handle&) const = default;
    bool operator!=(const Handle&) const = default;
};

// Typed handles — distinct types prevent accidental mixing
struct BodyHandle {
    uint32_t index      = 0;
    uint32_t generation = 0;

    bool operator==(const BodyHandle&) const = default;
    bool operator!=(const BodyHandle&) const = default;
};

struct ShapeHandle {
    uint32_t index      = 0;
    uint32_t generation = 0;

    bool operator==(const ShapeHandle&) const = default;
    bool operator!=(const ShapeHandle&) const = default;
};

} // namespace stan2d
```

### Step 4: Implement SparseSet (GREEN)

**File:** `include/stan2d/core/sparse_set.hpp`

```cpp
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
            sparse_index = free_list_.back();
            free_list_.pop_back();
        } else {
            sparse_index = static_cast<uint32_t>(sparse_.size());
            sparse_.push_back(0);
            generations_.push_back(0); // will be incremented below
        }

        uint32_t dense_index = static_cast<uint32_t>(dense_to_sparse_.size());
        sparse_[sparse_index] = dense_index;
        dense_to_sparse_.push_back(sparse_index);
        generations_[sparse_index] += 1;

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
```

### Step 5: Run tests to verify they pass

Run: `cmake --build build && ctest --test-dir build --output-on-failure`

Expected: PASS — all SparseSet tests green

### Step 6: Commit

```bash
git add include/stan2d/core/handle.hpp include/stan2d/core/sparse_set.hpp tests/unit/test_sparse_set.cpp
git commit -m "feat: SparseSet with generation handles, swap-and-pop, and free list"
```

---