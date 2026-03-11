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
    (void)set.deallocate(h);

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
    (void)set.deallocate(h1);

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

    (void)set.deallocate(h1);
    Handle h2 = set.allocate(); // should reuse the freed slot

    EXPECT_EQ(h2.index, slot);
    EXPECT_EQ(h2.generation, h1.generation + 1);
}

TEST(SparseSet, StaleHandleRejected) {
    SparseSet set;
    set.reserve(100);

    Handle h_old = set.allocate();
    (void)set.deallocate(h_old);
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
    (void)set.deallocate(b);
    EXPECT_EQ(set.dense_index(a), 0u);
    EXPECT_EQ(set.dense_index(d), 1u);
    EXPECT_EQ(set.dense_index(c), 2u);

    // Remove a (dense[0]) — c swaps into dense[0]
    (void)set.deallocate(a);
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
        (void)set.deallocate(h);
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

    (void)set.allocate(); // dense[0]
    (void)set.allocate(); // dense[1]
    (void)set.allocate(); // dense[2]

    (void)set.allocate(); // dense[3]  — but we allocated earlier too
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
