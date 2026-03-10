#include <gtest/gtest.h>
#include <stan2d/collision/aabb_tree.hpp>

using namespace stan2d;

// ── Insertion ─────────────────────────────────────────────────────

TEST(AABBTree, InsertSingleNode) {
    AABBTree tree;
    int32_t proxy = tree.insert(AABB{{0.0f, 0.0f}, {1.0f, 1.0f}}, 0);
    EXPECT_GE(proxy, 0);
}

TEST(AABBTree, InsertMultipleNodes) {
    AABBTree tree;
    int32_t a = tree.insert(AABB{{0.0f, 0.0f}, {1.0f, 1.0f}}, 0);
    int32_t b = tree.insert(AABB{{2.0f, 2.0f}, {3.0f, 3.0f}}, 1);
    int32_t c = tree.insert(AABB{{0.5f, 0.5f}, {1.5f, 1.5f}}, 2);
    EXPECT_NE(a, b);
    EXPECT_NE(b, c);
}

// ── Removal ───────────────────────────────────────────────────────

TEST(AABBTree, RemoveNode) {
    AABBTree tree;
    int32_t a = tree.insert(AABB{{0.0f, 0.0f}, {1.0f, 1.0f}}, 0);
    int32_t b = tree.insert(AABB{{2.0f, 2.0f}, {3.0f, 3.0f}}, 1);

    tree.remove(a);

    // b should still be queryable
    std::vector<CollisionPair> pairs;
    tree.query_pairs(pairs);
    // With only one node left, no pairs
    EXPECT_TRUE(pairs.empty());
}

TEST(AABBTree, RemoveAllNodes) {
    AABBTree tree;
    int32_t a = tree.insert(AABB{{0.0f, 0.0f}, {1.0f, 1.0f}}, 0);
    int32_t b = tree.insert(AABB{{0.5f, 0.5f}, {1.5f, 1.5f}}, 1);

    tree.remove(a);
    tree.remove(b);

    std::vector<CollisionPair> pairs;
    tree.query_pairs(pairs);
    EXPECT_TRUE(pairs.empty());
}

// ── Pair query ────────────────────────────────────────────────────

TEST(AABBTree, OverlappingPairsDetected) {
    AABBTree tree;
    tree.insert(AABB{{0.0f, 0.0f}, {2.0f, 2.0f}}, 0);
    tree.insert(AABB{{1.0f, 1.0f}, {3.0f, 3.0f}}, 1);

    std::vector<CollisionPair> pairs;
    tree.query_pairs(pairs);

    EXPECT_EQ(pairs.size(), 1u);
    // Order: smaller user_data first
    EXPECT_EQ(pairs[0].user_data_a, 0u);
    EXPECT_EQ(pairs[0].user_data_b, 1u);
}

TEST(AABBTree, NonOverlappingPairsNotReported) {
    AABBTree tree;
    tree.insert(AABB{{0.0f, 0.0f}, {1.0f, 1.0f}}, 0);
    tree.insert(AABB{{5.0f, 5.0f}, {6.0f, 6.0f}}, 1);

    std::vector<CollisionPair> pairs;
    tree.query_pairs(pairs);

    EXPECT_TRUE(pairs.empty());
}

TEST(AABBTree, ThreeOverlappingBodies) {
    AABBTree tree;
    // All three overlap with each other
    tree.insert(AABB{{0.0f, 0.0f}, {3.0f, 3.0f}}, 0);
    tree.insert(AABB{{1.0f, 1.0f}, {4.0f, 4.0f}}, 1);
    tree.insert(AABB{{2.0f, 2.0f}, {5.0f, 5.0f}}, 2);

    std::vector<CollisionPair> pairs;
    tree.query_pairs(pairs);

    // Expect 3 pairs: (0,1), (0,2), (1,2)
    EXPECT_EQ(pairs.size(), 3u);
}

TEST(AABBTree, OnlyPartialOverlap) {
    AABBTree tree;
    // A overlaps B, B overlaps C, but A does NOT overlap C
    tree.insert(AABB{{0.0f, 0.0f}, {2.0f, 2.0f}}, 0);  // A
    tree.insert(AABB{{1.5f, 0.0f}, {3.5f, 2.0f}}, 1);   // B
    tree.insert(AABB{{3.0f, 0.0f}, {5.0f, 2.0f}}, 2);    // C

    std::vector<CollisionPair> pairs;
    tree.query_pairs(pairs);

    // A-B overlap, B-C overlap, A-C do NOT (gap at x: A.max=2 < C.min=3)
    EXPECT_EQ(pairs.size(), 2u);
}

// ── Update (move) ─────────────────────────────────────────────────

TEST(AABBTree, UpdateMovedNodeDetectsNewOverlap) {
    AABBTree tree;
    int32_t a = tree.insert(AABB{{0.0f, 0.0f}, {1.0f, 1.0f}}, 0);
    int32_t b = tree.insert(AABB{{5.0f, 5.0f}, {6.0f, 6.0f}}, 1);

    // Initially no overlap
    std::vector<CollisionPair> pairs;
    tree.query_pairs(pairs);
    EXPECT_TRUE(pairs.empty());

    // Move A to overlap with B
    tree.update(a, AABB{{4.5f, 4.5f}, {5.5f, 5.5f}});

    pairs.clear();
    tree.query_pairs(pairs);
    EXPECT_EQ(pairs.size(), 1u);
}

// ── Get AABB ──────────────────────────────────────────────────────

TEST(AABBTree, GetFattenedAABB) {
    AABBTree tree;
    AABB tight{{1.0f, 1.0f}, {2.0f, 2.0f}};
    int32_t proxy = tree.insert(tight, 0);

    const AABB& fat = tree.get_aabb(proxy);
    // Fattened AABB should be at least as large as tight
    EXPECT_LE(fat.min.x, tight.min.x);
    EXPECT_LE(fat.min.y, tight.min.y);
    EXPECT_GE(fat.max.x, tight.max.x);
    EXPECT_GE(fat.max.y, tight.max.y);
}
