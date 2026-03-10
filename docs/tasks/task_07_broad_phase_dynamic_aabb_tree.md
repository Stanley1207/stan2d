> **Note for Claude**: 牢记 `docs/tasks/00_master_plan.md` 中的四大铁律。

## Task 7: Broad Phase (Dynamic AABB Tree)

**Goal:** Dynamic AABB tree for spatial partitioning. Insert/remove bodies, query overlapping pairs. Fattened AABBs reduce re-insertions. Tree nodes store enlarged AABBs; leaf nodes map to body handles.

**Files:**
- Create: `include/stan2d/collision/aabb_tree.hpp`
- Create: `tests/unit/test_aabb_tree.cpp`

**Depends on:** Task 2

### Step 1: Write failing tests (RED)

**File:** `tests/unit/test_aabb_tree.cpp`

```cpp
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
```

### Step 2: Run tests to verify they fail

Run: `cmake --build build && ctest --test-dir build --output-on-failure`

Expected: FAIL — `fatal error: 'stan2d/collision/aabb_tree.hpp' file not found`

### Step 3: Implement (GREEN)

**File:** `include/stan2d/collision/aabb_tree.hpp`

```cpp
#pragma once

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <vector>

#include <stan2d/core/math_types.hpp>

namespace stan2d {

inline constexpr float AABB_FATTEN_MARGIN = 0.1f;

struct CollisionPair {
    uint32_t user_data_a;
    uint32_t user_data_b;
};

class AABBTree {
public:
    static constexpr int32_t NULL_NODE = -1;

    int32_t insert(const AABB& tight_aabb, uint32_t user_data) {
        int32_t leaf = allocate_node();
        nodes_[leaf].aabb = aabb_expand(tight_aabb, AABB_FATTEN_MARGIN);
        nodes_[leaf].user_data = user_data;
        nodes_[leaf].is_leaf = true;

        if (root_ == NULL_NODE) {
            root_ = leaf;
            return leaf;
        }

        insert_leaf(leaf);
        return leaf;
    }

    void remove(int32_t proxy) {
        assert(proxy >= 0 && proxy < static_cast<int32_t>(nodes_.size()));
        assert(nodes_[proxy].is_leaf);

        remove_leaf(proxy);
        free_node(proxy);
    }

    void update(int32_t proxy, const AABB& new_tight_aabb) {
        assert(proxy >= 0 && proxy < static_cast<int32_t>(nodes_.size()));
        assert(nodes_[proxy].is_leaf);

        AABB fattened = aabb_expand(new_tight_aabb, AABB_FATTEN_MARGIN);

        // If the new AABB is still inside the old fattened AABB, skip
        if (aabb_contains_aabb(nodes_[proxy].aabb, new_tight_aabb)) {
            return;
        }

        remove_leaf(proxy);
        nodes_[proxy].aabb = fattened;
        insert_leaf(proxy);
    }

    void query_pairs(std::vector<CollisionPair>& out) const {
        out.clear();
        if (root_ == NULL_NODE) return;

        // Collect all leaf nodes
        std::vector<int32_t> leaves;
        collect_leaves(root_, leaves);

        // Brute-force leaf-vs-leaf on fattened AABBs
        // For small N this is fine; for large N the tree structure
        // enables query(aabb, callback) per-leaf against the tree
        for (size_t i = 0; i < leaves.size(); ++i) {
            for (size_t j = i + 1; j < leaves.size(); ++j) {
                const auto& a = nodes_[leaves[i]];
                const auto& b = nodes_[leaves[j]];
                if (aabb_overlaps(a.aabb, b.aabb)) {
                    uint32_t ua = a.user_data;
                    uint32_t ub = b.user_data;
                    if (ua > ub) std::swap(ua, ub);
                    out.push_back(CollisionPair{ua, ub});
                }
            }
        }
    }

    [[nodiscard]] const AABB& get_aabb(int32_t proxy) const {
        assert(proxy >= 0 && proxy < static_cast<int32_t>(nodes_.size()));
        return nodes_[proxy].aabb;
    }

private:
    struct Node {
        AABB     aabb{};
        uint32_t user_data = 0;
        int32_t  parent    = NULL_NODE;
        int32_t  left      = NULL_NODE;
        int32_t  right     = NULL_NODE;
        int32_t  height    = 0;
        bool     is_leaf   = false;
    };

    int32_t root_ = NULL_NODE;
    std::vector<Node> nodes_;
    std::vector<int32_t> free_list_;

    static bool aabb_contains_aabb(const AABB& outer, const AABB& inner) {
        return outer.min.x <= inner.min.x && outer.min.y <= inner.min.y
            && outer.max.x >= inner.max.x && outer.max.y >= inner.max.y;
    }

    int32_t allocate_node() {
        if (!free_list_.empty()) {
            int32_t id = free_list_.back();
            free_list_.pop_back();
            nodes_[id] = Node{};
            return id;
        }
        int32_t id = static_cast<int32_t>(nodes_.size());
        nodes_.push_back(Node{});
        return id;
    }

    void free_node(int32_t id) {
        nodes_[id] = Node{};
        free_list_.push_back(id);
    }

    void insert_leaf(int32_t leaf) {
        if (root_ == NULL_NODE) {
            root_ = leaf;
            nodes_[leaf].parent = NULL_NODE;
            return;
        }

        // Find best sibling using surface area heuristic
        int32_t sibling = find_best_sibling(leaf);

        // Create new parent
        int32_t old_parent = nodes_[sibling].parent;
        int32_t new_parent = allocate_node();
        nodes_[new_parent].parent = old_parent;
        nodes_[new_parent].aabb = aabb_merge(nodes_[leaf].aabb, nodes_[sibling].aabb);
        nodes_[new_parent].height = nodes_[sibling].height + 1;
        nodes_[new_parent].is_leaf = false;

        if (old_parent != NULL_NODE) {
            if (nodes_[old_parent].left == sibling) {
                nodes_[old_parent].left = new_parent;
            } else {
                nodes_[old_parent].right = new_parent;
            }
        } else {
            root_ = new_parent;
        }

        nodes_[new_parent].left = sibling;
        nodes_[new_parent].right = leaf;
        nodes_[sibling].parent = new_parent;
        nodes_[leaf].parent = new_parent;

        // Walk up and refit AABBs
        refit(nodes_[leaf].parent);
    }

    void remove_leaf(int32_t leaf) {
        if (leaf == root_) {
            root_ = NULL_NODE;
            return;
        }

        int32_t parent = nodes_[leaf].parent;
        int32_t grandparent = nodes_[parent].parent;
        int32_t sibling = (nodes_[parent].left == leaf)
                            ? nodes_[parent].right
                            : nodes_[parent].left;

        if (grandparent != NULL_NODE) {
            if (nodes_[grandparent].left == parent) {
                nodes_[grandparent].left = sibling;
            } else {
                nodes_[grandparent].right = sibling;
            }
            nodes_[sibling].parent = grandparent;
            free_node(parent);
            refit(grandparent);
        } else {
            root_ = sibling;
            nodes_[sibling].parent = NULL_NODE;
            free_node(parent);
        }

        nodes_[leaf].parent = NULL_NODE;
    }

    int32_t find_best_sibling(int32_t leaf) const {
        // Simple: walk tree greedily choosing child that minimizes combined perimeter
        int32_t current = root_;
        while (!nodes_[current].is_leaf) {
            int32_t left  = nodes_[current].left;
            int32_t right = nodes_[current].right;

            float area_combined_left  = aabb_perimeter(aabb_merge(nodes_[left].aabb, nodes_[leaf].aabb));
            float area_combined_right = aabb_perimeter(aabb_merge(nodes_[right].aabb, nodes_[leaf].aabb));

            if (area_combined_left < area_combined_right) {
                current = left;
            } else {
                current = right;
            }
        }
        return current;
    }

    void refit(int32_t node) {
        while (node != NULL_NODE) {
            int32_t left  = nodes_[node].left;
            int32_t right = nodes_[node].right;

            nodes_[node].aabb = aabb_merge(nodes_[left].aabb, nodes_[right].aabb);
            nodes_[node].height = 1 + std::max(nodes_[left].height, nodes_[right].height);

            node = nodes_[node].parent;
        }
    }

    void collect_leaves(int32_t node, std::vector<int32_t>& out) const {
        if (node == NULL_NODE) return;
        if (nodes_[node].is_leaf) {
            out.push_back(node);
            return;
        }
        collect_leaves(nodes_[node].left, out);
        collect_leaves(nodes_[node].right, out);
    }
};

} // namespace stan2d
```

### Step 4: Run tests to verify they pass

Run: `cmake --build build && ctest --test-dir build --output-on-failure`

Expected: PASS — all AABBTree tests green

### Step 5: Commit

```bash
git add include/stan2d/collision/aabb_tree.hpp tests/unit/test_aabb_tree.cpp
git commit -m "feat: Dynamic AABB tree broad phase with insert/remove/update/query_pairs"
```

---