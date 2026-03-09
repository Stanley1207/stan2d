# Phase 1 Implementation Plan — stan2d 2D Physics Engine

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a complete 2D rigid body physics engine with collision detection, constraint solving, state management, and debug rendering.

**Architecture:** Handle + SoA Hybrid — users interact via lightweight Handles (index + generation), engine stores data in Structure of Arrays for cache efficiency. SparseSet with swap-and-pop maintains contiguous dense arrays. Physics pipeline: Apply Forces → Integrate Velocities → Broad Phase → Narrow Phase → Solve Constraints → Integrate Positions → Post Step.

**Tech Stack:** C++20, CMake, vcpkg, Google Test, glm, nlohmann-json, SDL2

**Reference:** `docs/plans/2026-03-09-2d-physics-engine-design.md`

---

## Task 1: Project Scaffolding

**Goal:** Set up CMake + vcpkg build system with C++20, determinism compiler flags, and a passing smoke test.

**Files:**
- Create: `CMakeLists.txt`
- Create: `vcpkg.json`
- Create: `src/stan2d/core/.gitkeep`
- Create: `src/stan2d/dynamics/.gitkeep`
- Create: `src/stan2d/collision/.gitkeep`
- Create: `src/stan2d/constraints/.gitkeep`
- Create: `src/stan2d/world/.gitkeep`
- Create: `src/stan2d/export/.gitkeep`
- Create: `src/debug_renderer/.gitkeep`
- Create: `include/stan2d/core/.gitkeep`
- Create: `include/stan2d/dynamics/.gitkeep`
- Create: `include/stan2d/collision/.gitkeep`
- Create: `include/stan2d/constraints/.gitkeep`
- Create: `include/stan2d/world/.gitkeep`
- Create: `include/stan2d/export/.gitkeep`
- Create: `tests/unit/.gitkeep`
- Create: `tests/integration/.gitkeep`
- Create: `examples/.gitkeep`
- Create: `tests/unit/test_smoke.cpp`

**Depends on:** Nothing

### Step 1: Create vcpkg.json

**File:** `vcpkg.json`

```json
{
  "name": "stan2d",
  "version-string": "0.1.0",
  "dependencies": [
    "glm",
    "nlohmann-json",
    "gtest",
    "sdl2"
  ]
}
```

### Step 2: Create root CMakeLists.txt

**File:** `CMakeLists.txt`

```cmake
cmake_minimum_required(VERSION 3.24)
project(stan2d VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ── Dependencies via vcpkg ──────────────────────────────────────────
find_package(glm CONFIG REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)
find_package(GTest CONFIG REQUIRED)

# SDL2 is optional (debug renderer only)
find_package(SDL2 CONFIG QUIET)

# ── Core engine library ────────────────────────────────────────────
file(GLOB_RECURSE STAN2D_SOURCES "src/stan2d/*.cpp")

add_library(stan2d STATIC ${STAN2D_SOURCES})

target_include_directories(stan2d
    PUBLIC  include
    PRIVATE src
)

target_link_libraries(stan2d
    PUBLIC  glm::glm
    PRIVATE nlohmann_json::nlohmann_json
)

# Determinism: disable fast-math and FP contraction
target_compile_options(stan2d PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:
        -fno-fast-math
        -ffp-contract=off
    >
    $<$<CXX_COMPILER_ID:MSVC>:
        /fp:precise
    >
)

# ── Debug renderer (optional) ──────────────────────────────────────
if(SDL2_FOUND)
    file(GLOB_RECURSE DEBUG_RENDERER_SOURCES "src/debug_renderer/*.cpp")
    if(DEBUG_RENDERER_SOURCES)
        add_library(stan2d_debug_renderer STATIC ${DEBUG_RENDERER_SOURCES})
        target_include_directories(stan2d_debug_renderer PUBLIC include)
        target_link_libraries(stan2d_debug_renderer
            PUBLIC  stan2d
            PRIVATE $<IF:$<TARGET_EXISTS:SDL2::SDL2>,SDL2::SDL2,SDL2::SDL2-static>
        )
    endif()
endif()

# ── Tests ──────────────────────────────────────────────────────────
enable_testing()

file(GLOB_RECURSE TEST_SOURCES "tests/unit/*.cpp" "tests/integration/*.cpp")

if(TEST_SOURCES)
    add_executable(stan2d_tests ${TEST_SOURCES})
    target_link_libraries(stan2d_tests
        PRIVATE stan2d GTest::gtest GTest::gtest_main
    )
    include(GoogleTest)
    gtest_discover_tests(stan2d_tests)
endif()
```

### Step 3: Create directory structure

Run:
```bash
mkdir -p src/stan2d/{core,dynamics,collision,constraints,world,export}
mkdir -p src/debug_renderer
mkdir -p include/stan2d/{core,dynamics,collision,constraints,world,export}
mkdir -p tests/{unit,integration}
mkdir -p examples
touch src/stan2d/{core,dynamics,collision,constraints,world,export}/.gitkeep
touch src/debug_renderer/.gitkeep
touch include/stan2d/{core,dynamics,collision,constraints,world,export}/.gitkeep
touch tests/{unit,integration}/.gitkeep
touch examples/.gitkeep
```

### Step 4: Write the smoke test (RED)

**File:** `tests/unit/test_smoke.cpp`

```cpp
#include <gtest/gtest.h>

TEST(Smoke, BuildSystemWorks) {
    EXPECT_EQ(1 + 1, 2);
}
```

### Step 5: Build and run tests

Run:
```bash
cmake -B build -S . -DCMAKE_TOOLCHAIN_FILE=$VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake
cmake --build build
ctest --test-dir build --output-on-failure
```

Expected: PASS — `1 test passed`

### Step 6: Commit

```bash
git init
git add CMakeLists.txt vcpkg.json tests/unit/test_smoke.cpp
git add src/ include/ tests/ examples/ docs/
git commit -m "chore: project scaffolding with CMake, vcpkg, and smoke test"
```

---

## Task 2: Core Math Types

**Goal:** Define Vec2/Mat2 aliases over glm, AABB struct with helper functions.

**Files:**
- Create: `include/stan2d/core/math_types.hpp`
- Create: `tests/unit/test_math_types.cpp`

**Depends on:** Task 1

### Step 1: Write failing tests (RED)

**File:** `tests/unit/test_math_types.cpp`

```cpp
#include <gtest/gtest.h>
#include <stan2d/core/math_types.hpp>

using namespace stan2d;

// ── Vec2 alias ────────────────────────────────────────────────────

TEST(MathTypes, Vec2DefaultConstruction) {
    Vec2 v{};
    EXPECT_FLOAT_EQ(v.x, 0.0f);
    EXPECT_FLOAT_EQ(v.y, 0.0f);
}

TEST(MathTypes, Vec2Arithmetic) {
    Vec2 a{1.0f, 2.0f};
    Vec2 b{3.0f, 4.0f};
    Vec2 sum = a + b;
    EXPECT_FLOAT_EQ(sum.x, 4.0f);
    EXPECT_FLOAT_EQ(sum.y, 6.0f);
}

TEST(MathTypes, Vec2ScalarMultiply) {
    Vec2 v{2.0f, 3.0f};
    Vec2 result = v * 2.0f;
    EXPECT_FLOAT_EQ(result.x, 4.0f);
    EXPECT_FLOAT_EQ(result.y, 6.0f);
}

TEST(MathTypes, Vec2Dot) {
    Vec2 a{1.0f, 0.0f};
    Vec2 b{0.0f, 1.0f};
    EXPECT_FLOAT_EQ(glm::dot(a, b), 0.0f);

    Vec2 c{3.0f, 4.0f};
    EXPECT_FLOAT_EQ(glm::dot(c, c), 25.0f);
}

// ── AABB ──────────────────────────────────────────────────────────

TEST(AABB, ContainsPoint) {
    AABB box{{0.0f, 0.0f}, {10.0f, 10.0f}};
    EXPECT_TRUE(aabb_contains(box, {5.0f, 5.0f}));
    EXPECT_TRUE(aabb_contains(box, {0.0f, 0.0f}));   // boundary inclusive
    EXPECT_FALSE(aabb_contains(box, {-1.0f, 5.0f}));
    EXPECT_FALSE(aabb_contains(box, {5.0f, 11.0f}));
}

TEST(AABB, OverlapsTrue) {
    AABB a{{0.0f, 0.0f}, {5.0f, 5.0f}};
    AABB b{{3.0f, 3.0f}, {8.0f, 8.0f}};
    EXPECT_TRUE(aabb_overlaps(a, b));
    EXPECT_TRUE(aabb_overlaps(b, a));  // commutative
}

TEST(AABB, OverlapsFalse) {
    AABB a{{0.0f, 0.0f}, {2.0f, 2.0f}};
    AABB b{{5.0f, 5.0f}, {8.0f, 8.0f}};
    EXPECT_FALSE(aabb_overlaps(a, b));
}

TEST(AABB, OverlapsTouching) {
    AABB a{{0.0f, 0.0f}, {5.0f, 5.0f}};
    AABB b{{5.0f, 0.0f}, {10.0f, 5.0f}};
    // Touching edges: overlaps should return true (<=)
    EXPECT_TRUE(aabb_overlaps(a, b));
}

TEST(AABB, Merge) {
    AABB a{{1.0f, 2.0f}, {5.0f, 6.0f}};
    AABB b{{3.0f, 0.0f}, {8.0f, 4.0f}};
    AABB merged = aabb_merge(a, b);
    EXPECT_FLOAT_EQ(merged.min.x, 1.0f);
    EXPECT_FLOAT_EQ(merged.min.y, 0.0f);
    EXPECT_FLOAT_EQ(merged.max.x, 8.0f);
    EXPECT_FLOAT_EQ(merged.max.y, 6.0f);
}

TEST(AABB, SurfaceArea) {
    // "Surface area" in 2D = perimeter
    AABB box{{0.0f, 0.0f}, {3.0f, 4.0f}};
    EXPECT_FLOAT_EQ(aabb_perimeter(box), 14.0f);  // 2*(3+4)
}

TEST(AABB, ExpandByMargin) {
    AABB box{{1.0f, 1.0f}, {5.0f, 5.0f}};
    AABB expanded = aabb_expand(box, 0.5f);
    EXPECT_FLOAT_EQ(expanded.min.x, 0.5f);
    EXPECT_FLOAT_EQ(expanded.min.y, 0.5f);
    EXPECT_FLOAT_EQ(expanded.max.x, 5.5f);
    EXPECT_FLOAT_EQ(expanded.max.y, 5.5f);
}
```

### Step 2: Run tests to verify they fail

Run: `cmake --build build && ctest --test-dir build --output-on-failure`

Expected: FAIL — `fatal error: 'stan2d/core/math_types.hpp' file not found`

### Step 3: Implement (GREEN)

**File:** `include/stan2d/core/math_types.hpp`

```cpp
#pragma once

#include <glm/glm.hpp>
#include <glm/gtc/constants.hpp>

namespace stan2d {

// ── Vector / Matrix aliases ───────────────────────────────────────

using Vec2 = glm::vec2;
using Mat2 = glm::mat2;

// ── AABB ──────────────────────────────────────────────────────────

struct AABB {
    Vec2 min;
    Vec2 max;
};

[[nodiscard]] inline bool aabb_contains(const AABB& box, Vec2 point) {
    return point.x >= box.min.x && point.x <= box.max.x
        && point.y >= box.min.y && point.y <= box.max.y;
}

[[nodiscard]] inline bool aabb_overlaps(const AABB& a, const AABB& b) {
    return a.min.x <= b.max.x && a.max.x >= b.min.x
        && a.min.y <= b.max.y && a.max.y >= b.min.y;
}

[[nodiscard]] inline AABB aabb_merge(const AABB& a, const AABB& b) {
    return AABB{
        .min = {glm::min(a.min.x, b.min.x), glm::min(a.min.y, b.min.y)},
        .max = {glm::max(a.max.x, b.max.x), glm::max(a.max.y, b.max.y)}
    };
}

[[nodiscard]] inline float aabb_perimeter(const AABB& box) {
    Vec2 d = box.max - box.min;
    return 2.0f * (d.x + d.y);
}

[[nodiscard]] inline AABB aabb_expand(const AABB& box, float margin) {
    return AABB{
        .min = box.min - Vec2{margin, margin},
        .max = box.max + Vec2{margin, margin}
    };
}

} // namespace stan2d
```

### Step 4: Run tests to verify they pass

Run: `cmake --build build && ctest --test-dir build --output-on-failure`

Expected: PASS — all AABB and Vec2 tests green

### Step 5: Commit

```bash
git add include/stan2d/core/math_types.hpp tests/unit/test_math_types.cpp
git commit -m "feat: core math types — Vec2/Mat2 aliases and AABB utilities"
```

---

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

<!-- Tasks 4-12: To be added -->
