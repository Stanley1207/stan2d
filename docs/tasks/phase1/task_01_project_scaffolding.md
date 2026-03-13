> **Note for Claude**: 牢记 `docs/tasks/00_master_plan.md` 中的四大铁律。

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