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
        PRIVATE stan2d GTest::gtest GTest::gtest_main nlohmann_json::nlohmann_json
    )
    include(GoogleTest)
    gtest_discover_tests(stan2d_tests)
endif()
