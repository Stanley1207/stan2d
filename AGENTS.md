# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## Project Overview

**Stan2D** is a C++20 2D rigid body physics engine designed as a foundation for ML/world model research. Key design goals: deterministic simulation, cache-friendly SoA storage, zero allocations in the hot path (`step()`), and ML-friendly state export.

## Build & Test Commands

**Prerequisites:** CMake 3.24+, vcpkg at `$HOME/vcpkg`, C++20 compiler (Clang/GCC/MSVC)

```bash
# Configure
cmake --preset default

# Build
cmake --build build

# Run all tests
ctest --test-dir build

# Run tests directly (verbose)
./build/stan2d_tests

# Run a single test binary
./build/stan2d_tests --gtest_filter=WorldBasicsTest.*
```

**SDL2 debug renderer** is built automatically if SDL2 is found by vcpkg.

## Architecture

### Module Dependency Order
```
core → dynamics → collision → constraints → world
```

### Key Design Pattern: Handle + SoA Hybrid
- Users interact via lightweight **Handles** (`BodyHandle`, `ShapeHandle`) — value types with index + generation counter
- Internal data stored in **Structure-of-Arrays (SoA)** for cache efficiency and ML tensor compatibility
- **SparseSet** provides O(1) dense iteration via swap-and-pop deletion with generation validation

### Core Modules

| Module | Path | Description |
|--------|------|-------------|
| `core` | `include/stan2d/core/` | Math types (Vec2/Mat2/AABB), Handles, SparseSet, Shapes, ShapeRegistry |
| `dynamics` | `include/stan2d/dynamics/` | BodyStorage (SoA kinematics), Symplectic Euler integrator |
| `collision` | `include/stan2d/collision/` | Dynamic AABB tree (broad phase), SAT narrow phase, ContactManifold |
| `constraints` | `include/stan2d/constraints/` | Sequential impulse solver with warm starting, Coulomb friction |
| `world` | `include/stan2d/world/` | World class orchestrating the full pipeline |
| `export` | `include/stan2d/export/` | State serialization (planned: Task 11–12) |

Most code lives in **headers** (for inline performance). Only `src/stan2d/world/world.cpp` has significant implementation (~107 LOC currently).

### Physics Pipeline (per `world.step(dt)`)
1. Apply forces (gravity + user)
2. Integrate velocities (Symplectic Euler: `v += (F/m + g) * dt`)
3. Broad phase (AABB tree → candidate pairs)
4. Narrow phase (SAT → ContactManifolds with 1–2 points each)
5. Solve constraints (8–16 iterations, warm starting, Baumgarte bias)
6. Integrate positions (`x += v * dt`, `θ += ω * dt`)

### Body Types
- **Static**: Infinite mass, immovable (floors, walls)
- **Dynamic**: Fully simulated
- **Kinematic**: User-controlled velocity, infinite mass in collision response

## Implementation Status

Tasks 1–9 complete. Upcoming:
- **Task 10**: Wire `World::step()` full pipeline
- **Task 11**: State system (View + Snapshot)
- **Task 12**: Trajectory recorder + state export

See `docs/tasks/00_master_plan.md` for the full roadmap and `docs/PROGRESS.md` for current status.

## Critical Constraints

- **Determinism**: `-fno-fast-math` and `-ffp-contract=off` are set in CMake — never override these
- **Zero allocations in `step()`**: All storage is pre-reserved at `World` construction via `WorldConfig` capacity limits
- **Max 8 vertices per polygon**: Hard limit in `PolygonShape`
- **Contact manifolds**: 1–2 contact points per collision pair (stored in fixed-size array)

## Dependencies (via vcpkg)

| Package | Use |
|---------|-----|
| `glm` | `Vec2 = glm::vec2`, `Mat2 = glm::mat2` |
| `nlohmann-json` | State export serialization |
| `gtest` | Unit testing |
| `sdl2` | Optional debug renderer |

## Key Files

- `include/stan2d/world/world.hpp` — Public API entry point
- `include/stan2d/world/world_config.hpp` — Capacity configuration
- `src/stan2d/world/world.cpp` — World implementation
- `include/stan2d/constraints/solver.hpp` — Sequential impulse solver
- `include/stan2d/collision/narrow_phase.hpp` — SAT collision detection
- `docs/2d-physics-engine-design.md` — Full design specification
