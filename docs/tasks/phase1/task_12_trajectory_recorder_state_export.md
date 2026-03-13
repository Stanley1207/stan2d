> **Note for Claude**: 牢记 `docs/tasks/00_master_plan.md` 中的四大铁律。

## Task 12: Trajectory Recorder + State Export

**Goal:** TrajectoryRecorder captures per-frame state in fixed-stride ML-friendly layout (reshapeable to `[frames, max_bodies, channels]`). State export writes JSON (debug) and binary (ML training) formats. Supports full deterministic replay via snapshot + export round-trip.

**Files:**
- Create: `include/stan2d/export/trajectory_recorder.hpp`
- Create: `include/stan2d/export/state_export.hpp`
- Create: `src/stan2d/export/state_export.cpp`
- Create: `tests/unit/test_trajectory_recorder.cpp`
- Create: `tests/unit/test_state_export.cpp`

**Depends on:** Task 10, Task 11

### Step 1: Write failing tests — TrajectoryRecorder (RED)

**File:** `tests/unit/test_trajectory_recorder.cpp`

```cpp
#include <gtest/gtest.h>
#include <stan2d/export/trajectory_recorder.hpp>
#include <stan2d/world/world.hpp>

using namespace stan2d;

// ── Basic recording ───────────────────────────────────────────────

TEST(TrajectoryRecorder, ConstructWithWorld) {
    World world(WorldConfig{.max_bodies = 100, .max_shapes = 100, .max_contacts = 100});
    TrajectoryRecorder recorder(world, 60);

    EXPECT_EQ(recorder.max_frames(), 60u);
    EXPECT_EQ(recorder.current_frame(), 0u);
}

TEST(TrajectoryRecorder, CaptureIncreasesFrameCount) {
    World world(WorldConfig{.max_bodies = 100, .max_shapes = 100, .max_contacts = 100});

    ShapeHandle shape = world.create_shape(CircleShape{.radius = 0.5f});
    world.create_body({.position = {0.0f, 0.0f}, .shape = shape, .mass = 1.0f});

    TrajectoryRecorder recorder(world, 60);
    recorder.start();
    recorder.capture();

    EXPECT_EQ(recorder.current_frame(), 1u);
}

TEST(TrajectoryRecorder, CaptureRecordsPositions) {
    World world(WorldConfig{.max_bodies = 10, .max_shapes = 10, .max_contacts = 10});

    ShapeHandle shape = world.create_shape(CircleShape{.radius = 0.5f});
    world.create_body({.position = {1.0f, 2.0f}, .shape = shape, .mass = 1.0f});
    world.create_body({.position = {3.0f, 4.0f}, .shape = shape, .mass = 1.0f});

    TrajectoryRecorder recorder(world, 10);
    recorder.start();
    recorder.capture();

    // Frame 0, body 0
    Vec2 pos = recorder.get_position(0, 0);
    EXPECT_FLOAT_EQ(pos.x, 1.0f);
    EXPECT_FLOAT_EQ(pos.y, 2.0f);

    // Frame 0, body 1
    Vec2 pos2 = recorder.get_position(0, 1);
    EXPECT_FLOAT_EQ(pos2.x, 3.0f);
    EXPECT_FLOAT_EQ(pos2.y, 4.0f);
}

TEST(TrajectoryRecorder, MultipleFramesTrackMotion) {
    World world(WorldConfig{.max_bodies = 10, .max_shapes = 10, .max_contacts = 10});
    world.set_gravity({0.0f, -10.0f});

    ShapeHandle shape = world.create_shape(CircleShape{.radius = 0.5f});
    world.create_body({.position = {0.0f, 10.0f}, .shape = shape, .mass = 1.0f});

    TrajectoryRecorder recorder(world, 100);
    recorder.start();

    float dt = 1.0f / 60.0f;
    for (int i = 0; i < 10; ++i) {
        recorder.capture();
        world.step(dt);
    }

    EXPECT_EQ(recorder.current_frame(), 10u);

    // Each frame should show the ball lower than the previous
    for (uint32_t f = 1; f < 10; ++f) {
        float y_prev = recorder.get_position(f - 1, 0).y;
        float y_curr = recorder.get_position(f, 0).y;
        EXPECT_LT(y_curr, y_prev);
    }
}

// ── Fixed stride layout ───────────────────────────────────────────

TEST(TrajectoryRecorder, FixedStrideLayout) {
    World world(WorldConfig{.max_bodies = 10, .max_shapes = 10, .max_contacts = 10});

    ShapeHandle shape = world.create_shape(CircleShape{.radius = 0.5f});
    world.create_body({.position = {1.0f, 2.0f}, .shape = shape, .mass = 1.0f});

    TrajectoryRecorder recorder(world, 5);
    recorder.start();
    recorder.capture();

    // Get raw position buffer: should have max_bodies slots per frame
    std::span<const Vec2> raw_positions = recorder.raw_positions();
    // Total size = max_frames * max_bodies = 5 * 10 = 50
    EXPECT_EQ(raw_positions.size(), 50u);
}

TEST(TrajectoryRecorder, ActiveCountsTracked) {
    World world(WorldConfig{.max_bodies = 10, .max_shapes = 10, .max_contacts = 10});

    ShapeHandle shape = world.create_shape(CircleShape{.radius = 0.5f});
    world.create_body({.position = {1.0f, 2.0f}, .shape = shape, .mass = 1.0f});

    TrajectoryRecorder recorder(world, 10);
    recorder.start();
    recorder.capture();

    EXPECT_EQ(recorder.get_active_count(0), 1u);
}

// ── Capacity limit ────────────────────────────────────────────────

TEST(TrajectoryRecorder, StopsAtMaxFrames) {
    World world(WorldConfig{.max_bodies = 10, .max_shapes = 10, .max_contacts = 10});

    ShapeHandle shape = world.create_shape(CircleShape{.radius = 0.5f});
    world.create_body({.position = {0.0f, 0.0f}, .shape = shape, .mass = 1.0f});

    TrajectoryRecorder recorder(world, 3);
    recorder.start();

    recorder.capture();
    recorder.capture();
    recorder.capture();
    recorder.capture();  // Should be ignored (at capacity)

    EXPECT_EQ(recorder.current_frame(), 3u);
}

// ── Reset ─────────────────────────────────────────────────────────

TEST(TrajectoryRecorder, ResetClearsFrames) {
    World world(WorldConfig{.max_bodies = 10, .max_shapes = 10, .max_contacts = 10});

    ShapeHandle shape = world.create_shape(CircleShape{.radius = 0.5f});
    world.create_body({.position = {0.0f, 0.0f}, .shape = shape, .mass = 1.0f});

    TrajectoryRecorder recorder(world, 10);
    recorder.start();
    recorder.capture();
    recorder.capture();

    EXPECT_EQ(recorder.current_frame(), 2u);

    recorder.start();  // Reset
    EXPECT_EQ(recorder.current_frame(), 0u);
}
```

### Step 2: Write failing tests — State Export (RED)

**File:** `tests/unit/test_state_export.cpp`

```cpp
#include <gtest/gtest.h>
#include <filesystem>
#include <fstream>
#include <stan2d/export/state_export.hpp>
#include <stan2d/world/world.hpp>
#include <nlohmann/json.hpp>

using namespace stan2d;

class ExportFixture : public ::testing::Test {
protected:
    std::filesystem::path temp_dir_;

    void SetUp() override {
        temp_dir_ = std::filesystem::temp_directory_path() / "stan2d_test_export";
        std::filesystem::create_directories(temp_dir_);
    }

    void TearDown() override {
        std::filesystem::remove_all(temp_dir_);
    }
};

// ── JSON export ───────────────────────────────────────────────────

TEST_F(ExportFixture, ExportStateJSON) {
    World world(WorldConfig{.max_bodies = 100, .max_shapes = 100, .max_contacts = 100});
    world.set_gravity({0.0f, -9.81f});

    ShapeHandle shape = world.create_shape(CircleShape{.radius = 0.5f});
    world.create_body({
        .position = {1.0f, 2.0f},
        .velocity = {3.0f, 4.0f},
        .shape    = shape,
        .mass     = 5.0f
    });

    auto path = temp_dir_ / "state.json";
    export_state(world, path.string(), ExportFormat::JSON);

    EXPECT_TRUE(std::filesystem::exists(path));

    // Parse and verify
    std::ifstream file(path);
    nlohmann::json j;
    file >> j;

    EXPECT_EQ(j["body_count"], 1);
    EXPECT_FLOAT_EQ(j["bodies"][0]["position"][0].get<float>(), 1.0f);
    EXPECT_FLOAT_EQ(j["bodies"][0]["position"][1].get<float>(), 2.0f);
    EXPECT_FLOAT_EQ(j["bodies"][0]["velocity"][0].get<float>(), 3.0f);
    EXPECT_FLOAT_EQ(j["bodies"][0]["velocity"][1].get<float>(), 4.0f);
    EXPECT_FLOAT_EQ(j["bodies"][0]["mass"].get<float>(), 5.0f);
}

TEST_F(ExportFixture, ExportMultipleBodiesJSON) {
    World world(WorldConfig{.max_bodies = 100, .max_shapes = 100, .max_contacts = 100});

    ShapeHandle shape = world.create_shape(CircleShape{.radius = 0.5f});
    world.create_body({.position = {1.0f, 0.0f}, .shape = shape, .mass = 1.0f});
    world.create_body({.position = {2.0f, 0.0f}, .shape = shape, .mass = 2.0f});
    world.create_body({.position = {3.0f, 0.0f}, .shape = shape, .mass = 3.0f});

    auto path = temp_dir_ / "multi.json";
    export_state(world, path.string(), ExportFormat::JSON);

    std::ifstream file(path);
    nlohmann::json j;
    file >> j;

    EXPECT_EQ(j["body_count"], 3);
    EXPECT_EQ(j["bodies"].size(), 3u);
}

// ── Binary export ─────────────────────────────────────────────────

TEST_F(ExportFixture, ExportStateBinary) {
    World world(WorldConfig{.max_bodies = 100, .max_shapes = 100, .max_contacts = 100});

    ShapeHandle shape = world.create_shape(CircleShape{.radius = 0.5f});
    world.create_body({
        .position = {1.0f, 2.0f},
        .velocity = {3.0f, 4.0f},
        .shape    = shape,
        .mass     = 5.0f
    });

    auto path = temp_dir_ / "state.bin";
    export_state(world, path.string(), ExportFormat::Binary);

    EXPECT_TRUE(std::filesystem::exists(path));
    EXPECT_GT(std::filesystem::file_size(path), 0u);
}

TEST_F(ExportFixture, BinaryRoundTrip) {
    World world(WorldConfig{.max_bodies = 100, .max_shapes = 100, .max_contacts = 100});

    ShapeHandle shape = world.create_shape(CircleShape{.radius = 0.5f});
    world.create_body({
        .position = {1.5f, 2.5f},
        .velocity = {-1.0f, 3.0f},
        .shape    = shape,
        .mass     = 7.0f
    });

    auto path = temp_dir_ / "roundtrip.bin";
    export_state(world, path.string(), ExportFormat::Binary);

    // Read back and verify header
    std::ifstream file(path, std::ios::binary);
    uint32_t magic = 0;
    uint32_t version = 0;
    uint32_t body_count = 0;

    file.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    file.read(reinterpret_cast<char*>(&version), sizeof(version));
    file.read(reinterpret_cast<char*>(&body_count), sizeof(body_count));

    EXPECT_EQ(magic, 0x53324450u);  // "S2DP"
    EXPECT_EQ(version, 1u);
    EXPECT_EQ(body_count, 1u);

    // Read position
    float px = 0.0f, py = 0.0f;
    file.read(reinterpret_cast<char*>(&px), sizeof(float));
    file.read(reinterpret_cast<char*>(&py), sizeof(float));

    EXPECT_FLOAT_EQ(px, 1.5f);
    EXPECT_FLOAT_EQ(py, 2.5f);
}

// ── Trajectory binary export ──────────────────────────────────────

TEST_F(ExportFixture, TrajectoryBinaryExport) {
    World world(WorldConfig{.max_bodies = 10, .max_shapes = 10, .max_contacts = 10});
    world.set_gravity({0.0f, -10.0f});

    ShapeHandle shape = world.create_shape(CircleShape{.radius = 0.5f});
    world.create_body({.position = {0.0f, 10.0f}, .shape = shape, .mass = 1.0f});

    TrajectoryRecorder recorder(world, 5);
    recorder.start();

    float dt = 1.0f / 60.0f;
    for (int i = 0; i < 5; ++i) {
        recorder.capture();
        world.step(dt);
    }

    auto path = temp_dir_ / "trajectory.bin";
    recorder.save(path.string(), ExportFormat::Binary);

    EXPECT_TRUE(std::filesystem::exists(path));
    EXPECT_GT(std::filesystem::file_size(path), 0u);
}

TEST_F(ExportFixture, TrajectoryJSONExport) {
    World world(WorldConfig{.max_bodies = 10, .max_shapes = 10, .max_contacts = 10});

    ShapeHandle shape = world.create_shape(CircleShape{.radius = 0.5f});
    world.create_body({.position = {5.0f, 5.0f}, .shape = shape, .mass = 1.0f});

    TrajectoryRecorder recorder(world, 3);
    recorder.start();
    recorder.capture();
    recorder.capture();
    recorder.capture();

    auto path = temp_dir_ / "trajectory.json";
    recorder.save(path.string(), ExportFormat::JSON);

    std::ifstream file(path);
    nlohmann::json j;
    file >> j;

    EXPECT_EQ(j["frame_count"], 3);
    EXPECT_EQ(j["max_bodies"], 10);
    EXPECT_EQ(j["frames"].size(), 3u);
}

// ── Empty world export ────────────────────────────────────────────

TEST_F(ExportFixture, EmptyWorldExportJSON) {
    World world(WorldConfig{.max_bodies = 100, .max_shapes = 100, .max_contacts = 100});

    auto path = temp_dir_ / "empty.json";
    export_state(world, path.string(), ExportFormat::JSON);

    std::ifstream file(path);
    nlohmann::json j;
    file >> j;

    EXPECT_EQ(j["body_count"], 0);
    EXPECT_TRUE(j["bodies"].empty());
}
```

### Step 3: Run tests to verify they fail

Run: `cmake --build build && ctest --test-dir build --output-on-failure`

Expected: FAIL — `fatal error: 'stan2d/export/trajectory_recorder.hpp' file not found`

### Step 4: Implement TrajectoryRecorder

**File:** `include/stan2d/export/trajectory_recorder.hpp`

```cpp
#pragma once

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <span>
#include <string>
#include <vector>

#include <stan2d/core/math_types.hpp>
#include <stan2d/world/state_view.hpp>

namespace stan2d {

class World;  // Forward declaration

enum class ExportFormat : uint8_t {
    JSON,
    Binary
};

class TrajectoryRecorder {
public:
    TrajectoryRecorder(const World& world, uint32_t max_frames);

    void start();
    void capture();
    void save(const std::string& path, ExportFormat fmt) const;

    [[nodiscard]] uint32_t max_frames() const { return max_frames_; }
    [[nodiscard]] uint32_t current_frame() const { return current_frame_; }
    [[nodiscard]] uint32_t max_bodies() const { return max_bodies_; }

    [[nodiscard]] Vec2 get_position(uint32_t frame, uint32_t body) const {
        return all_positions_[frame * max_bodies_ + body];
    }

    [[nodiscard]] Vec2 get_velocity(uint32_t frame, uint32_t body) const {
        return all_velocities_[frame * max_bodies_ + body];
    }

    [[nodiscard]] float get_rotation(uint32_t frame, uint32_t body) const {
        return all_rotations_[frame * max_bodies_ + body];
    }

    [[nodiscard]] uint32_t get_active_count(uint32_t frame) const {
        return frame_active_counts_[frame];
    }

    [[nodiscard]] std::span<const Vec2> raw_positions() const {
        return std::span<const Vec2>(all_positions_.data(), all_positions_.size());
    }

    [[nodiscard]] std::span<const Vec2> raw_velocities() const {
        return std::span<const Vec2>(all_velocities_.data(), all_velocities_.size());
    }

private:
    const World& world_;
    uint32_t max_frames_;
    uint32_t max_bodies_;
    uint32_t current_frame_ = 0;

    // Pre-allocated: max_frames * max_bodies (fixed stride)
    std::vector<Vec2>     all_positions_;
    std::vector<Vec2>     all_velocities_;
    std::vector<float>    all_rotations_;
    std::vector<uint32_t> frame_active_counts_;
};

} // namespace stan2d
```

### Step 5: Implement TrajectoryRecorder (GREEN)

**File:** `src/stan2d/export/trajectory_recorder.cpp`

Create this new source file:

```cpp
#include <stan2d/export/trajectory_recorder.hpp>
#include <stan2d/world/world.hpp>

#include <cassert>
#include <fstream>
#include <nlohmann/json.hpp>

namespace stan2d {

TrajectoryRecorder::TrajectoryRecorder(const World& world, uint32_t max_frames)
    : world_(world)
    , max_frames_(max_frames)
    , max_bodies_(world.config().max_bodies)
{
    uint32_t total = max_frames_ * max_bodies_;
    all_positions_.resize(total, Vec2{0.0f, 0.0f});
    all_velocities_.resize(total, Vec2{0.0f, 0.0f});
    all_rotations_.resize(total, 0.0f);
    frame_active_counts_.resize(max_frames_, 0);
}

void TrajectoryRecorder::start() {
    current_frame_ = 0;
    // Zero-fill buffers
    std::fill(all_positions_.begin(), all_positions_.end(), Vec2{0.0f, 0.0f});
    std::fill(all_velocities_.begin(), all_velocities_.end(), Vec2{0.0f, 0.0f});
    std::fill(all_rotations_.begin(), all_rotations_.end(), 0.0f);
    std::fill(frame_active_counts_.begin(), frame_active_counts_.end(), 0u);
}

void TrajectoryRecorder::capture() {
    if (current_frame_ >= max_frames_) return;

    WorldStateView view = world_.get_state_view();
    uint32_t active = view.active_body_count;
    uint32_t offset = current_frame_ * max_bodies_;

    // Copy active body data into fixed-stride slot
    if (active > 0) {
        std::memcpy(&all_positions_[offset], view.positions.data(),
                    active * sizeof(Vec2));
        std::memcpy(&all_velocities_[offset], view.velocities.data(),
                    active * sizeof(Vec2));
        std::memcpy(&all_rotations_[offset], view.rotations.data(),
                    active * sizeof(float));
    }

    frame_active_counts_[current_frame_] = active;
    ++current_frame_;
}

void TrajectoryRecorder::save(const std::string& path, ExportFormat fmt) const {
    if (fmt == ExportFormat::JSON) {
        nlohmann::json j;
        j["frame_count"] = current_frame_;
        j["max_bodies"] = max_bodies_;
        j["frames"] = nlohmann::json::array();

        for (uint32_t f = 0; f < current_frame_; ++f) {
            nlohmann::json frame;
            frame["active_count"] = frame_active_counts_[f];
            frame["positions"] = nlohmann::json::array();
            frame["velocities"] = nlohmann::json::array();

            uint32_t offset = f * max_bodies_;
            uint32_t active = frame_active_counts_[f];

            for (uint32_t b = 0; b < active; ++b) {
                Vec2 pos = all_positions_[offset + b];
                Vec2 vel = all_velocities_[offset + b];
                frame["positions"].push_back({pos.x, pos.y});
                frame["velocities"].push_back({vel.x, vel.y});
            }

            j["frames"].push_back(frame);
        }

        std::ofstream file(path);
        file << j.dump(2);

    } else {
        // Binary: header + flat arrays
        std::ofstream file(path, std::ios::binary);

        uint32_t magic = 0x53324454;  // "S2DT" (Trajectory)
        uint32_t version = 1;
        file.write(reinterpret_cast<const char*>(&magic), sizeof(magic));
        file.write(reinterpret_cast<const char*>(&version), sizeof(version));
        file.write(reinterpret_cast<const char*>(&current_frame_), sizeof(current_frame_));
        file.write(reinterpret_cast<const char*>(&max_bodies_), sizeof(max_bodies_));

        // Active counts per frame
        file.write(reinterpret_cast<const char*>(frame_active_counts_.data()),
                   current_frame_ * sizeof(uint32_t));

        // Positions: current_frame_ * max_bodies_ entries
        uint32_t total = current_frame_ * max_bodies_;
        file.write(reinterpret_cast<const char*>(all_positions_.data()),
                   total * sizeof(Vec2));
        file.write(reinterpret_cast<const char*>(all_velocities_.data()),
                   total * sizeof(Vec2));
        file.write(reinterpret_cast<const char*>(all_rotations_.data()),
                   total * sizeof(float));
    }
}

} // namespace stan2d
```

### Step 6: Implement state export functions

**File:** `include/stan2d/export/state_export.hpp`

```cpp
#pragma once

#include <string>

#include <stan2d/export/trajectory_recorder.hpp>

namespace stan2d {

class World;

void export_state(const World& world, const std::string& path, ExportFormat fmt);

} // namespace stan2d
```

**File:** `src/stan2d/export/state_export.cpp`

```cpp
#include <stan2d/export/state_export.hpp>
#include <stan2d/world/world.hpp>

#include <fstream>
#include <nlohmann/json.hpp>

namespace stan2d {

void export_state(const World& world, const std::string& path, ExportFormat fmt) {
    WorldStateView view = world.get_state_view();

    if (fmt == ExportFormat::JSON) {
        nlohmann::json j;
        j["body_count"] = view.active_body_count;
        j["bodies"] = nlohmann::json::array();

        for (uint32_t i = 0; i < view.active_body_count; ++i) {
            nlohmann::json body;
            body["position"] = {view.positions[i].x, view.positions[i].y};
            body["velocity"] = {view.velocities[i].x, view.velocities[i].y};
            body["rotation"] = view.rotations[i];
            body["angular_velocity"] = view.angular_velocities[i];
            body["mass"] = view.masses[i];
            j["bodies"].push_back(body);
        }

        std::ofstream file(path);
        file << j.dump(2);

    } else {
        // Binary format: magic + version + body_count + flat SoA arrays
        std::ofstream file(path, std::ios::binary);

        uint32_t magic = 0x53324450;  // "S2DP" (Physics state)
        uint32_t version = 1;
        uint32_t count = view.active_body_count;

        file.write(reinterpret_cast<const char*>(&magic), sizeof(magic));
        file.write(reinterpret_cast<const char*>(&version), sizeof(version));
        file.write(reinterpret_cast<const char*>(&count), sizeof(count));

        // Write SoA arrays contiguously
        file.write(reinterpret_cast<const char*>(view.positions.data()),
                   count * sizeof(Vec2));
        file.write(reinterpret_cast<const char*>(view.velocities.data()),
                   count * sizeof(Vec2));
        file.write(reinterpret_cast<const char*>(view.rotations.data()),
                   count * sizeof(float));
        file.write(reinterpret_cast<const char*>(view.angular_velocities.data()),
                   count * sizeof(float));
        file.write(reinterpret_cast<const char*>(view.masses.data()),
                   count * sizeof(float));
    }
}

} // namespace stan2d
```

### Step 7: Add `config()` accessor to World

**File:** `include/stan2d/world/world.hpp` — add to public section:

```cpp
    [[nodiscard]] const WorldConfig& config() const { return config_; }
```

### Step 8: Update CMakeLists.txt — add export sources and link nlohmann-json

Ensure `src/stan2d/export/state_export.cpp` and `src/stan2d/export/trajectory_recorder.cpp` are included in the `GLOB_RECURSE` pattern (they should be, since they're under `src/stan2d/`). No changes needed if using `GLOB_RECURSE`.

`nlohmann_json` is already linked as `PRIVATE` in the stan2d target. The export source files include `<nlohmann/json.hpp>` in `.cpp` files only, keeping the public API clean.

### Step 9: Run tests to verify they pass

Run: `cmake --build build && ctest --test-dir build --output-on-failure`

Expected: PASS — all trajectory recorder and state export tests green

### Step 10: Commit

```bash
git add include/stan2d/export/trajectory_recorder.hpp \
        include/stan2d/export/state_export.hpp \
        src/stan2d/export/trajectory_recorder.cpp \
        src/stan2d/export/state_export.cpp \
        include/stan2d/world/world.hpp \
        tests/unit/test_trajectory_recorder.cpp \
        tests/unit/test_state_export.cpp
git commit -m "feat: TrajectoryRecorder with fixed-stride ML layout and JSON/Binary state export"
```