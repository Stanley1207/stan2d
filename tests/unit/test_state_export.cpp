#include <gtest/gtest.h>
#include <filesystem>
#include <fstream>
#include <stan2d/export/state_export.hpp>
#include <stan2d/export/trajectory_recorder.hpp>
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
