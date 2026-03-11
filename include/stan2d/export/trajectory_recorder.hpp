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
