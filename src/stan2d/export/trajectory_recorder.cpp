#include <stan2d/export/trajectory_recorder.hpp>
#include <stan2d/world/world.hpp>

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
