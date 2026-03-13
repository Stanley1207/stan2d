#include <stan2d/export/trajectory_recorder.hpp>
#include <stan2d/world/world.hpp>

#include <fstream>
#include <nlohmann/json.hpp>

namespace stan2d {

TrajectoryRecorder::TrajectoryRecorder(const World& world, uint32_t max_frames)
    : world_(world)
    , max_frames_(max_frames)
    , max_bodies_(world.config().max_bodies)
    , max_joints_(world.config().max_joints)
{
    uint32_t body_total = max_frames_ * max_bodies_;
    all_positions_.resize(body_total, Vec2{0.0f, 0.0f});
    all_velocities_.resize(body_total, Vec2{0.0f, 0.0f});
    all_rotations_.resize(body_total, 0.0f);
    frame_active_counts_.resize(max_frames_, 0);

    uint32_t joint_total = max_frames_ * max_joints_;
    all_joint_angles_.resize(joint_total, 0.0f);
    all_joint_angular_speeds_.resize(joint_total, 0.0f);
    all_joint_constraint_forces_.resize(joint_total, 0.0f);
    all_joint_lengths_.resize(joint_total, 0.0f);
    joint_frame_active_counts_.resize(max_frames_, 0);
}

void TrajectoryRecorder::start() {
    current_frame_ = 0;
    std::fill(all_positions_.begin(), all_positions_.end(), Vec2{0.0f, 0.0f});
    std::fill(all_velocities_.begin(), all_velocities_.end(), Vec2{0.0f, 0.0f});
    std::fill(all_rotations_.begin(), all_rotations_.end(), 0.0f);
    std::fill(frame_active_counts_.begin(), frame_active_counts_.end(), 0u);

    std::fill(all_joint_angles_.begin(), all_joint_angles_.end(), 0.0f);
    std::fill(all_joint_angular_speeds_.begin(), all_joint_angular_speeds_.end(), 0.0f);
    std::fill(all_joint_constraint_forces_.begin(), all_joint_constraint_forces_.end(), 0.0f);
    std::fill(all_joint_lengths_.begin(), all_joint_lengths_.end(), 0.0f);
    std::fill(joint_frame_active_counts_.begin(), joint_frame_active_counts_.end(), 0u);
}

void TrajectoryRecorder::capture() {
    if (current_frame_ >= max_frames_) return;

    WorldStateView view = world_.get_state_view();

    // Body data
    uint32_t active = view.active_body_count;
    uint32_t body_offset = current_frame_ * max_bodies_;
    if (active > 0) {
        std::memcpy(&all_positions_[body_offset], view.positions.data(),
                    active * sizeof(Vec2));
        std::memcpy(&all_velocities_[body_offset], view.velocities.data(),
                    active * sizeof(Vec2));
        std::memcpy(&all_rotations_[body_offset], view.rotations.data(),
                    active * sizeof(float));
    }
    frame_active_counts_[current_frame_] = active;

    // Joint data
    uint32_t joint_active = view.joints.active_joint_count;
    uint32_t joint_offset = current_frame_ * max_joints_;
    if (joint_active > 0) {
        std::memcpy(&all_joint_angles_[joint_offset], view.joints.angles.data(),
                    joint_active * sizeof(float));
        std::memcpy(&all_joint_angular_speeds_[joint_offset],
                    view.joints.angular_speeds.data(), joint_active * sizeof(float));
        std::memcpy(&all_joint_constraint_forces_[joint_offset],
                    view.joints.constraint_forces.data(), joint_active * sizeof(float));
        std::memcpy(&all_joint_lengths_[joint_offset], view.joints.lengths.data(),
                    joint_active * sizeof(float));
    }
    joint_frame_active_counts_[current_frame_] = joint_active;

    ++current_frame_;
}

void TrajectoryRecorder::save(const std::string& path, ExportFormat fmt) const {
    if (fmt == ExportFormat::JSON) {
        nlohmann::json j;
        j["frame_count"] = current_frame_;
        j["max_bodies"] = max_bodies_;
        j["max_joints"] = max_joints_;
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

            // Joint data
            frame["joint_active_count"] = joint_frame_active_counts_[f];
            frame["joint_angles"] = nlohmann::json::array();
            frame["joint_angular_speeds"] = nlohmann::json::array();
            frame["joint_constraint_forces"] = nlohmann::json::array();
            frame["joint_lengths"] = nlohmann::json::array();

            uint32_t j_offset = f * max_joints_;
            uint32_t j_active = joint_frame_active_counts_[f];

            for (uint32_t jj = 0; jj < j_active; ++jj) {
                frame["joint_angles"].push_back(all_joint_angles_[j_offset + jj]);
                frame["joint_angular_speeds"].push_back(
                    all_joint_angular_speeds_[j_offset + jj]);
                frame["joint_constraint_forces"].push_back(
                    all_joint_constraint_forces_[j_offset + jj]);
                frame["joint_lengths"].push_back(all_joint_lengths_[j_offset + jj]);
            }

            j["frames"].push_back(frame);
        }

        std::ofstream file(path);
        file << j.dump(2);

    } else {
        // Binary: header + flat arrays
        std::ofstream file(path, std::ios::binary);

        uint32_t magic = 0x53324454;  // "S2DT" (Trajectory)
        uint32_t version = 2;         // v2 adds joint data
        file.write(reinterpret_cast<const char*>(&magic), sizeof(magic));
        file.write(reinterpret_cast<const char*>(&version), sizeof(version));
        file.write(reinterpret_cast<const char*>(&current_frame_), sizeof(current_frame_));
        file.write(reinterpret_cast<const char*>(&max_bodies_), sizeof(max_bodies_));
        file.write(reinterpret_cast<const char*>(&max_joints_), sizeof(max_joints_));

        // Active counts per frame
        file.write(reinterpret_cast<const char*>(frame_active_counts_.data()),
                   current_frame_ * sizeof(uint32_t));

        // Body positions/velocities/rotations
        uint32_t body_total = current_frame_ * max_bodies_;
        file.write(reinterpret_cast<const char*>(all_positions_.data()),
                   body_total * sizeof(Vec2));
        file.write(reinterpret_cast<const char*>(all_velocities_.data()),
                   body_total * sizeof(Vec2));
        file.write(reinterpret_cast<const char*>(all_rotations_.data()),
                   body_total * sizeof(float));

        // Joint active counts per frame
        file.write(reinterpret_cast<const char*>(joint_frame_active_counts_.data()),
                   current_frame_ * sizeof(uint32_t));

        // Joint data
        uint32_t joint_total = current_frame_ * max_joints_;
        file.write(reinterpret_cast<const char*>(all_joint_angles_.data()),
                   joint_total * sizeof(float));
        file.write(reinterpret_cast<const char*>(all_joint_angular_speeds_.data()),
                   joint_total * sizeof(float));
        file.write(reinterpret_cast<const char*>(all_joint_constraint_forces_.data()),
                   joint_total * sizeof(float));
        file.write(reinterpret_cast<const char*>(all_joint_lengths_.data()),
                   joint_total * sizeof(float));
    }
}

} // namespace stan2d
