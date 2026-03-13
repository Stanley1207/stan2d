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

        // Joint data
        j["joint_count"] = view.joints.active_joint_count;
        j["joints"] = nlohmann::json::array();

        const char* type_names[] = {"Hinge", "Spring", "Distance", "Pulley"};

        for (uint32_t i = 0; i < view.joints.active_joint_count; ++i) {
            nlohmann::json joint;
            uint8_t type_val = view.joints.types[i];
            joint["type"] = (type_val < 4) ? type_names[type_val] : "Unknown";
            joint["angle"] = view.joints.angles[i];
            joint["angular_speed"] = view.joints.angular_speeds[i];
            joint["motor_enabled"] = view.joints.motor_enabled[i] != 0;
            joint["motor_target_speed"] = view.joints.motor_target_speeds[i];
            joint["constraint_force"] = view.joints.constraint_forces[i];
            joint["length"] = view.joints.lengths[i];
            j["joints"].push_back(joint);
        }

        std::ofstream file(path);
        file << j.dump(2);

    } else {
        // Binary format: magic + version + body_count + flat SoA arrays
        std::ofstream file(path, std::ios::binary);

        uint32_t magic = 0x53324450;  // "S2DP" (Physics state)
        uint32_t version = 2;         // v2 adds joint data
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

        // Joint count + joint observable data
        uint32_t jcount = view.joints.active_joint_count;
        file.write(reinterpret_cast<const char*>(&jcount), sizeof(jcount));
        if (jcount > 0) {
            file.write(reinterpret_cast<const char*>(view.joints.types.data()),
                       jcount * sizeof(uint8_t));
            file.write(reinterpret_cast<const char*>(view.joints.angles.data()),
                       jcount * sizeof(float));
            file.write(reinterpret_cast<const char*>(view.joints.angular_speeds.data()),
                       jcount * sizeof(float));
            file.write(reinterpret_cast<const char*>(view.joints.constraint_forces.data()),
                       jcount * sizeof(float));
            file.write(reinterpret_cast<const char*>(view.joints.lengths.data()),
                       jcount * sizeof(float));
        }
    }
}

} // namespace stan2d
