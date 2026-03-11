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
