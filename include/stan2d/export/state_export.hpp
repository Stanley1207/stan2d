#pragma once

#include <string>

#include <stan2d/export/trajectory_recorder.hpp>

namespace stan2d {

class World;

void export_state(const World& world, const std::string& path, ExportFormat fmt);

} // namespace stan2d
