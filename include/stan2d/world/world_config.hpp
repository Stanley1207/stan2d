#pragma once

#include <cstdint>

namespace stan2d {

struct WorldConfig {
    uint32_t max_bodies      = 10000;
    uint32_t max_constraints  = 5000;
    uint32_t max_contacts     = 20000;
    uint32_t max_shapes       = 10000;
};

} // namespace stan2d
