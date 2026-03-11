#pragma once

#include <cstdint>
#include <functional>

namespace stan2d {

struct Handle {
    uint32_t index      = 0;
    uint32_t generation = 0;

    bool operator==(const Handle&) const = default;
    bool operator!=(const Handle&) const = default;
};

// Typed handles — distinct types prevent accidental mixing
struct BodyHandle {
    uint32_t index      = 0;
    uint32_t generation = 0;

    bool operator==(const BodyHandle&) const = default;
    bool operator!=(const BodyHandle&) const = default;
};

struct ShapeHandle {
    uint32_t index      = 0;
    uint32_t generation = 0;

    bool operator==(const ShapeHandle&) const = default;
    bool operator!=(const ShapeHandle&) const = default;
};

} // namespace stan2d
