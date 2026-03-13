#pragma once

#include <stan2d/constraints/contact_constraint.hpp>
#include <stan2d/constraints/solver.hpp>
#include <stan2d/dynamics/body_storage.hpp>
#include <stan2d/joints/joint_storage.hpp>

namespace stan2d {

// Compute effective masses and cache per-frame values for all joints.
// Call once per step, before warm_start_joints().
void prepare_joint_constraints(JointStorage& joints, const BodyStorage& bodies, float dt);

// Apply accumulated impulses from previous frame for all joints.
// Call after prepare_joint_constraints(), before iteration loop.
void warm_start_joints(const JointStorage& joints, BodyStorage& bodies);

// Solve all joint constraints for one iteration.
// Call inside the iteration loop, after solve_contacts().
void solve_joints(JointStorage& joints, BodyStorage& bodies,
                  const SolverConfig& config, float dt);

} // namespace stan2d
