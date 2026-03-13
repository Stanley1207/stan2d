# Phase 1 Implementation Plan — stan2d 2D Physics Engine

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a complete 2D rigid body physics engine with collision detection, constraint solving, state management, and debug rendering.

**Architecture:** Handle + SoA Hybrid — users interact via lightweight Handles (index + generation), engine stores data in Structure of Arrays for cache efficiency. SparseSet with swap-and-pop maintains contiguous dense arrays. Physics pipeline: Apply Forces → Integrate Velocities → Broad Phase → Narrow Phase → Solve Constraints → Integrate Positions → Post Step.

**Tech Stack:** C++20, CMake, vcpkg, Google Test, glm, nlohmann-json, SDL2

**Reference:** `docs/plans/2026-03-09-2d-physics-engine-design.md`

---