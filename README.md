# Stan2D

A C++20 2D rigid body physics engine designed as a foundation for ML and world model research.

**Design goals:** deterministic simulation across platforms, cache-friendly Structure-of-Arrays (SoA) storage compatible with ML tensors, zero heap allocations in the hot path (`step()`), and clean handle-based API.

---

## Building

**Prerequisites:** CMake 3.24+, [vcpkg](https://vcpkg.io) installed at `$HOME/vcpkg`, C++20 compiler (Clang / GCC / MSVC)

```bash
cmake --preset default
cmake --build build
```

An optional SDL2 debug renderer (`stan2d_debug_renderer`) is built automatically if SDL2 is found via vcpkg.

## Running Tests

```bash
ctest --test-dir build

# Verbose / single suite
./build/stan2d_tests --gtest_filter=WorldBasicsTest.*
```

---

## Usage

### 1. Create a World

All memory is pre-allocated at construction — no allocations occur during simulation.

```cpp
#include <stan2d/world/world.hpp>

using namespace stan2d;

World world(WorldConfig{
    .max_bodies      = 1000,
    .max_shapes      = 1000,
    .max_constraints = 500,
    .max_contacts    = 2000,
});

world.set_gravity({0.0f, -9.81f});
```

### 2. Create Shapes

Shapes are registered once and can be shared across many bodies.

```cpp
// Circle
ShapeHandle circle = world.create_shape(CircleShape{.radius = 0.5f});

// Capsule
ShapeHandle capsule = world.create_shape(CapsuleShape{
    .point_a = {0.0f, -0.5f},
    .point_b = {0.0f,  0.5f},
    .radius  = 0.1f,
});

// Polygon (up to 8 vertices)
PolygonShape box{};
box.vertex_count = 4;
box.vertices[0] = {-0.5f, -0.5f};
box.vertices[1] = { 0.5f, -0.5f};
box.vertices[2] = { 0.5f,  0.5f};
box.vertices[3] = {-0.5f,  0.5f};
ShapeHandle poly = world.create_shape(box);
```

### 3. Create Bodies

**Two-step** (shared shape handle):

```cpp
BodyHandle body = world.create_body({
    .position  = {0.0f, 10.0f},
    .velocity  = {1.0f,  0.0f},
    .shape     = circle,
    .mass      = 1.0f,
    .body_type = BodyType::Dynamic,
});
```

**One-step** (inline shape, engine manages the handle):

```cpp
BodyHandle body = world.create_body({
    .position   = {0.0f, 0.0f},
    .shape_data = CircleShape{.radius = 0.3f},
    .mass       = 2.0f,
});
```

**Static body** (floor, wall — infinite mass, never moves):

```cpp
BodyHandle floor = world.create_body({
    .position  = {0.0f, 0.0f},
    .shape     = poly,
    .body_type = BodyType::Static,
});
```

**Kinematic body** (user-controlled velocity, not affected by forces):

```cpp
BodyHandle platform = world.create_body({
    .position  = {0.0f, 2.0f},
    .velocity  = {1.0f, 0.0f},
    .shape     = poly,
    .body_type = BodyType::Kinematic,
});
```

### 4. Step and Query

```cpp
const float dt = 1.0f / 60.0f;

world.step(dt);

Vec2 pos = world.get_position(body);
Vec2 vel = world.get_velocity(body);
float mass = world.get_mass(body);
```

### 5. Destroy Bodies

Handles become invalid after destruction. Stale handles are safely detected.

```cpp
world.destroy_body(body);

world.is_valid(body); // → false
```

---

## Architecture

```
core → dynamics → collision → constraints → world
```

| Module        | Responsibility |
|---------------|----------------|
| `core`        | Math types (Vec2, Mat2, AABB), Handles, SparseSet, Shapes, ShapeRegistry |
| `dynamics`    | SoA body storage (position, velocity, mass), Symplectic Euler integrator |
| `collision`   | Dynamic AABB tree (broad phase), SAT narrow phase, ContactManifold |
| `constraints` | Sequential impulse solver with warm starting and Coulomb friction |
| `world`       | Orchestrates the full pipeline |

**Handle system:** Users hold lightweight `BodyHandle` / `ShapeHandle` values (index + generation counter). Internally, a `SparseSet` keeps body data in contiguous dense arrays for cache efficiency, using swap-and-pop deletion. Stale handles from destroyed bodies are always detectable.

**Physics pipeline** (per `step(dt)`):
1. Apply forces (gravity + user forces)
2. Integrate velocities — Symplectic Euler: `v += (F/m + g) × dt`
3. Broad phase — AABB tree produces candidate pairs
4. Narrow phase — SAT produces `ContactManifold` (1–2 points per pair)
5. Solve constraints — 8–16 iterations, warm starting, Baumgarte position correction
6. Integrate positions — `x += v × dt`, `θ += ω × dt`

---

## Status

| Task | Description | Status |
|------|-------------|--------|
| 1 | Project scaffolding | ✅ |
| 2 | Core math types | ✅ |
| 3 | SparseSet | ✅ |
| 4 | ShapeRegistry | ✅ |
| 5 | BodyStorage + World basics | ✅ |
| 6 | Symplectic Euler integrator | ✅ |
| 7 | Dynamic AABB tree (broad phase) | ✅ |
| 8 | SAT narrow phase | ✅ |
| 9 | Sequential impulse solver | ✅ |
| 10 | Wire `World::step()` full pipeline | ✅ |
| 11 | State view + snapshot system | ✅ |
| 12 | Trajectory recorder + state export | ✅ |

---

---

# Stan2D（中文）

一个 C++20 的 2D 刚体物理引擎，专为机器学习和世界模型研究而设计。

**核心设计目标：** 跨平台确定性模拟、与 ML 张量兼容的列式（SoA）内存布局、热路径（`step()`）零堆分配、以及简洁的 Handle 句柄 API。

---

## 构建

**前置条件：** CMake 3.24+、[vcpkg](https://vcpkg.io) 安装于 `$HOME/vcpkg`、支持 C++20 的编译器（Clang / GCC / MSVC）

```bash
cmake --preset default
cmake --build build
```

若 vcpkg 中存在 SDL2，则会自动构建可选的调试渲染器（`stan2d_debug_renderer`）。

## 运行测试

```bash
ctest --test-dir build

# 带详细输出 / 运行单个测试套件
./build/stan2d_tests --gtest_filter=WorldBasicsTest.*
```

---

## 使用方法

### 1. 创建世界

所有内存在构造时预分配——模拟过程中不会发生堆分配。

```cpp
#include <stan2d/world/world.hpp>

using namespace stan2d;

World world(WorldConfig{
    .max_bodies      = 1000,
    .max_shapes      = 1000,
    .max_constraints = 500,
    .max_contacts    = 2000,
});

world.set_gravity({0.0f, -9.81f});
```

### 2. 创建形状

形状只需注册一次，可被多个刚体共享。

```cpp
// 圆形
ShapeHandle circle = world.create_shape(CircleShape{.radius = 0.5f});

// 胶囊形
ShapeHandle capsule = world.create_shape(CapsuleShape{
    .point_a = {0.0f, -0.5f},
    .point_b = {0.0f,  0.5f},
    .radius  = 0.1f,
});

// 多边形（最多 8 个顶点）
PolygonShape box{};
box.vertex_count = 4;
box.vertices[0] = {-0.5f, -0.5f};
box.vertices[1] = { 0.5f, -0.5f};
box.vertices[2] = { 0.5f,  0.5f};
box.vertices[3] = {-0.5f,  0.5f};
ShapeHandle poly = world.create_shape(box);
```

### 3. 创建刚体

**两步式**（复用已有 ShapeHandle）：

```cpp
BodyHandle body = world.create_body({
    .position  = {0.0f, 10.0f},
    .velocity  = {1.0f,  0.0f},
    .shape     = circle,
    .mass      = 1.0f,
    .body_type = BodyType::Dynamic,
});
```

**一步式**（内联形状，引擎内部管理 Handle）：

```cpp
BodyHandle body = world.create_body({
    .position   = {0.0f, 0.0f},
    .shape_data = CircleShape{.radius = 0.3f},
    .mass       = 2.0f,
});
```

**静态刚体**（地板、墙壁——无限质量，永不移动）：

```cpp
BodyHandle floor = world.create_body({
    .position  = {0.0f, 0.0f},
    .shape     = poly,
    .body_type = BodyType::Static,
});
```

**运动学刚体**（由用户控制速度，不受力影响）：

```cpp
BodyHandle platform = world.create_body({
    .position  = {0.0f, 2.0f},
    .velocity  = {1.0f, 0.0f},
    .shape     = poly,
    .body_type = BodyType::Kinematic,
});
```

### 4. 推进模拟并查询状态

```cpp
const float dt = 1.0f / 60.0f;

world.step(dt);

Vec2  pos  = world.get_position(body);
Vec2  vel  = world.get_velocity(body);
float mass = world.get_mass(body);
```

### 5. 销毁刚体

销毁后 Handle 自动失效，过期的 Handle 可被安全检测。

```cpp
world.destroy_body(body);

world.is_valid(body); // → false
```

---

## 架构

```
core → dynamics → collision → constraints → world
```

| 模块          | 职责 |
|---------------|------|
| `core`        | 数学类型（Vec2、Mat2、AABB）、句柄、SparseSet、形状、形状注册表 |
| `dynamics`    | SoA 刚体存储（位置、速度、质量）、辛欧拉积分器 |
| `collision`   | 动态 AABB 树（粗检测）、SAT 精检测、接触流形 |
| `constraints` | 序贯冲量求解器，支持温启动与库仑摩擦 |
| `world`       | 协调完整的物理管线 |

**句柄系统：** 用户持有轻量级的 `BodyHandle` / `ShapeHandle`（索引 + 代际计数器）。内部使用 `SparseSet` 将刚体数据保存在连续的密集数组中，删除时使用"交换删除（swap-and-pop）"策略。已销毁刚体的过期 Handle 始终可被检测。

**物理管线**（每次 `step(dt)`）：
1. 施加力（重力 + 用户力）
2. 速度积分——辛欧拉：`v += (F/m + g) × dt`
3. 粗检测——AABB 树生成候选碰撞对
4. 精检测——SAT 生成接触流形（每对 1~2 个接触点）
5. 约束求解——8~16 次迭代，温启动，Baumgarte 位置修正
6. 位置积分——`x += v × dt`，`θ += ω × dt`

---

## 实现进度

| 任务 | 描述 | 状态 |
|------|------|------|
| 1 | 项目脚手架 | ✅ |
| 2 | 核心数学类型 | ✅ |
| 3 | SparseSet | ✅ |
| 4 | ShapeRegistry 形状注册表 | ✅ |
| 5 | BodyStorage + World 基础 | ✅ |
| 6 | 辛欧拉积分器 | ✅ |
| 7 | 动态 AABB 树（粗检测） | ✅ |
| 8 | SAT 精检测 | ✅ |
| 9 | 序贯冲量求解器 | ✅ |
| 10 | `World::step()` 完整管线接入 | ✅ |
| 11 | 状态视图与快照系统 | ✅ |
| 12 | 轨迹录制与状态导出 | ✅ |
