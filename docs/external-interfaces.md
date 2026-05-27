# 外部接口：model-explorer

本文档定义 `model-explorer` 与 `dev-platform-constraints`、`a_gcs_ws-2.0.1` 的工程接口边界。项目职责边界入口见
[`../PROJECT_BOUNDARY.md`](../PROJECT_BOUNDARY.md)。

## 与 dev-platform-constraints

`dev-platform-constraints` 是建模底座。`model-explorer` 只消费它暴露给上层的解释性 JSON 摘要，并把地图、约束、可信度和候选目标基础评分视为外部输入。

### 消费的稳定契约

主接口是 `model-explorer-contract/v1`。`model-explorer` 必须能消费以下稳定字段：

| 字段 | 用途 |
|---|---|
| `schema_version` | 确认契约版本，当前固定为 `model-explorer-contract/v1` |
| `grid.width`、`grid.height`、`grid.resolution`、`grid.frame_id`、`grid.origin`、`grid.layers` | 建立目标坐标、世界坐标转换和可用图层认知 |
| `constraints.violation_count`、`constraints.passable_ratio`、`constraints.reason_counts` | 判断当前约束状态、风险解释和实验指标 |
| `top_goals[].cell`、`top_goals[].utility`、`top_goals[].reachable` | 选择或过滤单步探索目标 |
| `top_sequences[].cells`、`top_sequences[].utility`、`top_sequences[].coverage_area` | 评估多步目标序列 |
| `observation_update` | 读取观测更新后的可信度变化、可见格和更新格等报告字段 |

`top_goals[].cell` 和 `top_sequences[].cells` 使用栅格坐标 `[x, y]`。转换到世界坐标时使用：

```text
world_x = grid.origin[0] + x * grid.resolution
world_y = grid.origin[1] + y * grid.resolution
```

### 实验字段规则

`experimental_fields` 中列出的字段只用于解释研究原型和消融实验，例如 `information_gain`、`confidence_gain`、`risk`、`segment_path_costs`、`unreachable_reasons`。这些字段可以参与日志、可视化和调试，但不能作为 `model-explorer` 最小可运行闭环的必需依赖。

在 `model-explorer-contract/v1` 内，稳定字段不得删除、改名或改变语义。若稳定字段需要破坏性调整，应由 `dev-platform-constraints` 发布新的契约版本，并保留 v1 示例直到上层完成迁移。

### 请求方向

`model-explorer` 可以请求 `dev-platform-constraints` 执行以下底座能力：

- 生成或更新地图、硬约束、代价图和可信度层。
- 评估候选目标或目标序列。
- 模拟或应用局部观测更新。
- 输出契约报告、数据契约报告和实验指标。

`dev-platform-constraints` 不承诺在线重规划、完整自主探索状态机、Hybrid A*、IRIS、GCS 或车辆动力学执行接口。这些能力应由 `model-explorer` 编排或由 `a_gcs_ws-2.0.1` 提供。

## 与 a_gcs_ws-2.0.1

`a_gcs_ws-2.0.1` 是执行/局部轨迹层。它不理解完整研究语义地图，也不负责探索目标选择、任务价值、信息增益、可信度融合或观测更新。

### 发送给执行层

`model-explorer` 或适配层需要把上游语义输入转换为执行层需要的几何和约束输入：

| 逻辑字段 | 含义 |
|---|---|
| `source_pose` | 起点位姿，至少包含世界坐标位置和航向角 |
| `target_pose` | 目标位姿，由已选探索目标转换得到 |
| `path_skeleton` | 可选的局部路径骨架，通常来自全局路径或候选路径 |
| `obstacle_map` / `c_space` / `workspace_regions` | 从硬约束、障碍、可通行区域或安全走廊适配后的执行层输入 |
| `vehicle_params` | 车辆几何、速度、加速度、转向和曲率限制 |
| `trajectory_constraints` | 轨迹级速度、加速度、曲率、连续性和工作空间约束 |
| `cost_weights` | 时间、路径长度、能量或其他执行层代价权重 |

这些字段是 `model-explorer` 的适配协议，不要求 `a_gcs_ws-2.0.1` 当前直接暴露同名 JSON API。实现时应映射到该项目已有的数据结构，例如 `SE2ConfigurationSpace`、`EndpointState`、`VehicleParams`、`TrajectoryConstraints` 和 `AckermannGCSPlanner.plan_trajectory(...)`。

### 从执行层接收

执行层结果应被适配为以下逻辑字段：

| 逻辑字段 | 含义 |
|---|---|
| `feasible` | 局部轨迹是否可执行 |
| `trajectory` / `samples` | 优化轨迹对象或可序列化采样点 |
| `trajectory_cost` | 执行层报告的轨迹代价或可比较成本 |
| `path_length` | 轨迹长度或局部路径长度 |
| `solve_time` | 求解耗时 |
| `constraint_violations` | 速度、加速度、曲率、工作空间或连续性违反报告 |
| `solver_status` | 求解器状态或收敛原因 |
| `failure_reason` | 失败、不可达或约束违反的可解释原因 |

`model-explorer` 只能把这些结果用于路径反馈融合、重规划触发和实验指标，不应修改 `a_gcs_ws-2.0.1` 内部求解逻辑。

## 首版集成策略

首版 `model-explorer` 应先以 `dev-platform-constraints` 的 `model-explorer-contract/v1` 为主输入建立最小闭环。`a_gcs_ws-2.0.1` 作为可选局部可执行性检查接入：当全局候选目标已选定，并且需要验证局部轨迹可执行性时，再构造执行层适配请求。

在完整 Drake、IRIS、GCS 或 Ackermann 链路通过前，Windows 本机只做静态接口核对和轻量验证；最终轨迹链路通过必须在目标 Ubuntu 环境中确认。
