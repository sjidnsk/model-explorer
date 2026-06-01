# 外部接口：model-explorer

本文档定义 `model-explorer` 与 `dev-platform-constraints`、`path-planner` 的工程接口边界。项目职责边界入口见
[`../PROJECT_BOUNDARY.md`](../PROJECT_BOUNDARY.md)。

## 文档定位

本文档只定义外部接口和契约稳定性。候选排序、覆盖率策略和强化学习网络可以消费这里描述的稳定字段与实验字段，但不得要求破坏 `model-explorer-contract/v1` 稳定字段。策略网络设计见
[`rl-policy-network-design.md`](rl-policy-network-design.md)。

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

策略网络只能把实验字段作为可选特征。实验字段缺失时，`model-explorer` 必须继续支持基于 `top_goals[].utility` 和 `top_goals[].reachable` 的兼容选择行为。

### 请求方向

`model-explorer` 可以请求 `dev-platform-constraints` 执行以下底座能力：

- 生成或更新地图、硬约束、代价图和可信度层。
- 评估候选目标或目标序列。
- 模拟或应用局部观测更新。
- 输出契约报告、数据契约报告和实验指标。

`dev-platform-constraints` 不承诺在线重规划、完整自主探索状态机、Hybrid A*、IRIS、GCS 或车辆动力学执行接口。这些能力应由 `model-explorer` 编排或由 `path-planner` 提供路径评估与执行可行性诊断。

## 与 path-planner

`path-planner` 已取代 `a_gcs_ws-2.0.1` 作为本仓库内的路径执行评估层。它不负责探索目标选择、任务价值、信息增益、可信度融合或观测更新；它消费规划请求，输出可达性、路径、平台约束后处理、安全走廊、跟踪仿真、轨迹优化和 IRIS 区域图诊断。

### 发送给执行层

`model-explorer` 或适配层需要把上游语义输入转换为 `path-planner-request/v1`：

| 逻辑字段 | 含义 |
|---|---|
| `schema_version` | 固定为 `path-planner-request/v1` |
| `grid` / `resolution` / `origin` | 从 `model-explorer-contract/v1` 的地图摘要和图层派生 |
| `start` | 当前平台所在栅格或世界坐标 |
| `goal` | 已选探索目标转换后的栅格或世界坐标 |
| `cost` / `passable_mask` | 从 `dev-platform-constraints` 的代价图和硬约束摘要派生 |
| `terrain_layers` | 可选坡度、崎岖度、光照、可信度等结构化图层 |
| `platform` / `platform_config` | 平台 key 或来自 `dev-platform-constraints/configs/platforms/` 的配置路径 |
| `options` | 安全裕度、跟踪误差、仿真、优化、IRIS 区域图等开关 |

这些字段是 `model-explorer` 的适配协议，实现时应映射到 `path-planner` 现有 JSON loader、CLI 或后续 Python API。`model-explorer` 不直接 import `path_planner`，首版通过 JSON 文件或子进程边界联调，以保持测试夹具和训练 smoke 的轻量性。

### 从执行层接收

`path-planner-route/v1` 结果应被适配为以下逻辑字段：

| 逻辑字段 | 含义 |
|---|---|
| `feasible` | 通常由 `reachable`、后处理状态和可选仿真/优化状态综合得到 |
| `geometric_path` / `postprocess.smoothed_path` | 原始 A* 路径和平滑路径 |
| `path_cost` / `path_length` | 路径代价和长度，用于目标效用惩罚和实验指标 |
| `diagnostics` | 搜索模式、约束来源、阻塞统计和可解释失败信息 |
| `postprocess` | 安全走廊、曲率报告、可跟踪路径和跟踪安全报告 |
| `tracking_simulation_report` | 可选低速纯追踪仿真指标 |
| `trajectory_optimization_report` | 可选固定走廊轨迹优化结果和 fallback 状态 |
| `region_graph_report` / `iris_region_report` | 可选 IRIS/区域图诊断；当前不是 GCS 轨迹或车辆可行性证明 |
| `failure_reason` | 失败、不可达或约束违反的可解释原因 |

`model-explorer` 只能把这些结果用于路径反馈融合、重规划触发和实验指标，不应修改 `path-planner` 内部求解逻辑。

## 首版集成策略

首版 `model-explorer` 应先以 `dev-platform-constraints` 的 `model-explorer-contract/v1` 为主输入建立最小闭环。`path-planner` 作为路径评估层接入：当候选目标已选定或需要比较 Top-K 目标时，构造 `path-planner-request/v1`，消费 `path-planner-route/v1` 中的可达性、路径代价、后处理和诊断字段。

在完整 Drake、IRIS、GCS 或 Ackermann 链路通过前，Windows 本机只做静态接口核对和轻量验证；最终轨迹链路通过必须在目标 Ubuntu 环境中确认。当前推荐的联调顺序是：合同代价代理 -> `path-planner` CLI JSON -> Python API 适配 -> 可选 Drake/IRIS 后端。

## 当前适配实现

`model_explorer.policy.planning.PathPlannerRouteAdapter` 已提供首版 `path_planner_route` 后端：

- 使用 `build_path_planner_request_dict(...)` 从 `ModelExplorerContract`、当前起点和已选目标生成 `path-planner-request/v1`。
- 使用 `path_plan_result_from_route_dict(...)` 解析 `path-planner-route/v1`，把 `reachable`、`path_cost`、`failure_reason`、`diagnostics`、`postprocess`、`tracking_simulation_report`、`trajectory_optimization_report`、`region_graph_report` 和 `iris_region_report` 映射进 `PathPlanResult.metadata`。
- 默认通过 `python -m path_planner.cli` 子进程运行 sibling `path-planner`，并通过 `PYTHONPATH=<path-planner>/src` 注入源码路径；`model-explorer` 进程内不直接 import `path_planner`。
- 测试也支持 `route_json` fixture 模式，用于稳定验证 route 解析和奖励/失败指标反馈。

当前 `model-explorer-contract/v1` 稳定字段只包含地图摘要、候选目标和约束摘要，不包含完整 `cost` 与 `passable_mask` 数组。因此适配器支持两种输入来源：

1. 推荐联调路径：在 planner config 或 `PathPlanRequest.metadata` 中显式提供 `cost` 和 `passable_mask`。
2. 最小 smoke 路径：缺失数组时生成全 1 代价、全可通行的 open-grid fallback，并在 request metadata 中记录 `cost_source = open_grid_fallback` 和 `passable_mask_source = open_grid_fallback`。

open-grid fallback 只用于接口 smoke 和合成测试，不代表真实月面路径风险。进入 `.npz` 半真实地图验证时，应由 `dev-platform-constraints` 或中间导出脚本提供真实代价图和硬约束掩膜。

### path-planner sidecar

半真实验证推荐使用 `path-planner-sidecar/v1`：

| 字段 | 含义 |
|---|---|
| `schema_version` | 固定为 `path-planner-sidecar/v1` |
| `grid` | `width`、`height`、`resolution`、`origin`、`frame_id` |
| `cost` | 与 `path-planner-request/v1` 兼容的二维非负代价数组 |
| `passable_mask` | 与 `cost` 同 shape 的二维 bool 硬约束掩膜 |
| `terrain_layers` | 可选坡度、崎岖度、光照、可信度、障碍、通行性等诊断层 |
| `metadata` | 场景 ID、地图来源、平台 key、passable ratio 等 |

`model-explorer` manifest 中可配置：

```json
{
  "planner": {
    "backend": "path_planner_route",
    "path_planner_sidecar": "outputs/path_planner_sidecars/npz_shadow_corridor.path-planner-sidecar.json"
  }
}
```

`model_explorer.policy.planning.evaluate_candidate_paths(...)` 支持对 Top-K reachable goals 批量调用 `PathPlanningAdapter`，并通过 `path_feedback_summary(...)` 汇总不可达、路径代价、失败原因、重规划需求、postprocess fallback、tracking safety、trajectory optimization 和 region graph 诊断。

`model_explorer.policy.path_feedback` 提供半真实实验 summary 入口。Manifest 示例：

```json
{
  "schema_version": "path-feedback-manifest/v1",
  "top_k": 3,
  "planner": {
    "backend": "path_planner_route"
  },
  "scenarios": [
    {
      "scenario_id": "npz_shadow_corridor",
      "contract": "../dev-platform-constraints/outputs/path_planner_sidecars/npz_shadow_corridor.contract.json",
      "sidecar": "../dev-platform-constraints/outputs/path_planner_sidecars/npz_shadow_corridor.path-planner-sidecar.json",
      "current_cell": [0, 0]
    }
  ],
  "outputs": {
    "summary": "outputs/path-feedback-summary.json",
    "report": "outputs/path-feedback-summary.md"
  }
}
```

运行：

```bash
PYTHONPATH=src python -m model_explorer path-feedback validate path-feedback.json
PYTHONPATH=src python -m model_explorer path-feedback dry-run path-feedback.json
PYTHONPATH=src python -m model_explorer path-feedback run path-feedback.json
```

Summary 输出包含：

- `selected_cell_before_path_feedback`：原始 `top_goals` 中第一个 reachable 目标。
- `selected_cell_after_path_feedback`：路径反馈后按可达、非 replan、低 `path_cost`、低 `risk`、高 `utility` 重排得到的目标。
- `selection_changed_by_path_feedback`：路径反馈是否改变目标。
- `selected_path_cost_before_feedback`、`selected_path_cost_after_feedback`、`path_cost_delta_after_feedback`：用于逐场景比较 baseline 选择与路径反馈选择的路径代价差异。
- `selection_changed_count`、`selection_changed_rate`：用于实验级汇总路径反馈改变目标选择的频次和比例。
- `open_grid_fallback_used`：是否使用了 open-grid fallback；半真实可信实验应为 `false`。
- `path_planning_failure_count`、`replan_count`、`tracking_safety_violation_count`、`trajectory_optimization_fallback_count`、`region_graph_disconnected_count`。
- `iris_requested_count`、`iris_report_count`、`iris_status_counts`、`iris_fallback_count`、`iris_failure_count`、`iris_region_count_total`、`iris_fallback_reasons`：汇总可选 workspace IRIS 诊断。
- `region_graph_source_counts`、`region_graph_fallback_count`、`region_graph_fallback_reasons`、`region_graph_start_goal_disconnected_count`：汇总 `iris` vs `grid_box` graph source、fallback 和 start-goal 断连诊断。
- `scenario_group_summary`：按 smoke、stress、mixed_stress 或 unknown 聚合 candidate、reachable、failure、replan、selection changed、IRIS 和 region graph 指标。
- `coverage_per_path_cost`：覆盖率增量与路径代价的比值，用于比较“单位路径代价覆盖收益”。

`path-feedback run` 的 stdout 默认是紧凑摘要；完整 `scenarios`、Top-K 候选明细、baseline-vs-feedback、IRIS diagnostics、Region Graph diagnostics 和 scenario-group 对比仍写入 manifest 指定的 JSON 与 Markdown 报告。

执行证据与诊断特征应分开解释：

- 可作为执行评估证据：`reachable`、`path_cost`、`failure_reason`、`diagnostics.search_mode`、`postprocess.tracking_safety_report`、`trajectory_optimization_report.fallback_status`。
- 仅作为诊断特征：`region_graph_report`、`iris_region_report`、IRIS region count、region fallback ratio、graph source 和 fallback reason。它们不是 GCS trajectory，也不是 Ackermann/skid-steer feasibility proof；IRIS fallback 不应把成功 A* 路径改成 unreachable。只有当这些指标稳定解释路径失败或高风险暴露后，才应推进完整 GCS/Drake 替换当前 fallback 链。
