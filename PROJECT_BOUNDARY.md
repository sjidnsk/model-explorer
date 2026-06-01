# 项目边界：model-explorer

本文档是 `model-explorer` 的项目边界入口。更细的外部接口字段见
[`docs/external-interfaces.md`](docs/external-interfaces.md)。

## 文档地图

- `PROJECT_BOUNDARY.md`：定义本项目在建模底座、决策编排和执行层之间的职责边界。
- `docs/model-based-exploration.md`：整理基于模型的自主探索问题、算法闭环和模块划分。
- `docs/external-interfaces.md`：定义与 `dev-platform-constraints`、`path-planner` 的工程接口和契约稳定性规则。
- `docs/candidate-list-policy-baseline.md`：定义覆盖率优先的确定性候选排序基线和策略学习边界。
- `docs/rl-policy-network-design.md`：定义候选列表强化学习策略网络、训练样本、奖励函数和评估方式。

## 定位

`model-explorer` 是月面巡视探索研究原型的探索决策与闭环编排层，位于建模底座和执行层之间。

它回答的问题是：下一步应该观测哪里，为什么选这个目标，路径代价是否可接受，观测后是否需要重新排序或重规划。

三层分工如下：

| 项目 | 层次 | 主要职责 |
|---|---|---|
| `dev-platform-constraints` | 建模底座 | 表达环境、平台能力、地图可信度、硬约束、代价图和候选目标基础评分 |
| `model-explorer` | 决策编排 | 选择探索目标，融合路径反馈，编排观测更新，触发重规划，输出实验指标 |
| `path-planner` | 路径执行评估 | 将已选目标转换为路径请求，输出平台约束 A*、后处理、安全走廊、跟踪仿真、轨迹优化和 IRIS 区域图诊断 |

## 负责范围

- 探索主循环：`load scenario -> select goal -> evaluate path -> execute or simulate observation -> update belief -> rerank`。
- 目标选择策略：基于信息增益、任务价值、可信度提升、风险、路径代价和可达性选择下一个目标。
- 路径反馈融合：消费 `dev-platform-constraints` 的 `model-explorer-contract/v1` JSON 摘要，必要时调用 `path-planner` 进行路径可达性、代价、平台约束和轨迹后处理评估。
- 观测反馈编排：把执行或仿真观测请求交给 `dev-platform-constraints`，再消费其可信度更新报告。
- 重规划触发逻辑：根据目标排序变化、路径风险、观测冲突或执行失败决定是否重新选择目标。
- 实验场景与评价：定义端到端闭环场景，输出探索步数、目标变化、总代价、可信度提升和失败原因。

## 不负责范围

- 不重写 `GridMap`、核心地图层契约、平台参数、硬约束、代价图或可信度融合。
- 不把 `dev-platform-constraints` 中已有的候选目标基础评分复制一份；只在其稳定输出之上做策略级决策。
- 不把 `dev-platform-constraints` 的解释性报告接口当成在线重规划、任务状态机或 Hybrid A* 接口。
- 不实现 IRIS、GCS、Ackermann 或复杂车辆动力学轨迹优化；这些能力由 `path-planner` 在自己的后端边界内逐步承载。
- 不直接维护 `path-planner` 内部的 A*、安全走廊、跟踪仿真、轨迹优化、IRIS 区域图或后续 GCS 求解逻辑。
- 不让 `path-planner` 直接消费完整研究语义的 `GridMap`；语义地图必须先转换为 `path-planner-request/v1` 所需的代价、可通行掩膜、起终点和约束输入。

## 依赖边界

从 `dev-platform-constraints` 消费：

- `schema_version = model-explorer-contract/v1`
- 地图摘要：`grid.width`、`grid.height`、`grid.resolution`、`grid.frame_id`、`grid.origin`、`grid.layers`
- 约束摘要：`constraints.violation_count`、`constraints.passable_ratio`、`constraints.reason_counts`
- 离散目标：`top_goals.cell`、`top_goals.utility`、`top_goals.reachable`
- 多步序列：`top_sequences.cells`、`top_sequences.utility`、`top_sequences.coverage_area`
- 观测更新摘要：`observation_update`

向 `dev-platform-constraints` 请求：

- 生成或更新地图、约束和代价图
- 评估候选目标或目标序列
- 模拟或应用局部观测更新
- 输出契约报告与实验指标

向 `path-planner` 发送：

- `path-planner-request/v1` JSON
- 已选目标转换后的起点/目标栅格或世界坐标
- 代价图、可通行掩膜、障碍/约束派生层和地图规格
- 平台 key、平台配置路径、安全裕度、跟踪误差、仿真和优化开关

从 `path-planner` 接收：

- `path-planner-route/v1` JSON
- `reachable`、`geometric_path`、`path_cost`、`failure_reason` 和 `diagnostics`
- `postprocess` 中的安全走廊、平滑路径、曲率报告、可跟踪路径和跟踪安全报告
- 可选 `tracking_simulation_report`、`trajectory_optimization_report`、`region_graph_report` 和 `iris_region_report`

## 首个代码化目标

本目录已经从文档目录推进为可运行代码包，当前最小模块包括：

- `interfaces.py`：定义与底座和执行层交互的数据结构，字段以 `docs/external-interfaces.md` 为准。
- `scenario.py`：加载确定性探索场景。
- `loop.py`：实现单步或短序列探索闭环。
- `policy/planning.py`：提供合同代价、直线代理、轻量栅格 A* 和 `path-planner` 路由适配占位。
- `tests/`：验证“目标选择 -> 路径评估 -> 观测更新 -> 目标重排”的最小闭环。

## 当前阶段原则

`model-explorer` 仍是三项目联调的主断点。优先把 `dev-platform-constraints` 的契约输出转换为 `path-planner-request/v1`，消费 `path-planner-route/v1` 做目标重排和重规划触发；不要重做底层地图模型，也不要把执行层求解器搬进本项目。
