# 覆盖率优先的候选列表策略基线

本文档记录 `model-explorer` 后续候选点排序、仿真实验和策略网络训练的基线方案。该方案只定义设计边界和接口期望，不表示已经实现对应代码。

## 文档定位

本文档定义确定性候选排序基线，以及策略网络可以学习的候选列表边界。强化学习训练方式、网络结构、样本记录和验收指标的细化设计见
[`rl-policy-network-design.md`](rl-policy-network-design.md)。外部字段契约见
[`external-interfaces.md`](external-interfaces.md)。

## 背景

候选点排序不应只最大化单步 `utility`，还需要显式考虑探索覆盖率。这里的探索覆盖率指有效地图中已经被传感器有效观测并被底座接受更新的累计唯一面积占比。

覆盖率目标需要避免重复计数。同一区域被多次观测时，只能在第一次有效覆盖时增加累计覆盖面积；后续重复观测可以继续影响可信度或一致性指标，但不应提高探索覆盖率。

## 覆盖率定义

首版使用有效地图覆盖率：

```text
coverage_rate = covered_valid_area / total_valid_area
```

其中：

- `total_valid_area`：`valid_mask` 中所有有效栅格对应的面积。
- `covered_valid_area`：已经被有效观测过的有效栅格累计唯一面积。
- `coverage_rate_delta`：本次观测后覆盖率的实际增量。
- `expected_coverage_rate_delta`：候选点在执行前估计的覆盖率增量。

候选点排序应优先使用预计新增覆盖，而不是原始 `coverage_area`。`coverage_area` 只表示候选视场或序列覆盖面积，不能保证排除已覆盖区域。

## 职责边界

`dev-platform-constraints` 负责：

- 维护有效地图、已覆盖 mask 和覆盖率统计。
- 根据候选点的传感器 footprint 估计新增覆盖面积。
- 输出候选点的覆盖率增量、基础收益、风险、路径代价和可达性。
- 在观测更新后输出累计覆盖率和本步实际覆盖率增量。

`model-explorer` 负责：

- 消费 `top_goals` 候选列表并执行策略级排序。
- 将覆盖率增量作为主收益项纳入目标选择。
- 保留可达性、风险和路径代价等安全边界。
- 在仿真日志中记录候选特征、选择动作和执行后覆盖率变化。

第一版不要求 `model-explorer` 直接读取完整覆盖 mask，也不让策略网络直接从整张地图中选择任意栅格。

## 契约扩展

保持 `model-explorer-contract/v1` 稳定字段不破坏。以下字段作为实验字段追加或正式列入 `experimental_fields`。

`observation_update` 建议输出：

- `total_valid_area`
- `covered_valid_area`
- `coverage_rate`
- `coverage_rate_delta`
- `total_valid_cell_count`
- `covered_valid_cell_count`

`top_goals[]` 建议输出：

- `coverage_area`
- `expected_new_coverage_area`
- `expected_coverage_rate_delta`
- `risk`
- `path_cost`
- `energy_cost`

`top_sequences[]` 后续可扩展：

- `expected_new_coverage_area`
- `expected_coverage_rate_delta`

其中 `coverage_area` 已在当前报告中出现，但它不能替代新增覆盖面积。策略排序应优先使用 `expected_new_coverage_area` 或 `expected_coverage_rate_delta`。

## 排序基线

确定性排序基线采用候选列表策略。先处理硬约束和可达性，再计算综合策略分：

```text
score(g) =
  0.35 * norm(expected_coverage_rate_delta)
+ 0.20 * norm(information_gain)
+ 0.20 * norm(confidence_gain)
+ 0.15 * norm(value)
- 0.15 * norm(risk)
- 0.10 * norm(path_cost)
- 0.05 * norm(energy_cost)
```

排序规则：

1. 优先过滤或强惩罚 `reachable = false` 的候选点。
2. 对各项输入做归一化或限幅，避免面积、路径长度或价值尺度单项主导排序。
3. 当覆盖实验字段缺失时，回退到当前 `utility/reachable` 排序，保证旧契约兼容。
4. 确定性 tie-break 使用：

```text
score desc -> utility desc -> cell.x asc -> cell.y asc
```

该公式是首版基线，权重用于启动实验，不应作为最终固定参数。后续可通过消融实验或策略学习调整。

## 策略网络基线

策略网络第一版只在 `top_goals` 候选列表内输出目标概率或排序分数。

强化学习版本采用同一候选列表边界：网络只学习可达候选之间的偏好，不直接读取完整覆盖 mask，也不改变 `model-explorer-contract/v1` 稳定字段。具体 masked policy、PPO 奖励和样本格式见
[`rl-policy-network-design.md`](rl-policy-network-design.md)。

推荐输入特征包括：

- 候选点坐标和相对当前位置的距离。
- `expected_coverage_rate_delta`
- `expected_new_coverage_area`
- `information_gain`
- `confidence_gain`
- `value`
- `risk`
- `path_cost`
- `energy_cost`
- `reachable`
- 当前全局 `coverage_rate`

网络输出为候选点概率分布。执行时仍保留确定性安全约束：

```text
reachable hard filter -> policy probability rank -> deterministic tie-break
```

这样网络只学习候选点之间的策略偏好，不绕过底座的地图、硬约束、A* 可达性和风险解释。

## 仿真奖励

仿真轨迹应记录每一步：

- 当前 `top_goals` 候选列表及全部可用特征。
- 被选择的候选点索引。
- 执行后的 `coverage_rate_delta`。
- 可信度提升、任务价值覆盖、路径代价、风险和失败原因。
- episode 结束时的最终 `coverage_rate`。

策略网络奖励以累计有效地图覆盖率为主：

```text
reward_t =
  coverage_rate_delta
+ confidence_gain_reward
+ value_coverage_reward
- path_cost_penalty
- risk_penalty
- failure_penalty
```

如果任务目标更强调科学价值，可提高 `value_coverage_reward`；如果任务目标更强调安全，可提高风险和失败惩罚。

## 测试要点

`dev-platform-constraints` 应验证：

- 重复观测同一区域不会增加 `covered_valid_area`。
- `expected_new_coverage_area` 只统计未覆盖的有效栅格。
- `coverage_rate` 等于 `covered_valid_area / total_valid_area`。
- 覆盖率统计在无有效栅格、全部已覆盖和遮挡场景下行为稳定。

`model-explorer` 应验证：

- 高 `expected_coverage_rate_delta` 的可达候选能在综合分中胜出。
- 缺失覆盖实验字段的旧契约仍按当前 `utility/reachable` 行为工作。
- 不可达候选不会因为覆盖率高而越过硬约束。
- tie-break 在同分候选上保持确定性。

策略训练数据应验证：

- 每条样本包含候选列表、动作索引和执行后覆盖率奖励。
- 候选特征缺失时有明确的默认值或样本过滤规则。
- episode 级指标包含最终覆盖率、总代价、失败次数和重规划次数。

## 后续比较

候选列表策略是首版基线。它的优势是接口小、可解释、训练样本需求低，并能复用 `dev-platform-constraints` 已有候选生成、安全约束和路径代价评估。

多步序列策略可作为第二阶段。覆盖率最大化天然涉及累计去重，多步序列能更直接优化长期覆盖收益，但需要先稳定每个候选点的覆盖集合或新增覆盖估计，否则序列覆盖容易重复计数。

全图动作空间放到更后。它理论表达能力更强，可以发现候选生成器漏掉的目标，但需要完整多层地图张量、已覆盖 mask、动作 mask 和后验可达性过滤，训练成本和安全验证成本都更高。

推荐路线：

```text
候选列表策略基线 -> 多步序列策略 -> 全图动作空间策略
```
