# RL 策略网络设计：候选列表覆盖率优先探索

本文档定义 `model-explorer` 中策略网络的首版设计。它把候选列表策略扩展为可训练的强化学习策略，但不改变 `model-explorer-contract/v1` 稳定字段，也不把网络推进到底层地图建模或执行层轨迹优化职责中。

## 1. 问题定义

策略网络回答的问题是：在当前 `top_goals` 候选列表中，下一步应该选择哪个可达目标。

首版网络只在 `dev-platform-constraints` 已经生成并评估过的候选目标内做策略级选择。它不直接读取完整多层地图张量，不从整张地图中生成任意目标，也不绕过硬约束、A* 可达性、风险解释和路径代价评估。

首版动作空间是候选目标索引：

```text
action = index in top_goals
```

执行边界固定为：

```text
reachable hard filter -> policy probability rank -> deterministic tie-break
```

这意味着 `reachable = false` 的候选目标和 padding 项必须被 action mask 屏蔽。缺失策略网络、缺失实验字段或网络输出不可用时，系统回退到确定性 `utility/reachable` 或覆盖率优先基线排序。

## 2. 架构选择

首版采用候选列表 masked policy，而不是全图端到端策略。

推荐训练方法为 PyTorch 实现的 masked PPO。PPO 的优势是交互式仿真成本可控、实现边界清楚，并且能同时训练 policy head 和 value head。若后续需要复用外部库，可以在保持相同 observation、action mask 和 reward 定义的前提下再接入 Stable-Baselines3 或其他框架。

不推荐首版直接使用全图 CNN、全图 Transformer 或图搜索替代网络，原因是：

- 当前稳定契约暴露的是 `top_goals` 和摘要字段，不是完整地图张量。
- 全图动作空间需要额外动作 mask、后验可达性过滤和更高训练成本。
- 安全边界应继续由 `dev-platform-constraints` 和执行层适配保证，网络只学习候选之间的偏好。

## 3. 状态、动作和奖励

每个训练 step 的 observation 由候选特征、全局特征和 action mask 组成。

候选特征来自 `top_goals[]`：

| 特征 | 来源 | 说明 |
|---|---|---|
| `cell.x`, `cell.y` | 稳定字段 | 栅格坐标，需归一化到地图宽高 |
| `relative_dx`, `relative_dy` | 派生字段 | 候选相对当前位置或上一目标的位置 |
| `relative_distance` | 派生字段 | 候选相对当前位置距离 |
| `utility` | 稳定字段 | 底座或确定性策略给出的基础效用 |
| `reachable` | 稳定字段 | action mask 的硬约束来源 |
| `expected_coverage_rate_delta` | 实验字段 | 预计新增覆盖率，是首要收益特征 |
| `expected_new_coverage_area` | 实验字段 | 预计新增覆盖面积 |
| `information_gain` | 实验字段 | 信息增益估计 |
| `confidence_gain` | 实验字段 | 可信度提升估计 |
| `value` | 实验字段 | 任务价值收益 |
| `risk` | 实验字段 | 地形、障碍、光照和可信度风险 |
| `path_cost` | 实验字段 | 候选路径代价 |
| `energy_cost` | 实验字段 | 能耗代价 |

全局特征来自 `grid`、`constraints` 和 `observation_update`：

| 特征 | 来源 | 说明 |
|---|---|---|
| `grid.width`, `grid.height`, `grid.resolution` | 稳定字段 | 地图尺度和分辨率 |
| `constraints.passable_ratio` | 稳定字段 | 当前可通行比例 |
| `constraints.violation_count` | 稳定字段 | 硬约束违规数量 |
| `coverage_rate` | 实验字段 | 当前累计有效覆盖率 |
| `step_index` | 闭环状态 | episode 内当前步数 |
| `remaining_steps` | 闭环状态 | episode 剩余预算 |

缺失实验字段时使用显式默认值：收益类字段默认为 `0.0`，代价类字段默认为候选内最大可用代价或 `0.0`，`coverage_rate` 默认为 `0.0`。这些默认值只用于保持旧契约兼容，不应被解释为真实观测值。

首版奖励以覆盖率增量为主：

```text
reward_t =
  1.00 * coverage_rate_delta
+ 0.20 * confidence_gain_reward
+ 0.20 * value_coverage_reward
- 0.10 * norm(path_cost)
- 0.20 * norm(risk)
- 1.00 * failure_penalty
```

其中 `coverage_rate_delta` 来自执行或仿真观测后的 `observation_update`。路径代价和风险项必须做归一化或限幅，避免尺度差异压制覆盖率目标。失败惩罚用于不可执行、观测失败、执行层返回不可行或 episode 提前终止的场景。

## 4. 样本获取

训练样本采用混合路线。

第一阶段以合成仿真为主。`dev-platform-constraints` 生成多种地图、障碍、坡度、崎岖度、光照、任务价值、平台参数和起点。每个 episode 执行：

```text
generate contract
-> extract observation and action mask
-> policy selects candidate index
-> simulate or apply observation update
-> compute reward
-> regenerate contract
-> continue until done
```

第二阶段引入真实月面数据。真实 DEM、影像或派生坡度/光照/障碍层应先适配到底座的多层地图表达，再通过同一 contract 输出进入 `model-explorer`。真实数据首要用途是评估和域随机化校准，只有在合成环境训练稳定后才用于微调。

每条训练 transition 应记录：

```text
observation
action_index
action_mask
log_prob
value
reward
next_observation
done
info
```

`info` 至少包含 selected cell、`coverage_rate_delta`、`path_cost`、`risk`、失败原因、episode 最终覆盖率、总代价、失败次数和重规划次数。

## 5. 网络设计

首版网络由四个部分组成。

`candidate_encoder` 对每个候选目标共享权重编码：

```text
candidate_features
-> Linear
-> LayerNorm
-> GELU
-> Linear
-> GELU
-> candidate_embedding
```

`global_encoder` 编码全局摘要：

```text
global_features
-> Linear
-> GELU
-> Linear
-> global_embedding
```

`policy_head` 对每个候选输出一个 logit：

```text
concat(candidate_embedding, global_embedding)
-> Linear
-> GELU
-> Linear(1)
-> masked logits
-> categorical policy
```

`value_head` 估计当前状态价值：

```text
masked_pool(candidate_embeddings)
-> concat(global_embedding)
-> Linear
-> GELU
-> Linear(1)
-> V(s)
```

默认隐藏维度为 64 或 128。候选数量默认固定到 `K = 16`；不足补 padding，超过则由底座或排序基线截断为 top-K。若后续发现候选之间的相互覆盖和去重关系对策略影响明显，再将 candidate encoder 升级为 1 到 2 层 self-attention。

## 6. 闭环流程

训练和执行共享同一策略边界：

```text
model-explorer-contract/v1
-> feature extraction
-> action mask from reachable and padding
-> policy network
-> selected top_goals[index]
-> optional execution feasibility check
-> observation update through dev-platform-constraints
-> reward and metrics
-> next contract
```

`model-explorer` 只记录和消费策略需要的摘要数据。地图更新、覆盖 mask、可信度融合、候选生成、A* 可达性和局部轨迹求解不在策略网络内部实现。

## 7. 评估与验收

策略网络必须与至少两个基线比较：

- 当前 `utility/reachable` 排序。
- 覆盖率优先的确定性候选列表排序。

核心指标包括：

| 指标 | 说明 |
|---|---|
| final `coverage_rate` | episode 结束时累计有效覆盖率 |
| cumulative `coverage_rate_delta` | episode 内累计新增覆盖 |
| total `path_cost` | 总路径代价 |
| average `risk` | 被选目标平均风险 |
| failure_count | 不可达、执行失败或观测失败次数 |
| replan_count | 重规划触发次数 |
| value_coverage | 高价值区域覆盖收益 |

验收条件：

- 不可达候选不会被策略采样或选中。
- 缺失实验字段时能回退到旧契约兼容行为。
- 固定随机种子下训练环境可复现。
- 策略网络在小规模合成场景能完成 rollout，并产生有限的 loss、entropy 和 value。
- 文档和实现均不要求修改 `model-explorer-contract/v1` 稳定字段。

