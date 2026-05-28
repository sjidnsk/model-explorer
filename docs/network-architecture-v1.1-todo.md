# Network Architecture v1.1 TODO

## Goal

把当前 masked candidate MLP v1 从“可运行训练原型”推进到“可比较、可复现、可验收的候选列表网络架构实验框架”。

本阶段不做全图端到端动作空间，不改变 `model-explorer-contract/v1` 稳定字段，不 import `dev-platform-constraints` 或 `a_gcs_ws-2.0.1`。

## Current Baseline

当前网络是 `mlp_v1`：

```text
candidate_features -> shared candidate_encoder -> candidate_embedding
global_features -> global_encoder -> global_embedding
concat(candidate_embedding, global_embedding) -> policy_head -> logits
logits + action_mask -> masked_logits -> policy distribution
masked_pool(candidate_embedding) + global_embedding -> value_head -> V(s)
```

主要实现位置：

- `src/model_explorer/policy/features.py`
- `src/model_explorer/policy/torch_policy.py`
- `src/model_explorer/policy/ppo.py`
- `src/model_explorer/policy/training.py`
- `src/model_explorer/policy/experiment.py`

## Non-Goals

- 不把网络输入扩展为完整地图张量。
- 不让策略选择 `top_goals` 之外的动作。
- 不让策略选择 `reachable=false` 或 padding action。
- 不把真实样本缺失解释为模型质量结论。
- 不要求 synthetic benchmark 上的训练策略必须优于 heuristic。

## Acceptance Rules

- `action_mask` 继续屏蔽不可达候选和 padding。
- 缺失实验字段时继续返回有限 reward、loss 和 metrics。
- checkpoint 必须记录 architecture、feature schema 和训练配置。
- 同 seed、同 manifest、同 architecture config 下结果可复现。
- report 必须能比较 `utility`、`coverage_heuristic`、`torch_policy`。
- `python -m unittest discover -s tests -v` 通过。
- `$env:PYTHONPATH='src'; python -m model_explorer verify` 通过。
- `git diff --check` 通过。

## Task Breakdown

### NA-0: Baseline Snapshot

目标：把当前 `mlp_v1` 的输入、输出、checkpoint 和验证边界固化为后续比较基线。

涉及文件：

- 修改：`docs/rl-policy-network-design.md`
- 修改：`tests/test_model_explorer.py`
- 修改：`tests/test_training_closure.py`

TODO:

- [x] 记录当前 `mlp_v1` 架构图和张量形状。
- [x] 增加测试确认 checkpoint metadata 包含当前 architecture 名称。
- [x] 增加测试确认旧 checkpoint 仍可加载。
- [x] 运行 `python -m unittest discover -s tests -v`。

验收：

- 当前 MLP 行为被命名为 `mlp_v1`。
- checkpoint 兼容旧格式。
- 不改变现有训练结果字段含义。

### NA-1: Observation Schema v1.1

目标：区分“真实 0.0”和“字段缺失 fallback 0.0”，避免训练信号污染。

涉及文件：

- 修改：`src/model_explorer/policy/features.py`
- 修改：`src/model_explorer/policy/rollout.py`
- 修改：`src/model_explorer/policy/rollout_io.py`
- 修改：`tests/test_model_explorer.py`
- 修改：`tests/test_training_closure.py`

TODO:

- [x] 在 `PolicyObservation` 中增加候选实验字段缺失指示。
- [x] 为 `expected_coverage_rate_delta`、`expected_new_coverage_area`、`information_gain`、`confidence_gain`、`value`、`risk`、`path_cost`、`energy_cost` 输出 missing indicator。
- [x] 确保 padding 行的 missing indicator 不会被当成有效候选特征。
- [x] 更新 rollout JSON 序列化和反序列化。
- [x] 增加缺失字段 fixture 回归测试。
- [x] 运行 `python -m unittest discover -s tests -v`。

验收：

- 缺失实验字段不会破坏旧 contract。
- `action_mask` 语义不变。
- 旧 rollout JSON 仍可读，缺失新字段时 fallback 到兼容默认值。

### NA-2: Feature Normalization Policy

目标：稳定输入尺度，减少不同 grid size、path cost、risk 尺度对训练的干扰。

涉及文件：

- 修改：`src/model_explorer/policy/features.py`
- 修改：`docs/rl-policy-network-design.md`
- 修改：`tests/test_model_explorer.py`

TODO:

- [x] 明确每个候选特征的归一化规则。
- [x] 明确每个全局特征的归一化规则。
- [x] 对 `grid_width`、`grid_height`、`violation_count`、`step_index`、`remaining_steps` 做稳定尺度处理。
- [x] 对 `risk`、`path_cost`、`energy_cost` 做限幅或候选内相对归一化。
- [x] 增加测试覆盖不同 grid size 下特征仍在可控范围。
- [x] 增加测试覆盖缺失代价字段时仍使用兼容 fallback。

验收：

- 归一化不改变 contract 字段语义。
- 所有 observation tensor 元素有限。
- 缺失字段和极端尺度 synthetic scenario 不产生非有限 loss。

### NA-3: Architecture Config and Registry

目标：允许通过 manifest 选择网络架构，而不是在代码中手动替换类。

涉及文件：

- 新增：`src/model_explorer/policy/architectures.py`
- 修改：`src/model_explorer/policy/torch_policy.py`
- 修改：`src/model_explorer/policy/training.py`
- 修改：`src/model_explorer/policy/experiment.py`
- 修改：`docs/experiment-manifest.md`
- 修改：`tests/test_training_closure.py`

TODO:

- [x] 定义 architecture 名称：`mlp_v1`、`mlp_missing_v1`、`candidate_attention_v1`。
- [x] 在 `train` manifest 中支持 `architecture` 字段。
- [x] checkpoint metadata 写入 `architecture`。
- [x] loader 根据 checkpoint metadata 构造对应网络。
- [x] 未指定 architecture 时默认使用 `mlp_v1`。
- [x] 未知 architecture 返回可读错误。
- [x] 增加 CLI experiment run 覆盖测试。

验收：

- 不配置 `train.architecture` 时旧 manifest 继续工作。
- 配置未知 architecture 时实验失败且错误 JSON 可读。
- checkpoint 可通过 metadata 恢复网络结构。

### NA-4: MLP Missing Indicators v1

目标：实现第一个可比较的新架构 `mlp_missing_v1`。

涉及文件：

- 修改：`src/model_explorer/policy/features.py`
- 修改：`src/model_explorer/policy/architectures.py`
- 修改：`src/model_explorer/policy/torch_policy.py`
- 修改：`tests/test_model_explorer.py`
- 修改：`tests/test_training_closure.py`

TODO:

- [x] 将 missing indicators 拼接进 candidate encoder 输入。
- [x] 保持 policy head 和 value head 与 `mlp_v1` 相同。
- [x] 增加测试确认 missing indicator 会改变输入维度。
- [x] 增加测试确认 masked logits 仍屏蔽不可达候选。
- [x] 增加训练 smoke 测试确认 loss、entropy、value 有限。

验收：

- `mlp_missing_v1` 能完成 checkpoint 保存和加载。
- 同一 manifest 可分别运行 `mlp_v1` 和 `mlp_missing_v1`。
- report 中能区分 architecture。

### NA-5: Candidate Attention v1

目标：在候选之间加入一层轻量上下文交互，验证候选相对关系是否改善策略表达。

涉及文件：

- 修改：`src/model_explorer/policy/architectures.py`
- 修改：`src/model_explorer/policy/torch_policy.py`
- 修改：`tests/test_model_explorer.py`
- 修改：`tests/test_training_closure.py`

TODO:

- [x] 实现 1 层 masked self-attention。
- [x] attention key padding mask 使用 `action_mask` 或独立 candidate-valid mask。
- [x] policy head 使用 contextual candidate embedding。
- [x] value head 使用 masked pooled contextual embedding。
- [x] 增加测试确认 padding candidate 不影响有效候选概率。
- [x] 增加训练 smoke 测试确认可反向传播。

验收：

- `candidate_attention_v1` 在 padding 和 unreachable case 中 mask-safe。
- 参数规模保持轻量，默认 hidden size 仍可用 64 或 128。
- 不要求 synthetic benchmark 指标优于 `mlp_v1`，只要求可比较。

### NA-6: Architecture Benchmark Matrix

目标：让 benchmark report 能按架构比较训练结果，而不是只输出单个 `torch_policy`。

涉及文件：

- 修改：`src/model_explorer/policy/experiment.py`
- 修改：`src/model_explorer/policy/benchmark.py`
- 修改：`docs/benchmark-readiness.md`
- 修改：`tests/test_daily_benchmark_entrypoints.py`

TODO:

- [x] 在 JSON summary 中记录 `training.architecture`。
- [x] 在 Markdown report 的 Training section 中显示 architecture。
- [x] 增加 architecture delta section，比较不同 architecture 与 baseline 的关系。
- [x] 增加固定 seed 的 architecture smoke manifest。
- [x] 增加测试确认无 PyTorch 时非训练 benchmark 仍通过。

验收：

- report 能说明当前训练使用哪个 architecture。
- benchmark 仍被标注为 synthetic smoke / regression suite。
- 不把 synthetic delta 解释为真实泛化能力。

### NA-7: Final Verification and Readiness Review

目标：完成 v1.1 设计闭环，给出是否进入更复杂策略学习阶段的判断。

涉及文件：

- 修改：`docs/network-architecture-v1.1-progress.md`
- 修改：`docs/rl-policy-network-design.md`

TODO:

- [x] 运行 `python -m unittest discover -s tests -v`。
- [x] 运行 `$env:PYTHONPATH='src'; python -m model_explorer verify`。
- [x] 运行 `git diff --check`。
- [x] 用 `rg` 检查无 `a_gcs_ws` 和 `dev-platform-constraints` import。
- [x] 更新进度文件中的 verification snapshot。
- [x] 总结是否进入 `candidate_attention_v2`、多步序列策略或真实 contract intake。

验收：

- 所有验证通过。
- 进度文件准确记录完成项、风险和下一步。
- 没有自动提交，除非收到明确提交指令。
