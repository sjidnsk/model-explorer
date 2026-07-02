# model-explorer 全局导出与长函数瘦身计划

## 背景

前一阶段已经完成 target module 和测试治理，`tests/test_norm_architecture.py` 的真实仓库静态检查重新回到 GREEN。当前剩余结构债务集中在两类：

1. 多个 production module 使用 `__all__ = [name for name in globals() ...]`，会把临时导入名、工具名或实现细节泄漏到 public export 面。
2. 若干核心函数继续承载过多职责，后续维护和验证成本偏高。

本计划先固化 RED 检查，再分阶段清理导出和瘦身函数。Task 0 只增加治理规则与当前违规计数，不做实际清理。

## Task 0 范围

- 新增本计划文档。
- 在 `tests/test_norm_architecture.py` 中加入 RED 治理断言。
- 在 `src/model_explorer/verification.py` 中加入：
  - `no_dynamic_globals_all`：扫描所有 `src/model_explorer/**/*.py` production modules，禁止基于 `globals()` 生成 `__all__`。
  - `function_line_limit`：对指定函数执行行数预算检查。
- 真实仓库静态检查必须返回 RED，并且只包含当前已知两类违规。

## Task 0 非目标

- 不清理任何 `__all__`。
- 不拆分或瘦身任何长函数。
- 不改变 public API、runtime behavior、training、planner 或 experiment semantics。
- 不删除文件，不回退用户或其他任务的改动。
- 不要求 `python -m model_explorer verify` 在本任务通过。

## 当前 RED 合同

动态 `__all__` 当前应精确报告 11 个 `no_dynamic_globals_all` 违规：

- `src/model_explorer/experiments/environment.py`
- `src/model_explorer/experiments/evaluation.py`
- `src/model_explorer/experiments/manifest.py`
- `src/model_explorer/experiments/reports.py`
- `src/model_explorer/experiments/selection.py`
- `src/model_explorer/experiments/training_matrix.py`
- `src/model_explorer/policy/path_feedback_runner.py`
- `src/model_explorer/policy/planning_adapters.py`
- `src/model_explorer/policy/planning_routes.py`
- `src/model_explorer/policy/planning_types.py`
- `src/model_explorer/policy/planning_utils.py`

函数预算统一为 `<= 180` 行，当前应精确报告 10 个 `function_line_limit` 违规：

- `src/model_explorer/experiments/quasi_real_matrix/reports.py::_markdown_report`
- `src/model_explorer/experiments/reports.py::_markdown_report`
- `src/model_explorer/policy/path_feedback_summary.py::compact_path_feedback_summary`
- `src/model_explorer/experiments/training_matrix.py::_run_training`
- `src/model_explorer/verification.py::_run_architecture_static_check`
- `src/model_explorer/policy/path_feedback_backend_diagnostics.py::_sampled_region_path_diagnostics`
- `src/model_explorer/policy/path_feedback_reports.py::render_path_feedback_markdown`
- `src/model_explorer/policy/collector.py::collect_dynamic_rollout_episode`
- `src/model_explorer/policy/evaluation.py::_evaluate_strategy`
- `src/model_explorer/experiments/quasi_real_matrix/architecture_selection.py::_architecture_selection_summary`

## 后续执行顺序

1. Task 1：逐个模块把动态 `__all__` 改为显式 export 列表，并保持 legacy import path 兼容。
2. Task 2：按职责边界瘦身长函数，优先抽出纯数据整形、Markdown 渲染、诊断聚合和策略评估 helper。
3. Task 3：收敛真实仓库静态检查，从 RED 计数断言切回 GREEN 断言，并运行完整验证。

## 验收

Task 0 的验收命令：

```powershell
$env:PYTHONPATH='src'; python -m pytest tests/test_norm_architecture.py -q
```

预期结果：测试通过，但 `_run_architecture_static_check(Path.cwd())` 对真实仓库仍返回 `returncode=1`，且规则计数包含 `no_dynamic_globals_all: 11` 与 `function_line_limit: 10`。

## 执行结果

本计划已完成并切回 GREEN：

- 11 个基于 `globals()` 的动态 `__all__` 已全部替换为显式 allowlist。
- `verification.py::_run_architecture_static_check` 已拆为多个 scan helper，主函数保留架构检查编排。
- Experiment 与 quasi-real Markdown report builder 已拆到 section helper。
- Path feedback compact summary 与 Markdown report builder 已拆到 compact/report section helper。
- Training matrix 已拆为 dimensions、execution、outputs 与 orchestration，保持 PyTorch lazy import。
- 其余长函数已分别拆分：sampled-region diagnostics、rollout collector、baseline evaluation strategy、quasi-real architecture selection summary。

当前验收状态：

```powershell
$env:PYTHONPATH='src'; python -m pytest tests -q
$env:PYTHONPATH='src'; python -m model_explorer verify
```

两条命令均通过；`architecture_static_check` 中 `no_dynamic_globals_all` 与 `function_line_limit` violations 均为空。
