# model-explorer 第四阶段目标模块治理检查计划

## 背景

前三阶段已经收紧 legacy facade、`*_impl.py` 与 runner 的职责边界。当前仍有多个目标业务模块承担过多逻辑，尤其是 path feedback diagnostics、feedback selection、planning anchor/diagnostics、quasi-real selection，以及两个巨型测试文件。第四阶段先建立 RED 架构测试，锁定最终拆分目标和预算。

## 目标

1. 将目标 facade 的最终预算固定为 250 行。
2. 将计划拆出的 production split module 预算固定为 800 行。
3. 禁止 split module 反向导入其 facade、runner 或对应 `*_impl.py`。
4. 禁止 production code 和 tests 从目标 facade 导入 `_private` helper。
5. 禁止治理范围内 production module 使用基于 `globals()` 的动态 `__all__`。
6. 将 `tests/test_model_explorer.py` 和 `tests/test_quasi_real_data_pipeline.py` 纳入巨型测试预算。

## 范围

目标 facade:

- `src/model_explorer/policy/path_feedback_diagnostics.py`
- `src/model_explorer/policy/feedback_selection.py`
- `src/model_explorer/policy/planning_anchor.py`
- `src/model_explorer/policy/planning_diagnostics.py`
- `src/model_explorer/experiments/quasi_real_matrix/selection.py`

计划 split module:

- path feedback diagnostics: aggregate、interpretation、backend diagnostics、candidate audits
- feedback selection: types、scoring、channel、trainability、anchor、sources
- planning anchor: evaluation、projection、grid
- planning diagnostics: backend summaries、platform feasibility、diagnostic interpretation
- quasi-real matrix selection: quality gates、decision diagnostics、architecture selection、stability

巨型测试预算:

- `tests/test_model_explorer.py <= 5600`
- `tests/test_quasi_real_data_pipeline.py <= 1050`

## 非目标

- 不执行实际模块拆分。
- 不迁移业务函数。
- 不改变 public API、runtime behavior、reward、training、planner 或 experiment semantics。
- 不删除文件，不回退其他工作区改动。

## 验收

- `verification.py` 报告第四阶段治理 metadata 和 violations。
- `tests/test_norm_architecture.py` 包含临时 fixture 测试，覆盖 oversized facade、missing split module、split module forbidden import、production/test private import、dynamic `__all__`、giant test budget。
- 当前仓库保持预期 RED：真实目标文件 oversized，计划 split module 尚未存在，后续拆分任务负责转绿。
