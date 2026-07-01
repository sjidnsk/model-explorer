# model-explorer Runner 职责收敛与合同加固计划

## 目标

在第二轮架构债收口后，继续压缩剩余 runner 的职责范围：runner 只做 orchestration，不承载 manifest 解析、summary 合同、Markdown rendering、诊断、选择、统计或 artifact 构建逻辑。旧 CLI、旧脚本、旧 public import、JSON schema、fixtures 和报告关键字段保持兼容。

## 范围

1. 固化第二轮 model-explorer 子模块指针。
2. 拆分 `policy/path_feedback_runner.py`，把 manifest、summary、reports、diagnostics、artifacts、feedback selection 下沉到目标模块。
3. 拆分 `experiments/runner.py`，把 manifest、training matrix、selection、evaluation、reports、environment 下沉到目标模块。
4. 拆分 `experiments/quasi_real_matrix/runner.py`，把 ROI manifest、scenario generation、selection、metrics、reports 下沉到目标模块。
5. 增加 golden contract tests，固定 JSON summary 和 Markdown report 的关键字段。
6. 收紧 `verification.py`，增加 runner 行数限制、split module 反向导入检查和 private import 检查。
7. 更新当前架构和迁移文档。

## 验收标准

- `path_feedback_runner.py <= 800` 行。
- `experiments/runner.py <= 800` 行。
- `experiments/quasi_real_matrix/runner.py <= 700` 行。
- 旧 facade 和 `*_impl.py` 保持兼容 re-export，不重新承载业务长函数。
- 测试不从 runner 或 legacy target module 新增 `_private` helper import。
- `python -m model_explorer verify` 包含并通过第三阶段架构检查。
- 全量 pytest 和父仓 smoke 通过。

## 验证命令

```powershell
$env:PYTHONPATH='src'; python -m pytest tests/path_feedback tests/experiments tests/quasi_real tests/test_norm_architecture.py -q
$env:PYTHONPATH='src'; python -m model_explorer verify
$env:PYTHONPATH='src'; python -m pytest tests -q
```
