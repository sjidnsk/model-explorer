# model-explorer 重构迁移说明

本说明用于规范 model-explorer 分层重构期间的导入路径和兼容策略，避免在迁移中破坏已有调用方。

## Public import 兼容

- 已发布或已被脚本、实验、测试使用的旧 public import 路径继续保留。
- 旧路径应通过 re-export 或薄 facade 转发到新的目标模块，保持函数名、类名和基础行为兼容。
- 兼容层不新增长期业务逻辑；如果必须临时补充适配逻辑，应保持范围小，并在后续迁移中下沉到目标模块。

## 新代码导入规则

- 新代码优先从目标模块导入，例如 `contracts`、`core/io`、`decision`、`planning`、`path_feedback`、`experiments`、`data`、`verification`。
- 只有在维护旧调用方、验证旧 API 或编写迁移测试时，才继续使用旧 public import 路径。
- 文档和示例应优先展示新模块路径，并在必要处标注旧路径仍兼容。

## 分阶段迁移流程

1. 先建立目标模块和合同边界，保持旧路径 re-export。
2. 将内部实现逐步迁移到目标模块，旧 facade 只转发。
3. 更新新代码、脚本和文档，使其优先导入新模块。
4. 使用 verification 检查依赖方向，特别是 `data` 不静态依赖 `policy`。
5. 在确认调用方完成迁移前，不删除旧 public 路径。

## 当前约束

- `training` 可选依赖继续只声明 `torch>=2.0`。
- 栅格数据能力使用独立的 `raster` 可选依赖声明 `Pillow>=10`。
- 聚合安装使用 `all`，包含训练和栅格能力所需依赖。

## 第二轮迁移规则

- 新测试不得再从 `model_explorer.policy.planning`、`model_explorer.policy.path_feedback`、`model_explorer.policy.experiment` 或 `model_explorer.data.evaluation_matrix` 导入 `_private` helper。
- 旧 private helper 若仍需覆盖，应通过新的目标模块暴露无下划线 API 后再测试。
- `data` 和 `decision` 层不得顶层静态导入 `policy`；需要策略 observation 时使用 `contracts.observations`。
- `*_impl.py` 与旧 facade 文件必须保持小型 compatibility shim，不承载新的业务实现。

## 第三轮 runner 迁移规则

- runner 私有 helper 不再作为测试入口。新增测试应从目标模块导入 public/internal API，例如 path feedback summary、report、diagnostics、artifacts，experiments selection、evaluation、report，quasi-real metrics、selection、report。
- 旧 facade 和 `*_impl.py` 只承担兼容 re-export。需要保留历史私有符号时，使用显式 `__all__` 与 allowlist shim，不得通过 `dir()` 广泛泄漏临时变量或导入对象。
- 生产代码不得从 legacy facade、`*_impl.py` 或 runner 回流导入已迁出的职责 helper。目标模块之间按职责依赖，目标模块不得静态导入自己的 runner。
- verification 已收紧 runner 行数限制、split module 反向导入检查和测试 private import 检查。新增 runner 逻辑前，应先判断它是否应下沉到 manifest、summary、reports、selection、metrics、diagnostics、artifacts、training matrix 或 evaluation 模块。

## 第四轮目标模块与测试治理规则

- 目标 facade 只保留兼容 re-export 和迁移说明，预算为 250 行；新拆出的职责模块预算为 800 行。超过预算前应继续按职责拆分，而不是压缩代码或把逻辑回流到 facade。
- 新 production module 不得从自己的 facade、runner 或对应 `*_impl.py` 导入；跨模块依赖应指向真实职责模块。
- 新测试不得从目标 facade 导入 `_private` helper。需要覆盖内部合同的，优先从 split module 导入无下划线 API；确需验证兼容私有符号时，只在专门的兼容性测试中断言 object identity。
- 不再使用 `__all__ = [name for name in globals() ...]` 这类动态导出。facade 和 split module 都应使用显式 allowlist，避免临时导入对象泄漏为 API。
- `tests/test_model_explorer.py` 只保留端到端 CLI、runner 和 integration smoke；path feedback selection、diagnostics、anchor projection 等 helper contract 放在 `tests/path_feedback/` 或 `tests/planning/`。
- `tests/test_quasi_real_data_pipeline.py` 只保留 quasi-real pipeline/integration smoke；selection decision、sample discriminativeness、decision diagnostics、quality gates、stability 等 helper contract 放在 `tests/quasi_real/`。
- `model_explorer verify` 已覆盖目标 facade 行数、split module 行数、反向导入、private import、动态 `__all__` 和巨型测试预算。新增模块或测试时应先跑 `python -m model_explorer verify`，再跑相关 pytest。
