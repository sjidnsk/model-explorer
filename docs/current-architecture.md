# model-explorer 当前架构

本文档描述 model-explorer 的目标分层和迁移期边界。当前代码仍保留部分旧模块路径；重构期间这些旧 facade 继续对外兼容，新代码优先落到目标层。

## 目标分层

- `contracts`：集中定义跨层共享的数据合同、字段名、版本标识和解析规则。其他层依赖它来保持输入输出可审计。
- `core/io`：承载基础接口、场景读取、manifest 读取和外部输入输出适配。这里负责把文件或外部数据转换成稳定合同对象。
- `decision`：负责候选动作选择、策略输出解释和决策结果封装。该层消费合同与规划反馈，不直接承担数据下载或训练流水线职责。
- `planning`：负责路径规划相关适配、候选可达性评估和规划器交互边界。默认规划能力保持可插拔，避免把具体 planner 细节泄露给决策层。
- `path_feedback`：负责把路径执行、覆盖、代价和可达性反馈转为决策或训练可消费的特征与诊断字段。
- `experiments`：负责实验配置、rollout、评估、训练入口和结果汇总。实验层可以编排多个下层能力，但不应反向成为基础数据或合同层依赖。
- `data`：负责地形、栅格、GeoTIFF、LOLA、评估矩阵等数据资产读取与转换。`data` 不应静态依赖 `policy` 或训练实现；需要策略相关解释时，应通过上层编排传入接口或配置。
- `cli/scripts`：负责命令行入口和一次性脚本编排。CLI 可以调用目标层 API，但业务逻辑应尽量下沉到对应模块。
- `verification`：负责架构边界、合同一致性和轻量 smoke 检查。验证代码应帮助发现静态依赖倒置、旧路径遗漏和文档声明漂移。

## 依赖方向

目标依赖方向是从外层编排指向内层合同和能力：

`cli/scripts`、`experiments` -> `decision`、`planning`、`path_feedback`、`data`、`core/io` -> `contracts`

其中 `data` 是基础数据能力层，不应静态导入 `policy`。策略、训练和实验逻辑可以使用 `data`，但 `data` 不能反向绑定训练或 policy 实现。

## 兼容原则

- 旧 public import 路径在迁移期继续可用，并通过 re-export 或薄 facade 转发到新模块。
- 旧 facade 只承担兼容和迁移说明职责，不继续承载新的核心实现。
- 新增代码优先导入目标模块；只有维护旧调用方时才使用旧 public 路径。
- 删除旧路径前必须先有迁移说明、验证覆盖和调用方审计结果。

## 第二轮架构债收口状态

- `policy/planning_impl.py`、`policy/path_feedback_impl.py`、`experiments/experiment_impl.py` 和 `experiments/quasi_real_matrix/evaluation_matrix_impl.py` 已退化为兼容 re-export 层，真实实现位于目标职责模块。
- `decision` 层不再静态依赖 `policy`；策略 observation 提取能力下沉到 `contracts.observations`，`policy.features` 仅保留兼容导出。
- Path feedback 的诊断、artifact、selection 入口分别暴露在 `path_feedback_diagnostics.py`、`path_feedback_artifacts.py` 和 `feedback_selection.py`。
- Experiment selection、training matrix、report、environment helper 使用 `experiments.*` 入口；旧 `policy.experiment` 只作为兼容 facade。
- Quasi-real matrix 的 manifest、runner、scenario generation、selection、reports 使用 `experiments.quasi_real_matrix.*` 入口；旧 `data.evaluation_matrix` 只作为兼容 facade。

## 第三轮 runner 职责收敛状态

第三轮后，三个 runner 只保留入口编排、I/O glue 和高层流程控制：

- `model_explorer.policy.path_feedback_runner`：保留 validate、dry-run、run、单 scenario orchestration。manifest 解析在 `path_feedback_manifest.py`，summary 合同在 `path_feedback_summary.py`，Markdown 在 `path_feedback_reports.py`，诊断在 `path_feedback_diagnostics.py`，artifact 和 triage 在 `path_feedback_artifacts.py`，反馈选择在 `feedback_selection.py`。
- `model_explorer.experiments.runner`：保留实验 manifest validate、dry-run、run 的 orchestration。manifest、training matrix、selection、evaluation、report、environment metadata 分别由同名目标模块承接。
- `model_explorer.experiments.quasi_real_matrix.runner`：保留 quasi-real matrix 的高层 validate、dry-run、run。ROI manifest、scenario generation、selection、metrics、report 分别由目标模块承接。

兼容层仍保留旧 public import 路径，但真实业务逻辑不得回流到 legacy facade 或 `*_impl.py`。当前 verification 对 runner 行数设硬限制：path feedback 和 experiment runner 不超过 800 行，quasi-real runner 不超过 700 行；目标模块也不得静态导入自己的 runner。
