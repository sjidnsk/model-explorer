# Ubuntu Readiness

本文档记录 `model-explorer` 在共享 Conda 环境和 Ubuntu 目标环境中的验证、
训练 smoke 边界。Windows 本机验证用于日常开发和静态回归；Ubuntu 目标验证
用于确认 Linux shell、Python 运行方式和可选训练依赖在目标系统中可复制。

## Target

- Ubuntu 24.04
- Python 3.12
- 本项目只依赖本仓库代码和 `model-explorer-contract/v1` JSON 文件。
- Windows 本机默认使用 `D:\conda_envs\lunar-explorer`。
- 默认安装不强制安装 PyTorch；训练路径只在需要 smoke 或 checkpoint 时额外安装
  PyTorch。

## Environment

PowerShell：

```powershell
conda activate D:\conda_envs\lunar-explorer
$env:PYTHONPATH='src'
```

Linux shell 可使用等价 Conda 环境；运行项目代码时只需要把源码目录加入
`PYTHONPATH`，不需要安装 `model-explorer` editable 包：

```bash
conda activate lunar-explorer
export PYTHONPATH=src
```

训练 smoke 或 checkpoint 相关命令需要在该环境中额外安装 PyTorch；不要通过
安装本地 editable 包来获得训练入口。

## Verify

Linux shell 版本验证命令：

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
PYTHONPATH=src python -m model_explorer verify
git diff --check
rg "^\s*(from|import)\s+.*(a_gcs_ws|dev-platform-constraints|dev_platform_constraints)" src tests scripts
```

`model_explorer verify` 使用 Python 原生 forbidden import check，因此不依赖
PowerShell-only 命令，也不要求 Ubuntu 环境预装 `rg`。单独的 `rg` 命令只作为
人工验收时的额外静态检查。

## Boundary

Windows 本机验证：

- 日常运行 `PYTHONPATH=src python -m unittest discover -s tests -v`。
- 运行 `$env:PYTHONPATH='src'; python -m model_explorer verify`。
- 用于确认非训练路径、合成 benchmark smoke 和报告生成。

Ubuntu 目标验证：

- 使用上面的 Linux shell 命令重新验证。
- 只在需要训练 smoke 时安装 PyTorch，然后通过 `PYTHONPATH=src` 运行训练入口。
- 确认没有引入 `a_gcs_ws`、`dev-platform-constraints` 或 Windows-only 入口。

当前 benchmark 是 synthetic smoke / regression suite，不代表真实月面泛化能力。
