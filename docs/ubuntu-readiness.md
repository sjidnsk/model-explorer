# Ubuntu Readiness

本文档记录 `model-explorer` 在 Ubuntu 目标环境中的安装、验证和训练 smoke
边界。Windows 本机验证用于日常开发和静态回归；Ubuntu 目标验证用于确认
Linux shell、Python 包安装和可选训练依赖在目标系统中可复制。

## Target

- Ubuntu 24.04
- Python 3.12
- 本项目只依赖本仓库代码和 `model-explorer-contract/v1` JSON 文件。
- 默认安装不强制安装 PyTorch；训练路径通过 `model-explorer[training]`
  extra 安装 PyTorch。

## Install

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e .
```

训练 smoke 或 checkpoint 相关命令需要额外安装训练依赖：

```bash
pip install -e .[training]
```

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

- 日常运行 `python -m unittest discover -s tests -v`。
- 运行 `$env:PYTHONPATH='src'; python -m model_explorer verify`。
- 用于确认非训练路径、合成 benchmark smoke 和报告生成。

Ubuntu 目标验证：

- 使用上面的 Linux shell 命令重新安装和验证。
- 使用 `pip install -e .[training]` 后运行训练 smoke。
- 确认没有引入 `a_gcs_ws`、`dev-platform-constraints` 或 Windows-only 入口。

当前 benchmark 是 synthetic smoke / regression suite，不代表真实月面泛化能力。
