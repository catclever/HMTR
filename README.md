# H-M-T-R 项目 / Project

基于 Julia 的 Hierarchical Mamba-Transformer-RNN 架构实现。
**当前版本核心：联合训练 (Joint Training)**
Current Version Core: Joint Training

Hierarchical Mamba-Transformer-RNN Architecture implementation in Julia.
**Core Focus: Joint Training of Stage 1 (AutoEncoder) and Stage 2 (Latent Reasoner).**

## 项目结构 / Project Structure

- `hmtr.jl`: 统一入口 / Unified entry point.
- `src/HMTR.jl`: 主模块 / Main module.
- `src/model.jl`: 核心组件 / Core components.
- `src/train_stage_joint.jl`: **联合训练逻辑 (Joint Training Logic)**.
- `src/train_stage1.jl`: Stage 1 辅助逻辑 / Stage 1 auxiliary logic.
- `src/data.jl`: 数据预处理 / Data preprocessing.
- `data/`: 数据集目录 / Dataset directory.

## 联合训练参数 / Joint Training Parameters

运行 `hmtr.jl train_stage_joint` 命令时可用的参数。
Available parameters when running `hmtr.jl train_stage_joint`.

### 基础训练配置 / Basic Training Config

| 参数 / Parameter | 类型 / Type | 默认值 / Default | 说明 / Description |
| :--- | :--- | :--- | :--- |
| `--data-file` | String | `data/processed.jld2` | 训练数据文件路径。<br>Path to training data file. |
| `--meta-file` | String | (Auto) | 元数据文件路径。<br>Path to metadata file. |
| `--checkpoint-dir` | String | `checkpoints` | 检查点保存目录。<br>Directory to save checkpoints. |
| `--checkpoint-prefix` | String | `ckpt_stage_joint` | 检查点文件名前缀。<br>Prefix for checkpoint filenames. |
| `--epochs` | Int | 5 | 训练轮数。<br>Number of training epochs. |
| `--batch-size` | Int | 24 | 批次大小。<br>Batch size. |
| `--lr` | Float | 1e-3 | 学习率。<br>Learning rate. |
| `--warmup-steps` | Int | 500 | 预热步数。<br>Warmup steps. |
| `--max-batches` | Int | 0 | 每轮最大批次数（0为全部）。<br>Max batches per epoch (0 means all). |
| `--save-every` | Int | 0 | 每隔多少步保存一次（0为仅每轮保存）。<br>Save checkpoint every N steps. |
| `--grad-clip-norm` | Float | 5.0 | 梯度裁剪范数。<br>Gradient clipping norm. |
| `--loss-spike-threshold` | Float | 10.0 | 损失突增阈值。<br>Loss spike threshold. |
| `--skip-on-spike` | Bool | true | 损失突增时跳过更新。<br>Skip update on loss spike. |
| `--force-cpu` | Bool | false | 强制使用 CPU。<br>Force CPU usage. |
| `--add-timestamp` | Bool | true | 文件名添加时间戳。<br>Add timestamp to filenames. |
| `--use-parallel` | Bool | true | 是否启用并行计算优化。<br>Enable parallel computation optimization. |
| `--dtype` | String | "fp32" | 参数与状态精度（`fp32`/`fp16`/`bf16`）。显存紧张时建议优先尝试 `bf16`。<br>Parameter/state precision (`fp32`/`fp16`/`bf16`). Under memory pressure, try `bf16` first. |

### 模型架构参数 / Model Architecture Params

| 参数 / Parameter | 类型 / Type | 默认值 / Default | 说明 / Description |
| :--- | :--- | :--- | :--- |
| `--dim` | Int | 256 | 模型维度。<br>Model dimension. |
| `--mamba-d-state` | Int | 32 | Mamba 状态维度。<br>Mamba state dimension. |
| `--block-size` | Int | 8 | 压缩块大小。<br>Compression block size. |
| `--seq-len` | Int | 512 | 序列长度。<br>Sequence length. |
| `--k-streams` | Int | 4 | 并行流数量 (Best-of-K)。<br>Number of parallel streams. |
| `--heads` | Int | 8 | Transformer 头数。<br>Number of Transformer heads. |
| `--num-layers` | Int | 4 | 层数。<br>Number of layers. |
| `--k-drop-threshold` | Float | 0.0 | K流丢弃阈值（按流能量筛选）。`0.0` 等价于关闭 K-drop；建议先用 `0.0`，如需轻度稀疏可尝试 `0.01~0.05`。<br>K-stream drop threshold (energy-based stream pruning). `0.0` effectively disables K-drop; start with `0.0`, then try `0.01~0.05` for mild sparsification. |
| `--k-drop-min` | Int | 1 | 最少保留的流数量（防止全部被丢弃）。建议在 `1~2`；当 `k-streams` 较小（如 2~4）时通常设为 `1`。<br>Minimum number of streams to keep (prevents dropping all streams). Recommended `1~2`; with small `k-streams` (e.g. 2~4), usually set to `1`. |

### 损失权重与任务开关 / Loss Weights & Task Switches

| 参数 / Parameter | 类型 / Type | 默认值 / Default | 说明 / Description |
| :--- | :--- | :--- | :--- |
| `--recon-weight` | Float | 1.0 | 重构损失权重。<br>Reconstruction loss weight. |
| `--kl-weight` | Float | 0.0 | KL 散度权重。<br>KL divergence weight. |
| `--pred-weight` | Float | 0.3 | 预测损失权重。<br>Prediction loss weight. |
| `--var-dir-weight` | Float | 0.1 | 方差方向损失权重。<br>Variance direction loss weight. |
| `--var-mag-weight` | Float | 0.01 | 方差幅度损失权重。<br>Variance magnitude loss weight. |
| `--var-mag-low` | Float | 0.2 | 方差幅度惩罚下限。<br>Variance magnitude penalty lower bound. |
| `--var-mag-high` | Float | 3.5 | 方差幅度惩罚上限。<br>Variance magnitude penalty upper bound. |
| `--enable-recon-task` | Bool | true | 启用重构任务。<br>Enable reconstruction task. |
| `--enable-kl-loss` | Bool | true | 启用 KL 损失。<br>Enable KL loss. |
| `--enable-pred-task` | Bool | true | 启用预测任务。<br>Enable prediction task. |
| `--enable-var-dir-loss` | Bool | true | 启用方差方向损失。<br>Enable variance direction loss. |
| `--enable-var-mag-loss` | Bool | true | 启用方差幅度损失。<br>Enable variance magnitude loss. |

### 训练策略参数 / Training Strategy Params

| 参数 / Parameter | 类型 / Type | 默认值 / Default | 说明 / Description |
| :--- | :--- | :--- | :--- |
| `--teacher-forcing` | Bool | false | 是否启用 Teacher Forcing（默认关闭）。<br>Enable Teacher Forcing (disabled by default). |
| `--pred-warmup-frac` | Float | 0.1 | 兼容旧参数：当前主要作为 `pred-full-frac` 的默认回退值；建议优先使用 `pred-start-frac` + `pred-full-frac`。<br>Legacy compatibility: currently used mainly as fallback default for `pred-full-frac`; prefer `pred-start-frac` + `pred-full-frac`. |
| `--pred-start-frac` | Float | 0.1 | 预测损失开始生效的进度比例。<br>Progress fraction when prediction loss starts ramping. |
| `--pred-full-frac` | Float | 0.1 | 预测损失达到满权重的进度比例。<br>Progress fraction when prediction loss reaches full weight. |
| `--var-dir-start-frac` | Float | 0.6 | 方差方向损失开始生效的进度比例。<br>Progress fraction when variance-direction loss starts ramping. |
| `--var-dir-full-frac` | Float | 0.85 | 方差方向损失达到满权重的进度比例。<br>Progress fraction when variance-direction loss reaches full weight. |
| `--pred-select-mode` | String | "best" | 预测选择模式 (best/mean)。<br>Prediction selection mode. |

### 方差学习率控制 (新特性) / Variance LR Control (New)

用于解决方差被一侧主导的问题，独立控制 `var_head` 的学习率曲线。
Controls `var_head` learning rate independently to prevent one-sided variance dominance.

| 参数 / Parameter | 类型 / Type | 默认值 / Default | 说明 / Description |
| :--- | :--- | :--- | :--- |
| `--var-lr-base-scale` | Float | 1.0 | 方差层基础学习率缩放倍数。<br>Base LR scale for variance layer. |
| `--var-lr-start-frac` | Float | 0.0 | 方差学习率开始预热的进度比例 (0.0-1.0)。<br>Progress fraction to start variance LR warmup. |
| `--var-lr-full-frac` | Float | 0.0 | 方差学习率达到全速的进度比例。<br>Progress fraction to reach full variance LR. |
| `--var-lr-decay-start-frac` | Float | 1.0 | 方差学习率开始衰减的进度比例。<br>Progress fraction to start variance LR decay. |
| `--var-lr-end-scale` | Float | 1.0 | 衰减结束时的最终缩放倍数。<br>Final LR scale after decay. |

## 快速开始 / Quick Start

### 1. 环境准备 / Setup
```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

### 2. 联合训练 / Joint Training
```bash
julia --project=. hmtr.jl train_stage_joint --data-file data/processed.jld2
```

## REPL 指引 / REPL Guide

```julia
using Revise
includet("src/HMTR.jl")
using .HMTR

# 运行联合训练 (Run Joint Training)
HMTR.main([
    "train_stage_joint",
    "--epochs", "3",
    "--batch-size", "24",
    "--k-streams", "4",
    "--heads", "8",
    "--num-layers", "4",
    "--k-drop-threshold", "0.0",
    "--k-drop-min", "1",
    "--pred-select-mode", "best",
    "--pred-start-frac", "0.15",
    "--pred-full-frac", "0.55",
    "--var-dir-start-frac", "0.65",
    "--var-dir-full-frac", "0.90",
    "--var-lr-base-scale", "0.6",
    "--var-lr-start-frac", "0.0",
    "--var-lr-full-frac", "0.15",
    "--var-lr-decay-start-frac", "0.75",
    "--var-lr-end-scale", "0.5"
])
```
