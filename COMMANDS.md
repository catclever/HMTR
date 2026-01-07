# 🚀 HMTR 开发指南 (Interactive Workflow)

本项目强烈推荐使用 **Julia REPL** 进行交互式开发。
相比于每次通过命令行启动（需要重复经历 Julia 的启动和编译延迟），REPL 模式可以让代码**常驻内存**，且支持**热重载**，极大地提升开发效率。

## 🛠️ 1. 环境启动 (Setup & Hot Reload)

### 🚀 推荐流程（启用热重载）
强烈建议使用 `Revise.jl` 进行开发，这样修改代码后**无需重启**即可生效。

在 `julia>` 提示符下按顺序输入：

```julia
# 1. 加载 Revise (必须在加载代码前执行)
using Revise

# 2. 跟踪加载项目入口
# 注意使用 includet (include tracked)
includet("src/HMTR.jl")
```

此时，您可以直接运行任务命令。如果修改了 `src/` 下的代码（例如调整了 `get_target_lr` 或模型逻辑），再次运行命令时会自动使用新代码。

### 备用流程（无热重载）
如果 Revise 遇到问题，可以使用普通加载（每次修改代码需重启 REPL）：
```julia
include("hmtr.jl")
```

---

## � 2. 任务执行 (Tasks)

你可以通过 `HMTR.main(["command", ...])` 来模拟命令行调用。

### 📦 数据准备 (Data Preparation)

将 Parquet 数据转换为训练所需的 JLD2 格式。

```julia
# 基础用法 (自动处理 data/ 下的 .parquet 文件)
HMTR.main(["data"])

# 指定 Block Size (例如 64)
HMTR.main(["data", "--block-size", "64"])
```

### 🏋️ 训练 Stage 1 (AutoEncoder)

#### ⚡ 快速测试 (Debug Run)
用于验证代码逻辑，跑少量 Batch。
```julia
HMTR.main([
    "train_stage1",
    "--data-file", "data/processed_char_bs32_20260106_163247.jld2",
    "--epochs", "1",
    "--max-batches", "10",
    "--batch-size", "8",
    "--dim", "64",
    "--warmup-steps", "5",
    "--save-every", "0"
])
```

#### 🔥 正式训练 (Full Training)
```julia
HMTR.main([
    "train_stage1",
    "--data-file", "data/processed_char_bs32_20260106_163247.jld2",
    "--dim", "256",
    "--batch-size", "128",
    "--epochs", "10",
    "--lr", "1e-3",
    "--warmup-steps", "500",
    "--save-every", "2000",
    "--checkpoint-dir", "checkpoints",
    "--grad-clip-norm", "5.0",
    "--loss-spike-threshold", "10.0",
    "--skip-on-spike", "1"
])
```

**关键参数说明:**
- `--dim`: 模型维度 (默认 256)
- `--lr`: 学习率 (默认 1e-3)
- `--warmup-steps`: 预热步数 (默认 500)，在此期间 LR 线性增加
- `--grad-clip-norm`: 梯度裁剪阈值 (默认 5.0)
- `--loss-spike-threshold`: Loss 尖峰检测阈值 (默认 10.0)。若 Batch Loss 超过此值，将跳过更新。
- `--skip-on-spike`: 是否跳过尖峰 (1: 是, 0: 否)

#### 🔄 继续训练 (Resume)
中断后恢复训练（自动恢复权重、优化器状态和步数）。
```julia
HMTR.main([
    "train_stage1",
    "--data-file", "data/processed_char_bs32_20260106_163247.jld2",
    "--dim", "256",
    "--resume-ckpt", "checkpoints/ckpt_stage1_epoch2_step5000.jld2"
])
```

### 🤖 推理 (Inference)

#### 💬 交互模式
启动后可以直接在 REPL 中输入文本查看重构结果。
```julia
HMTR.main([
    "infer_stage1",
    "--checkpoint-file", "checkpoints/ckpt_stage1_latest.jld2",
    "--data-file", "data/processed_char_bs32_20260106_163247.jld2",
    "--interactive"
])
```

#### 📝 单次推理
```julia
HMTR.main([
    "infer_stage1",
    "--checkpoint-file", "checkpoints/ckpt_stage1_latest.jld2",
    "--data-file", "data/processed_char_bs32_20260106_163247.jld2",
    "--text", "你好，世界",
    "--force-cpu"
])
```

**关键参数说明:**
- `--interactive`: 进入交互式模式
- `--text`: 单次推理输入的文本
- `--force-cpu`: 强制使用 CPU (默认自动检测 GPU)

---

## 🖥️ 3. 命令行 (CLI) 备忘

如果你需要在服务器后台运行（非交互式），上述命令完全对应于 CLI 参数。
只需将 `HMTR.main([...])` 中的内容传给 `hmtr.jl` 即可。

**示例：后台运行训练**
```bash
nohup julia --project=. hmtr.jl train_stage1 \
  --data-file data/processed_char_bs32_20260106_163247.jld2 \
  --dim 256 \
  --epochs 10 \
  > train.log 2>&1 &
```
