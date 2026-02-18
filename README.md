# H-M-T-R 项目 / Project

基于 Julia 的 Hierarchical Mamba-Transformer-RNN 架构实现。
（说明：Stage 1 当前使用 **GRU-based NanoDecoder（RNN）**，尚未切换为 Mamba Decoder。）

Hierarchical Mamba-Transformer-RNN Architecture implementation in Julia.
(Note: Stage 1 currently uses **GRU-based NanoDecoder (RNN)**, not Mamba Decoder.)

## 项目结构 / Project Structure

- `hmtr.jl`: 统一入口 / Unified entry point.
- `src/HMTR.jl`: 主模块 / Main module.
- `src/model.jl`: 核心组件与 `HMTR_Stage1_AutoEncoder` / Core components and `HMTR_Stage1_AutoEncoder`.
- `src/train_stage1.jl`: Stage 1 训练逻辑（含分桶）/ Stage 1 training logic (bucketing).
- `src/infer_stage1.jl`: Stage 1 推理逻辑 / Stage 1 inference logic.
- `src/ve_model.jl`: VE 版本模型实现 / VE model implementation.
- `src/ve_train_stage1.jl`: VE 版本 Stage 1 训练 / VE Stage 1 training.
- `src/ve_infer_stage1.jl`: VE 版本 Stage 1 推理 / VE Stage 1 inference.
- `src/data.jl`: 数据预处理（Parquet -> JLD2 + 分桶）/ Data preprocessing (Parquet -> JLD2 + bucketing).
- `data/`: 数据集目录 / Dataset directory.

## 数据构建参数 / Data Construction Parameters

运行 `hmtr.jl data` 命令时可用的参数：
Available parameters when running `hmtr.jl data`:

| 参数 / Parameter | 类型 / Type | 默认值 / Default | 说明 / Description |
| :--- | :--- | :--- | :--- |
| `--data-dir` | String | `./data` | 包含 .parquet 文件的目录。<br>Directory containing .parquet files. |
| `--parquet-file` | String | (Auto) | 指定要加载的 Parquet 文件路径。若未指定，自动使用 `data-dir` 下的第一个。<br>Specific parquet file to load. Defaults to first file in `data-dir`. |
| `--tokenizer-name` | String | "" | HuggingFace 分词器名称（如 `bert-base-uncased`）。留空则使用字符级分词 (Char-level)。<br>HuggingFace tokenizer name. Leave empty for Character-level tokenization. |
| `--output-file` | String | (Auto) | 输出的 `.jld2` 文件路径。默认自动生成带时间戳的文件名。<br>Output .jld2 file path. Defaults to auto-generated name with timestamp. |
| `--max-docs` | Int | 0 | 限制处理的文档数量（用于测试）。0 表示处理所有文档。<br>Limit number of documents (for testing). 0 means process all. |
| `--char-vocab-docs` | Int | 10000 | 字符级模式下，用于构建词表的采样文档数。<br>Number of docs to sample for building vocab in Char-level mode. |

**示例 / Examples:**

1. **构建完整字符级数据集 (Build full char-level dataset):**
   ```bash
   julia --project=. hmtr.jl data --data-dir ./data --output-file ./data/processed_full.jld2
   ```

2. **使用 BERT 分词器构建测试集 (Build test set with BERT tokenizer):**
   ```bash
   julia --project=. hmtr.jl data --tokenizer-name bert-base-uncased --max-docs 100
   ```

## 快速开始 / Quick Start
1. **环境准备 / Setup**:
   安装 Julia 并初始化依赖：
   Ensure Julia is installed. Then instantiate the project:
   ```bash
   julia --project=. -e 'using Pkg; Pkg.instantiate()'
   ```

2. **开始训练 / Run Training**:
   ```bash
   julia --project=. hmtr.jl train_stage1
   ```
   ```bash
   julia --project=. hmtr.jl train_stage1_ve
   ```

详细参数与示例请参考 / See [COMMANDS.md](COMMANDS.md).

## REPL 指引 / REPL Guide

建议在项目根目录启动 REPL，配合 Revise 热重载：
Start REPL at project root with Revise hot reload:

```julia
using Revise
includet("src/HMTR.jl")
using .HMTR
```

之后可以直接调用：
Then run commands directly:

```julia
HMTR.main(["lint"])
HMTR.main(["typecheck"])
HMTR.main(["train_stage1", "--data-file", "data/processed_char_buckets_20260109_140845.jld2"])
```

如果 Revise 不稳定，可使用：
If Revise is unstable, use:

```julia
include("hmtr.jl")
```

更多参数示例请参考 / See [COMMANDS.md](COMMANDS.md).

## Stage 1：AutoEncoder / Stage 1: AutoEncoder

目标：训练 Mamba Encoder 将文本压缩为 capsules，并由 Mamba Decoder 重构。
Objective: Train the Mamba Encoder to compress text into capsules and the Mamba Decoder to reconstruct it.

- **输入 / Input**: 变长序列，分桶 8/16/32/64/128。
- **模型 / Model**: `MambaCompressor` -> `Capsules` -> `NanoDecoder (GRU)`.
- **损失 / Loss**: 重构交叉熵 / Reconstruction CrossEntropy.

## 下一步（Stage 2）/ Next Steps (Stage 2)

当 Stage 1 收敛后：
Once Stage 1 loss converges:
1. 冻结 `Encoder` 与 `Decoder` / Freeze `Encoder` and `Decoder`.
2. 提取全量 capsules / Extract capsules for the whole dataset.
3. 训练 `LatentReasoner` 预测下一 capsule / Train `LatentReasoner` to predict the next capsule.
