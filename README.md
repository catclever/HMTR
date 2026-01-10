# H-M-T-R Project

Hierarchical Mamba-Transformer-RNN Architecture implementation in Julia.
(Note: Stage 1 now uses **Mamba Decoder** instead of RNN, achieving a fully Mamba-based AutoEncoder).

## Project Structure

- `hmtr.jl`: **Unified Entry Point**.
- `src/HMTR.jl`: Main module.
- `src/model.jl`: Defines the core components (MambaCompressor, MambaDecoder) and the `HMTR_Stage1_AutoEncoder`.
- `src/train_stage1.jl`: Training logic for Stage 1 (AutoEncoder with Bucketing strategy).
- `src/infer_stage1.jl`: Inference logic for Stage 1.
- `src/data.jl`: Data preprocessing logic (Parquet -> JLD2 with Length Bucketing).
- `data/`: Directory for datasets.

## Quick Start
1. **Setup**:
   Ensure Julia is installed. Then instantiate the project:
   ```bash
   julia --project=. -e 'using Pkg; Pkg.instantiate()'
   ```

2. **Run Training**:
   ```bash
   julia --project=. hmtr.jl train_stage1
   ```

See [COMMANDS.md](COMMANDS.md) for detailed usage instructions.

## Stage 1: AutoEncoder

Objective: Train the Mamba Encoder to compress text into "capsules" and the Mamba Decoder to reconstruct it.

- **Input**: Token sequences (Variable lengths, bucketed into 8, 16, 32, 64, 128).
- **Model**: `MambaCompressor` -> `Capsules` -> `MambaDecoder`.
- **Loss**: Reconstruction CrossEntropy.

## Next Steps (Stage 2)

Once Stage 1 loss converges:
1. Freeze `Encoder` and `Decoder`.
2. Extract capsules for the whole dataset.
3. Train `LatentReasoner` (Transformer) to predict the next capsule.
