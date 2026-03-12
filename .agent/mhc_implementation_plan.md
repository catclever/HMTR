# Implementation Plan: DeepSeek Manifold-Constrained Hyper-Connections (mHC) Architecture

## Goal Description
Implement the **Multiple Hypothesis Conclusion (MHC)** strategy using **Manifold-Constrained Hyper-Connections (mHC)**, inspired by DeepSeek V3 architecture.
This architecture forms the core of **Stage 2 (Latent Reasoning)**. It replaces the standard Transformer with a parallelized structure that:
1.  **Instruction Operator**: Generates $K$ distinct initial hypotheses based on instruction semantics.
2.  **MHC Reasoner**: Processes these hypotheses simultaneously using Doubly Stochastic Mixing.
3.  **Training**: The Operator and Reasoner are trained **end-to-end** in Stage 2, with the Stage 1 VAE (Encoder/Decoder) frozen.

## User Review Required
> [!IMPORTANT]
> This plan implements the full mHC architecture:
> 1.  **Parallel Streams**: The Reasoner maintains $K$ parallel latent states $[D, K, B]$.
> 2.  **Hyper-Connections**: Between layers, streams are mixed using a learnable matrix $M$.
> 3.  **Manifold Constraint**: $M$ is constrained to be **Doubly Stochastic** (rows/cols sum to 1, non-negative) using Sinkhorn-Knopp normalization.
> 
> Confirm this aligns with your "Low-cost exploration" goal (running $K$ Reasoner paths in parallel on latent space).

## Proposed Changes

### 1. Stage 2 核心架构：通用的多流自回归预测 (Transformer + mHC)
这是 Stage 2 的基础骨架。
- **任务目标**：纯通用的自回归预测（Next-Capsule Prediction）。给定前面的胶囊 $z_{1:t}$，预测 $z_{t+1}$。
- **结构设计 (`MHCLatentReasoner`)**：
  - 取代原本单一的 Transformer，改为支持 $K$ 条并行流（Stream）的网络。
  - $K$ 的大小是参数化的，在初始化时可以设定为定值（如 $K=4$ 或 $K=8$），未来也可扩展为动态。
  - **模块化 `MHCBlock`**：在标准的 Transformer 块（Self-Attention + FFN）之间，插入一个**混合层（Mixing Layer）**。
  - 混合层通过一个学习到的双随机矩阵 $M_{K \times K}$（由 `sinkhorn_knopp` 约束），让不同流在每步推理中交换信息。

### 2. 初始混合层与多流输入 (Dynamic Multi-Stream Input)
为了让 `MHCLatentReasoner` 起步阶段就有多个不同的假设基础，需要在最开始设计一个分流机制。
- **结构设计**：在 Stage 1 给出单流胶囊 $z_{base}$ 后，第一层就要做分裂。
- **实现方式**：引入一个**初始混合层（Initial Mixing Layer）**。在没有额外任务指令干预时（通用预测），可以通过微小的可学习偏置或噪声，将 $z_{base}$ 衍生为初始的 $K$ 条流 $Z_{1:K}$，输入给 Reasoner。

### 3. 指令编码与任务算子 (Task Encoder & Operator)
这是用来解决“各种不同任务要求”的关键层（即之前的指导层）。
- **任务目标**：将文字形式的“任务要求”（Instruction/Task）编码为算子，作用于输入胶囊。
- **结构设计 (`TaskEncoder` & `TaskOperator`)**：
  - 使用一个新的 **Mamba 编码器** 专门处理任务文本，提取出任务的高维表达 $E_{task}$。
  - 将 $E_{task}$ 转化为一个操作算子（例如低秩矩阵 $U V^T$ 或者通道缩放参数）。
  - **交互计算**：在步骤 2 的初始混合层中，或者作为第一步，用这个算子对原始输入 $z_{base}$（或 $\mu, \sigma$）进行调制（Modulation），产生带有强烈任务倾向的 $K$ 个假设流。
- **训练设计**：如前述讨论，为了保持步骤1中 Transformer 的通用性，在训练时会混入无指令数据（Null Instruction）。

### 4. TTT (Test-Time Training) 的集成
TTT 是一种动态适应机制，让模型在推理阶段也能更新自身状态。
- **与通用的 Transformer 在一起还是分开？**
  - **建议紧密结合（Integrated）**：DeepSeek V3 的论文中，TTT 往往被设计为标准 Attention/FFN 的替代品，形成双路架构（Dual Path）。
  - **结构设计**：在 `MHCBlock` 内部，我们可以设计一个分支为普通的 Self-Attention，另一个分支为 TTT-Linear 层。两条流的输出结果在混合层再次汇聚。这样 TTT 就成为了 Transformer 架构内部的一种新型“动态记忆层”。

---

## 实施路径与第一步确认

这 4 个步骤逻辑非常清晰，为了稳妥推进，我分析了它们的**依赖关系**：
*   **步骤 1** 是地基。没有它，多流、算子、TTT 都无从施展。
*   **步骤 2** 是步骤 1 的前置接口。
*   **步骤 3 (任务编码)** 是上层建筑，依赖于有通用模型可以接受它的输入。
*   **步骤 4 (TTT)** 是复杂度的极度攀升，需要基础的 `MHCBlock` 已经运作良好。

### 第一步应该做什么？

**强烈建议第一步只做【步骤 1 + 步骤 2 的最简版】**，目标是跑通：**“一个纯净的 $K$ 路 Transformer，基于已有的无监督 Stream数据，完成 Next-Capsule 预测”**。

**具体动作（First Step）：**
1.  **在 `src/model.jl` 中编写 `sinkhorn_knopp` 算法**，确保我们能生成双随机矩阵 $M$。
2.  **构建 `MHCBlock`**：写一个包含普通 Transformer 注意力 + Sinkhorn 混合层的结构。
3.  **构建 `MHCLatentReasoner`**：堆叠几层 `MHCBlock`，它的输入是 $[Dim, K, L_{seq}, Batch]$，输出同理。
4.  **编写验证脚本 `verify_mhc.jl`**：构造随机假数据，跑一个前向传播，证实 $K$ 条流确实在 $M$ 矩阵的作用下进行了信息交换。

**为什么这样安排？**
一旦这个无任务指令的 "N路未来预测器" 能稳定收敛 Loss，就证明了我们 Stage 2 的主干引擎没有架构 Bug。之后再引入步骤 3 的 Mamba 任务编码器，就变成了顺理成章的“条件输入（Conditional Input）”添加。

### Manual Verification
- Run `julia --project verify_mhc.jl` and confirm constraint satisfaction.
