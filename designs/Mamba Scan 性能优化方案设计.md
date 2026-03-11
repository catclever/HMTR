# HMTR：Stage2 Reasoner + Mamba Scan 性能优化方案设计

更新时间：2026-01-06

## 1. 背景

当前 Stage1 使用 Lux + Zygote 训练，Encoder 侧包含 Mamba 风格的 scan（按序列长度 L 做递推）。在 GPU 上，逐步 scan 循环通常会引入大量 kernel launch 与中间张量分配，导致 GPU 利用率偏低、吞吐不足；同时 Zygote 对循环与带状态递推的代码也存在已知的性能/可用性问题。

本设计文档评估两类可行路线，并给出建议的落地路径：

- 方案 A：切换 AD 引擎到 Enzyme（LLVM 层微分）
- 方案 B：为 scan 写自定义反向传播（ChainRules rrule），必要时配合融合 kernel

## 2. 目标与非目标

### 2.1 目标

- 提升训练吞吐，降低 AD 与循环递推带来的额外开销。
- 保持数值正确性：自定义梯度与参考实现一致（可接受一定误差）。
- 支持先在纯 CPU 环境验证正确性，再迁移到 GPU 做性能优化。
- 为后续 Stage2 Reasoner 训练提供可持续的工程路径。

### 2.2 非目标

- 本阶段不追求一次性实现“最优 kernel”，优先拿到可验证、可迭代的结构。
- 不承诺直接复用外部预训练 Transformer 权重（涉及对齐与收益不确定性）。

## 3. 现状概览（与本方案相关）

- Stage1 的向量宽度 `dim` 已参数化（默认 256，可在训练脚本 `--dim` 指定），并在非默认值时写入 ckpt 前缀。
- Encoder 输出 capsule 形状为 `[dim, L_cap, B]`，后续 Stage2 Reasoner 的 hidden size 需要与其对齐（`dim` 或投影到目标维度）。

## 4. 方案评估

## 4.1 方案 A：切换 AD 引擎（Enzyme）

### 核心思路

用 Enzyme 替代 Zygote，减少 “构建图 / pullback” 等高层 AD 的开销，提升包含循环的代码的求导效率与稳定性。

### 优点

- 工程改动相对集中：主要在训练入口与 AD 后端选择。
- 对循环代码在 CPU 上通常更友好（速度与稳定性）。
- 可作为“低成本试验”，快速判断 AD 是否是主要瓶颈。

### 局限（尤其对 GPU）

- 在 GPU 上，scan 循环的瓶颈通常来自 L 次 kernel launch 与中间张量分配；更换 AD 并不必然解决该执行形态问题。
- Enzyme 在 Julia + CUDA 的端到端训练体验相对更新，可能出现兼容性与调试成本。

### 适用场景

- 当前训练经常被 Zygote 的循环相关问题阻塞（报错/极慢）。
- 你想快速验证 “AD 是否是主要瓶颈”。

### 预期收益

- CPU：收益常见且明显。
- GPU：可能有改善，但通常难以触及 GPU 吞吐上限。

## 4.2 方案 B：手写 scan 的反向传播（ChainRules rrule）

### 核心思路

把 Mamba 的 scan 作为一个“自定义算子”，为其实现 `rrule`（解析梯度）。这样 AD 不需要展开整个循环构图，只把 scan 当作黑盒并使用手写的 backward。

### 优点

- 这是深度学习里非常通用的做法（custom op + backward），对 scan/RNN/DP 类任务普适。
- 可在 CPU 上先验证正确性与数值稳定性，再迁移到 GPU 做融合与性能调优。
- 一旦结合融合 kernel（forward/backward），最有机会把 GPU 利用率拉满。

### 代价与风险

- 工程复杂度高：需要严谨推导梯度、保存/复用必要的中间量、处理数值稳定性。
- 仅写 rrule 但不改 forward 执行形态，GPU 收益会被 “循环多次 launch” 限制；要达到高吞吐需进一步做融合。
- 需要为不同变体维护对应的 rrule（例如不同参数化/门控/是否加入额外项）。

### 适用场景

- 你把 scan/RNN 的吞吐作为核心目标（尤其是 GPU）。
- 你愿意为关键算子投入较高的工程成本，换取长期可复用的性能路径。

### 预期收益

- CPU：可显著减少 AD 展开/中间量开销。
- GPU：若配合融合 kernel，潜在收益最大。

## 5. 推荐结论

- 若目标是“快速减轻 Zygote 的循环痛点、尽快跑稳”：先做 A 的小试验有价值。
- 若目标是“长期把 4090 吞吐拉满、把 scan 变成可复用的高性能模块”：B 更对症，也更可能成为决定性提升。

建议采用“两段式路线”：

1) 先在 CPU 做 B 的正确性验证（rrule），建立可靠的梯度实现与测试框架；
2) 再把同一接口迁移到 GPU，逐步替换为融合实现（forward/backward kernel）。

## 6. B 方案落地设计

## 6.1 模块边界与接口

目标是把 scan 抽象为：

- `scan_forward(x, params...) -> y, aux`
- `scan_backward(ȳ, aux, x, params...) -> (x̄, params̄...)`

并通过 `ChainRulesCore.rrule` 暴露给 AD：

- forward 返回 `y`
- pullback 使用手写 backward 产出各输入的梯度

其中 `aux` 存放 backward 所需的最小中间量（可选 checkpointing：不存、少存、重算）。

## 6.2 CPU 正确性验证策略（强烈推荐先做）

在纯 CPU 环境完成以下验证：

- forward 对齐：与现有参考实现输出一致（相同随机种子/相同输入）。
- gradient 对齐：
  - 小尺寸下与 Zygote 的梯度对比（作为 baseline）
  - 或有限差分（finite difference）对 x 与关键参数做抽检
- 稳定性：在不同 dtype（Float32 / BF16 仅做前向或混合）下不出现 NaN/Inf。

建议从极小配置开始：

- `d_model` 取 4~16
- `d_state` 取 2~8
- `L` 取 8~32
- `B` 取 1~4

确保每一步都可复现、可定位。

## 6.3 GPU 性能路径（分阶段）

阶段 1：保持现有 forward（逐步循环），先接入 rrule，确认训练可跑且数值一致。

阶段 2：减少中间张量与广播（以 in-place/预分配为主），降低分配与同步。

阶段 3：融合 kernel（关键阶段）

- forward：将 scan 的时间步递推尽可能融合到更少的 kernel 调度中（取决于实现方式与并行策略）。
- backward：同理，避免把每个时间步的反传拆成大量小 kernel。

可选技术栈：

- CUDA.jl 自定义 kernel（最底层，控制力最大）
- KernelAbstractions.jl（可移植，但需要评估性能）
- 针对 scan 的专用实现（如果未来引入外部库/现成实现）

## 6.4 风险与缓解

- 梯度公式错误：通过小尺寸对比 + 抽样 finite difference 缓解。
- 显存占用：通过 checkpointing（少存中间量，重算）与分阶段优化缓解。
- 工程维护：将 scan 相关代码集中在单一模块文件中，确保接口清晰、测试覆盖到位。

## 7. 与 Stage2 Reasoner 的关系（维度对齐）

Stage2 的 Transformer Reasoner 若在 Julia/Lux 内实现，推荐直接使用 `dim` 作为 Transformer 的 d_model（或在 Reasoner 前后加投影以对齐外部模型维度）。

若计划尝试复用外部预训练 Transformer 权重，需要做维度对齐：

- 直接对齐：Stage1 `dim == H`（H 为外部模型 hidden size），但会显著增加训练开销与显存。
- 投影对齐：增加 `W_in: dim -> H` 与 `W_out: H -> dim`，保持 Stage1 小维度的同时对接外部 H 维 Transformer。

## 8. 里程碑（建议）

- M0：CPU 上 scan rrule 正确性验证通过（forward/grad 对齐）
- M1：GPU 上接入 rrule，训练可跑（不追性能）
- M2：减少分配/广播，吞吐改善明显
- M3：融合 forward/backward kernel，GPU 利用率显著提升

