---
layout: wiki
title: FlashAttention
wiki_type: concepts
category: wiki
tags: [flash-attention, attention, optimization, gpu, io-aware]
description: IO 感知的精确注意力算法，通过分块计算大幅降低 HBM 访问次数，速度快 2-4×
related: [attention-mechanism, kv-cache, transformer]
updated: 2026-04-09
---

## 问题背景

标准注意力的时间复杂度 $O(n^2)$ 其实在 GPU 上还好，真正的瓶颈是**内存带宽**（HBM 访问）：

1. 计算 $S = QK^T$，写入 HBM：$O(n^2)$ 内存
2. 计算 $P = \text{softmax}(S)$，读取 S，写入 P
3. 计算 $O = PV$，读取 P

每步都需要读写 $n \times n$ 的矩阵到 HBM，极其耗时。

## FlashAttention 的核心思想

**分块（Tiling）+ 在线 Softmax**：将 Q/K/V 分成小块，在 SRAM（on-chip 快速缓存）中完成一块的全部计算，避免反复读写 HBM。

关键数学：softmax 可以分块增量计算。对于 $m_{\text{new}} = \max(m_{\text{old}}, m_{\text{block}})$：

$$o_{\text{new}} = \frac{e^{m_{\text{old}} - m_{\text{new}}} \cdot l_{\text{old}} \cdot o_{\text{old}} + e^{m_{\text{block}} - m_{\text{new}}} \cdot o_{\text{block}}}{l_{\text{new}}}$$

## 版本对比

| 版本 | 改进 | 速度提升 |
|------|------|----------|
| FlashAttention v1 (2022) | Tiling + 在线 softmax | 2-4× vs 标准 |
| FlashAttention v2 (2023) | 更优的并行策略、减少 non-matmul FLOP | 2× vs v1 |
| FlashAttention v3 (2024) | 针对 Hopper 架构（H100）优化 | 1.5-2× vs v2 |

## 内存复杂度

| | 时间 | 空间（HBM）|
|-|------|------------|
| 标准注意力 | $O(n^2 d)$ | $O(n^2)$ |
| FlashAttention | $O(n^2 d)$ | $O(n)$ |

时间复杂度相同，空间复杂度从 $O(n^2)$ 降到 $O(n)$！

## 应用场景

- 训练加速（尤其长序列）
- 减少显存占用，支持更长上下文
- 已集成进 PyTorch 2.0+（`F.scaled_dot_product_attention`）

## 参考资料

- Dao et al., "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness" (2022)
- [一文搞懂FlashAttention](/blog/一文搞懂FlashAttention copy.html)（博客文章）
- [FlashAttention伪代码分析](/blog/FlashAttention伪代码分析.html)（博客文章）
