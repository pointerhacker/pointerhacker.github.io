---
layout: wiki
title: 注意力机制
wiki_type: concepts
category: wiki
tags: [attention, transformer, llm, self-attention]
description: Transformer 的核心组件，通过计算 Query-Key-Value 的相关性实现动态上下文聚合
related: [transformer, kv-cache, flash-attention, rope]
updated: 2026-04-09
---

## 什么是注意力机制

注意力机制（Attention Mechanism）是 Transformer 架构的核心，允许模型在处理序列时动态地"关注"其他位置的信息，而不是依赖固定的循环结构。

## 核心公式

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

其中：
- $Q$（Query）：当前位置的查询向量
- $K$（Key）：所有位置的键向量
- $V$（Value）：所有位置的值向量
- $d_k$：Key 的维度（用于缩放，防止点积过大）

## 多头注意力

$$\text{MultiHead}(Q,K,V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h)W^O$$

$$\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$$

多头注意力允许模型从不同的表示子空间同时关注不同的信息。

## 计算复杂度

| 维度 | 复杂度 |
|------|--------|
| 时间 | $O(n^2 d)$ |
| 空间 | $O(n^2)$ |

其中 $n$ 为序列长度，$d$ 为模型维度。$n^2$ 的瓶颈是长序列的核心挑战，也是 [FlashAttention](/wiki/concepts/flash-attention/) 等优化工作的出发点。

## Self-Attention vs Cross-Attention

- **Self-Attention**：Q、K、V 来自同一序列，用于编码器/解码器内部
- **Cross-Attention**：Q 来自一个序列，K/V 来自另一个序列，用于编解码器之间

## 因果注意力（Causal Attention）

自回归语言模型使用因果注意力（也叫 masked attention），通过上三角掩码确保位置 $i$ 只能看到 $\leq i$ 的位置：

$$M_{ij} = \begin{cases} 0 & \text{if } j \leq i \\ -\infty & \text{if } j > i \end{cases}$$

## 关键优化方向

- **KV Cache**：推理时缓存 K/V 矩阵，避免重复计算 → [[kv-cache]]
- **FlashAttention**：IO 感知的注意力算法，大幅降低显存占用 → [[flash-attention]]
- **旋转位置编码（RoPE）**：通过旋转矩阵编码相对位置信息 → [[rope]]
- **多查询注意力（MQA/GQA）**：共享 K/V 头，降低 KV Cache 大小

## 参考资料

- Vaswani et al., "Attention Is All You Need" (2017)
- [一文搞懂KVCache](/blog/一文搞懂KVCache.html)
