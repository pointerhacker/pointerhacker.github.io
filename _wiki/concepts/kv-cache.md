---
layout: wiki
title: KV Cache
wiki_type: concepts
category: wiki
tags: [kv-cache, inference, optimization, attention, llm]
description: 推理时缓存 Key-Value 矩阵以避免重复计算，是自回归生成的核心优化
related: [attention-mechanism, transformer, flash-attention]
updated: 2026-04-09
---

## 为什么需要 KV Cache

自回归语言模型每次生成一个 token，新 token 需要与所有历史 token 做注意力计算。

若不缓存，每步 forward 都需要对整个历史序列重新计算 $K$ 和 $V$，复杂度为 $O(n^2)$（n 为当前序列长度）。

**KV Cache** 将已计算过的 $K$ 和 $V$ 缓存起来，每步只需计算新 token 的 $Q$、$K$、$V$，然后将新 $K/V$ 追加到缓存，最后对全部 $K/V$ 做注意力。

## 显存占用

每层每个 token 的 KV Cache 大小（FP16）：

$$\text{Size per token} = 2 \times 2 \times n_{\text{heads}} \times d_{\text{head}} \text{ bytes}$$

对于 LLaMA-7B（32 层，32 头，$d_{\text{head}}=128$）：

$$\text{Total} = L \times 2 \times 2 \times n_h \times d_h = 32 \times 2 \times 2 \times 32 \times 128 = 524\text{KB/token}$$

1K 上下文约需 512MB，16K 上下文约需 8GB——显存瓶颈往往在 KV Cache 而非模型权重。

## 优化方向

| 方法 | 思路 | 代表工作 |
|------|------|----------|
| **MQA** (Multi-Query Attention) | 所有 Q 头共享一组 K/V | Shazeer 2019 |
| **GQA** (Grouped-Query Attention) | K/V 头数 < Q 头数 | LLaMA-2, Mistral |
| **PagedAttention** | 非连续内存分页管理 KV Cache | vLLM |
| **量化 KV Cache** | INT8/FP8 量化 K/V | 多种工作 |
| **StreamingLLM** | 只保留 attention sink + 滑动窗口 | Xiao et al. 2023 |

## GQA 详解

GQA 将 $h$ 个 Q 头分成 $g$ 组，每组共享一对 K/V：

- $g = 1$：等价于 MQA（极度压缩）
- $g = h$：等价于标准 MHA（无压缩）
- $1 < g < h$：GQA（平衡质量和效率）

LLaMA-2-70B 使用 GQA，Q 头数 64，KV 头数 8（压缩 8×）。

## 参考资料

- [一文搞懂KVCache](/blog/一文搞懂KVCache.html)（博客文章）
- Shazeer, "Fast Transformer Decoding: One Write-Head is All You Need" (2019)
- Ainslie et al., "GQA: Training Generalized Multi-Query Transformer Models" (2023)
