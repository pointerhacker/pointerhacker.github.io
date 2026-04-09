---
layout: wiki
title: 旋转位置编码（RoPE）
wiki_type: concepts
category: wiki
tags: [rope, positional-encoding, transformer, llm]
description: 通过旋转矩阵在复数空间编码相对位置，支持长度外推，被 LLaMA/Qwen 等广泛采用
related: [attention-mechanism, transformer]
updated: 2026-04-09
---

## 动机

标准 Transformer 使用绝对位置编码（正弦/学习）。绝对编码的问题：
1. 泛化能力差，超过训练长度后性能急剧下降
2. 无法直接表达相对位置关系

RoPE（Rotary Position Embedding）通过将位置信息编码为旋转矩阵，使注意力得分自然地依赖于相对位置。

## 核心思想

对于位置 $m$ 的 Query 向量 $q_m$ 和位置 $n$ 的 Key 向量 $k_n$，RoPE 的目标是：

$$\langle f(q_m, m), f(k_n, n) \rangle = g(q_m, k_n, m-n)$$

即内积只依赖相对位置 $m-n$，而非绝对位置。

## 二维情形

将 $d$ 维向量视为 $d/2$ 个二维子空间，在第 $i$ 个子空间：

$$f(q, m)_{2i:2i+2} = \begin{pmatrix} \cos(m\theta_i) & -\sin(m\theta_i) \\ \sin(m\theta_i) & \cos(m\theta_i) \end{pmatrix} \begin{pmatrix} q_{2i} \\ q_{2i+1} \end{pmatrix}$$

其中 $\theta_i = 10000^{-2i/d}$（与原始 Transformer 的正弦编码基底相同）。

## 内积的相对性

$$\langle f(q,m), f(k,n) \rangle = \text{Re}\left[\sum_i (q_{[i]} k_{[i]}^*) e^{i(m-n)\theta_i}\right]$$

其中 $q_{[i]} = q_{2i} + iq_{2i+1}$。内积只依赖 $m-n$，性质得证。

## 长度外推

RoPE 的相对位置特性使其天然支持一定程度的长度外推。进一步改进：

| 方法 | 思路 |
|------|------|
| **YaRN** | 分段缩放 $\theta_i$ |
| **LongRoPE** | 非均匀缩放各维度 |
| **Dynamic NTK** | 推理时动态调整基底 |

## 采用 RoPE 的模型

- LLaMA / LLaMA-2 / LLaMA-3
- Mistral
- Qwen / Qwen2
- DeepSeek
- Gemma

## 参考资料

- Su et al., "RoFormer: Enhanced Transformer with Rotary Position Embedding" (2021)
- [旋转位置编码如此简单](/blog/旋转位置编码如此简单.html)（博客文章）
