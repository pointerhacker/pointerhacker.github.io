---
layout: wiki
title: Transformer 架构
wiki_type: concepts
category: wiki
tags: [transformer, architecture, attention, llm]
description: "Attention Is All You Need" 提出的序列建模架构，现代大语言模型的基础
related: [attention-mechanism, rope, kv-cache]
updated: 2026-04-09
---

## 概述

Transformer 是 Vaswani 等人于 2017 年提出的神经网络架构，完全基于注意力机制，摒弃了 RNN/CNN 结构。它是 BERT、GPT 系列等现代大语言模型的基础。

## 架构组成

### Encoder（编码器）

每层包含：
1. **Multi-Head Self-Attention**（多头自注意力）
2. **Add & Norm**（残差连接 + Layer Normalization）
3. **Feed-Forward Network**（FFN，两层线性变换 + ReLU）
4. **Add & Norm**

### Decoder（解码器）

每层包含：
1. **Masked Multi-Head Self-Attention**（因果自注意力）
2. **Add & Norm**
3. **Cross-Attention**（与 Encoder 输出交叉注意力）
4. **Add & Norm**
5. **Feed-Forward Network**
6. **Add & Norm**

### Decoder-Only 架构

现代大语言模型（GPT、LLaMA、Qwen 等）通常只用 Decoder，去掉 Cross-Attention，使用因果注意力实现自回归生成。

## 位置编码

原始 Transformer 使用正弦/余弦位置编码：

$$PE_{(pos, 2i)} = \sin(pos / 10000^{2i/d_{model}})$$
$$PE_{(pos, 2i+1)} = \cos(pos / 10000^{2i/d_{model}})$$

现代 LLM 通常使用 RoPE（旋转位置编码）替代。→ [[rope]]

## FFN 的作用

FFN 的参数量约占整个模型的 2/3，被认为是模型的"知识存储"部分：

$$\text{FFN}(x) = \max(0, xW_1 + b_1)W_2 + b_2$$

现代变体通常使用 SwiGLU 激活函数：

$$\text{SwiGLU}(x) = \text{Swish}(xW_1) \odot (xW_2)$$

## 关键超参数

| 参数 | 说明 | 典型值 |
|------|------|--------|
| `d_model` | 模型隐藏层维度 | 4096（7B 模型） |
| `n_heads` | 注意力头数 | 32 |
| `n_layers` | Transformer 层数 | 32（7B 模型） |
| `d_ffn` | FFN 中间层维度 | 4×d_model 或 8/3×d_model |
| `context_length` | 最大序列长度 | 2K~128K |

## 参考资料

- Vaswani et al., "Attention Is All You Need", NeurIPS 2017
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
