---
layout: wiki
title: "Attention Is All You Need (2017)"
wiki_type: sources
category: wiki
tags: [paper, transformer, attention, vaswani]
description: 提出 Transformer 架构的奠基性论文，2017年 NeurIPS，作者 Vaswani 等
related: [transformer, attention-mechanism]
updated: 2026-04-09
---

## 基本信息

- **标题**：Attention Is All You Need
- **作者**：Vaswani, Shazeer, Parmar, Uszkoreit, Jones, Gomez, Kaiser, Polosukhin
- **发表**：NeurIPS 2017
- **引用量**：100,000+（截至 2025）

## 核心贡献

1. **提出 Transformer 架构**：完全基于注意力机制的 Seq2Seq 模型，无 RNN/CNN
2. **多头注意力**：$h=8$ 个并行注意力头，$d_k=d_v=64$
3. **位置编码**：正弦/余弦固定编码
4. **在机器翻译上达到 SOTA**：英德翻译 BLEU 28.4

## 架构细节

- 编码器：6层，$d_{\text{model}}=512$，$d_{ff}=2048$，$h=8$
- 解码器：6层，同上，增加 Cross-Attention
- 优化器：Adam，带 warmup 的学习率调度

## 关键公式

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

详见 [[attention-mechanism]]。

## 历史意义

这篇论文彻底改变了 NLP 和 AI 领域：
- 成为 BERT、GPT 等预训练模型的基础
- 后来扩展到 Vision Transformer（ViT）、扩散模型等
- 8位作者后来均离开 Google，各自创业（OpenAI、Cohere 等）

## 延伸阅读

- [[transformer]]（本 wiki 的 Transformer 概念页）
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
