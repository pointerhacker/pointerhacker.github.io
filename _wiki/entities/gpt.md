---
layout: wiki
title: GPT 系列模型
wiki_type: entities
category: wiki
tags: [gpt, openai, llm, decoder-only, autoregressive]
description: OpenAI 开发的 GPT 系列自回归语言模型，从 GPT-1 到 GPT-4 推动了 LLM 革命
related: [transformer, attention-mechanism, rlhf]
updated: 2026-04-09
---

## 概述

GPT（Generative Pre-trained Transformer）是 OpenAI 开发的自回归语言模型系列，采用 Decoder-Only Transformer 架构，通过海量文本预训练后微调/对齐。

## 版本演进

| 版本 | 发布 | 参数量 | 关键特性 |
|------|------|--------|----------|
| GPT-1 | 2018 | 117M | 预训练+微调范式 |
| GPT-2 | 2019 | 1.5B | Zero-shot 生成 |
| GPT-3 | 2020 | 175B | In-context learning，few-shot |
| InstructGPT | 2022 | 175B | RLHF 对齐 |
| ChatGPT | 2022.11 | 未公开 | 对话优化 |
| GPT-4 | 2023 | 未公开（估计 1T+） | 多模态，顶级推理 |
| GPT-4o | 2024 | 未公开 | 原生多模态（文本/音频/视觉） |
| GPT-4.1 | 2025 | 未公开 | 长上下文（1M token）|

## 架构特点

- **Decoder-Only**：仅使用 Transformer 解码器，因果自注意力
- **BPE Tokenization**：字节对编码分词
- **自回归生成**：逐 token 预测下一个词
- **规模定律（Scaling Law）**：性能随参数量、数据量、计算量可预测提升

## 关键技术

### RLHF（来自 InstructGPT）
通过人类反馈的强化学习对齐模型行为：
1. 监督微调（SFT）
2. 训练奖励模型（RM）
3. PPO 优化策略

详见 [[rlhf]]。

### In-context Learning（GPT-3）
无需梯度更新，仅通过 prompt 中的示例即可适应新任务——浮现出的 few-shot 能力。

## 对行业的影响

- 确立了"预训练+对齐"的现代 LLM 范式
- 推动了 LLaMA、Qwen、DeepSeek 等开源复现工作
- GPT-4 的多模态能力开启了新阶段

## 参考资料

- Radford et al., "Improving Language Understanding by Generative Pre-Training" (2018)
- Brown et al., "Language Models are Few-Shot Learners" (NeurIPS 2020)
- Ouyang et al., "Training language models to follow instructions with human feedback" (2022)
