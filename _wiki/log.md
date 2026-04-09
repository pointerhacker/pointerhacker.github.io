---
layout: wiki
title: 操作日志
wiki_type: concepts
category: wiki
nav_exclude: true
tags: [log, meta]
description: 知识库的操作记录日志
related: []
updated: 2026-04-09
---

## 日志格式

每条记录格式：`## [日期] 操作类型 | 简要描述`

---

## [2026-04-09] 初始化 | 知识库首次创建

- 创建知识库框架（Jekyll collection + wiki 布局）
- 新增初始条目：注意力机制、Transformer、KV Cache、RoPE、FlashAttention、RLHF
- 新增实体页：GPT 系列

---

## [2026-04-09] ingest | Claude Managed Agents

- 新增实体页：`_wiki/entities/claude-managed-agents.md`
- 来源：微信公众号「金色传说大聪明」，解析 Anthropic 2026-04-08 发布的 Managed Agents 平台
- 覆盖内容：三层虚拟化架构（Session/Harness/Sandbox）、从宠物到牲畜的设计演进、定价模型、早期用户案例

---

## 待补充条目

以下是规划中但尚未创建的条目（stub list）：

### Concepts
- [ ] MoE（混合专家模型）
- [ ] 激活函数（SwiGLU、GELU、ReLU）
- [ ] Layer Normalization vs RMSNorm
- [ ] Tokenization（BPE、SentencePiece）
- [ ] 位置编码总览（绝对 vs 相对 vs RoPE vs ALiBi）
- [ ] 量化（INT8/INT4/GPTQ/AWQ）
- [ ] LoRA / PEFT
- [ ] Speculative Decoding
- [ ] 推理优化总览
- [ ] 预训练数据处理

### Entities
- [ ] LLaMA 系列
- [ ] Qwen 系列
- [ ] DeepSeek 系列
- [ ] Mistral / Mixtral
- [ ] Claude 系列
- [ ] Andrej Karpathy（重要研究者）

### Sources
- [ ] "Attention Is All You Need" 摘要
- [ ] "Scaling Laws for Neural Language Models" 摘要
- [ ] nanoGPT 代码分析
