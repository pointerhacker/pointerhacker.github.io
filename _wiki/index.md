---
layout: wiki
title: 知识库概览
wiki_type: concepts
category: wiki
nav_exclude: true
tags: [llm, overview, index]
description: LLM 知识库的概览和使用指南
related: [openviking]
updated: 2026-04-09
---

## 什么是这个知识库

这是一个基于 [Karpathy LLM Wiki](https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f) 模式构建的个人 LLM 知识库。

核心思想：不同于 RAG（每次查询时重新从原始文档检索），这里的 LLM **增量构建并维护一个持久化的 Wiki**——一个结构化的、相互关联的 Markdown 文件集合。

## 三层架构

| 层次 | 说明 | 操作者 |
|------|------|--------|
| 原始资料 (`raw/`) | 不可变的原始资料 | 人工收集 |
| Wiki (`_wiki/`) | 结构化知识页面 | LLM 写作 |
| Schema (`CLAUDE.md`) | 维护规则和工作流 | 人机共同演化 |

## 分类体系

- **concepts/** — 概念页（仅限跨实体可复用的独立概念）
- **entities/** — 实体页（模型、论文、人物、机构等）
- **sources/** — 原始资料摘要

> **聚合原则：** 如果一个概念仅在某一个实体的上下文中才有意义，就作为该实体页的一个章节展开，**不单独建 concept 条目**。

## 索引

### 实体（Entities）

- [[entities/voxcpm-2|VoxCPM 2]] — 面壁智能开源多语种高保真 TTS 模型，2B 参数，支持 30 种语言和 9 种中文方言
- [[entities/openviking|OpenViking]] — 字节火山引擎开源的 AI Agent 上下文管理框架（文件系统范式 + 分层上下文加载 + 目录递归检索）
- [[entities/claude-managed-agents|Claude Managed Agents]] — Anthropic 2026-04-08 发布的 Agent 托管平台
- [[entities/gpt|GPT 系列]] — OpenAI GPT 模型系列

### 概念（Concepts）

- [[concepts/attention-mechanism|注意力机制]] — Transformer 的核心机制
- [[concepts/flash-attention|Flash Attention]] — 高效注意力计算实现
- [[concepts/kv-cache|KV Cache]] — LLM 推理优化的关键技术
- [[concepts/rope|RoPE]] — 旋转位置编码
- [[concepts/transformer|Transformer]] — 注意力机制架构
- [[concepts/rlhf|RLHF]] — 基于人类反馈的强化学习

- [[entities/agentrun|AgentRun]] — AIO Sandbox 浏览器自动化平台，支持 Cookie 持久化、VNC 人机协作
- [[entities/multica|Multica]] — 多人 Agent 协作平台，4K Stars，Agent 即队友，看板管理+技能沉淀
- [[entities/li-2026-mempo|MemPO]] — 让 LLM Agent 自主管理记忆的 RL 算法，通过记忆级 dense reward 解决长程任务信用分配问题

## 如何贡献新条目

在 `_wiki/concepts/`、`_wiki/entities/` 或 `_wiki/sources/` 下新建 Markdown 文件，使用以下 front matter：

```yaml
---
layout: wiki
title: 条目标题
wiki_type: concepts  # concepts | entities | sources
category: wiki
tags: [tag1, tag2]
description: 一行摘要
related: [other-slug]
updated: 2026-04-09
---
```
