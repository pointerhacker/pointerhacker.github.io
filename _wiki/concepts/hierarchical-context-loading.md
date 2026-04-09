---
layout: wiki
title: 分层上下文加载
wiki_type: concepts
category: wiki
tags: [上下文管理, LLM, Agent]
description: LLM 上下文分层管理范式，将信息分为 L0 摘要、L1 概览、L2 详情，按需加载。
related: [openviking, recursive-directory-retrieval]
updated: 2026-04-09
---

# 分层上下文加载

一种 LLM 上下文管理范式，将海量上下文信息按详略程度分为多个层级，按需加载。

## 三层结构

| 层级 | 内容 | 使用场景 |
|------|------|---------|
| **L0 摘要层** | 高度浓缩的摘要 | 快速检索、相关性判断 |
| **L1 概览层** | 核心信息和功能描述 | Agent 规划阶段决策 |
| **L2 详情层** | 原始完整数据 | 需要深度阅读时 |

## 优势

- 节省 Token 成本
- 减少无效信息干扰
- 让 Agent 只调用当前需要的信息

## 参考资料

- [[entities/openviking|OpenViking]]
