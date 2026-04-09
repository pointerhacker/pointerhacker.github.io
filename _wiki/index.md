---
layout: wiki
title: 知识库概览
wiki_type: concepts
category: wiki
nav_exclude: true
tags: [llm, overview, index]
description: LLM 知识库的概览和使用指南
related: []
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

- **concepts/** — 概念页（注意力机制、RoPE、KV Cache 等）
- **entities/** — 实体页（模型、论文、人物、机构等）
- **sources/** — 原始资料摘要

## 如何使用

1. 浏览左侧导航树，找到感兴趣的条目
2. 使用顶部搜索框全文搜索（覆盖博客和知识库）
3. 点击条目底部的"相关条目"和"反向链接"探索关联

## 如何贡献新条目

在 `_wiki/concepts/`、`_wiki/entities/` 或 `_wiki/sources/` 下新建 Markdown 文件，使用以下 front matter：

```yaml
---
layout: wiki
title: 条目标题
wiki_type: concepts  # concepts | entities | sources
category: wiki
tags: [tag1, tag2]
description: 一行摘要（显示在首页和导航 tooltip）
related: [other-slug]  # 相关条目的文件名（不含扩展名）
updated: 2026-04-09
---
```
