# CLAUDE.md — LLM Wiki Schema

这是 pointerhacker.github.io 的 LLM 知识库维护指南。本文件告诉 Claude Code 如何维护这个知识库。

## 知识库结构

```
_wiki/
├── index.md          # 知识库概览
├── log.md            # 操作日志（append-only）
├── concepts/         # 概念页（算法、机制、方法等）
├── entities/         # 实体页（模型、论文、人物、机构等）
└── sources/          # 原始资料摘要
```

## 页面 Front Matter 规范

```yaml
---
layout: wiki
title: 页面标题
wiki_type: concepts  # concepts | entities | sources
category: wiki
tags: [tag1, tag2, tag3]
description: 一行摘要（显示在首页卡片和导航 tooltip，保持 <80 字符）
related: [slug1, slug2]   # 相关页面的 slug（文件名不含扩展名和路径）
updated: 2026-04-09       # 最后更新日期（YYYY-MM-DD）
---
```

**Slug 规则**：`related` 字段填写文件名（不含 `.md` 扩展名），如 `attention-mechanism`、`gpt`。

## 操作工作流

### Ingest（新增条目）

当添加新资料时：

1. **读取资料**，与用户讨论关键要点
2. **选择分类**：是概念（concepts）、实体（entities）还是资料摘要（sources）？
3. **创建页面**：在对应目录下新建 `.md` 文件，文件名使用 kebab-case（小写+连字符）
4. **更新相关页**：检查现有 wiki 页面，在其 `related` 字段中添加新页面的 slug（如有关联）
5. **更新 log.md**：在 `## [日期] ingest | 标题` 格式下追加一条记录

### Query（回答问题）

回答 wiki 相关问题时：
1. 优先阅读相关 wiki 页面（`_wiki/` 目录）
2. 结合 `_posts/` 中的博客文章
3. 如果答案值得保存，将其作为新条目或更新现有条目

### Lint（健康检查）

定期执行以下检查：
- 是否有孤立页面（`related` 引用了不存在的 slug）？
- 是否有重要概念在多页提及但无独立条目（stub 候选）？
- `log.md` 的"待补充条目"是否有可以填充的？
- 各页的 `description` 是否简洁准确？

## 写作规范

- **中文为主**，专有名词（论文标题、模型名、作者名）保留英文
- **公式**：使用 LaTeX 行内公式 `$...$` 和块级公式 `$$...$$`
- **内部链接**：引用其他 wiki 页面时使用 `[[slug]]` 风格注释（文档中），实际链接用 `/wiki/concepts/slug/`
- **外部链接**：链接到博客文章或外部资源时注明来源
- **表格**：复杂对比信息优先使用 Markdown 表格
- **每页长度**：核心概念 300-800 字，资料摘要 100-400 字

## 文件命名

| 分类 | 示例 |
|------|------|
| 概念 | `attention-mechanism.md`, `kv-cache.md`, `flash-attention.md` |
| 实体（模型） | `gpt.md`, `llama.md`, `qwen.md` |
| 实体（论文） | `attention-is-all-you-need.md`, `scaling-laws.md` |
| 实体（人物） | `andrej-karpathy.md` |

## 与博客的关系

`_posts/` 博客文章是原创深度内容（不删除），wiki 页面是提炼后的知识图谱。两者关系：

- 博客文章 → 资料来源，wiki 摘要引用博客
- wiki 条目 → 博客索引，为博客提供概念背景
- 搜索同时覆盖两者

## 注意事项

- **不要修改 `_posts/` 下的博客文章**
- **不要修改 `css/site.css` 和 `css/blog.css`**（除非专门优化博客样式）
- `_wiki/index.md` 和 `_wiki/log.md` 是特殊文件，`wiki_type` 设为 `concepts`，但不出现在 concepts 分组（通过 slug 区分）
- Wiki 页面的 URL 格式为 `/wiki/concepts/slug/`、`/wiki/entities/slug/`、`/wiki/sources/slug/`
