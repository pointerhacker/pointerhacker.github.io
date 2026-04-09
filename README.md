# pointerhacker.github.io

个人博客 + LLM 知识库，基于 Jekyll + GitHub Pages 构建。  
博客由 Obsidian 写作推送；知识库由 Claude Code 协助维护，参考 [Karpathy LLM Wiki](https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f) 模式。

**线上地址**：[pointerhacker.github.io](https://pointerhacker.github.io)

---

## 这是什么

```
博客 (_posts/)              知识库 (_wiki/)
  │                            │
  │  原创深度文章               │  结构化知识图谱
  │  ・长篇推导                 │  ・概念条目
  │  ・代码实现                 │  ・实体页面
  │  ・实验记录                 │  ・论文摘要
  │                            │
  └──────── 统一全文搜索 ────────┘
```

- **博客** (`/blog/`)：自己写的深度文章，Obsidian 写完自动推送
- **知识库** (`/wiki/`)：LLM 知识图谱，Claude Code 协助维护
- **搜索** (`/search.html`)：全文搜索，同时覆盖博客和知识库

---

## 目录结构

```
.
├── _posts/              ← 博客文章（Obsidian 写，obsidian-git 自动推送）
├── _wiki/               ← 知识库条目
│   ├── concepts/        ← 概念（注意力机制、RoPE、KV Cache…）
│   ├── entities/        ← 实体（模型、论文、人物…）
│   ├── sources/         ← 原始资料摘要
│   ├── index.md         ← 知识库概览页
│   └── log.md           ← 操作日志（append-only）
├── _layouts/
│   ├── default.html     ← 博客默认布局
│   ├── post.html        ← 博客文章布局
│   ├── wiki.html        ← Wiki 双栏布局
│   └── wiki-index.html  ← 知识库首页布局
├── css/
│   ├── site.css         ← 全局样式
│   ├── blog.css         ← 博客样式
│   └── wiki.css         ← 知识库样式
├── wiki/index.html      ← 知识库入口页
├── search.html          ← 全文搜索页（覆盖 blog + wiki）
├── CLAUDE.md            ← Claude Code 操作规范（Wiki Schema）
└── skill.md             ← Claude Code 提示词手册（即本文档的姐妹文件）
```

---

## 写博客

用 Obsidian 打开仓库根目录，在 `_posts/` 下新建文件：

```
_posts/YYYY-MM-DD-标题.md
```

Front matter 模板：

```yaml
---
layout: post
title: "文章标题"
categories: LLM
tags: [tag1, tag2]
description: 一句话摘要
---
```

obsidian-git 插件自动 commit + push，GitHub Pages 约 1-3 分钟构建完成。

---

## 维护知识库

### 让 Claude Code 处理（推荐）

打开 Claude Code，用 `skill.md` 里的提示词触发操作：

```bash
# 新增一篇论文摘要
把这篇论文整理成知识库条目：[粘贴摘要]

# 从博客提炼条目
把博客《一文搞懂KVCache》提炼成 wiki 条目

# 定期健康检查
对知识库做一次 lint

# 回答问题（会先查 wiki 再答）
根据知识库，解释 GQA 和 MQA 的区别
```

详细提示词见 [`skill.md`](./skill.md)，操作规范见 [`CLAUDE.md`](./CLAUDE.md)。

### 手动新建条目

在 `_wiki/concepts/`、`_wiki/entities/` 或 `_wiki/sources/` 下新建 `.md` 文件：

```yaml
---
layout: wiki
title: 条目标题
wiki_type: concepts        # concepts | entities | sources
category: wiki
tags: [tag1, tag2]
description: 一行摘要（<80字符，显示在首页卡片）
related: [other-slug]      # 相关页面文件名（不含 .md）
updated: 2026-04-09
---
```

文件名用 kebab-case，例如 `my-concept.md`。

---

## Wiki 条目 URL 规则

| 文件路径 | 访问 URL |
|----------|----------|
| `_wiki/concepts/attention-mechanism.md` | `/wiki/concepts/attention-mechanism/` |
| `_wiki/entities/gpt.md` | `/wiki/entities/gpt/` |
| `_wiki/sources/attention-is-all-you-need.md` | `/wiki/sources/attention-is-all-you-need/` |

---

## 本地预览

```bash
bundle exec jekyll serve
# 打开 http://localhost:4000
```

无 Ruby 环境可直接 push 后在 GitHub Pages 查看。

---

## 致谢

- 博客主题基于 [Sébastien Saunier](https://github.com/ssaunier) 和 [marcgg](http://marcgg.com/) 的模板
- 知识库模式参考 [Karpathy LLM Wiki](https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f)
