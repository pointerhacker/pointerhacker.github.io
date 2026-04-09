---
layout: wiki
title: OpenViking
wiki_type: entities
category: wiki
tags: [Agent, 上下文管理, 开源, 字节跳动]
description: 字节火山引擎开源的 AI Agent 上下文管理框架，以文件系统范式统一管理记忆、资源、技能。
related: [hierarchical-context-loading, recursive-directory-retrieval]
updated: 2026-04-09
---

# OpenViking

字节火山引擎推出的开源 AI Agent 框架（[GitHub 21.7k Stars](https://github.com/volcengine/OpenViking)），核心解决 Agent 上下文管理的痛点。

## 核心设计理念

将 Agent 所需的所有上下文（记忆、资源、技能）映射为**虚拟文件系统**，通过 `viking://` 协议赋予每个信息唯一 URI，让开发者像用 `ls`、`find` 命令一样管理智能体大脑。

## 五大核心能力

### 1. 文件系统管理范式

将 Agent 所有上下文整合为分层虚拟文件系统：

- `resources/`：外部资源（文档、代码仓库、网页）
- `user/`：用户偏好、习惯
- `agent/`：技能、指令、任务记忆

### 2. 分层上下文加载（L0/L1/L2）

| 层级 | 名称 | 内容 | 用途 |
|------|------|------|------|
| **L0** | 摘要层 | 高度浓缩摘要 | 快速检索和相关性判断 |
| **L1** | 概览层 | 核心信息和功能描述 | Agent 规划阶段决策 |
| **L2** | 详情层 | 原始完整数据 | 需要深度阅读时 |

### 3. 目录递归检索

5步策略：意图分析 → 初始定位 → 精细探索 → 递归深入 → 结果聚合。

### 4. 可视化检索轨迹

完整记录每次检索的目录浏览和文件定位轨迹，便于调试。

### 5. 自动会话管理

会话结束后自动更新用户记忆和智能体经验，实现自我进化。

## 参考资料

- GitHub: https://github.com/volcengine/OpenViking
- 公众号文章: https://mp.weixin.qq.com/s/02wS4VyySI6Zumjx0kCGdQ
