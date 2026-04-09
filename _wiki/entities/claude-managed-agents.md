---
layout: wiki
title: Claude Managed Agents
wiki_type: entities
category: wiki
tags: [anthropic, agent, infrastructure, api, harness]
description: Anthropic 于 2026-04-08 发布的云托管 AI Agent 基础设施平台，提供 Harness 编排引擎、沙箱执行环境和长会话管理
related: []
updated: 2026-04-09
---

## 概述

Claude Managed Agents 是 Anthropic 于 2026 年 4 月 8 日发布的一套可组合 API，用于构建和部署云托管的 AI Agent。核心卖点是 **Harness（Agent 编排引擎）**，而非单纯的模型 API。

这次发布标志着 Anthropic 从「模型提供商」向「Agent 基础设施提供商」的定位转移。

## 核心能力

| 功能 | 说明 |
|------|------|
| 生产级沙箱 | 每个 Agent 跑在隔离云容器中，可预装 Python/Node.js/Go 等环境 |
| 长时间 Session | Agent 可自主运行数小时，断连自动恢复，进度持久化 |
| 内置 Harness | 经调优的编排循环，自动处理工具调用决策、上下文管理、错误恢复 |
| 多 Agent 协调 | （研究预览）Agent 可启动子 Agent 并行处理子任务 |
| 自评估能力 | （研究预览）Agent 自迭代直到满足用户定义的成功标准，结构化任务成功率提升最多 10 个百分点 |
| 治理工具 | Scoped permissions、身份管理、Session tracing、执行追踪 |

Agent 定义支持自然语言描述或 YAML 文件，兼容 MCP 服务器和 Agent Skills。

## Harness 架构：把大脑和手分开

参见工程博客「Scaling Managed Agents: Decoupling the brain from the hands」。

### 三层虚拟化

Managed Agents 将 Agent 虚拟化为三个解耦接口：

- **Session（会话）**：所有事件的追加写入日志，持久存储在 Harness 之外
- **Harness（编排循环）**：调用 Claude、将工具调用路由到对应基础设施的循环
- **Sandbox（执行环境）**：Agent 执行代码和文件操作的隔离容器

三者完全解耦，任一故障不影响其他两层。

### 从宠物到牲畜

早期设计把所有组件放在同一容器里，容器变成了「宠物」（坏了得救）。解耦后容器变成「牲畜」（坏了换一头）：

- Harness 通过 `execute(name, input) → string` 调用沙箱，与调用其他工具无异
- Session 日志独立存储于 Harness 之外，Harness 崩溃不丢数据
- 新 Harness 通过 `wake(sessionId)` 从断点继续
- 安全边界变干净：凭证永远不进沙箱，OAuth 令牌通过代理调用

### Session ≠ 上下文窗口

Session 是独立于上下文窗口的持久化对象，Harness 通过 `getEvents()` 按需取事件流切片，避免压缩/裁剪导致的信息不可逆丢失。

### 性能提升

解耦后推理可在容器启动前开始：**p50 TTFT 下降约 60%，p95 下降超过 90%**。

### Meta-harness 设计哲学

Anthropic 虚拟化了 Harness 的接口而非固化某一具体实现，与操作系统设计哲学一致：**抽象比实现活得久**，为「尚未想到的程序」设计系统。

## 定价

- 标准 Claude Platform token 费率
- Session 活跃运行时间：**$0.08/小时**（按毫秒计量，空闲等待不计费）
- Web 搜索：**$10/千次**

`$0.08/session-hour` 代表 Anthropic 在 token 收入之外新增了更稳定的运行时收入流。

## 早期用户

- **Notion**：团队与 Agent 协作平台
- **Rakuten**：跨部门专项 Agent，一周内部署
- **Asana**：AI Teammates，参与项目管理流程
- **Sentry**：调试 Agent 自动写补丁、开 PR，集成周期从数月缩至数周
- **Atlassian**：将 Agent 构建进 Jira 工作流

## 参考资料

- [Claude Managed Agents 产品公告](https://claude.com/blog/claude-managed-agents)
- [工程博客：Scaling Managed Agents](https://www.anthropic.com/engineering/managed-agents)
- [API 文档](https://platform.claude.com/docs/en/managed-agents/overview)
- 来源：微信公众号「金色传说大聪明」，2026-04-08
