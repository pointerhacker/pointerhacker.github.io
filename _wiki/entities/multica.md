---
layout: wiki
title: Multica
wiki_type: entities
category: wiki
tags: [多Agent协作, 任务管理, Agent平台, 开源, Claude Code, Codex, OpenClaw]
description: Multica 开源平台——把编码 Agent 变成真正的队友，支持任务指派/看板/执行追踪/技能沉淀，解决多 Agent 协同中的任务分配、进度追踪、经验复用三大痛点。
related: [multi-agent, claude-code, codex]
updated: 2026-04-10
---

# Multica

> 把编码 Agent 变成真正的队友

来源：[公众号 开源星探 - Multica 实践指南](https://mp.weixin.qq.com/s/b2I88JnT_PMLPdW8czf4pw) | GitHub 4K+ Stars

## 核心定位

Multica 是一个开源的**托管式 Agent 协作平台**，核心理念是"把编码 Agent 变成真正的队友"。

解决的根本问题：多 Agent 协同时，缺乏统一的任务管理和经验沉淀——Agent 各干各的、不透明、无法积累。

## 核心价值

> 缺乏统一的项目管理和经验沉淀，是目前多 Agent 协同最大的死穴。

**当前痛点**：
- 任务指派靠复制粘贴
- 进度黑盒，不知道谁在做什么
- Agent A 踩过的坑，Agent B 重来一遍
- 无任务看板、无追踪、无沉淀

**Multica 的解法**：像管理人类员工一样管理 Agent——有身份、有档案、有看板、有任务追踪。

## 支持的 Agent 后端

**厂商中立**，不绑定任何一家 AI 服务商：

- Claude Code
- Codex
- OpenClaw
- OpenCode

## 核心功能

### 1. Agent 即队友

- 有个人档案、出现在看板上
- 能发表评论、创建 Issue
- 能主动报告阻塞问题
- 像指派同事一样分配任务

### 2. 自主执行

- 设置后无需管理
- 完整的任务生命周期：排队 → 认领 → 执行 → 完成/失败
- 通过 WebSocket 实时推送进度
- 人在看板上实时可见执行状态

### 3. 可复用技能

- 每个解决方案都成为全团队可复用的技能
- 部署、数据库迁移、代码审查等
- 团队能力随时间持续增长，不会因 Agent 切换而丢失

### 4. 统一运行时

- 一个控制台管理所有算力
- 本地 daemon + 云端运行时
- 自动检测 PATH 里可用的 Agent CLI（claude、codex、openclaw、opencode）
- 实时监控

### 5. 多工作区

- 按团队组织工作
- 工作区级别隔离
- 每个工作区有独立的 Agent、Issue 和设置

## 工作原理

```
用户 (Web UI / CLI)
    ↓ 指派任务
Multica 平台
    ↓ WebSocket 推送进度
本地 Daemon (multica daemon start)
    ↓ 自动检测可用 CLI
Agent (Claude Code / Codex / OpenClaw / OpenCode)
    ↓
在用户机器上隔离执行
    ↓
结果汇报回平台
```

**Daemon 的角色**：连接本地机器和 Multica 平台的桥梁，创建隔离环境，运行 Agent，汇报结果。

## 快速上手

### 方式一：Multica Cloud（最快）

访问 [multica.ai](https://multica.ai)，注册即可用，无需任何配置。

### 方式二：自托管（Docker）

```bash
git clone https://github.com/multica-ai/multica.git
cd multica
cp .env.example .env
# 编辑 .env —— 至少修改 JWT_SECRET
docker compose up -d          # 启动 PostgreSQL
cd server && go run ./cmd/migrate up && cd ..  # 运行数据库迁移
make start                   # 启动应用
```

### 安装 CLI 并连接机器

```bash
# 安装
brew tap multica-ai/tap
brew install multica

# 认证并启动
multica login
multica daemon start
```

Daemon 启动后会自动检测 PATH 中的 Agent CLI，并在 Agent 被分配任务时创建隔离环境执行。

## 4 步指派第一个任务

1. **登录并启动 daemon**
   ```bash
   multica login
   multica daemon start
   ```
2. **验证 Runtime**：Web 应用 → Settings → Runtimes，确认机器状态为活跃
3. **创建 Agent**：Settings → Agents → New Agent，选择 Runtime 和提供商，起名字
4. **指派任务**：从看板创建 Issue（或 `multica issue create`），指派给 Agent，Agent 自动接手执行并汇报

## 解决的三大痛点

| 痛点 | 传统方式 | Multica |
|------|----------|---------|
| 任务分配 | 复制粘贴 prompt | 像指派同事一样自然 |
| 执行追踪 | 盯着终端日志 | 全生命周期看板可视 |
| 经验沉淀 | Agent 换人经验全丢 | 每次解决变可复用技能 |

## 参考资料

- [Multica GitHub](https://github.com/multica-ai/multica)
- [Multica Cloud](https://multica.ai)
- [CLI 安装指南](https://github.com/multica-ai/multica/blob/main/CLI_INSTALL.md)
