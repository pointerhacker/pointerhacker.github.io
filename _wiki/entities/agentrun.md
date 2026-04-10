---
layout: wiki
title: AgentRun
wiki_type: entities
category: wiki
tags: [Agent平台, Sandbox, Serverless, 模型治理, MCP, 企业级AI]
description: AgentRun 是企业级一站式 Agentic AI 基础设施平台，提供 Agent 运行时管理、多类型沙箱、模型治理、工具管理、凭证安全等能力。
related: [puppeteer, browser-automation]
updated: 2026-04-10
---

# AgentRun

> 企业级一站式 Agentic AI 基础设施平台

来源：[AgentRun 官方文档](https://docs.agent.run/)

## 产品定位

AgentRun 是以**高代码为核心，开放生态、灵活组装**的一站式 Agentic AI 基础设施平台，为企业级 Agentic 应用提供**开发、部署与运维全生命周期管理**。

核心价值：
- **Serverless 架构**：强隔离运行时 + 弹性伸缩
- **开放生态**：深度集成 LangChain、AgentScope、LangGraph、CrewAI 等主流框架
- **数据不出域**：私有网络部署，数据物理隔离

## 核心模块

### 1. AgentRuntime — 智能体运行时

Agent 运行时的生命周期管理单元。

**部署方式**：上传代码包（本地/OSS）、在线编码、自定义容器镜像

**运行时特性**：
- 多语言：Python 3.10/3.12、Node.js 18/20、Java 8/11/17 等
- 多开发模式：无代码（AI Studio）、低代码（快速创建）、高代码（代码创建）
- 会话亲和 + Serverless 弹性 + 多实例并发
- 版本管理 + Endpoint 灰度发布

**集成方式**：SDK、API（OpenAI Chat Completions 兼容）、UI、MCP

### 2. Sandbox — 沙箱环境

为代码执行和浏览器操作提供安全、高性能的 Serverless 沙箱。

**沙箱类型**：
- **Code Interpreter**：Python/Shell 代码执行，完整文件系统操作
- **Browser Use**：浏览器自动化
- **All-in-One**：全功能沙箱（浏览器 + 代码执行，适合复杂多步骤任务）
- **……**（更多类型持续更新）

**隔离与弹性**：
- MicroVM 安全容器，多级隔离
- 支持缩容到 0，按请求弹性调度
- 毫秒级唤醒，万级实例/分钟极速交付

**集成方式**：SDK 调用、MCP 工具方式

### 3. Model — 模型管理与治理

统一的大模型接入、管理与治理中心。

**模型来源**：
- 第三方模型：通义千问、DeepSeek 等
- 开源托管模型：vLLM / SGLang / Ollama / LMDeploy 等框架
- 向量模型

**模型治理能力**：
- 多模型负载代理 + Fallback + 并发控制
- 超时与缓存
- 内容安全 + Token 限流 + 成本监控
- Serverless GPU 弹性交付

### 4. ToolSet — 工具管理

统一的工具定义、调用和治理中心。

**协议支持**：MCP + Function Call 双协议

**Tool Hub 生态**：
- 提供大量常用工具，一键接入
- 支持自定义工具发布与分享

**智能扩展**（规划中）：Hook 注入、语义分析、智能路由、AI 自动生成工具定义

### 5. Credential — 凭证管理

安全集中管理 Agent / Sandbox / LLM / 工具访问所需的凭证。

**支持类型**：API Key、JWT、Basic Auth、AK/SK 等

**安全特性**：
- 动态凭证注入：运行时自动注入，无需硬编码
- 启用/禁用控制：一键禁用疑似泄露的凭证

## SDK 架构

```
┌─────────────────────────────────────┐
│         集成模块 + 工具类             │  ← 上层：框架适配器、配置、日志
├─────────────────────────────────────┤
│         资源对象层                   │  ← AgentRuntime、Model、Sandbox 等
├─────────────────────────────────────┤
│   Control API   │   Data API        │  ← 中层：资源管理 / 运行交互
└─────────────────────────────────────┘
```

## 适用场景

| 场景 | 说明 |
|------|------|
| 智能客服 | 企业级对话 Agent，多模型 Fallback |
| 数据分析助手 | Code Interpreter 沙箱执行 Python 分析 |
| 代码生成工具 | 代码解释器 + 浏览器自动化组合 |
| 网页自动化机器人 | Browser Use / AIO Sandbox |
| 合规敏感场景 | 数据不出域 + 私有网络部署 |

## 技术指标

- **弹性**：缩容到 0，毫秒级唤醒，万级实例/分钟交付
- **隔离**：MicroVM 安全容器，多级隔离
- **集成**：LangChain / AgentScope / LangGraph / CrewAI 原生支持
- **协议**：OpenAI Chat Completions 兼容、MCP 协议

## 参考资料

- [AgentRun 官方文档](https://docs.agent.run/)
- [Sandbox 教程](https://docs.agent.run/docs/tutorial/core/sandbox)
- [Demo 仓库](https://github.com/devsapp/agentrun-sandbox-demos)
