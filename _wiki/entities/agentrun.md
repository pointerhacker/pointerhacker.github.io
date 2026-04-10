---
layout: wiki
title: AgentRun
wiki_type: entities
category: wiki
tags: [浏览器自动化, Sandbox, Puppeteer, AI Agent, Cookie持久化, VNC]
description: AgentRun 平台：AIO Sandbox 实现浏览器自动化，支持 Cookie 持久化、VNC 人机协作、多步骤任务状态保持。
related: [puppeteer, browser-automation]
updated: 2026-04-10
---

# AgentRun

AgentRun 实践指南：Agent 的宝藏工具——All-In-One Sandbox

来源：[公众号 AgentRun - Cloud Native](https://mp.weixin.qq.com/s/BgrcbJnD_Q27xQT4cLBtfg)

## 核心产品：AIO Sandbox

All-In-One Browser Sandbox，为 AI Agent 提供浏览器自动化能力。核心解决传统浏览器自动化需要登录、验证码、多步骤状态保持的问题。

## 核心设计理念

### 人机协作，而非完全自动化

承认 AI 无法处理验证码、滑块、短信等环节，采用：
- **可观测性优先**：通过 VNC 让执行过程完全透明
- **人机协作**：人工介入搞定验证后，无缝衔接自动化
- **状态持久化**：浏览器会话和 Cookie 可跨步骤保存和恢复

## 核心技术点

### 1. 必须用 connect()，别用 launch()

```javascript
// 错误：启动新浏览器 (1-3秒)
const browser = await puppeteer.launch();

// 正确：连接已运行的浏览器 (<100ms)
const browser = await puppeteer.connect({
  browserWSEndpoint: 'ws://localhost:5000/ws/automation'
});
```

原因：浏览器在容器启动时就已运行，`launch()` 会启动第二个浏览器浪费资源。

### 2. 必须用 disconnect()，别用 close()

| 方法 | 效果 |
|------|------|
| `browser.close()` | 关闭所有页面，终止进程，状态全丢 |
| `browser.disconnect()` | 仅断开 Puppeteer 连接，浏览器继续运行，状态保留 |

### 3. Cookie 持久化是王道

登录状态的本质是 Cookie，没有持久化会导致：Sandbox 重建后状态全丢、Cookie 过期后需重新登录。

```javascript
// 保存 Cookie
const cookies = await page.cookies();
fs.writeFileSync('cookies.json', JSON.stringify(cookies));

// 加载 Cookie
const cookies = JSON.parse(fs.readFileSync('cookies.json'));
await page.setCookie(...cookies);
```

### 4. 多步骤任务用文件系统传递状态

```javascript
// 步骤1：登录 → 保存结果到文件
fs.writeFileSync('/home/user/data/user.json', JSON.stringify(userData));

// 步骤2：读取文件，继续执行
const userData = JSON.parse(fs.readFileSync('/home/user/data/user.json'));
```

## 登录流程拆分

1. 打开登录页 → 人在 VNC 中手动完成
2. 保存 Cookie → 程序自动保存到文件系统
3. 执行任务 → 加载 Cookie，恢复登录状态

## 7 条黄金法则

1. 必须用 `puppeteer.connect()`，禁止 `launch()`
2. 必须用 `browser.disconnect()`，禁止 `close()`
3. 必须保存数据到 `/home/user/data/` 目录
4. 登录流程拆分：打开登录页 → 人工登录 → 保存 Cookie → 执行任务
5. Cookie 先访问域名再设置，避免跨域问题
6. 多步骤任务用文件系统传递状态，别用全局变量
7. 重要操作必须加错误处理，别让错误静默失败

## 系统提示词设计

### 基础模板

```
你是 AgentRun AIO Sandbox 的代码生成助手。
【环境信息】
- 浏览器：Chromium (已预启动)
- 连接端点：ws://localhost:5000/ws/automation
- 文件系统：/home/user/data/ (持久化目录)
- 超时限制：单次执行 300 秒
【代码规范】
1. 连接浏览器：puppeteer.connect({ browserWSEndpoint: '...' })
2. 结束会话：browser.disconnect()
3. 文件读写：使用 /home/user/data/ 目录
4. 错误处理：必须用 try-catch 包裹核心操作
【输出要求】
- 生成完整的 JavaScript 代码
- 包含必要的错误处理
- 关键步骤用 console.log() 记录
- 重要结果保存到文件系统
```

## 进阶技巧

### 性能优化：禁用不必要的资源

```javascript
await page.setRequestInterception(true);
page.on('request', (req) => {
  if (['image', 'stylesheet', 'font', 'media'].includes(req.resourceType())) {
    req.abort();
  } else {
    req.continue();
  }
});
```

### 安全注意事项

- Cookie 文件必须加入 `.gitignore`
- 用户输入需白名单验证，防止代码注入

## 技术收益

1. **启动延迟低**：从多个 sandbox 优化为一个，降低至少 50% 启动时间
2. **状态保持轻量**：用本地文件系统实现状态保持，符合最佳实践
3. **VNC 人工介入**：有效解决验证码等自动化卡点

## 参考资料

- [AgentRun 官方文档](https://docs.agent.run/)
- [Sandbox 教程](https://docs.agent.run/docs/tutorial/core/sandbox)
- [Demo 仓库](https://github.com/devsapp/agentrun-sandbox-demos)
