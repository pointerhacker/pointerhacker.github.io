---
layout: wiki
title: RLHF（基于人类反馈的强化学习）
wiki_type: concepts
category: wiki
tags: [rlhf, alignment, ppo, reward-model, llm]
description: 通过人类偏好反馈训练奖励模型，再用 PPO 优化语言模型的对齐方法
related: [gpt, transformer]
updated: 2026-04-09
---

## 概述

RLHF（Reinforcement Learning from Human Feedback）是将人类偏好融入语言模型训练的核心方法，由 InstructGPT（2022）将其推广到大规模 LLM。

## 三阶段流程

### 阶段 1：监督微调（SFT）
收集人工标注的高质量对话数据，对预训练模型做监督微调，得到 SFT 模型。

### 阶段 2：奖励模型（RM）训练
对同一提示生成多个回答，由人类标注偏好（哪个更好），训练奖励模型 $r_\phi(x, y)$ 拟合人类偏好。

损失函数（Pairwise ranking loss）：

$$\mathcal{L}_{RM} = -\mathbb{E}_{(x, y_w, y_l)}\left[\log \sigma(r_\phi(x, y_w) - r_\phi(x, y_l))\right]$$

其中 $y_w$ 为偏好回答，$y_l$ 为不偏好回答。

### 阶段 3：PPO 强化学习优化

用奖励模型作为奖励信号，通过 PPO 优化 SFT 模型。为防止偏离原始分布过远，加入 KL 惩罚：

$$r(x, y) = r_\phi(x, y) - \beta \cdot \text{KL}[\pi_\theta(\cdot|x) \| \pi_{\text{SFT}}(\cdot|x)]$$

## RLHF 的挑战

- **奖励欺骗（Reward Hacking）**：模型学会在奖励模型上得高分而非真正符合人类意图
- **分布偏移**：PPO 训练会使模型偏离 SFT 初始化
- **人类标注质量**：标注者偏见影响 RM 质量
- **计算开销大**：PPO 需要同时运行 4 个模型

## 替代方法

| 方法 | 思路 | 代表工作 |
|------|------|----------|
| **DPO** | 直接偏好优化，无需 RM | Rafailov et al. 2023 |
| **GRPO** | 组相对策略优化 | DeepSeek-R1 |
| **RLAIF** | 用 AI 反馈替代人类反馈 | Constitutional AI |

## 参考资料

- Ouyang et al., "Training language models to follow instructions with human feedback" (2022)
- [大模型强化学习原理与实现](/blog/大模型强化学习原理与实现.html)（博客文章）
