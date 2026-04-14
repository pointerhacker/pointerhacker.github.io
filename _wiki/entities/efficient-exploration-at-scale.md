---
layout: wiki
title: Efficient Exploration at Scale
wiki_type: entities
category: wiki
tags: [RLHF, reinforcement-learning, efficient-exploration, google-deepmind, LLM-alignment, Gemma]
description: Google DeepMind 提出在线学习算法，将 RLHF 数据效率提升 10 倍以上，用 20K 标签达到离线 RLHF 用 200K 标签的效果。
related: []
updated: 2026-04-13
---

# Efficient Exploration at Scale

**团队：** Google DeepMind — Efficient Agent Team

**作者：** Mohammad Asghari, Chris Chute, Vikranth Dwaracherla, Xiuyuan Lu, Mehdi Jafarnia, Victor Minden, Zheng Wen, Benjamin Van Roy

**来源：** arXiv:2603.17378（cs.LG）
**提交：** 2026-03-18
**链接：** https://arxiv.org/abs/2603.17378

---

## 核心贡献

提出一种**在线学习算法**，大幅提升 RLHF（从人类反馈中进行强化学习）的数据效率。

核心思路：不像传统离线 RLHF 那样将偏好数据当作静态数据集批量训练，而是**随着 choice 数据到来逐步更新 reward model 和 language model**。

- **Reward Model (RM)**：拟合 choice 数据
- **Language Model (LM)**：通过一种 REINFORCE 变体更新，由 reward model 提供强化信号

### 三个关键使能技术

1. **Affirmative Nudge**：在每个强化信号上添加一个小的"肯定推力"，稳定在线学习
2. **Epistemic Neural Network（认知神经网络）**：建模 reward 的不确定性
3. **Information-Directed Exploration（信息导向探索）**：引导探索策略

## 效果

在 **Gemma LLM** 上的实验结果：

| 配置 | 效果 |
|------|------|
| 在线算法用 **<20K** 标签 | 匹配离线 RLHF 用 **200K** 标签的效果 |
| 数据效率提升 | **>10 倍** |
| 推算：用 **1M** 标签 | 可匹配离线 RLHF 用 **1B** 标签的效果 |
| 推算提升 | **1000 倍** |

这是首次有结果表明如此大规模的效率提升是可能的。

## 关键词

- RLHF（Reinforcement Learning from Human Feedback）
- Online Learning / Incremental Learning
- Information-Directed Exploration
- Epistemic Neural Networks
- LLM Alignment
- Data Efficiency

## 参考资料

- 原始论文：https://arxiv.org/abs/2603.17378
