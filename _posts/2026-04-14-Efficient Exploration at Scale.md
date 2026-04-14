---
layout: post
title: 论文精读：Efficient Exploration at Scale
categories:
  - 论文
tags:
  - RL
  - 数据增强
---

**Google DeepMind 如何用探索策略实现 RLHF 1000x 数据效率提升**

> **论文**：Efficient Exploration at Scale
> **机构**：Google DeepMind, The Efficient Agent Team
> **时间**：2026 年 3 月
> **链接**：https://arxiv.org/abs/2603.17378

---

## 一、论文解决什么问题？

当前 RLHF（Reinforcement Learning from Human Feedback）是大语言模型对齐人类偏好的主流方法，但存在一个根本性问题：**数据效率极低**。Offline RLHF 需要海量的人类标注才能获得显著性能提升，而每一条人类标注都意味着高昂的时间和金钱成本。

本文提出的核心问题是：

> 能不能让每一次人类标注都发挥最大价值，从而用远少于传统方法的数据量达到同样甚至更好的效果？

答案是：**可以，而且提升幅度惊人。**

---

## 二、核心结果

论文的实验结果具有里程碑意义：

- 在仅使用 **< 20K** 标签时，信息导向探索算法**匹配了 Offline RLHF 使用 200K 标签**的性能 → **超过 10x 数据效率提升**
- 外推预测：使用 **1M** 标签可匹配 Offline RLHF **1B** 标签 → **约 1,000x 提升**

这意味着在 scaling law 的视角下，高效探索策略能够从根本上**改变 RLHF 的数据-性能缩放曲线**。

---

## 三、实验设置

| 要素 | 详情 |
|---|---|
| 基础模型 | Gemma 9B（预训练 + SFT，未经 RLHF） |
| 人类反馈模拟器 | 基于 Gemini 1.5 Pro 的奖励模型（远大于 9B，确保模拟行为足够复杂） |
| Prompt 集 | 202K 条多样化 prompt（写作、代码、数学、摘要等） |
| 评估指标 | 与基线策略（SFT top-1 policy）的 win rate |
| 采样策略 | Top-5 sampling 生成候选 response，Top-1 用于评估 |
| Batch 大小 | 64 prompts / batch |

---

## 四、四种对比算法

### 4.1 Offline RLHF

最传统的方式：先用固定策略 π_θ₀ 生成所有 response，收集全部人类反馈，然后一次性训练 RM，最后优化策略。

**问题**：response 来自固定分布，无法根据学习进度调整数据收集策略，存在数据覆盖不足的问题。

### 4.2 Periodic RLHF

每 400 个 batch 为一个周期，周期结束后用新策略重新采样 response、重训 RM 和策略。相比 Offline 有改善，但每次重训计算开销巨大。

### 4.3 Online RLHF

每个 batch 后**增量更新** RM 和策略，不再从头重训。这是关键的效率跃升。但面临一个严重问题——**tanking（性能崩塌）**。

### 4.4 Information-Directed Exploration

在 Online RLHF 基础上，增加 ENN 不确定性建模和信息导向的 response pair 选择。这是论文最核心的贡献，表现最佳。

---

## 五、三大关键创新深度解析

### 5.1 Affirmative Nudge（肯定性微调）—— 解决 Tanking 问题

#### 问题背景

Online RLHF 在训练过程中，策略性能会先上升然后突然崩塌（tanking）。之前的应对方式要么保存 checkpoint 回退，要么降低学习率——都会牺牲最终性能。

#### 策略更新公式（公式 4）

论文给出的策略更新规则包含两大部分：

**第一项——策略梯度（REINFORCE）：**

$$\left( p_{\hat{\phi}_t}(Y \succeq Y' | X) - \frac{1}{2} \right) \nabla_{\theta_t} \ln \pi_{\theta_t}(Y|X)$$

- $p_{\hat{\phi}_t}(Y \succeq Y' | X)$：奖励模型预测 response Y 优于 Y' 的概率，取值 ∈ [0,1]
- $-\frac{1}{2}$：居中基线，使强化信号以 0 为中心
- $\nabla_{\theta_t} \ln \pi_{\theta_t}(Y|X)$：整个 response 在当前策略下的对数概率梯度

RM 说好 → 正信号 → 强化生成概率；RM 说差 → 负信号 → 抑制生成概率。

**第二项——KL 正则化：**

$$\beta \sum_{\ell=1}^{\text{len}(Y)} \pi_{\bar{\theta}_t}(Y_\ell | X, Y_{1:\ell-1}) \nabla_{\theta_t} \ln \frac{\pi_{\bar{\theta}_t}(Y_\ell | X, Y_{1:\ell-1})}{\pi_{\theta_t}(Y_\ell | X, Y_{1:\ell-1})}$$

- $\bar{\theta}_t$：锚点参数（历史参数的指数移动平均）
- 本质是 KL(π_θ̄ₜ ‖ π_θₜ) 对 θₜ 的梯度
- 作用：防止当前策略偏离历史稳定策略太远

两项合在一起构成经典的「探索-约束」平衡：第一项驱动学习，第二项保持稳定。

#### Affirmative Nudge 的改进

加入 nudge 后的公式（公式 5），唯一的变化是在强化信号中加入小正数 ε：

$$p_{\hat{\phi}_t}(Y \succeq Y' | X) - \frac{1}{2} + \epsilon$$

效果：

- **负信号被系统性削弱**：惩罚比原来小了 ε，防止模型过度远离当前策略分布
- **隐式锚定当前策略**：即使 RM 给出中性判断（概率=0.5），模型仍微弱强化当前 response
- **不牺牲学习速度**：ε 很小，正向信号几乎不受影响

**Affirmative（肯定）+ Nudge（轻推）**——对模型生成的每个 response 都给予一点肯定性轻推，防止过度自我否定导致崩塌。

---

### 5.2 认知神经网络（Epistemic Neural Network, ENN）—— 量化不确定性

#### 核心思想

普通奖励模型只输出点估计 r_φ(Y|X)，无法回答「我对这个估计有多确信」。ENN 通过引入**认知索引（epistemic index）Z**，让同一模型给出多种可能的奖励估计，从而量化不确定性。

#### 架构

在 Gemma 9B 的 transformer backbone（去掉 unembedding + softmax）的 last-layer embedding 之上，挂载三类 head：

| 组件 | 结构 | 数量 | 是否训练 |
|---|---|---|---|
| 点估计 head（mlp₀） | 2 隐层 × 1024 宽，线性输出 | 1 | 是 |
| Prior networks（mlp₁~mlp₁₀₀） | 2 隐层 × 256 宽 | 100 | **否**（随机初始化后冻结） |
| Differential networks（mlp'₁~mlp'₁₀₀） | 2 隐层 × 1024 宽 | 100 | 是 |

新增参数量不到总 9B 参数的 5%。

#### 推理路径

**Z = 0（点估计）：** 只用点估计 head → r_φ(Y|X, 0) = mlp₀(backbone(X,Y))，等同普通 RM。

**Z = k（k=1,...,100，集成粒子）：** 点估计 + 第 k 个 prior + 第 k 个 differential 的输出相加。

核心设计思想（Randomized Prior Functions）：

- **Prior network（冻结）** 提供随机先验 → 100 个粒子天然有多样性
- **Differential network（可训练）** 学习数据驱动的修正 → 数据越多，粒子趋同 → 不确定性降低

#### 训练流程

| 步骤 | 更新对象 | Backbone | 其他组件 |
|---|---|---|---|
| 步骤 1：更新点估计 | mlp₀ + backbone | 更新 | 冻结 |
| 步骤 2：更新集成 | 各 mlp'_k | 冻结 | Prior 永远冻结 |

---

### 5.3 信息导向探索（Information-Directed Exploration）—— 选最值得标注的 pair

#### 核心流程

对每个 prompt X，用当前策略采样 16 个 response，然后：

1. 对所有 response pair (Y, Y')，让 100 个 ENN 粒子各自给出选择概率（公式 6）
2. 计算方差：Var[p_φₜ(Y≥Y'|X, Z)]，Z = 1,...,100
3. 选方差最大的 pair 送给人类标注（公式 7）

#### 直觉理解

| Pair | 各粒子给出的概率 | 方差 | 含义 |
|---|---|---|---|
| (Yₐ, Y_b) | 0.91, 0.89, ..., 0.92 | 很小 | 所有粒子都认为 Yₐ 更好 → RM 很确定 → 不值得问 |
| (Y_c, Y_d) | 0.82, 0.23, ..., 0.61 | 很大 | 粒子们严重分歧 → RM 很不确定 → **最值得问** |

#### 论文中的实例

对于一个情感分析 prompt：

**Infomin pair（方差最小，不值得问）：**
- Response 1: "Positive."
- Response 2: "Positive sentiment."
- → 本质相同，标了也白标

**Infomax pair（方差最大，最值得问）：**
- Response 1: "positive"
- Response 2: "Neutral."
- → 有实质分歧，人类选择能有效校准 RM

---

## 六、四种算法综合对比

| 维度 | Offline RLHF | Periodic RLHF | Online RLHF | Info-Directed |
|---|---|---|---|---|
| Response 生成 | 固定用 θ₀ | 周期性更新 | 每步用最新 θₜ | 每步用最新 θₜ |
| 送标 pair 选择 | 随机 | 随机 | 随机 | **方差最大化** |
| RM 更新 | 全部收完再训 | 周期性重训 | 增量更新 | 增量 + ENN |
| 不确定性建模 | 无 | 无 | 无 | ENN（100 粒子） |
| 数据效率 | 1x（基线） | ~3-5x | ~5-8x | **>10x，预计 1000x** |

---

## 七、结果与洞察

### 7.1 Win Rate 对比

四种算法在相同数据量下的表现差距显著。Information-directed exploration 在 20K 标签时就达到了 Offline RLHF 200K+ 标签的效果。

### 7.2 缩放曲线的本质变化

论文特别强调了**用对数刻度**观察缩放曲线的重要性。线性刻度下各算法差异不明显，但对数刻度清晰揭示了：高效探索从根本上**平移了缩放曲线**——相同性能所需数据量降低了数量级。

### 7.3 Response 质量对比

论文给出了一个数学题的例子：
- **Offline RLHF** 生成的 response：逻辑混乱，最终答案错误（得出 100/3 km，声称没有正确选项）
- **Information-directed exploration** 生成的 response：推理简洁清晰，正确得出答案 50 km（选项 A）

---

## 八、未来方向

论文提出了多个值得继续探索的方向：

1. **改进探索算法**：在更深层建模不确定性，不仅是 RM 的不确定性，还包括 LM 本身的不确定性
2. **Prompt 选择**：从「选最有信息量的 response pair」扩展到「选最有信息量的 prompt」
3. **多轮对话**：将方法扩展到优化多轮对话质量，引入价值模型预测未来奖励
4. **Agent 场景**：扩展到 AI Agent 的优化，处理动作产生延迟后果的场景
5. **AI 辅助反馈**：用 AI 辅助框架帮助人类做出更有信息量的评判

---

## 九、总结

这篇论文的核心启示是：**在 RLHF 中，「问什么」比「问多少」重要得多。**

三个创新分别解决了三个层面的问题：

- **Affirmative Nudge**（工程层）：一个极简但高效的技巧，解决了在线学习的稳定性问题
- **ENN**（建模层）：以极低的参数开销实现了对奖励不确定性的建模
- **Information-Directed Exploration**（决策层）：将不确定性转化为探索策略，让每次标注都用在刀刃上

三者结合，实现了从 10x 到预期 1000x 的数据效率提升。这不仅是一个算法上的进步，更指向了 RLHF 乃至整个 AI 对齐领域的一个根本性转变：**从被动收集数据到主动选择数据**。
