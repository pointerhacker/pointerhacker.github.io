---
layout: wiki
title: MemPO（Self-Memory Policy Optimization）
wiki_type: entities
category: wiki
tags: [LLM Agent, Reinforcement Learning, Memory Mechanism, Long-Horizon Task, arXiv, 2026]
description: 让 LLM Agent 自主管理记忆的 RL 算法，通过记忆级 dense reward 解决长程任务中的信用分配问题
related: [grpo]
updated: 2026-04-12
---

# MemPO: Self-Memory Policy Optimization for Long-Horizon Agents

> **论文**：[arXiv:2603.00680](https://arxiv.org/abs/2603.00680) | 提交于 2026-02-28 | 作者：Ruoran Li 等

## 核心问题

长程 Agent 在与环境多轮交互时，上下文不断膨胀，导致性能下降、稳定性变差。现有方案依赖**外部记忆模块 + RAG 检索**，缺点是：
- 模型无法主动管理记忆内容
- 检索只基于 embedding 相似度，不一定找到对解题最有用的信息
- 无法端到端联合优化

## 核心思想

**让 Agent 自主"决定记什么、忘什么"**，把记忆管理变成模型自身能力的一部分，而不是外挂的存储系统。

推理时只保留**上一步的交互内容**作为下一轮输入，更早的历史全部通过 `<mem>` 机制压缩进记忆。

## 技术方案

### Agent 输出格式

每个交互步骤 $s_t$ 由四部分组成：

$$
s_t = \{ s_t^{mem},\ s_t^{think},\ s_t^{call},\ s_t^{resp} \}
$$

- `<mem> </mem>`：模型生成的记忆摘要
- `<think>  </span>`：推理过程
- `<tool_call> </tool_call>`：工具调用
- `<information> </information>`：工具返回结果

### 双层 Advantage 机制（MemPO 核心）

基于 GRPO（Group Relative Policy Optimization），设计**双层 Advantage**：

#### ① Trajectory-level Advantage（轨迹级）

标准 GRPO 逻辑，衡量最终答案是否正确：

$$
R^T(\tau_i) = \begin{cases} 1 & \text{答案正确 + 格式正确} \\ 0 & \text{否则} \end{cases}
$$

$$
A^T(\tau_i) = \frac{R^T(\tau_i) - \text{mean}(R^T)}{\text{std}(R^T)}
$$

这是**稀疏奖励**，只衡量最终结果，不反映中间每步记忆的质量。

#### ② Memory-level Advantage（记忆级）— 核心创新

对每个 `<mem>` 动作设计**密集奖励**：

$$
R^M(\tau_i, s_t^{mem}) = P[\text{正确回答} \mid s_t^{mem}] - \epsilon
$$

**实现方式**：

1. 提取 `<mem> </mem>` 内的记忆文本 $s_t^{mem}$
2. 仅以这段记忆为上下文（不加其他历史），让模型回答原问题 q
3. 计算正确答案的条件概率 $P(s^{ans} \mid s^{mem}) = \exp(-\text{NLL}(s^{ans} \mid s^{mem}))$
4. 减 $\epsilon$ 作为最终记忆奖励

**直觉**：记忆越好，仅凭它就能让模型答对（猜对）的概率越高。

在组内归一化后：

$$
A^M(\tau_i, s_t^{mem}) = \frac{R^M - \text{mean}(R^M)}{\text{std}(R^M)}
$$

#### ③ 最终 Advantage

$$
A = A^T + \alpha \cdot A^M
$$

$\alpha$ 控制记忆奖励的权重。记忆 token 的 advantage 由两层合并得到，其他 token（如 think、call）只有 $A^T$。

### 训练流程

1. **Behavior Cloning 热启动**：用 GPT-4.1 在公开数据集上推理，滤掉答案错误的轨迹，生成约 10K 条符合格式的轨迹
2. **MemPO RL 微调**：基于 GRPO 框架，结合双层 advantage 优化策略

## 实验结果

| 指标 | 对比基线 | 对比 SOTA |
|------|---------|-----------|
| F1 提升 | +25.98% | +7.1% |
| Token 消耗降低 | -67.58% | -73.12% |

在 5 个长程基准任务上验证，基线包括 MemGPT、Mem0、RAG-based Memory、MEM1。

## 关键洞察

1. **记忆质量的显式信用分配**：把"记忆好不好"从最终答案对错中解耦出来，用条件概率显式衡量——这是 MemPO 最大的创新
2. **不依赖外部标注**：正确答案来自训练集本身，条件概率可由模型 forward pass 直接计算，无需额外人工打分
3. **局限**：条件概率只能衡量记忆对答案的**直接贡献**，中介性记忆（需结合后续推理才生效）可能被低估

## 局限性

- $P(ans \mid mem)$ 只能反映记忆对最终答案的直接贡献，无法捕捉记忆在中间推理步骤中的中介价值
- 论文对 $\epsilon$ 的取值和 $\alpha$ 的 ablation 分析较简略
- 实验任务数量有限（5个），在更复杂任务（如超长程规划）上的效果有待验证

## 参考资料

- [MemPO 原文 arXiv:2603.00680](https://arxiv.org/abs/2603.00680)
- GRPO：[Shao et al., 2024](https://arxiv.org/abs/2402.03300)
