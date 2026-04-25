---
layout: wiki
title: AutoMetrics (Ryan et al., 2026)
wiki_type: entities
category: wiki
tags: [LLM评估, LLM-as-a-Judge, 度量学习, ICLR2026]
description: AutoMetrics — 从稀疏人类反馈中自动合成评估指标，在 5 个任务上比 LLM-as-a-Judge 的 Kendall 相关性提升最高 33.4%
related: [llm-evaluation, metric-learning]
updated: 2026-04-25
---

# AutoMetrics: Approximate Human Judgments with Automatically Generated Evaluators

> 来源：[OpenReview](https://openreview.net/pdf?id=ymJuBifPUy) | ICLR 2026
> 作者：Michael J. Ryan, Yanzhe Zhang (Stanford), Amol Salunkhe, Yi Chu, Di Xu (American Express), Diyi Yang (Stanford)

## 一句话总结

从少于 100 个稀疏人类反馈信号中，自动合成可解释的评估指标，Kendall 相关性比 LLM-as-a-Judge 最高提升 33.4%。

## 核心问题

开放域 AI 应用（旅行规划、临床笔记生成、对话等）的评估难题：
- **用户反馈**（点赞/踩）是金标准，但原型阶段稀少，优化系统太慢
- **Reward Model** 需要数千条标签，成本高
- **LLM-as-a-Judge** 依赖清晰的任务定义，不保证严格遵循评分标准

## 方法：AutoMetrics 四步流程

```
(1) Generate → (2) Retrieve → (3) Regress → (4) Report
```

### Step 1: Generate（生成候选指标）

为每个任务生成 4 类候选指标：
- **Single Criterion LLM Judge** × 10：用 LLM 直接打分（单维度）
- **Rubric LLM-Judge** × 5：分步骤 rubrics 评分
- **Example-based LLM-Judge** × 1：few-shot 示例优化
- **Prompt-Optimized LLM-Judge** × 1：prompt 工程优化

### Step 2: Retrieve（检索）

从两个来源检索候选指标：
1. **自己生成的候选**（Step 1）
2. **MetricBank**：NLP 文献中精选的 48 个指标，每个有标准 Metric Card

混合 ColBERT + LLM 检索，过滤到最相关的指标组合。

### Step 3: Regress（回归加权）

用 **Partial Least Squares (PLS) 回归**将候选指标组合为预测信号：

- PLS 投影到与人类标签最相关的方向
- 两阶段：第一阶段筛选 top-n 指标，第二阶段重新拟合
- 移除负相关的 LLM 生成指标（设计期望正相关）

数学：
$$w^* = \arg\max_{\|w\|_2=1} \text{cov}(Xw, y)^2$$

### Step 4: Report（输出报告）

产出：可解释的指标权重 + 相关性分析 + LLM 判断推理痕迹，供人检查和系统优化使用。

## 核心实验结果

| 任务 | 相对 LLM-as-a-Judge 提升 |
|------|--------------------------|
| 5 个不同任务 | Kendall 相关性最高 +33.4% |

- **数据效率**：仅需 ~80 个反馈点
- **作为 Proxy Reward**：效果可等同于可验证奖励（verifiable reward），直接优化下游 AI 系统

## 关键洞察

1. **自适应指标生成**比固定 rubric 更重要——MetaMetrics 用预定义指标组合，效果不如 AutoMetrics
2. **PLS 回归**在指标数量接近样本量、指标间高度相关时优于 OLS
3. **轻量人类反馈**（<100 点）足够诱导有效指标
4. 产出的指标可解释——能揭示"用户真正在乎什么"

## 局限性

- 依赖 LLM Judge 的质量上限
- 少数反馈点时，生成指标多样性可能不足
- MetricBank 目前仅覆盖 48 个 NLP 指标

## 与相关工作的区别

| 工作 | 区别 |
|------|------|
| G-Eval | 用 CoT + logprob 打分，但不给 rubric |
| ChatEval | 多 Agent 辩论，不做指标合成 |
| EvalGen | 有人类反馈 + rubric 迭代精炼，但不做自动检索和回归加权 |
| MetaMetrics | 预定义指标组合，不是自适应生成 |
| AutoMetrics | **自适应生成 + MetricBank 检索 + PLS 回归加权** |

## 资源

- 代码：https://github.com/SALT-NLP/autometrics
- MetricBank：48 个精选 NLP 评估指标库
