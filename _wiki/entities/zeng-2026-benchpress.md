---
layout: wiki
title: You Don't Need to Run Every Eval (BENCHPRESS)
wiki_type: entities
category: wiki
tags: [llm, evaluation, benchmark, low-rank, matrix-completion, microsoft]
description: 微软研究院证明大模型基准分矩阵近似 rank-2，BENCHPRESS 仅用 5 项测试即可补全全部成绩单
related: [coeval, autometrics-ryan-2026]
updated: 2026-06-28
---

# You Don't Need to Run Every Eval（BENCHPRESS）

Zeng & Papailiopoulos, *"You Don't Need to Run Every Eval"*, Microsoft Research, 2026。

针对大模型开发中“跑全套评估成本过高”（一遍可能要数千美元 + 数天算力）的痛点，论文给出一个有数据支撑的结论：**绝大多数基准测试是冗余的，只需跑 5 个关键测试即可预测剩余分数。**

## 核心发现：分数矩阵近似 rank-2

研究收集了 **84 个前沿模型 × 133 个基准** 共 2,604 个真实分数。通过奇异值分解（SVD）发现：

> the score matrix is effectively rank-2: a model's scores across all 133 benchmarks are largely determined by just two numbers.

即一个模型在 133 个维度上的表现可压缩为 2 个核心维度，可大致理解为：

- **基础语言/常识能力**
- **逻辑/推理硬实力**

确定这两个值后，其余分数基本可推导。

## BENCHPRESS 预测器

基于低秩特性的成绩单补全工具，原理类似推荐系统的“猜你喜欢”：

1. **输入**：模型已知的几个分数（如 MMLU、GSM8K）
2. **计算**：在“模型 × 测试”矩阵中定位，利用低秩规律做 logit 空间矩阵补全
3. **输出**：补全整张成绩单，中位绝对误差（MedAE）约 **4.6 分**

工作流：新模型 → 跑 5 个核心测试 → BENCHPRESS（矩阵补全 + 可靠性分析层）→ 全维度分数预测 + 哪些预测可信。

## 两套“黄金探测器”组合

只跑这 5 项即可高精度还原整体水平：

**最佳预测组合（MedAE 3.93）** — 不计成本求最准：
1. GPQA-D（研究生级问答）
2. HLE（Humanity's Last Exam）
3. Codeforces（编程竞技）
4. MMLU-Pro（强化版多任务理解）
5. ARC-AGI-1（抽象推理）

**低成本组合（MedAE 4.55）** — 省钱跑得快：
1. GPQA-D
2. MMLU-Pro
3. Aider Polyglot（多语言编码）
4. MATH-500
5. AIME 2026

选对“风向标”误差迅速降到 4 分以内；随便选则跑 10 个也看不清真实成色。

## 可靠性估计器

混合可靠性估计器用 **集成离散度（ensemble spread）+ 矩阵支撑度（matrix support）** 在跑测试前判断哪些预测低风险。预测通常在以下情况失效：

- **模型太特殊**：如垂直领域（医疗）模型，能力分布偏离通用模型规律
- **测试集太新**：矩阵中尚无该基准数据，无法补全

## 开发者避坑要点

1. 不要迷信全量榜单——许多分数强相关，看完 A 已知 B
2. 重视高区分度硬核测试（GPQA、MATH）作为预测“锚点”
3. 动态监测——随推理模型崛起，低秩结构可能变化，需定期更新矩阵

## 参考资料

- 论文：Zeng & Papailiopoulos, "You Don't Need to Run Every Eval", Microsoft Research (2026)
- [Rank-2 Geometry](https://www.alphaxiv.org/abs/2606.24020v1?page=1)
- [Reliability](https://www.alphaxiv.org/abs/2606.24020v1?page=19)
