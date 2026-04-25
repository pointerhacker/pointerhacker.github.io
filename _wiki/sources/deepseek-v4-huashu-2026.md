---
layout: wiki
title: DeepSeek V4 论文深度解读（花叔）
wiki_type: sources
category: wiki
tags: [DeepSeek, 大模型, MoE, 长上下文, 论文解读]
description: 花叔对 DeepSeek V4 技术报告的 58 页超长解读，涵盖架构改进、训练细节与后训练范式变化
related: [deepseek-v3, mhc, mixture-of-experts]
updated: 2026-04-25
---

# DeepSeek V4 论文深度解读（花叔）

> 来源：花叔 Twitter (@AlchainHust) | https://x.com/AlchainHust/status/2047711336357126372
> 原始链接：https://huasheng.ai/

## 核心结论

**不是冲破 AGI 天花板的模型，而是让普通开发者第一次能用上百万上下文 Agent 模型的发布。**

- 闭源旗舰卷天花板，开源模型卷地板。V4 把地板往上抬了抬。
- V4-Flash（便价版）才是 DeepSeek 一贯风格，参数只有 V4-Pro 的 1/6，但很多基础能力已反超 V3.2。

## V4-Pro 和 V4-Flash：两个定位

| 模型 | 参数量 | 激活参数 | 定位 | 价格 |
|------|--------|----------|------|------|
| V4-Pro | 1.6T（V3 的 2.4 倍） | 49B（V3 的 1.3 倍） | 开源阵营能与闭源旗舰掰手腕 | 较贵（高端算力受限） |
| V4-Flash | 约 270B | 未明确 | 真正符合 DeepSeek 一贯风格 | 约为同类快速模型的 1/7~1/18 |

## 三大架构改动

### 1. mHC：残差连接升级

**解决的问题：** 深度堆叠时数值不稳定，限制模型做大。

- 2024 年底的 Hyper-Connections（HC）把单通道残差扩展成多通道，但训练不稳定——27B 模型上信号放大峰值达 3000 倍。
- mHC 加了一道「只准收缩不准放大」的数学护栏，用「双随机矩阵」保证守恒。
- **直接结果：** V4 从 V3 的 671B 推到 1.6T，参数量 2.4 倍增长，训练稳定性反而更好。

### 2. CSA + HCA：注意力机制拆分

**解决的问题：** 100 万上下文的 KV cache 爆炸，算力扛不住。4K → 1M，内积数量是 4000 倍，成本高约 6 万倍。

- **CSA（Compressed Sparse Attention）**：每 64 个 token 压缩成 1 块，用 Lightning Indexer 挑最相关的 top-k 块做精细注意力。→ 「扫小标题定位」
- **HCA（Heavily Compressed Attention）**：1024 个 token 压成 1 块，新 token 全扫描所有压缩块。→ 「翻目录看大意」
- 两者交替安排在不同层。
- **实际效果：** 100 万上下文单次推理成本只有 V3.2 的约 1/4，KV cache 占用只有传统 GQA8 baseline 的约 2%。

### 3. Muon：优化器替换 AdamW

**解决的问题：** AdamW 单独调每个旋钮，多旋钮联动时探索范围窄，模型会偏科。

- Muon 看整组旋钮的协同方向，把扁椭圆「掰成正圆」，让每个方向都走一样远。
- 矩阵分解太贵，用 Newton-Schulz 迭代 10 步近似。
- embedding、prediction head、RMSNorm 等非矩阵参数仍用 AdamW。

## 1.6T 怎么训稳的：两个不完全理解的 trick

### Anticipatory Routing（预判式路由）

路由器用「昨天的脑子」做「今天的决定」——主干网络更新与路由器解耦，路由器查前几步历史参数，恶性循环断了。

### SwiGiLU Clamping

给 SwiGLU 激活函数加上下限（-10 到 10），哪怕某神经元想输出一万，也只能给 10。

> DeepSeek 自己在论文里写："the underlying principles remain insufficiently understood."

## 训练数据

- **规模：** Pro 33T tokens，Flash 32T tokens（V3 为 14.8T）
- **关键动作：**
  - 反模型坍缩：过滤批量 AI 生成内容
  - 中期训练引入 Agent 数据（工具调用、多步推理）
  - 多语言扩容（中英外长尾语言）
  - 精选长文档（科学论文、技术报告）
- **序列长度分阶段扩展：** 4K → 16K → 64K → 1M
- **稀疏注意力分阶段引入：** 前 1T tokens 用 dense attention 热身，到 64K 时切到 sparse attention

## 后训练：Specialist + OPD（被低估的范式变化）

**核心变化：** 混合 RL 阶段被彻底替换为 On-Policy Distillation（OPD）。

### 为什么替换

传统 SFT+RLHF 混炼路子：多任务联合优化时，数学、代码、Agent、对话会互相打架（负迁移问题）。

### 怎么做

- **Stage 1 Specialist**：每个领域（推理/数学/代码/Agent/对话）单独训练专家模型，先 SFT 再 GRPO 做 RL
- **Stage 2 OPD**：十多个专家当老师，通过**反向 KL loss**蒸馏出统一学生模型

### 反向 KL 关键

- 正向 KL：学生 cover 老师所有模式 → 四不像
- 反向 KL：学生集中在老师的高概率区域 → 自动路由：数学任务对齐数学专家，代码任务对齐代码专家

### 意义

- MoE 是推理时混合，OPD 是训练时混合，组合空间大得多
- 适合小团队（多训小专家再蒸馏）、垂直应用（法律/医疗/代码各训一个）、持续学习（加新能力只需加新专家）
- **可能是比 MoE 更深刻的范式变化**

## 评测结果

### 强项

- **数学推理**：V4-Pro 在 AIME 等 benchmark 拿到开源前所未有高分
- **编程**：LiveCodeBench 93.5 分，Codeforces Rating 3206（人类选手第 23 名），首次开源模型追平闭源
- **中文写作**：V4-Pro 对 Gemini 碾压级，但复杂指令跟随仍落后 Opus 4.5

### 弱项

- **Agent 能力**：全方位落后闭源，Terminal Bench 2.0 落后 GPT-5.4 整整 7 分
- **真实工程编程**：SWE 系列与 Opus 4.6 Thinking 差约 13 个百分点
- **品味型任务**：创意写作输 Opus 4.5，长程 Agent 落后 GPT-5.4

### 长上下文

- 128K 以内稳如狗（MRCR 8-needle 测试 0.9 以上）
- 256K 开始掉到 0.82，1024K 降至 0.59

## 为什么 V4 这么偏科

> DeepSeek 招聘以竞赛获奖选手为主——擅长在给定规则下把单点做到极致，擅长解有明确答案的题。

- **强项**：有明确答案的任务（数学竞赛、竞赛编程）
- **弱项**：需要综合品味的任务（创意写作、长链 Agent、通用工程编程）

## 花叔的评价

> "不诱于誉，不恐于诽，率道而行，端然正己。"
> ——DeepSeek 发布 V4 时引用的话，这句话可能比 58 页论文技术细节更能解释这家公司。

- DeepSeek 的「Open」：R1 的 86 页更新补全了训练账单和数据配方，V4 的 58 页继续补全基础设施的每个缝隙。不是「开源权重就完了」的 Open，是一份让别人真的能复现的 Open。
- V5 轮廓预测：原生多模态、引入可扩展查找式记忆（Engram 路线）、进一步降低延迟、更长的 long-horizon multi-round agentic 能力。

## 参考资料

- DeepSeek V4 技术报告：https://github.com/deepseek-ai/DeepSeek-V4
- 花叔频道：https://huasheng.ai/
- 花叔 YouTube/B站：20万+粉丝频道
- DeepSeek R1 论文 v2（86页）：https://arxiv.org/abs/2501.12948
