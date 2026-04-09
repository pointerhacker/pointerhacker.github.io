---
layout: wiki
title: VoxCPM 2
wiki_type: entities
category: wiki
tags: [TTS, 开源, 多语种, 面壁智能, OpenBMB]
description: 面壁智能联合 OpenBMB 开源的多语种高保真 TTS 模型，2B 参数，支持 30 种语言和 9 种中文方言，48kHz 高音质。
related: []
updated: 2026-04-09
---

# VoxCPM 2

面壁智能联合 OpenBMB 开源社区、清华大学人机语音交互实验室研发的**多语种高保真 TTS（文本转语音）模型**，仅 2B 参数，主打「小钢炮」系列。

开源地址：[GitHub](https://github.com/OpenBMB/VoxCPM/) · [ModelScope](https://modelscope.cn/models/OpenBMB/VoxCPM2)

## 核心能力

VoxCPM 2 将多个传统 TTS 系统的能力统一到一个 2B 模型中：

| 能力 | 说明 |
|------|------|
| 多语种 | 支持全球 30 种主流语言 |
| 方言支持 | 9 种中国方言 |
| 音色设计 | 纯文字描述凭空创造全新音色 |
| 音色克隆 | 通用音色可控，高度还原原声 |
| 高音质 | 48kHz 高保真（Hi-Fi） |

## 一、全球通：30国语言 + 9大方言

### 国际语言

覆盖全球 30 种主流语言，尤其在东南亚 8 国语种上做了专项优化：

越南语、泰语、印尼语、老挝语、缅甸语、柬埔寨语、菲律宾语、马来西亚语。

### 中国方言

掌握 9 大中国方言：四川话、粤语、吴语（上海话）、东北话、河南话、陕西话、山东话、天津话、闽南语。

## 二、百变声优：音色设计

不需要参考声音，纯靠文字描述凭空创造全新音色。

输入一段文字描述（音色、情绪、性别、年龄等要求），模型一口气生成 7 个不同音色 вариантов。

典型场景：配音师不想暴露真人声音、又找不到合适配音时。

## 三、千人千面：通用音色可控（语音克隆）

传统 TTS 的语音克隆往往只支持少数固定音色控制。VoxCPM 2 采用**扩散自回归连续表征（Continuous Representation）**技术路线，实现真正意义上的通用音色可控。

相比主流 Token-based 方法，扩散自回归架构能保留更多声学信息，实现情感起伏、情绪变化乃至细微呼吸声的高度还原。

## 四、影视级配音：高音质 + 高表现力

| 采样率 | 质量等级 |
|--------|---------|
| 8kHz | 仅满足通话 |
| 16kHz | 清晰音质 |
| **48kHz** | **高保真 Hi-Fi（VoxCPM 2）** |

高音质让 AI 语音可进入对音质要求更高的领域，如影视配音、有声书、视频创作。

## 技术特性

- **全参数微调** + **LoRA 微调** 均支持
- 原生 PyTorch 推理
- 企业级算力和个人小破本均可运行
- 开源、免费

## VoxCPM 1 vs VoxCPM 2

VoxCPM 1 仅支持中英双语；VoxCPM 2 扩展至 30 种语言 + 9 种中文方言，并新增音色设计、通用音色克隆能力。

## 参考资料

- GitHub: https://github.com/OpenBMB/VoxCPM/
- ModelScope: https://modelscope.cn/models/OpenBMB/VoxCPM2
- Demo: https://modelscope.cn/studios/OpenBMB/VoxCPM2-Demo
- 公众号原文: https://mp.weixin.qq.com/s/QEuziYy4eAZ-2DXNKFk8Ug
