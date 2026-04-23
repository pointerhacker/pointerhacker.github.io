---
layout: wiki
title: PaddleOCR-VL
wiki_type: entities
category: wiki
tags: [OCR, VLM, 百度, PaddlePaddle, 文档解析, PDF]
description: 百度开源轻量级视觉-语言文档解析模型，0.9B参数支持109语言，OmniDocBench全球SOTA
related: []
updated: 2026-04-23
---

# PaddleOCR-VL

百度开源的**多模态文档解析视觉-语言模型**，将复杂文档解析任务分解为两阶段：版面分析（PP-DocLayoutV2） + 细粒度元素识别。

## 核心架构

**PaddleOCR-VL-0.9B** = NaViT 风格动态分辨率视觉编码器 + 轻量级 ERNIE-4.5-0.3B 语言模型

- **NaViT 动态分辨率**：处理任意尺寸文档图像，避免切图丢失跨区域上下文
- **轻量语言模型**：0.9B 参数，资源消耗低，适合高效部署
- 两阶段 Pipeline：
  1. **PP-DocLayoutV2** 版面分析 → 定位语义区域 + 预测阅读顺序
  2. **PaddleOCR-VL-0.9B** → 对文本、表格、公式、图表进行细粒度识别
  3. 聚合输出 → 结构化 Markdown / JSON

## 版本迭代

| 版本 | 发布日期 | 关键特性 |
|------|----------|----------|
| PaddleOCR-VL 1.0 | 2025-10-16 | 首次开源，0.9B 参数，109 语言，连续5天登顶 HuggingFace/ModelScope 总趋势榜 |
| **PaddleOCR-VL 1.5** | **2026-01-29** | **OmniDocBench 94.5% 全球第一，异形框定位突破，印章识别，跨页表格合并** |

## PDF / 文档解析能力

> 这是该模型的核心能力，也是相比传统 OCR 的本质区别。

### 支持的文档元素

| 元素类型 | 说明 |
|---------|------|
| 文本 | 手写汉字、印刷体、古籍文献、生僻字 |
| 表格 | 表格结构理解（含跨页表格自动合并） |
| 公式 | 数学公式、化学公式 |
| 图表 | 图表识别与描述 |
| 印章/印章 | 新增（1.5版） |
| 版面逻辑 | 阅读顺序预测、段落标题识别（跨页） |

### 异形框定位（1.5版核心突破）

> 全球首次实现 OCR 模型的"异形框定位"能力，使机器能够精准识别倾斜、弯折、拍照畸变等非规则文档形态。

解决传统 OCR 在以下真实场景中的识别失败问题：
- 📱 **移动拍照**：手机拍摄文档产生的透视畸变
- 📄 **扫描件变形**：老旧扫描仪导致的弯曲、倾斜
- 💡 **复杂光照**：反光、阴影、背景干扰
- 📐 **异形文档**：不规则边界文档

### 多语言支持

- 覆盖 **109 种语言**
- 新增支持：藏语、孟加拉语
- 优化场景：下划线、复选框等复杂结构

## 性能表现

### OmniDocBench v1.5 评测（全球权威文档解析榜单）

| 指标 | PaddleOCR-VL 1.5 | 对比模型 |
|------|-----------------|---------|
| **综合精度** | **94.5%** | 超 Gemini-3-Pro、DeepSeek-OCR2、Qwen3-VL-235B-A22B、GPT-5.2 |
| **表格结构理解** | **92.8 分** | 领先 Gemini-3-Pro、DeepSeek-OCR 2-5 分 |
| **阅读顺序预测** | **95.8 分** | 版面逻辑解析错误率仅为同类模型约一半 |

### 推理速度（A100）

| 模型 | Token/s |
|------|---------|
| **PaddleOCR-VL** | **1881** |
| MinerU 2.5 | 快 14.2% |
| dots.ocr | 快 253.01% |

## 硬件支持

| 推理方式 | NVIDIA GPU | x64 CPU | Apple Silicon | 华为昇腾 NPU | AMD GPU |
|---------|-----------|---------|---------------|-------------|---------|
| PaddlePaddle | ✅ | ✅ | ✅ | 🚧 | ✅ |
| + vLLM | ✅ | - | - | ✅ | ✅ |
| + llama.cpp | ✅ | ✅ | - | 🚧 | 🚧 |
| + MLX-VLM | - | - | ✅ | - | - |

## 快速使用

### CLI

```bash
paddleocr doc_parser -i ./document.pdf
# 启用文档方向分类
paddleocr doc_parser -i ./doc.png --use_doc_orientation_classify True
# 启用文档去扭曲
paddleocr doc_parser -i ./doc.png --use_doc_unwarping True
# 指定 pipeline 版本（1.0 或 1.5）
paddleocr doc_parser -i ./doc.pdf --pipeline_version v1.5
```

### Python API

```python
from paddleocr import PaddleOCR

ocr = PaddleOCR(
    doc_parser=True,
    pipeline_version='v1.5',
    use_doc_unwarping=True,
    use_doc_orientation_classify=True
)
result = ocr.doc_parser('./document.pdf')
```

## 部署方式

- 🐳 **Docker**：百度官方镜像（一键启动，离线镜像约 10GB）
  ```bash
  docker run --gpus all ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddleocr-vl:latest-nvidia-gpu
  ```
- ☁️ **API 服务**：百度智能云千帆平台
- 🔧 **微调**：支持针对特定业务场景微调

## 相关条目

- [[PaddlePaddle]] — 百度深度学习平台

## 参考资料

- [GitHub - PaddlePaddle/PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)
- [PaddleOCR 官方文档](http://www.paddleocr.ai/v3.3.0/en/version3.x/pipeline_usage/PaddleOCR-VL.html)
- [ModelScope 发布博客](https://www.modelscope.cn/learn/2078)
- [CSDN PaddleOCR-VL-1.5 发布解读](https://csdnnews.blog.csdn.net/article/details/157516813)
