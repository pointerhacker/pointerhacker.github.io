---
layout: wiki
title: PaddleOCR-VL
wiki_type: entities
category: wiki
tags: [OCR, VLM, 百度, PaddlePaddle, 文档解析]
description: 百度开源的轻量级视觉-语言模型，用于文档解析，支持109种语言
related: []
updated: 2026-04-23
---

# PaddleOCR-VL

百度开源的**轻量级视觉-语言模型（VLM）**，专为文档解析任务设计。

## 核心架构

核心组件为 **PaddleOCR-VL-0.9B**，融合了：
- **NaViT 风格动态分辨率视觉编码器**：处理任意尺寸的文档图像
- **ERNIE-4.5-0.3B 语言模型**：理解文档语义

## 版本迭代

| 版本 | 发布日期 | 关键特性 |
|------|----------|----------|
| PaddleOCR-VL-1.0 | 2025年10月16日 | 初始开源，支持109种语言 |
| PaddleOCR-VL-1.5 | 2026年1月29日 | OmniDocBench 准确率 94.5%，支持异形框定位、印章识别 |

## 主要能力

- 📄 **文档元素识别**：文本、表格、公式、图表
- 🌍 **多语言支持**：覆盖 109 种语言
- 📐 **异形框定位**：全球首次突破倾斜、弯曲、拍照、光照、扫描等复杂场景
- 🔏 **印章识别**：新增印章/印章检测和识别能力
- ⚡ **轻量高效**：仅 0.9B 参数，资源消耗低

## 性能表现

- **OmniDocBench v1.5** 评测综合性能全球第一
- 准确率达 94.5%
- 推理速度显著优于传统 Pipeline 方案和通用多模态大模型

## 推理硬件支持

| 推理方式 | NVIDIA GPU | x64 CPU | Apple Silicon | 华为昇腾 NPU |
|---------|-----------|---------|---------------|-------------|
| PaddlePaddle | ✅ | ✅ | ✅ | 🚧 |
| PaddlePaddle + vLLM | ✅ | - | - | ✅ |
| PaddlePaddle + llama.cpp | ✅ | ✅ | - | 🚧 |
| PaddlePaddle + MLX-VLM | - | - | ✅ | - |

## 快速使用

```bash
# NVIDIA GPU
paddleocr doc_parser -i https://example.com/doc.png

# 启用文档方向分类
paddleocr doc_parser -i ./doc.png --use_doc_orientation_classify True

# 启用文档去扭曲
paddleocr doc_parser -i ./doc.png --use_doc_unwarping True
```

## 相关条目

- [[PaddlePaddle]] - 百度深度学习平台

## 参考资料

- [GitHub - PaddlePaddle/PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)
- [PaddleOCR 官方文档](http://www.paddleocr.ai/v3.3.0/en/version3.x/pipeline_usage/PaddleOCR-VL.html)
- [PaddleOCR-VL-1.5 发布博客](https://csdnnews.blog.csdn.net/article/details/157516813)
