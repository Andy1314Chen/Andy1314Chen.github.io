---
title: "VLM (Vision Language Model) 介绍"
date: 2026-04-13
tags:
  - "VLM"
  - "多模态"
  - "InternVL"
  - "推理优化"
description: "从 InternVL2 出发，系统梳理 VLM 的三大组件、Token 处理机制、推理优化策略，以及主流模型横向对比。"
toc: true
draft: false
---

## 一、架构概览

VLM 由三个核心组件串联而成：

- **视觉编码器 Vision Encoder**：将原始图像/视频转换为视觉特征向量
- **模态投影层 Projector**：将视觉特征空间映射到语言模型的语义空间，解决"视觉-语言鸿沟"
- **语言模型 LLM Backbone**：理解对齐后的视觉-文本联合表示，生成文本输出

以 InternVL2-2B 为例：

| 模型           | 视觉编码器          | 语言模型       | 总参数量 | 推理显存    |
| -------------- | ------------------- | -------------- | -------- | ----------- |
| InternVL2-2B   | InternViT-300M      | Qwen2-1.5B     | ~2B      | ~4.2 GB     |

```text
输入图像 (支持动态分辨率)
         |
         v
+---------------------------------------------------+
|  动态高分辨率处理 (Dynamic Tiling)                |
|  · 图像分割: 将大图切分为 N 个 448x448 tile       |
|  · 网格排列: 支持 1x1 到 6x6 (最多 40 个 tile)   |
+---------------------------------------------------+
         |  N x (448x448 像素)
         v
+---------------------------------------------------+
|  视觉编码器 (Vision Encoder)                      |
|  InternViT-300M-448px                             |
|  · 参数量: 304M                                   |
|  · 层数: 24 层 Transformer                        |
|  · 隐藏维度: 1024                                 |
|  · Patch Size: 14x14  ->  32x32 = 1024 tokens    |
|  · 位置编码: 2D 绝对位置编码                      |
+---------------------------------------------------+
         |  1024 tokens/tile (每个 1024 维)
         |  Pixel Unshuffle (空间压缩 4:1)
         v  256 tokens/tile  (减少计算量 75%)
+---------------------------------------------------+
|  模态投影层 (MLP Projector)                       |
|  · 类型: 2 层全连接网络 + GELU 激活               |
|  · 作用: 视觉空间 (1024d) -> 语言空间 (1536d)     |
|  · 可学习参数，随机初始化                         |
+---------------------------------------------------+
         |  256xN visual tokens (每个 1536 维)
         v
+---------------------------------------------------+
|  语言模型 (LLM Backbone)                          |
|  Qwen2-1.5B-Instruct                              |
|  · 参数量: 1.5B                                   |
|  · 架构: Decoder-only Transformer，28 层          |
|  · 注意力: 12 头 GQA                              |
|  · 隐藏维度: 1536 / 上下文: 32K / 词表: 151936   |
|  · 位置编码: RoPE                                 |
|                                                   |
|  输入序列: [ V0..V(256xN) | T0..Tn ]              |
|            visual tokens    text tokens           |
+---------------------------------------------------+
         |
         v
输出文本 (自回归生成: 描述 / 问答 / OCR / Grounding)
```

---

## 二、Token 与 Patch

> Patch 是将图像分割为小块的基本单元，是模型处理视觉信息的最小粒度。

Patch 与文本 LLM 的 token 并不是同一概念，三者的区别：

| 概念               | 含义                       | 在 VLM 中的流转                   |
| ------------------ | -------------------------- | --------------------------------- |
| **Image Patch**    | 图像的局部区域 (14x14 像素) | 原始图像 -> ViT                   |
| **Visual Token**   | Patch 经编码后的特征向量    | ViT 输出 -> MLP Projector -> LLM  |
| **Text Token**     | 文本经 tokenizer 后的 ID   | Tokenizer -> LLM                  |

### 文本 Token 的工作方式

```text
文本: "你好"
  | Tokenizer (BPE/SentencePiece)
  v
Token ID: [13244, 9125]        <- 整数索引
  | Embedding Layer (查表)
  v
Embedding: [0.23, -0.45, ...]  (1536 维向量)
  | 输入 Transformer
  v
```

### 视觉 Token 的工作方式

以 InternViT-300M 为例（Patch Size = 14x14）：

```text
图像: 448x448 像素
  | 分割为 14x14 的小块
  v
Patch 网格: 32x32 = 1024 个 patches  (每个 Pi 是 14x14x3 = 588 维)
  | 线性投影到模型维度
  v
Embedding: 1024 个向量  (每个 Ei 是 1024 维)
  | InternViT-300M (24 层 Transformer)
  v
特征图: 1024 个 patch 特征  (每个 1024 维)
  | Pixel Unshuffle  (2x2 -> 1，压缩 4:1)
  v
压缩特征: 256 个向量  (每个 1024 维)
  | MLP Projector
  v
Visual Tokens: 256 个向量  (每个 1536 维)  <- 直接是向量，无整数 ID！
  | 直接输入 LLM 的 Transformer 层
  v
```

Visual token 没有经过 tokenizer，没有对应的整数 ID，是 ViT 输出的连续特征，经过 MLP 对齐后直接输入 LLM。

### 关键维度流转

| 阶段                | 形状变化                                  | 说明                          |
| ------------------- | ----------------------------------------- | ----------------------------- |
| **输入图像**         | `(H, W, 3)` -> `(N, 448, 448, 3)`        | 动态切分为 N 个 tile           |
| **Patch Embed**     | `(N, 448, 448, 3)` -> `(N, 1024, 1024)`  | 14x14 patch，1024 tokens/tile |
| **Pixel Unshuffle** | `(N, 1024, 1024)` -> `(N, 256, 1024)`    | 空间重排压缩 4x               |
| **MLP Projector**   | `(Nx256, 1024)` -> `(Nx256, 1536)`       | 对齐到 Qwen2 维度              |
| **Qwen2-1.5B**      | `(seq_len, 1536)` -> `(vocab_size)`      | 自回归生成文本                 |

### 实际输入序列（InternVL2）

```text
# 假设 1 张图 + 文本 "描述这张图片"
input_sequence = [
    [V0], [V1], [V2], ..., [V255],  # 256 个 visual tokens (向量，来自 ViT + Projector)
    [T0], [T1], [T2], ..., [Tn]    # n 个 text tokens  (来自 Tokenizer + Embedding)
]
```

---

## 三、推理优化

### 3.1 推理两阶段

VLM 推理比纯文本 LLM 多了图像编码步骤：

```text
阶段 1: 图像编码 (Image Prefill)
  输入图像
    | Vision Encoder + Pixel Unshuffle + MLP Projector
    v
  生成 256xN visual tokens，计算其 KV Cache
  特点: 计算密集，耗时随 tile 数量 N 线性增长

                    |
                    v

阶段 2: 文本解码 (Text Decode)
  [visual KV Cache] + text tokens
    | LLM 自回归生成
    v
  输出文本
  特点: 内存带宽瓶颈，与纯文本 LLM 相同
```

| 阶段     | 瓶颈     | 优化手段                                   |
| -------- | -------- | ------------------------------------------ |
| 图像编码 | 计算密集 | 减少 tile 数 / visual token 压缩 / ViT 量化 |
| 文本解码 | 内存带宽 | 量化 / KV Cache 复用 / Flash Attention      |

### 3.2 Visual Token 压缩

Pixel Unshuffle 是 ViT 内部的第一次压缩（4:1，固定）。进入 LLM 后，visual token 仍是序列长度的主要来源——多图或高分辨率场景下可达数千 tokens，因此有进一步压缩的需求：

```text
ViT 输出:  1024 tokens/tile x N tiles
         | Pixel Unshuffle (ViT 内部，固定 4:1)
         v
MLP 输入:   256 tokens/tile x N tiles  <- 第一次压缩
         | MLP Projector
         v
LLM 输入:   256xN visual tokens        <- N=40 时达 10240 tokens
         | Token Compression (可选，推理期叠加)
         v
压缩后:     更少 tokens -> 更快 LLM 推理
```

| 方法                    | 思路                                    | 压缩位置   | 精度损失 |
| ----------------------- | --------------------------------------- | ---------- | -------- |
| **Pixel Unshuffle**     | 空间重排，2x2 区域合并为 1 token（已有） | ViT 内部   | 极低     |
| **FastV**               | 第 K 层后丢弃注意力权重低的 visual token  | LLM 中间层 | 低       |
| **Token Merging (ToMe)** | 按 cosine 相似度合并相邻 token           | LLM 输入前 | 低       |
| **LLaVA-PruMerge**      | 注意力分数剪枝 + 邻近 token 合并         | LLM 输入前 | 中       |

### 3.3 KV Cache 复用

VLM 与纯文本 LLM 最大的 KV Cache 差异：**visual token 的 KV Cache 可跨请求复用**。

```text
场景：同一张图，问多个问题

请求 1: [V0..V255] + "这张图是什么地方？"
         | LLM prefill
         v  计算并缓存 visual tokens 的 KV Cache
        输出答案

请求 2: [V0..V255] + "图中有几个人？"
         | 命中 Prefix Cache
         v  直接复用 visual KV Cache  <- 节省约 70% prefill 时间
            只需计算新 text tokens 的 KV
        输出答案
```

工程实现：
- **vLLM**：开启 `prefix_caching=True`，相同 visual token 前缀自动命中 KV Cache
- **llama.cpp**：`--cache-type-k q8_0 --cache-type-v q8_0`，KV Cache 量化节省约 50% 显存

### 3.4 量化策略

VLM 的两个组件量化敏感度不同，应差异化处理：

| 组件                     | 量化敏感度 | 推荐精度      | 原因                     |
| ------------------------ | ---------- | ------------- | ------------------------ |
| **Vision Encoder (ViT)** | 高         | FP16 / INT8   | 图像特征提取对低比特敏感 |
| **MLP Projector**        | 中         | FP16 / INT8   | 参数量小，影响有限       |
| **LLM Backbone**         | 低         | INT4 / Q4_K_M | 参数量大，量化收益最高   |

在 llama.cpp 中，VLM 权重对应拆分为两个 GGUF 文件：

```text
model-Q4_K_M.gguf   <- LLM Backbone（量化，节省显存）
mmproj-f16.gguf     <- Vision Encoder + MLP Projector（FP16，保精度）
```

```bash
llama-server \
  -m model-Q4_K_M.gguf \       # LLM 主模型（量化）
  --mmproj mmproj-f16.gguf \   # 视觉编码器（FP16）
  -fa 1 \                      # Flash Attention
  --cache-type-k q8_0 \        # KV Cache 量化
  --cache-type-v q8_0
```

---

## 四、主流模型对比

### 4.1 架构横向对比

| 模型               | Vision Encoder       | Projector       | LLM Backbone      | 参数规格             |
| ------------------ | -------------------- | --------------- | ----------------- | -------------------- |
| **InternVL2**      | InternViT-300M / 6B  | MLP (2层)       | InternLM2 / Qwen2 | 2B / 8B / 26B / 76B  |
| **LLaVA-1.5**      | CLIP ViT-L/14        | MLP (2层)       | Vicuna / LLaMA2   | 7B / 13B             |
| **Qwen-VL**        | ViT（自研）          | Cross-Attention | Qwen              | 7B                   |
| **MiniCPM-V 2.6**  | SigLIP-400M          | LDM Resampler   | MiniCPM-3B        | 8B                   |
| **PaliGemma**      | SigLIP-So400m        | Linear          | Gemma-2B          | 3B                   |
| **Phi-3.5-Vision** | CLIP                 | MLP             | Phi-3.5-mini      | 4.2B                 |
| **SmolVLM**        | SigLIP-400M          | MLP             | SmolLM2           | 256M / 500M / 2B     |

### 4.2 Projector 设计对比

```text
MLP Projector  (InternVL2, LLaVA-1.5)
  visual tokens (N, d_v)
    | Linear -> GELU -> Linear
    v
  aligned tokens (N, d_llm)      <- token 数量不变
  优点: 简单高效
  缺点: 大量 visual token 直接进入 LLM，序列较长

------------------------------------------------------------

Cross-Attention  (Qwen-VL, Flamingo)
  text query
    | Cross-Attn (K=visual, V=visual)
    v
  aligned tokens                  <- LLM 主动查询视觉信息
  优点: 按需提取，语义对齐更灵活
  缺点: 计算开销大

------------------------------------------------------------

Q-Former  (BLIP-2)
  N 个可学习 query tokens
    | Cross-Attn with visual features
    v
  固定 32 个输出 tokens            <- 与输入图像大小无关
  优点: LLM 侧序列长度恒定
  缺点: 信息压缩过激，OCR/Grounding 等细粒度任务效果差

------------------------------------------------------------

Resampler  (MiniCPM-V)
  可学习 query + 注意力机制压缩     <- 介于 MLP 和 Q-Former 之间
  优点: 可灵活控制压缩比
```

### 4.3 轻量模型对比（端侧 / 边缘部署）

| 模型               | 参数量 | 推理显存  | 亮点                         |
| ------------------ | ------ | --------- | ---------------------------- |
| **SmolVLM-500M**   | 500M   | ~1.5 GB   | 极致轻量，适合嵌入式 / IoT    |
| **InternVL2-2B**   | 2B     | ~4.2 GB   | 综合性能最强 2B，动态高分辨率 |
| **MiniCPM-V 2.0**  | 2.4B   | ~5 GB     | 端侧优化，支持手机部署        |
| **PaliGemma-3B**   | 3B     | ~6 GB     | SigLIP 视觉强，Google 出品    |
| **Phi-3.5-Vision** | 4.2B   | ~8 GB     | 多帧 / 视频支持，微软出品     |

### 4.4 视觉编码器选型

| 编码器              | 预训练方式        | 典型分辨率     | 擅长任务                 |
| ---------------- | ------------ | --------- | -------------------- |
| **CLIP ViT-L**   | 图文对比学习       | 224 / 336 | 图文匹配、零样本分类           |
| **SigLIP**       | Sigmoid 对比学习 | 224~512   | 细粒度理解，综合优于 CLIP      |
| **InternViT-6B** | 渐进式多分辨率训练    | 448（动态）   | OCR、Grounding、高分辨率理解 |
| **DINOv2**       | 自监督（无文本监督）   | 224 / 518 | 密集预测、语义分割            |

InternViT 的核心优势：专门针对动态高分辨率输入训练，适合需要细粒度定位的任务（OCR、遥感目标检测）。
