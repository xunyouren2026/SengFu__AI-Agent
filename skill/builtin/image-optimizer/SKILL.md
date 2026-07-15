---
id: image-optimizer
name: 图片优化器
description: 图片压缩和格式转换工具
version: "1.0.0"
author: AGI Framework Team
license: MIT
category: media
tags: [image, optimization, compression, conversion]
icon: ��️
parameters:
  - name: input_path
    type: file
    description: 输入文件或目录
    required: true
  - name: output_path
    type: file
    description: 输出文件或目录
    required: true
  - name: format
    type: string
    description: 输出格式
    required: false
    default: auto
    options: [auto, jpeg, png, webp, gif, avif]
  - name: quality
    type: integer
    description: 压缩质量（1-100）
    required: false
    default: 85
    min_value: 1
    max_value: 100
  - name: max_width
    type: integer
    description: 最大宽度
    required: false
  - name: max_height
    type: integer
    description: 最大高度
    required: false
  - name: preserve_aspect
    type: boolean
    description: 保持宽高比
    required: false
    default: true
  - name: strip_metadata
    type: boolean
    description: 移除元数据
    required: false
    default: false
allowed_tools:
  - Read
  - Write
  - Glob
execution:
  interpreter: python
  timeout: 300
---

# 图片优化器

图片压缩和格式转换工具。

## 功能特性

- 智能压缩
- 格式转换
- 批量处理
- 尺寸调整
- 元数据管理

## 支持的格式

### 输入
- JPEG/JPG
- PNG
- GIF
- WebP
- BMP
- TIFF
- RAW (CR2, NEF, ARW 等)

### 输出
- JPEG
- PNG
- WebP（推荐）
- GIF
- AVIF

## 压缩模式

### 有损压缩
- JPEG: 质量 60-95
- WebP: 质量 70-95

### 无损压缩
- PNG: 优化调色板
- WebP: 无损模式

## 使用方法

```bash
# 压缩单个文件
agi skill run image-optimizer --input_path=photo.jpg --output_path=photo_optimized.jpg --quality=80

# 转换为 WebP
agi skill run image-optimizer --input_path=photo.png --output_path=photo.webp --format=webp

# 批量处理
agi skill run image-optimizer --input_path=./images --output_path=./optimized --format=webp --quality=85

# 调整尺寸
agi skill run image-optimizer --input_path=photo.jpg --output_path=photo_small.jpg --max_width=800 --max_height=600
```

## 优化建议

| 用途 | 推荐格式 | 质量 |
|------|----------|------|
| 网页 | WebP | 85 |
| 照片 | JPEG | 90 |
| 图标 | PNG | 无损 |
| 透明图 | PNG/WebP | 无损 |
