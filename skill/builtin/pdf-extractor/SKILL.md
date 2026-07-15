---
id: pdf-extractor
name: PDF 提取器
description: PDF 文本、图片和元数据提取工具
version: "1.0.0"
author: AGI Framework Team
license: MIT
category: data
tags: [pdf, extraction, text, image, document]
icon: ��
parameters:
  - name: input_file
    type: file
    description: 输入 PDF 文件
    required: true
  - name: output_dir
    type: directory
    description: 输出目录
    required: true
  - name: extract_text
    type: boolean
    description: 提取文本
    required: false
    default: true
  - name: extract_images
    type: boolean
    description: 提取图片
    required: false
    default: false
  - name: extract_tables
    type: boolean
    description: 提取表格
    required: false
    default: false
  - name: extract_metadata
    type: boolean
    description: 提取元数据
    required: false
    default: true
  - name: ocr
    type: boolean
    description: 对扫描件进行 OCR
    required: false
    default: false
  - name: page_range
    type: string
    description: 页码范围（如 "1-10" 或 "1,3,5"）
    required: false
allowed_tools:
  - Read
  - Write
execution:
  interpreter: python
  timeout: 300
---

# PDF 提取器

PDF 文本、图片和元数据提取工具。

## 功能特性

- 文本提取（保留布局）
- 图片提取
- 表格提取
- 元数据提取
- OCR 支持（扫描件）
- 批量处理

## 提取内容

### 文本
- 纯文本
- 格式化文本
- 带位置信息的文本
- 书签和链接

### 图片
- 提取为 PNG、JPEG
- 保留原始分辨率
- 支持透明背景

### 表格
- 提取为 CSV
- 提取为 Excel
- 保留表格结构

### 元数据
- 标题、作者
- 创建日期、修改日期
- 页数、页面大小
- PDF 版本

## 使用方法

```bash
# 提取文本
agi skill run pdf-extractor --input_file=document.pdf --output_dir=./output

# 提取所有内容
agi skill run pdf-extractor --input_file=document.pdf --output_dir=./output \
  --extract_text=true --extract_images=true --extract_tables=true

# OCR 扫描件
agi skill run pdf-extractor --input_file=scan.pdf --output_dir=./output --ocr=true

# 指定页码范围
agi skill run pdf-extractor --input_file=document.pdf --output_dir=./output --page_range=1-10
```

## 输出结构

```
output/
├── text.txt          # 提取的文本
├── metadata.json     # 元数据
├── images/           # 提取的图片
│   ├── page_1_img_1.png
│   └── page_2_img_1.png
└── tables/           # 提取的表格
    ├── page_3_table_1.csv
    └── page_5_table_1.csv
```