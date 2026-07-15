---
id: doc-translator
name: 文档翻译器
description: 文档自动翻译工具，支持多种格式和语言
version: "1.0.0"
author: AGI Framework Team
license: MIT
category: productivity
tags: [translation, document, i18n, language]
icon: ��
parameters:
  - name: input_file
    type: file
    description: 输入文档路径
    required: true
  - name: output_file
    type: file
    description: 输出文档路径
    required: true
  - name: source_lang
    type: string
    description: 源语言
    required: false
    default: auto
  - name: target_lang
    type: string
    description: 目标语言
    required: true
  - name: preserve_formatting
    type: boolean
    description: 保留原始格式
    required: false
    default: true
  - name: translate_images
    type: boolean
    description: 翻译图片中的文字
    required: false
    default: false
  - name: glossary
    type: file
    description: 术语表文件
    required: false
allowed_tools:
  - Read
  - Write
  - WebSearch
execution:
  interpreter: python
  timeout: 600
---

# 文档翻译器

文档自动翻译工具，支持多种格式和语言。

## 功能特性

- 多格式支持（Markdown、Word、PDF、HTML）
- 100+ 语言支持
- 保留原始格式
- 术语表支持
- 批量翻译

## 支持的格式

- Markdown (.md)
- Microsoft Word (.docx, .doc)
- PDF (.pdf)
- HTML (.html, .htm)
- 纯文本 (.txt)
- reStructuredText (.rst)
- LaTeX (.tex)

## 支持的语言

- 中文（简体、繁体）
- 英语
- 日语
- 韩语
- 法语
- 德语
- 西班牙语
- 俄语
- 阿拉伯语
- 葡萄牙语
- 意大利语
- 荷兰语
- ... 等 100+ 种语言

## 使用方法

```bash
# 基本翻译
agi skill run doc-translator --input_file=doc.md --output_file=doc_en.md --target_lang=en

# 指定源语言
agi skill run doc-translator --input_file=doc.md --output_file=doc_ja.md --source_lang=zh --target_lang=ja

# 使用术语表
agi skill run doc-translator --input_file=doc.md --output_file=doc_en.md --target_lang=en --glossary=terms.json
```

## 术语表格式

```json
{
  "术语": "Term",
  "人工智能": "Artificial Intelligence",
  "机器学习": "Machine Learning"
}
```

## 输出格式

翻译后的文档保留原始格式，包括：
- 标题层级
- 列表和表格
- 代码块
- 链接和图片