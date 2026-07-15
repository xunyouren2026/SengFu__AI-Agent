---
id: barcode
name: 条形码生成器
description: 多种格式条形码生成和解析工具
version: "1.0.0"
author: AGI Framework Team
license: MIT
category: utility
tags: [barcode, generator, scanner]
icon: █▄
tags: [barcode, generator, scanner, utility]
parameters:
  - name: action
    type: string
    description: 操作类型
    required: true
    options: [generate, decode]
  - name: data
    type: string
    description: 要编码的数据
    required: false
  - name: barcode_type
    type: string
    description: 条形码类型
    required: false
    default: code128
    options: [code128, code39, ean13, ean8, upca, upce, isbn13, isbn10, qr]
  - name: input_file
    type: file
    description: 输入图片文件（用于解码）
    required: false
  - name: output_file
    type: file
    description: 输出文件路径
    required: false
  - name: width
    type: integer
    description: 图片宽度
    required: false
    default: 400
  - name: height
    type: integer
    description: 图片高度
    required: false
    default: 100
  - name: show_text
    type: boolean
    description: 显示文字
    required: false
    default: true
allowed_tools:
  - Read
  - Write
execution:
  interpreter: python
  timeout: 60
---

# 条形码生成器

多种格式条形码生成和解析工具。

## 功能特性

- 多种条形码格式支持
- 条形码解码
- 自定义尺寸
- 文字显示选项
- 批量生成

## 支持的条形码类型

| 类型 | 描述 | 数据长度 |
|------|------|----------|
| CODE128 | 高密度字母数字 | 可变 |
| CODE39 | 工业标准 | 可变 |
| EAN13 | 零售商品 | 13 位 |
| EAN8 | 小包装商品 | 8 位 |
| UPC-A | 北美零售 | 12 位 |
| UPC-E | 压缩 UPC | 8 位 |
| ISBN13 | 图书编码 | 13 位 |
| ISBN10 | 旧版图书 | 10 位 |

## 使用方法

### 生成条形码

```bash
# 生成 CODE128
agi skill run barcode --action=generate --data="ABC123" --barcode_type=code128 --output_file=barcode.png

# 生成 EAN13
agi skill run barcode --action=generate --data="1234567890123" --barcode_type=ean13

# 自定义尺寸
agi skill run barcode --action=generate --data="TEST" --width=600 --height=150

# 无文字
agi skill run barcode --action=generate --data="12345" --show_text=false
```

### 解码条形码

```bash
agi skill run barcode --action=decode --input_file=barcode.png
```

## 应用场景

### 库存管理
使用 CODE128 编码 SKU 编号。

### 零售
使用 EAN13/UPC-A 编码商品。

### 图书管理
使用 ISBN13 编码图书。

### 资产追踪
使用 CODE39 编码资产编号。

## 输出示例

```
条形码生成成功
==============

类型: CODE128
数据: ABC123456
尺寸: 400x100

保存至: barcode.png
```

## 注意事项

- EAN/UPC 类型需要特定长度的数据
- ISBN 需要有效的校验位
- 打印时确保足够的分辨率以便扫描
