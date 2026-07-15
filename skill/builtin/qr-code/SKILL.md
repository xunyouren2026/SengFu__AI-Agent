---
id: qr-code
name: QR 码生成器
description: QR 码生成和解析工具
version: "1.0.0"
author: AGI Framework Team
license: MIT
category: utility
tags: [qr-code, barcode, generator, scanner]
icon: ▣
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
  - name: input_file
    type: file
    description: 输入图片文件（用于解码）
    required: false
  - name: output_file
    type: file
    description: 输出文件路径
    required: false
  - name: size
    type: integer
    description: 图片尺寸（像素）
    required: false
    default: 300
    min_value: 100
    max_value: 2000
  - name: error_correction
    type: string
    description: 纠错级别
    required: false
    default: M
    options: [L, M, Q, H]
  - name: fill_color
    type: string
    description: 前景色
    required: false
    default: black
  - name: back_color
    type: string
    description: 背景色
    required: false
    default: white
  - name: logo
    type: file
    description: 中心 Logo 图片
    required: false
allowed_tools:
  - Read
  - Write
execution:
  interpreter: python
  timeout: 60
---

# QR 码生成器

QR 码生成和解析工具。

## 功能特性

- QR 码生成
- QR 码解码
- 自定义样式
- Logo 嵌入
- 批量生成

## 纠错级别

| 级别 | 容错率 | 适用场景 |
|------|--------|----------|
| L | 7% | 清洁环境 |
| M | 15% | 一般场景 |
| Q | 25% | 较脏环境 |
| H | 30% | 严重污损 |

## 支持的数据类型

- 纯文本
- URL
- 邮箱地址
- 电话号码
- WiFi 配置
- vCard 联系人
- 地理位置
- 日历事件

## 使用方法

### 生成 QR 码

```bash
# 基本生成
agi skill run qr-code --action=generate --data="Hello World" --output_file=qr.png

# 生成 URL QR 码
agi skill run qr-code --action=generate --data="https://example.com" --output_file=url_qr.png

# 自定义样式
agi skill run qr-code --action=generate --data="Hello" --size=500 \
  --fill_color=blue --back_color=white

# 带 Logo
agi skill run qr-code --action=generate --data="Brand" --logo=logo.png
```

### 解码 QR 码

```bash
agi skill run qr-code --action=decode --input_file=qr.png
```

## WiFi QR 码格式

```
WIFI:T:WPA;S:NetworkName;P:Password;H:false;;
```

## vCard 格式

```
BEGIN:VCARD
VERSION:3.0
FN:John Doe
TEL:+1234567890
EMAIL:john@example.com
END:VCARD
```

## 输出示例

```
QR 码生成成功
=============

数据: https://example.com
尺寸: 300x300
纠错级别: M (15%)
版本: 4

保存至: qr.png
```
