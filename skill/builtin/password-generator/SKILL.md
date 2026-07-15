---
id: password-generator
name: 密码生成器
description: 安全密码生成工具
version: "1.0.0"
author: AGI Framework Team
license: MIT
category: security
tags: [password, security, generator]
icon: ��
parameters:
  - name: length
    type: integer
    description: 密码长度
    required: false
    default: 16
    min_value: 8
    max_value: 128
  - name: count
    type: integer
    description: 生成数量
    required: false
    default: 1
    min_value: 1
    max_value: 100
  - name: include_uppercase
    type: boolean
    description: 包含大写字母
    required: false
    default: true
  - name: include_lowercase
    type: boolean
    description: 包含小写字母
    required: false
    default: true
  - name: include_numbers
    type: boolean
    description: 包含数字
    required: false
    default: true
  - name: include_symbols
    type: boolean
    description: 包含特殊字符
    required: false
    default: true
  - name: exclude_ambiguous
    type: boolean
    description: 排除易混淆字符
    required: false
    default: true
  - name: passphrase_mode
    type: boolean
    description: 使用密码短语模式
    required: false
    default: false
  - name: word_count
    type: integer
    description: 密码短语单词数
    required: false
    default: 4
allowed_tools:
  - Write
execution:
  interpreter: python
  timeout: 60
---

# 密码生成器

安全密码生成工具。

## 功能特性

- 强密码生成
- 密码短语模式
- 可定制字符集
- 密码强度评估
- 批量生成

## 密码模式

### 随机密码
使用随机字符组合生成高强度密码。

### 密码短语
使用多个随机单词组合，易于记忆且安全。
例如：`correct-horse-battery-staple`

## 字符选项

- 大写字母 (A-Z)
- 小写字母 (a-z)
- 数字 (0-9)
- 特殊符号 (!@#$%^&*)
- 排除易混淆字符 (0, O, l, 1, I)

## 使用方法

```bash
# 生成默认密码
agi skill run password-generator

# 生成长密码
agi skill run password-generator --length=32

# 生成密码短语
agi skill run password-generator --passphrase_mode=true --word_count=5

# 仅包含字母和数字
agi skill run password-generator --include_symbols=false

# 批量生成
agi skill run password-generator --count=10
```

## 输出示例

```
生成的密码
==========

1. k9#mP2$vL8@nQ5*w
   强度: 非常强
   熵值: 95.4 bits

2. correct-horse-battery-staple
   强度: 强
   熵值: 88.0 bits
   估计破解时间: > 1000 年
```

## 密码强度标准

| 强度 | 熵值 | 建议用途 |
|------|------|----------|
| 弱 | < 50 bits | 不推荐 |
| 中等 | 50-75 bits | 临时密码 |
| 强 | 75-100 bits | 一般账户 |
| 非常强 | > 100 bits | 重要账户 |

## 安全建议

1. 使用至少 12 位密码
2. 每个账户使用不同密码
3. 使用密码管理器
4. 定期更换重要账户密码
5. 启用双因素