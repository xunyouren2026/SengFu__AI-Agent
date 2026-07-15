---
id: code-review
name: 代码审查助手
description: 自动进行代码审查，检测潜在问题并提供改进建议
version: "1.0.0"
author: AGI Framework Team
license: MIT
category: development
tags: [code-review, quality, analysis]
icon: ��
parameters:
  - name: file_path
    type: file
    description: 要审查的代码文件路径
    required: true
  - name: language
    type: string
    description: 编程语言
    required: false
    default: auto
    options: [auto, python, javascript, typescript, java, go, rust, cpp, c]
  - name: strictness
    type: string
    description: 审查严格程度
    required: false
    default: normal
    options: [lenient, normal, strict]
  - name: focus_areas
    type: array
    description: 重点关注领域
    required: false
    default: [security, performance, readability]
allowed_tools:
  - Read
  - Write
  - Grep
execution:
  interpreter: python
  timeout: 300
---

# 代码审查助手

自动进行代码审查，检测潜在问题并提供改进建议。

## 功能特性

- 多语言支持（Python、JavaScript、TypeScript、Java、Go、Rust、C/C++）
- 安全性检查
- 性能优化建议
- 代码可读性分析
- 最佳实践检测

## 使用方法

```bash
# 审查单个文件
agi skill run code-review --file_path=src/main.py

# 指定语言和严格程度
agi skill run code-review --file_path=src/app.js --language=javascript --strictness=strict

# 关注特定领域
agi skill run code-review --file_path=src/api.py --focus_areas=[security,performance]
```

## 审查维度

### 安全性
- SQL 注入检测
- XSS 漏洞检测
- 敏感信息泄露
- 不安全的函数调用

### 性能
- 算法复杂度分析
- 内存使用优化
- I/O 操作优化
- 循环优化建议

### 可读性
- 命名规范检查
- 代码复杂度分析
- 注释覆盖率
- 函数长度检查

## 输出格式

```json
{
  "summary": {
    "total_issues": 10,
    "critical": 2,
    "warning": 5,
    "info": 3
  },
  "issues": [
    {
      "line": 42,
      "severity": "critical",
      "category": "security",
      "message": "检测到 SQL 注入风险",
      "suggestion": "使用