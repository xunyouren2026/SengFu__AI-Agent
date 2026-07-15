---
id: security-scanner
name: 安全扫描器
description: 基础安全漏洞扫描工具
version: "1.0.0"
author: AGI Framework Team
license: MIT
category: security
tags: [security, vulnerability, scanning, audit]
icon: ��
parameters:
  - name: target
    type: string
    description: 扫描目标（URL 或路径）
    required: true
  - name: scan_type
    type: string
    description: 扫描类型
    required: false
    default: full
    options: [full, quick, owasp-top10, dependencies, secrets]
  - name: depth
    type: integer
    description: 扫描深度
    required: false
    default: 3
    min_value: 1
    max_value: 10
  - name: output_format
    type: string
    description: 输出格式
    required: false
    default: json
    options: [json, html, sarif]
  - name: severity_filter
    type: array
    description: 严重程度过滤
    required: false
    default: [high, critical]
allowed_tools:
  - Read
  - Write
  - Grep
  - Bash
execution:
  interpreter: python
  timeout: 600
---

# 安全扫描器

基础安全漏洞扫描工具。

## 功能特性

- OWASP Top 10 检测
- 依赖漏洞扫描
- 敏感信息泄露检测
- 配置安全检查
- 生成合规报告

## 扫描类型

### full
完整扫描，包含所有检查项。

### quick
快速扫描，仅检查高风险项。

### owasp-top10
针对 OWASP Top 10 的专项扫描。

### dependencies
依赖包漏洞扫描。

### secrets
敏感信息（密钥、密码等）泄露扫描。

## 检测项目

### 注入攻击
- SQL 注入
- NoSQL 注入
- 命令注入
- XSS

### 安全配置
- 不安全的 HTTP 头
- 敏感信息泄露
- 弱加密算法
- 默认凭证

### 依赖安全
- 已知 CVE 漏洞
- 过期依赖
- 许可证风险

## 使用方法

```bash
# 完整扫描
agi skill run security-scanner --target=https://example.com --scan_type=full

# 快速扫描
agi skill run security-scanner --target=./src --scan_type=quick

# 依赖漏洞扫描
agi skill run security-scanner --target=./package.json --scan_type=dependencies

# 敏感信息扫描
agi skill run security-scanner --target=./ --scan_type=secrets
```

## 输出报告

```json
{
  "scan_summary": {
    "target": "https://example.com",
    "scan_type": "full",
    "duration": 120,
    "vulnerabilities_found": 5
  },
  "vulnerabilities": [
    {
      "id": "CVE-2024-XXXX",
      "severity": "high",
      "category": "injection",
      "title": "SQL Injection",
      "description": "...",
      "remediation": "..."
    }
  ]
}
```

## 严重级别

- **Critical**: 立即修复
- **High**: 24小时内修复
- **Medium**: 一周内修复
- **Low**: 下次迭代修复
- **Info