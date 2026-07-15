---
id: dependency-checker
name: 依赖检查器
description: 项目依赖检查和更新建议工具
version: "1.0.0"
author: AGI Framework Team
license: MIT
category: development
tags: [dependency, package, update, security]
icon: ��
parameters:
  - name: project_path
    type: directory
    description: 项目路径
    required: true
  - name: package_manager
    type: string
    description: 包管理器
    required: false
    default: auto
    options: [auto, npm, yarn, pip, poetry, maven, gradle, go, cargo]
  - name: check_outdated
    type: boolean
    description: 检查过期依赖
    required: false
    default: true
  - name: check_vulnerabilities
    type: boolean
    description: 检查安全漏洞
    required: false
    default: true
  - name: check_licenses
    type: boolean
    description: 检查许可证兼容性
    required: false
    default: false
  - name: update_recommendations
    type: boolean
    description: 提供更新建议
    required: false
    default: true
allowed_tools:
  - Read
  - Write
  - Bash
execution:
  interpreter: python
  timeout: 300
---

# 依赖检查器

项目依赖检查和更新建议工具。

## 功能特性

- 多包管理器支持
- 过期依赖检测
- 安全漏洞扫描
- 许可证检查
- 更新建议生成

## 支持的包管理器

- **JavaScript**: npm, yarn, pnpm
- **Python**: pip, poetry, pipenv
- **Java**: Maven, Gradle
- **Go**: Go Modules
- **Rust**: Cargo

## 检查内容

### 过期依赖
- 当前版本
- 最新版本
- 更新类型（major/minor/patch）
- 更新风险评级

### 安全漏洞
- 已知 CVE
- 严重级别
- 修复版本

### 许可证
- 依赖许可证
- 兼容性检查
- 合规性报告

## 使用方法

```bash
# 检查项目依赖
agi skill run dependency-checker --project_path=./my-project

# 仅检查安全漏洞
agi skill run dependency-checker --project_path=./my-project \
  --check_outdated=false --check_vulnerabilities=true

# 包含许可证检查
agi skill run dependency-checker --project_path=./my-project --check_licenses=true
```

## 输出报告

```markdown
# 依赖检查报告

## 摘要
- 总依赖数: 150
- 过期依赖: 20
- 安全漏洞: 3
- 许可证问题: 0

## 过期依赖

### Major 更新（可能有破坏性变更）
| 包名 | 当前版本 | 最新版本 | 风险 |
|------|----------|----------|------|
| react | 17.0.0 | 18.2.0 | 中 |

### Minor/Patch 更新（安全）
| 包名 | 当前版本 | 最新版本 | 风险 |
|------|----------|----------|------|
| lodash | 4.17.20 | 4.17.21 | 低 |

## 安全漏洞

### High Severity
1. **lodash** < 4.17.21
   - CVE-2021-23337
   - 建议更新到 4.17.21

## 建议操作

```bash
# 安全更新
npm update

# 需要审查的更新
npm install react@18.2.0
```
```