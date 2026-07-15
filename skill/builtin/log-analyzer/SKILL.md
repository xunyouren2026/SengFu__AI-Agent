---
id: log-analyzer
name: 日志分析器
description: 日志文件分析和异常检测工具
version: "1.0.0"
author: AGI Framework Team
license: MIT
category: development
tags: [log, analysis, monitoring, debugging]
icon: ��
parameters:
  - name: log_file
    type: file
    description: 日志文件路径
    required: true
  - name: log_format
    type: string
    description: 日志格式
    required: false
    default: auto
    options: [auto, apache, nginx, syslog, json, csv, custom]
  - name: time_range
    type: string
    description: 时间范围
    required: false
  - name: level_filter
    type: array
    description: 日志级别过滤
    required: false
    default: [ERROR, WARN]
  - name: pattern
    type: string
    description: 搜索模式（正则表达式）
    required: false
  - name: top_n
    type: integer
    description: 显示前 N 个结果
    required: false
    default: 20
  - name: output_format
    type: string
    description: 输出格式
    required: false
    default: text
    options: [text, json, html]
allowed_tools:
  - Read
  - Write
  - Grep
execution:
  interpreter: python
  timeout: 300
---

# 日志分析器

日志文件分析和异常检测工具。

## 功能特性

- 多格式日志解析
- 异常检测
- 趋势分析
- 性能指标提取
- 可视化报告

## 支持的日志格式

- Apache/Nginx 访问日志
- Syslog
- JSON 结构化日志
- CSV 日志
- 自定义格式

## 分析维度

### 错误分析
- 错误率统计
- 错误类型分布
- 错误趋势

### 性能分析
- 响应时间分布
- 吞吐量统计
- 慢请求识别

### 流量分析
- 请求量趋势
- 峰值识别
- 来源分布

## 使用方法

```bash
# 分析错误日志
agi skill run log-analyzer --log_file=error.log --level_filter=[ERROR]

# 分析访问日志
agi skill run log-analyzer --log_file=access.log --log_format=nginx

# 搜索特定模式
agi skill run log-analyzer --log_file=app.log --pattern="timeout"

# 指定时间范围
agi skill run log-analyzer --log_file=app.log --time_range="2024-01-01 to 2024-01-31"
```

## 输出示例

```
日志分析报告
============

时间范围: 2024-01-15 00:00 - 23:59
总行数: 100,000

错误统计:
- ERROR: 150 (0.15%)
- WARN: 500 (0.5%)
- INFO: 99,350 (99.35%)

Top 10 错误:
1. Connection timeout: 50
2. Database error: 45
3. NullPointerException: 30
...

性能指标:
- 平均响应时间: 120ms
- P95 响应时间: 500ms
- P99 响应时间: 1000ms
```