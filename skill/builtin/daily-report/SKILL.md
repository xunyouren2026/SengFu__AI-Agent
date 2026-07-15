---
id: daily-report
name: 日报生成器
description: 自动生成每日工作报告，汇总任务完成情况和明日计划
version: "1.0.0"
author: AGI Framework Team
license: MIT
category: productivity
tags: [report, daily, productivity, work]
icon: ��
parameters:
  - name: date
    type: date
    description: 报告日期
    required: false
    default: today
  - name: data_source
    type: string
    description: 数据来源
    required: false
    default: auto
    options: [auto, git, todo, calendar, manual]
  - name: format
    type: string
    description: 输出格式
    required: false
    default: markdown
    options: [markdown, html, pdf, json]
  - name: include_metrics
    type: boolean
    description: 是否包含统计数据
    required: false
    default: true
  - name: recipient
    type: email
    description: 报告接收邮箱
    required: false
allowed_tools:
  - Read
  - Write
  - Bash
  - WebSearch
execution:
  interpreter: python
  timeout: 120
---

# 日报生成器

自动生成每日工作报告，汇总任务完成情况和明日计划。

## 功能特性

- 自动从多个数据源收集信息
- 智能任务分类和汇总
- 支持多种输出格式
- 邮件自动发送
- 自定义模板

## 数据来源

### Git
- 代码提交记录
- 分支合并情况
- Pull Request 状态

### Todo
- 完成任务列表
- 待办事项统计
- 优先级分析

### Calendar
- 会议记录
- 时间分配
- 日程完成情况

## 使用方法

```bash
# 生成今日报告
agi skill run daily-report

# 生成指定日期报告
agi skill run daily-report --date=2024-01-15

# 生成 PDF 格式并发送邮件
agi skill run daily-report --format=pdf --recipient=manager@company.com
```

## 报告模板

### Markdown 格式

```markdown
# 工作日报 - 2024年1月15日

## 今日完成
- [x] 任务 A - 已完成
- [x] 任务 B - 已完成

## 进行中
- [ ] 任务 C - 50%

## 明日计划
- [ ] 任务 D
- [ ] 任务 E

## 数据统计
- 完成任务: 5
- 代码提交: 12
- 会议时长: 2小时
```

## 配置

```json
{
  "data_sources": ["git", "todo", "calendar"],
  "git_repos": ["/path/to/repo1", "/path/to/repo2"],
  "output_format": "markdown",
  "auto_send": false,
  "template": "default"
