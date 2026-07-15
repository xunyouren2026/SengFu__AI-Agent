---
id: email-summarizer
name: 邮件摘要器
description: 自动汇总邮件内容，提取关键信息和待办事项
version: "1.0.0"
author: AGI Framework Team
license: MIT
category: productivity
tags: [email, summary, inbox, productivity]
icon: ✉️
parameters:
  - name: mailbox
    type: string
    description: 邮箱类型
    required: false
    default: gmail
    options: [gmail, outlook, imap]
  - name: folder
    type: string
    description: 邮件文件夹
    required: false
    default: inbox
  - name: date_range
    type: string
    description: 日期范围
    required: false
    default: today
    options: [today, yesterday, week, month]
  - name: max_emails
    type: integer
    description: 最大处理邮件数
    required: false
    default: 50
    min_value: 1
    max_value: 500
  - name: extract_tasks
    type: boolean
    description: 提取待办事项
    required: false
    default: true
  - name: priority_filter
    type: string
    description: 优先级过滤
    required: false
    default: all
    options: [all, high, normal, low]
allowed_tools:
  - WebFetch
  - Write
execution:
  interpreter: python
  timeout: 300
---

# 邮件摘要器

自动汇总邮件内容，提取关键信息和待办事项。

## 功能特性

- 多邮箱支持（Gmail、Outlook、IMAP）
- 智能内容摘要
- 待办事项自动提取
- 优先级分类
- 发件人统计

## 使用方法

```bash
# 汇总今日邮件
agi skill run email-summarizer

# 汇总指定文件夹的邮件
agi skill run email-summarizer --folder=work --date_range=week

# 仅处理高优先级邮件
agi skill run email-summarizer --priority_filter=high --max_emails=20
```

## 输出格式

```markdown
# 邮件摘要报告
**时间范围:** 2024-01-15  
**处理邮件:** 25 封

## 重要邮件

### �� 来自: boss@company.com
**主题:** 项目进度汇报  
**摘要:** 要求本周五前提交项目进度报告...  
**待办:** 
- [ ] 准备项目进度报告 (截止: 周五)

---

## 按类别统计
- 工作相关: 15 封
- 订阅邮件: 8 封
- 社交邮件: 2 封

## 待办事项汇总
1. [高] 回复客户询问
2. [中] 审查 PR
3. [低] 更新文档
```

## 配置

```json
{
  "gmail": {
    "credentials_path": "~/.credentials/gmail.json"
  },
  "imap": {
    "server": "imap.example.com",
    "port": 993,
    "username": "user@example.com"
  },
  "summary_options": {
    "max_length": 200,
    "include_attachments": true
  }
}
```
