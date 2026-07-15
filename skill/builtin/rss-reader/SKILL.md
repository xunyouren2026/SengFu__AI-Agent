---
id: rss-reader
name: RSS 阅读器
description: RSS 订阅源聚合和摘要生成工具
version: "1.0.0"
author: AGI Framework Team
license: MIT
category: productivity
tags: [rss, news, feed, summary]
icon: ��
parameters:
  - name: feeds
    type: array
    description: RSS 订阅源 URL 列表
    required: true
  - name: max_items
    type: integer
    description: 每个源最大条目数
    required: false
    default: 10
    min_value: 1
    max_value: 100
  - name: generate_summary
    type: boolean
    description: 是否生成 AI 摘要
    required: false
    default: true
  - name: output_format
    type: string
    description: 输出格式
    required: false
    default: markdown
    options: [markdown, html, json]
  - name: filter_keywords
    type: array
    description: 关键词过滤
    required: false
    default: []
allowed_tools:
  - WebFetch
  - Write
  - WebSearch
execution:
  interpreter: python
  timeout: 180
---

# RSS 阅读器

RSS 订阅源聚合和摘要生成工具。

## 功能特性

- 多源 RSS 聚合
- AI 智能摘要
- 关键词过滤
- 多种输出格式
- 定时更新

## 支持的格式

- RSS 2.0
- Atom
- JSON Feed

## 使用方法

```bash
# 读取单个 RSS 源
agi skill run rss-reader --feeds=["https://example.com/feed.xml"]

# 读取多个源并生成摘要
agi skill run rss-reader --feeds=["url1","url2"] --generate_summary=true

# 按关键词过滤
agi skill run rss-reader --feeds=["url"] --filter_keywords=["AI","machine learning"]
```

## 输出示例

```markdown
# RSS 摘要 - 2024-01-15

## 科技新闻

### Article Title
**来源:** Tech Blog  
**时间:** 2024-01-15 10:30  
**摘要:** 这是一篇关于最新技术发展的文章摘要...

[阅读原文](https://example.com/article)

---

## 行业动态
...
```

## 配置

```json
{
  "default_feeds": [
    "https://news.ycombinator.com/rss",
    "https://www.reddit.com/r/technology/.rss"
  ],
  "update_interval": 3600,
  "max_items_per_feed": 10,
  "summary_length": 200