---
id: meeting-notes
name: 会议记录助手
description: 会议录音转录和智能摘要生成
version: "1.0.0"
author: AGI Framework Team
license: MIT
category: productivity
tags: [meeting, transcription, notes, summary]
icon: ��️
parameters:
  - name: audio_file
    type: file
    description: 会议录音文件
    required: false
  - name: transcript
    type: file
    description: 转录文本文件
    required: false
  - name: meeting_type
    type: string
    description: 会议类型
    required: false
    default: general
    options: [general, standup, review, planning, retrospective]
  - name: participants
    type: array
    description: 参会人员列表
    required: false
    default: []
  - name: extract_action_items
    type: boolean
    description: 提取行动项
    required: false
    default: true
  - name: output_format
    type: string
    description: 输出格式
    required: false
    default: markdown
    options: [markdown, html, docx]
allowed_tools:
  - Read
  - Write
execution:
  interpreter: python
  timeout: 600
---

# 会议记录助手

会议录音转录和智能摘要生成。

## 功能特性

- 音频转录（支持多种格式）
- 说话人识别
- 智能摘要生成
- 行动项自动提取
- 多格式输出

## 支持的音频格式

- MP3
- WAV
- M4A
- FLAC
- OGG

## 使用方法

```bash
# 从音频文件生成会议记录
agi skill run meeting-notes --audio_file=meeting.mp3

# 从转录文本生成
agi skill run meeting-notes --transcript=transcript.txt

# 指定会议类型和参会人员
agi skill run meeting-notes --audio_file=daily.mp3 --meeting_type=standup --participants=["Alice","Bob","Charlie"]
```

## 输出格式

```markdown
# 会议记录

**日期:** 2024-01-15  
**时间:** 14:00 - 15:30  
**类型:** 周会  
**参会人员:** Alice, Bob, Charlie

## 议程
1. 上周回顾
2. 本周计划
3. 问题讨论

## 讨论要点

### 上周回顾
- Alice 完成了用户认证模块
- Bob 修复了 5 个 bug
- Charlie 更新了文档

### 本周计划
- 启动支付模块开发
- 进行性能优化
- 准备发布文档

## 行动项

| 任务 | 负责人 | 截止日期 |
|------|--------|----------|
| 完成 API 设计 | Alice | 2024-01-17 |
| 编写测试用例 | Bob | 2024-01-18 |
| 更新用户手册 | Charlie | 2024-01-19 |

## 决定事项
1. 采用新的缓存策略
2. 推迟低优先级功能到下个版本
```

## 配置

```json
{
  "transcription": {
    "engine": "whisper",
    "language": "zh",
    "speaker_detection": true
  },
  "summary": {
    "max_length": 500,
    "include_timestamps": true
  }
