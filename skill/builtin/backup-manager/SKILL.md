---
id: backup-manager
name: 备份管理器
description: 文件备份自动化工具
version: "1.0.0"
author: AGI Framework Team
license: MIT
category: system
tags: [backup, automation, archive, restore]
icon: ��
parameters:
  - name: action
    type: string
    description: 操作类型
    required: true
    options: [backup, restore, list, cleanup]
  - name: source
    type: file
    description: 备份源路径
    required: false
  - name: destination
    type: directory
    description: 备份目标路径
    required: false
  - name: backup_name
    type: string
    description: 备份名称
    required: false
  - name: compression
    type: string
    description: 压缩格式
    required: false
    default: zip
    options: [zip, tar, tar.gz, 7z, none]
  - name: exclude_patterns
    type: array
    description: 排除模式
    required: false
    default: ["node_modules", ".git", "__pycache__", ".env"]
  - name: encrypt
    type: boolean
    description: 加密备份
    required: false
    default: false
  - name: retention_days
    type: integer
    description: 保留天数
    required: false
    default: 30
allowed_tools:
  - Read
  - Write
  - Bash
  - Glob
execution:
  interpreter: python
  timeout: 600
---

# 备份管理器

文件备份自动化工具。

## 功能特性

- 增量/全量备份
- 压缩和加密
- 自动清理
- 定时任务
- 恢复功能

## 操作类型

### backup
创建新备份。

```bash
agi skill run backup-manager --action=backup --source=./project --destination=./backups --backup_name=project_backup
```

### restore
从备份恢复。

```bash
agi skill run backup-manager --action=restore --source=./backups/project_backup_20240115.zip --destination=./restored
```

### list
列出所有备份。

```bash
agi skill run backup-manager --action=list --destination=./backups
```

### cleanup
清理过期备份。

```bash
agi skill run backup-manager --action=cleanup --destination=./backups --retention_days=30
```

## 备份策略

### 全量备份
备份所有文件。

### 增量备份
仅备份自上次备份后修改的文件。

### 差异备份
备份自上次全量备份后修改的文件。

## 压缩选项

| 格式 | 压缩率 | 速度 | 兼容性 |
|------|--------|------|--------|
| zip | 中 | 快 | 高 |
| tar.gz | 高 | 中 | 高 |
| 7z | 很高 | 慢 | 中 |

## 使用方法

```bash
# 创建加密备份
agi skill run backup-manager --action=backup --source=./data --destination=./backups \
  --backup_name=data_backup --encrypt=true

# 排除特定文件
agi skill run backup-manager --action=backup --source=./project --destination=./backups \
  --exclude_patterns=["*.log","*.tmp",".git"]

# 恢复特定备份
agi skill run backup-manager --action=restore --source=./backups/data_backup_20240115.zip \
  --destination=./restored
```

## 配置

```json
{
  "default_destination": "~/backups",
  "compression": "zip",
  "encryption": {
    "enabled": false,
    "algorithm": "AES-256"
  },
  "schedule": {
    "enabled": true,
    "cron": "0 2 * * *"
  },
  "retention": {
    "daily": 7,
    "weekly": 4,
    "monthly": 12
  }
}
```
