---
id: git-assistant
name: Git 助手
description: Git 工作流辅助工具，提供提交建议、分支管理和冲突解决
version: "1.0.0"
author: AGI Framework Team
license: MIT
category: development
tags: [git, version-control, workflow, automation]
icon: ��
parameters:
  - name: action
    type: string
    description: 执行的操作
    required: true
    options: [suggest-commit, analyze-branch, resolve-conflict, generate-changelog, cleanup-branches]
  - name: repo_path
    type: directory
    description: 仓库路径
    required: false
    default: .
  - name: commit_style
    type: string
    description: 提交信息风格
    required: false
    default: conventional
    options: [conventional, simple, detailed]
  - name: target_branch
    type: string
    description: 目标分支
    required: false
    default: main
  - name: dry_run
    type: boolean
    description: 仅预览不执行
    required: false
    default: true
allowed_tools:
  - Read
  - Write
  - Bash
  - Grep
execution:
  interpreter: python
  timeout: 120
---

# Git 助手

Git 工作流辅助工具，提供提交建议、分支管理和冲突解决。

## 功能特性

- 智能提交信息生成
- 分支分析
- 冲突解决建议
- 变更日志生成
- 分支清理

## 操作类型

### suggest-commit
分析暂存区变更，生成提交信息建议。

```bash
agi skill run git-assistant --action=suggest-commit --commit_style=conventional
```

### analyze-branch
分析当前分支状态，提供合并建议。

```bash
agi skill run git-assistant --action=analyze-branch --target_branch=main
```

### resolve-conflict
分析冲突文件，提供解决建议。

```bash
agi skill run git-assistant --action=resolve-conflict
```

### generate-changelog
根据提交历史生成变更日志。

```bash
agi skill run git-assistant --action=generate-changelog
```

### cleanup-branches
清理已合并的本地和远程分支。

```bash
agi skill run git-assistant --action=cleanup-branches --dry_run=false
```

## 提交信息风格

### Conventional Commits
```
feat: add user authentication
fix: resolve login timeout issue
docs: update API documentation
```

### Simple
```
Add user authentication
Fix login timeout
Update documentation
```

### Detailed
```
Add user authentication module

- Implement JWT token generation
- Add password hashing
- Create login/logout endpoints
```

## 配置

```json
{
  "commit_style": "conventional",
  "auto_stage": false,
  "protected_branches": ["main", "master", "develop"],
  "cleanup": {
    "keep_recent": 10,
    "remote_prune": true
