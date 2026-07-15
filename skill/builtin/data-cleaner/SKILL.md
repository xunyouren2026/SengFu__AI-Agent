---
id: data-cleaner
name: 数据清洗工具
description: CSV/Excel 数据清洗、转换和验证工具
version: "1.0.0"
author: AGI Framework Team
license: MIT
category: data
tags: [data, csv, excel, cleaning, etl]
icon: ��
parameters:
  - name: input_file
    type: file
    description: 输入文件路径
    required: true
  - name: output_file
    type: file
    description: 输出文件路径
    required: true
  - name: operations
    type: array
    description: 清洗操作列表
    required: false
    default: [remove_duplicates, trim_whitespace]
  - name: column_types
    type: object
    description: 列类型定义
    required: false
    default: {}
  - name: missing_strategy
    type: string
    description: 缺失值处理策略
    required: false
    default: ignore
    options: [ignore, drop, fill_mean, fill_median, fill_mode, fill_value]
  - name: fill_value
    type: string
    description: 填充值（当 missing_strategy=fill_value 时）
    required: false
  - name: validate_schema
    type: boolean
    description: 验证数据模式
    required: false
    default: false
allowed_tools:
  - Read
  - Write
execution:
  interpreter: python
  timeout: 300
---

# 数据清洗工具

CSV/Excel 数据清洗、转换和验证工具。

## 功能特性

- 重复数据删除
- 缺失值处理
- 数据类型转换
- 异常值检测
- 格式标准化
- 数据验证

## 支持的格式

- CSV
- Excel (.xlsx, .xls)
- JSON
- Parquet

## 清洗操作

### 基本操作
- `remove_duplicates` - 删除重复行
- `trim_whitespace` - 去除空白字符
- `normalize_case` - 统一大小写
- `remove_empty_rows` - 删除空行
- `remove_empty_columns` - 删除空列

### 高级操作
- `standardize_dates` - 标准化日期格式
- `normalize_phone` - 标准化电话号码
- `clean_email` - 清洗邮箱地址
- `remove_outliers` - 移除异常值
- `encode_categories` - 编码分类变量

## 使用方法

```bash
# 基本清洗
agi skill run data-cleaner --input_file=data.csv --output_file=cleaned.csv

# 指定列类型
agi skill run data-cleaner --input_file=data.csv --output_file=cleaned.csv \
  --column_types='{"age": "integer", "price": "float", "date": "datetime"}'

# 处理缺失值
agi skill run data-cleaner --input_file=data.csv --output_file=cleaned.csv \
  --missing_strategy=fill_mean
```

## 配置示例

```json
{
  "operations": [
    "remove_duplicates",
    "trim_whitespace",
    "standardize_dates"
  ],
  "column_rules": {
    "email": {
      "type": "email",
      "required": true
    },
    "age": {
      "type": "integer",
      "min": 0,
      "max": 150
    }
  }
}
```

## 输出报告

```json
{
  "input_rows": 1000,
  "output_rows": 985,
  "removed_duplicates": 10,
  "filled_missing": 25,
  "errors": []
}
```