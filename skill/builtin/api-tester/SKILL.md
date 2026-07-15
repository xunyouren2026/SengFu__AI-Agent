---
id: api-tester
name: API 测试器
description: API 端点测试和性能分析工具
version: "1.0.0"
author: AGI Framework Team
license: MIT
category: development
tags: [api, testing, http, performance]
icon: ��
parameters:
  - name: spec_file
    type: file
    description: API 规范文件（OpenAPI/Swagger）
    required: false
  - name: base_url
    type: url
    description: API 基础 URL
    required: true
  - name: endpoints
    type: array
    description: 要测试的端点列表
    required: false
    default: []
  - name: method
    type: string
    description: HTTP 方法
    required: false
    default: GET
    options: [GET, POST, PUT, DELETE, PATCH, HEAD, OPTIONS]
  - name: headers
    type: object
    description: 请求头
    required: false
    default: {}
  - name: body
    type: string
    description: 请求体
    required: false
  - name: concurrent
    type: integer
    description: 并发请求数
    required: false
    default: 1
    min_value: 1
    max_value: 1000
  - name: requests
    type: integer
    description: 总请求数
    required: false
    default: 1
    min_value: 1
  - name: timeout
    type: integer
    description: 请求超时（秒）
    required: false
    default: 30
allowed_tools:
  - WebFetch
  - Write
execution:
  interpreter: python
  timeout: 300
---

# API 测试器

API 端点测试和性能分析工具。

## 功能特性

- 端点功能测试
- 负载测试
- 响应时间分析
- 并发测试
- OpenAPI 支持

## 测试类型

### 功能测试
验证 API 响应是否符合预期。

```bash
agi skill run api-tester --base_url=https://api.example.com --endpoints=["/users","/posts"]
```

### 负载测试
测试 API 在高负载下的表现。

```bash
agi skill run api-tester --base_url=https://api.example.com --endpoints=["/api/data"] \
  --concurrent=50 --requests=1000
```

### 基于 OpenAPI 的测试
从 OpenAPI 规范自动生成测试。

```bash
agi skill run api-tester --spec_file=api.yaml --base_url=https://api.example.com
```

## 输出报告

```json
{
  "summary": {
    "total_requests": 100,
    "successful": 98,
    "failed": 2,
    "avg_response_time": 150,
    "min_response_time": 50,
    "max_response_time": 500
  },
  "endpoints": [
    {
      "path": "/users",
      "method": "GET",
      "status_codes": {
        "200": 98,
        "500": 2
      },
      "response_times": {
        "avg": 150,
        "p50": 120,
        "p95": 400,
        "p99": 480
      }
    }
  ]
}
```

## 配置

```json
{
  "default_headers": {
    "Content-Type": "application/json",
    "Accept": "application/json"
  },
  "retry": {
    "enabled": true,
    "max_attempts": 3,
    "backoff": 1000
  },
  "assertions": {
    "max_response_time": 1000,
    "required_status_codes": [200, 201]
