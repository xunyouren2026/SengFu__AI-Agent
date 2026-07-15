---
id: crypto-tool
name: 加密工具
description: 加密/解密工具，支持多种算法
version: "1.0.0"
author: AGI Framework Team
license: MIT
category: security
tags: [encryption, decryption, cryptography, security]
icon: ��
parameters:
  - name: action
    type: string
    description: 操作类型
    required: true
    options: [encrypt, decrypt, generate-key, hash, sign, verify]
  - name: algorithm
    type: string
    description: 算法
    required: false
    default: aes-256-gcm
    options: [aes-256-gcm, aes-128-gcm, chacha20-poly1305, rsa-2048, rsa-4096]
  - name: input
    type: file
    description: 输入文件
    required: false
  - name: output
    type: file
    description: 输出文件
    required: false
  - name: key
    type: string
    description: 密钥（或密钥文件路径）
    required: false
  - name: password
    type: string
    description: 密码（用于基于密码的加密）
    required: false
  - name: input_format
    type: string
    description: 输入格式
    required: false
    default: auto
    options: [auto, text, base64, hex, binary]
  - name: output_format
    type: string
    description: 输出格式
    required: false
    default: base64
    options: [base64, hex, binary]
allowed_tools:
  - Read
  - Write
execution:
  interpreter: python
  timeout: 120
---

# 加密工具

加密/解密工具，支持多种算法。

## 功能特性

- 对称加密（AES、ChaCha20）
- 非对称加密（RSA）
- 哈希计算
- 数字签名
- 密钥生成

## 支持的算法

### 对称加密
- AES-256-GCM（推荐）
- AES-128-GCM
- ChaCha20-Poly1305

### 非对称加密
- RSA-2048
- RSA-4096

### 哈希算法
- SHA-256
- SHA-384
- SHA-512
- SHA3-256
- BLAKE2b

## 使用方法

### 加密文件

```bash
# 使用密码加密
agi skill run crypto-tool --action=encrypt --input=secret.txt --output=secret.enc --password=mypassword

# 使用密钥加密
agi skill run crypto-tool --action=encrypt --input=secret.txt --output=secret.enc --key=mykey
```

### 解密文件

```bash
agi skill run crypto-tool --action=decrypt --input=secret.enc --output=secret.txt --password=mypassword
```

### 生成密钥

```bash
agi skill run crypto-tool --action=generate-key --algorithm=aes-256-gcm --output=key.bin
```

### 计算哈希

```bash
agi skill run crypto-tool --action=hash --input=file.txt --algorithm=sha-256
```

### 数字签名

```bash
# 签名
agi skill run crypto-tool --action=sign --input=document.pdf --key=private.pem --output=signature.bin

# 验证
agi skill run crypto-tool --action=verify --input=document.pdf --key=public.pem --signature=signature.bin
```

## 安全建议

1. **密钥管理**: 使用安全的密钥管理系统
2. **密码强度**: 使用至少 12 位的强密码
3. **算法选择**: 优先使用 AES-256-GCM
4. **密钥分离**: 加密密钥和密文分开存储

## 输出格式

### 加密输出
```json
{
  "algorithm": "aes-256-gcm",
  "ciphertext": "base64encoded...",
  "nonce": "base64nonce...",
  "tag": "base64tag..."
}
```

### 哈希输出
```
SHA-256: a1b2c3d4e5f6...
```