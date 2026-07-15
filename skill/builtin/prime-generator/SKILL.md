---
id: prime-generator
name: 安全素数生成器
description: 密码学安全的大素数生成工具
version: "1.0.0"
author: AGI Framework Team
license: MIT
category: security
tags: [prime, cryptography, security, math]
icon: ��
parameters:
  - name: bits
    type: integer
    description: 素数位数
    required: false
    default: 2048
    min_value: 64
    max_value: 8192
  - name: count
    type: integer
    description: 生成数量
    required: false
    default: 1
    min_value: 1
    max_value: 100
  - name: safe_prime
    type: boolean
    description: 生成安全素数
    required: false
    default: true
  - name: output_format
    type: string
    description: 输出格式
    required: false
    default: hex
    options: [hex, decimal, binary, base64]
  - name: save_to_file
    type: file
    description: 保存到文件
    required: false
allowed_tools:
  - Write
execution:
  interpreter: python
  timeout: 600
---

# 安全素数生成器

密码学安全的大素数生成工具。

## 功能特性

- 大素数生成（最高 8192 位）
- 安全素数支持
- 多种输出格式
- Miller-Rabin 素性测试
- 高性能实现

## 什么是安全素数

安全素数是指形如 p = 2q + 1 的素数，其中 q 也是素数。
安全素数在密码学中非常重要，因为它们可以抵抗某些攻击。

## 算法说明

### 素数生成
1. 随机生成候选数
2. 进行初步筛选（小素数试除）
3. Miller-Rabin 素性测试
4. 验证为安全素数（如需要）

### Miller-Rabin 测试
- 确定性版本：对于 64 位整数
- 概率性版本：对于大数，错误率 < 2^-128

## 使用方法

```bash
# 生成 2048 位安全素数
agi skill run prime-generator --bits=2048 --safe_prime=true

# 生成多个素数
agi skill run prime-generator --bits=1024 --count=10 --output_format=hex

# 保存到文件
agi skill run prime-generator --bits=4096 --save_to_file=primes.txt
```

## 输出示例

```
素数生成结果
============

位数: 2048
安全素数: 是
生成数量: 1

素数 #1:
Hex: ffffffffffffffffc90fdaa22168c234c4c6628b80dc1cd129024e088a67cc74020bbea63b139b22514a08798e3404ddef9519b3cd3a431b302b0a6df25f14374fe1356d6d51c245e485b576625e7ec6f44c42e9a637ed6b0bff5cb6f406b7edee386bfb5a899fa5ae9f24117c4b1fe649286651ece45b3dc2007cb8a163bf0598da48361c55d39a69163fa8fd24cf5f83655d23dca3ad961c62f356208552bb9ed529077096966d670c354e4abc9804f1746c08ca18217c32905e462e36ce3be39e772c180e86039b2783a2ec07a28fb5c55df06f4c52c9de2bcbf6955817183995497cea956ae515d2261898fa051015728e5a8aacaa68ffffffffffffffff
Decimal: 323170060713110073007148766886...（省略）

验证:
- 通过 Miller-Rabin 测试
- 是安全素数
- (p-1)/2 也是素数
```

## 密码学应用

- RSA 密钥生成
- Diffie-Hellman 密钥交换
- DSA 签名
- ElGamal 加密

## 性能参考

| 位数 | 平均生成时间 |
|------|-------------|
| 512 | < 1s |
| 1024 | 1-5s |
| 2048 | 10-60s |
| 4096 | 5