# AGI Unified Framework - 开发者安装指南

## 系统要求

- Python 3.10+
- 操作系统: Linux, macOS, Windows
- 内存: 最低8GB，推荐16GB+
- 磁盘: 最低10GB可用空间

## 快速开始

### 1. 克隆仓库

```bash
git clone https://github.com/your-org/agi_unified_framework.git
cd agi_unified_framework
```

### 2. 创建虚拟环境

```bash
# 使用venv
python -m venv venv
source venv/bin/activate  # Linux/macOS
# 或
venv\Scripts\activate  # Windows

# 或使用conda
conda create -n agi python=3.10
conda activate agi
```

### 3. 安装依赖

```bash
pip install -r requirements.txt
```

### 4. 配置环境变量

创建 `.env` 文件：

```env
# LLM API配置
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key

# 云服务配置
AWS_ACCESS_KEY_ID=your_aws_key
AWS_SECRET_ACCESS_KEY=your_aws_secret
AZURE_CLIENT_ID=your_azure_client
AZURE_CLIENT_SECRET=your_azure_secret

# 安全配置
SECRET_KEY=your_secret_key
ENABLE_SANDBOX=true

# 日志配置
LOG_LEVEL=INFO
LOG_FORMAT=json
```

### 5. 验证安装

```bash
python -c "import agi_unified_framework; print('Installation successful!')"
```

## 详细安装选项

### 核心依赖

```bash
# 核心框架
pip install pydantic>=2.0
pip install python-dotenv
pip install pyyaml

# 异步支持
pip install asyncio
pip install aiohttp

# 日志和监控
pip install structlog
```

### LLM支持

```bash
# OpenAI
pip install openai>=1.0

# Anthropic
pip install anthropic

# 本地模型
pip install transformers
pip install torch
```

### 多模态支持

```bash
# 图像处理
pip install pillow
pip install opencv-python

# 音频处理
pip install librosa
pip install soundfile

# 视频处理
pip install ffmpeg-python
```

### 云服务支持

```bash
# AWS
pip install boto3

# Azure
pip install azure-storage-blob
pip install azure-identity
```

### 安全和沙箱

```bash
# Docker沙箱
pip install docker

# 加密
pip install cryptography
```

### 分布式训练

```bash
# 分布式
pip install ray
pip install deepspeed

# ML框架
pip install torch
pip install tensorflow
```

## 开发环境设置

### 安装开发依赖

```bash
pip install -r requirements-dev.txt
```

### 配置pre-commit

```bash
pip install pre-commit
pre-commit install
```

### 运行测试

```bash
# 运行所有测试
pytest

# 运行特定测试
pytest tests/test_core/

# 带覆盖率
pytest --cov=agi_unified_framework
```

### 代码风格

```bash
# 格式化
black agi_unified_framework/

# 类型检查
mypy agi_unified_framework/

# 代码检查
ruff check agi_unified_framework/
```

## Docker部署

### 构建镜像

```bash
docker build -t agi-framework:latest -f deployment/docker/Dockerfile .
```

### 运行容器

```bash
docker run -d \
  --name agi-framework \
  -p 8000:8000 \
  -v $(pwd)/config:/app/config \
  -e OPENAI_API_KEY=$OPENAI_API_KEY \
  agi-framework:latest
```

### Docker Compose

```bash
docker-compose -f deployment/docker/docker-compose.yml up -d
```

## Kubernetes部署

### 应用配置

```bash
kubectl apply -f deployment/kubernetes/configmap.yaml
kubectl apply -f deployment/kubernetes/secret.yaml
```

### 部署服务

```bash
kubectl apply -f deployment/kubernetes/deployment.yaml
kubectl apply -f deployment/kubernetes/service.yaml
kubectl apply -f deployment/kubernetes/ingress.yaml
```

## 配置说明

### 主配置文件 (config.yaml)

```yaml
# 核心配置
core:
  debug: false
  log_level: INFO
  
# LLM配置
llm:
  default_provider: openai
  cache_enabled: true
  max_retries: 3
  
# 安全配置
security:
  sandbox_enabled: true
  audit_enabled: true
  dlp_enabled: true
  
# 多智能体配置
multiagent:
  max_agents: 100
  coordination_timeout: 300
```

### 模块配置

每个模块可以有自己的配置文件：

```yaml
# config/llm.yaml
providers:
  openai:
    model: gpt-4
    temperature: 0.7
    max_tokens: 4096
    
  anthropic:
    model: claude-3-opus
    temperature: 0.7
```

## 故障排除

### 常见问题

1. **依赖冲突**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt --force-reinstall
   ```

2. **内存不足**
   - 增加系统内存
   - 减少批处理大小
   - 启用量化

3. **GPU不可用**
   ```bash
   # 检查CUDA
   python -c "import torch; print(torch.cuda.is_available())"
   
   # 安装正确的PyTorch版本
   pip install torch --index-url https://download.pytorch.org/whl/cu118
   ```

4. **权限问题**
   ```bash
   # Linux/macOS
   chmod +x scripts/*.py
   
   # 检查文件权限
   ls -la config/
   ```

### 日志查看

```bash
# 查看应用日志
tail -f logs/app.log

# 查看错误日志
grep ERROR logs/app.log
```

## 下一步

- 阅读[架构概览](../architecture/overview.md)
- 查看[API文档](../api/openapi.yaml)
- 运行示例代码 `examples/`

## 获取帮助

- GitHub Issues: https://github.com/your-org/agi_unified_framework/issues
- 文档: https://docs.agi-framework.org
- 社区: https://community.agi-framework.org
