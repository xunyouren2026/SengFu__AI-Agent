# UFO AGI 框架 - 快速开始指南

## 🚀 一键启动

### Windows
```batch
双击运行 start-all.bat
```

### Linux/Mac
```bash
chmod +x start-all.sh
./start-all.sh
```

---

## 📋 系统要求

- Python 3.8+
- Node.js 16+ (可选，用于React前端)
- 4GB+ 内存

---

## 🎯 核心功能

### 1. 智能对话 (Chat)
```python
# API调用
POST /api/v1/chat/conversations
{
  "message": "你好",
  "model": "agi-ultra"
}
```

### 2. 胜复学闭环 (Pendulum)
```python
# 执行胜复学闭环
POST /api/v1/algorithms/pendulum/cycle
{
  "observation": {"text": "当前状态"},
  "goal": "完成任务"
}
```

### 3. 多智能体协作
```python
# 创建联盟
POST /api/v1/multiagent/alliance/create
{
  "name": "开发团队",
  "description": "协作开发",
  "goals": ["代码审查", "功能开发"]
}
```

### 4. 训练算法
```python
# 启动RLHF训练
POST /api/v1/training-algo/rlhf/start
{
  "model_name": "gpt-2",
  "dataset_path": "/data/preferences.json"
}
```

---

## 🔌 外挂模型配置

### OpenAI
```env
OPENAI_API_KEY=sk-xxx
OPENAI_MODEL=gpt-4
```

### Claude
```env
ANTHROPIC_API_KEY=sk-xxx
ANTHROPIC_MODEL=claude-3-opus
```

### Ollama (本地)
```env
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=llama2
```

---

## 📚 API端点列表

| 模块 | 端点数 | 说明 |
|------|--------|------|
| `/algorithms` | 15+ | 胜复学核心算法 |
| `/multiagent` | 12+ | 多智能体协作 |
| `/training-algo` | 15+ | 训练算法 |
| `/chat` | 10+ | 对话系统 |
| `/models` | 8+ | 模型管理 |
| `/workflows` | 5+ | 工作流编排 |

**总端点: 380+**

---

## 🎨 前端页面

访问 http://localhost:3000 查看36个功能页面：

- Dashboard - 系统仪表盘
- Chat - 智能对话
- MultiAgent - 多智能体
- Training - 训练管理
- RAG - 知识库
- Settings - 系统设置

---

## 📖 详细文档

- API文档: http://localhost:18888/docs
- GitHub: https://github.com/ufo-agi
- 问题反馈: issues

---

## ⚡ 快速测试

```bash
# 测试后端
curl http://localhost:18888/api/v1/models

# 测试胜复学闭环
curl -X POST http://localhost:18888/api/v1/algorithms/swing/execute \
  -H "Content-Type: application/json" \
  -d '{"observation": "test", "training": true}'
```

---

## 🆘 常见问题

**Q: 启动失败？**
A: 检查Python版本和依赖安装

**Q: 数据库错误？**
A: 运行 `python scripts/init_db.py`

**Q: 前端无法访问？**
A: 检查端口3000是否被占用

---

## 📞 支持

- 文档: /docs
- 示例: /examples
- 社区: Discord
