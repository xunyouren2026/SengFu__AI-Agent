# 🚀 UFO AGI 统一框架 - 安装指南

## 📦 快速开始（推荐）

### 方式一：一键启动脚本（最简单）

#### Linux / macOS
```bash
# 1. 进入项目目录
cd agi_framework/UFO

# 2. 运行启动脚本（首次会自动安装）
./start.sh

# 3. 访问应用
# Web界面: http://localhost:8000
# API文档: http://localhost:8000/docs
```

#### Windows
```cmd
:: 1. 进入项目目录
cd agi_framework\UFO

:: 2. 运行启动脚本（首次会自动安装）
start.bat

:: 3. 访问应用
:: Web界面: http://localhost:8000
:: API文档: http://localhost:8000/docs
```

**启动脚本功能：**
- ✅ 自动检查Python环境
- ✅ 自动创建虚拟环境
- ✅ 自动安装依赖
- ✅ 自动初始化数据库
- ✅ 自动创建配置文件
- ✅ 后台运行，带日志输出

**常用命令：**
```bash
./start.sh           # 启动应用
./start.sh --setup   # 重新安装配置
./start.sh --stop    # 停止应用
./start.sh --status  # 查看状态
./start.sh --logs    # 查看实时日志
```

---

### 方式二：Docker（推荐生产环境）

```bash
# 1. 确保已安装Docker和Docker Compose
# https://docs.docker.com/get-docker/

# 2. 进入项目目录
cd agi_framework/UFO

# 3. 启动所有服务（应用 + PostgreSQL + Redis）
docker-compose up -d

# 4. 查看日志
docker-compose logs -f

# 5. 停止服务
docker-compose down
```

**Docker优势：**
- ✅ 一键部署完整环境
- ✅ 包含PostgreSQL数据库
- ✅ 包含Redis缓存
- ✅ 数据持久化
- ✅ 健康检查
- ✅ 自动重启

---

### 方式三：手动安装（开发者）

```bash
# 1. 确保Python 3.9+ 已安装
python --version

# 2. 创建虚拟环境
python -m venv .venv

# 3. 激活虚拟环境
# Linux/macOS:
source .venv/bin/activate
# Windows:
.venv\Scripts\activate

# 4. 安装依赖
pip install -r requirements.txt

# 5. 创建配置文件
cp .env.example .env
# 编辑 .env 配置你的API Key

# 6. 初始化数据库
python init_database.py

# 7. 启动应用
python main.py
```

---

### 方式四：打包成可执行文件（Windows用户）

```bash
# 1. 安装PyInstaller
pip install pyinstaller

# 2. 运行打包脚本
python build.py

# 3. 在 dist/ 目录找到可执行文件
# UFO-AGI.exe

# 4. 双击运行即可
```

---

## ⚙️ 配置说明

### 必需配置（首次运行）

编辑 `.env` 文件，配置以下API Key：

```env
# OpenAI (推荐)
OPENAI_API_KEY=sk-your-openai-api-key

# 或 Anthropic Claude
ANTHROPIC_API_KEY=sk-ant-your-anthropic-key

# 或 Azure OpenAI
AZURE_OPENAI_KEY=your-azure-key
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
```

> 💡 **提示**：如果没有API Key，可以使用本地模型（Ollama/LM Studio）

---

### 可选配置

```env
# 数据库（Docker已自动配置）
DB_HOST=localhost
DB_PORT=5432
DB_USER=agi_user
DB_PASSWORD=your-password
DB_NAME=agi_framework

# JWT密钥（生产环境必须修改）
JWT_SECRET_KEY=your-secret-key-change-this

# 渠道配置（Webhook）
TELEGRAM_BOT_TOKEN=your-bot-token
SLACK_BOT_TOKEN=xoxb-your-token
DINGTALK_WEBHOOK=https://oapi.dingtalk.com/robot/send?access_token=xxx
```

---

## 🔧 常见问题

### 问题1：端口被占用
```bash
# 修改 .env 文件
SERVER_PORT=8001  # 改为其他端口
```

### 问题2：权限不足（Linux/macOS）
```bash
chmod +x start.sh
```

### 问题3：Docker启动失败
```bash
# 检查Docker服务
sudo systemctl start docker

# 或者使用sudo
docker-compose up -d
```

### 问题4：数据库连接失败
```bash
# 使用SQLite（无需PostgreSQL）
# 修改 .env
DB_HOST=sqlite
DB_NAME=./data/ufo.db
```

---

## 🌐 访问应用

启动成功后，访问以下地址：

| 地址 | 说明 |
|------|------|
| http://localhost:8000 | Web界面 |
| http://localhost:8000/docs | API文档（Swagger）|
| http://localhost:8000/redoc | API文档（ReDoc）|
| http://localhost:8000/health | 健康检查 |

---

## 📚 下一步

1. **配置模型**：在Web界面添加你的API Key
2. **创建智能体**：使用Agent管理功能
3. **创建工作流**：使用Workflow设计器
4. **配置渠道**：添加Telegram/钉钉等渠道

---

## 🆘 获取帮助

- 📖 完整文档：http://localhost:8000/docs
- 🐛 问题反馈：查看 logs/ufo.log
- 💬 社区支持：[GitHub Discussions]

---

**🎉 恭喜！UFO AGI 框架已准备就绪！**
