#!/bin/bash
# =============================================================================
# AGI Unified Framework - Setup Script
# =============================================================================
# 一键安装脚本，用于自动化部署AGI Unified Framework
# 支持Docker Compose和Kubernetes两种部署模式
# =============================================================================

set -euo pipefail

# -----------------------------------------------------------------------------
# 颜色定义
# -----------------------------------------------------------------------------
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly NC='\033[0m' # No Color

# -----------------------------------------------------------------------------
# 全局变量
# -----------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DEPLOYMENT_DIR="${PROJECT_ROOT}/deployment"
LOG_FILE="/var/log/agi-framework-setup.log"
DEPLOY_MODE="docker"  # docker 或 kubernetes
ENVIRONMENT="production"
SKIP_CONFIRM=false
VERBOSE=false

# -----------------------------------------------------------------------------
# 日志函数
# -----------------------------------------------------------------------------
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1" | tee -a "$LOG_FILE"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1" | tee -a "$LOG_FILE"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1" | tee -a "$LOG_FILE"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1" | tee -a "$LOG_FILE"
}

log_debug() {
    if [[ "$VERBOSE" == true ]]; then
        echo -e "${BLUE}[DEBUG]${NC} $1" | tee -a "$LOG_FILE"
    fi
}

# -----------------------------------------------------------------------------
# 打印Banner
# -----------------------------------------------------------------------------
print_banner() {
    cat << 'EOF'
    _    ____ ___   _   _ _   _ ____  _   _ _____ _     _     
   / \  / ___|_ _| | | | | | | | __ )| | | | ____| |   | |    
  / _ \| |  _ | |  | | | | | | |  _ \| | | |  _| | |   | |    
 / ___ \ |_| || |  | |_| | |_| | |_) | |_| | |___| |___| |___ 
/_/   \_\____|___|  \___/ \___/|____/ \___/|_____|_____|_____|
                                                              
EOF
    echo "AGI Unified Framework - Setup Script v1.0.0"
    echo "================================================"
    echo ""
}

# -----------------------------------------------------------------------------
# 显示帮助信息
# -----------------------------------------------------------------------------
show_help() {
    cat << EOF
Usage: $0 [OPTIONS]

Options:
    -m, --mode          部署模式: docker 或 kubernetes (默认: docker)
    -e, --environment   环境: development, staging, production (默认: production)
    -y, --yes           跳过确认提示
    -v, --verbose       显示详细输出
    -h, --help          显示此帮助信息

Examples:
    $0                                    # 使用默认配置部署
    $0 -m docker -e production            # Docker模式生产环境部署
    $0 -m kubernetes -e staging           # K8s模式测试环境部署
    $0 -m docker -y -v                    # 自动确认并详细输出

EOF
}

# -----------------------------------------------------------------------------
# 解析命令行参数
# -----------------------------------------------------------------------------
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -m|--mode)
                DEPLOY_MODE="$2"
                shift 2
                ;;
            -e|--environment)
                ENVIRONMENT="$2"
                shift 2
                ;;
            -y|--yes)
                SKIP_CONFIRM=true
                shift
                ;;
            -v|--verbose)
                VERBOSE=true
                shift
                ;;
            -h|--help)
                show_help
                exit 0
                ;;
            *)
                log_error "未知参数: $1"
                show_help
                exit 1
                ;;
        esac
    done

    # 验证部署模式
    if [[ "$DEPLOY_MODE" != "docker" && "$DEPLOY_MODE" != "kubernetes" ]]; then
        log_error "无效的部署模式: $DEPLOY_MODE. 必须是 'docker' 或 'kubernetes'"
        exit 1
    fi

    # 验证环境
    if [[ "$ENVIRONMENT" != "development" && "$ENVIRONMENT" != "staging" && "$ENVIRONMENT" != "production" ]]; then
        log_error "无效的环境: $ENVIRONMENT"
        exit 1
    fi
}

# -----------------------------------------------------------------------------
# 检查系统要求
# -----------------------------------------------------------------------------
check_prerequisites() {
    log_info "检查系统要求..."

    local missing_deps=()

    # 检查必需命令
    if ! command -v curl &> /dev/null; then
        missing_deps+=("curl")
    fi

    if ! command -v git &> /dev/null; then
        missing_deps+=("git")
    fi

    # 根据部署模式检查特定依赖
    if [[ "$DEPLOY_MODE" == "docker" ]]; then
        if ! command -v docker &> /dev/null; then
            missing_deps+=("docker")
        fi
        if ! command -v docker-compose &> /dev/null; then
            missing_deps+=("docker-compose")
        fi
    else
        if ! command -v kubectl &> /dev/null; then
            missing_deps+=("kubectl")
        fi
        if ! command -v helm &> /dev/null; then
            missing_deps+=("helm")
        fi
    fi

    if [[ ${#missing_deps[@]} -gt 0 ]]; then
        log_error "缺少必需的依赖: ${missing_deps[*]}"
        log_info "请安装缺失的依赖后重试"
        exit 1
    fi

    # 检查Docker服务状态
    if [[ "$DEPLOY_MODE" == "docker" ]]; then
        if ! docker info &> /dev/null; then
            log_error "Docker服务未运行"
            exit 1
        fi
    fi

    # 检查Kubernetes连接
    if [[ "$DEPLOY_MODE" == "kubernetes" ]]; then
        if ! kubectl cluster-info &> /dev/null; then
            log_error "无法连接到Kubernetes集群"
            exit 1
        fi
    fi

    log_success "系统要求检查通过"
}

# -----------------------------------------------------------------------------
# 生成环境配置文件
# -----------------------------------------------------------------------------
generate_env_file() {
    log_info "生成环境配置文件..."

    local env_file="${PROJECT_ROOT}/.env"
    
    # 如果.env已存在，备份它
    if [[ -f "$env_file" ]]; then
        cp "$env_file" "${env_file}.backup.$(date +%Y%m%d%H%M%S)"
        log_warn "已备份现有 .env 文件"
    fi

    # 生成随机密钥
    local app_secret=$(openssl rand -hex 32)
    local db_password=$(openssl rand -hex 16)
    local redis_password=$(openssl rand -hex 16)
    local jwt_secret=$(openssl rand -hex 32)

    cat > "$env_file" << EOF
# =============================================================================
# AGI Unified Framework - Environment Configuration
# =============================================================================
# 生成时间: $(date)
# 环境: ${ENVIRONMENT}
# =============================================================================

# 应用配置
APP_NAME=AGI Unified Framework
APP_ENV=${ENVIRONMENT}
APP_DEBUG=false
APP_SECRET_KEY=${app_secret}
APP_VERSION=1.0.0

# 服务器配置
SERVER_HOST=0.0.0.0
SERVER_PORT=8000
WORKERS=4
TIMEOUT=120

# 数据库配置
DB_USER=agi_user
DB_PASSWORD=${db_password}
DB_NAME=agi_framework
DB_HOST=postgres
DB_PORT=5432
DB_POOL_SIZE=20
DB_MAX_OVERFLOW=10

# Redis配置
REDIS_HOST=redis
REDIS_PORT=6379
REDIS_PASSWORD=${redis_password}
REDIS_DB=0

# JWT配置
JWT_SECRET_KEY=${jwt_secret}
JWT_ALGORITHM=HS256
JWT_ACCESS_TOKEN_EXPIRE_MINUTES=30
JWT_REFRESH_TOKEN_EXPIRE_DAYS=7

# 消息队列配置
CELERY_BROKER_URL=redis://redis:6379/1
CELERY_RESULT_BACKEND=redis://redis:6379/2

# 日志配置
LOG_LEVEL=info
LOG_FORMAT=json

# 监控配置
METRICS_ENABLED=true
METRICS_PORT=9090

# 外部API密钥（请填入实际值）
# OPENAI_API_KEY=your-openai-api-key
# ANTHROPIC_API_KEY=your-anthropic-api-key

# 对象存储配置
S3_ENDPOINT=minio:9000
S3_ACCESS_KEY=minioadmin
S3_SECRET_KEY=minioadmin
S3_BUCKET_NAME=agi-assets

# 域名配置
DOMAIN=localhost

# 端口配置
HTTP_PORT=80
HTTPS_PORT=443
API_PORT=8000
DB_PORT=5432
REDIS_PORT=6379
MINIO_PORT=9000
MINIO_CONSOLE_PORT=9001
GRAFANA_PORT=3000
PROMETHEUS_PORT=9090

# 数据目录
DATA_DIR=./data
EOF

    chmod 600 "$env_file"
    log_success "环境配置文件已生成: $env_file"
}

# -----------------------------------------------------------------------------
# Docker Compose部署
# -----------------------------------------------------------------------------
deploy_docker() {
    log_info "开始Docker Compose部署..."

    cd "${DEPLOYMENT_DIR}/docker"

    # 创建数据目录
    mkdir -p "${PROJECT_ROOT}/data"/{postgres,redis,minio,logs,app,prometheus,grafana}

    # 拉取镜像
    log_info "拉取Docker镜像..."
    docker-compose pull

    # 启动服务
    log_info "启动服务..."
    docker-compose up -d

    # 等待服务就绪
    log_info "等待服务就绪..."
    sleep 10

    # 检查服务状态
    if docker-compose ps | grep -q "Up"; then
        log_success "Docker Compose部署成功"
        log_info "应用访问地址: http://localhost:8000"
        log_info "Grafana监控: http://localhost:3000"
        log_info "MinIO控制台: http://localhost:9001"
    else
        log_error "部署失败，请检查日志"
        docker-compose logs
        exit 1
    fi
}

# -----------------------------------------------------------------------------
# Kubernetes部署
# -----------------------------------------------------------------------------
deploy_kubernetes() {
    log_info "开始Kubernetes部署..."

    local namespace="agi-framework-${ENVIRONMENT}"

    # 创建命名空间
    kubectl create namespace "$namespace" --dry-run=client -o yaml | kubectl apply -f -

    # 部署依赖服务
    cd "${DEPLOYMENT_DIR}/helm/agi-framework"

    # 添加Helm仓库
    helm repo add bitnami https://charts.bitnami.com/bitnami
    helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
    helm repo add ingress-nginx https://kubernetes.github.io/ingress-nginx
    helm repo add jetstack https://charts.jetstack.io
    helm repo update

    # 安装Chart
    log_info "安装Helm Chart..."
    helm upgrade --install agi-framework . \
        --namespace "$namespace" \
        --set global.environment="$ENVIRONMENT" \
        --wait \
        --timeout 600s

    # 检查部署状态
    if kubectl rollout status deployment/agi-framework-api -n "$namespace"; then
        log_success "Kubernetes部署成功"
        log_info "命名空间: $namespace"
        kubectl get svc -n "$namespace"
    else
        log_error "部署失败"
        kubectl get pods -n "$namespace"
        exit 1
    fi
}

# -----------------------------------------------------------------------------
# 健康检查
# -----------------------------------------------------------------------------
health_check() {
    log_info "执行健康检查..."

    local max_attempts=30
    local attempt=1

    while [[ $attempt -le $max_attempts ]]; do
        if curl -sf http://localhost:8000/health &> /dev/null; then
            log_success "健康检查通过"
            return 0
        fi
        log_debug "健康检查尝试 $attempt/$max_attempts..."
        sleep 5
        ((attempt++))
    done

    log_error "健康检查失败"
    return 1
}

# -----------------------------------------------------------------------------
# 显示部署信息
# -----------------------------------------------------------------------------
show_deployment_info() {
    echo ""
    echo "================================================"
    log_success "AGI Unified Framework 部署完成"
    echo "================================================"
    echo ""
    echo "部署模式: $DEPLOY_MODE"
    echo "环境: $ENVIRONMENT"
    echo ""
    echo "访问地址:"
    echo "  - 应用API: http://localhost:8000"
    echo "  - 健康检查: http://localhost:8000/health"
    echo ""
    
    if [[ "$DEPLOY_MODE" == "docker" ]]; then
        echo "监控面板:"
        echo "  - Grafana: http://localhost:3000 (admin/admin)"
        echo "  - Prometheus: http://localhost:9090"
        echo "  - MinIO: http://localhost:9001"
        echo ""
        echo "常用命令:"
        echo "  - 查看日志: docker-compose logs -f"
        echo "  - 停止服务: docker-compose down"
        echo "  - 重启服务: docker-compose restart"
    else
        echo "查看Pod状态:"
        echo "  kubectl get pods -n agi-framework-${ENVIRONMENT}"
        echo ""
        echo "查看服务:"
        echo "  kubectl get svc -n agi-framework-${ENVIRONMENT}"
    fi
    echo ""
    echo "================================================"
}

# -----------------------------------------------------------------------------
# 主函数
# -----------------------------------------------------------------------------
main() {
    # 初始化日志文件
    sudo touch "$LOG_FILE"
    sudo chmod 666 "$LOG_FILE"

    print_banner
    parse_args "$@"

    log_info "开始部署 AGI Unified Framework"
    log_info "部署模式: $DEPLOY_MODE"
    log_info "环境: $ENVIRONMENT"

    # 确认提示
    if [[ "$SKIP_CONFIRM" == false ]]; then
        read -p "确认开始部署? [Y/n]: " confirm
        if [[ ! "$confirm" =~ ^[Yy]$ && ! "$confirm" == "" ]]; then
            log_info "部署已取消"
            exit 0
        fi
    fi

    # 执行部署步骤
    check_prerequisites
    generate_env_file

    if [[ "$DEPLOY_MODE" == "docker" ]]; then
        deploy_docker
    else
        deploy_kubernetes
    fi

    # 健康检查
    if [[ "$DEPLOY_MODE" == "docker" ]]; then
        health_check
    fi

    show_deployment_info
}

# 执行主函数
main "$@"
