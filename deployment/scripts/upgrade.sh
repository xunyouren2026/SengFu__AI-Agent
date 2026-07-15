#!/bin/bash
# =============================================================================
# AGI Unified Framework - Upgrade Script
# =============================================================================
# 自动化升级脚本，支持版本检查、备份、升级和回滚
# 确保升级过程安全可控
# =============================================================================

set -euo pipefail

# -----------------------------------------------------------------------------
# 配置变量
# -----------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BACKUP_DIR="${BACKUP_DIR:-/var/backups/agi-framework}"
LOG_FILE="/var/log/agi-framework-upgrade.log"
CURRENT_VERSION="1.0.0"
TARGET_VERSION=""

# 部署模式
DEPLOY_MODE="${DEPLOY_MODE:-docker}"

# 回滚配置
ROLLBACK_ENABLED=true
ROLLBACK_VERSION=""

# -----------------------------------------------------------------------------
# 颜色定义
# -----------------------------------------------------------------------------
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly NC='\033[0m'

# -----------------------------------------------------------------------------
# 日志函数
# -----------------------------------------------------------------------------
log_info() {
    echo -e "${BLUE}[INFO]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a "$LOG_FILE"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a "$LOG_FILE"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a "$LOG_FILE"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a "$LOG_FILE"
}

# -----------------------------------------------------------------------------
# 显示帮助信息
# -----------------------------------------------------------------------------
show_help() {
    cat << EOF
Usage: $0 [OPTIONS]

Options:
    -v, --version       目标版本号 (必需)
    -m, --mode          部署模式: docker 或 kubernetes (默认: docker)
    --skip-backup       跳过备份步骤
    --skip-migration    跳过数据库迁移
    --rollback          执行回滚到指定版本
    --force             强制升级，跳过确认
    -h, --help          显示此帮助信息

Examples:
    $0 -v 1.1.0                           # 升级到 1.1.0
    $0 -v 1.1.0 -m kubernetes             # K8s环境升级
    $0 --rollback -v 1.0.0                # 回滚到 1.0.0
    $0 -v 1.1.0 --skip-backup             # 跳过备份

EOF
}

# -----------------------------------------------------------------------------
# 解析命令行参数
# -----------------------------------------------------------------------------
SKIP_BACKUP=false
SKIP_MIGRATION=false
ROLLBACK_MODE=false
FORCE=false

parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -v|--version)
                TARGET_VERSION="$2"
                shift 2
                ;;
            -m|--mode)
                DEPLOY_MODE="$2"
                shift 2
                ;;
            --skip-backup)
                SKIP_BACKUP=true
                shift
                ;;
            --skip-migration)
                SKIP_MIGRATION=true
                shift
                ;;
            --rollback)
                ROLLBACK_MODE=true
                shift
                ;;
            --force)
                FORCE=true
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

    # 验证必需参数
    if [[ -z "$TARGET_VERSION" ]]; then
        log_error "必须指定目标版本号 (-v)"
        show_help
        exit 1
    fi

    # 验证部署模式
    if [[ "$DEPLOY_MODE" != "docker" && "$DEPLOY_MODE" != "kubernetes" ]]; then
        log_error "无效的部署模式: $DEPLOY_MODE"
        exit 1
    fi
}

# -----------------------------------------------------------------------------
# 检查当前版本
# -----------------------------------------------------------------------------
check_current_version() {
    log_info "检查当前版本..."

    # 从环境变量或配置文件读取当前版本
    if [[ -f "${PROJECT_ROOT}/.env" ]]; then
        CURRENT_VERSION=$(grep "APP_VERSION" "${PROJECT_ROOT}/.env" | cut -d= -f2 || echo "1.0.0")
    fi

    log_info "当前版本: $CURRENT_VERSION"
    log_info "目标版本: $TARGET_VERSION"

    # 版本比较
    if [[ "$CURRENT_VERSION" == "$TARGET_VERSION" ]]; then
        log_warn "当前版本与目标版本相同"
        if [[ "$FORCE" != true ]]; then
            read -p "确认重新部署? [y/N]: " confirm
            if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
                log_info "升级已取消"
                exit 0
            fi
        fi
    fi

    # 检查版本兼容性
    local current_major=$(echo "$CURRENT_VERSION" | cut -d. -f1)
    local target_major=$(echo "$TARGET_VERSION" | cut -d. -f1)
    
    if [[ "$current_major" != "$target_major" ]]; then
        log_warn "主版本号变更，可能存在不兼容的更改"
        if [[ "$FORCE" != true ]]; then
            read -p "确认继续升级? [y/N]: " confirm
            if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
                log_info "升级已取消"
                exit 0
            fi
        fi
    fi
}

# -----------------------------------------------------------------------------
# 执行备份
# -----------------------------------------------------------------------------
perform_backup() {
    if [[ "$SKIP_BACKUP" == true ]]; then
        log_warn "跳过备份步骤"
        return 0
    fi

    log_info "执行升级前备份..."

    local backup_script="${SCRIPT_DIR}/backup.sh"
    if [[ -x "$backup_script" ]]; then
        "$backup_script" -t full -r 30
        log_success "备份完成"
    else
        log_error "备份脚本不存在或不可执行: $backup_script"
        exit 1
    fi

    # 记录回滚版本
    ROLLBACK_VERSION="$CURRENT_VERSION"
}

# -----------------------------------------------------------------------------
# 下载新版本
# -----------------------------------------------------------------------------
download_new_version() {
    log_info "准备新版本 $TARGET_VERSION..."

    # 拉取新镜像
    if [[ "$DEPLOY_MODE" == "docker" ]]; then
        log_info "拉取Docker镜像..."
        docker pull "registry.company.com/agi-framework:${TARGET_VERSION}"
    else
        log_info "更新Helm Chart..."
        cd "${PROJECT_ROOT}/deployment/helm/agi-framework"
        helm dependency update
    fi

    log_success "新版本准备完成"
}

# -----------------------------------------------------------------------------
# 执行数据库迁移
# -----------------------------------------------------------------------------
run_migrations() {
    if [[ "$SKIP_MIGRATION" == true ]]; then
        log_warn "跳过数据库迁移"
        return 0
    fi

    log_info "执行数据库迁移..."

    if [[ "$DEPLOY_MODE" == "docker" ]]; then
        # 使用临时容器执行迁移
        docker run --rm \
            --env-file "${PROJECT_ROOT}/.env" \
            --network agi-network \
            "registry.company.com/agi-framework:${TARGET_VERSION}" \
            alembic upgrade head
    else
        # Kubernetes环境下使用Job执行迁移
        local namespace="agi-framework-prod"
        
        cat <<EOF | kubectl apply -f -
apiVersion: batch/v1
kind: Job
metadata:
  name: db-migration-$(date +%s)
  namespace: ${namespace}
spec:
  ttlSecondsAfterFinished: 3600
  template:
    spec:
      restartPolicy: OnFailure
      containers:
        - name: migration
          image: registry.company.com/agi-framework:${TARGET_VERSION}
          command: ["alembic", "upgrade", "head"]
          envFrom:
            - secretRef:
                name: agi-framework-secrets
            - configMapRef:
                name: agi-framework-config
EOF

        # 等待迁移完成
        kubectl wait --for=condition=complete job/db-migration -n "$namespace" --timeout=300s
    fi

    log_success "数据库迁移完成"
}

# -----------------------------------------------------------------------------
# 升级服务
# -----------------------------------------------------------------------------
upgrade_services() {
    log_info "升级服务..."

    if [[ "$DEPLOY_MODE" == "docker" ]]; then
        upgrade_docker
    else
        upgrade_kubernetes
    fi
}

# -----------------------------------------------------------------------------
# Docker升级
# -----------------------------------------------------------------------------
upgrade_docker() {
    log_info "Docker Compose升级..."

    cd "${PROJECT_ROOT}/deployment/docker"

    # 更新镜像标签
    export AGI_FRAMEWORK_VERSION="$TARGET_VERSION"

    # 拉取新镜像
    docker-compose pull

    # 优雅地重启服务
    docker-compose up -d --no-deps --scale agi-api=2 agi-api
    
    # 等待新实例就绪
    sleep 10
    
    # 滚动更新
    docker-compose up -d

    # 清理旧镜像
    docker image prune -f

    log_success "Docker服务升级完成"
}

# -----------------------------------------------------------------------------
# Kubernetes升级
# -----------------------------------------------------------------------------
upgrade_kubernetes() {
    log_info "Kubernetes升级..."

    local namespace="agi-framework-prod"
    local release_name="agi-framework"

    cd "${PROJECT_ROOT}/deployment/helm/agi-framework"

    # 使用Helm升级
    helm upgrade "$release_name" . \
        --namespace "$namespace" \
        --set api.image.tag="$TARGET_VERSION" \
        --set worker.image.tag="$TARGET_VERSION" \
        --set beat.image.tag="$TARGET_VERSION" \
        --wait \
        --timeout 600s

    # 等待滚动更新完成
    kubectl rollout status deployment/agi-framework-api -n "$namespace" --timeout=300s
    kubectl rollout status deployment/agi-framework-worker -n "$namespace" --timeout=300s

    log_success "Kubernetes服务升级完成"
}

# -----------------------------------------------------------------------------
# 验证升级
# -----------------------------------------------------------------------------
verify_upgrade() {
    log_info "验证升级结果..."

    local max_attempts=30
    local attempt=1

    # 等待服务就绪
    while [[ $attempt -le $max_attempts ]]; do
        if curl -sf http://localhost:8000/health > /dev/null 2>&1; then
            break
        fi
        log_info "等待服务就绪... ($attempt/$max_attempts)"
        sleep 5
        ((attempt++))
    done

    if [[ $attempt -gt $max_attempts ]]; then
        log_error "服务启动超时"
        return 1
    fi

    # 验证版本
    local running_version=$(curl -sf http://localhost:8000/version 2>/dev/null | grep -o '"version":"[^"]*"' | cut -d'"' -f4 || echo "")
    
    if [[ "$running_version" == "$TARGET_VERSION" ]]; then
        log_success "版本验证通过: $running_version"
    else
        log_warn "版本验证失败: 期望 $TARGET_VERSION, 实际 $running_version"
    fi

    # 健康检查
    if curl -sf http://localhost:8000/health > /dev/null 2>&1; then
        log_success "健康检查通过"
    else
        log_error "健康检查失败"
        return 1
    fi

    log_success "升级验证完成"
}

# -----------------------------------------------------------------------------
# 回滚操作
# -----------------------------------------------------------------------------
rollback() {
    if [[ "$ROLLBACK_ENABLED" != true ]]; then
        log_error "回滚未启用"
        return 1
    fi

    log_error "升级失败，执行回滚..."

    if [[ "$DEPLOY_MODE" == "docker" ]]; then
        # Docker回滚
        cd "${PROJECT_ROOT}/deployment/docker"
        export AGI_FRAMEWORK_VERSION="$ROLLBACK_VERSION"
        docker-compose up -d
    else
        # Kubernetes回滚
        helm rollback agi-framework 0 -n agi-framework-prod
    fi

    # 数据库回滚
    if [[ "$SKIP_MIGRATION" != true ]]; then
        log_info "回滚数据库迁移..."
        # 注意：数据库回滚需要手动处理或使用特定迁移版本
        log_warn "数据库回滚需要手动执行: alembic downgrade"
    fi

    log_success "回滚完成"
}

# -----------------------------------------------------------------------------
# 更新版本文件
# -----------------------------------------------------------------------------
update_version_file() {
    log_info "更新版本文件..."

    # 更新.env文件
    if [[ -f "${PROJECT_ROOT}/.env" ]]; then
        sed -i "s/APP_VERSION=.*/APP_VERSION=${TARGET_VERSION}/" "${PROJECT_ROOT}/.env"
    fi

    # 记录升级历史
    echo "$(date): Upgraded from $CURRENT_VERSION to $TARGET_VERSION" >> "${PROJECT_ROOT}/.upgrade_history"

    log_success "版本文件更新完成"
}

# -----------------------------------------------------------------------------
# 发送通知
# -----------------------------------------------------------------------------
send_notification() {
    local status="$1"
    local message="$2"

    logger -t "agi-framework-upgrade" "[$status] $message"
    
    # 这里可以集成Slack、邮件等通知
    # 示例：
    # curl -X POST -H 'Content-type: application/json' \
    #     --data '{"text":"AGI Framework Upgrade: '"$status"' - '"$message"'"}' \
    #     $SLACK_WEBHOOK_URL
}

# -----------------------------------------------------------------------------
# 主函数
# -----------------------------------------------------------------------------
main() {
    parse_args "$@"

    log_info "========================================"
    if [[ "$ROLLBACK_MODE" == true ]]; then
        log_info "AGI Framework Rollback"
        log_info "Target Version: $TARGET_VERSION"
    else
        log_info "AGI Framework Upgrade"
        log_info "Target Version: $TARGET_VERSION"
        log_info "Mode: $DEPLOY_MODE"
    fi
    log_info "========================================"

    # 确认操作
    if [[ "$FORCE" != true ]]; then
        if [[ "$ROLLBACK_MODE" == true ]]; then
            echo -e "${YELLOW}警告: 将执行回滚到版本 $TARGET_VERSION${NC}"
        else
            echo -e "${YELLOW}警告: 将执行升级到版本 $TARGET_VERSION${NC}"
        fi
        read -p "确认继续? [y/N]: " confirm
        if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
            log_info "操作已取消"
            exit 0
        fi
    fi

    if [[ "$ROLLBACK_MODE" == true ]]; then
        # 执行回滚
        rollback
        exit 0
    fi

    # 执行升级流程
    check_current_version
    perform_backup
    download_new_version
    
    # 执行升级（带错误处理）
    if run_migrations && upgrade_services && verify_upgrade; then
        update_version_file
        log_info "========================================"
        log_success "升级成功完成"
        log_info "版本: $CURRENT_VERSION -> $TARGET_VERSION"
        log_info "========================================"
        send_notification "SUCCESS" "Upgraded to $TARGET_VERSION"
    else
        log_error "========================================"
        log_error "升级失败"
        log_error "========================================"
        rollback
        send_notification "FAILED" "Upgrade to $TARGET_VERSION failed"
        exit 1
    fi
}

# 执行主函数
main "$@"
