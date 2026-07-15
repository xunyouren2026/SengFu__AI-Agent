#!/bin/bash
# =============================================================================
# AGI Unified Framework - Backup Script
# =============================================================================
# 自动化备份脚本，用于备份数据库、配置文件和重要数据
# 支持定时备份、增量备份和远程存储
# =============================================================================

set -euo pipefail

# -----------------------------------------------------------------------------
# 配置变量
# -----------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BACKUP_DIR="${BACKUP_DIR:-/var/backups/agi-framework}"
RETENTION_DAYS="${RETENTION_DAYS:-30}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_NAME="agi-framework-backup-${TIMESTAMP}"
LOG_FILE="/var/log/agi-framework-backup.log"

# 远程存储配置（可选）
S3_BUCKET="${S3_BUCKET:-}"
S3_ENDPOINT="${S3_ENDPOINT:-}"
AWS_ACCESS_KEY_ID="${AWS_ACCESS_KEY_ID:-}"
AWS_SECRET_ACCESS_KEY="${AWS_SECRET_ACCESS_KEY:-}"

# 数据库配置
DB_HOST="${DB_HOST:-localhost}"
DB_PORT="${DB_PORT:-5432}"
DB_USER="${DB_USER:-agi_user}"
DB_NAME="${DB_NAME:-agi_framework}"
DB_PASSWORD="${DB_PASSWORD:-}"

# Redis配置
REDIS_HOST="${REDIS_HOST:-localhost}"
REDIS_PORT="${REDIS_PORT:-6379}"
REDIS_PASSWORD="${REDIS_PASSWORD:-}"

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
    -t, --type          备份类型: full, db, config, data (默认: full)
    -d, --destination   备份目标目录 (默认: /var/backups/agi-framework)
    -r, --retention     保留天数 (默认: 30)
    -s, --s3            同时上传到S3
    -c, --compress      压缩级别: 0-9 (默认: 6)
    -h, --help          显示此帮助信息

Examples:
    $0                              # 执行完整备份
    $0 -t db                        # 仅备份数据库
    $0 -t config -d /backup         # 备份配置到指定目录
    $0 -s                           # 备份并上传到S3

EOF
}

# -----------------------------------------------------------------------------
# 解析命令行参数
# -----------------------------------------------------------------------------
BACKUP_TYPE="full"
COMPRESS_LEVEL=6
UPLOAD_S3=false

parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -t|--type)
                BACKUP_TYPE="$2"
                shift 2
                ;;
            -d|--destination)
                BACKUP_DIR="$2"
                shift 2
                ;;
            -r|--retention)
                RETENTION_DAYS="$2"
                shift 2
                ;;
            -s|--s3)
                UPLOAD_S3=true
                shift
                ;;
            -c|--compress)
                COMPRESS_LEVEL="$2"
                shift 2
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
}

# -----------------------------------------------------------------------------
# 初始化备份环境
# -----------------------------------------------------------------------------
init_backup() {
    log_info "初始化备份环境..."

    # 创建备份目录
    mkdir -p "$BACKUP_DIR"
    
    # 创建临时目录
    TEMP_DIR=$(mktemp -d)
    trap "rm -rf $TEMP_DIR" EXIT

    # 检查依赖
    if ! command -v pg_dump &> /dev/null; then
        log_error "未找到 pg_dump，请安装 PostgreSQL 客户端"
        exit 1
    fi

    log_success "备份环境初始化完成"
}

# -----------------------------------------------------------------------------
# 备份数据库
# -----------------------------------------------------------------------------
backup_database() {
    log_info "开始备份数据库..."

    local db_backup_dir="${TEMP_DIR}/database"
    mkdir -p "$db_backup_dir"

    # 设置密码环境变量
    export PGPASSWORD="$DB_PASSWORD"

    # 创建数据库备份
    local backup_file="${db_backup_dir}/${DB_NAME}_${TIMESTAMP}.sql"
    
    if pg_dump -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" \
               --verbose --no-owner --no-privileges \
               > "$backup_file" 2>> "$LOG_FILE"; then
        
        # 压缩备份文件
        gzip -"$COMPRESS_LEVEL" "$backup_file"
        log_success "数据库备份完成: ${backup_file}.gz"
    else
        log_error "数据库备份失败"
        return 1
    fi

    # 备份Redis数据（如果可访问）
    if command -v redis-cli &> /dev/null; then
        local redis_backup_file="${db_backup_dir}/redis_${TIMESTAMP}.rdb"
        if redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" ${REDIS_PASSWORD:+-a "$REDIS_PASSWORD"} \
                    --rdb "$redis_backup_file" 2>> "$LOG_FILE"; then
            gzip -"$COMPRESS_LEVEL" "$redis_backup_file"
            log_success "Redis备份完成: ${redis_backup_file}.gz"
        else
            log_warn "Redis备份失败（可能未启用持久化）"
        fi
    fi

    unset PGPASSWORD
}

# -----------------------------------------------------------------------------
# 备份配置文件
# -----------------------------------------------------------------------------
backup_config() {
    log_info "开始备份配置文件..."

    local config_backup_dir="${TEMP_DIR}/config"
    mkdir -p "$config_backup_dir"

    # 备份环境文件
    if [[ -f "${PROJECT_ROOT}/.env" ]]; then
        cp "${PROJECT_ROOT}/.env" "${config_backup_dir}/env_${TIMESTAMP}"
        log_success "环境文件备份完成"
    fi

    # 备份Docker Compose配置
    if [[ -d "${PROJECT_ROOT}/deployment/docker" ]]; then
        cp -r "${PROJECT_ROOT}/deployment/docker" "${config_backup_dir}/"
        log_success "Docker配置备份完成"
    fi

    # 备份Kubernetes配置
    if [[ -d "${PROJECT_ROOT}/deployment/kubernetes" ]]; then
        cp -r "${PROJECT_ROOT}/deployment/kubernetes" "${config_backup_dir}/"
        log_success "Kubernetes配置备份完成"
    fi

    # 备份Helm配置
    if [[ -d "${PROJECT_ROOT}/deployment/helm" ]]; then
        cp -r "${PROJECT_ROOT}/deployment/helm" "${config_backup_dir}/"
        log_success "Helm配置备份完成"
    fi

    # 创建配置文件清单
    find "$config_backup_dir" -type f > "${config_backup_dir}/manifest.txt"
}

# -----------------------------------------------------------------------------
# 备份应用数据
# -----------------------------------------------------------------------------
backup_data() {
    log_info "开始备份应用数据..."

    local data_backup_dir="${TEMP_DIR}/data"
    mkdir -p "$data_backup_dir"

    # 备份上传的文件
    local uploads_dir="${PROJECT_ROOT}/data/uploads"
    if [[ -d "$uploads_dir" ]]; then
        tar czf "${data_backup_dir}/uploads_${TIMESTAMP}.tar.gz" -C "$uploads_dir" .
        log_success "上传文件备份完成"
    fi

    # 备份日志文件（最近7天）
    local logs_dir="${PROJECT_ROOT}/data/logs"
    if [[ -d "$logs_dir" ]]; then
        find "$logs_dir" -name "*.log" -mtime -7 -exec tar czf "${data_backup_dir}/logs_${TIMESTAMP}.tar.gz" {} +
        log_success "日志文件备份完成"
    fi

    # 备份MinIO数据（如果存在）
    local minio_dir="${PROJECT_ROOT}/data/minio"
    if [[ -d "$minio_dir" ]]; then
        tar czf "${data_backup_dir}/minio_${TIMESTAMP}.tar.gz" -C "$minio_dir" .
        log_success "MinIO数据备份完成"
    fi
}

# -----------------------------------------------------------------------------
# 创建备份归档
# -----------------------------------------------------------------------------
create_archive() {
    log_info "创建备份归档..."

    local archive_name="${BACKUP_NAME}.tar.gz"
    local archive_path="${BACKUP_DIR}/${archive_name}"

    # 创建备份信息文件
    cat > "${TEMP_DIR}/backup_info.txt" << EOF
AGI Framework Backup
====================
Backup Name: ${BACKUP_NAME}
Backup Type: ${BACKUP_TYPE}
Created At: $(date)
Hostname: $(hostname)
Version: 1.0.0
EOF

    # 创建压缩归档
    tar czf "$archive_path" -C "$TEMP_DIR" .

    # 计算校验和
    local checksum=$(md5sum "$archive_path" | awk '{print $1}')
    echo "MD5: ${checksum}" >> "${BACKUP_DIR}/${BACKUP_NAME}.md5"

    log_success "备份归档创建完成: $archive_path"
    log_info "归档大小: $(du -h "$archive_path" | cut -f1)"
    log_info "MD5校验和: $checksum"
}

# -----------------------------------------------------------------------------
# 上传到S3
# -----------------------------------------------------------------------------
upload_to_s3() {
    if [[ "$UPLOAD_S3" != true ]]; then
        return 0
    fi

    log_info "上传到S3..."

    if [[ -z "$S3_BUCKET" ]]; then
        log_error "未配置S3_BUCKET"
        return 1
    fi

    # 检查AWS CLI
    if ! command -v aws &> /dev/null; then
        log_error "未找到AWS CLI"
        return 1
    fi

    local archive_name="${BACKUP_NAME}.tar.gz"
    local archive_path="${BACKUP_DIR}/${archive_name}"
    local s3_key="backups/${BACKUP_TYPE}/${archive_name}"

    # 配置AWS凭证
    export AWS_ACCESS_KEY_ID
    export AWS_SECRET_ACCESS_KEY

    if aws s3 cp "$archive_path" "s3://${S3_BUCKET}/${s3_key}" \
                  --endpoint-url "$S3_ENDPOINT" 2>> "$LOG_FILE"; then
        log_success "备份已上传到S3: s3://${S3_BUCKET}/${s3_key}"
    else
        log_error "上传到S3失败"
        return 1
    fi

    unset AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY
}

# -----------------------------------------------------------------------------
# 清理旧备份
# -----------------------------------------------------------------------------
cleanup_old_backups() {
    log_info "清理旧备份（保留${RETENTION_DAYS}天）..."

    # 清理本地备份
    find "$BACKUP_DIR" -name "agi-framework-backup-*.tar.gz" -mtime +$RETENTION_DAYS -delete
    find "$BACKUP_DIR" -name "agi-framework-backup-*.md5" -mtime +$RETENTION_DAYS -delete

    log_success "旧备份清理完成"

    # 清理S3备份（如果配置了）
    if [[ "$UPLOAD_S3" == true && -n "$S3_BUCKET" ]]; then
        log_info "清理S3旧备份..."
        
        local cutoff_date=$(date -d "${RETENTION_DAYS} days ago" +%Y-%m-%d)
        
        aws s3 ls "s3://${S3_BUCKET}/backups/${BACKUP_TYPE}/" \
            --endpoint-url "$S3_ENDPOINT" 2>/dev/null | \
            while read -r line; do
                local file_date=$(echo "$line" | awk '{print $1}')
                local file_name=$(echo "$line" | awk '{print $4}')
                
                if [[ "$file_date" < "$cutoff_date" ]]; then
                    aws s3 rm "s3://${S3_BUCKET}/backups/${BACKUP_TYPE}/${file_name}" \
                              --endpoint-url "$S3_ENDPOINT" 2>> "$LOG_FILE"
                    log_info "已删除S3旧备份: $file_name"
                fi
            done
    fi
}

# -----------------------------------------------------------------------------
# 发送通知
# -----------------------------------------------------------------------------
send_notification() {
    local status="$1"
    local message="$2"

    # 这里可以集成邮件、Slack、钉钉等通知
    # 示例：写入系统日志
    logger -t "agi-framework-backup" "[$status] $message"
}

# -----------------------------------------------------------------------------
# 主函数
# -----------------------------------------------------------------------------
main() {
    parse_args "$@"
    
    log_info "========================================"
    log_info "AGI Framework Backup Started"
    log_info "Backup Type: $BACKUP_TYPE"
    log_info "========================================"

    init_backup

    case $BACKUP_TYPE in
        full)
            backup_database
            backup_config
            backup_data
            ;;
        db)
            backup_database
            ;;
        config)
            backup_config
            ;;
        data)
            backup_data
            ;;
        *)
            log_error "未知的备份类型: $BACKUP_TYPE"
            exit 1
            ;;
    esac

    create_archive
    upload_to_s3
    cleanup_old_backups

    log_info "========================================"
    log_success "Backup completed successfully"
    log_info "Backup Location: ${BACKUP_DIR}/${BACKUP_NAME}.tar.gz"
    log_info "========================================"

    send_notification "SUCCESS" "Backup completed: ${BACKUP_NAME}"
}

# 执行主函数
main "$@"
