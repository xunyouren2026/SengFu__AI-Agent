#!/bin/bash
# =============================================================================
# AGI Unified Framework - Restore Script
# =============================================================================
# 数据恢复脚本，用于从备份中恢复数据库、配置和数据
# 支持完整恢复和选择性恢复
# =============================================================================

set -euo pipefail

# -----------------------------------------------------------------------------
# 配置变量
# -----------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BACKUP_DIR="${BACKUP_DIR:-/var/backups/agi-framework}"
LOG_FILE="/var/log/agi-framework-restore.log"

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
Usage: $0 [OPTIONS] <backup_file>

Options:
    -t, --type          恢复类型: full, db, config, data (默认: full)
    -f, --file          备份文件路径
    -s, --s3            从S3下载备份
    -b, --bucket        S3存储桶名称
    -k, --key           S3对象键
    --dry-run           模拟运行，不实际执行恢复
    --force             强制恢复，不提示确认
    -h, --help          显示此帮助信息

Examples:
    $0 -f /backups/agi-framework-backup-20240101_120000.tar.gz
    $0 -t db -f backup.tar.gz                    # 仅恢复数据库
    $0 -s -b my-bucket -k backups/backup.tar.gz  # 从S3恢复
    $0 -f backup.tar.gz --dry-run                # 模拟运行

EOF
}

# -----------------------------------------------------------------------------
# 解析命令行参数
# -----------------------------------------------------------------------------
RESTORE_TYPE="full"
BACKUP_FILE=""
FROM_S3=false
S3_BUCKET=""
S3_KEY=""
DRY_RUN=false
FORCE=false

parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -t|--type)
                RESTORE_TYPE="$2"
                shift 2
                ;;
            -f|--file)
                BACKUP_FILE="$2"
                shift 2
                ;;
            -s|--s3)
                FROM_S3=true
                shift
                ;;
            -b|--bucket)
                S3_BUCKET="$2"
                shift 2
                ;;
            -k|--key)
                S3_KEY="$2"
                shift 2
                ;;
            --dry-run)
                DRY_RUN=true
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
                if [[ -z "$BACKUP_FILE" && ! "$1" =~ ^- ]]; then
                    BACKUP_FILE="$1"
                    shift
                else
                    log_error "未知参数: $1"
                    show_help
                    exit 1
                fi
                ;;
        esac
    done
}

# -----------------------------------------------------------------------------
# 从S3下载备份
# -----------------------------------------------------------------------------
download_from_s3() {
    if [[ "$FROM_S3" != true ]]; then
        return 0
    fi

    log_info "从S3下载备份..."

    if [[ -z "$S3_BUCKET" || -z "$S3_KEY" ]]; then
        log_error "S3_BUCKET 和 S3_KEY 必须指定"
        exit 1
    fi

    if ! command -v aws &> /dev/null; then
        log_error "未找到AWS CLI"
        exit 1
    fi

    local temp_file="/tmp/$(basename "$S3_KEY")"
    
    if aws s3 cp "s3://${S3_BUCKET}/${S3_KEY}" "$temp_file"; then
        BACKUP_FILE="$temp_file"
        log_success "备份文件下载完成: $BACKUP_FILE"
    else
        log_error "从S3下载备份失败"
        exit 1
    fi
}

# -----------------------------------------------------------------------------
# 验证备份文件
# -----------------------------------------------------------------------------
verify_backup() {
    log_info "验证备份文件..."

    if [[ ! -f "$BACKUP_FILE" ]]; then
        log_error "备份文件不存在: $BACKUP_FILE"
        exit 1
    fi

    # 检查文件完整性
    if ! tar tzf "$BACKUP_FILE" > /dev/null 2>&1; then
        log_error "备份文件损坏或格式不正确"
        exit 1
    fi

    # 查找MD5校验文件
    local backup_name=$(basename "$BACKUP_FILE" .tar.gz)
    local md5_file="${BACKUP_DIR}/${backup_name}.md5"
    
    if [[ -f "$md5_file" ]]; then
        local expected_md5=$(grep "MD5:" "$md5_file" | awk '{print $2}')
        local actual_md5=$(md5sum "$BACKUP_FILE" | awk '{print $1}')
        
        if [[ "$expected_md5" != "$actual_md5" ]]; then
            log_error "MD5校验失败，备份文件可能已损坏"
            log_error "期望: $expected_md5"
            log_error "实际: $actual_md5"
            exit 1
        fi
        log_success "MD5校验通过"
    else
        log_warn "未找到MD5校验文件，跳过校验"
    fi

    # 显示备份信息
    log_info "备份内容:"
    tar tzf "$BACKUP_FILE" | head -20
    local total_files=$(tar tzf "$BACKUP_FILE" | wc -l)
    log_info "总文件数: $total_files"
}

# -----------------------------------------------------------------------------
# 解压备份
# -----------------------------------------------------------------------------
extract_backup() {
    log_info "解压备份文件..."

    TEMP_DIR=$(mktemp -d)
    trap "rm -rf $TEMP_DIR" EXIT

    if [[ "$DRY_RUN" == true ]]; then
        log_info "[DRY RUN] 将解压到: $TEMP_DIR"
        return 0
    fi

    tar xzf "$BACKUP_FILE" -C "$TEMP_DIR"
    log_success "备份文件解压完成"

    # 显示备份信息
    if [[ -f "${TEMP_DIR}/backup_info.txt" ]]; then
        log_info "备份信息:"
        cat "${TEMP_DIR}/backup_info.txt"
    fi
}

# -----------------------------------------------------------------------------
# 恢复数据库
# -----------------------------------------------------------------------------
restore_database() {
    log_info "开始恢复数据库..."

    local db_backup_dir="${TEMP_DIR}/database"
    
    if [[ ! -d "$db_backup_dir" ]]; then
        log_warn "备份中未找到数据库文件，跳过数据库恢复"
        return 0
    fi

    # 查找最新的数据库备份
    local db_backup=$(find "$db_backup_dir" -name "${DB_NAME}_*.sql.gz" | sort -r | head -1)
    
    if [[ -z "$db_backup" ]]; then
        log_warn "未找到数据库备份文件"
        return 0
    fi

    log_info "使用备份文件: $(basename "$db_backup")"

    if [[ "$DRY_RUN" == true ]]; then
        log_info "[DRY RUN] 将恢复数据库: $DB_NAME"
        return 0
    fi

    # 确认恢复
    if [[ "$FORCE" != true ]]; then
        echo -e "${YELLOW}警告: 这将覆盖现有数据库 '$DB_NAME'${NC}"
        read -p "确认继续? [y/N]: " confirm
        if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
            log_info "数据库恢复已取消"
            return 0
        fi
    fi

    # 设置密码环境变量
    export PGPASSWORD="$DB_PASSWORD"

    # 创建临时数据库用于验证
    local temp_db="${DB_NAME}_restore_temp_$(date +%s)"
    
    log_info "创建临时数据库..."
    createdb -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" "$temp_db"

    # 恢复数据到临时数据库
    log_info "恢复数据到临时数据库..."
    if gunzip -c "$db_backup" | psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$temp_db" 2>> "$LOG_FILE"; then
        log_success "数据恢复到临时数据库成功"
    else
        log_error "数据恢复失败"
        dropdb -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" "$temp_db"
        exit 1
    fi

    # 验证恢复的数据
    log_info "验证恢复的数据..."
    local table_count=$(psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$temp_db" -t -c "SELECT COUNT(*) FROM information_schema.tables WHERE table_schema='public';" | xargs)
    log_info "恢复的表数量: $table_count"

    # 替换生产数据库
    log_info "替换生产数据库..."
    
    # 重命名现有数据库
    local old_db="${DB_NAME}_old_$(date +%s)"
    psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d postgres -c "ALTER DATABASE \"$DB_NAME\" RENAME TO \"$old_db\";" 2>> "$LOG_FILE" || true
    
    # 重命名临时数据库为生产数据库
    psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d postgres -c "ALTER DATABASE \"$temp_db\" RENAME TO \"$DB_NAME\";"
    
    # 删除旧数据库
    psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d postgres -c "DROP DATABASE IF EXISTS \"$old_db\";" 2>> "$LOG_FILE" || true

    log_success "数据库恢复完成"

    # 恢复Redis数据
    local redis_backup=$(find "$db_backup_dir" -name "redis_*.rdb.gz" | sort -r | head -1)
    if [[ -n "$redis_backup" ]]; then
        log_info "恢复Redis数据..."
        # 注意：Redis恢复需要停止服务，这里仅记录
        log_warn "Redis数据恢复需要手动执行: gunzip -c $redis_backup > /var/lib/redis/dump.rdb"
    fi

    unset PGPASSWORD
}

# -----------------------------------------------------------------------------
# 恢复配置
# -----------------------------------------------------------------------------
restore_config() {
    log_info "开始恢复配置文件..."

    local config_backup_dir="${TEMP_DIR}/config"
    
    if [[ ! -d "$config_backup_dir" ]]; then
        log_warn "备份中未找到配置文件，跳过配置恢复"
        return 0
    fi

    if [[ "$DRY_RUN" == true ]]; then
        log_info "[DRY RUN] 将恢复配置文件"
        return 0
    fi

    # 备份当前配置
    local current_config_backup="${PROJECT_ROOT}/config_backup_$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$current_config_backup"
    
    if [[ -f "${PROJECT_ROOT}/.env" ]]; then
        cp "${PROJECT_ROOT}/.env" "$current_config_backup/"
    fi

    # 恢复环境文件
    local env_backup=$(find "$config_backup_dir" -name "env_*" | head -1)
    if [[ -n "$env_backup" ]]; then
        cp "$env_backup" "${PROJECT_ROOT}/.env"
        chmod 600 "${PROJECT_ROOT}/.env"
        log_success "环境文件恢复完成"
    fi

    # 恢复部署配置
    if [[ -d "${config_backup_dir}/docker" ]]; then
        rm -rf "${PROJECT_ROOT}/deployment/docker"
        cp -r "${config_backup_dir}/docker" "${PROJECT_ROOT}/deployment/"
        log_success "Docker配置恢复完成"
    fi

    if [[ -d "${config_backup_dir}/kubernetes" ]]; then
        rm -rf "${PROJECT_ROOT}/deployment/kubernetes"
        cp -r "${config_backup_dir}/kubernetes" "${PROJECT_ROOT}/deployment/"
        log_success "Kubernetes配置恢复完成"
    fi

    log_success "配置文件恢复完成"
}

# -----------------------------------------------------------------------------
# 恢复数据
# -----------------------------------------------------------------------------
restore_data() {
    log_info "开始恢复应用数据..."

    local data_backup_dir="${TEMP_DIR}/data"
    
    if [[ ! -d "$data_backup_dir" ]]; then
        log_warn "备份中未找到数据文件，跳过数据恢复"
        return 0
    fi

    if [[ "$DRY_RUN" == true ]]; then
        log_info "[DRY RUN] 将恢复应用数据"
        return 0
    fi

    # 恢复上传的文件
    local uploads_backup=$(find "$data_backup_dir" -name "uploads_*.tar.gz" | head -1)
    if [[ -n "$uploads_backup" ]]; then
        local uploads_dir="${PROJECT_ROOT}/data/uploads"
        mkdir -p "$uploads_dir"
        tar xzf "$uploads_backup" -C "$uploads_dir"
        log_success "上传文件恢复完成"
    fi

    # 恢复MinIO数据
    local minio_backup=$(find "$data_backup_dir" -name "minio_*.tar.gz" | head -1)
    if [[ -n "$minio_backup" ]]; then
        local minio_dir="${PROJECT_ROOT}/data/minio"
        mkdir -p "$minio_dir"
        tar xzf "$minio_backup" -C "$minio_dir"
        log_success "MinIO数据恢复完成"
    fi

    log_success "应用数据恢复完成"
}

# -----------------------------------------------------------------------------
# 验证恢复
# -----------------------------------------------------------------------------
verify_restore() {
    log_info "验证恢复结果..."

    if [[ "$DRY_RUN" == true ]]; then
        log_info "[DRY RUN] 跳过验证"
        return 0
    fi

    # 验证数据库连接
    export PGPASSWORD="$DB_PASSWORD"
    if psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -c "SELECT 1;" > /dev/null 2>&1; then
        log_success "数据库连接正常"
    else
        log_error "数据库连接失败"
        return 1
    fi
    unset PGPASSWORD

    # 验证应用健康
    if curl -sf http://localhost:8000/health > /dev/null 2>&1; then
        log_success "应用健康检查通过"
    else
        log_warn "应用健康检查失败，可能需要重启服务"
    fi

    log_success "恢复验证完成"
}

# -----------------------------------------------------------------------------
# 主函数
# -----------------------------------------------------------------------------
main() {
    parse_args "$@"

    log_info "========================================"
    log_info "AGI Framework Restore Started"
    log_info "Restore Type: $RESTORE_TYPE"
    if [[ "$DRY_RUN" == true ]]; then
        log_info "Mode: DRY RUN (模拟运行)"
    fi
    log_info "========================================"

    # 下载备份（如果需要）
    download_from_s3

    # 验证备份文件
    verify_backup

    # 确认恢复
    if [[ "$FORCE" != true && "$DRY_RUN" != true ]]; then
        echo -e "${YELLOW}警告: 恢复操作将覆盖现有数据${NC}"
        read -p "确认继续恢复? [y/N]: " confirm
        if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
            log_info "恢复已取消"
            exit 0
        fi
    fi

    # 解压备份
    extract_backup

    # 执行恢复
    case $RESTORE_TYPE in
        full)
            restore_database
            restore_config
            restore_data
            ;;
        db)
            restore_database
            ;;
        config)
            restore_config
            ;;
        data)
            restore_data
            ;;
        *)
            log_error "未知的恢复类型: $RESTORE_TYPE"
            exit 1
            ;;
    esac

    # 验证恢复
    verify_restore

    log_info "========================================"
    if [[ "$DRY_RUN" == true ]]; then
        log_success "模拟运行完成"
    else
        log_success "恢复完成"
    fi
    log_info "========================================"
}

# 执行主函数
main "$@"
