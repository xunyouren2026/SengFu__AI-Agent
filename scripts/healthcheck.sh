#!/bin/bash
# =============================================================================
# AGI Unified Framework - Health Check Script
# =============================================================================
# Docker容器健康检查脚本
# =============================================================================

set -e

HEALTH_URL="${HEALTH_URL:-http://localhost:${PORT:-8000}/api/v1/health}"
TIMEOUT="${HEALTH_TIMEOUT:-10}"
MAX_RETRIES="${HEALTH_RETRIES:-3}"

check_health() {
    # 尝试使用 curl 进行健康检查
    if command -v curl &> /dev/null; then
        HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" \
            --max-time "${TIMEOUT}" \
            "${HEALTH_URL}" 2>/dev/null || echo "000")

        if [ "${HTTP_CODE}" -ge 200 ] && [ "${HTTP_CODE}" -lt 300 ]; then
            echo "healthy"
            return 0
        else
            echo "unhealthy (HTTP ${HTTP_CODE})"
            return 1
        fi
    # 回退到使用 python
    elif command -v python &> /dev/null; then
        python -c "
import urllib.request
import sys
try:
    req = urllib.request.Request('${HEALTH_URL}')
    with urllib.request.urlopen(req, timeout=${TIMEOUT}) as resp:
        if 200 <= resp.status < 300:
            print('healthy')
            sys.exit(0)
        else:
            print(f'unhealthy (HTTP {resp.status})')
            sys.exit(1)
except Exception as e:
    print(f'unhealthy ({e})')
    sys.exit(1)
" 2>/dev/null
        return $?
    else
        echo "unhealthy (no curl or python available)"
        return 1
    fi
}

# 执行健康检查
check_health
