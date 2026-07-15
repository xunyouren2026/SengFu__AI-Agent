"""
AGI Unified Framework - Celery Application Configuration

Celery 异步任务队列配置模块。
提供任务队列、定时任务和后台处理功能。

使用示例:
    >>> from api.celery_app import celery
    >>> celery.send_task('my_task', args=[1, 2])
"""

from __future__ import annotations

import os
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

# Celery 实例将在 Celery 可用时初始化
_celery_app = None


def get_celery_app():
    """
    获取 Celery 应用实例

    Returns:
        Celery 应用实例，如果 Celery 未安装则返回 None
    """
    global _celery_app

    if _celery_app is not None:
        return _celery_app

    try:
        from celery import Celery

        broker_url = os.environ.get(
            'CELERY_BROKER_URL',
            os.environ.get('REDIS_URL', 'redis://localhost:6379/1')
        )
        result_backend = os.environ.get(
            'CELERY_RESULT_BACKEND',
            os.environ.get('REDIS_URL', 'redis://localhost:6379/2')
        )

        _celery_app = Celery(
            'agi_framework',
            broker=broker_url,
            backend=result_backend,
            include=[
                'api.tasks',
            ],
        )

        _celery_app.conf.update(
            task_serializer='json',
            accept_content=['json'],
            result_serializer='json',
            timezone='Asia/Shanghai',
            enable_utc=True,
            task_track_started=True,
            task_acks_late=True,
            worker_prefetch_multiplier=1,
            result_expires=3600,
        )

        logger.info("Celery application initialized")
        return _celery_app

    except ImportError:
        logger.warning("Celery is not installed. Task queue features are disabled.")
        return None


# 延迟初始化的 celery 实例属性
class _CeleryLazyProxy:
    """Celery 延迟代理，支持 celery -A api.celery_app 的调用方式"""

    def __getattr__(self, name):
        app = get_celery_app()
        if app is None:
            raise RuntimeError(
                "Celery is not installed. Install it with: pip install celery"
            )
        return getattr(app, name)

    def __call__(self, *args, **kwargs):
        app = get_celery_app()
        if app is None:
            raise RuntimeError(
                "Celery is not installed. Install it with: pip install celery"
            )
        return app(*args, **kwargs)


# 模块级 celery 实例，供 celery -A api.celery_app 使用
celery = _CeleryLazyProxy()

__all__ = ['celery', 'get_celery_app']
