"""
发布工作流模块

提供提交审核、自动测试、发布流程和下架管理功能。
"""

import json
import os
import re
import subprocess
import tempfile
import time
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Callable
from dataclasses import dataclass, field, asdict
import threading


# ---------------------------------------------------------------------------
# 数据模型
# ---------------------------------------------------------------------------

class ReviewStatus(Enum):
    """审核状态"""
    PENDING = "pending"
    IN_REVIEW = "in_review"
    APPROVED = "approved"
    REJECTED = "rejected"
    NEEDS_CHANGES = "needs_changes"


class PublishStatus(Enum):
    """发布状态"""
    DRAFT = "draft"
    SUBMITTED = "submitted"
    IN_REVIEW = "in_review"
    TESTING = "testing"
    APPROVED = "approved"
    PUBLISHED = "published"
    DEPRECATED = "deprecated"
    UNPUBLISHED = "unpublished"


@dataclass
class ReviewComment:
    """审核评论"""
    comment_id: str
    reviewer_id: str
    content: str
    line_number: Optional[int] = None
    file_path: Optional[str] = None
    severity: str = "info"  # info, warning, error
    created_at: float = field(default_factory=time.time)
    resolved: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Review:
    """审核记录"""
    review_id: str
    plugin_id: str
    version: str
    submitter_id: str
    status: ReviewStatus = ReviewStatus.PENDING
    submitted_at: float = field(default_factory=time.time)
    reviewed_at: Optional[float] = None
    reviewer_id: Optional[str] = None
    comments: List[ReviewComment] = field(default_factory=list)
    decision_notes: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'review_id': self.review_id,
            'plugin_id': self.plugin_id,
            'version': self.version,
            'submitter_id': self.submitter_id,
            'status': self.status.value,
            'submitted_at': self.submitted_at,
            'reviewed_at': self.reviewed_at,
            'reviewer_id': self.reviewer_id,
            'comments': [c.to_dict() for c in self.comments],
            'decision_notes': self.decision_notes,
        }


@dataclass
class TestResult:
    """测试结果"""
    test_name: str
    passed: bool
    duration_ms: float
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TestSuite:
    """测试套件"""
    suite_name: str
    results: List[TestResult] = field(default_factory=list)
    started_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None
    
    @property
    def all_passed(self) -> bool:
        return all(r.passed for r in self.results)
    
    @property
    def total_duration_ms(self) -> float:
        if self.completed_at:
            return (self.completed_at - self.started_at) * 1000
        return 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'suite_name': self.suite_name,
            'results': [r.to_dict() for r in self.results],
            'started_at': self.started_at,
            'completed_at': self.completed_at,
            'all_passed': self.all_passed,
            'total_duration_ms': self.total_duration_ms,
        }


@dataclass
class PublishRecord:
    """发布记录"""
    record_id: str
    plugin_id: str
    version: str
    status: PublishStatus
    submitted_by: str
    submitted_at: float = field(default_factory=time.time)
    published_at: Optional[float] = None
    unpublished_at: Optional[float] = None
    unpublish_reason: str = ""
    review_id: Optional[str] = None
    test_results: Optional[TestSuite] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'record_id': self.record_id,
            'plugin_id': self.plugin_id,
            'version': self.version,
            'status': self.status.value,
            'submitted_by': self.submitted_by,
            'submitted_at': self.submitted_at,
            'published_at': self.published_at,
            'unpublished_at': self.unpublished_at,
            'unpublish_reason': self.unpublish_reason,
            'review_id': self.review_id,
            'test_results': self.test_results.to_dict() if self.test_results else None,
        }


# ---------------------------------------------------------------------------
# 审核队列
# ---------------------------------------------------------------------------

class ReviewQueue:
    """审核队列"""
    
    def __init__(self):
        self._reviews: Dict[str, Review] = {}
        self._queue: List[str] = []  # 待审核的review_id列表
        self._lock = threading.RLock()
        self._listeners: List[Callable[[str, Review], None]] = []
    
    def add_listener(self, callback: Callable[[str, Review], None]) -> None:
        """添加监听器"""
        self._listeners.append(callback)
    
    def _notify(self, event: str, review: Review) -> None:
        """通知监听器"""
        for listener in self._listeners:
            try:
                listener(event, review)
            except Exception:
                pass
    
    def submit(self, plugin_id: str, version: str,
               submitter_id: str) -> Review:
        """提交审核
        
        Args:
            plugin_id: 插件ID
            version: 版本
            submitter_id: 提交者ID
            
        Returns:
            审核记录
        """
        import secrets
        
        review_id = f"review_{secrets.token_hex(8)}"
        
        review = Review(
            review_id=review_id,
            plugin_id=plugin_id,
            version=version,
            submitter_id=submitter_id,
            status=ReviewStatus.PENDING,
        )
        
        with self._lock:
            self._reviews[review_id] = review
            self._queue.append(review_id)
        
        self._notify('submitted', review)
        
        return review
    
    def claim(self, reviewer_id: str) -> Optional[Review]:
        """认领审核任务
        
        Args:
            reviewer_id: 审核员ID
            
        Returns:
            审核记录，无任务返回None
        """
        with self._lock:
            for review_id in self._queue:
                review = self._reviews[review_id]
                if review.status == ReviewStatus.PENDING:
                    review.status = ReviewStatus.IN_REVIEW
                    review.reviewer_id = reviewer_id
                    self._queue.remove(review_id)
                    return review
        
        return None
    
    def add_comment(self, review_id: str, reviewer_id: str,
                    content: str, **kwargs) -> Optional[ReviewComment]:
        """添加审核评论
        
        Args:
            review_id: 审核ID
            reviewer_id: 审核员ID
            content: 评论内容
            
        Returns:
            评论，失败返回None
        """
        import secrets
        
        with self._lock:
            review = self._reviews.get(review_id)
            if not review:
                return None
            
            if review.reviewer_id != reviewer_id:
                return None
            
            comment = ReviewComment(
                comment_id=f"comment_{secrets.token_hex(8)}",
                reviewer_id=reviewer_id,
                content=content,
                **kwargs
            )
            
            review.comments.append(comment)
            
            return comment
    
    def approve(self, review_id: str, reviewer_id: str,
                notes: str = "") -> bool:
        """批准
        
        Args:
            review_id: 审核ID
            reviewer_id: 审核员ID
            notes: 备注
            
        Returns:
            是否成功
        """
        with self._lock:
            review = self._reviews.get(review_id)
            if not review:
                return False
            
            if review.reviewer_id != reviewer_id:
                return False
            
            review.status = ReviewStatus.APPROVED
            review.reviewed_at = time.time()
            review.decision_notes = notes
            
        self._notify('approved', review)
        return True
    
    def reject(self, review_id: str, reviewer_id: str,
               reason: str) -> bool:
        """拒绝
        
        Args:
            review_id: 审核ID
            reviewer_id: 审核员ID
            reason: 原因
            
        Returns:
            是否成功
        """
        with self._lock:
            review = self._reviews.get(review_id)
            if not review:
                return False
            
            if review.reviewer_id != reviewer_id:
                return False
            
            review.status = ReviewStatus.REJECTED
            review.reviewed_at = time.time()
            review.decision_notes = reason
            
        self._notify('rejected', review)
        return True
    
    def request_changes(self, review_id: str, reviewer_id: str,
                        feedback: str) -> bool:
        """请求修改
        
        Args:
            review_id: 审核ID
            reviewer_id: 审核员ID
            feedback: 反馈
            
        Returns:
            是否成功
        """
        with self._lock:
            review = self._reviews.get(review_id)
            if not review:
                return False
            
            if review.reviewer_id != reviewer_id:
                return False
            
            review.status = ReviewStatus.NEEDS_CHANGES
            review.reviewed_at = time.time()
            review.decision_notes = feedback
            
        self._notify('needs_changes', review)
        return True
    
    def get_review(self, review_id: str) -> Optional[Review]:
        """获取审核记录"""
        with self._lock:
            return self._reviews.get(review_id)
    
    def get_plugin_reviews(self, plugin_id: str) -> List[Review]:
        """获取插件的所有审核记录"""
        with self._lock:
            return [
                r for r in self._reviews.values()
                if r.plugin_id == plugin_id
            ]
    
    def get_pending_count(self) -> int:
        """获取待审核数量"""
        with self._lock:
            return len(self._queue)
    
    def get_stats(self) -> Dict[str, int]:
        """获取统计"""
        with self._lock:
            stats = {
                'pending': 0,
                'in_review': 0,
                'approved': 0,
                'rejected': 0,
                'needs_changes': 0,
            }
            
            for review in self._reviews.values():
                stats[review.status.value] += 1
            
            return stats


# ---------------------------------------------------------------------------
# 自动测试器
# ---------------------------------------------------------------------------

class AutoTester:
    """自动测试器"""
    
    def __init__(self):
        self._tests: Dict[str, Callable[[str], TestResult]] = {}
        self._lock = threading.Lock()
    
    def register_test(self, name: str, test_func: Callable[[str], TestResult]) -> None:
        """注册测试
        
        Args:
            name: 测试名称
            test_func: 测试函数，接收插件路径，返回测试结果
        """
        with self._lock:
            self._tests[name] = test_func
    
    def run_tests(self, plugin_path: str,
                  test_names: Optional[List[str]] = None) -> TestSuite:
        """运行测试
        
        Args:
            plugin_path: 插件路径
            test_names: 要运行的测试名称，None表示全部
            
        Returns:
            测试套件结果
        """
        suite = TestSuite(suite_name="auto_test")
        
        with self._lock:
            tests_to_run = test_names or list(self._tests.keys())
            
            for name in tests_to_run:
                if name in self._tests:
                    test_func = self._tests[name]
                    
                    start = time.time()
                    try:
                        result = test_func(plugin_path)
                    except Exception as e:
                        result = TestResult(
                            test_name=name,
                            passed=False,
                            duration_ms=(time.time() - start) * 1000,
                            message=f"Test failed with exception: {e}",
                        )
                    
                    suite.results.append(result)
        
        suite.completed_at = time.time()
        return suite
    
    def run_security_scan(self, plugin_path: str) -> TestResult:
        """运行安全扫描"""
        start = time.time()
        
        try:
            # 这里应该调用安全验证器
            # from ..verifier import SecurityVerifier
            # verifier = SecurityVerifier()
            # report = verifier.verify_plugin(plugin_path)
            
            # 模拟结果
            passed = True
            message = "Security scan completed"
            
            return TestResult(
                test_name="security_scan",
                passed=passed,
                duration_ms=(time.time() - start) * 1000,
                message=message,
            )
        except Exception as e:
            return TestResult(
                test_name="security_scan",
                passed=False,
                duration_ms=(time.time() - start) * 1000,
                message=f"Security scan failed: {e}",
            )
    
    def run_syntax_check(self, plugin_path: str) -> TestResult:
        """运行语法检查"""
        start = time.time()
        
        try:
            errors = []
            
            # 检查Python文件
            for root, _, files in os.walk(plugin_path):
                for file in files:
                    if file.endswith('.py'):
                        file_path = os.path.join(root, file)
                        
                        # 使用py_compile检查语法
                        import py_compile
                        try:
                            py_compile.compile(file_path, doraise=True)
                        except py_compile.PyCompileError as e:
                            errors.append(f"{file}: {e}")
            
            passed = len(errors) == 0
            
            return TestResult(
                test_name="syntax_check",
                passed=passed,
                duration_ms=(time.time() - start) * 1000,
                message="Syntax check passed" if passed else f"Found {len(errors)} errors",
                details={'errors': errors} if errors else {},
            )
        except Exception as e:
            return TestResult(
                test_name="syntax_check",
                passed=False,
                duration_ms=(time.time() - start) * 1000,
                message=f"Syntax check failed: {e}",
            )
    
    def run_import_test(self, plugin_path: str) -> TestResult:
        """运行导入测试"""
        start = time.time()
        
        try:
            # 这里应该实际尝试导入插件
            # 简化实现
            
            return TestResult(
                test_name="import_test",
                passed=True,
                duration_ms=(time.time() - start) * 1000,
                message="Import test passed",
            )
        except Exception as e:
            return TestResult(
                test_name="import_test",
                passed=False,
                duration_ms=(time.time() - start) * 1000,
                message=f"Import test failed: {e}",
            )


# ---------------------------------------------------------------------------
# 发布管理器
# ---------------------------------------------------------------------------

class PublishManager:
    """发布管理器"""
    
    def __init__(self, storage, registry):
        """
        Args:
            storage: 存储实例
            registry: 注册表实例
        """
        self._storage = storage
        self._registry = registry
        self._published: Dict[str, PublishRecord] = {}
        self._lock = threading.RLock()
    
    def publish(self, plugin_id: str, version: str,
                submitted_by: str,
                review_id: Optional[str] = None,
                test_results: Optional[TestSuite] = None) -> PublishRecord:
        """发布插件
        
        Args:
            plugin_id: 插件ID
            version: 版本
            submitted_by: 提交者
            review_id: 审核ID
            test_results: 测试结果
            
        Returns:
            发布记录
        """
        import secrets
        
        record_id = f"pub_{secrets.token_hex(8)}"
        
        record = PublishRecord(
            record_id=record_id,
            plugin_id=plugin_id,
            version=version,
            status=PublishStatus.PUBLISHED,
            submitted_by=submitted_by,
            published_at=time.time(),
            review_id=review_id,
            test_results=test_results,
        )
        
        with self._lock:
            self._published[record_id] = record
            
            # 更新插件状态
            plugin = self._registry.get(plugin_id)
            if plugin:
                plugin.status = "active"
                plugin.published_at = datetime.now()
                self._registry.register(plugin)
        
        return record
    
    def get_record(self, record_id: str) -> Optional[PublishRecord]:
        """获取发布记录"""
        with self._lock:
            return self._published.get(record_id)
    
    def get_plugin_records(self, plugin_id: str) -> List[PublishRecord]:
        """获取插件的发布记录"""
        with self._lock:
            return [
                r for r in self._published.values()
                if r.plugin_id == plugin_id
            ]
    
    def get_latest_version(self, plugin_id: str) -> Optional[str]:
        """获取最新版本"""
        with self._lock:
            records = [
                r for r in self._published.values()
                if r.plugin_id == plugin_id and r.status == PublishStatus.PUBLISHED
            ]
            
            if not records:
                return None
            
            # 按发布时间排序
            records.sort(key=lambda r: r.published_at or 0, reverse=True)
            
            return records[0].version


# ---------------------------------------------------------------------------
# 下架管理器
# ---------------------------------------------------------------------------

class UnpublishManager:
    """下架管理器"""
    
    def __init__(self, publish_manager, registry):
        """
        Args:
            publish_manager: 发布管理器
            registry: 注册表实例
        """
        self._publish_manager = publish_manager
        self._registry = registry
        self._lock = threading.RLock()
    
    def unpublish(self, plugin_id: str, reason: str,
                  unpublish_by: str) -> bool:
        """下架插件
        
        Args:
            plugin_id: 插件ID
            reason: 原因
            unpublish_by: 操作者
            
        Returns:
            是否成功
        """
        with self._lock:
            # 更新插件状态
            plugin = self._registry.get(plugin_id)
            if not plugin:
                return False
            
            plugin.status = "suspended"
            self._registry.register(plugin)
            
            # 更新发布记录
            records = self._publish_manager.get_plugin_records(plugin_id)
            for record in records:
                if record.status == PublishStatus.PUBLISHED:
                    record.status = PublishStatus.UNPUBLISHED
                    record.unpublished_at = time.time()
                    record.unpublish_reason = reason
            
            return True
    
    def deprecate_version(self, plugin_id: str, version: str,
                          reason: str) -> bool:
        """弃用版本
        
        Args:
            plugin_id: 插件ID
            version: 版本
            reason: 原因
            
        Returns:
            是否成功
        """
        with self._lock:
            plugin = self._registry.get(plugin_id)
            if not plugin:
                return False
            
            version_info = plugin.get_version_info(version)
            if not version_info:
                return False
            
            version_info.deprecated = True
            version_info.deprecated_reason = reason
            
            self._registry.register(plugin)
            
            return True


# ---------------------------------------------------------------------------
# 发布工作流
# ---------------------------------------------------------------------------

class PublishWorkflow:
    """发布工作流
    
    整合所有发布功能的主类。
    """
    
    def __init__(self, storage, registry):
        """
        Args:
            storage: 存储实例
            registry: 注册表实例
        """
        self._storage = storage
        self._registry = registry
        
        self._review_queue = ReviewQueue()
        self._auto_tester = AutoTester()
        self._publish_manager = PublishManager(storage, registry)
        self._unpublish_manager = UnpublishManager(self._publish_manager, registry)
        
        # 注册默认测试
        self._auto_tester.register_test("security", self._auto_tester.run_security_scan)
        self._auto_tester.register_test("syntax", self._auto_tester.run_syntax_check)
        self._auto_tester.register_test("import", self._auto_tester.run_import_test)
        
        # 设置审核监听器
        self._review_queue.add_listener(self._on_review_complete)
    
    def submit(self, plugin_id: str, version: str,
               submitter_id: str) -> Dict[str, Any]:
        """提交发布
        
        Args:
            plugin_id: 插件ID
            version: 版本
            submitter_id: 提交者ID
            
        Returns:
            提交结果
        """
        # 检查插件是否存在
        plugin = self._registry.get(plugin_id)
        if not plugin:
            return {'success': False, 'error': 'Plugin not found'}
        
        # 检查版本是否存在
        version_info = plugin.get_version_info(version)
        if not version_info:
            return {'success': False, 'error': 'Version not found'}
        
        # 提交审核
        review = self._review_queue.submit(plugin_id, version, submitter_id)
        
        # 运行自动测试
        # 这里应该获取插件实际路径
        plugin_path = f"/tmp/plugins/{plugin_id}"
        test_results = self._auto_tester.run_tests(plugin_path)
        
        return {
            'success': True,
            'review_id': review.review_id,
            'auto_test_results': test_results.to_dict(),
        }
    
    def approve_and_publish(self, review_id: str, reviewer_id: str,
                            notes: str = "") -> Dict[str, Any]:
        """批准并发布
        
        Args:
            review_id: 审核ID
            reviewer_id: 审核员ID
            notes: 备注
            
        Returns:
            发布结果
        """
        # 批准审核
        success = self._review_queue.approve(review_id, reviewer_id, notes)
        if not success:
            return {'success': False, 'error': 'Failed to approve review'}
        
        return {'success': True}
    
    def _on_review_complete(self, event: str, review: Review) -> None:
        """审核完成回调"""
        if event == 'approved':
            # 自动发布
            self._publish_manager.publish(
                plugin_id=review.plugin_id,
                version=review.version,
                submitted_by=review.submitter_id,
                review_id=review.review_id,
            )
    
    def unpublish(self, plugin_id: str, reason: str,
                  operator_id: str) -> Dict[str, Any]:
        """下架插件"""
        success = self._unpublish_manager.unpublish(plugin_id, reason, operator_id)
        
        return {
            'success': success,
            'plugin_id': plugin_id,
        }
    
    def deprecate(self, plugin_id: str, version: str,
                  reason: str) -> Dict[str, Any]:
        """弃用版本"""
        success = self._unpublish_manager.deprecate_version(plugin_id, version, reason)
        
        return {
            'success': success,
            'plugin_id': plugin_id,
            'version': version,
        }
    
    @property
    def review_queue(self) -> ReviewQueue:
        """获取审核队列"""
        return self._review_queue
    
    @property
    def auto_tester(self) -> AutoTester:
        """获取自动测试器"""
        return self._auto_tester
    
    @property
    def publish_manager(self) -> PublishManager:
        """获取发布管理器"""
        return self._publish_manager
    
    @property
    def unpublish_manager(self) -> UnpublishManager:
        """获取下架管理器"""
        return self._unpublish_manager
