"""
API测试插件

提供HTTP请求、响应验证和断言测试功能。
"""

import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from enum import Enum
import threading


class HTTPMethod(Enum):
    """HTTP方法"""
    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    DELETE = "DELETE"
    PATCH = "PATCH"
    HEAD = "HEAD"
    OPTIONS = "OPTIONS"


@dataclass
class APIResponse:
    """API响应"""
    status_code: int
    headers: Dict[str, str] = field(default_factory=dict)
    body: str = ""
    response_time_ms: float = 0.0
    success: bool = True
    error: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'status_code': self.status_code,
            'headers': self.headers,
            'body': self.body,
            'response_time_ms': self.response_time_ms,
            'success': self.success,
            'error': self.error,
        }


@dataclass
class APITestCase:
    """API测试用例"""
    name: str
    url: str
    method: HTTPMethod = HTTPMethod.GET
    headers: Dict[str, str] = field(default_factory=dict)
    body: Optional[str] = None
    expected_status: int = 200
    expected_body_contains: str = ""
    timeout: int = 30
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'url': self.url,
            'method': self.method.value,
            'headers': self.headers,
            'body': self.body,
            'expected_status': self.expected_status,
            'expected_body_contains': self.expected_body_contains,
            'timeout': self.timeout,
        }


class APITesterPlugin:
    """API测试插件
    
    提供HTTP请求、响应验证和断言测试。
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Args:
            config: 配置字典
        """
        self._config = config or {}
        self._default_headers = self._config.get('default_headers', {
            'User-Agent': 'ClawHub-API-Tester/1.0',
        })
        self._lock = threading.RLock()
    
    def request(self, url: str,
                method: HTTPMethod = HTTPMethod.GET,
                headers: Optional[Dict[str, str]] = None,
                body: Optional[str] = None,
                timeout: int = 30) -> APIResponse:
        """发送HTTP请求
        
        Args:
            url: URL
            method: HTTP方法
            headers: 请求头
            body: 请求体
            timeout: 超时（秒）
            
        Returns:
            API响应
        """
        import urllib.request
        import urllib.error
        
        start = time.time()
        
        # 合并请求头
        req_headers = dict(self._default_headers)
        if headers:
            req_headers.update(headers)
        
        # 创建请求
        req = urllib.request.Request(
            url=url,
            data=body.encode() if body else None,
            headers=req_headers,
            method=method.value,
        )
        
        try:
            with urllib.request.urlopen(req, timeout=timeout) as response:
                response_time = (time.time() - start) * 1000
                
                return APIResponse(
                    status_code=response.status,
                    headers=dict(response.headers),
                    body=response.read().decode('utf-8'),
                    response_time_ms=response_time,
                    success=True,
                )
        except urllib.error.HTTPError as e:
            response_time = (time.time() - start) * 1000
            
            return APIResponse(
                status_code=e.code,
                headers=dict(e.headers) if e.headers else {},
                body=e.read().decode('utf-8') if e.fp else "",
                response_time_ms=response_time,
                success=False,
                error=str(e),
            )
        except Exception as e:
            response_time = (time.time() - start) * 1000
            
            return APIResponse(
                status_code=0,
                response_time_ms=response_time,
                success=False,
                error=str(e),
            )
    
    def test(self, test_case: APITestCase) -> Dict[str, Any]:
        """执行测试用例
        
        Args:
            test_case: 测试用例
            
        Returns:
            测试结果
        """
        response = self.request(
            url=test_case.url,
            method=test_case.method,
            headers=test_case.headers,
            body=test_case.body,
            timeout=test_case.timeout,
        )
        
        # 验证
        passed = True
        failures = []
        
        if response.status_code != test_case.expected_status:
            passed = False
            failures.append(
                f"Status code mismatch: expected {test_case.expected_status}, "
                f"got {response.status_code}"
            )
        
        if test_case.expected_body_contains:
            if test_case.expected_body_contains not in response.body:
                passed = False
                failures.append(
                    f"Body does not contain expected text: {test_case.expected_body_contains}"
                )
        
        return {
            'name': test_case.name,
            'passed': passed,
            'response': response.to_dict(),
            'failures': failures,
        }
    
    def run_tests(self, test_cases: List[APITestCase]) -> Dict[str, Any]:
        """运行多个测试用例
        
        Args:
            test_cases: 测试用例列表
            
        Returns:
            测试结果汇总
        """
        results = []
        passed_count = 0
        failed_count = 0
        
        for test_case in test_cases:
            result = self.test(test_case)
            results.append(result)
            
            if result['passed']:
                passed_count += 1
            else:
                failed_count += 1
        
        return {
            'total': len(test_cases),
            'passed': passed_count,
            'failed': failed_count,
            'results': results,
        }
    
    def get_metadata(self) -> Dict[str, Any]:
        """获取插件元数据"""
        return {
            'name': 'api_tester',
            'version': '1.0.0',
            'description': 'API testing plugin with HTTP request support',
            'methods': [m.value for m in HTTPMethod],
        }
