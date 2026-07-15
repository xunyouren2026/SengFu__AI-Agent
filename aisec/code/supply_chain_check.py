"""
供应链检查 - 检测供应链安全问题
"""
import re
import json
import hashlib
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


class SupplyChainRiskType(Enum):
    """供应链风险类型"""
    TYPOSQUATTING = "typosquatting"       # 仿冒包名
    DEPENDENCY_CONFUSION = "dependency_confusion"  # 依赖混淆
    MALICIOUS_PACKAGE = "malicious_package"  # 恶意包
    OUTDATED_PACKAGE = "outdated_package"   # 过时包
    UNVERIFIED_SOURCE = "unverified_source"  # 未验证来源
    SUSPICIOUS_PERMISSION = "suspicious_permission"  # 可疑权限
    PRE_INSTALL_HOOK = "pre_install_hook"   # 安装前钩子
    POST_INSTALL_HOOK = "post_install_hook"  # 安装后钩子
    NETWORK_ACCESS = "network_access"       # 网络访问
    FILE_ACCESS = "file_access"             # 文件访问


@dataclass
class SupplyChainRisk:
    """供应链风险"""
    risk_type: SupplyChainRiskType
    package_name: str
    severity: str  # critical, high, medium, low
    description: str
    evidence: str = ""
    recommendation: str = ""


@dataclass
class PackageInfo:
    """包信息"""
    name: str
    version: str
    source: str
    hash: Optional[str] = None
    author: Optional[str] = None
    homepage: Optional[str] = None
    license: Optional[str] = None
    dependencies: List[str] = field(default_factory=list)


class SupplyChainChecker:
    """供应链检查器"""
    
    def __init__(self):
        # 常见包名列表（用于检测仿冒）
        self._popular_packages = {
            'requests', 'numpy', 'pandas', 'django', 'flask', 'tensorflow',
            'torch', 'scipy', 'matplotlib', 'pillow', 'selenium', 'beautifulsoup4',
            'pytest', 'celery', 'redis', 'sqlalchemy', 'jinja2', 'pyyaml',
            'cryptography', 'paramiko', 'boto3', 'google-cloud-storage',
            'azure-storage-blob', 'openai', 'anthropic', 'langchain', 'transformers'
        }
        
        # 已知恶意包列表
        self._known_malicious = {
            'python3-dateutil': "仿冒python-dateutil的恶意包",
            'pypi-package': "已知的恶意包",
            'py-modules': "已知的恶意包",
        }
        
        # 可疑权限模式
        self._suspicious_patterns = [
            (r'eval\s*\(', "使用eval函数"),
            (r'exec\s*\(', "使用exec函数"),
            (r'__import__\s*\(', "动态导入"),
            (r'os\.system\s*\(', "系统命令执行"),
            (r'subprocess\..*shell\s*=\s*True', "shell注入风险"),
            (r'pickle\.loads?\s*\(', "不安全的反序列化"),
            (r'requests\.(get|post)\s*\([^)]*http', "网络请求"),
            (r'open\s*\([^)]*[\'\"]w[\'\"]', "文件写入"),
            (r'\.fetch\s*\(', "网络请求"),
        ]
    
    def check_typosquatting(self, package_name: str) -> Optional[SupplyChainRisk]:
        """检查仿冒包名"""
        name_lower = package_name.lower()
        
        # 检查已知恶意包
        if name_lower in self._known_malicious:
            return SupplyChainRisk(
                risk_type=SupplyChainRiskType.MALICIOUS_PACKAGE,
                package_name=package_name,
                severity="critical",
                description=self._known_malicious[name_lower],
                recommendation="立即移除此包"
            )
        
        # 检查与流行包的相似度
        for popular in self._popular_packages:
            if self._is_typosquat(name_lower, popular):
                return SupplyChainRisk(
                    risk_type=SupplyChainRiskType.TYPOSQUATTING,
                    package_name=package_name,
                    severity="high",
                    description=f"可能是'{popular}'的仿冒包",
                    evidence=f"包名'{package_name}'与'{popular}'相似",
                    recommendation=f"确认是否应使用'{popular}'"
                )
        
        return None
    
    def _is_typosquat(self, name: str, target: str) -> bool:
        """检查是否为仿冒名"""
        if name == target:
            return False
        
        # 编辑距离检查
        if len(name) == len(target):
            diff = sum(1 for a, b in zip(name, target) if a != b)
            if diff <= 2:
                return True
        
        # 常见仿冒模式
        patterns = [
            lambda n, t: n == t + 's',  # 复数形式
            lambda n, t: n == t[:-1],   # 少一个字母
            lambda n, t: n == t + '-',  # 添加连字符
            lambda n, t: n.replace('-', '') == t,  # 移除连字符
            lambda n, t: n == t + '2' or n == t + '3',  # 数字后缀
        ]
        
        for pattern in patterns:
            try:
                if pattern(name, target):
                    return True
            except:
                pass
        
        return False
    
    def check_setup_py(self, setup_content: str, package_name: str) -> List[SupplyChainRisk]:
        """检查setup.py内容"""
        risks = []
        
        # 检查可疑模式
        for pattern, desc in self._suspicious_patterns:
            if re.search(pattern, setup_content, re.IGNORECASE):
                if 'network' in desc.lower() or 'request' in desc.lower():
                    risk_type = SupplyChainRiskType.NETWORK_ACCESS
                elif 'file' in desc.lower() or 'write' in desc.lower():
                    risk_type = SupplyChainRiskType.FILE_ACCESS
                else:
                    risk_type = SupplyChainRiskType.SUSPICIOUS_PERMISSION
                
                risks.append(SupplyChainRisk(
                    risk_type=risk_type,
                    package_name=package_name,
                    severity="high",
                    description=f"setup.py中{desc}",
                    recommendation="审查setup.py内容"
                ))
        
        # 检查安装钩子
        if re.search(r'cmdclass\s*=', setup_content):
            risks.append(SupplyChainRisk(
                risk_type=SupplyChainRiskType.PRE_INSTALL_HOOK,
                package_name=package_name,
                severity="medium",
                description="setup.py定义了自定义安装命令",
                recommendation="检查自定义安装命令的安全性"
            ))
        
        return risks
    
    def check_package_json(self, content: str, package_name: str) -> List[SupplyChainRisk]:
        """检查package.json内容"""
        risks = []
        
        try:
            data = json.loads(content)
            
            # 检查pre/post install脚本
            scripts = data.get('scripts', {})
            for script_type in ['preinstall', 'postinstall', 'prepublish']:
                if script_type in scripts:
                    risks.append(SupplyChainRisk(
                        risk_type=SupplyChainRiskType.PRE_INSTALL_HOOK if 'pre' in script_type else SupplyChainRiskType.POST_INSTALL_HOOK,
                        package_name=package_name,
                        severity="high",
                        description=f"定义了{script_type}脚本",
                        evidence=scripts[script_type][:100],
                        recommendation="审查安装脚本内容"
                    ))
            
            # 检查可疑依赖
            deps = {**data.get('dependencies', {}), **data.get('devDependencies', {})}
            for dep_name in deps:
                typo_risk = self.check_typosquatting(dep_name)
                if typo_risk:
                    risks.append(typo_risk)
        
        except json.JSONDecodeError:
            pass
        
        return risks
    
    def check_lock_file(self, lock_content: str, lock_type: str) -> List[SupplyChainRisk]:
        """检查锁文件完整性"""
        risks = []
        
        if lock_type == 'requirements':
            # 检查是否有固定版本
            for line in lock_content.strip().split('\n'):
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                if not re.search(r'[<>=!]', line):
                    # 没有版本约束
                    pkg_name = re.match(r'^([a-zA-Z0-9_-]+)', line)
                    if pkg_name:
                        risks.append(SupplyChainRisk(
                            risk_type=SupplyChainRiskType.OUTDATED_PACKAGE,
                            package_name=pkg_name.group(1),
                            severity="medium",
                            description="依赖未指定版本",
                            recommendation="固定依赖版本"
                        ))
        
        elif lock_type == 'package-lock':
            try:
                data = json.loads(lock_content)
                
                # 检查integrity
                for pkg_name, pkg_data in data.get('packages', {}).items():
                    if pkg_name and 'integrity' not in pkg_data and 'resolved' in pkg_data:
                        risks.append(SupplyChainRisk(
                            risk_type=SupplyChainRiskType.UNVERIFIED_SOURCE,
                            package_name=pkg_name.split('node_modules/')[-1] if 'node_modules/' in pkg_name else pkg_name,
                            severity="medium",
                            description="包缺少完整性校验",
                            recommendation="使用package-lock.json确保完整性"
                        ))
            except json.JSONDecodeError:
                pass
        
        return risks
    
    def check_dependency_confusion(
        self,
        public_deps: List[str],
        private_deps: List[str]
    ) -> List[SupplyChainRisk]:
        """检查依赖混淆攻击"""
        risks = []
        
        for private_dep in private_deps:
            if private_dep in public_deps:
                risks.append(SupplyChainRisk(
                    risk_type=SupplyChainRiskType.DEPENDENCY_CONFUSION,
                    package_name=private_dep,
                    severity="critical",
                    description="私有包名与公共包名冲突",
                    evidence=f"'{private_dep}'同时存在于私有和公共仓库",
                    recommendation="重命名私有包或使用私有仓库优先"
                ))
        
        return risks
    
    def scan_project(self, project_path: str) -> List[SupplyChainRisk]:
        """扫描项目供应链风险"""
        risks = []
        path = Path(project_path)
        
        # 检查requirements.txt
        req_file = path / "requirements.txt"
        if req_file.exists():
            content = req_file.read_text(encoding='utf-8', errors='ignore')
            
            for line in content.strip().split('\n'):
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                match = re.match(r'^([a-zA-Z0-9_-]+)', line)
                if match:
                    pkg_name = match.group(1)
                    typo_risk = self.check_typosquatting(pkg_name)
                    if typo_risk:
                        risks.append(typo_risk)
            
            risks.extend(self.check_lock_file(content, 'requirements'))
        
        # 检查package.json
        pkg_file = path / "package.json"
        if pkg_file.exists():
            content = pkg_file.read_text(encoding='utf-8', errors='ignore')
            try:
                data = json.loads(content)
                pkg_name = data.get('name', 'unknown')
                risks.extend(self.check_package_json(content, pkg_name))
            except:
                pass
        
        # 检查package-lock.json
        lock_file = path / "package-lock.json"
        if lock_file.exists():
            content = lock_file.read_text(encoding='utf-8', errors='ignore')
            risks.extend(self.check_lock_file(content, 'package-lock'))
        
        # 检查setup.py
        setup_file = path / "setup.py"
        if setup_file.exists():
            content = setup_file.read_text(encoding='utf-8', errors='ignore')
            # 尝试提取包名
            name_match = re.search(r'name\s*=\s*["\']([^"\']+)["\']', content)
            pkg_name = name_match.group(1) if name_match else "unknown"
            risks.extend(self.check_setup_py(content, pkg_name))
        
        return risks
    
    def add_known_malicious(self, package_name: str, description: str) -> None:
        """添加已知恶意包"""
        self._known_malicious[package_name.lower()] = description
    
    def add_popular_package(self, package_name: str) -> None:
        """添加流行包（用于仿冒检测）"""
        self._popular_packages.add(package_name.lower())
    
    def get_summary(self, risks: List[SupplyChainRisk]) -> Dict[str, Any]:
        """获取风险摘要"""
        by_type = {}
        by_severity = {}
        
        for risk in risks:
            type_name = risk.risk_type.value
            by_type[type_name] = by_type.get(type_name, 0) + 1
            by_severity[risk.severity] = by_severity.get(risk.severity, 0) + 1
        
        return {
            "total_risks": len(risks),
            "by_type": by_type,
            "by_severity": by_severity,
            "critical_count": by_severity.get("critical", 0),
            "high_count": by_severity.get("high", 0)
        }
