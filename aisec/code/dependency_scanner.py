"""
依赖漏洞扫描 - 检测依赖包中的已知漏洞
"""
import re
import json
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


class VulnerabilitySeverity(Enum):
    """漏洞严重程度"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


@dataclass
class Vulnerability:
    """漏洞信息"""
    cve_id: str
    package_name: str
    affected_versions: str
    fixed_versions: str
    severity: VulnerabilitySeverity
    description: str
    references: List[str] = field(default_factory=list)
    cvss_score: Optional[float] = None
    exploit_available: bool = False
    publish_date: Optional[str] = None


@dataclass
class Dependency:
    """依赖信息"""
    name: str
    version: str
    source: str  # pip, npm, maven, etc.
    direct: bool = True  # 是否为直接依赖
    dependencies: List[str] = field(default_factory=list)


@dataclass
class ScanResult:
    """扫描结果"""
    dependencies: List[Dependency]
    vulnerabilities: List[Vulnerability]
    vulnerable_packages: Dict[str, List[Vulnerability]] = field(default_factory=dict)


class DependencyScanner:
    """依赖漏洞扫描器"""
    
    def __init__(self):
        self._vulnerability_db = self._load_vulnerability_db()
        self._custom_vulnerabilities: List[Vulnerability] = []
    
    def _load_vulnerability_db(self) -> Dict[str, List[Vulnerability]]:
        """加载漏洞数据库（内置已知漏洞）"""
        # 这里包含一些常见的已知漏洞作为示例
        db: Dict[str, List[Vulnerability]] = {
            "requests": [
                Vulnerability(
                    cve_id="CVE-2018-18074",
                    package_name="requests",
                    affected_versions="<2.20.0",
                    fixed_versions=">=2.20.0",
                    severity=VulnerabilitySeverity.MEDIUM,
                    description="requests在特定情况下可能泄露凭据",
                    references=["https://nvd.nist.gov/vuln/detail/CVE-2018-18074"],
                    cvss_score=6.5
                ),
            ],
            "urllib3": [
                Vulnerability(
                    cve_id="CVE-2021-33503",
                    package_name="urllib3",
                    affected_versions="<1.26.5",
                    fixed_versions=">=1.26.5",
                    severity=VulnerabilitySeverity.HIGH,
                    description="urllib3正则表达式拒绝服务漏洞",
                    references=["https://nvd.nist.gov/vuln/detail/CVE-2021-33503"],
                    cvss_score=7.5
                ),
            ],
            "pyyaml": [
                Vulnerability(
                    cve_id="CVE-2020-14343",
                    package_name="pyyaml",
                    affected_versions="<5.4",
                    fixed_versions=">=5.4",
                    severity=VulnerabilitySeverity.CRITICAL,
                    description="PyYAML反序列化任意代码执行漏洞",
                    references=["https://nvd.nist.gov/vuln/detail/CVE-2020-14343"],
                    cvss_score=9.8,
                    exploit_available=True
                ),
            ],
            "jinja2": [
                Vulnerability(
                    cve_id="CVE-2019-8341",
                    package_name="jinja2",
                    affected_versions="<2.10.1",
                    fixed_versions=">=2.10.1",
                    severity=VulnerabilitySeverity.HIGH,
                    description="Jinja2服务端模板注入漏洞",
                    references=["https://nvd.nist.gov/vuln/detail/CVE-2019-8341"],
                    cvss_score=8.8
                ),
            ],
            "flask": [
                Vulnerability(
                    cve_id="CVE-2018-1000656",
                    package_name="flask",
                    affected_versions="<0.12.3",
                    fixed_versions=">=0.12.3",
                    severity=VulnerabilitySeverity.MEDIUM,
                    description="Flask开放重定向漏洞",
                    references=["https://nvd.nist.gov/vuln/detail/CVE-2018-1000656"],
                    cvss_score=6.1
                ),
            ],
            "django": [
                Vulnerability(
                    cve_id="CVE-2022-28346",
                    package_name="django",
                    affected_versions="<2.2.28,<3.2.13,<4.0.5",
                    fixed_versions=">=2.2.28,>=3.2.13,>=4.0.5",
                    severity=VulnerabilitySeverity.HIGH,
                    description="Django SQL注入漏洞",
                    references=["https://nvd.nist.gov/vuln/detail/CVE-2022-28346"],
                    cvss_score=8.8
                ),
            ],
            "numpy": [
                Vulnerability(
                    cve_id="CVE-2021-33430",
                    package_name="numpy",
                    affected_versions="<1.20.0",
                    fixed_versions=">=1.20.0",
                    severity=VulnerabilitySeverity.MEDIUM,
                    description="NumPy缓冲区溢出漏洞",
                    references=["https://nvd.nist.gov/vuln/detail/CVE-2021-33430"],
                    cvss_score=5.5
                ),
            ],
            "pillow": [
                Vulnerability(
                    cve_id="CVE-2022-22817",
                    package_name="pillow",
                    affected_versions="<9.0.0",
                    fixed_versions=">=9.0.0",
                    severity=VulnerabilitySeverity.HIGH,
                    description="Pillow任意代码执行漏洞",
                    references=["https://nvd.nist.gov/vuln/detail/CVE-2022-22817"],
                    cvss_score=8.8
                ),
            ],
            "cryptography": [
                Vulnerability(
                    cve_id="CVE-2020-36247",
                    package_name="cryptography",
                    affected_versions="<3.3.2",
                    fixed_versions=">=3.3.2",
                    severity=VulnerabilitySeverity.MEDIUM,
                    description="cryptography整数溢出漏洞",
                    references=["https://nvd.nist.gov/vuln/detail/CVE-2020-36247"],
                    cvss_score=5.9
                ),
            ],
            "paramiko": [
                Vulnerability(
                    cve_id="CVE-2018-16395",
                    package_name="paramiko",
                    affected_versions="<2.4.2",
                    fixed_versions=">=2.4.2",
                    severity=VulnerabilitySeverity.HIGH,
                    description="Paramiko服务器端用户枚举漏洞",
                    references=["https://nvd.nist.gov/vuln/detail/CVE-2018-16395"],
                    cvss_score=7.5
                ),
            ],
        }
        return db
    
    def parse_requirements(self, file_path: str) -> List[Dependency]:
        """解析requirements.txt"""
        dependencies = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    
                    # 解析包名和版本
                    match = re.match(r'^([a-zA-Z0-9_-]+)\s*([<>=!~]+)\s*([^\s;#]+)', line)
                    if match:
                        name = match.group(1).lower()
                        version = match.group(3)
                        dependencies.append(Dependency(
                            name=name,
                            version=version,
                            source="pip"
                        ))
        except Exception:
            pass
        
        return dependencies
    
    def parse_setup_py(self, file_path: str) -> List[Dependency]:
        """解析setup.py"""
        dependencies = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 查找install_requires
            match = re.search(r'install_requires\s*=\s*\[(.*?)\]', content, re.DOTALL)
            if match:
                requires_str = match.group(1)
                for req in re.findall(r'["\']([^"\']+)["\']', requires_str):
                    pkg_match = re.match(r'^([a-zA-Z0-9_-]+)\s*([<>=!~]+)?\s*([^\s;#]+)?', req)
                    if pkg_match:
                        dependencies.append(Dependency(
                            name=pkg_match.group(1).lower(),
                            version=pkg_match.group(3) or "*",
                            source="pip"
                        ))
        except Exception:
            pass
        
        return dependencies
    
    def parse_pyproject_toml(self, file_path: str) -> List[Dependency]:
        """解析pyproject.toml"""
        dependencies = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 简单解析dependencies部分
            deps_match = re.search(r'dependencies\s*=\s*\[(.*?)\]', content, re.DOTALL)
            if deps_match:
                deps_str = deps_match.group(1)
                for dep in re.findall(r'["\']([^"\']+)["\']', deps_str):
                    pkg_match = re.match(r'^([a-zA-Z0-9_-]+)\s*([<>=!~^]+)?\s*([^\s;#]+)?', dep)
                    if pkg_match:
                        dependencies.append(Dependency(
                            name=pkg_match.group(1).lower(),
                            version=pkg_match.group(3) or "*",
                            source="pip"
                        ))
        except Exception:
            pass
        
        return dependencies
    
    def parse_package_json(self, file_path: str) -> List[Dependency]:
        """解析package.json (Node.js)"""
        dependencies = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            for dep_type in ['dependencies', 'devDependencies']:
                if dep_type in data:
                    for name, version in data[dep_type].items():
                        dependencies.append(Dependency(
                            name=name.lower(),
                            version=version.lstrip('^~'),
                            source="npm",
                            direct=(dep_type == 'dependencies')
                        ))
        except Exception:
            pass
        
        return dependencies
    
    def parse_pom_xml(self, file_path: str) -> List[Dependency]:
        """解析pom.xml (Maven)"""
        dependencies = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 简单正则解析
            pattern = r'<dependency>.*?<groupId>([^<]+)</groupId>.*?<artifactId>([^<]+)</artifactId>.*?<version>([^<]+)</version>.*?</dependency>'
            for match in re.finditer(pattern, content, re.DOTALL):
                group_id = match.group(1)
                artifact_id = match.group(2)
                version = match.group(3)
                dependencies.append(Dependency(
                    name=f"{group_id}:{artifact_id}",
                    version=version,
                    source="maven"
                ))
        except Exception:
            pass
        
        return dependencies
    
    def check_vulnerability(self, dependency: Dependency) -> List[Vulnerability]:
        """检查依赖的漏洞"""
        vulnerabilities = []
        
        # 检查内置数据库
        if dependency.name in self._vulnerability_db:
            for vuln in self._vulnerability_db[dependency.name]:
                if self._is_affected(dependency.version, vuln.affected_versions):
                    vulnerabilities.append(vuln)
        
        # 检查自定义漏洞
        for vuln in self._custom_vulnerabilities:
            if vuln.package_name.lower() == dependency.name.lower():
                if self._is_affected(dependency.version, vuln.affected_versions):
                    vulnerabilities.append(vuln)
        
        return vulnerabilities
    
    def _is_affected(self, version: str, affected_versions: str) -> bool:
        """检查版本是否受影响"""
        if version == "*" or not version:
            return True  # 未知版本，假设受影响
        
        # 解析版本
        try:
            v_parts = [int(x) for x in re.findall(r'\d+', version)]
        except:
            return True
        
        # 检查每个受影响范围
        for condition in affected_versions.split(','):
            condition = condition.strip()
            
            if condition.startswith('<'):
                threshold = condition.lstrip('<=').strip()
                threshold_parts = [int(x) for x in re.findall(r'\d+', threshold)]
                
                if condition.startswith('<='):
                    return self._compare_versions(v_parts, threshold_parts) <= 0
                else:
                    return self._compare_versions(v_parts, threshold_parts) < 0
            
            elif condition.startswith('>'):
                threshold = condition.lstrip('>=').strip()
                threshold_parts = [int(x) for x in re.findall(r'\d+', threshold)]
                
                if condition.startswith('>='):
                    return self._compare_versions(v_parts, threshold_parts) >= 0
                else:
                    return self._compare_versions(v_parts, threshold_parts) > 0
            
            elif condition.startswith('='):
                threshold = condition.lstrip('=').strip()
                threshold_parts = [int(x) for x in re.findall(r'\d+', threshold)]
                return self._compare_versions(v_parts, threshold_parts) == 0
        
        return False
    
    def _compare_versions(self, v1: List[int], v2: List[int]) -> int:
        """比较版本号"""
        max_len = max(len(v1), len(v2))
        v1 = v1 + [0] * (max_len - len(v1))
        v2 = v2 + [0] * (max_len - len(v2))
        
        for a, b in zip(v1, v2):
            if a < b:
                return -1
            elif a > b:
                return 1
        return 0
    
    def scan_project(self, project_path: str) -> ScanResult:
        """扫描项目依赖"""
        all_dependencies: List[Dependency] = []
        path = Path(project_path)
        
        # 检查各种依赖文件
        requirements_file = path / "requirements.txt"
        if requirements_file.exists():
            all_dependencies.extend(self.parse_requirements(str(requirements_file)))
        
        setup_file = path / "setup.py"
        if setup_file.exists():
            all_dependencies.extend(self.parse_setup_py(str(setup_file)))
        
        pyproject_file = path / "pyproject.toml"
        if pyproject_file.exists():
            all_dependencies.extend(self.parse_pyproject_toml(str(pyproject_file)))
        
        package_file = path / "package.json"
        if package_file.exists():
            all_dependencies.extend(self.parse_package_json(str(package_file)))
        
        pom_file = path / "pom.xml"
        if pom_file.exists():
            all_dependencies.extend(self.parse_pom_xml(str(pom_file)))
        
        # 检查漏洞
        vulnerabilities = []
        vulnerable_packages: Dict[str, List[Vulnerability]] = {}
        
        for dep in all_dependencies:
            vulns = self.check_vulnerability(dep)
            if vulns:
                vulnerabilities.extend(vulns)
                vulnerable_packages[dep.name] = vulns
        
        return ScanResult(
            dependencies=all_dependencies,
            vulnerabilities=vulnerabilities,
            vulnerable_packages=vulnerable_packages
        )
    
    def add_custom_vulnerability(self, vulnerability: Vulnerability) -> None:
        """添加自定义漏洞"""
        self._custom_vulnerabilities.append(vulnerability)
    
    def get_summary(self, result: ScanResult) -> Dict[str, Any]:
        """获取扫描摘要"""
        severity_counts = {}
        for vuln in result.vulnerabilities:
            sev = vuln.severity.value
            severity_counts[sev] = severity_counts.get(sev, 0) + 1
        
        return {
            "total_dependencies": len(result.dependencies),
            "vulnerable_packages": len(result.vulnerable_packages),
            "total_vulnerabilities": len(result.vulnerabilities),
            "by_severity": severity_counts,
            "critical_and_high": severity_counts.get("critical", 0) + severity_counts.get("high", 0)
        }
