"""
依赖解析器模块

提供依赖图构建、冲突检测、自动解决和循环检测功能。
"""

import json
import os
import re
import threading
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Set, Tuple, Callable
from collections import defaultdict, deque


# ---------------------------------------------------------------------------
# 数据模型
# ---------------------------------------------------------------------------

@dataclass
class DependencyNode:
    """依赖节点"""
    name: str
    version: str = ""
    constraint: str = "*"
    optional: bool = False
    dependencies: List["DependencyNode"] = field(default_factory=list)
    
    def __hash__(self) -> int:
        return hash(self.name)
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, DependencyNode):
            return False
        return self.name == other.name
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'version': self.version,
            'constraint': self.constraint,
            'optional': self.optional,
            'dependencies': [d.to_dict() for d in self.dependencies],
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DependencyNode":
        node = cls(
            name=data['name'],
            version=data.get('version', ''),
            constraint=data.get('constraint', '*'),
            optional=data.get('optional', False),
        )
        node.dependencies = [cls.from_dict(d) for d in data.get('dependencies', [])]
        return node


@dataclass
class ResolutionResult:
    """解析结果"""
    success: bool
    resolved: Dict[str, str] = field(default_factory=dict)
    conflicts: List[Dict[str, Any]] = field(default_factory=list)
    missing: List[str] = field(default_factory=list)
    circular: List[List[str]] = field(default_factory=list)
    installation_order: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ConflictInfo:
    """冲突信息"""
    package: str
    required_versions: List[str]
    current_resolution: Optional[str] = None
    involved_packages: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# 版本约束解析
# ---------------------------------------------------------------------------

class VersionConstraintParser:
    """版本约束解析器"""
    
    @staticmethod
    def parse(constraint: str) -> Callable[[str], bool]:
        """解析约束字符串为验证函数
        
        Args:
            constraint: 约束字符串，如 ">=1.0.0,<2.0.0"
            
        Returns:
            验证函数
        """
        if not constraint or constraint == "*":
            return lambda v: True
        
        constraints = []
        
        # 分割多个约束
        for part in constraint.split(','):
            part = part.strip()
            if not part:
                continue
            
            # 解析操作符
            if part.startswith('>='):
                target = part[2:]
                constraints.append(lambda v, t=target: VersionConstraintParser._compare(v, t) >= 0)
            elif part.startswith('>'):
                target = part[1:]
                constraints.append(lambda v, t=target: VersionConstraintParser._compare(v, t) > 0)
            elif part.startswith('<='):
                target = part[2:]
                constraints.append(lambda v, t=target: VersionConstraintParser._compare(v, t) <= 0)
            elif part.startswith('<'):
                target = part[1:]
                constraints.append(lambda v, t=target: VersionConstraintParser._compare(v, t) < 0)
            elif part.startswith('^'):
                target = part[1:]
                constraints.append(lambda v, t=target: VersionConstraintParser._compatible(v, t))
            elif part.startswith('~'):
                target = part[1:]
                constraints.append(lambda v, t=target: VersionConstraintParser._approximate(v, t))
            elif part.startswith('='):
                target = part[1:]
                constraints.append(lambda v, t=target: v == t)
            else:
                # 默认为精确匹配
                constraints.append(lambda v, t=part: v == t)
        
        def validator(version: str) -> bool:
            return all(c(version) for c in constraints)
        
        return validator
    
    @staticmethod
    def _compare(v1: str, v2: str) -> int:
        """比较两个版本号"""
        def normalize(v: str) -> List[int]:
            # 提取数字部分
            parts = re.findall(r'\d+', v)
            return [int(p) for p in parts] + [0] * (4 - len(parts))
        
        n1 = normalize(v1)
        n2 = normalize(v2)
        
        for a, b in zip(n1, n2):
            if a != b:
                return 1 if a > b else -1
        return 0
    
    @staticmethod
    def _compatible(version: str, target: str) -> bool:
        """检查兼容版本 (^)"""
        # ^1.2.3 表示 >=1.2.3 <2.0.0
        comp = VersionConstraintParser._compare(version, target)
        if comp < 0:
            return False
        
        # 获取主版本号
        major = int(re.search(r'\d+', target).group())
        major_v = int(re.search(r'\d+', version).group())
        
        return major_v == major
    
    @staticmethod
    def _approximate(version: str, target: str) -> bool:
        """检查近似版本 (~)"""
        # ~1.2.3 表示 >=1.2.3 <1.3.0
        comp = VersionConstraintParser._compare(version, target)
        if comp < 0:
            return False
        
        # 获取主次版本号
        parts_target = re.findall(r'\d+', target)[:2]
        parts_version = re.findall(r'\d+', version)[:2]
        
        return parts_target == parts_version


# ---------------------------------------------------------------------------
# 依赖图构建器
# ---------------------------------------------------------------------------

class DependencyGraphBuilder:
    """依赖图构建器"""
    
    def __init__(self):
        self._nodes: Dict[str, DependencyNode] = {}
        self._edges: Dict[str, Set[str]] = defaultdict(set)  # package -> dependencies
        self._reverse_edges: Dict[str, Set[str]] = defaultdict(set)  # package -> dependents
        self._lock = threading.Lock()
    
    def add_package(self, name: str, version: str,
                    dependencies: Dict[str, str] = None) -> DependencyNode:
        """添加包到图
        
        Args:
            name: 包名
            version: 版本
            dependencies: 依赖字典 {name: constraint}
            
        Returns:
            创建的节点
        """
        with self._lock:
            node = DependencyNode(name=name, version=version)
            self._nodes[name] = node
            
            if dependencies:
                for dep_name, constraint in dependencies.items():
                    dep_node = DependencyNode(name=dep_name, constraint=constraint)
                    node.dependencies.append(dep_node)
                    self._edges[name].add(dep_name)
                    self._reverse_edges[dep_name].add(name)
            
            return node
    
    def get_node(self, name: str) -> Optional[DependencyNode]:
        """获取节点"""
        with self._lock:
            return self._nodes.get(name)
    
    def get_dependencies(self, name: str, recursive: bool = False) -> Set[str]:
        """获取依赖
        
        Args:
            name: 包名
            recursive: 是否递归获取
            
        Returns:
            依赖包名集合
        """
        with self._lock:
            if not recursive:
                return self._edges.get(name, set()).copy()
            
            visited = set()
            stack = list(self._edges.get(name, set()))
            
            while stack:
                dep = stack.pop()
                if dep in visited:
                    continue
                visited.add(dep)
                stack.extend(self._edges.get(dep, set()) - visited)
            
            return visited
    
    def get_dependents(self, name: str, recursive: bool = False) -> Set[str]:
        """获取被依赖
        
        Args:
            name: 包名
            recursive: 是否递归获取
            
        Returns:
            依赖该包的包名集合
        """
        with self._lock:
            if not recursive:
                return self._reverse_edges.get(name, set()).copy()
            
            visited = set()
            stack = list(self._reverse_edges.get(name, set()))
            
            while stack:
                dep = stack.pop()
                if dep in visited:
                    continue
                visited.add(dep)
                stack.extend(self._reverse_edges.get(dep, set()) - visited)
            
            return visited
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        with self._lock:
            return {
                'nodes': {name: node.to_dict() for name, node in self._nodes.items()},
                'edges': {k: list(v) for k, v in self._edges.items()},
            }
    
    def build_from_requirements(self, requirements: Dict[str, str]) -> "DependencyGraphBuilder":
        """从需求字典构建图
        
        Args:
            requirements: {package: constraint}
            
        Returns:
            self
        """
        for name, constraint in requirements.items():
            self.add_package(name, "", {name: constraint})
        
        return self


# ---------------------------------------------------------------------------
# 循环检测器
# ---------------------------------------------------------------------------

class CycleDetector:
    """循环依赖检测器"""
    
    def detect_cycles(self, graph: DependencyGraphBuilder) -> List[List[str]]:
        """检测图中的循环
        
        Args:
            graph: 依赖图
            
        Returns:
            循环列表，每个循环是包名列表
        """
        cycles = []
        visited = set()
        rec_stack = set()
        
        def dfs(node: str, path: List[str]) -> None:
            visited.add(node)
            rec_stack.add(node)
            path.append(node)
            
            for neighbor in graph._edges.get(node, set()):
                if neighbor not in visited:
                    dfs(neighbor, path)
                elif neighbor in rec_stack:
                    # 发现循环
                    cycle_start = path.index(neighbor)
                    cycle = path[cycle_start:] + [neighbor]
                    # 避免重复循环
                    if cycle not in cycles:
                        cycles.append(cycle)
            
            path.pop()
            rec_stack.remove(node)
        
        for node in graph._nodes:
            if node not in visited:
                dfs(node, [])
        
        return cycles
    
    def has_cycles(self, graph: DependencyGraphBuilder) -> bool:
        """检查是否存在循环"""
        return len(self.detect_cycles(graph)) > 0
    
    def find_cycles_involving(self, graph: DependencyGraphBuilder,
                              package: str) -> List[List[str]]:
        """查找涉及特定包的循环"""
        all_cycles = self.detect_cycles(graph)
        return [c for c in all_cycles if package in c]


# ---------------------------------------------------------------------------
# 冲突检测器
# ---------------------------------------------------------------------------

class ConflictDetector:
    """依赖冲突检测器"""
    
    def __init__(self):
        self._parser = VersionConstraintParser()
    
    def detect_conflicts(self, requirements: Dict[str, List[str]]) -> List[ConflictInfo]:
        """检测版本冲突
        
        Args:
            requirements: {package: [constraint1, constraint2, ...]}
            
        Returns:
            冲突信息列表
        """
        conflicts = []
        
        for package, constraints in requirements.items():
            if len(constraints) <= 1:
                continue
            
            # 检查约束是否兼容
            if not self._are_constraints_compatible(constraints):
                conflicts.append(ConflictInfo(
                    package=package,
                    required_versions=constraints,
                ))
        
        return conflicts
    
    def _are_constraints_compatible(self, constraints: List[str]) -> bool:
        """检查多个约束是否兼容"""
        # 简化实现：检查是否有重叠的版本范围
        # 实际实现可能需要更复杂的约束求解
        
        # 提取所有版本号
        versions = set()
        for c in constraints:
            versions.update(re.findall(r'\d+\.\d+(?:\.\d+)?', c))
        
        if not versions:
            return True
        
        # 检查每个版本是否满足所有约束
        validators = [self._parser.parse(c) for c in constraints]
        
        for v in versions:
            if all(vld(v) for vld in validators):
                return True
        
        return False
    
    def check_installation(self, resolved: Dict[str, str],
                           requirements: Dict[str, Dict[str, str]]) -> List[ConflictInfo]:
        """检查已解析的安装是否满足所有需求
        
        Args:
            resolved: {package: version}
            requirements: {dependent: {dependency: constraint}}
            
        Returns:
            冲突信息列表
        """
        conflicts = []
        
        for dependent, deps in requirements.items():
            for dependency, constraint in deps.items():
                if dependency not in resolved:
                    continue
                
                installed_version = resolved[dependency]
                validator = self._parser.parse(constraint)
                
                if not validator(installed_version):
                    conflicts.append(ConflictInfo(
                        package=dependency,
                        required_versions=[constraint],
                        current_resolution=installed_version,
                        involved_packages=[dependent],
                    ))
        
        return conflicts


# ---------------------------------------------------------------------------
# 自动解析器
# ---------------------------------------------------------------------------

class AutoResolver:
    """依赖自动解析器"""
    
    def __init__(self, available_packages: Dict[str, List[str]] = None):
        """
        Args:
            available_packages: 可用包及其版本 {package: [versions]}
        """
        self._available = available_packages or {}
        self._parser = VersionConstraintParser()
        self._detector = ConflictDetector()
    
    def resolve(self, requirements: Dict[str, str],
                installed: Optional[Dict[str, str]] = None) -> ResolutionResult:
        """解析依赖
        
        Args:
            requirements: 需求 {package: constraint}
            installed: 已安装的包 {package: version}
            
        Returns:
            解析结果
        """
        installed = installed or {}
        resolved = dict(installed)
        conflicts = []
        missing = []
        
        # 收集所有需求
        all_requirements = self._collect_requirements(requirements)
        
        # 尝试解析每个包
        for package, constraints in all_requirements.items():
            if package in resolved:
                # 检查已安装的版本是否满足
                if not self._satisfies_all(resolved[package], constraints):
                    # 尝试找到兼容版本
                    new_version = self._find_compatible_version(package, constraints)
                    if new_version:
                        resolved[package] = new_version
                    else:
                        conflicts.append({
                            'package': package,
                            'required': constraints,
                            'installed': resolved[package],
                        })
            else:
                # 查找可用版本
                version = self._find_compatible_version(package, constraints)
                if version:
                    resolved[package] = version
                else:
                    missing.append(package)
        
        # 检测循环
        cycles = self._detect_cycles_in_resolution(resolved, all_requirements)
        
        # 计算安装顺序
        order = self._compute_installation_order(resolved, all_requirements)
        
        return ResolutionResult(
            success=len(conflicts) == 0 and len(missing) == 0 and len(cycles) == 0,
            resolved=resolved,
            conflicts=conflicts,
            missing=missing,
            circular=cycles,
            installation_order=order,
        )
    
    def _collect_requirements(self, requirements: Dict[str, str]) -> Dict[str, List[str]]:
        """收集所有需求（包括传递依赖）"""
        all_reqs: Dict[str, List[str]] = defaultdict(list)
        
        for package, constraint in requirements.items():
            all_reqs[package].append(constraint)
            
            # 获取传递依赖
            deps = self._get_package_dependencies(package, constraint)
            for dep, dep_constraint in deps.items():
                all_reqs[dep].append(dep_constraint)
        
        return dict(all_reqs)
    
    def _get_package_dependencies(self, package: str,
                                   constraint: str) -> Dict[str, str]:
        """获取包的依赖（模拟）"""
        # 实际实现应该从包元数据中获取
        return {}
    
    def _satisfies_all(self, version: str, constraints: List[str]) -> bool:
        """检查版本是否满足所有约束"""
        for c in constraints:
            validator = self._parser.parse(c)
            if not validator(version):
                return False
        return True
    
    def _find_compatible_version(self, package: str,
                                  constraints: List[str]) -> Optional[str]:
        """查找满足所有约束的版本"""
        available = self._available.get(package, [])
        
        for version in sorted(available, reverse=True):
            if self._satisfies_all(version, constraints):
                return version
        
        return None
    
    def _detect_cycles_in_resolution(self, resolved: Dict[str, str],
                                      requirements: Dict[str, List[str]]) -> List[List[str]]:
        """检测解析结果中的循环"""
        # 构建依赖图
        graph = DependencyGraphBuilder()
        
        for package in resolved:
            deps = self._get_package_dependencies(package, resolved[package])
            graph.add_package(package, resolved[package], deps)
        
        detector = CycleDetector()
        return detector.detect_cycles(graph)
    
    def _compute_installation_order(self, resolved: Dict[str, str],
                                    requirements: Dict[str, List[str]]) -> List[str]:
        """计算安装顺序（拓扑排序）"""
        # 构建依赖图
        in_degree = {p: 0 for p in resolved}
        edges = defaultdict(set)
        
        for package in resolved:
            deps = self._get_package_dependencies(package, resolved[package])
            for dep in deps:
                if dep in resolved:
                    edges[package].add(dep)
                    in_degree[dep] += 1
        
        # Kahn算法
        queue = [p for p, d in in_degree.items() if d == 0]
        order = []
        
        while queue:
            node = queue.pop(0)
            order.append(node)
            
            for dependent, deps in edges.items():
                if node in deps:
                    in_degree[dependent] -= 1
                    if in_degree[dependent] == 0:
                        queue.append(dependent)
        
        # 添加无依赖的包
        for package in resolved:
            if package not in order:
                order.append(package)
        
        return order


# ---------------------------------------------------------------------------
# 依赖解析器
# ---------------------------------------------------------------------------

class DependencyResolver:
    """依赖解析器
    
    整合所有依赖解析功能的主类。
    """
    
    def __init__(self, storage_path: Optional[str] = None):
        """
        Args:
            storage_path: 存储路径
        """
        self._storage_path = storage_path or os.path.join(
            os.path.expanduser("~"), ".clawhub", "dependencies"
        )
        
        self._graph_builder = DependencyGraphBuilder()
        self._cycle_detector = CycleDetector()
        self._conflict_detector = ConflictDetector()
        self._auto_resolver = AutoResolver()
        
        self._resolution_cache: Dict[str, ResolutionResult] = {}
        self._lock = threading.Lock()
        
        os.makedirs(self._storage_path, exist_ok=True)
    
    def add_package_info(self, name: str, version: str,
                         dependencies: Dict[str, str] = None) -> None:
        """添加包信息"""
        self._graph_builder.add_package(name, version, dependencies)
    
    def resolve(self, requirements: Dict[str, str],
                installed: Optional[Dict[str, str]] = None,
                use_cache: bool = True) -> ResolutionResult:
        """解析依赖
        
        Args:
            requirements: 需求 {package: constraint}
            installed: 已安装的包
            use_cache: 是否使用缓存
            
        Returns:
            解析结果
        """
        cache_key = json.dumps({'reqs': requirements, 'inst': installed}, sort_keys=True)
        
        if use_cache:
            with self._lock:
                if cache_key in self._resolution_cache:
                    return self._resolution_cache[cache_key]
        
        result = self._auto_resolver.resolve(requirements, installed)
        
        if use_cache:
            with self._lock:
                self._resolution_cache[cache_key] = result
        
        return result
    
    def check_cycles(self) -> List[List[str]]:
        """检查循环依赖"""
        return self._cycle_detector.detect_cycles(self._graph_builder)
    
    def check_conflicts(self, requirements: Dict[str, List[str]]) -> List[ConflictInfo]:
        """检查版本冲突"""
        return self._conflict_detector.detect_conflicts(requirements)
    
    def get_dependency_tree(self, package: str,
                            max_depth: int = 10) -> Optional[DependencyNode]:
        """获取依赖树
        
        Args:
            package: 包名
            max_depth: 最大深度
            
        Returns:
            依赖树根节点
        """
        node = self._graph_builder.get_node(package)
        if not node:
            return None
        
        # 深拷贝并限制深度
        return self._copy_tree_limited(node, max_depth, set())
    
    def _copy_tree_limited(self, node: DependencyNode, max_depth: int,
                           visited: Set[str]) -> DependencyNode:
        """复制树并限制深度"""
        if max_depth <= 0 or node.name in visited:
            return DependencyNode(name=node.name, version=node.version)
        
        visited.add(node.name)
        
        copy_node = DependencyNode(
            name=node.name,
            version=node.version,
            constraint=node.constraint,
            optional=node.optional,
        )
        
        for dep in node.dependencies:
            copy_node.dependencies.append(
                self._copy_tree_limited(dep, max_depth - 1, visited.copy())
            )
        
        return copy_node
    
    def get_dependents(self, package: str, recursive: bool = False) -> Set[str]:
        """获取依赖该包的包"""
        return self._graph_builder.get_dependents(package, recursive)
    
    def validate_installation(self, installed: Dict[str, str]) -> List[str]:
        """验证安装是否完整
        
        Args:
            installed: 已安装的包 {package: version}
            
        Returns:
            问题列表
        """
        issues = []
        
        for package, version in installed.items():
            node = self._graph_builder.get_node(package)
            if not node:
                continue
            
            for dep in node.dependencies:
                if dep.name not in installed:
                    issues.append(f"Missing dependency: {package} requires {dep.name}")
                else:
                    validator = self._parser.parse(dep.constraint)
                    if not validator(installed[dep.name]):
                        issues.append(
                            f"Version mismatch: {package} requires {dep.name} {dep.constraint}, "
                            f"but {installed[dep.name]} is installed"
                        )
        
        return issues
    
    def suggest_fixes(self, conflicts: List[ConflictInfo]) -> List[str]:
        """建议修复方案
        
        Args:
            conflicts: 冲突信息列表
            
        Returns:
            建议列表
        """
        suggestions = []
        
        for conflict in conflicts:
            pkg = conflict.package
            versions = conflict.required_versions
            
            suggestions.append(f"For package '{pkg}':")
            
            if len(versions) == 2:
                suggestions.append(f"  - Consider upgrading all packages to use compatible versions")
                suggestions.append(f"  - Or use a version that satisfies both: {versions[0]} and {versions[1]}")
            else:
                suggestions.append(f"  - Multiple version constraints found: {versions}")
                suggestions.append(f"  - Try to find a common compatible version")
        
        return suggestions
    
    def export_graph(self, format: str = "json") -> str:
        """导出依赖图
        
        Args:
            format: 导出格式 (json, dot)
            
        Returns:
            导出的字符串
        """
        if format == "json":
            return json.dumps(self._graph_builder.to_dict(), indent=2)
        
        elif format == "dot":
            lines = ["digraph dependencies {"]
            
            for package, node in self._graph_builder._nodes.items():
                for dep in node.dependencies:
                    lines.append(f'    "{package}" -> "{dep.name}";')
            
            lines.append("}")
            return "\n".join(lines)
        
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            'total_packages': len(self._graph_builder._nodes),
            'total_dependencies': sum(
                len(node.dependencies)
                for node in self._graph_builder._nodes.values()
            ),
            'packages_with_most_dependents': sorted(
                [(p, len(deps)) for p, deps in self._graph_builder._reverse_edges.items()],
                key=lambda x: x[1],
                reverse=True,
            )[:10],
        }
    
    def clear_cache(self) -> None:
        """清除缓存"""
        with self._lock:
            self._resolution_cache.clear()


# 导出别名
ResolutionResult = ResolutionResult
DependencyNode = DependencyNode
ConflictInfo = ConflictInfo
