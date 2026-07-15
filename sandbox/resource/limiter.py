"""
资源限制器
设置cgroup限制，控制进程资源使用
"""

import os
import subprocess
import time
import signal
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum


class CgroupVersion(Enum):
    """Cgroup版本"""
    V1 = "v1"
    V2 = "v2"
    UNIFIED = "unified"


@dataclass
class CgroupConfig:
    """Cgroup配置"""
    name: str
    cpu_quota: Optional[int] = None      # CPU配额（微秒）
    cpu_period: Optional[int] = None     # CPU周期（微秒）
    cpu_shares: Optional[int] = None     # CPU权重
    memory_limit: Optional[int] = None   # 内存限制（字节）
    memory_swap_limit: Optional[int] = None  # Swap限制（字节）
    io_read_limit: Optional[int] = None  # IO读取限制（字节/秒）
    io_write_limit: Optional[int] = None # IO写入限制（字节/秒）
    pids_limit: Optional[int] = None     # 进程数限制
    cpuset_cpus: Optional[str] = None    # CPU核心绑定
    cpuset_mems: Optional[str] = None    # 内存节点绑定


class CgroupController:
    """
    Cgroup控制器
    管理cgroup的创建、配置和销毁
    """
    
    def __init__(self, base_path: str = "/sys/fs/cgroup"):
        self.base_path = Path(base_path)
        self.version = self._detect_version()
        self._created_cgroups: List[str] = []
    
    def _detect_version(self) -> CgroupVersion:
        """检测cgroup版本"""
        # 检查是否存在cgroup.controllers文件（v2特征）
        if (self.base_path / "cgroup.controllers").exists():
            return CgroupVersion.V2
        
        # 检查是否存在子系统目录（v1特征）
        if (self.base_path / "memory").is_dir() or (self.base_path / "cpu").is_dir():
            return CgroupVersion.V1
        
        return CgroupVersion.UNIFIED
    
    def is_available(self) -> bool:
        """检查cgroup是否可用"""
        return self.base_path.exists() and self.version != CgroupVersion.UNIFIED
    
    def create_cgroup(self, config: CgroupConfig) -> bool:
        """
        创建cgroup
        
        Args:
            config: Cgroup配置
            
        Returns:
            是否成功
        """
        if self.version == CgroupVersion.V2:
            return self._create_cgroup_v2(config)
        else:
            return self._create_cgroup_v1(config)
    
    def _create_cgroup_v2(self, config: CgroupConfig) -> bool:
        """创建cgroup v2"""
        cgroup_path = self.base_path / config.name
        
        try:
            # 创建cgroup目录
            cgroup_path.mkdir(parents=True, exist_ok=True)
            
            # 启用需要的控制器
            controllers = []
            if config.cpu_quota or config.cpu_shares:
                controllers.append("cpu")
            if config.memory_limit:
                controllers.append("memory")
            if config.pids_limit:
                controllers.append("pids")
            if config.io_read_limit or config.io_write_limit:
                controllers.append("io")
            
            if controllers:
                controller_file = cgroup_path / "cgroup.subtree_control"
                # 先从父cgroup启用控制器
                parent_controllers = self.base_path / "cgroup.subtree_control"
                if parent_controllers.exists():
                    try:
                        existing = parent_controllers.read_text().strip()
                        for ctrl in controllers:
                            if f"+{ctrl}" not in existing:
                                with open(parent_controllers, "a") as f:
                                    f.write(f"+{ctrl} ")
                    except IOError:
                        pass
            
            # 设置CPU限制
            if config.cpu_quota and config.cpu_period:
                cpu_max_file = cgroup_path / "cpu.max"
                cpu_max_file.write_text(f"{config.cpu_quota} {config.cpu_period}")
            elif config.cpu_quota:
                cpu_max_file = cgroup_path / "cpu.max"
                cpu_max_file.write_text(f"{config.cpu_quota} 100000")
            
            if config.cpu_shares:
                cpu_weight_file = cgroup_path / "cpu.weight"
                # v2使用weight而不是shares，需要转换
                weight = min(10000, max(1, config.cpu_shares // 10))
                cpu_weight_file.write_text(str(weight))
            
            # 设置内存限制
            if config.memory_limit:
                memory_max_file = cgroup_path / "memory.max"
                memory_max_file.write_text(str(config.memory_limit))
            
            if config.memory_swap_limit:
                memory_swap_max_file = cgroup_path / "memory.swap.max"
                memory_swap_max_file.write_text(str(config.memory_swap_limit))
            
            # 设置进程数限制
            if config.pids_limit:
                pids_max_file = cgroup_path / "pids.max"
                pids_max_file.write_text(str(config.pids_limit))
            
            # 设置CPU核心绑定
            if config.cpuset_cpus:
                cpus_file = cgroup_path / "cpuset.cpus"
                cpus_file.write_text(config.cpuset_cpus)
            
            if config.cpuset_mems:
                mems_file = cgroup_path / "cpuset.mems"
                mems_file.write_text(config.cpuset_mems)
            
            self._created_cgroups.append(config.name)
            return True
            
        except IOError as e:
            return False
    
    def _create_cgroup_v1(self, config: CgroupConfig) -> bool:
        """创建cgroup v1"""
        try:
            # CPU cgroup
            if config.cpu_quota or config.cpu_shares:
                cpu_path = self.base_path / "cpu" / config.name
                cpu_path.mkdir(parents=True, exist_ok=True)
                
                if config.cpu_quota and config.cpu_period:
                    (cpu_path / "cpu.cfs_quota_us").write_text(str(config.cpu_quota))
                    (cpu_path / "cpu.cfs_period_us").write_text(str(config.cpu_period))
                
                if config.cpu_shares:
                    (cpu_path / "cpu.shares").write_text(str(config.cpu_shares))
            
            # CPUSET cgroup
            if config.cpuset_cpus or config.cpuset_mems:
                cpuset_path = self.base_path / "cpuset" / config.name
                cpuset_path.mkdir(parents=True, exist_ok=True)
                
                # 继承父cgroup的配置
                parent = cpuset_path.parent
                if (parent / "cpuset.cpus").exists():
                    cpus = (parent / "cpuset.cpus").read_text().strip()
                    if cpus and not config.cpuset_cpus:
                        (cpuset_path / "cpuset.cpus").write_text(cpus)
                
                if (parent / "cpuset.mems").exists():
                    mems = (parent / "cpuset.mems").read_text().strip()
                    if mems and not config.cpuset_mems:
                        (cpuset_path / "cpuset.mems").write_text(mems)
                
                if config.cpuset_cpus:
                    (cpuset_path / "cpuset.cpus").write_text(config.cpuset_cpus)
                
                if config.cpuset_mems:
                    (cpuset_path / "cpuset.mems").write_text(config.cpuset_mems)
            
            # Memory cgroup
            if config.memory_limit:
                memory_path = self.base_path / "memory" / config.name
                memory_path.mkdir(parents=True, exist_ok=True)
                (memory_path / "memory.limit_in_bytes").write_text(str(config.memory_limit))
                
                if config.memory_swap_limit:
                    (memory_path / "memory.memsw.limit_in_bytes").write_text(str(config.memory_swap_limit))
            
            # PIDs cgroup
            if config.pids_limit:
                pids_path = self.base_path / "pids" / config.name
                pids_path.mkdir(parents=True, exist_ok=True)
                (pids_path / "pids.max").write_text(str(config.pids_limit))
            
            # Block IO cgroup
            if config.io_read_limit or config.io_write_limit:
                blkio_path = self.base_path / "blkio" / config.name
                blkio_path.mkdir(parents=True, exist_ok=True)
                # blkio限制需要设备ID，这里简化处理
            
            self._created_cgroups.append(config.name)
            return True
            
        except IOError:
            return False
    
    def add_process(self, cgroup_name: str, pid: int) -> bool:
        """
        将进程添加到cgroup
        
        Args:
            cgroup_name: Cgroup名称
            pid: 进程ID
            
        Returns:
            是否成功
        """
        if self.version == CgroupVersion.V2:
            cgroup_path = self.base_path / cgroup_name / "cgroup.procs"
        else:
            # v1需要添加到所有子系统
            subsystems = ["cpu", "memory", "pids", "cpuset", "blkio"]
            success = True
            for subsystem in subsystems:
                cgroup_path = self.base_path / subsystem / cgroup_name / "tasks"
                if cgroup_path.exists():
                    try:
                        cgroup_path.write_text(str(pid))
                    except IOError:
                        success = False
            return success
        
        try:
            cgroup_path.write_text(str(pid))
            return True
        except IOError:
            return False
    
    def remove_process(self, cgroup_name: str, pid: int) -> bool:
        """
        从cgroup移除进程
        
        Args:
            cgroup_name: Cgroup名称
            pid: 进程ID
            
        Returns:
            是否成功
        """
        # 将进程移回根cgroup
        return self.add_process("", pid)
    
    def delete_cgroup(self, cgroup_name: str) -> bool:
        """
        删除cgroup
        
        Args:
            cgroup_name: Cgroup名称
            
        Returns:
            是否成功
        """
        if self.version == CgroupVersion.V2:
            cgroup_path = self.base_path / cgroup_name
        else:
            # v1需要删除所有子系统中的目录
            success = True
            for subsystem in ["cpu", "memory", "pids", "cpuset", "blkio"]:
                cgroup_path = self.base_path / subsystem / cgroup_name
                if cgroup_path.exists():
                    try:
                        cgroup_path.rmdir()
                    except IOError:
                        success = False
            if cgroup_name in self._created_cgroups:
                self._created_cgroups.remove(cgroup_name)
            return success
        
        try:
            cgroup_path.rmdir()
            if cgroup_name in self._created_cgroups:
                self._created_cgroups.remove(cgroup_name)
            return True
        except IOError:
            return False
    
    def get_stats(self, cgroup_name: str) -> Dict[str, Any]:
        """
        获取cgroup统计信息
        
        Args:
            cgroup_name: Cgroup名称
            
        Returns:
            统计信息字典
        """
        stats = {}
        
        if self.version == CgroupVersion.V2:
            cgroup_path = self.base_path / cgroup_name
            
            # CPU统计
            cpu_stat_file = cgroup_path / "cpu.stat"
            if cpu_stat_file.exists():
                cpu_stats = {}
                for line in cpu_stat_file.read_text().strip().split('\n'):
                    parts = line.split()
                    if len(parts) == 2:
                        cpu_stats[parts[0]] = int(parts[1])
                stats['cpu'] = cpu_stats
            
            # 内存统计
            memory_stat_file = cgroup_path / "memory.stat"
            if memory_stat_file.exists():
                memory_stats = {}
                for line in memory_stat_file.read_text().strip().split('\n'):
                    parts = line.split()
                    if len(parts) == 2:
                        memory_stats[parts[0]] = int(parts[1])
                stats['memory'] = memory_stats
            
            memory_current_file = cgroup_path / "memory.current"
            if memory_current_file.exists():
                stats['memory']['current'] = int(memory_current_file.read_text().strip())
            
            # PIDs统计
            pids_current_file = cgroup_path / "pids.current"
            if pids_current_file.exists():
                stats['pids'] = {'current': int(pids_current_file.read_text().strip())}
        
        else:
            # v1统计
            # CPU
            cpu_path = self.base_path / "cpu" / cgroup_name
            if cpu_path.exists():
                stats['cpu'] = {}
                cpu_stat_file = cpu_path / "cpuacct.stat"
                if cpu_stat_file.exists():
                    for line in cpu_stat_file.read_text().strip().split('\n'):
                        parts = line.split()
                        if len(parts) == 2:
                            stats['cpu'][parts[0]] = int(parts[1])
            
            # Memory
            memory_path = self.base_path / "memory" / cgroup_name
            if memory_path.exists():
                stats['memory'] = {}
                memory_stat_file = memory_path / "memory.stat"
                if memory_stat_file.exists():
                    for line in memory_stat_file.read_text().strip().split('\n'):
                        parts = line.split()
                        if len(parts) == 2:
                            stats['memory'][parts[0]] = int(parts[1])
                
                memory_usage_file = memory_path / "memory.usage_in_bytes"
                if memory_usage_file.exists():
                    stats['memory']['usage'] = int(memory_usage_file.read_text().strip())
        
        return stats
    
    def cleanup(self) -> None:
        """清理所有创建的cgroup"""
        for cgroup_name in list(self._created_cgroups):
            self.delete_cgroup(cgroup_name)


class ResourceLimiter:
    """
    资源限制器
    使用cgroup限制进程资源使用
    """
    
    def __init__(
        self,
        cgroup_base: str = "/sys/fs/cgroup",
        prefix: str = "sandbox_"
    ):
        """
        初始化资源限制器
        
        Args:
            cgroup_base: Cgroup基础路径
            prefix: Cgroup名称前缀
        """
        self.controller = CgroupController(cgroup_base)
        self.prefix = prefix
        self._active_limits: Dict[int, str] = {}  # pid -> cgroup_name
    
    def is_available(self) -> bool:
        """检查是否可用"""
        return self.controller.is_available()
    
    def apply_limits(
        self,
        pid: int,
        cpu_quota: Optional[float] = None,      # CPU核心数（如1.5表示1.5核）
        memory_limit: Optional[int] = None,     # 内存限制（字节）
        memory_swap: Optional[int] = None,      # Swap限制（字节）
        pids_limit: Optional[int] = None,       # 进程数限制
        io_read_bps: Optional[int] = None,      # IO读取限制（字节/秒）
        io_write_bps: Optional[int] = None,     # IO写入限制（字节/秒）
        cpuset_cpus: Optional[str] = None,      # CPU核心绑定
        name: Optional[str] = None              # 自定义cgroup名称
    ) -> bool:
        """
        应用资源限制
        
        Args:
            pid: 进程ID
            cpu_quota: CPU配额
            memory_limit: 内存限制
            memory_swap: Swap限制
            pids_limit: 进程数限制
            io_read_bps: IO读取限制
            io_write_bps: IO写入限制
            cpuset_cpus: CPU核心绑定
            name: 自定义名称
            
        Returns:
            是否成功
        """
        cgroup_name = name or f"{self.prefix}{pid}_{int(time.time())}"
        
        # 转换CPU配额
        cpu_quota_us = None
        cpu_period_us = 100000  # 默认100ms周期
        if cpu_quota is not None:
            cpu_quota_us = int(cpu_quota * cpu_period_us)
        
        config = CgroupConfig(
            name=cgroup_name,
            cpu_quota=cpu_quota_us,
            cpu_period=cpu_period_us,
            memory_limit=memory_limit,
            memory_swap_limit=memory_swap,
            pids_limit=pids_limit,
            io_read_limit=io_read_bps,
            io_write_limit=io_write_bps,
            cpuset_cpus=cpuset_cpus
        )
        
        if not self.controller.create_cgroup(config):
            return False
        
        if not self.controller.add_process(cgroup_name, pid):
            self.controller.delete_cgroup(cgroup_name)
            return False
        
        self._active_limits[pid] = cgroup_name
        return True
    
    def remove_limits(self, pid: int) -> bool:
        """
        移除资源限制
        
        Args:
            pid: 进程ID
            
        Returns:
            是否成功
        """
        cgroup_name = self._active_limits.pop(pid, None)
        if cgroup_name:
            self.controller.remove_process(cgroup_name, pid)
            return self.controller.delete_cgroup(cgroup_name)
        return True
    
    def get_process_stats(self, pid: int) -> Dict[str, Any]:
        """
        获取进程资源统计
        
        Args:
            pid: 进程ID
            
        Returns:
            统计信息
        """
        cgroup_name = self._active_limits.get(pid)
        if cgroup_name:
            return self.controller.get_stats(cgroup_name)
        return {}
    
    def get_all_limits(self) -> Dict[int, str]:
        """获取所有活动的限制"""
        return dict(self._active_limits)
    
    def cleanup(self) -> None:
        """清理所有资源限制"""
        for pid in list(self._active_limits.keys()):
            self.remove_limits(pid)
        self.controller.cleanup()


class ProcessResourceLimiter:
    """
    进程资源限制器
    使用Python resource模块限制当前进程资源
    """
    
    def __init__(self):
        self._original_limits: Dict[str, tuple] = {}
    
    def set_memory_limit(self, max_memory_bytes: int) -> bool:
        """
        设置内存限制
        
        Args:
            max_memory_bytes: 最大内存字节数
            
        Returns:
            是否成功
        """
        try:
            import resource
            # 保存原始限制
            self._original_limits['memory'] = resource.getrlimit(resource.RLIMIT_AS)
            # 设置新限制
            resource.setrlimit(resource.RLIMIT_AS, (max_memory_bytes, max_memory_bytes))
            return True
        except (ImportError, ValueError, resource.error):
            return False
    
    def set_cpu_time_limit(self, max_seconds: int) -> bool:
        """
        设置CPU时间限制
        
        Args:
            max_seconds: 最大CPU时间秒数
            
        Returns:
            是否成功
        """
        try:
            import resource
            self._original_limits['cpu'] = resource.getrlimit(resource.RLIMIT_CPU)
            resource.setrlimit(resource.RLIMIT_CPU, (max_seconds, max_seconds))
            return True
        except (ImportError, ValueError, resource.error):
            return False
    
    def set_file_limit(self, max_files: int) -> bool:
        """
        设置打开文件数限制
        
        Args:
            max_files: 最大文件数
            
        Returns:
            是否成功
        """
        try:
            import resource
            self._original_limits['files'] = resource.getrlimit(resource.RLIMIT_NOFILE)
            resource.setrlimit(resource.RLIMIT_NOFILE, (max_files, max_files))
            return True
        except (ImportError, ValueError, resource.error):
            return False
    
    def set_process_limit(self, max_processes: int) -> bool:
        """
        设置进程数限制
        
        Args:
            max_processes: 最大进程数
            
        Returns:
            是否成功
        """
        try:
            import resource
            self._original_limits['processes'] = resource.getrlimit(resource.RLIMIT_NPROC)
            resource.setrlimit(resource.RLIMIT_NPROC, (max_processes, max_processes))
            return True
        except (ImportError, ValueError, resource.error):
            return False
    
    def set_stack_limit(self, max_stack_bytes: int) -> bool:
        """
        设置栈大小限制
        
        Args:
            max_stack_bytes: 最大栈大小字节数
            
        Returns:
            是否成功
        """
        try:
            import resource
            self._original_limits['stack'] = resource.getrlimit(resource.RLIMIT_STACK)
            resource.setrlimit(resource.RLIMIT_STACK, (max_stack_bytes, max_stack_bytes))
            return True
        except (ImportError, ValueError, resource.error):
            return False
    
    def apply_all_limits(
        self,
        memory: Optional[int] = None,
        cpu_time: Optional[int] = None,
        files: Optional[int] = None,
        processes: Optional[int] = None,
        stack: Optional[int] = None
    ) -> Dict[str, bool]:
        """
        应用所有限制
        
        Args:
            memory: 内存限制
            cpu_time: CPU时间限制
            files: 文件数限制
            processes: 进程数限制
            stack: 栈大小限制
            
        Returns:
            各项限制是否成功的字典
        """
        results = {}
        
        if memory is not None:
            results['memory'] = self.set_memory_limit(memory)
        
        if cpu_time is not None:
            results['cpu_time'] = self.set_cpu_time_limit(cpu_time)
        
        if files is not None:
            results['files'] = self.set_file_limit(files)
        
        if processes is not None:
            results['processes'] = self.set_process_limit(processes)
        
        if stack is not None:
            results['stack'] = self.set_stack_limit(stack)
        
        return results
    
    def restore_limits(self) -> None:
        """恢复原始限制"""
        try:
            import resource
            
            if 'memory' in self._original_limits:
                resource.setrlimit(resource.RLIMIT_AS, self._original_limits['memory'])
            
            if 'cpu' in self._original_limits:
                resource.setrlimit(resource.RLIMIT_CPU, self._original_limits['cpu'])
            
            if 'files' in self._original_limits:
                resource.setrlimit(resource.RLIMIT_NOFILE, self._original_limits['files'])
            
            if 'processes' in self._original_limits:
                resource.setrlimit(resource.RLIMIT_NPROC, self._original_limits['processes'])
            
            if 'stack' in self._original_limits:
                resource.setrlimit(resource.RLIMIT_STACK, self._original_limits['stack'])
            
        except (ImportError, ValueError, resource.error):
            pass
        
        self._original_limits.clear()
    
    def __enter__(self) -> 'ProcessResourceLimiter':
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.restore_limits()
