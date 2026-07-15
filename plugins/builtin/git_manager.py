"""
Git管理插件

提供仓库操作、分支管理和提交记录功能。
"""

import os
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional
import threading


@dataclass
class RepositoryInfo:
    """仓库信息"""
    path: str
    remote_url: str = ""
    current_branch: str = ""
    is_clean: bool = True
    commit_count: int = 0
    last_commit: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'path': self.path,
            'remote_url': self.remote_url,
            'current_branch': self.current_branch,
            'is_clean': self.is_clean,
            'commit_count': self.commit_count,
            'last_commit': self.last_commit.isoformat() if self.last_commit else None,
        }


@dataclass
class GitOperation:
    """Git操作"""
    command: str
    success: bool
    output: str = ""
    error: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'command': self.command,
            'success': self.success,
            'output': self.output,
            'error': self.error,
        }


class GitManagerPlugin:
    """Git管理插件
    
    提供仓库操作、分支管理和提交记录。
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Args:
            config: 配置字典
        """
        self._config = config or {}
        self._default_author = self._config.get('default_author', {
            'name': 'ClawHub',
            'email': 'clawhub@example.com',
        })
        self._lock = threading.RLock()
    
    def _run_git(self, repo_path: str, *args) -> GitOperation:
        """运行Git命令
        
        Args:
            repo_path: 仓库路径
            *args: Git参数
            
        Returns:
            操作结果
        """
        cmd = ['git'] + list(args)
        
        try:
            result = subprocess.run(
                cmd,
                cwd=repo_path,
                capture_output=True,
                text=True,
            )
            
            return GitOperation(
                command=' '.join(cmd),
                success=result.returncode == 0,
                output=result.stdout,
                error=result.stderr,
            )
        except Exception as e:
            return GitOperation(
                command=' '.join(cmd),
                success=False,
                error=str(e),
            )
    
    def clone(self, url: str, dest_path: str) -> GitOperation:
        """克隆仓库
        
        Args:
            url: 远程URL
            dest_path: 目标路径
            
        Returns:
            操作结果
        """
        return self._run_git('.', 'clone', url, dest_path)
    
    def get_info(self, repo_path: str) -> RepositoryInfo:
        """获取仓库信息
        
        Args:
            repo_path: 仓库路径
            
        Returns:
            仓库信息
        """
        info = RepositoryInfo(path=repo_path)
        
        # 获取当前分支
        result = self._run_git(repo_path, 'branch', '--show-current')
        if result.success:
            info.current_branch = result.output.strip()
        
        # 检查是否有未提交更改
        result = self._run_git(repo_path, 'status', '--porcelain')
        if result.success:
            info.is_clean = not result.output.strip()
        
        # 获取提交数量
        result = self._run_git(repo_path, 'rev-list', '--count', 'HEAD')
        if result.success:
            info.commit_count = int(result.output.strip())
        
        # 获取远程URL
        result = self._run_git(repo_path, 'remote', 'get-url', 'origin')
        if result.success:
            info.remote_url = result.output.strip()
        
        return info
    
    def commit(self, repo_path: str, message: str,
               files: Optional[List[str]] = None) -> GitOperation:
        """提交更改
        
        Args:
            repo_path: 仓库路径
            message: 提交信息
            files: 要提交的文件，None表示所有
            
        Returns:
            操作结果
        """
        # 添加文件
        if files:
            for file in files:
                self._run_git(repo_path, 'add', file)
        else:
            self._run_git(repo_path, 'add', '.')
        
        # 提交
        return self._run_git(repo_path, 'commit', '-m', message)
    
    def push(self, repo_path: str, remote: str = 'origin',
             branch: Optional[str] = None) -> GitOperation:
        """推送
        
        Args:
            repo_path: 仓库路径
            remote: 远程名称
            branch: 分支名称，None表示当前分支
            
        Returns:
            操作结果
        """
        if branch:
            return self._run_git(repo_path, 'push', remote, branch)
        else:
            return self._run_git(repo_path, 'push', remote)
    
    def pull(self, repo_path: str, remote: str = 'origin',
             branch: Optional[str] = None) -> GitOperation:
        """拉取
        
        Args:
            repo_path: 仓库路径
            remote: 远程名称
            branch: 分支名称
            
        Returns:
            操作结果
        """
        if branch:
            return self._run_git(repo_path, 'pull', remote, branch)
        else:
            return self._run_git(repo_path, 'pull', remote)
    
    def create_branch(self, repo_path: str, branch_name: str,
                      base: Optional[str] = None) -> GitOperation:
        """创建分支
        
        Args:
            repo_path: 仓库路径
            branch_name: 分支名称
            base: 基础分支
            
        Returns:
            操作结果
        """
        if base:
            return self._run_git(repo_path, 'checkout', '-b', branch_name, base)
        else:
            return self._run_git(repo_path, 'checkout', '-b', branch_name)
    
    def switch_branch(self, repo_path: str, branch_name: str) -> GitOperation:
        """切换分支
        
        Args:
            repo_path: 仓库路径
            branch_name: 分支名称
            
        Returns:
            操作结果
        """
        return self._run_git(repo_path, 'checkout', branch_name)
    
    def list_branches(self, repo_path: str) -> List[str]:
        """列出分支
        
        Args:
            repo_path: 仓库路径
            
        Returns:
            分支列表
        """
        result = self._run_git(repo_path, 'branch', '-a')
        
        if result.success:
            branches = []
            for line in result.output.strip().split('\n'):
                branch = line.strip().lstrip('* ')
                if branch:
                    branches.append(branch)
            return branches
        
        return []
    
    def get_log(self, repo_path: str, count: int = 10) -> List[Dict[str, str]]:
        """获取提交记录
        
        Args:
            repo_path: 仓库路径
            count: 记录数量
            
        Returns:
            提交记录列表
        """
        result = self._run_git(
            repo_path, 'log', f'-{count}',
            '--pretty=format:%H|%an|%ae|%ad|%s'
        )
        
        commits = []
        
        if result.success:
            for line in result.output.strip().split('\n'):
                parts = line.split('|', 4)
                if len(parts) >= 5:
                    commits.append({
                        'hash': parts[0],
                        'author': parts[1],
                        'email': parts[2],
                        'date': parts[3],
                        'message': parts[4],
                    })
        
        return commits
    
    def get_metadata(self) -> Dict[str, Any]:
        """获取插件元数据"""
        return {
            'name': 'git_manager',
            'version': '1.0.0',
            'description': 'Git management plugin with repository operations',
        }
