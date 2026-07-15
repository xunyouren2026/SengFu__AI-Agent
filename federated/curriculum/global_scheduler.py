"""
全局课程调度
"""
from typing import Dict, List, Optional, Any, Tuple, Set
from datetime import datetime
from enum import Enum
import math


class DifficultyLevel(Enum):
    """难度级别"""
    VERY_EASY = 0
    EASY = 1
    MEDIUM = 2
    HARD = 3
    VERY_HARD = 4


class CurriculumStage:
    """课程阶段"""
    
    def __init__(
        self,
        stage_id: int,
        name: str,
        difficulty: DifficultyLevel,
        data_filter: Optional[Dict[str, Any]] = None
    ):
        self.stage_id = stage_id
        self.name = name
        self.difficulty = difficulty
        self.data_filter = data_filter or {}
        self.threshold: float = 0.7  # 进入下一阶段的阈值
        self.min_rounds: int = 5  # 最少训练轮数
        self.max_rounds: int = 50  # 最多训练轮数


class ClientProgress:
    """客户端进度"""
    
    def __init__(self, client_id: str):
        self.client_id = client_id
        self.current_stage: int = 0
        self.stage_scores: Dict[int, List[float]] = {}
        self.rounds_in_stage: int = 0
        self.total_rounds: int = 0
        self.completed_stages: Set[int] = set()
    
    def add_score(self, stage: int, score: float) -> None:
        """添加分数"""
        if stage not in self.stage_scores:
            self.stage_scores[stage] = []
        self.stage_scores[stage].append(score)
    
    def get_avg_score(self, stage: int) -> float:
        """获取平均分数"""
        scores = self.stage_scores.get(stage, [])
        return sum(scores) / len(scores) if scores else 0.0
    
    def can_advance(self, stage: CurriculumStage) -> bool:
        """是否可以进入下一阶段"""
        if self.rounds_in_stage < stage.min_rounds:
            return False
        
        avg_score = self.get_avg_score(stage.stage_id)
        return avg_score >= stage.threshold


class GlobalCurriculumScheduler:
    """
    全局课程调度器
    
    管理联邦学习中的课程学习策略
    """
    
    def __init__(
        self,
        num_stages: int = 5,
        auto_advance: bool = True,
        global_sync: bool = True
    ):
        self.num_stages = num_stages
        self.auto_advance = auto_advance
        self.global_sync = global_sync
        
        # 课程阶段
        self._stages: Dict[int, CurriculumStage] = {}
        self._init_stages()
        
        # 客户端进度
        self._client_progress: Dict[str, ClientProgress] = {}
        
        # 全局进度
        self._global_stage: int = 0
        self._global_round: int = 0
        
        # 统计
        self._advancement_history: List[Dict[str, Any]] = []
    
    def _init_stages(self) -> None:
        """初始化课程阶段"""
        difficulties = list(DifficultyLevel)
        
        for i in range(self.num_stages):
            diff = difficulties[min(i, len(difficulties) - 1)]
            stage = CurriculumStage(
                stage_id=i,
                name=f"Stage_{i}",
                difficulty=diff
            )
            # 根据难度调整阈值
            stage.threshold = 0.5 + 0.1 * (self.num_stages - i)
            self._stages[i] = stage
    
    def register_client(self, client_id: str) -> None:
        """注册客户端"""
        if client_id not in self._client_progress:
            self._client_progress[client_id] = ClientProgress(client_id)
    
    def unregister_client(self, client_id: str) -> None:
        """注销客户端"""
        self._client_progress.pop(client_id, None)
    
    def get_stage(self, stage_id: int) -> Optional[CurriculumStage]:
        """获取课程阶段"""
        return self._stages.get(stage_id)
    
    def get_client_stage(
        self,
        client_id: str,
        use_global: bool = False
    ) -> int:
        """
        获取客户端当前阶段
        
        Args:
            client_id: 客户端ID
            use_global: 是否使用全局阶段
        """
        if use_global and self.global_sync:
            return self._global_stage
        
        progress = self._client_progress.get(client_id)
        return progress.current_stage if progress else 0
    
    def report_progress(
        self,
        client_id: str,
        score: float,
        round_num: int
    ) -> Optional[int]:
        """
        报告进度
        
        Args:
            client_id: 客户端ID
            score: 当前分数
            round_num: 轮次
        
        Returns:
            新阶段（如果升级），否则None
        """
        if client_id not in self._client_progress:
            self.register_client(client_id)
        
        progress = self._client_progress[client_id]
        current_stage = progress.current_stage
        
        # 记录分数
        progress.add_score(current_stage, score)
        progress.rounds_in_stage += 1
        progress.total_rounds += 1
        
        # 检查是否可以升级
        if self.auto_advance and current_stage < self.num_stages - 1:
            stage = self._stages[current_stage]
            
            if progress.can_advance(stage):
                new_stage = current_stage + 1
                self._advance_client(client_id, new_stage)
                return new_stage
        
        return None
    
    def _advance_client(
        self,
        client_id: str,
        new_stage: int
    ) -> None:
        """推进客户端到新阶段"""
        progress = self._client_progress[client_id]
        old_stage = progress.current_stage
        
        progress.completed_stages.add(old_stage)
        progress.current_stage = new_stage
        progress.rounds_in_stage = 0
        
        # 记录历史
        self._advancement_history.append({
            'client_id': client_id,
            'from_stage': old_stage,
            'to_stage': new_stage,
            'timestamp': datetime.now().timestamp()
        })
    
    def advance_global(self) -> bool:
        """
        推进全局阶段
        
        当足够多客户端进入下一阶段时推进
        """
        if self._global_stage >= self.num_stages - 1:
            return False
        
        # 计算进入下一阶段的客户端比例
        next_stage = self._global_stage + 1
        clients_at_next = sum(
            1 for p in self._client_progress.values()
            if p.current_stage >= next_stage
        )
        
        total_clients = len(self._client_progress)
        if total_clients == 0:
            return False
        
        ratio = clients_at_next / total_clients
        
        # 超过50%客户端进入下一阶段
        if ratio >= 0.5:
            self._global_stage = next_stage
            return True
        
        return False
    
    def get_data_filter(
        self,
        client_id: str
    ) -> Dict[str, Any]:
        """
        获取数据过滤器
        
        根据当前阶段返回数据过滤条件
        """
        stage_id = self.get_client_stage(client_id)
        stage = self._stages.get(stage_id)
        
        if stage:
            return stage.data_filter.copy()
        
        return {}
    
    def get_stage_distribution(self) -> Dict[int, int]:
        """获取阶段分布"""
        distribution: Dict[int, int] = {}
        
        for progress in self._client_progress.values():
            stage = progress.current_stage
            distribution[stage] = distribution.get(stage, 0) + 1
        
        return distribution
    
    def get_difficulty_weights(
        self,
        client_id: str
    ) -> Dict[int, float]:
        """
        获取难度权重
        
        用于加权采样不同难度的数据
        """
        stage_id = self.get_client_stage(client_id)
        
        weights: Dict[int, float] = {}
        
        for sid, stage in self._stages.items():
            # 当前阶段权重最高，相邻阶段次之
            distance = abs(sid - stage_id)
            weights[sid] = math.exp(-distance)
        
        # 归一化
        total = sum(weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}
        
        return weights
    
    def set_stage_threshold(
        self,
        stage_id: int,
        threshold: float
    ) -> None:
        """设置阶段阈值"""
        if stage_id in self._stages:
            self._stages[stage_id].threshold = threshold
    
    def set_stage_rounds(
        self,
        stage_id: int,
        min_rounds: int,
        max_rounds: int
    ) -> None:
        """设置阶段轮数范围"""
        if stage_id in self._stages:
            self._stages[stage_id].min_rounds = min_rounds
            self._stages[stage_id].max_rounds = max_rounds
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        stage_dist = self.get_stage_distribution()
        
        avg_stage = (
            sum(s * c for s, c in stage_dist.items()) /
            sum(stage_dist.values())
            if stage_dist else 0
        )
        
        return {
            'num_stages': self.num_stages,
            'global_stage': self._global_stage,
            'total_clients': len(self._client_progress),
            'stage_distribution': stage_dist,
            'average_client_stage': avg_stage,
            'total_advancements': len(self._advancement_history),
            'auto_advance': self.auto_advance,
            'global_sync': self.global_sync
        }
