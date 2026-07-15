"""
AGI统一框架 - 自我博弈课程学习
实现AlphaZero风格的自我博弈、课程学习、任务难度自适应调整
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass, field
import math
from collections import deque, defaultdict
import random
from abc import ABC, abstractmethod
import copy


# ==================== 配置类 ====================

@dataclass
class SelfPlayConfig:
    """自我博弈配置"""
    # MCTS配置
    num_simulations: int = 800
    c_puct: float = 1.0  # 探索常数
    dirichlet_alpha: float = 0.3  # Dirichlet噪声参数
    dirichlet_epsilon: float = 0.25  # 噪声权重
    temperature: float = 1.0  # 温度参数
    temperature_threshold: int = 30  # 温度衰减阈值
    
    # 自我博弈配置
    num_games: int = 1000
    max_moves: int = 500
    num_parallel_games: int = 8
    
    # 训练配置
    batch_size: int = 256
    buffer_size: int = 100000
    num_epochs: int = 10
    learning_rate: float = 0.001
    
    # 课程学习配置
    curriculum_enabled: bool = True
    initial_difficulty: float = 0.1
    difficulty_increment: float = 0.05
    performance_threshold: float = 0.7
    
    # 对手池配置
    opponent_pool_size: int = 10
    save_checkpoint_interval: int = 100


# ==================== 游戏环境接口 ====================

class GameEnvironment(ABC):
    """游戏环境抽象基类"""
    
    @abstractmethod
    def reset(self) -> np.ndarray:
        """重置环境，返回初始状态"""
        pass
    
    @abstractmethod
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """执行动作，返回(新状态, 奖励, 是否结束, 信息)"""
        pass
    
    @abstractmethod
    def get_legal_actions(self) -> List[int]:
        """获取合法动作"""
        pass
    
    @abstractmethod
    def is_terminal(self) -> bool:
        """检查是否终止"""
        pass
    
    @abstractmethod
    def get_result(self) -> Optional[float]:
        """获取游戏结果 (1=胜, -1=负, 0=平, None=未结束)"""
        pass
    
    @abstractmethod
    def state_to_tensor(self) -> torch.Tensor:
        """将状态转换为张量"""
        pass
    
    @abstractmethod
    def clone(self) -> 'GameEnvironment':
        """克隆环境"""
        pass


# ==================== MCTS节点 ====================

class MCTSNode:
    """蒙特卡洛树搜索节点"""
    
    def __init__(self, state: np.ndarray, prior: float = 0.0, 
                 parent: Optional['MCTSNode'] = None):
        self.state = state
        self.prior = prior
        self.parent = parent
        
        self.children: Dict[int, 'MCTSNode'] = {}
        self.visit_count = 0
        self.total_value = 0.0
        
    @property
    def q_value(self) -> float:
        """Q值"""
        if self.visit_count == 0:
            return 0.0
        return self.total_value / self.visit_count
    
    def u_value(self, c_puct: float, total_visits: int) -> float:
        """U值 (探索项)"""
        if self.parent is None:
            return 0.0
        return c_puct * self.prior * math.sqrt(total_visits) / (1 + self.visit_count)
    
    def select_child(self, c_puct: float) -> Tuple[int, 'MCTSNode']:
        """选择最佳子节点"""
        total_visits = sum(child.visit_count for child in self.children.values())
        
        best_score = float('-inf')
        best_action = None
        best_child = None
        
        for action, child in self.children.items():
            score = child.q_value + child.u_value(c_puct, total_visits)
            if score > best_score:
                best_score = score
                best_action = action
                best_child = child
                
        return best_action, best_child
    
    def expand(self, action_probs: Dict[int, float]) -> None:
        """扩展节点"""
        for action, prob in action_probs.items():
            if action not in self.children:
                self.children[action] = MCTSNode(
                    state=None,  # 状态在backup时设置
                    prior=prob,
                    parent=self
                )
    
    def backup(self, value: float) -> None:
        """反向传播"""
        self.visit_count += 1
        self.total_value += value
        
        if self.parent is not None:
            self.parent.backup(-value)  # 对手视角取负


# ==================== MCTS搜索 ====================

class MCTS:
    """蒙特卡洛树搜索"""
    
    def __init__(self, network: nn.Module, config: SelfPlayConfig):
        self.network = network
        self.config = config
        
    def search(self, env: GameEnvironment, 
               add_noise: bool = True) -> Dict[int, float]:
        """执行MCTS搜索，返回动作概率"""
        # 创建根节点
        root_state = env.state_to_tensor().numpy()
        root = MCTSNode(root_state)
        
        # 获取初始策略
        state_tensor = env.state_to_tensor().unsqueeze(0)
        with torch.no_grad():
            policy_logits, value = self.network(state_tensor)
        
        policy = F.softmax(policy_logits.squeeze(), dim=0).numpy()
        
        # 添加Dirichlet噪声
        if add_noise:
            noise = np.random.dirichlet(
                [self.config.dirichlet_alpha] * len(policy)
            )
            policy = (1 - self.config.dirichlet_epsilon) * policy + \
                     self.config.dirichlet_epsilon * noise
        
        # 获取合法动作
        legal_actions = env.get_legal_actions()
        action_probs = {a: policy[a] for a in legal_actions}
        
        # 归一化
        total_prob = sum(action_probs.values())
        if total_prob > 0:
            action_probs = {a: p / total_prob for a, p in action_probs.items()}
        
        # 扩展根节点
        root.expand(action_probs)
        
        # 执行模拟
        for _ in range(self.config.num_simulations):
            self._simulate(env.clone(), root)
        
        # 计算访问次数分布
        visit_counts = {a: child.visit_count for a, child in root.children.items()}
        total_visits = sum(visit_counts.values())
        
        if total_visits > 0:
            probs = {a: c / total_visits for a, c in visit_counts.items()}
        else:
            probs = action_probs
            
        return probs
    
    def _simulate(self, env: GameEnvironment, node: MCTSNode) -> None:
        """执行一次模拟"""
        # 选择
        path = [node]
        current = node
        
        while current.children:
            action, current = current.select_child(self.config.c_puct)
            env.step(action)
            path.append(current)
        
        # 评估
        if env.is_terminal():
            value = env.get_result()
        else:
            state_tensor = env.state_to_tensor().unsqueeze(0)
            with torch.no_grad():
                policy_logits, value = self.network(state_tensor)
            
            policy = F.softmax(policy_logits.squeeze(), dim=0).numpy()
            value = value.item()
            
            # 扩展
            legal_actions = env.get_legal_actions()
            action_probs = {a: policy[a] for a in legal_actions}
            total_prob = sum(action_probs.values())
            if total_prob > 0:
                action_probs = {a: p / total_prob for a, p in action_probs.items()}
            
            current.expand(action_probs)
        
        # 反向传播
        current.backup(value)


# ==================== 自我博弈训练器 ====================

class SelfPlayTrainer:
    """自我博弈训练器"""
    
    def __init__(self, network: nn.Module, env_factory: Callable[[], GameEnvironment],
                 config: Optional[SelfPlayConfig] = None, device: str = 'cpu'):
        self.network = network.to(device)
        self.env_factory = env_factory
        self.config = config or SelfPlayConfig()
        self.device = device
        
        # MCTS
        self.mcts = MCTS(network, self.config)
        
        # 经验缓冲区
        self.replay_buffer: deque = deque(maxlen=self.config.buffer_size)
        
        # 对手池
        self.opponent_pool: List[nn.Module] = []
        
        # 统计
        self.game_count = 0
        self.win_count = 0
        self.loss_count = 0
        self.draw_count = 0
        
    def play_game(self, opponent: Optional[nn.Module] = None,
                  temperature: float = 1.0) -> List[Dict]:
        """执行一局自我博弈"""
        env = self.env_factory()
        history = []
        
        move_count = 0
        
        while not env.is_terminal() and move_count < self.config.max_moves:
            # 获取MCTS策略
            action_probs = self.mcts.search(env, add_noise=True)
            
            # 选择动作
            if move_count < self.config.temperature_threshold:
                # 使用温度采样
                actions = list(action_probs.keys())
                probs = np.array([action_probs[a] for a in actions])
                probs = probs ** (1 / temperature)
                probs = probs / probs.sum()
                action = np.random.choice(actions, p=probs)
            else:
                # 选择最优动作
                action = max(action_probs, key=action_probs.get)
            
            # 记录
            state = env.state_to_tensor()
            history.append({
                'state': state,
                'action_probs': action_probs,
                'action': action
            })
            
            # 执行动作
            env.step(action)
            move_count += 1
        
        # 获取结果
        result = env.get_result()
        
        # 分配价值
        for i, entry in enumerate(history):
            # 从当前玩家视角
            if (len(history) - i) % 2 == 0:
                entry['value'] = -result if result is not None else 0
            else:
                entry['value'] = result if result is not None else 0
        
        return history
    
    def collect_data(self, num_games: int) -> None:
        """收集自我博弈数据"""
        for _ in range(num_games):
            history = self.play_game()
            
            for entry in history:
                self.replay_buffer.append(entry)
            
            # 更新统计
            result = entry['value']
            if result > 0:
                self.win_count += 1
            elif result < 0:
                self.loss_count += 1
            else:
                self.draw_count += 1
            
            self.game_count += 1
    
    def train_step(self) -> Dict[str, float]:
        """执行一步训练"""
        if len(self.replay_buffer) < self.config.batch_size:
            return {'loss': 0.0}
        
        # 采样
        batch = random.sample(list(self.replay_buffer), self.config.batch_size)
        
        states = torch.stack([e['state'] for e in batch]).to(self.device)
        
        # 目标策略
        num_actions = states.size(1)  # 假设动作数等于状态维度
        target_policies = torch.zeros(len(batch), num_actions, device=self.device)
        for i, e in enumerate(batch):
            for a, p in e['action_probs'].items():
                if a < num_actions:
                    target_policies[i, a] = p
        
        target_values = torch.tensor(
            [e['value'] for e in batch], 
            dtype=torch.float32, device=self.device
        )
        
        # 前向传播
        policy_logits, values = self.network(states)
        
        # 损失
        policy_loss = F.cross_entropy(policy_logits, target_policies)
        value_loss = F.mse_loss(values.squeeze(), target_values)
        
        total_loss = policy_loss + value_loss
        
        return {
            'loss': total_loss.item(),
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item()
        }
    
    def train(self, num_iterations: int = 1000,
              callback: Optional[Callable] = None) -> Dict[str, Any]:
        """执行完整训练"""
        optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=self.config.learning_rate
        )
        
        history = {'loss': [], 'win_rate': []}
        
        for iteration in range(num_iterations):
            # 收集数据
            self.collect_data(self.config.num_games)
            
            # 训练
            for _ in range(self.config.num_epochs):
                losses = self.train_step()
                
                optimizer.zero_grad()
                # 需要重新计算以获取梯度
                if len(self.replay_buffer) >= self.config.batch_size:
                    batch = random.sample(list(self.replay_buffer), self.config.batch_size)
                    states = torch.stack([e['state'] for e in batch]).to(self.device)
                    
                    num_actions = states.size(1)
                    target_policies = torch.zeros(len(batch), num_actions, device=self.device)
                    for i, e in enumerate(batch):
                        for a, p in e['action_probs'].items():
                            if a < num_actions:
                                target_policies[i, a] = p
                    
                    target_values = torch.tensor(
                        [e['value'] for e in batch],
                        dtype=torch.float32, device=self.device
                    )
                    
                    policy_logits, values = self.network(states)
                    policy_loss = F.cross_entropy(policy_logits, target_policies)
                    value_loss = F.mse_loss(values.squeeze(), target_values)
                    total_loss = policy_loss + value_loss
                    
                    total_loss.backward()
                    optimizer.step()
            
            # 记录
            win_rate = self.win_count / max(self.game_count, 1)
            history['loss'].append(losses['loss'])
            history['win_rate'].append(win_rate)
            
            # 保存检查点
            if iteration % self.config.save_checkpoint_interval == 0:
                self._save_to_pool()
            
            if callback:
                callback(iteration, {'win_rate': win_rate, **losses})
        
        return history
    
    def _save_to_pool(self) -> None:
        """保存到对手池"""
        snapshot = copy.deepcopy(self.network)
        self.opponent_pool.append(snapshot)
        
        if len(self.opponent_pool) > self.config.opponent_pool_size:
            self.opponent_pool.pop(0)
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            'game_count': self.game_count,
            'win_count': self.win_count,
            'loss_count': self.loss_count,
            'draw_count': self.draw_count,
            'win_rate': self.win_count / max(self.game_count, 1),
            'buffer_size': len(self.replay_buffer),
            'opponent_pool_size': len(self.opponent_pool)
        }


# ==================== 课程学习 ====================

class TaskDifficulty:
    """任务难度管理"""
    
    def __init__(self, initial_difficulty: float = 0.1,
                 min_difficulty: float = 0.0,
                 max_difficulty: float = 1.0):
        self.current_difficulty = initial_difficulty
        self.min_difficulty = min_difficulty
        self.max_difficulty = max_difficulty
        
        self.performance_history: deque = deque(maxlen=100)
        
    def update(self, performance: float, threshold: float = 0.7,
               increment: float = 0.05) -> float:
        """更新难度"""
        self.performance_history.append(performance)
        
        # 计算平均性能
        if len(self.performance_history) >= 10:
            avg_performance = np.mean(list(self.performance_history)[-10:])
            
            if avg_performance > threshold:
                # 性能好，增加难度
                self.current_difficulty = min(
                    self.max_difficulty,
                    self.current_difficulty + increment
                )
            elif avg_performance < threshold - 0.2:
                # 性能差，降低难度
                self.current_difficulty = max(
                    self.min_difficulty,
                    self.current_difficulty - increment * 0.5
                )
        
        return self.current_difficulty


class CurriculumScheduler:
    """课程学习调度器"""
    
    def __init__(self, tasks: List[Dict], 
                 strategy: str = "linear"):
        """
        tasks: 任务列表，每个任务包含难度等级
        strategy: 调度策略 (linear, exponential, adaptive)
        """
        self.tasks = sorted(tasks, key=lambda t: t.get('difficulty', 0))
        self.strategy = strategy
        self.current_task_idx = 0
        
        self.difficulty_manager = TaskDifficulty()
        
    def get_next_task(self, performance: float) -> Dict:
        """获取下一个任务"""
        # 更新难度
        current_difficulty = self.difficulty_manager.update(performance)
        
        # 选择任务
        if self.strategy == "linear":
            # 线性递增
            eligible_tasks = [
                t for t in self.tasks 
                if t.get('difficulty', 0) <= current_difficulty + 0.1
            ]
        elif self.strategy == "exponential":
            # 指数递增
            threshold = 0.1 * (1.5 ** self.current_task_idx)
            eligible_tasks = [
                t for t in self.tasks
                if t.get('difficulty', 0) <= threshold
            ]
        else:  # adaptive
            # 自适应
            eligible_tasks = [
                t for t in self.tasks
                if abs(t.get('difficulty', 0) - current_difficulty) < 0.2
            ]
        
        if not eligible_tasks:
            eligible_tasks = self.tasks
        
        # 随机选择
        task = random.choice(eligible_tasks)
        
        return task
    
    def get_curriculum_progress(self) -> float:
        """获取课程进度"""
        return self.difficulty_manager.current_difficulty


class CurriculumSelfPlay(SelfPlayTrainer):
    """带课程学习的自我博弈"""
    
    def __init__(self, network: nn.Module, 
                 env_factory: Callable[[float], GameEnvironment],
                 config: Optional[SelfPlayConfig] = None,
                 device: str = 'cpu'):
        super().__init__(network, lambda: env_factory(0.0), config, device)
        
        self.env_factory_with_difficulty = env_factory
        
        # 课程学习
        if config and config.curriculum_enabled:
            self.curriculum = CurriculumScheduler(
                [{'difficulty': d, 'name': f'task_{d}'} 
                 for d in np.linspace(0.1, 1.0, 10)],
                strategy="adaptive"
            )
        else:
            self.curriculum = None
            
        self.current_difficulty = config.initial_difficulty if config else 0.1
        
    def play_game_curriculum(self) -> List[Dict]:
        """执行带课程学习的游戏"""
        # 根据当前难度创建环境
        env = self.env_factory_with_difficulty(self.current_difficulty)
        
        # 执行游戏
        history = []
        move_count = 0
        
        while not env.is_terminal() and move_count < self.config.max_moves:
            action_probs = self.mcts.search(env, add_noise=True)
            
            if move_count < self.config.temperature_threshold:
                actions = list(action_probs.keys())
                probs = np.array([action_probs[a] for a in actions])
                probs = probs ** (1 / self.config.temperature)
                probs = probs / probs.sum()
                action = np.random.choice(actions, p=probs)
            else:
                action = max(action_probs, key=action_probs.get)
            
            state = env.state_to_tensor()
            history.append({
                'state': state,
                'action_probs': action_probs,
                'action': action,
                'difficulty': self.current_difficulty
            })
            
            env.step(action)
            move_count += 1
        
        result = env.get_result()
        
        for i, entry in enumerate(history):
            if (len(history) - i) % 2 == 0:
                entry['value'] = -result if result is not None else 0
            else:
                entry['value'] = result if result is not None else 0
        
        return history
    
    def train_curriculum(self, num_iterations: int = 1000,
                         callback: Optional[Callable] = None) -> Dict[str, Any]:
        """带课程学习的训练"""
        optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=self.config.learning_rate
        )
        
        history = {'loss': [], 'win_rate': [], 'difficulty': []}
        
        for iteration in range(num_iterations):
            # 收集数据
            game_history = self.play_game_curriculum()
            
            for entry in game_history:
                self.replay_buffer.append(entry)
            
            # 更新难度
            result = game_history[-1]['value'] if game_history else 0
            if result > 0:
                self.win_count += 1
            elif result < 0:
                self.loss_count += 1
            else:
                self.draw_count += 1
            self.game_count += 1
            
            # 课程更新
            if self.curriculum:
                performance = self.win_count / max(self.game_count, 1)
                task = self.curriculum.get_next_task(performance)
                self.current_difficulty = task.get('difficulty', self.current_difficulty)
            
            # 训练
            if len(self.replay_buffer) >= self.config.batch_size:
                for _ in range(self.config.num_epochs):
                    losses = self.train_step()
                    
                    # 反向传播
                    optimizer.zero_grad()
                    batch = random.sample(list(self.replay_buffer), self.config.batch_size)
                    states = torch.stack([e['state'] for e in batch]).to(self.device)
                    
                    num_actions = states.size(1)
                    target_policies = torch.zeros(len(batch), num_actions, device=self.device)
                    for i, e in enumerate(batch):
                        for a, p in e['action_probs'].items():
                            if a < num_actions:
                                target_policies[i, a] = p
                    
                    target_values = torch.tensor(
                        [e['value'] for e in batch],
                        dtype=torch.float32, device=self.device
                    )
                    
                    policy_logits, values = self.network(states)
                    policy_loss = F.cross_entropy(policy_logits, target_policies)
                    value_loss = F.mse_loss(values.squeeze(), target_values)
                    total_loss = policy_loss + value_loss
                    
                    total_loss.backward()
                    optimizer.step()
            
            # 记录
            win_rate = self.win_count / max(self.game_count, 1)
            history['loss'].append(losses.get('loss', 0))
            history['win_rate'].append(win_rate)
            history['difficulty'].append(self.current_difficulty)
            
            if callback:
                callback(iteration, {
                    'win_rate': win_rate,
                    'difficulty': self.current_difficulty,
                    **losses
                })
        
        return history


# ==================== 对抗训练 ====================

class AdversarialTraining:
    """对抗训练"""
    
    def __init__(self, player1: nn.Module, player2: nn.Module,
                 env_factory: Callable[[], GameEnvironment],
                 config: Optional[SelfPlayConfig] = None):
        self.player1 = player1
        self.player2 = player2
        self.env_factory = env_factory
        self.config = config or SelfPlayConfig()
        
        self.mcts1 = MCTS(player1, self.config)
        self.mcts2 = MCTS(player2, self.config)
        
        # 统计
        self.p1_wins = 0
        self.p2_wins = 0
        self.draws = 0
        
    def play_match(self) -> int:
        """执行一场对抗"""
        env = self.env_factory()
        
        move_count = 0
        current_player = 1
        
        while not env.is_terminal() and move_count < self.config.max_moves:
            if current_player == 1:
                action_probs = self.mcts1.search(env, add_noise=False)
            else:
                action_probs = self.mcts2.search(env, add_noise=False)
            
            action = max(action_probs, key=action_probs.get)
            env.step(action)
            
            current_player = -current_player
            move_count += 1
        
        result = env.get_result()
        
        if result is not None:
            if result > 0:
                self.p1_wins += 1
                return 1
            elif result < 0:
                self.p2_wins += 1
                return -1
            else:
                self.draws += 1
                return 0
        return 0
    
    def evaluate(self, num_games: int = 100) -> Dict[str, float]:
        """评估两个玩家"""
        self.p1_wins = 0
        self.p2_wins = 0
        self.draws = 0
        
        for _ in range(num_games):
            self.play_match()
        
        return {
            'p1_win_rate': self.p1_wins / num_games,
            'p2_win_rate': self.p2_wins / num_games,
            'draw_rate': self.draws / num_games
        }


# ==================== 联盟训练 ====================

class LeagueTraining:
    """联盟训练 (AlphaStar风格)"""
    
    def __init__(self, network_class: type, 
                 env_factory: Callable[[], GameEnvironment],
                 config: Optional[SelfPlayConfig] = None,
                 num_players: int = 3):
        self.network_class = network_class
        self.env_factory = env_factory
        self.config = config or SelfPlayConfig()
        self.num_players = num_players
        
        # 创建玩家池
        self.players: List[nn.Module] = []
        self.player_stats: List[Dict] = []
        
        for i in range(num_players):
            player = network_class()
            self.players.append(player)
            self.player_stats.append({
                'wins': 0,
                'losses': 0,
                'games': 0
            })
        
        # 主玩家
        self.main_player_idx = 0
        
    def get_matchup(self) -> Tuple[int, int]:
        """获取对战配对"""
        # 主玩家 vs 历史玩家
        opponent_idx = random.choice([
            i for i in range(self.num_players) if i != self.main_player_idx
        ])
        return self.main_player_idx, opponent_idx
    
    def train_league(self, num_iterations: int = 1000) -> Dict[str, Any]:
        """联盟训练"""
        results = []
        
        for iteration in range(num_iterations):
            # 获取配对
            p1_idx, p2_idx = self.get_matchup()
            
            # 对抗
            adversarial = AdversarialTraining(
                self.players[p1_idx],
                self.players[p2_idx],
                self.env_factory,
                self.config
            )
            
            result = adversarial.play_match()
            
            # 更新统计
            self.player_stats[p1_idx]['games'] += 1
            self.player_stats[p2_idx]['games'] += 1
            
            if result > 0:
                self.player_stats[p1_idx]['wins'] += 1
                self.player_stats[p2_idx]['losses'] += 1
            elif result < 0:
                self.player_stats[p1_idx]['losses'] += 1
                self.player_stats[p2_idx]['wins'] += 1
            
            results.append(result)
            
            # 定期更新主玩家快照
            if iteration % 100 == 0:
                snapshot_idx = (self.main_player_idx + 1) % self.num_players
                self.players[snapshot_idx] = copy.deepcopy(self.players[self.main_player_idx])
        
        return {
            'results': results,
            'player_stats': self.player_stats
        }
