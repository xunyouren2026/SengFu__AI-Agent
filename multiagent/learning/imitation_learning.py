"""
模仿学习模块 - 收集专家轨迹进行行为克隆

实现模仿学习(Imitation Learning)算法，通过收集专家演示轨迹，
训练Agent模仿专家行为。支持DAgger、行为克隆等算法。
"""

from typing import Dict, List, Any, Optional, Callable, Tuple, Union
from dataclasses import dataclass, field
from collections import deque
import random
import math


@dataclass
class Trajectory:
    """轨迹数据类"""
    states: List[Any] = field(default_factory=list)
    actions: List[Any] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    next_states: List[Any] = field(default_factory=list)
    dones: List[bool] = field(default_factory=list)
    
    def __len__(self) -> int:
        return len(self.states)
    
    def append(self, state: Any, action: Any, reward: float = 0.0, 
               next_state: Any = None, done: bool = False):
        """添加一个时间步"""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.dones.append(done)
    
    def get_transitions(self) -> List[Tuple[Any, Any, float, Any, bool]]:
        """获取所有转移"""
        return list(zip(self.states, self.actions, self.rewards, 
                       self.next_states, self.dones))


@dataclass
class ExpertDemonstration:
    """专家演示数据"""
    trajectory: Trajectory
    expert_id: str
    task_id: str
    success: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


class ExpertPolicy:
    """专家策略接口"""
    
    def __init__(self, policy_fn: Optional[Callable[[Any], Any]] = None):
        self.policy_fn = policy_fn or self._default_policy
        self.demonstrations: List[ExpertDemonstration] = []
        
    def _default_policy(self, state: Any) -> Any:
        """默认策略"""
        return random.choice(['left', 'right', 'up', 'down'])
    
    def select_action(self, state: Any) -> Any:
        """选择动作"""
        return self.policy_fn(state)
    
    def generate_trajectory(self, env: Any, max_steps: int = 1000) -> Trajectory:
        """生成专家轨迹"""
        trajectory = Trajectory()
        state = env.reset() if hasattr(env, 'reset') else env
        
        for _ in range(max_steps):
            action = self.select_action(state)
            
            # 执行动作
            if hasattr(env, 'step'):
                next_state, reward, done, info = env.step(action)
            else:
                next_state = state
                reward = 0.0
                done = False
            
            trajectory.append(state, action, reward, next_state, done)
            state = next_state
            
            if done:
                break
        
        return trajectory


class ImitationLearner:
    """模仿学习器"""
    
    def __init__(self, feature_extractor: Optional[Callable[[Any], List[float]]] = None):
        self.feature_extractor = feature_extractor or self._default_features
        self.demonstrations: List[ExpertDemonstration] = []
        self.policy_params: Dict[str, float] = {}
        self.learning_rate = 0.01
        self.batch_size = 32
        
    def _default_features(self, state: Any) -> List[float]:
        """默认特征提取"""
        if isinstance(state, (list, tuple)):
            return [float(x) for x in state]
        elif isinstance(state, dict):
            return [float(v) for v in state.values() if isinstance(v, (int, float))]
        return [float(state)]
    
    def collect_demonstrations(self, expert: ExpertPolicy, env: Any, 
                               num_episodes: int = 10) -> List[ExpertDemonstration]:
        """收集专家演示"""
        new_demos = []
        
        for i in range(num_episodes):
            trajectory = expert.generate_trajectory(env)
            demo = ExpertDemonstration(
                trajectory=trajectory,
                expert_id=f"expert_{i}",
                task_id=f"task_{i}",
                success=True
            )
            new_demos.append(demo)
            self.demonstrations.append(demo)
        
        return new_demos
    
    def behavior_cloning(self, epochs: int = 100) -> Dict[str, Any]:
        """
        行为克隆训练
        
        使用监督学习从专家演示中学习策略
        """
        if not self.demonstrations:
            return {'status': 'no_data', 'loss': float('inf')}
        
        # 准备训练数据
        X, y = self._prepare_training_data()
        
        # 训练
        losses = []
        for epoch in range(epochs):
            # 随机打乱
            indices = list(range(len(X)))
            random.shuffle(indices)
            
            epoch_loss = 0.0
            for i in range(0, len(X), self.batch_size):
                batch_indices = indices[i:i+self.batch_size]
                batch_X = [X[j] for j in batch_indices]
                batch_y = [y[j] for j in batch_indices]
                
                # 计算损失并更新
                loss = self._train_step(batch_X, batch_y)
                epoch_loss += loss
            
            losses.append(epoch_loss / max(1, len(X) // self.batch_size))
        
        return {
            'status': 'success',
            'final_loss': losses[-1],
            'loss_history': losses,
            'num_demonstrations': len(self.demonstrations)
        }
    
    def _prepare_training_data(self) -> Tuple[List[List[float]], List[Any]]:
        """准备训练数据"""
        X, y = [], []
        
        for demo in self.demonstrations:
            for state, action in zip(demo.trajectory.states, demo.trajectory.actions):
                features = self.feature_extractor(state)
                X.append(features)
                y.append(action)
        
        return X, y
    
    def _train_step(self, X: List[List[float]], y: List[Any]) -> float:
        """训练步骤"""
        # 简化的线性模型训练
        loss = 0.0
        for features, action in zip(X, y):
            # 预测
            prediction = self._predict(features)
            
            # 计算损失
            if isinstance(action, (int, float)):
                loss += (prediction - float(action)) ** 2
            else:
                loss += 0.0 if prediction == action else 1.0
            
            # 更新参数
            self._update_params(features, action)
        
        return loss / len(X)
    
    def _predict(self, features: List[float]) -> Union[float, Any]:
        """预测"""
        # 线性预测
        result = sum(f * self.policy_params.get(f'w{i}', 0.1) 
                    for i, f in enumerate(features))
        return result
    
    def _update_params(self, features: List[float], target: Any):
        """更新参数"""
        for i, f in enumerate(features):
            key = f'w{i}'
            current = self.policy_params.get(key, 0.1)
            if isinstance(target, (int, float)):
                gradient = 2 * (current * f - float(target)) * f
            else:
                gradient = random.uniform(-0.1, 0.1)
            self.policy_params[key] = current - self.learning_rate * gradient
    
    def select_action(self, state: Any) -> Any:
        """使用学习到的策略选择动作"""
        features = self.feature_extractor(state)
        return self._predict(features)


class DAggerAlgorithm:
    """
    DAgger (Dataset Aggregation) 算法
    
    在线模仿学习算法，通过迭代收集学生在实际执行中遇到的状态，
    并请求专家标注，逐步改进策略。
    """
    
    def __init__(self, expert: ExpertPolicy, learner: ImitationLearner,
                 beta_schedule: Optional[Callable[[int], float]] = None):
        self.expert = expert
        self.learner = learner
        self.beta_schedule = beta_schedule or self._default_beta_schedule
        self.iteration = 0
        self.aggregated_dataset: List[Tuple[Any, Any]] = []
        
    def _default_beta_schedule(self, iteration: int) -> float:
        """默认beta衰减 schedule: beta = 0.9^iteration"""
        return 0.9 ** iteration
    
    def run_iteration(self, env: Any, num_episodes: int = 10) -> Dict[str, Any]:
        """运行一个DAgger迭代"""
        beta = self.beta_schedule(self.iteration)
        new_data = []
        
        for _ in range(num_episodes):
            trajectory = self._rollout(env, beta)
            # 为访问的状态请求专家标注
            for state in trajectory.states:
                expert_action = self.expert.select_action(state)
                new_data.append((state, expert_action))
        
        # 聚合数据集
        self.aggregated_dataset.extend(new_data)
        
        # 用聚合数据重新训练
        metrics = self._train_on_aggregated_data()
        
        self.iteration += 1
        
        return {
            'iteration': self.iteration,
            'beta': beta,
            'new_data_points': len(new_data),
            'total_data_points': len(self.aggregated_dataset),
            'training_metrics': metrics
        }
    
    def _rollout(self, env: Any, beta: float) -> Trajectory:
        """执行rollout，混合使用专家和learner策略"""
        trajectory = Trajectory()
        state = env.reset() if hasattr(env, 'reset') else env
        
        max_steps = 1000
        for _ in range(max_steps):
            # 以概率beta使用专家策略
            if random.random() < beta:
                action = self.expert.select_action(state)
            else:
                action = self.learner.select_action(state)
            
            # 执行动作
            if hasattr(env, 'step'):
                next_state, reward, done, info = env.step(action)
            else:
                next_state = state
                reward = 0.0
                done = False
            
            trajectory.append(state, action, reward, next_state, done)
            state = next_state
            
            if done:
                break
        
        return trajectory
    
    def _train_on_aggregated_data(self) -> Dict[str, Any]:
        """在聚合数据上训练"""
        # 将聚合数据转换为demonstrations格式
        self.learner.demonstrations = []
        
        # 分批处理
        batch_size = 100
        for i in range(0, len(self.aggregated_dataset), batch_size):
            batch = self.aggregated_dataset[i:i+batch_size]
            trajectory = Trajectory()
            for state, action in batch:
                trajectory.append(state, action)
            
            demo = ExpertDemonstration(
                trajectory=trajectory,
                expert_id='dagger_aggregate',
                task_id=f'batch_{i//batch_size}'
            )
            self.learner.demonstrations.append(demo)
        
        # 训练
        return self.learner.behavior_cloning(epochs=50)


class InverseImitationLearning:
    """
    逆向模仿学习 (Inverse RL)
    
    从专家演示中恢复奖励函数，然后基于恢复的奖励函数进行强化学习。
    """
    
    def __init__(self, feature_dim: int = 10):
        self.feature_dim = feature_dim
        self.reward_weights: List[float] = [0.0] * feature_dim
        self.learning_rate = 0.01
        
    def extract_features(self, state: Any, action: Any) -> List[float]:
        """提取状态-动作特征"""
        # 简化的特征提取
        if isinstance(state, (list, tuple)):
            features = list(state)[:self.feature_dim]
        elif isinstance(state, dict):
            features = [float(v) for v in list(state.values())[:self.feature_dim]]
        else:
            features = [float(state)]
        
        # 填充到固定维度
        while len(features) < self.feature_dim:
            features.append(0.0)
        
        return features[:self.feature_dim]
    
    def compute_reward(self, state: Any, action: Any) -> float:
        """计算奖励"""
        features = self.extract_features(state, action)
        return sum(w * f for w, f in zip(self.reward_weights, features))
    
    def learn_reward_function(self, expert_demos: List[ExpertDemonstration],
                             learner_trajectories: List[Trajectory]) -> Dict[str, Any]:
        """
        学习奖励函数
        
        使用最大边际方法(max-margin)区分专家行为和当前策略行为
        """
        # 计算专家特征期望
        expert_feature_expectation = self._compute_feature_expectation(expert_demos)
        
        # 计算当前策略特征期望
        learner_feature_expectation = self._compute_trajectory_features(learner_trajectories)
        
        # 更新权重以最大化边际
        for i in range(self.feature_dim):
            gradient = expert_feature_expectation[i] - learner_feature_expectation[i]
            self.reward_weights[i] += self.learning_rate * gradient
        
        # 归一化
        norm = math.sqrt(sum(w ** 2 for w in self.reward_weights))
        if norm > 0:
            self.reward_weights = [w / norm for w in self.reward_weights]
        
        return {
            'expert_feature_expectation': expert_feature_expectation,
            'learner_feature_expectation': learner_feature_expectation,
            'reward_weights': self.reward_weights.copy()
        }
    
    def _compute_feature_expectation(self, demonstrations: List[ExpertDemonstration]) -> List[float]:
        """计算特征期望"""
        total_features = [0.0] * self.feature_dim
        total_steps = 0
        
        for demo in demonstrations:
            for state, action in zip(demo.trajectory.states, demo.trajectory.actions):
                features = self.extract_features(state, action)
                for i, f in enumerate(features):
                    total_features[i] += f
                total_steps += 1
        
        if total_steps > 0:
            return [f / total_steps for f in total_features]
        return total_features
    
    def _compute_trajectory_features(self, trajectories: List[Trajectory]) -> List[float]:
        """计算轨迹特征期望"""
        demos = [ExpertDemonstration(trajectory=t, expert_id='learner', task_id='temp')
                for t in trajectories]
        return self._compute_feature_expectation(demos)


class TrajectoryAugmentation:
    """轨迹数据增强"""
    
    @staticmethod
    def add_noise(trajectory: Trajectory, noise_level: float = 0.1) -> Trajectory:
        """向状态添加噪声"""
        augmented = Trajectory()
        
        for state, action, reward, next_state, done in trajectory.get_transitions():
            if isinstance(state, (list, tuple)):
                noisy_state = [s + random.uniform(-noise_level, noise_level) for s in state]
            elif isinstance(state, dict):
                noisy_state = {k: v + random.uniform(-noise_level, noise_level) 
                              for k, v in state.items()}
            else:
                noisy_state = state
            
            augmented.append(noisy_state, action, reward, next_state, done)
        
        return augmented
    
    @staticmethod
    def subsample(trajectory: Trajectory, ratio: float = 0.5) -> Trajectory:
        """子采样轨迹"""
        subsampled = Trajectory()
        indices = sorted(random.sample(range(len(trajectory)), 
                                      int(len(trajectory) * ratio)))
        
        for i in indices:
            subsampled.append(
                trajectory.states[i],
                trajectory.actions[i],
                trajectory.rewards[i],
                trajectory.next_states[i],
                trajectory.dones[i]
            )
        
        return subsampled
