"""
DAgger在线学习模块 - Dataset Aggregation

实现DAgger算法，通过在线收集自身执行数据并请求专家标注，
逐步改进策略，解决行为克隆中的分布偏移问题。
"""

from typing import Dict, List, Any, Optional, Callable, Tuple, Union
from dataclasses import dataclass, field
from collections import deque
import random
import math


@dataclass
class StateActionPair:
    """状态-动作对"""
    state: Any
    expert_action: Any
    learner_action: Any
    timestamp: float = 0.0
    annotated: bool = False


@dataclass
class DAggerConfig:
    """DAgger配置"""
    beta_initial: float = 1.0          # 初始混合概率
    beta_decay: float = 0.9            # beta衰减率
    min_beta: float = 0.0              # 最小beta值
    max_iterations: int = 100          # 最大迭代次数
    episodes_per_iteration: int = 10   # 每次迭代的episode数
    max_steps_per_episode: int = 1000  # 每个episode的最大步数
    batch_size: int = 32               # 训练批次大小
    learning_rate: float = 0.01        # 学习率
    aggregation_strategy: str = 'all'  # 聚合策略: 'all', 'error', 'uncertain'


class ExpertOracle:
    """
    专家预言机接口
    
    提供专家策略查询，支持人工标注和自动策略
    """
    
    def __init__(
        self,
        policy_fn: Optional[Callable[[Any], Any]] = None,
        annotation_callback: Optional[Callable[[Any], Any]] = None
    ):
        self.policy_fn = policy_fn
        self.annotation_callback = annotation_callback
        self.query_count = 0
        self.annotation_history: List[Tuple[Any, Any, float]] = []
    
    def query(self, state: Any) -> Any:
        """
        查询专家动作
        
        Args:
            state: 当前状态
            
        Returns:
            专家推荐的动作
        """
        self.query_count += 1
        
        if self.policy_fn is not None:
            action = self.policy_fn(state)
        elif self.annotation_callback is not None:
            action = self.annotation_callback(state)
        else:
            raise ValueError("必须提供policy_fn或annotation_callback")
        
        self.annotation_history.append((state, action, self.query_count))
        return action
    
    def batch_query(self, states: List[Any]) -> List[Any]:
        """批量查询专家动作"""
        return [self.query(s) for s in states]
    
    def get_annotation_cost(self) -> int:
        """获取标注成本（查询次数）"""
        return self.query_count


class LearnerPolicy:
    """
    学习者策略
    
    维护可训练的策略参数，支持在线更新
    """
    
    def __init__(
        self,
        feature_dim: int = 10,
        action_space: Optional[List[Any]] = None,
        learning_rate: float = 0.01
    ):
        self.feature_dim = feature_dim
        self.action_space = action_space or ['action_0', 'action_1']
        self.learning_rate = learning_rate
        
        # 策略参数 (简单的线性策略)
        self.weights: Dict[str, List[float]] = {}
        self._initialize_weights()
        
        # 经验缓冲
        self.buffer: deque = deque(maxlen=10000)
        
    def _initialize_weights(self) -> None:
        """初始化权重"""
        for action in self.action_space:
            self.weights[str(action)] = [random.uniform(-0.1, 0.1) 
                                         for _ in range(self.feature_dim)]
    
    def extract_features(self, state: Any) -> List[float]:
        """提取状态特征"""
        if isinstance(state, (list, tuple)):
            features = [float(x) for x in state][:self.feature_dim]
        elif isinstance(state, dict):
            features = [float(v) for v in state.values() 
                       if isinstance(v, (int, float))][:self.feature_dim]
        elif isinstance(state, (int, float)):
            features = [float(state)]
        else:
            features = [0.0]
        
        # 填充到固定维度
        while len(features) < self.feature_dim:
            features.append(0.0)
        
        return features[:self.feature_dim]
    
    def select_action(self, state: Any) -> Any:
        """选择动作"""
        features = self.extract_features(state)
        
        # 计算每个动作的得分
        scores = {}
        for action in self.action_space:
            weight = self.weights.get(str(action), [0.0] * self.feature_dim)
            score = sum(w * f for w, f in zip(weight, features))
            scores[action] = score
        
        # 选择得分最高的动作
        return max(scores, key=scores.get)
    
    def get_action_probabilities(self, state: Any) -> Dict[Any, float]:
        """获取动作概率分布（softmax）"""
        features = self.extract_features(state)
        
        scores = {}
        for action in self.action_space:
            weight = self.weights.get(str(action), [0.0] * self.feature_dim)
            score = sum(w * f for w, f in zip(weight, features))
            scores[action] = score
        
        # Softmax
        max_score = max(scores.values())
        exp_scores = {a: math.exp(s - max_score) for a, s in scores.items()}
        total = sum(exp_scores.values())
        
        return {a: e / total for a, e in exp_scores.items()}
    
    def update(self, state: Any, target_action: Any) -> float:
        """
        更新策略参数
        
        使用简单的梯度下降
        """
        features = self.extract_features(state)
        current_action = self.select_action(state)
        
        # 如果预测正确，不需要更新
        if current_action == target_action:
            return 0.0
        
        # 增加目标动作权重，减少当前动作权重
        target_key = str(target_action)
        current_key = str(current_action)
        
        for i, f in enumerate(features):
            if target_key in self.weights:
                self.weights[target_key][i] += self.learning_rate * f
            if current_key in self.weights:
                self.weights[current_key][i] -= self.learning_rate * f
        
        return 1.0  # 返回损失
    
    def batch_update(self, data: List[Tuple[Any, Any]]) -> float:
        """批量更新"""
        total_loss = 0.0
        for state, action in data:
            total_loss += self.update(state, action)
        return total_loss / len(data) if data else 0.0


class DAggerLoop:
    """
    DAgger在线学习循环
    
    实现完整的DAgger算法流程:
    1. 初始化策略（从专家演示）
    2. 迭代执行:
       - 使用混合策略收集轨迹
       - 为访问的状态请求专家标注
       - 聚合数据并重新训练
    3. 返回最终策略
    """
    
    def __init__(
        self,
        expert: ExpertOracle,
        learner: LearnerPolicy,
        config: Optional[DAggerConfig] = None
    ):
        self.expert = expert
        self.learner = learner
        self.config = config or DAggerConfig()
        
        # 聚合数据集
        self.aggregated_data: List[StateActionPair] = []
        
        # 迭代历史
        self.iteration_history: List[Dict[str, Any]] = []
        
        # 当前迭代次数
        self.current_iteration = 0
        
        # 当前beta值
        self.current_beta = self.config.beta_initial
    
    def get_beta(self, iteration: int) -> float:
        """计算当前beta值"""
        beta = self.config.beta_initial * (self.config.beta_decay ** iteration)
        return max(beta, self.config.min_beta)
    
    def mixed_policy_action(self, state: Any) -> Tuple[Any, bool]:
        """
        混合策略选择动作
        
        Returns:
            (动作, 是否为专家动作)
        """
        if random.random() < self.current_beta:
            return self.expert.query(state), True
        else:
            return self.learner.select_action(state), False
    
    def collect_episode(
        self,
        env: Any,
        max_steps: int
    ) -> Tuple[List[Any], List[Any], float]:
        """
        收集一个episode的数据
        
        Returns:
            (状态序列, 动作序列, 累计奖励)
        """
        states = []
        actions = []
        total_reward = 0.0
        
        state = env.reset() if hasattr(env, 'reset') else env
        
        for step in range(max_steps):
            states.append(state)
            action, _ = self.mixed_policy_action(state)
            actions.append(action)
            
            # 执行动作
            if hasattr(env, 'step'):
                result = env.step(action)
                if len(result) == 4:
                    next_state, reward, done, info = result
                else:
                    next_state, reward, done = result
                    info = {}
            else:
                next_state = state
                reward = 0.0
                done = False
                info = {}
            
            total_reward += reward
            state = next_state
            
            if done:
                break
        
        return states, actions, total_reward
    
    def request_annotations(self, states: List[Any]) -> List[Any]:
        """为状态请求专家标注"""
        return self.expert.batch_query(states)
    
    def aggregate_data(
        self,
        states: List[Any],
        expert_actions: List[Any]
    ) -> int:
        """聚合新数据"""
        added_count = 0
        
        for state, expert_action in zip(states, expert_actions):
            learner_action = self.learner.select_action(state)
            
            # 根据聚合策略决定是否添加
            should_add = self._should_aggregate(state, expert_action, learner_action)
            
            if should_add:
                pair = StateActionPair(
                    state=state,
                    expert_action=expert_action,
                    learner_action=learner_action,
                    annotated=True
                )
                self.aggregated_data.append(pair)
                added_count += 1
        
        return added_count
    
    def _should_aggregate(
        self,
        state: Any,
        expert_action: Any,
        learner_action: Any
    ) -> bool:
        """判断是否应该聚合该数据点"""
        strategy = self.config.aggregation_strategy
        
        if strategy == 'all':
            return True
        elif strategy == 'error':
            # 只聚合学习者犯错的样本
            return expert_action != learner_action
        elif strategy == 'uncertain':
            # 只聚合学习者不确定的样本
            probs = self.learner.get_action_probabilities(state)
            max_prob = max(probs.values())
            return max_prob < 0.7  # 最高概率小于0.7认为不确定
        
        return True
    
    def train_on_aggregated_data(self, epochs: int = 10) -> Dict[str, float]:
        """在聚合数据上训练"""
        if not self.aggregated_data:
            return {'loss': 0.0, 'accuracy': 0.0}
        
        # 准备训练数据
        data = [(p.state, p.expert_action) for p in self.aggregated_data]
        
        losses = []
        for epoch in range(epochs):
            # 打乱数据
            random.shuffle(data)
            
            # 批量训练
            for i in range(0, len(data), self.config.batch_size):
                batch = data[i:i + self.config.batch_size]
                loss = self.learner.batch_update(batch)
                losses.append(loss)
        
        # 计算准确率
        correct = sum(
            1 for p in self.aggregated_data
            if self.learner.select_action(p.state) == p.expert_action
        )
        accuracy = correct / len(self.aggregated_data)
        
        return {
            'loss': sum(losses) / len(losses) if losses else 0.0,
            'accuracy': accuracy
        }
    
    def run_iteration(
        self,
        env: Any,
        num_episodes: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        运行一次DAgger迭代
        
        Args:
            env: 环境
            num_episodes: episode数量
            
        Returns:
            迭代统计信息
        """
        num_episodes = num_episodes or self.config.episodes_per_iteration
        
        # 更新beta
        self.current_beta = self.get_beta(self.current_iteration)
        
        # 收集数据
        all_states = []
        all_rewards = []
        
        for _ in range(num_episodes):
            states, actions, reward = self.collect_episode(
                env, self.config.max_steps_per_episode
            )
            all_states.extend(states)
            all_rewards.append(reward)
        
        # 请求专家标注
        expert_actions = self.request_annotations(all_states)
        
        # 聚合数据
        added = self.aggregate_data(all_states, expert_actions)
        
        # 训练
        train_metrics = self.train_on_aggregated_data()
        
        # 记录历史
        iteration_info = {
            'iteration': self.current_iteration,
            'beta': self.current_beta,
            'episodes': num_episodes,
            'states_collected': len(all_states),
            'data_added': added,
            'total_data': len(self.aggregated_data),
            'avg_reward': sum(all_rewards) / len(all_rewards) if all_rewards else 0.0,
            'annotation_cost': self.expert.get_annotation_cost(),
            'train_loss': train_metrics['loss'],
            'train_accuracy': train_metrics['accuracy']
        }
        
        self.iteration_history.append(iteration_info)
        self.current_iteration += 1
        
        return iteration_info
    
    def run(
        self,
        env: Any,
        max_iterations: Optional[int] = None,
        convergence_threshold: float = 0.95,
        verbose: bool = False
    ) -> Dict[str, Any]:
        """
        运行完整的DAgger训练
        
        Args:
            env: 环境
            max_iterations: 最大迭代次数
            convergence_threshold: 收敛阈值（准确率）
            verbose: 是否打印详细信息
            
        Returns:
            训练结果
        """
        max_iterations = max_iterations or self.config.max_iterations
        
        for i in range(max_iterations):
            info = self.run_iteration(env)
            
            if verbose:
                print(f"Iteration {i}: beta={info['beta']:.3f}, "
                      f"accuracy={info['train_accuracy']:.3f}, "
                      f"data_size={info['total_data']}")
            
            # 检查收敛
            if info['train_accuracy'] >= convergence_threshold:
                if verbose:
                    print(f"Converged at iteration {i}")
                break
        
        return {
            'final_iteration': self.current_iteration,
            'final_accuracy': self.iteration_history[-1]['train_accuracy'] 
                             if self.iteration_history else 0.0,
            'total_data': len(self.aggregated_data),
            'total_annotations': self.expert.get_annotation_cost(),
            'iteration_history': self.iteration_history
        }
    
    def get_policy(self) -> LearnerPolicy:
        """获取训练好的策略"""
        return self.learner
    
    def evaluate(
        self,
        env: Any,
        num_episodes: int = 10
    ) -> Dict[str, float]:
        """评估当前策略"""
        rewards = []
        successes = 0
        
        for _ in range(num_episodes):
            state = env.reset() if hasattr(env, 'reset') else env
            total_reward = 0.0
            
            for _ in range(self.config.max_steps_per_episode):
                action = self.learner.select_action(state)
                
                if hasattr(env, 'step'):
                    result = env.step(action)
                    if len(result) == 4:
                        next_state, reward, done, info = result
                    else:
                        next_state, reward, done = result
                        info = {}
                else:
                    next_state = state
                    reward = 0.0
                    done = False
                
                total_reward += reward
                state = next_state
                
                if done:
                    if reward > 0:
                        successes += 1
                    break
            
            rewards.append(total_reward)
        
        return {
            'avg_reward': sum(rewards) / len(rewards) if rewards else 0.0,
            'success_rate': successes / num_episodes,
            'num_episodes': num_episodes
        }


class DAggerWithReset(DAggerLoop):
    """
    带重置的DAgger
    
    在训练过程中定期重置到专家演示的初始状态，
    提高数据收集效率
    """
    
    def __init__(
        self,
        expert: ExpertOracle,
        learner: LearnerPolicy,
        config: Optional[DAggerConfig] = None,
        reset_probability: float = 0.1
    ):
        super().__init__(expert, learner, config)
        self.reset_probability = reset_probability
        self.expert_demonstrations: List[Tuple[Any, Any]] = []
    
    def add_expert_demonstration(self, states: List[Any], actions: List[Any]) -> None:
        """添加专家演示"""
        for state, action in zip(states, actions):
            self.expert_demonstrations.append((state, action))
    
    def collect_episode(
        self,
        env: Any,
        max_steps: int
    ) -> Tuple[List[Any], List[Any], float]:
        """收集episode，带随机重置"""
        states = []
        actions = []
        total_reward = 0.0
        
        state = env.reset() if hasattr(env, 'reset') else env
        
        for step in range(max_steps):
            # 随机重置到专家演示的状态
            if (self.expert_demonstrations and 
                random.random() < self.reset_probability):
                idx = random.randint(0, len(self.expert_demonstrations) - 1)
                state, _ = self.expert_demonstrations[idx]
            
            states.append(state)
            action, _ = self.mixed_policy_action(state)
            actions.append(action)
            
            if hasattr(env, 'step'):
                result = env.step(action)
                if len(result) == 4:
                    next_state, reward, done, info = result
                else:
                    next_state, reward, done = result
            else:
                next_state = state
                reward = 0.0
                done = False
            
            total_reward += reward
            state = next_state
            
            if done:
                break
        
        return states, actions, total_reward


class ActiveDAgger(DAggerLoop):
    """
    主动学习DAgger
    
    使用不确定性采样主动选择最有价值的状态进行标注
    """
    
    def __init__(
        self,
        expert: ExpertOracle,
        learner: LearnerPolicy,
        config: Optional[DAggerConfig] = None,
        uncertainty_threshold: float = 0.3
    ):
        super().__init__(expert, learner, config)
        self.uncertainty_threshold = uncertainty_threshold
    
    def compute_uncertainty(self, state: Any) -> float:
        """计算状态的不确定性（熵）"""
        probs = self.learner.get_action_probabilities(state)
        
        # 计算熵
        entropy = 0.0
        for p in probs.values():
            if p > 0:
                entropy -= p * math.log(p)
        
        # 归一化
        max_entropy = math.log(len(self.learner.action_space))
        return entropy / max_entropy if max_entropy > 0 else 0.0
    
    def select_states_for_annotation(
        self,
        states: List[Any],
        budget: int
    ) -> List[int]:
        """
        选择最有价值的状态进行标注
        
        Args:
            states: 候选状态列表
            budget: 标注预算
            
        Returns:
            选中的状态索引
        """
        # 计算每个状态的不确定性
        uncertainties = [(i, self.compute_uncertainty(s)) for i, s in enumerate(states)]
        
        # 按不确定性排序
        uncertainties.sort(key=lambda x: x[1], reverse=True)
        
        # 选择前budget个
        selected = [idx for idx, _ in uncertainties[:budget]]
        
        return selected
    
    def aggregate_data(
        self,
        states: List[Any],
        expert_actions: List[Any]
    ) -> int:
        """聚合数据，使用主动学习策略"""
        added_count = 0
        
        for state, expert_action in zip(states, expert_actions):
            uncertainty = self.compute_uncertainty(state)
            
            # 只聚合高不确定性的样本
            if uncertainty >= self.uncertainty_threshold:
                learner_action = self.learner.select_action(state)
                pair = StateActionPair(
                    state=state,
                    expert_action=expert_action,
                    learner_action=learner_action,
                    annotated=True
                )
                self.aggregated_data.append(pair)
                added_count += 1
        
        return added_count
