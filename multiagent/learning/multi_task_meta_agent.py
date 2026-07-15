"""
多任务元学习模块 - Multi-Task Meta-Learning

实现MAML (Model-Agnostic Meta-Learning) 算法，
使Agent能够快速适应不同的协作风格和任务类型。
"""

from typing import Dict, List, Any, Optional, Callable, Tuple, Union
from dataclasses import dataclass, field
from collections import defaultdict
import random
import math
import copy


@dataclass
class TaskSpecification:
    """任务规格"""
    task_id: str
    task_type: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    reward_function: Optional[Callable] = None
    transition_function: Optional[Callable] = None


@dataclass
class AdaptationResult:
    """适应结果"""
    task_id: str
    initial_loss: float
    final_loss: float
    adaptation_steps: int
    adaptation_time: float
    success: bool


@dataclass
class MAMLConfig:
    """MAML配置"""
    meta_learning_rate: float = 0.001     # 元学习率
    inner_learning_rate: float = 0.01     # 内层学习率
    num_inner_steps: int = 5              # 内层梯度步数
    num_meta_iterations: int = 1000       # 元迭代次数
    tasks_per_iteration: int = 4          # 每次元迭代采样的任务数
    first_order: bool = False             # 是否使用一阶近似
    meta_batch_size: int = 4              # 元批次大小


class MetaParameter:
    """
    元参数容器
    
    存储和操作模型的元参数，支持梯度更新
    """
    
    def __init__(self, params: Optional[Dict[str, List[float]]] = None):
        self.params: Dict[str, List[float]] = params or {}
        self.gradients: Dict[str, List[float]] = {}
    
    def initialize(self, layer_dims: List[Tuple[str, int]], 
                   init_scale: float = 0.1) -> None:
        """初始化参数"""
        for name, dim in layer_dims:
            self.params[name] = [random.uniform(-init_scale, init_scale) 
                                for _ in range(dim)]
    
    def copy(self) -> 'MetaParameter':
        """深拷贝参数"""
        new_params = {k: v.copy() for k, v in self.params.items()}
        return MetaParameter(new_params)
    
    def update(self, gradients: Dict[str, List[float]], 
               learning_rate: float) -> None:
        """参数更新"""
        for name, grad in gradients.items():
            if name in self.params:
                for i in range(len(self.params[name])):
                    self.params[name][i] -= learning_rate * grad[i]
    
    def add_gradient(self, gradients: Dict[str, List[float]], 
                     weight: float = 1.0) -> None:
        """累加梯度"""
        for name, grad in gradients.items():
            if name not in self.gradients:
                self.gradients[name] = [0.0] * len(grad)
            for i in range(len(grad)):
                self.gradients[name][i] += weight * grad[i]
    
    def clear_gradients(self) -> None:
        """清空梯度"""
        self.gradients = {}
    
    def get_param(self, name: str) -> List[float]:
        """获取参数"""
        return self.params.get(name, [])
    
    def set_param(self, name: str, value: List[float]) -> None:
        """设置参数"""
        self.params[name] = value


class TaskEnvironment:
    """
    任务环境接口
    
    定义不同任务的环境和奖励函数
    """
    
    def __init__(self, spec: TaskSpecification):
        self.spec = spec
        self.state: Any = None
        self.step_count = 0
    
    def reset(self) -> Any:
        """重置环境"""
        self.step_count = 0
        self.state = self._init_state()
        return self.state
    
    def _init_state(self) -> Any:
        """初始化状态"""
        return [0.0] * 10  # 默认10维状态
    
    def step(self, action: Any) -> Tuple[Any, float, bool, Dict]:
        """执行动作"""
        self.step_count += 1
        
        # 使用自定义转移函数或默认
        if self.spec.transition_function:
            next_state = self.spec.transition_function(self.state, action)
        else:
            next_state = self._default_transition(self.state, action)
        
        # 计算奖励
        if self.spec.reward_function:
            reward = self.spec.reward_function(self.state, action, next_state)
        else:
            reward = self._default_reward(self.state, action, next_state)
        
        # 检查是否结束
        done = self.step_count >= 100 or self._is_terminal(next_state)
        
        self.state = next_state
        return next_state, reward, done, {}
    
    def _default_transition(self, state: Any, action: Any) -> Any:
        """默认转移函数"""
        if isinstance(state, list):
            new_state = state.copy()
            if isinstance(action, (int, float)):
                new_state[0] = action
            return new_state
        return state
    
    def _default_reward(self, state: Any, action: Any, next_state: Any) -> float:
        """默认奖励函数"""
        if isinstance(state, list) and isinstance(next_state, list):
            # 奖励基于状态变化
            diff = sum(abs(s - n) for s, n in zip(state, next_state))
            return -diff  # 最小化变化
        return 0.0
    
    def _is_terminal(self, state: Any) -> bool:
        """检查是否终止"""
        return False


class CollaborativeTaskEnvironment(TaskEnvironment):
    """
    协作任务环境
    
    模拟多Agent协作场景
    """
    
    def __init__(
        self, 
        spec: TaskSpecification,
        num_agents: int = 2,
        collaboration_style: str = 'cooperative'
    ):
        super().__init__(spec)
        self.num_agents = num_agents
        self.collaboration_style = collaboration_style
        self.agent_states: Dict[str, Any] = {}
    
    def reset(self) -> Any:
        """重置"""
        super().reset()
        for i in range(self.num_agents):
            self.agent_states[f'agent_{i}'] = self._init_state()
        return self.state
    
    def step_with_agents(
        self, 
        actions: Dict[str, Any]
    ) -> Tuple[Any, float, bool, Dict]:
        """多Agent步进"""
        total_reward = 0.0
        
        for agent_id, action in actions.items():
            if agent_id in self.agent_states:
                # 更新Agent状态
                self.agent_states[agent_id] = self._default_transition(
                    self.agent_states[agent_id], action
                )
                
                # 计算个体奖励
                individual_reward = self._compute_individual_reward(
                    agent_id, action
                )
                total_reward += individual_reward
        
        # 协作奖励
        collaboration_reward = self._compute_collaboration_reward()
        total_reward += collaboration_reward
        
        # 更新全局状态
        self.state = self._aggregate_agent_states()
        self.step_count += 1
        
        done = self.step_count >= 100
        
        return self.state, total_reward, done, {
            'collaboration_reward': collaboration_reward
        }
    
    def _compute_individual_reward(self, agent_id: str, action: Any) -> float:
        """计算个体奖励"""
        return 1.0 if action is not None else -0.1
    
    def _compute_collaboration_reward(self) -> float:
        """计算协作奖励"""
        if self.collaboration_style == 'cooperative':
            # 合作风格：奖励一致性
            states = list(self.agent_states.values())
            if len(states) >= 2:
                consistency = self._compute_consistency(states)
                return consistency * 2.0
        elif self.collaboration_style == 'competitive':
            # 竞争风格：奖励差异
            states = list(self.agent_states.values())
            if len(states) >= 2:
                diversity = self._compute_diversity(states)
                return diversity * 2.0
        return 0.0
    
    def _compute_consistency(self, states: List[Any]) -> float:
        """计算一致性"""
        if not states:
            return 0.0
        avg = [sum(s[i] if isinstance(s, list) else 0 
                   for s in states) / len(states) 
               for i in range(min(len(s) for s in states if isinstance(s, list)))]
        return sum(-abs(s - a) for s, a in zip(states[0] if isinstance(states[0], list) else [], avg))
    
    def _compute_diversity(self, states: List[Any]) -> float:
        """计算多样性"""
        if len(states) < 2:
            return 0.0
        total_diff = 0.0
        for i in range(len(states)):
            for j in range(i + 1, len(states)):
                if isinstance(states[i], list) and isinstance(states[j], list):
                    total_diff += sum(abs(a - b) for a, b in zip(states[i], states[j]))
        return total_diff / (len(states) * (len(states) - 1) / 2)
    
    def _aggregate_agent_states(self) -> Any:
        """聚合Agent状态"""
        states = list(self.agent_states.values())
        if not states:
            return [0.0] * 10
        return [sum(s[i] if isinstance(s, list) and i < len(s) else 0 
                   for s in states) / len(states) 
               for i in range(10)]


class MAMLMetaLearner:
    """
    MAML元学习器
    
    实现Model-Agnostic Meta-Learning算法，
    学习一个好的初始化参数，使Agent能够快速适应新任务
    """
    
    def __init__(
        self,
        config: Optional[MAMLConfig] = None,
        model_architecture: Optional[List[Tuple[str, int]]] = None
    ):
        self.config = config or MAMLConfig()
        
        # 初始化元参数
        self.meta_params = MetaParameter()
        if model_architecture:
            self.meta_params.initialize(model_architecture)
        else:
            # 默认架构
            self.meta_params.initialize([
                ('W1', 100), ('b1', 10),
                ('W2', 100), ('b2', 10),
                ('W3', 50), ('b3', 5)
            ])
        
        # 任务池
        self.task_pool: List[TaskSpecification] = []
        
        # 适应历史
        self.adaptation_history: List[AdaptationResult] = []
    
    def add_task(self, task: TaskSpecification) -> None:
        """添加任务到任务池"""
        self.task_pool.append(task)
    
    def add_tasks(self, tasks: List[TaskSpecification]) -> None:
        """批量添加任务"""
        self.task_pool.extend(tasks)
    
    def sample_tasks(self, num_tasks: int) -> List[TaskSpecification]:
        """随机采样任务"""
        if len(self.task_pool) <= num_tasks:
            return self.task_pool.copy()
        return random.sample(self.task_pool, num_tasks)
    
    def forward(
        self, 
        params: MetaParameter, 
        state: Any
    ) -> List[float]:
        """前向传播"""
        # 提取状态特征
        if isinstance(state, list):
            x = [float(s) for s in state]
        elif isinstance(state, dict):
            x = [float(v) for v in state.values() if isinstance(v, (int, float))]
        else:
            x = [float(state)]
        
        # 填充到固定维度
        while len(x) < 10:
            x.append(0.0)
        x = x[:10]
        
        # 第一层
        W1 = params.get_param('W1')
        b1 = params.get_param('b1')
        h1 = self._linear_relu(x, W1, b1, 10)
        
        # 第二层
        W2 = params.get_param('W2')
        b2 = params.get_param('b2')
        h2 = self._linear_relu(h1, W2, b2, 10)
        
        # 输出层
        W3 = params.get_param('W3')
        b3 = params.get_param('b3')
        output = self._linear(h2, W3, b3, 5)
        
        return output
    
    def _linear_relu(
        self, 
        x: List[float], 
        W: List[float], 
        b: List[float],
        out_dim: int
    ) -> List[float]:
        """线性层 + ReLU"""
        in_dim = len(x)
        output = []
        
        for j in range(out_dim):
            val = b[j] if j < len(b) else 0.0
            for i in range(in_dim):
                idx = i * out_dim + j
                if idx < len(W):
                    val += x[i] * W[idx]
            output.append(max(0, val))  # ReLU
        
        return output
    
    def _linear(
        self, 
        x: List[float], 
        W: List[float], 
        b: List[float],
        out_dim: int
    ) -> List[float]:
        """线性层"""
        in_dim = len(x)
        output = []
        
        for j in range(out_dim):
            val = b[j] if j < len(b) else 0.0
            for i in range(in_dim):
                idx = i * out_dim + j
                if idx < len(W):
                    val += x[i] * W[idx]
            output.append(val)
        
        return output
    
    def compute_loss(
        self, 
        params: MetaParameter, 
        env: TaskEnvironment,
        num_samples: int = 10
    ) -> float:
        """计算损失"""
        total_loss = 0.0
        
        for _ in range(num_samples):
            state = env.reset()
            
            for _ in range(10):  # 最多10步
                # 前向传播
                output = self.forward(params, state)
                
                # 选择动作（输出中最大值对应的动作）
                action = output.index(max(output)) if output else 0
                
                # 执行动作
                next_state, reward, done, _ = env.step(action)
                
                # 损失为负奖励（最小化损失 = 最大化奖励）
                total_loss -= reward
                
                state = next_state
                if done:
                    break
        
        return total_loss / num_samples
    
    def compute_gradients(
        self, 
        params: MetaParameter, 
        env: TaskEnvironment
    ) -> Dict[str, List[float]]:
        """计算梯度（数值梯度）"""
        gradients = {}
        epsilon = 1e-4
        
        for name, values in params.params.items():
            grad = []
            for i in range(len(values)):
                # 前向差分
                params_plus = params.copy()
                params_plus.params[name][i] += epsilon
                loss_plus = self.compute_loss(params_plus, env)
                
                params_minus = params.copy()
                params_minus.params[name][i] -= epsilon
                loss_minus = self.compute_loss(params_minus, env)
                
                grad.append((loss_plus - loss_minus) / (2 * epsilon))
            
            gradients[name] = grad
        
        return gradients
    
    def inner_loop_adapt(
        self, 
        task: TaskSpecification,
        num_steps: Optional[int] = None
    ) -> Tuple[MetaParameter, float]:
        """
        内层循环适应
        
        在特定任务上进行梯度下降
        """
        num_steps = num_steps or self.config.num_inner_steps
        
        # 复制元参数
        adapted_params = self.meta_params.copy()
        
        # 创建环境
        env = TaskEnvironment(task)
        
        # 梯度下降
        for _ in range(num_steps):
            gradients = self.compute_gradients(adapted_params, env)
            adapted_params.update(gradients, self.config.inner_learning_rate)
        
        # 计算最终损失
        final_loss = self.compute_loss(adapted_params, env)
        
        return adapted_params, final_loss
    
    def meta_update(
        self, 
        tasks: List[TaskSpecification]
    ) -> float:
        """
        元更新
        
        聚合多个任务的梯度，更新元参数
        """
        self.meta_params.clear_gradients()
        
        total_loss = 0.0
        
        for task in tasks:
            # 内层适应
            adapted_params, task_loss = self.inner_loop_adapt(task)
            total_loss += task_loss
            
            # 计算适应后参数的梯度
            env = TaskEnvironment(task)
            
            if self.config.first_order:
                # 一阶近似：直接使用内层梯度
                gradients = self.compute_gradients(adapted_params, env)
            else:
                # 二阶：需要通过链式法则计算
                gradients = self._compute_meta_gradients(task, adapted_params)
            
            # 累加梯度
            self.meta_params.add_gradient(gradients, 1.0 / len(tasks))
        
        # 元参数更新
        self.meta_params.update(
            self.meta_params.gradients, 
            self.config.meta_learning_rate
        )
        
        return total_loss / len(tasks)
    
    def _compute_meta_gradients(
        self, 
        task: TaskSpecification,
        adapted_params: MetaParameter
    ) -> Dict[str, List[float]]:
        """计算元梯度（二阶）"""
        env = TaskEnvironment(task)
        
        # 简化：使用一阶近似
        return self.compute_gradients(adapted_params, env)
    
    def train(
        self,
        num_iterations: Optional[int] = None,
        tasks_per_iter: Optional[int] = None,
        callback: Optional[Callable[[int, float], None]] = None
    ) -> Dict[str, Any]:
        """
        训练元学习器
        
        Args:
            num_iterations: 元迭代次数
            tasks_per_iter: 每次迭代采样的任务数
            callback: 回调函数
            
        Returns:
            训练统计
        """
        num_iterations = num_iterations or self.config.num_meta_iterations
        tasks_per_iter = tasks_per_iter or self.config.tasks_per_iteration
        
        losses = []
        
        for iteration in range(num_iterations):
            # 采样任务
            tasks = self.sample_tasks(tasks_per_iter)
            
            if not tasks:
                continue
            
            # 元更新
            avg_loss = self.meta_update(tasks)
            losses.append(avg_loss)
            
            # 回调
            if callback:
                callback(iteration, avg_loss)
        
        return {
            'num_iterations': num_iterations,
            'final_loss': losses[-1] if losses else 0.0,
            'loss_history': losses,
            'best_loss': min(losses) if losses else 0.0
        }
    
    def adapt_to_task(
        self, 
        task: TaskSpecification,
        num_steps: Optional[int] = None
    ) -> Tuple[MetaParameter, AdaptationResult]:
        """
        适应到新任务
        
        Args:
            task: 新任务
            num_steps: 适应步数
            
        Returns:
            (适应后的参数, 适应结果)
        """
        import time
        
        start_time = time.time()
        
        # 计算初始损失
        env = TaskEnvironment(task)
        initial_loss = self.compute_loss(self.meta_params, env)
        
        # 内层适应
        adapted_params, final_loss = self.inner_loop_adapt(task, num_steps)
        
        adaptation_time = time.time() - start_time
        
        result = AdaptationResult(
            task_id=task.task_id,
            initial_loss=initial_loss,
            final_loss=final_loss,
            adaptation_steps=num_steps or self.config.num_inner_steps,
            adaptation_time=adaptation_time,
            success=final_loss < initial_loss
        )
        
        self.adaptation_history.append(result)
        
        return adapted_params, result
    
    def get_adapted_policy(
        self, 
        adapted_params: MetaParameter
    ) -> Callable[[Any], int]:
        """获取适应后的策略函数"""
        def policy(state: Any) -> int:
            output = self.forward(adapted_params, state)
            return output.index(max(output)) if output else 0
        return policy


class MultiTaskMetaAgent:
    """
    多任务元Agent
    
    使用元学习快速适应不同协作风格的Agent
    """
    
    def __init__(
        self,
        agent_id: str,
        config: Optional[MAMLConfig] = None
    ):
        self.agent_id = agent_id
        self.config = config or MAMLConfig()
        
        # 元学习器
        self.meta_learner = MAMLMetaLearner(self.config)
        
        # 当前任务和适应后的参数
        self.current_task: Optional[TaskSpecification] = None
        self.current_params: Optional[MetaParameter] = None
        
        # 协作风格
        self.collaboration_styles = [
            'cooperative',    # 合作型
            'competitive',    # 竞争型
            'hierarchical',   # 层级型
            'democratic',     # 民主型
            'specialized'     # 专业化型
        ]
        
        # 风格-任务映射
        self.style_tasks: Dict[str, List[TaskSpecification]] = defaultdict(list)
        
        # 经验缓冲
        self.experience_buffer: List[Tuple[Any, Any, float, Any]] = []
    
    def register_collaboration_style(
        self, 
        style: str, 
        tasks: List[TaskSpecification]
    ) -> None:
        """注册协作风格及其对应任务"""
        self.style_tasks[style] = tasks
        self.meta_learner.add_tasks(tasks)
    
    def pre_train(
        self,
        num_iterations: int = 100,
        verbose: bool = False
    ) -> Dict[str, Any]:
        """预训练元学习器"""
        def callback(iteration: int, loss: float) -> None:
            if verbose and iteration % 10 == 0:
                print(f"Iteration {iteration}: loss = {loss:.4f}")
        
        return self.meta_learner.train(
            num_iterations=num_iterations,
            callback=callback
        )
    
    def adapt_to_style(
        self, 
        style: str,
        num_steps: int = 5
    ) -> AdaptationResult:
        """适应到特定协作风格"""
        if style not in self.style_tasks or not self.style_tasks[style]:
            raise ValueError(f"未知的协作风格: {style}")
        
        # 随机选择该风格的一个任务
        task = random.choice(self.style_tasks[style])
        self.current_task = task
        
        # 适应
        self.current_params, result = self.meta_learner.adapt_to_task(
            task, num_steps
        )
        
        return result
    
    def select_action(self, state: Any) -> int:
        """选择动作"""
        if self.current_params is None:
            # 使用元参数
            output = self.meta_learner.forward(self.meta_learner.meta_params, state)
        else:
            output = self.meta_learner.forward(self.current_params, state)
        
        return output.index(max(output)) if output else 0
    
    def observe_transition(
        self, 
        state: Any, 
        action: int, 
        reward: float, 
        next_state: Any
    ) -> None:
        """观察转移"""
        self.experience_buffer.append((state, action, reward, next_state))
        
        # 限制缓冲大小
        if len(self.experience_buffer) > 10000:
            self.experience_buffer = self.experience_buffer[-10000:]
    
    def fine_tune_online(
        self, 
        learning_rate: float = 0.01,
        batch_size: int = 32
    ) -> float:
        """在线微调"""
        if len(self.experience_buffer) < batch_size:
            return 0.0
        
        # 采样批次
        batch = random.sample(self.experience_buffer, batch_size)
        
        total_loss = 0.0
        
        for state, action, reward, next_state in batch:
            if self.current_params:
                output = self.meta_learner.forward(self.current_params, state)
                predicted_action = output.index(max(output)) if output else 0
                
                # 简单的损失计算
                if predicted_action != action:
                    total_loss += 1.0
        
        return total_loss / batch_size
    
    def evaluate_style(
        self, 
        style: str, 
        num_episodes: int = 10
    ) -> Dict[str, float]:
        """评估在特定风格上的表现"""
        if style not in self.style_tasks or not self.style_tasks[style]:
            return {'avg_reward': 0.0, 'success_rate': 0.0}
        
        task = random.choice(self.style_tasks[style])
        env = CollaborativeTaskEnvironment(
            task, 
            num_agents=2, 
            collaboration_style=style
        )
        
        total_rewards = []
        successes = 0
        
        for _ in range(num_episodes):
            state = env.reset()
            episode_reward = 0.0
            
            for _ in range(100):
                action = self.select_action(state)
                
                # 模拟其他Agent的动作
                actions = {
                    'agent_0': action,
                    'agent_1': random.randint(0, 4)
                }
                
                next_state, reward, done, info = env.step_with_agents(actions)
                episode_reward += reward
                state = next_state
                
                if done:
                    if reward > 0:
                        successes += 1
                    break
            
            total_rewards.append(episode_reward)
        
        return {
            'avg_reward': sum(total_rewards) / len(total_rewards),
            'success_rate': successes / num_episodes,
            'style': style
        }
    
    def get_meta_knowledge(self) -> Dict[str, Any]:
        """获取元知识"""
        return {
            'agent_id': self.agent_id,
            'num_tasks_in_pool': len(self.meta_learner.task_pool),
            'registered_styles': list(self.style_tasks.keys()),
            'adaptation_history': [
                {
                    'task_id': r.task_id,
                    'improvement': r.initial_loss - r.final_loss,
                    'success': r.success
                }
                for r in self.meta_learner.adaptation_history[-10:]
            ]
        }
    
    def transfer_to_new_task(
        self, 
        new_task: TaskSpecification,
        num_adaptation_steps: int = 5
    ) -> Tuple[Callable[[Any], int], AdaptationResult]:
        """
        迁移到新任务
        
        使用元学习的初始化快速适应新任务
        """
        adapted_params, result = self.meta_learner.adapt_to_task(
            new_task, num_adaptation_steps
        )
        
        policy = self.meta_learner.get_adapted_policy(adapted_params)
        
        return policy, result


class StyleAwareMetaAgent(MultiTaskMetaAgent):
    """
    风格感知元Agent
    
    能够识别当前协作风格并自动适应
    """
    
    def __init__(
        self, 
        agent_id: str, 
        config: Optional[MAMLConfig] = None
    ):
        super().__init__(agent_id, config)
        
        # 风格识别器
        self.style_history: List[Tuple[str, float]] = []
        self.current_style: Optional[str] = None
    
    def detect_style(
        self, 
        interaction_history: List[Dict[str, Any]]
    ) -> str:
        """
        检测协作风格
        
        基于交互历史判断当前协作风格
        """
        if not interaction_history:
            return 'cooperative'  # 默认合作
        
        # 计算风格特征
        cooperation_score = 0.0
        competition_score = 0.0
        
        for interaction in interaction_history[-20:]:  # 最近20次交互
            # 检查动作一致性
            if 'action_agreement' in interaction:
                if interaction['action_agreement']:
                    cooperation_score += 1
                else:
                    competition_score += 1
            
            # 检查奖励分配
            if 'reward_difference' in interaction:
                diff = abs(interaction['reward_difference'])
                if diff < 0.1:
                    cooperation_score += 1
                else:
                    competition_score += 1
        
        # 确定风格
        if cooperation_score > competition_score * 1.5:
            detected_style = 'cooperative'
        elif competition_score > cooperation_score * 1.5:
            detected_style = 'competitive'
        else:
            detected_style = 'democratic'
        
        self.style_history.append((detected_style, time.time()))
        self.current_style = detected_style
        
        return detected_style
    
    def auto_adapt(
        self, 
        interaction_history: List[Dict[str, Any]]
    ) -> Optional[AdaptationResult]:
        """自动适应检测到的风格"""
        detected_style = self.detect_style(interaction_history)
        
        if detected_style in self.style_tasks:
            return self.adapt_to_style(detected_style)
        
        return None
    
    def get_style_confidence(self) -> Dict[str, float]:
        """获取风格置信度"""
        if not self.style_history:
            return {style: 0.0 for style in self.collaboration_styles}
        
        # 统计最近的风格分布
        style_counts = defaultdict(int)
        for style, _ in self.style_history[-50:]:
            style_counts[style] += 1
        
        total = sum(style_counts.values())
        
        return {
            style: style_counts.get(style, 0) / total
            for style in self.collaboration_styles
        }


# 导入time用于风格检测
import time
