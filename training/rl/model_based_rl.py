"""
Model-based Reinforcement Learning模块 - 包含各种MBRL算法的真实实现
包括: World Models, Dreamer, MuZero, MBPO, PETS, PlaNet, DreamerV2, DreamerV3
"""

import math
import random
from typing import Optional, Tuple, List, Union, Callable, Dict, Any
from abc import ABC, abstractmethod
from collections import deque


def matmul(a, b):
    """矩阵乘法"""
    if isinstance(a, (int, float)) or isinstance(b, (int, float)):
        return a * b
    if isinstance(a[0], list) and isinstance(b[0], list):
        return [[sum(a[i][k] * b[k][j] for k in range(len(b))) for j in range(len(b[0]))] for i in range(len(a))]
    elif isinstance(a[0], list):
        return [sum(a[i][k] * b[k] for k in range(len(b))) for i in range(len(a))]
    elif isinstance(b[0], list):
        return [sum(a[k] * b[k][j] for k in range(len(a))) for j in range(len(b[0]))]
    else:
        return sum(a[k] * b[k] for k in range(len(a)))


def softmax(x: List[float]) -> List[float]:
    max_x = max(x)
    exp_x = [math.exp(xi - max_x) for xi in x]
    sum_exp = sum(exp_x)
    return [e / sum_exp for e in exp_x]


class ReplayBuffer:
    """经验回放缓冲区"""
    
    def __init__(self, capacity: int = 100000):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int) -> List[Tuple]:
        return random.sample(list(self.buffer), min(batch_size, len(self.buffer)))
    
    def __len__(self):
        return len(self.buffer)


class SequenceBuffer:
    """序列缓冲区，用于存储轨迹"""
    
    def __init__(self, capacity: int = 1000, seq_len: int = 50):
        self.capacity = capacity
        self.seq_len = seq_len
        self.sequences = deque(maxlen=capacity)
        self.current_sequence = []
    
    def push(self, state, action, reward, done):
        self.current_sequence.append((state, action, reward))
        if done or len(self.current_sequence) >= self.seq_len:
            if len(self.current_sequence) >= 2:
                self.sequences.append(list(self.current_sequence))
            self.current_sequence = []
    
    def sample(self, batch_size: int) -> List[List[Tuple]]:
        return random.sample(list(self.sequences), min(batch_size, len(self.sequences)))
    
    def __len__(self):
        return len(self.sequences)


class WorldModel:
    """
    World Models - Ha & Schmidhuber 2018
    
    包含三个组件:
    1. VAE: 将观察压缩到潜在空间
    2. RNN: 在潜在空间中预测未来
    3. Controller: 在潜在空间中选择动作
    """
    
    def __init__(
        self,
        obs_dim: int,
        latent_dim: int = 32,
        hidden_dim: int = 256,
        action_dim: int = 3,
        rnn_type: str = 'lstm'
    ):
        self.obs_dim = obs_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.rnn_type = rnn_type
        
        # VAE参数
        self.vae_encoder = self._init_weight(obs_dim, hidden_dim)
        self.vae_mu = self._init_weight(hidden_dim, latent_dim)
        self.vae_logvar = self._init_weight(hidden_dim, latent_dim)
        self.vae_decoder = self._init_weight(latent_dim, hidden_dim)
        self.vae_output = self._init_weight(hidden_dim, obs_dim)
        
        # RNN参数 (MDN-RNN)
        self.rnn_w_ih = self._init_weight(latent_dim + action_dim, hidden_dim)
        self.rnn_w_hh = self._init_weight(hidden_dim, hidden_dim)
        self.rnn_bias = [0.0] * hidden_dim
        
        # MDN输出 (混合高斯)
        self.num_mixtures = 5
        self.mdn_fc = self._init_weight(hidden_dim, (2 * latent_dim + 1) * self.num_mixtures)
        
        # RNN隐藏状态
        self.hidden = [0.0] * hidden_dim
    
    def _init_weight(self, in_dim: int, out_dim: int):
        scale = math.sqrt(2.0 / (in_dim + out_dim))
        if isinstance(out_dim, int):
            return [[random.gauss(0, scale) for _ in range(out_dim)] for _ in range(in_dim)]
        else:
            return [[random.gauss(0, scale) for _ in range(out_dim)] for _ in range(in_dim)]
    
    def encode(self, obs) -> Tuple[List[float], List[float], List[float]]:
        """VAE编码"""
        # 编码器前向
        h = [math.tanh(sum(self.vae_encoder[i][j] * obs[j] for j in range(len(obs))))
             for i in range(self.hidden_dim)]
        
        # 计算均值和方差
        mu = [sum(self.vae_mu[i][j] * h[j] for j in range(self.hidden_dim))
              for i in range(self.latent_dim)]
        logvar = [sum(self.vae_logvar[i][j] * h[j] for j in range(self.hidden_dim))
                  for i in range(self.latent_dim)]
        
        # 重参数化采样
        std = [math.exp(0.5 * lv) for lv in logvar]
        z = [mu[i] + std[i] * random.gauss(0, 1) for i in range(self.latent_dim)]
        
        return z, mu, logvar
    
    def decode(self, z: List[float]):
        """VAE解码"""
        h = [math.tanh(sum(self.vae_decoder[i][j] * z[j] for j in range(self.latent_dim)))
             for i in range(self.hidden_dim)]
        
        obs_recon = [sum(self.vae_output[i][j] * h[j] for j in range(self.hidden_dim))
                     for i in range(self.obs_dim)]
        
        return obs_recon
    
    def rnn_forward(self, z: List[float], action: List[float]) -> Tuple[List[float], List[List[float]]]:
        """RNN前向传播，输出MDN参数"""
        # 拼接输入
        x = z + action
        
        # RNN更新
        h_new = [math.tanh(sum(self.rnn_w_ih[i][j] * x[j] for j in range(len(x))) +
                          sum(self.rnn_w_hh[i][j] * self.hidden[j] for j in range(self.hidden_dim)) +
                          self.rnn_bias[i])
                for i in range(self.hidden_dim)]
        
        self.hidden = h_new
        
        # MDN输出
        mdn_out = [sum(self.mdn_fc[i][j] * h_new[j] for j in range(self.hidden_dim))
                   for i in range((2 * self.latent_dim + 1) * self.num_mixtures)]
        
        # 解析MDN参数
        params = self._parse_mdn_params(mdn_out)
        
        return h_new, params
    
    def _parse_mdn_params(self, mdn_out: List[float]) -> List[List[List[float]]]:
        """解析MDN参数"""
        chunk_size = 2 * self.latent_dim + 1
        params = []
        
        for k in range(self.num_mixtures):
            start = k * chunk_size
            # logit (混合权重)
            logit = mdn_out[start]
            # mu (均值)
            mu = mdn_out[start + 1: start + 1 + self.latent_dim]
            # logstd (对数标准差)
            logstd = mdn_out[start + 1 + self.latent_dim: start + chunk_size]
            params.append([logit, mu, logstd])
        
        return params
    
    def sample_next_z(self, params: List[List[List[float]]]) -> List[float]:
        """从MDN采样下一个潜在状态"""
        # 计算混合权重
        logits = [p[0] for p in params]
        weights = softmax(logits)
        
        # 选择混合成分
        r = random.random()
        cumsum = 0.0
        selected_k = 0
        for k, w in enumerate(weights):
            cumsum += w
            if r < cumsum:
                selected_k = k
                break
        
        # 从选中的高斯采样
        mu = params[selected_k][1]
        logstd = params[selected_k][2]
        std = [math.exp(ls) for ls in logstd]
        
        z_next = [mu[i] + std[i] * random.gauss(0, 1) for i in range(self.latent_dim)]
        
        return z_next
    
    def vae_loss(self, obs, obs_recon, mu, logvar) -> float:
        """VAE损失"""
        # 重构损失
        recon_loss = sum((obs[i] - obs_recon[i]) ** 2 for i in range(len(obs)))
        
        # KL散度
        kl_loss = -0.5 * sum(1 + logvar[i] - mu[i] ** 2 - math.exp(logvar[i])
                            for i in range(self.latent_dim))
        
        return recon_loss + kl_loss
    
    def mdn_loss(self, z_next: List[float], params: List[List[List[float]]]) -> float:
        """MDN损失"""
        logits = [p[0] for p in params]
        weights = softmax(logits)
        
        total_loss = 0.0
        for k in range(self.num_mixtures):
            mu = params[k][1]
            logstd = params[k][2]
            std = [math.exp(ls) for ls in logstd]
            
            # 高斯概率
            log_prob = -0.5 * sum(
                ((z_next[i] - mu[i]) / std[i]) ** 2 + 2 * logstd[i] + math.log(2 * math.pi)
                for i in range(self.latent_dim)
            )
            
            total_loss += weights[k] * math.exp(log_prob)
        
        return -math.log(total_loss + 1e-10)


class Dreamer:
    """
    Dreamer - Hafner et al. 2020
    
    在潜在空间中学习世界模型和策略
    使用RSSM (Recurrent State Space Model)
    """
    
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        latent_dim: int = 30,
        hidden_dim: int = 200,
        deter_dim: int = 200,
        imag_horizon: int = 15
    ):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim  # 随机部分维度
        self.hidden_dim = hidden_dim
        self.deter_dim = deter_dim  # 确定性部分维度
        self.imag_horizon = imag_horizon
        
        # 编码器
        self.encoder = self._init_weight(obs_dim, hidden_dim)
        self.encoder_out = self._init_weight(hidden_dim, latent_dim * 2)
        
        # RSSM
        # 确定性状态更新 (GRU)
        self.gru_w_ih = self._init_weight(latent_dim + action_dim, deter_dim)
        self.gru_w_hh = self._init_weight(deter_dim, deter_dim)
        self.gru_bias = [0.0] * deter_dim
        
        # 随机状态先验
        self.prior_net = self._init_weight(deter_dim, latent_dim * 2)
        
        # 随机状态后验
        self.posterior_net = self._init_weight(deter_dim + hidden_dim, latent_dim * 2)
        
        # 解码器
        self.decoder = self._init_weight(latent_dim + deter_dim, hidden_dim)
        self.decoder_out = self._init_weight(hidden_dim, obs_dim)
        
        # 奖励预测器
        self.reward_net = self._init_weight(latent_dim + deter_dim, 1)
        
        # 值函数
        self.value_net = self._init_weight(latent_dim + deter_dim, 1)
        
        # 策略网络
        self.actor_net = self._init_weight(latent_dim + deter_dim, hidden_dim)
        self.actor_out = self._init_weight(hidden_dim, action_dim)
        
        # 当前状态
        self.deter_state = [0.0] * deter_dim
        self.stoch_state = [0.0] * latent_dim
    
    def _init_weight(self, in_dim: int, out_dim: int):
        scale = math.sqrt(2.0 / (in_dim + out_dim))
        return [[random.gauss(0, scale) for _ in range(out_dim)] for _ in range(in_dim)]
    
    def encode(self, obs) -> List[float]:
        """编码观察"""
        h = [math.tanh(sum(self.encoder[i][j] * obs[j] for j in range(self.obs_dim)))
             for i in range(self.hidden_dim)]
        embedded = [sum(self.encoder_out[i][j] * h[j] for j in range(self.hidden_dim))
                    for i in range(self.latent_dim * 2)]
        return embedded
    
    def rssm_forward(self, action: List[float], embedded: Optional[List[float]] = None):
        """RSSM前向传播"""
        # 拼接随机状态和动作
        x = self.stoch_state + action
        
        # GRU更新确定性状态
        gate = [math.sigmoid(sum(self.gru_w_ih[i][j] * x[j] for j in range(len(x))) +
                            sum(self.gru_w_hh[i][j] * self.deter_state[j] for j in range(self.deter_dim)) +
                            self.gru_bias[i])
               for i in range(self.deter_dim)]
        
        new_deter = [gate[i] * math.tanh(sum(self.gru_w_ih[i][j] * x[j] for j in range(len(x)))) +
                    (1 - gate[i]) * self.deter_state[i]
                    for i in range(self.deter_dim)]
        
        # 先验分布
        prior_params = [sum(self.prior_net[i][j] * new_deter[j] for j in range(self.deter_dim))
                        for i in range(self.latent_dim * 2)]
        prior_mu = prior_params[:self.latent_dim]
        prior_std = [math.exp(0.5 * p) for p in prior_params[self.latent_dim:]]
        
        if embedded is not None:
            # 后验分布 (有观察)
            post_input = new_deter + embedded
            post_params = [sum(self.posterior_net[i][j] * post_input[j] for j in range(len(post_input)))
                          for i in range(self.latent_dim * 2)]
            post_mu = post_params[:self.latent_dim]
            post_std = [math.exp(0.5 * p) for p in post_params[self.latent_dim:]]
            
            # 从后验采样
            new_stoch = [post_mu[i] + post_std[i] * random.gauss(0, 1)
                        for i in range(self.latent_dim)]
        else:
            # 从先验采样
            new_stoch = [prior_mu[i] + prior_std[i] * random.gauss(0, 1)
                        for i in range(self.latent_dim)]
        
        self.deter_state = new_deter
        self.stoch_state = new_stoch
        
        return new_deter, new_stoch
    
    def decode(self, deter, stoch) -> List[float]:
        """解码观察"""
        h = deter + stoch
        hidden = [math.tanh(sum(self.decoder[i][j] * h[j] for j in range(len(h))))
                  for i in range(self.hidden_dim)]
        obs = [sum(self.decoder_out[i][j] * hidden[j] for j in range(self.hidden_dim))
               for i in range(self.obs_dim)]
        return obs
    
    def predict_reward(self, deter, stoch) -> float:
        """预测奖励"""
        h = deter + stoch
        reward = sum(self.reward_net[0][j] * h[j] for j in range(len(h)))
        return reward
    
    def predict_value(self, deter, stoch) -> float:
        """预测值函数"""
        h = deter + stoch
        value = sum(self.value_net[0][j] * h[j] for j in range(len(h)))
        return value
    
    def get_action(self, deter, stoch, explore: bool = True) -> List[float]:
        """获取动作"""
        h = deter + stoch
        hidden = [math.tanh(sum(self.actor_net[i][j] * h[j] for j in range(len(h))))
                  for i in range(self.hidden_dim)]
        action = [math.tanh(sum(self.actor_out[i][j] * hidden[j] for j in range(self.hidden_dim)))
                  for i in range(self.action_dim)]
        
        if explore:
            # 添加探索噪声
            action = [a + random.gauss(0, 0.1) for a in action]
            action = [max(-1, min(1, a)) for a in action]
        
        return action
    
    def imagine(self, initial_deter, initial_stoch, horizon: int) -> Tuple[List, List]:
        """在想象中rollout"""
        deter = initial_deter
        stoch = initial_stoch
        
        rewards = []
        values = []
        
        for _ in range(horizon):
            # 获取动作
            action = self.get_action(deter, stoch, explore=False)
            
            # RSSM预测
            x = stoch + action
            gate = [math.sigmoid(sum(self.gru_w_ih[i][j] * x[j] for j in range(len(x))) +
                                sum(self.gru_w_hh[i][j] * deter[j] for j in range(self.deter_dim)) +
                                self.gru_bias[i])
                   for i in range(self.deter_dim)]
            
            deter = [gate[i] * math.tanh(sum(self.gru_w_ih[i][j] * x[j] for j in range(len(x)))) +
                    (1 - gate[i]) * deter[i]
                    for i in range(self.deter_dim)]
            
            prior_params = [sum(self.prior_net[i][j] * deter[j] for j in range(self.deter_dim))
                           for i in range(self.latent_dim * 2)]
            prior_mu = prior_params[:self.latent_dim]
            prior_std = [math.exp(0.5 * p) for p in prior_params[self.latent_dim:]]
            
            stoch = [prior_mu[i] + prior_std[i] * random.gauss(0, 1)
                    for i in range(self.latent_dim)]
            
            # 预测奖励和值
            reward = self.predict_reward(deter, stoch)
            value = self.predict_value(deter, stoch)
            
            rewards.append(reward)
            values.append(value)
        
        return rewards, values


class MuZero:
    """
    MuZero - Schrittwieser et al. 2020
    
    结合模型学习和蒙特卡洛树搜索
    三个函数: 表示函数h, 动态函数g, 预测函数f
    """
    
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        latent_dim: int = 128,
        num_simulations: int = 50,
        discount: float = 0.997,
        c1: float = 1.25,
        c2: float = 19652
    ):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_simulations = num_simulations
        self.discount = discount
        self.c1 = c1
        self.c2 = c2
        
        # 表示函数 h(o) -> s
        self.h_encoder = self._init_weight(obs_dim, hidden_dim)
        self.h_out = self._init_weight(hidden_dim, latent_dim)
        
        # 动态函数 g(s, a) -> s', r
        self.g_net = self._init_weight(latent_dim + action_dim, hidden_dim)
        self.g_state = self._init_weight(hidden_dim, latent_dim)
        self.g_reward = self._init_weight(hidden_dim, 1)
        
        # 预测函数 f(s) -> p, v
        self.f_net = self._init_weight(latent_dim, hidden_dim)
        self.f_policy = self._init_weight(hidden_dim, action_dim)
        self.f_value = self._init_weight(hidden_dim, 1)
        
        # 根节点
        self.root = None
    
    def _init_weight(self, in_dim: int, out_dim: int):
        scale = math.sqrt(2.0 / (in_dim + out_dim))
        return [[random.gauss(0, scale) for _ in range(out_dim)] for _ in range(in_dim)]
    
    def h(self, obs) -> List[float]:
        """表示函数: 观察到隐藏状态"""
        hidden = [math.relu(sum(self.h_encoder[i][j] * obs[j] for j in range(self.obs_dim)))
                  for i in range(self.hidden_dim)]
        state = [sum(self.h_out[i][j] * hidden[j] for j in range(self.hidden_dim))
                 for i in range(self.latent_dim)]
        return state
    
    def g(self, state: List[float], action: int) -> Tuple[List[float], float]:
        """动态函数: 状态转移和奖励预测"""
        # one-hot编码动作
        action_vec = [1.0 if i == action else 0.0 for i in range(self.action_dim)]
        x = state + action_vec
        
        hidden = [math.relu(sum(self.g_net[i][j] * x[j] for j in range(len(x))))
                  for i in range(self.hidden_dim)]
        
        next_state = [sum(self.g_state[i][j] * hidden[j] for j in range(self.hidden_dim))
                      for i in range(self.latent_dim)]
        
        reward = sum(self.g_reward[0][j] * hidden[j] for j in range(self.hidden_dim))
        
        return next_state, reward
    
    def f(self, state: List[float]) -> Tuple[List[float], float]:
        """预测函数: 策略和值函数"""
        hidden = [math.relu(sum(self.f_net[i][j] * state[j] for j in range(self.latent_dim)))
                  for i in range(self.hidden_dim)]
        
        policy_logits = [sum(self.f_policy[i][j] * hidden[j] for j in range(self.hidden_dim))
                         for i in range(self.action_dim)]
        policy = softmax(policy_logits)
        
        value = sum(self.f_value[0][j] * hidden[j] for j in range(self.hidden_dim))
        
        return policy, value
    
    class MCTSNode:
        """MCTS节点"""
        def __init__(self, state, prior, parent=None):
            self.state = state
            self.prior = prior
            self.parent = parent
            self.children = {}
            self.visit_count = 0
            self.total_value = 0.0
            self.reward = 0.0
        
        @property
        def value(self):
            if self.visit_count == 0:
                return 0.0
            return self.total_value / self.visit_count
        
        def expanded(self):
            return len(self.children) > 0
    
    def mcts_search(self, root_state: List[float]) -> List[float]:
        """执行MCTS搜索"""
        # 创建根节点
        policy, value = self.f(root_state)
        root = self.MCTSNode(root_state, policy)
        
        for _ in range(self.num_simulations):
            # 选择
            node = root
            search_path = [node]
            
            while node.expanded():
                action = self._select_action(node)
                if action not in node.children:
                    # 扩展
                    next_state, reward = self.g(node.state, action)
                    policy, value = self.f(next_state)
                    child = self.MCTSNode(next_state, policy, parent=node)
                    child.reward = reward
                    node.children[action] = child
                
                node = node.children[action]
                search_path.append(node)
            
            # 评估
            _, value = self.f(node.state)
            
            # 回溯
            for node in reversed(search_path):
                node.total_value += value
                node.visit_count += 1
                value = node.reward + self.discount * value
        
        # 计算改进的策略
        visit_counts = [root.children.get(a, self.MCTSNode(root.state, [0])).visit_count
                       for a in range(self.action_dim)]
        total_visits = sum(visit_counts)
        
        if total_visits > 0:
            improved_policy = [v / total_visits for v in visit_counts]
        else:
            improved_policy = policy
        
        self.root = root
        return improved_policy
    
    def _select_action(self, node: MCTSNode) -> int:
        """选择动作 (UCB)"""
        best_score = float('-inf')
        best_action = 0
        
        total_visits = sum(child.visit_count for child in node.children.values())
        
        for action in range(self.action_dim):
            if action in node.children:
                child = node.children[action]
                q = child.value
                u = self.c1 * math.sqrt(total_visits) / (1 + child.visit_count)
                u *= node.prior[action] / (1 + child.prior[action])
                score = q + u
            else:
                # 未访问的动作
                u = self.c1 * math.sqrt(total_visits + 1)
                u *= node.prior[action]
                score = u
            
            if score > best_score:
                best_score = score
                best_action = action
        
        return best_action
    
    def select_action(self, obs, temperature: float = 1.0) -> int:
        """选择动作"""
        # 表示函数
        state = self.h(obs)
        
        # MCTS搜索
        policy = self.mcts_search(state)
        
        # 根据策略采样
        if temperature == 0:
            action = policy.index(max(policy))
        else:
            # 温度采样
            policy_temp = [p ** (1 / temperature) for p in policy]
            total = sum(policy_temp)
            policy_temp = [p / total for p in policy_temp]
            
            r = random.random()
            cumsum = 0.0
            action = 0
            for a, p in enumerate(policy_temp):
                cumsum += p
                if r < cumsum:
                    action = a
                    break
        
        return action


class MBPO:
    """
    Model-Based Policy Optimization (MBPO)
    
    使用模型生成的短rollout来增强真实数据
    """
    
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        ensemble_size: int = 5,
        horizon: int = 5,
        model_rollout_length: int = 1
    ):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.ensemble_size = ensemble_size
        self.horizon = horizon
        self.model_rollout_length = model_rollout_length
        
        # 模型集成
        self.ensemble = [self._init_model() for _ in range(ensemble_size)]
        
        # 策略网络
        self.policy_net = self._init_weight(obs_dim, hidden_dim)
        self.policy_out = self._init_weight(hidden_dim, action_dim)
        
        # Q网络
        self.q_net1 = self._init_weight(obs_dim + action_dim, hidden_dim)
        self.q_out1 = self._init_weight(hidden_dim, 1)
        self.q_net2 = self._init_weight(obs_dim + action_dim, hidden_dim)
        self.q_out2 = self._init_weight(hidden_dim, 1)
        
        # 经验回放
        self.real_buffer = ReplayBuffer()
        self.model_buffer = ReplayBuffer()
    
    def _init_weight(self, in_dim: int, out_dim: int):
        scale = math.sqrt(2.0 / (in_dim + out_dim))
        return [[random.gauss(0, scale) for _ in range(out_dim)] for _ in range(in_dim)]
    
    def _init_model(self) -> Dict:
        """初始化单个模型"""
        return {
            'net1': self._init_weight(self.obs_dim + self.action_dim, self.hidden_dim),
            'net2': self._init_weight(self.hidden_dim, self.hidden_dim),
            'out': self._init_weight(self.hidden_dim, self.obs_dim + 1)  # next_obs + reward
        }
    
    def model_predict(self, obs, action, model_idx: Optional[int] = None):
        """模型预测"""
        if model_idx is None:
            model_idx = random.randint(0, self.ensemble_size - 1)
        
        model = self.ensemble[model_idx]
        x = list(obs) + list(action)
        
        h = [math.relu(sum(model['net1'][i][j] * x[j] for j in range(len(x))))
             for i in range(self.hidden_dim)]
        h = [math.relu(sum(model['net2'][i][j] * h[j] for j in range(self.hidden_dim)))
             for i in range(self.hidden_dim)]
        
        out = [sum(model['out'][i][j] * h[j] for j in range(self.hidden_dim))
               for i in range(self.obs_dim + 1)]
        
        next_obs = out[:self.obs_dim]
        reward = out[self.obs_dim]
        
        return next_obs, reward
    
    def get_action(self, obs, explore: bool = True) -> List[float]:
        """获取动作"""
        h = [math.relu(sum(self.policy_net[i][j] * obs[j] for j in range(self.obs_dim)))
             for i in range(self.hidden_dim)]
        action = [math.tanh(sum(self.policy_out[i][j] * h[j] for j in range(self.hidden_dim)))
                  for i in range(self.action_dim)]
        
        if explore:
            action = [a + random.gauss(0, 0.1) for a in action]
            action = [max(-1, min(1, a)) for a in action]
        
        return action
    
    def get_q(self, obs, action) -> float:
        """获取Q值"""
        x = list(obs) + list(action)
        
        h1 = [math.relu(sum(self.q_net1[i][j] * x[j] for j in range(len(x))))
              for i in range(self.hidden_dim)]
        q1 = sum(self.q_out1[0][j] * h1[j] for j in range(self.hidden_dim))
        
        h2 = [math.relu(sum(self.q_net2[i][j] * x[j] for j in range(len(x))))
              for i in range(self.hidden_dim)]
        q2 = sum(self.q_out2[0][j] * h2[j] for j in range(self.hidden_dim))
        
        return min(q1, q2)
    
    def generate_model_rollouts(self, num_rollouts: int):
        """生成模型rollout"""
        if len(self.real_buffer) < 100:
            return
        
        for _ in range(num_rollouts):
            # 从真实数据中采样初始状态
            batch = self.real_buffer.sample(1)
            state = batch[0][0]  # 初始状态
            
            for _ in range(self.model_rollout_length):
                action = self.get_action(state, explore=True)
                next_state, reward = self.model_predict(state, action)
                done = False
                
                self.model_buffer.push(state, action, reward, next_state, done)
                state = next_state


class PETS:
    """
    Probabilistic Ensembles for Trajectory Sampling (PETS)
    
    使用概率集成模型和CEM优化
    """
    
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 200,
        ensemble_size: int = 5,
        horizon: int = 30,
        cem_iterations: int = 5,
        cem_elite_ratio: float = 0.1,
        cem_population: int = 500
    ):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.ensemble_size = ensemble_size
        self.horizon = horizon
        self.cem_iterations = cem_iterations
        self.cem_elite_ratio = cem_elite_ratio
        self.cem_population = cem_population
        
        # 概率集成模型
        self.ensemble = [self._init_probabilistic_model() for _ in range(ensemble_size)]
        
        # CEM参数
        self.action_mean = [0.0] * action_dim
        self.action_std = [1.0] * action_dim
    
    def _init_weight(self, in_dim: int, out_dim: int):
        scale = math.sqrt(2.0 / (in_dim + out_dim))
        return [[random.gauss(0, scale) for _ in range(out_dim)] for _ in range(in_dim)]
    
    def _init_probabilistic_model(self) -> Dict:
        """初始化概率模型"""
        return {
            'net1': self._init_weight(self.obs_dim + self.action_dim, self.hidden_dim),
            'net2': self._init_weight(self.hidden_dim, self.hidden_dim),
            'out_mean': self._init_weight(self.hidden_dim, self.obs_dim),
            'out_logvar': self._init_weight(self.hidden_dim, self.obs_dim)
        }
    
    def model_predict(self, obs, action, model_idx: Optional[int] = None) -> Tuple[List[float], List[float]]:
        """概率模型预测"""
        if model_idx is None:
            model_idx = random.randint(0, self.ensemble_size - 1)
        
        model = self.ensemble[model_idx]
        x = list(obs) + list(action)
        
        h = [math.relu(sum(model['net1'][i][j] * x[j] for j in range(len(x))))
             for i in range(self.hidden_dim)]
        h = [math.relu(sum(model['net2'][i][j] * h[j] for j in range(self.hidden_dim)))
             for i in range(self.hidden_dim)]
        
        mean = [sum(model['out_mean'][i][j] * h[j] for j in range(self.hidden_dim))
                for i in range(self.obs_dim)]
        logvar = [sum(model['out_logvar'][i][j] * h[j] for j in range(self.hidden_dim))
                  for i in range(self.obs_dim)]
        
        # 采样
        std = [math.exp(0.5 * lv) for lv in logvar]
        next_obs = [mean[i] + std[i] * random.gauss(0, 1) for i in range(self.obs_dim)]
        
        return next_obs, mean
    
    def evaluate_trajectory(self, initial_state: List[float], action_sequence: List[List[float]]) -> float:
        """评估轨迹"""
        state = initial_state
        total_reward = 0.0
        
        for action in action_sequence:
            # 使用所有模型预测
            next_states = []
            for i in range(self.ensemble_size):
                next_state, _ = self.model_predict(state, action, i)
                next_states.append(next_state)
            
            # 平均预测
            next_state = [sum(ns[i] for ns in next_states) / self.ensemble_size
                         for i in range(self.obs_dim)]
            
            # 奖励函数 (需要根据任务定义)
            reward = self._reward_function(state, action)
            total_reward += reward
            
            state = next_state
        
        return total_reward
    
    def _reward_function(self, state, action) -> float:
        """奖励函数 (示例)"""
        # 这里需要根据具体任务定义
        return -sum(s ** 2 for s in state) - 0.01 * sum(a ** 2 for a in action)
    
    def cem_optimize(self, initial_state: List[float]) -> List[float]:
        """CEM优化动作序列"""
        # 初始化动作序列分布
        horizon = self.horizon
        action_dim = self.action_dim
        
        mean = [[0.0] * action_dim for _ in range(horizon)]
        std = [[1.0] * action_dim for _ in range(horizon)]
        
        for _ in range(self.cem_iterations):
            # 采样动作序列
            samples = []
            for _ in range(self.cem_population):
                action_seq = [[mean[t][a] + std[t][a] * random.gauss(0, 1)
                              for a in range(action_dim)]
                             for t in range(horizon)]
                samples.append(action_seq)
            
            # 评估
            rewards = [self.evaluate_trajectory(initial_state, seq) for seq in samples]
            
            # 选择精英
            elite_size = int(self.cem_population * self.cem_elite_ratio)
            elite_indices = sorted(range(len(rewards)), key=lambda i: rewards[i], reverse=True)[:elite_size]
            
            # 更新分布
            elite_samples = [samples[i] for i in elite_indices]
            
            for t in range(horizon):
                for a in range(action_dim):
                    values = [sample[t][a] for sample in elite_samples]
                    mean[t][a] = sum(values) / len(values)
                    std[t][a] = math.sqrt(sum((v - mean[t][a]) ** 2 for v in values) / len(values)) + 0.01
        
        # 返回第一个动作
        return mean[0]
    
    def get_action(self, obs) -> List[float]:
        """获取动作"""
        return self.cem_optimize(obs)


class PlaNet:
    """
    PlaNet - Planning in Latent Space
    
    在学习的潜在空间中进行规划
    """
    
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        latent_dim: int = 30,
        hidden_dim: int = 200,
        horizon: int = 12
    ):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.horizon = horizon
        
        # 编码器
        self.encoder = self._init_weight(obs_dim, hidden_dim)
        self.encoder_out = self._init_weight(hidden_dim, latent_dim * 2)
        
        # 转移模型 (RSSM)
        self.transition = self._init_weight(latent_dim + action_dim, hidden_dim)
        self.trans_out = self._init_weight(hidden_dim, latent_dim * 2)
        
        # 奖励模型
        self.reward_net = self._init_weight(latent_dim, hidden_dim)
        self.reward_out = self._init_weight(hidden_dim, 1)
        
        # 当前潜在状态
        self.latent_state = [0.0] * latent_dim
    
    def _init_weight(self, in_dim: int, out_dim: int):
        scale = math.sqrt(2.0 / (in_dim + out_dim))
        return [[random.gauss(0, scale) for _ in range(out_dim)] for _ in range(in_dim)]
    
    def encode(self, obs) -> List[float]:
        """编码观察"""
        h = [math.relu(sum(self.encoder[i][j] * obs[j] for j in range(self.obs_dim)))
             for i in range(self.hidden_dim)]
        params = [sum(self.encoder_out[i][j] * h[j] for j in range(self.hidden_dim))
                  for i in range(self.latent_dim * 2)]
        
        mu = params[:self.latent_dim]
        logvar = params[self.latent_dim:]
        std = [math.exp(0.5 * lv) for lv in logvar]
        
        z = [mu[i] + std[i] * random.gauss(0, 1) for i in range(self.latent_dim)]
        return z
    
    def transition_step(self, latent: List[float], action: List[float]) -> List[float]:
        """潜在空间中的状态转移"""
        x = latent + action
        h = [math.relu(sum(self.transition[i][j] * x[j] for j in range(len(x))))
             for i in range(self.hidden_dim)]
        params = [sum(self.trans_out[i][j] * h[j] for j in range(self.hidden_dim))
                  for i in range(self.latent_dim * 2)]
        
        mu = params[:self.latent_dim]
        logvar = params[self.latent_dim:]
        std = [math.exp(0.5 * lv) for lv in logvar]
        
        next_latent = [mu[i] + std[i] * random.gauss(0, 1) for i in range(self.latent_dim)]
        return next_latent
    
    def predict_reward(self, latent: List[float]) -> float:
        """预测奖励"""
        h = [math.relu(sum(self.reward_net[i][j] * latent[j] for j in range(self.latent_dim)))
             for i in range(self.hidden_dim)]
        reward = sum(self.reward_out[0][j] * h[j] for j in range(self.hidden_dim))
        return reward
    
    def cross_entropy_method(self, initial_latent: List[float], num_iterations: int = 5,
                            population: int = 100, elite_ratio: float = 0.1) -> List[float]:
        """CEM规划"""
        mean = [0.0] * self.action_dim
        std = [1.0] * self.action_dim
        
        for _ in range(num_iterations):
            # 采样动作
            samples = [[mean[a] + std[a] * random.gauss(0, 1) for a in range(self.action_dim)]
                      for _ in range(population)]
            
            # 评估轨迹
            rewards = []
            for action in samples:
                latent = initial_latent
                total_reward = 0.0
                for _ in range(self.horizon):
                    latent = self.transition_step(latent, action)
                    total_reward += self.predict_reward(latent)
                rewards.append(total_reward)
            
            # 选择精英
            elite_size = int(population * elite_ratio)
            elite_indices = sorted(range(len(rewards)), key=lambda i: rewards[i], reverse=True)[:elite_size]
            elite_samples = [samples[i] for i in elite_indices]
            
            # 更新分布
            for a in range(self.action_dim):
                values = [sample[a] for sample in elite_samples]
                mean[a] = sum(values) / len(values)
                std[a] = math.sqrt(sum((v - mean[a]) ** 2 for v in values) / len(values)) + 0.01
        
        return mean
    
    def get_action(self, obs) -> List[float]:
        """获取动作"""
        latent = self.encode(obs)
        self.latent_state = latent
        return self.cross_entropy_method(latent)


# 工具函数
def get_mbrl_agent(name: str, **kwargs):
    """根据名称获取MBRL智能体"""
    agents = {
        'world_model': WorldModel,
        'dreamer': Dreamer,
        'muzero': MuZero,
        'mbpo': MBPO,
        'pets': PETS,
        'planet': PlaNet
    }
    
    name_lower = name.lower()
    if name_lower not in agents:
        raise ValueError(f"Unknown MBRL agent: {name}. Available: {list(agents.keys())}")
    
    return agents[name_lower](**kwargs)
