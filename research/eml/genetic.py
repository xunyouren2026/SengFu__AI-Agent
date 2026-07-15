"""
EML Genetic Programming - EML遗传编程模块

基于遗传编程(GP)的EML算子自动发现与优化。
通过进化算法搜索最优的EML组合表达式，用于函数逼近和符号回归。

核心思想:
- 将EML算子组合表示为表达式树
- 使用遗传编程(选择、交叉、变异)进化种群
- 通过适应度函数评估表达式质量
- 自动发现有效的EML数学表达式

⚠️ 研究用途警告: 本模块为实验性实现
"""

import math
import random
import copy
import time
import statistics
from typing import Optional, List, Dict, Tuple, Any, Callable
from dataclasses import dataclass, field
from enum import Enum, auto


# ============================================================
# 表达式树节点定义
# ============================================================

class NodeType(Enum):
    """表达式树节点类型"""
    CONSTANT = auto()    # 常数节点
    VARIABLE = auto()    # 变量节点 (x)
    EML = auto()         # EML算子节点 eml(left, right)
    ADD = auto()         # 加法节点
    SUB = auto()         # 减法节点
    MUL = auto()         # 乘法节点
    NEG = auto()         # 取负节点 (一元)


class ExpressionNode:
    """
    表达式树节点

    表示遗传编程中的单个基因/表达式节点。
    每个节点可以是常数、变量或运算符。
    """

    def __init__(self, node_type: NodeType, value: float = 0.0,
                 left: Optional['ExpressionNode'] = None,
                 right: Optional['ExpressionNode'] = None):
        self.node_type = node_type
        self.value = value
        self.left = left
        self.right = right

    def copy(self) -> 'ExpressionNode':
        """深拷贝表达式树"""
        return ExpressionNode(
            node_type=self.node_type,
            value=self.value,
            left=self.left.copy() if self.left else None,
            right=self.right.copy() if self.right else None,
        )

    def depth(self) -> int:
        """计算表达式树深度"""
        if self.node_type in (NodeType.CONSTANT, NodeType.VARIABLE):
            return 1
        elif self.node_type == NodeType.NEG:
            return 1 + (self.left.depth() if self.left else 0)
        else:
            left_d = self.left.depth() if self.left else 0
            right_d = self.right.depth() if self.right else 0
            return 1 + max(left_d, right_d)

    def size(self) -> int:
        """计算表达式树节点总数"""
        count = 1
        if self.left:
            count += self.left.size()
        if self.right:
            count += self.right.size()
        return count

    def to_string(self) -> str:
        """将表达式树转换为可读字符串"""
        if self.node_type == NodeType.CONSTANT:
            return f"{self.value:.4f}"
        elif self.node_type == NodeType.VARIABLE:
            return "x"
        elif self.node_type == NodeType.EML:
            left_s = self.left.to_string() if self.left else "?"
            right_s = self.right.to_string() if self.right else "?"
            return f"eml({left_s}, {right_s})"
        elif self.node_type == NodeType.ADD:
            left_s = self.left.to_string() if self.left else "?"
            right_s = self.right.to_string() if self.right else "?"
            return f"({left_s} + {right_s})"
        elif self.node_type == NodeType.SUB:
            left_s = self.left.to_string() if self.left else "?"
            right_s = self.right.to_string() if self.right else "?"
            return f"({left_s} - {right_s})"
        elif self.node_type == NodeType.MUL:
            left_s = self.left.to_string() if self.left else "?"
            right_s = self.right.to_string() if self.right else "?"
            return f"({left_s} * {right_s})"
        elif self.node_type == NodeType.NEG:
            left_s = self.left.to_string() if self.left else "?"
            return f"(-{left_s})"
        return "?"

    def evaluate(self, x: float) -> float:
        """
        在给定x值下求值表达式树

        Args:
            x: 变量值

        Returns:
            表达式求值结果，出错时返回0.0
        """
        try:
            if self.node_type == NodeType.CONSTANT:
                return self.value
            elif self.node_type == NodeType.VARIABLE:
                return x
            elif self.node_type == NodeType.EML:
                left_val = self.left.evaluate(x) if self.left else 0.0
                right_val = self.right.evaluate(x) if self.right else 1.0
                # EML: e^left - ln(right)
                if right_val <= 1e-10:
                    right_val = 1e-10  # 安全下界
                exp_val = math.exp(min(left_val, 700))  # 防止溢出
                log_val = math.log(right_val)
                result = exp_val - log_val
                # 限制输出范围
                if math.isnan(result) or math.isinf(result):
                    return 0.0
                return max(-1e10, min(1e10, result))
            elif self.node_type == NodeType.ADD:
                left_val = self.left.evaluate(x) if self.left else 0.0
                right_val = self.right.evaluate(x) if self.right else 0.0
                return left_val + right_val
            elif self.node_type == NodeType.SUB:
                left_val = self.left.evaluate(x) if self.left else 0.0
                right_val = self.right.evaluate(x) if self.right else 0.0
                return left_val - right_val
            elif self.node_type == NodeType.MUL:
                left_val = self.left.evaluate(x) if self.left else 0.0
                right_val = self.right.evaluate(x) if self.right else 0.0
                return left_val * right_val
            elif self.node_type == NodeType.NEG:
                left_val = self.left.evaluate(x) if self.left else 0.0
                return -left_val
        except (OverflowError, ValueError, ZeroDivisionError):
            return 0.0
        return 0.0

    def get_all_nodes(self) -> List[Tuple['ExpressionNode', Optional['ExpressionNode'], str]]:
        """
        获取所有节点及其父节点和方向

        Returns:
            列表，每个元素为 (节点, 父节点, 方向) 其中方向为 'left'/'right'/None
        """
        nodes = []
        self._collect_nodes(nodes, None, None)
        return nodes

    def _collect_nodes(self, nodes: list, parent: Optional['ExpressionNode'], direction: Optional[str]):
        """递归收集所有节点"""
        nodes.append((self, parent, direction))
        if self.left:
            self.left._collect_nodes(nodes, self, 'left')
        if self.right:
            self.right._collect_nodes(nodes, self, 'right')


# ============================================================
# 遗传编程配置
# ============================================================

@dataclass
class GPConfig:
    """遗传编程配置参数"""

    # 种群参数
    population_size: int = 100           # 种群大小
    max_generations: int = 200           # 最大进化代数
    max_tree_depth: int = 6              # 表达式树最大深度
    min_tree_depth: int = 2              # 表达式树最小深度

    # 遗传操作参数
    crossover_rate: float = 0.8          # 交叉概率
    mutation_rate: float = 0.15          # 变异概率
    reproduction_rate: float = 0.05      # 直接复制概率
    elitism_count: int = 2               # 精英保留数量

    # 变异参数
    constant_mutation_range: float = 2.0 # 常数变异范围
    subtree_mutation_depth: int = 3      # 子树变异最大深度

    # 选择参数
    tournament_size: int = 5             # 锦标赛选择大小

    # 终止条件
    fitness_threshold: float = 1e-6      # 适应度阈值（达到后停止）
    patience: int = 20                   # 连续无改善代数（达到后停止）

    # 随机种子
    seed: Optional[int] = None


# ============================================================
# 适应度评估结果
# ============================================================

@dataclass
class FitnessResult:
    """适应度评估结果"""
    fitness: float               # 适应度值（越小越好）
    mse: float                   # 均方误差
    mae: float                   # 平均绝对误差
    max_error: float             # 最大误差
    valid_ratio: float           # 有效输出比例
    expression_complexity: int   # 表达式复杂度（节点数）


# ============================================================
# 进化统计信息
# ============================================================

@dataclass
class EvolutionStats:
    """进化过程统计信息"""
    generation: int = 0
    best_fitness: float = float('inf')
    avg_fitness: float = float('inf')
    worst_fitness: float = float('inf')
    best_expression: str = ""
    diversity: float = 0.0       # 种群多样性
    elapsed_time: float = 0.0


# ============================================================
# EML遗传编程主类
# ============================================================

class EMLGeneticProgramming:
    """
    EML遗传编程引擎

    使用遗传编程自动发现和优化EML算子的组合表达式。
    适用于符号回归、函数逼近等任务。

    工作流程:
    1. 初始化随机种群（随机表达式树）
    2. 评估每个个体的适应度
    3. 选择优秀个体
    4. 通过交叉、变异产生新个体
    5. 替换旧种群，重复进化

    使用示例:
        >>> gp = EMLGeneticProgramming()
        >>> # 定义目标函数
        >>> target_fn = lambda x: math.exp(x) - math.log(x + 1)
        >>> # 准备训练数据
        >>> train_x = [i * 0.1 for i in range(1, 50)]
        >>> train_y = [target_fn(xi) for xi in train_x]
        >>> # 运行进化
        >>> result = gp.evolve(train_x, train_y)
        >>> print(result['best_expression'])
    """

    def __init__(self, config: Optional[GPConfig] = None):
        """
        初始化遗传编程引擎

        Args:
            config: 遗传编程配置，为None时使用默认配置
        """
        self.config = config or GPConfig()

        # 设置随机种子
        if self.config.seed is not None:
            random.seed(self.config.seed)

        # 种群：每个个体是一棵表达式树
        self.population: List[ExpressionNode] = []

        # 进化历史记录
        self.history: List[EvolutionStats] = []

        # 当前最优个体
        self.best_individual: Optional[ExpressionNode] = None
        self.best_fitness: float = float('inf')

    # ----------------------------------------------------------
    # 种群初始化
    # ----------------------------------------------------------

    def initialize_population(self) -> List[ExpressionNode]:
        """
        初始化种群

        使用混合策略生成初始种群:
        - 一半使用满方法(grow)生成，保证多样性
        - 一半使用完全方法(full)生成，保证结构完整

        Returns:
            初始化后的种群列表
        """
        self.population = []
        half = self.config.population_size // 2

        # 满方法(grow)：允许不同深度的分支
        for _ in range(half):
            tree = self._generate_tree_grow(
                max_depth=self.config.max_tree_depth,
                min_depth=self.config.min_tree_depth
            )
            self.population.append(tree)

        # 完全方法(full)：所有分支达到最大深度
        for _ in range(self.config.population_size - half):
            tree = self._generate_tree_full(
                max_depth=self.config.max_tree_depth
            )
            self.population.append(tree)

        return self.population

    def _generate_tree_grow(self, max_depth: int,
                            min_depth: int = 1,
                            current_depth: int = 0) -> ExpressionNode:
        """
        使用grow方法生成随机表达式树

        grow方法允许不同分支有不同深度，增加多样性。

        Args:
            max_depth: 最大允许深度
            min_depth: 最小深度
            current_depth: 当前深度

        Returns:
            随机生成的表达式树根节点
        """
        # 终止条件：达到最大深度或随机决定终止
        if current_depth >= max_depth:
            return self._generate_terminal()

        if current_depth >= min_depth and random.random() < 0.3:
            return self._generate_terminal()

        # 选择运算符
        node_type = self._random_operator()
        node = ExpressionNode(node_type=node_type)

        if node_type == NodeType.NEG:
            # 一元运算符：只有一个子节点
            node.left = self._generate_tree_grow(
                max_depth, min_depth, current_depth + 1
            )
        elif node_type == NodeType.EML:
            # EML是二元运算符
            node.left = self._generate_tree_grow(
                max_depth, min_depth, current_depth + 1
            )
            # EML的右操作数应尽量为正，使用常数或变量
            node.right = self._generate_tree_grow(
                max_depth, min_depth, current_depth + 1
            )
        else:
            # 二元运算符
            node.left = self._generate_tree_grow(
                max_depth, min_depth, current_depth + 1
            )
            node.right = self._generate_tree_grow(
                max_depth, min_depth, current_depth + 1
            )

        return node

    def _generate_tree_full(self, max_depth: int,
                            current_depth: int = 0) -> ExpressionNode:
        """
        使用full方法生成随机表达式树

        full方法确保所有分支都达到指定深度，生成结构完整的树。

        Args:
            max_depth: 目标深度
            current_depth: 当前深度

        Returns:
            随机生成的表达式树根节点
        """
        # 终止条件：达到最大深度时生成终端节点
        if current_depth >= max_depth:
            return self._generate_terminal()

        # 选择运算符
        node_type = self._random_operator()
        node = ExpressionNode(node_type=node_type)

        if node_type == NodeType.NEG:
            node.left = self._generate_tree_full(max_depth, current_depth + 1)
        else:
            node.left = self._generate_tree_full(max_depth, current_depth + 1)
            node.right = self._generate_tree_full(max_depth, current_depth + 1)

        return node

    def _generate_terminal(self) -> ExpressionNode:
        """
        生成终端节点（常数或变量）

        Returns:
            终端节点
        """
        if random.random() < 0.5:
            # 变量节点
            return ExpressionNode(node_type=NodeType.VARIABLE)
        else:
            # 随机常数节点，范围[-5, 5]
            value = random.uniform(-5.0, 5.0)
            return ExpressionNode(node_type=NodeType.CONSTANT, value=value)

    def _random_operator(self) -> NodeType:
        """
        随机选择一个运算符节点类型

        EML算子有更高的权重，因为这是EML遗传编程的核心。

        Returns:
            随机选择的节点类型
        """
        # 运算符权重：EML权重更高
        operators = [
            NodeType.EML, NodeType.EML,  # EML权重加倍
            NodeType.ADD,
            NodeType.SUB,
            NodeType.MUL,
            NodeType.NEG,
        ]
        return random.choice(operators)

    # ----------------------------------------------------------
    # 适应度评估
    # ----------------------------------------------------------

    def evaluate_fitness(self, individual: ExpressionNode,
                         train_x: List[float],
                         train_y: List[float]) -> FitnessResult:
        """
        评估单个个体的适应度

        使用均方误差(MSE)作为主要适应度指标，
        同时考虑表达式复杂度作为惩罚项。

        Args:
            individual: 待评估的表达式树
            train_x: 训练输入数据
            train_y: 训练目标输出

        Returns:
            适应度评估结果
        """
        errors = []
        valid_count = 0

        for xi, yi in zip(train_x, train_y):
            try:
                predicted = individual.evaluate(xi)
                if math.isfinite(predicted):
                    error = (predicted - yi) ** 2
                    errors.append(error)
                    valid_count += 1
                else:
                    errors.append(1e10)  # 无效输出给予大惩罚
            except Exception:
                errors.append(1e10)

        # 计算误差指标
        if errors:
            mse = statistics.mean(errors)
            mae = statistics.mean([math.sqrt(e) for e in errors])
            max_error = max(errors)
        else:
            mse = 1e10
            mae = 1e10
            max_error = 1e10

        valid_ratio = valid_count / len(train_x) if train_x else 0.0

        # 复杂度惩罚：节点数越多，惩罚越大
        complexity = individual.size()
        complexity_penalty = 0.001 * complexity

        # 综合适应度 = MSE + 复杂度惩罚
        fitness = mse + complexity_penalty

        return FitnessResult(
            fitness=fitness,
            mse=mse,
            mae=mae,
            max_error=max_error,
            valid_ratio=valid_ratio,
            expression_complexity=complexity,
        )

    def _evaluate_population(self, train_x: List[float],
                             train_y: List[float]) -> List[FitnessResult]:
        """
        评估整个种群的适应度

        Args:
            train_x: 训练输入
            train_y: 训练目标

        Returns:
            适应度结果列表，与种群顺序对应
        """
        results = []
        for individual in self.population:
            fitness = self.evaluate_fitness(individual, train_x, train_y)
            results.append(fitness)
        return results

    # ----------------------------------------------------------
    # 选择操作
    # ----------------------------------------------------------

    def tournament_selection(self, fitness_results: List[FitnessResult]) -> ExpressionNode:
        """
        锦标赛选择

        从种群中随机选取tournament_size个个体，
        选择适应度最好的一个作为父代。

        Args:
            fitness_results: 种群适应度结果列表

        Returns:
            选中的个体（表达式树）
        """
        # 随机选取参赛者索引
        indices = random.sample(
            range(len(self.population)),
            min(self.config.tournament_size, len(self.population))
        )

        # 找到适应度最好的参赛者
        best_idx = min(indices, key=lambda i: fitness_results[i].fitness)
        return self.population[best_idx].copy()

    def roulette_selection(self, fitness_results: List[FitnessResult]) -> ExpressionNode:
        """
        轮盘赌选择

        适应度越好的个体被选中的概率越高。
        使用适应度的倒数作为权重（因为适应度越小越好）。

        Args:
            fitness_results: 种群适应度结果列表

        Returns:
            选中的个体
        """
        # 计算选择权重（适应度倒数）
        weights = []
        for fr in fitness_results:
            # 使用倒数，加小常数避免除零
            w = 1.0 / (fr.fitness + 1e-10)
            weights.append(w)

        total_weight = sum(weights)
        if total_weight == 0:
            # 所有权重为零时，均匀随机选择
            return random.choice(self.population).copy()

        # 归一化
        weights = [w / total_weight for w in weights]

        # 轮盘赌选择
        r = random.random()
        cumulative = 0.0
        for i, w in enumerate(weights):
            cumulative += w
            if r <= cumulative:
                return self.population[i].copy()

        return self.population[-1].copy()

    # ----------------------------------------------------------
    # 交叉操作
    # ----------------------------------------------------------

    def crossover(self, parent1: ExpressionNode,
                  parent2: ExpressionNode) -> Tuple[ExpressionNode, ExpressionNode]:
        """
        子树交叉操作

        随机选择两个父代的子树进行交换，产生两个子代。

        Args:
            parent1: 第一个父代
            parent2: 第二个父代

        Returns:
            两个子代个体
        """
        child1 = parent1.copy()
        child2 = parent2.copy()

        # 获取所有节点
        nodes1 = child1.get_all_nodes()
        nodes2 = child2.get_all_nodes()

        if not nodes1 or not nodes2:
            return child1, child2

        # 随机选择交叉点
        cross_idx1 = random.randint(0, len(nodes1) - 1)
        cross_idx2 = random.randint(0, len(nodes2) - 1)

        node1, parent1_node, dir1 = nodes1[cross_idx1]
        node2, parent2_node, dir2 = nodes2[cross_idx2]

        # 检查交叉后深度是否超限
        depth1_remaining = self.config.max_tree_depth - self._depth_to_node(child1, node1)
        depth2_remaining = self.config.max_tree_depth - self._depth_to_node(child2, node2)

        if node2.depth() > depth1_remaining or node1.depth() > depth2_remaining:
            # 交叉后深度超限，放弃本次交叉
            return child1, child2

        # 执行交换
        if parent1_node is None:
            # node1是根节点
            child1 = node2.copy()
        elif dir1 == 'left':
            parent1_node.left = node2.copy()
        else:
            parent1_node.right = node2.copy()

        if parent2_node is None:
            child2 = node1.copy()
        elif dir2 == 'left':
            parent2_node.left = node1.copy()
        else:
            parent2_node.right = node1.copy()

        return child1, child2

    def _depth_to_node(self, root: ExpressionNode,
                       target: ExpressionNode) -> int:
        """
        计算从根节点到目标节点的深度

        Args:
            root: 根节点
            target: 目标节点

        Returns:
            深度值，未找到返回-1
        """
        if root is target:
            return 0
        if root.left:
            d = self._depth_to_node(root.left, target)
            if d >= 0:
                return d + 1
        if root.right:
            d = self._depth_to_node(root.right, target)
            if d >= 0:
                return d + 1
        return -1

    # ----------------------------------------------------------
    # 变异操作
    # ----------------------------------------------------------

    def mutate(self, individual: ExpressionNode) -> ExpressionNode:
        """
        变异操作

        随机选择一种变异方式:
        1. 常数变异：修改常数节点的值
        2. 子树变异：用随机子树替换选中子树
        3. 节点替换：替换运算符类型

        Args:
            individual: 待变异的个体

        Returns:
            变异后的个体
        """
        mutant = individual.copy()
        nodes = mutant.get_all_nodes()

        if not nodes:
            return mutant

        # 随机选择变异方式
        mutation_type = random.choice(['constant', 'subtree', 'operator'])

        if mutation_type == 'constant':
            mutant = self._mutate_constant(mutant, nodes)
        elif mutation_type == 'subtree':
            mutant = self._mutate_subtree(mutant, nodes)
        else:
            mutant = self._mutate_operator(mutant, nodes)

        return mutant

    def _mutate_constant(self, individual: ExpressionNode,
                         nodes: List) -> ExpressionNode:
        """
        常数变异：随机修改一个常数节点的值

        Args:
            individual: 个体
            nodes: 所有节点列表

        Returns:
            变异后的个体
        """
        # 找到所有常数节点
        constant_nodes = [(n, p, d) for n, p, d in nodes
                          if n.node_type == NodeType.CONSTANT]

        if not constant_nodes:
            return individual

        # 随机选择一个常数节点
        node, parent, direction = random.choice(constant_nodes)

        # 高斯变异
        node.value += random.gauss(0, self.config.constant_mutation_range)

        return individual

    def _mutate_subtree(self, individual: ExpressionNode,
                        nodes: List) -> ExpressionNode:
        """
        子树变异：用随机生成的子树替换选中的子树

        Args:
            individual: 个体
            nodes: 所有节点列表

        Returns:
            变异后的个体
        """
        # 随机选择一个非根节点进行替换
        replaceable = [(n, p, d) for n, p, d in nodes
                       if p is not None]  # 排除根节点

        if not replaceable:
            # 如果只有根节点，直接替换整棵树
            return self._generate_tree_grow(
                max_depth=self.config.subtree_mutation_depth,
                min_depth=1
            )

        node, parent, direction = random.choice(replaceable)

        # 计算剩余允许深度
        remaining_depth = self.config.max_tree_depth - self._depth_to_node(
            individual, node
        )
        new_depth = min(self.config.subtree_mutation_depth, remaining_depth)

        if new_depth < 1:
            return individual

        # 生成新子树
        new_subtree = self._generate_tree_grow(
            max_depth=new_depth,
            min_depth=1
        )

        # 替换
        if direction == 'left':
            parent.left = new_subtree
        else:
            parent.right = new_subtree

        return individual

    def _mutate_operator(self, individual: ExpressionNode,
                         nodes: List) -> ExpressionNode:
        """
        运算符变异：替换一个运算符节点的类型

        Args:
            individual: 个体
            nodes: 所有节点列表

        Returns:
            变异后的个体
        """
        # 找到所有运算符节点
        operator_nodes = [(n, p, d) for n, p, d in nodes
                          if n.node_type in (NodeType.EML, NodeType.ADD,
                                             NodeType.SUB, NodeType.MUL)]

        if not operator_nodes:
            return individual

        # 随机选择一个运算符节点
        node, parent, direction = random.choice(operator_nodes)

        # 选择新的运算符类型
        binary_ops = [NodeType.EML, NodeType.ADD, NodeType.SUB, NodeType.MUL]
        new_type = random.choice([op for op in binary_ops if op != node.node_type])
        node.node_type = new_type

        return individual

    # ----------------------------------------------------------
    # 进化循环
    # ----------------------------------------------------------

    def evolve(self, train_x: List[float], train_y: List[float],
               verbose: bool = False) -> Dict[str, Any]:
        """
        执行完整的进化过程

        Args:
            train_x: 训练输入数据
            train_y: 训练目标数据
            verbose: 是否打印进度信息

        Returns:
            进化结果字典，包含:
            - best_expression: 最优表达式字符串
            - best_fitness: 最优适应度
            - best_individual: 最优个体（表达式树）
            - generations: 实际进化代数
            - history: 进化历史
            - elapsed_time: 总耗时（秒）
        """
        start_time = time.time()

        # 步骤1: 初始化种群
        self.initialize_population()
        if verbose:
            print(f"[GP] 种群初始化完成，大小: {len(self.population)}")

        # 步骤2: 评估初始种群
        fitness_results = self._evaluate_population(train_x, train_y)

        # 记录初始最优
        best_idx = min(range(len(fitness_results)),
                       key=lambda i: fitness_results[i].fitness)
        self.best_individual = self.population[best_idx].copy()
        self.best_fitness = fitness_results[best_idx].fitness

        no_improvement_count = 0  # 连续无改善计数

        if verbose:
            print(f"[GP] 初始最优适应度: {self.best_fitness:.6f}")
            print(f"[GP] 初始最优表达式: {self.best_individual.to_string()}")

        # 步骤3: 进化循环
        for gen in range(self.config.max_generations):
            gen_start = time.time()

            # --- 选择与繁殖 ---
            new_population = []

            # 精英保留：直接复制最优的几个个体
            sorted_indices = sorted(range(len(fitness_results)),
                                    key=lambda i: fitness_results[i].fitness)
            for i in range(min(self.config.elitism_count, len(self.population))):
                new_population.append(self.population[sorted_indices[i]].copy())

            # 生成剩余个体
            while len(new_population) < self.config.population_size:
                r = random.random()

                if r < self.config.crossover_rate:
                    # 交叉操作
                    parent1 = self.tournament_selection(fitness_results)
                    parent2 = self.tournament_selection(fitness_results)
                    child1, child2 = self.crossover(parent1, parent2)
                    new_population.append(child1)
                    if len(new_population) < self.config.population_size:
                        new_population.append(child2)

                elif r < self.config.crossover_rate + self.config.mutation_rate:
                    # 变异操作
                    parent = self.tournament_selection(fitness_results)
                    child = self.mutate(parent)
                    new_population.append(child)

                else:
                    # 直接复制
                    parent = self.tournament_selection(fitness_results)
                    new_population.append(parent.copy())

            # 确保种群大小正确
            self.population = new_population[:self.config.population_size]

            # --- 评估新种群 ---
            fitness_results = self._evaluate_population(train_x, train_y)

            # --- 更新最优个体 ---
            current_best_idx = min(range(len(fitness_results)),
                                   key=lambda i: fitness_results[i].fitness)
            current_best_fitness = fitness_results[current_best_idx].fitness

            if current_best_fitness < self.best_fitness:
                self.best_fitness = current_best_fitness
                self.best_individual = self.population[current_best_idx].copy()
                no_improvement_count = 0
            else:
                no_improvement_count += 1

            # --- 记录统计信息 ---
            all_fitness = [fr.fitness for fr in fitness_results]
            avg_fitness = statistics.mean(all_fitness)
            worst_fitness = max(all_fitness)

            # 计算种群多样性（不同表达式的比例）
            expressions = set(ind.to_string() for ind in self.population)
            diversity = len(expressions) / len(self.population)

            gen_elapsed = time.time() - gen_start
            stats = EvolutionStats(
                generation=gen + 1,
                best_fitness=self.best_fitness,
                avg_fitness=avg_fitness,
                worst_fitness=worst_fitness,
                best_expression=self.best_individual.to_string(),
                diversity=diversity,
                elapsed_time=gen_elapsed,
            )
            self.history.append(stats)

            # --- 进度输出 ---
            if verbose and (gen + 1) % 10 == 0:
                print(
                    f"[GP] 代 {gen + 1:4d} | "
                    f"最优: {self.best_fitness:.6f} | "
                    f"平均: {avg_fitness:.6f} | "
                    f"多样性: {diversity:.2f} | "
                    f"耗时: {gen_elapsed:.2f}s"
                )

            # --- 终止条件检查 ---
            if self.best_fitness <= self.config.fitness_threshold:
                if verbose:
                    print(f"[GP] 达到适应度阈值 {self.config.fitness_threshold}，停止进化")
                break

            if no_improvement_count >= self.config.patience:
                if verbose:
                    print(f"[GP] 连续 {self.config.patience} 代无改善，停止进化")
                break

        total_time = time.time() - start_time

        if verbose:
            print(f"\n[GP] 进化完成")
            print(f"  总代数: {len(self.history)}")
            print(f"  最优适应度: {self.best_fitness:.6f}")
            print(f"  最优表达式: {self.best_individual.to_string()}")
            print(f"  总耗时: {total_time:.2f}s")

        return {
            'best_expression': self.best_individual.to_string(),
            'best_fitness': self.best_fitness,
            'best_individual': self.best_individual,
            'generations': len(self.history),
            'history': self.history,
            'elapsed_time': total_time,
        }

    # ----------------------------------------------------------
    # 辅助方法
    # ----------------------------------------------------------

    def get_population_expressions(self) -> List[str]:
        """
        获取当前种群中所有个体的表达式字符串

        Returns:
            表达式字符串列表
        """
        return [ind.to_string() for ind in self.population]

    def get_population_diversity(self) -> float:
        """
        计算种群多样性

        多样性定义为不同表达式占总数的比例。

        Returns:
            多样性值，范围[0, 1]
        """
        if not self.population:
            return 0.0
        expressions = set(ind.to_string() for ind in self.population)
        return len(expressions) / len(self.population)

    def get_evolution_summary(self) -> str:
        """
        生成进化过程摘要报告

        Returns:
            格式化的摘要字符串
        """
        lines = []
        lines.append("=" * 60)
        lines.append("EML Genetic Programming - 进化摘要报告")
        lines.append("=" * 60)
        lines.append("")

        if not self.history:
            lines.append("  尚未运行进化过程")
            return "\n".join(lines)

        lines.append(f"  总进化代数: {len(self.history)}")
        lines.append(f"  种群大小: {self.config.population_size}")
        lines.append(f"  最优适应度: {self.best_fitness:.6f}")
        lines.append(f"  最优表达式: {self.best_individual.to_string()}")
        lines.append("")

        lines.append("  进化趋势:")
        lines.append(f"    {'代数':>6s} | {'最优适应度':>12s} | {'平均适应度':>12s} | {'多样性':>8s}")
        lines.append("  " + "-" * 52)

        # 每隔一定代数输出一行
        step = max(1, len(self.history) // 10)
        for i in range(0, len(self.history), step):
            s = self.history[i]
            lines.append(
                f"    {s.generation:6d} | {s.best_fitness:12.6f} | "
                f"{s.avg_fitness:12.6f} | {s.diversity:8.2f}"
            )

        # 最后一行
        s = self.history[-1]
        lines.append(
            f"    {s.generation:6d} | {s.best_fitness:12.6f} | "
            f"{s.avg_fitness:12.6f} | {s.diversity:8.2f}"
        )

        lines.append("")
        lines.append("=" * 60)
        return "\n".join(lines)

    def reset(self):
        """重置遗传编程引擎状态"""
        self.population = []
        self.history = []
        self.best_individual = None
        self.best_fitness = float('inf')


# ============================================================
# 模块自测
# ============================================================

if __name__ == "__main__":
    print("EML Genetic Programming - 模块自测")
    print("=" * 60)

    # 创建遗传编程引擎
    gp = EMLGeneticProgramming(GPConfig(
        population_size=50,
        max_generations=30,
        max_tree_depth=4,
        seed=42,
    ))

    # 定义目标函数: eml(x, 1) = e^x
    def target_fn(x):
        return math.exp(x) - math.log(1.0)  # = e^x

    # 准备训练数据
    train_x = [i * 0.1 for i in range(1, 30)]
    train_y = [target_fn(xi) for xi in train_x]

    # 运行进化
    result = gp.evolve(train_x, train_y, verbose=True)

    # 输出摘要
    print("\n" + gp.get_evolution_summary())
