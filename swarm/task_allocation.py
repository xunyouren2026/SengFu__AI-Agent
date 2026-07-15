"""
任务分配模块

提供多种任务分配算法：匈牙利算法最优分配、拍卖机制分配、
贪心分配，以及Shapley值贡献度评估。仅使用Python标准库。
"""

import math
import threading
import time
import uuid
from itertools import permutations
from typing import Any, Dict, List, Optional, Set, Tuple


class TaskAssignment:
    """任务分配结果

    描述一个任务到智能体的分配结果。

    Attributes:
        task_id: 任务ID
        agent_id: 智能体ID
        cost: 分配成本
        confidence: 分配置信度
        metadata: 附加信息
    """

    def __init__(
        self,
        task_id: str,
        agent_id: str,
        cost: float = 0.0,
        confidence: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.task_id = task_id
        self.agent_id = agent_id
        self.cost = max(0.0, cost)
        self.confidence = max(0.0, min(1.0, confidence))
        self.metadata = metadata or {}
        self.assigned_at = time.time()

    def to_dict(self) -> Dict[str, Any]:
        """序列化为字典"""
        return {
            "task_id": self.task_id,
            "agent_id": self.agent_id,
            "cost": self.cost,
            "confidence": self.confidence,
            "metadata": self.metadata,
            "assigned_at": self.assigned_at,
        }

    def __repr__(self) -> str:
        return (
            f"TaskAssignment(task={self.task_id!r}, agent={self.agent_id!r}, "
            f"cost={self.cost:.2f})"
        )


class HungarianSolver:
    """匈牙利算法求解器

    使用标准匈牙利算法（Kuhn-Munkres算法）求解最优分配问题。
    算法步骤：行缩减 -> 列缩减 -> 覆盖零元素 -> 调整矩阵 -> 迭代。

    支持最小化和最大化两种模式。
    """

    def __init__(self, maximize: bool = False):
        """
        Args:
            maximize: 是否最大化（默认为最小化）
        """
        self.maximize = maximize

    def solve(self, cost_matrix: List[List[float]]) -> List[Tuple[int, int]]:
        """求解最优分配

        Args:
            cost_matrix: 成本矩阵，cost_matrix[i][j]表示将任务i分配给智能体j的成本

        Returns:
            最优分配列表，每个元素为(任务索引, 智能体索引)元组

        Raises:
            ValueError: 成本矩阵无效
        """
        if not cost_matrix or not cost_matrix[0]:
            return []

        n_rows = len(cost_matrix)
        n_cols = len(cost_matrix[0])

        # 构造方阵（补零填充）
        size = max(n_rows, n_cols)
        matrix = [[0.0] * size for _ in range(size)]
        for i in range(n_rows):
            for j in range(n_cols):
                val = cost_matrix[i][j]
                if self.maximize:
                    val = -val  # 转换为最小化
                matrix[i][j] = val

        # 步骤1：行缩减 - 每行减去该行最小值
        for i in range(size):
            row_min = min(matrix[i])
            for j in range(size):
                matrix[i][j] -= row_min

        # 步骤2：列缩减 - 每列减去该列最小值
        for j in range(size):
            col_min = min(matrix[i][j] for i in range(size))
            for i in range(size):
                matrix[i][j] -= col_min

        # 初始化标记
        # star: 已标记的零（候选分配）
        # prime: 临时标记的零
        # covered_rows, covered_cols: 被覆盖的行和列
        star = [[False] * size for _ in range(size)]
        prime = [[False] * size for _ in range(size)]
        covered_rows = [False] * size
        covered_cols = [False] * size

        # 步骤3：标记独立零
        for i in range(size):
            for j in range(size):
                if (
                    matrix[i][j] == 0.0
                    and not covered_rows[i]
                    and not covered_cols[j]
                ):
                    star[i][j] = True
                    covered_rows[i] = True
                    covered_cols[j] = True

        # 重置覆盖
        covered_rows = [False] * size
        covered_cols = [False] * size

        # 步骤4：覆盖有star零的列
        for i in range(size):
            for j in range(size):
                if star[i][j]:
                    covered_cols[j] = True

        while True:
            # 检查是否所有列都被覆盖
            if all(covered_cols):
                break

            # 步骤5：找未覆盖零
            found_prime = False
            prime_row = -1
            prime_col = -1

            for i in range(size):
                if covered_rows[i]:
                    continue
                for j in range(size):
                    if not covered_cols[j] and matrix[i][j] == 0.0:
                        prime[i][j] = True
                        prime_row = i
                        prime_col = j
                        found_prime = True
                        break
                if found_prime:
                    break

            if not found_prime:
                # 步骤7：调整矩阵
                # 找未覆盖行和覆盖列中的最小值
                min_val = float("inf")
                for i in range(size):
                    if not covered_rows[i]:
                        for j in range(size):
                            if not covered_cols[j]:
                                min_val = min(min_val, matrix[i][j])

                if min_val == float("inf"):
                    break

                # 未覆盖行减去最小值
                for i in range(size):
                    if not covered_rows[i]:
                        for j in range(size):
                            matrix[i][j] -= min_val

                # 覆盖列加上最小值
                for j in range(size):
                    if covered_cols[j]:
                        for i in range(size):
                            matrix[i][j] += min_val

                # 清除prime标记
                prime = [[False] * size for _ in range(size)]
                continue

            # 检查prime零所在行是否有star零
            star_in_row = False
            star_col = -1
            for j in range(size):
                if star[prime_row][j]:
                    star_in_row = True
                    star_col = j
                    break

            if star_in_row:
                # 步骤6：交替标记
                covered_rows[prime_row] = True
                covered_cols[star_col] = False
            else:
                # 步骤6续：增广路径
                # 从prime零开始，构建交替路径
                path = [(prime_row, prime_col)]
                while True:
                    # 找路径中最后一个prime零所在列的star零
                    last_col = path[-1][1]
                    star_row = -1
                    for i in range(size):
                        if star[i][last_col]:
                            star_row = i
                            break

                    if star_row == -1:
                        break

                    path.append((star_row, last_col))

                    # 找该star零所在行的prime零
                    last_row = path[-1][0]
                    prime_col_next = -1
                    for j in range(size):
                        if prime[last_row][j]:
                            prime_col_next = j
                            break

                    if prime_col_next == -1:
                        break

                    path.append((last_row, prime_col_next))

                # 翻转路径上的star标记
                for i, j in path:
                    star[i][j] = not star[i][j]

                # 清除标记
                prime = [[False] * size for _ in range(size)]
                covered_rows = [False] * size
                covered_cols = [False] * size

                # 重新覆盖有star零的列
                for i in range(size):
                    for j in range(size):
                        if star[i][j]:
                            covered_cols[j] = True

        # 提取分配结果
        assignments = []
        for i in range(n_rows):
            for j in range(n_cols):
                if star[i][j]:
                    assignments.append((i, j))
                    break

        return assignments

    def solve_with_cost(
        self, cost_matrix: List[List[float]]
    ) -> Tuple[List[Tuple[int, int]], float]:
        """求解并返回总成本

        Args:
            cost_matrix: 成本矩阵

        Returns:
            (分配列表, 总成本)
        """
        assignments = self.solve(cost_matrix)
        total_cost = 0.0
        for row, col in assignments:
            if row < len(cost_matrix) and col < len(cost_matrix[0]):
                total_cost += cost_matrix[row][col]
        if self.maximize:
            total_cost = -total_cost
        return assignments, total_cost


class VickreyAuction:
    """Vickrey（第二价格）拍卖

    出价最高的竞标者获胜，但只需支付第二高的出价价格。
    这鼓励竞标者提交真实估值。
    """

    def __init__(self):
        self._bids: List[Dict[str, Any]] = []
        self._lock = threading.Lock()
        self._closed = False

    def bid(self, agent_id: str, value: float, metadata: Optional[Dict] = None) -> bool:
        """提交出价

        Args:
            agent_id: 竞标者ID
            value: 出价
            metadata: 附加信息

        Returns:
            是否成功出价
        """
        if self._closed:
            return False
        if value < 0:
            return False

        with self._lock:
            self._bids.append({
                "agent_id": agent_id,
                "value": value,
                "metadata": metadata or {},
                "timestamp": time.time(),
            })
            return True

    def determine_winner(self) -> Optional[Dict[str, Any]]:
        """确定赢家

        Returns:
            拍卖结果字典，包含winner_id, winning_bid, second_price等信息
        """
        with self._lock:
            if not self._bids:
                return None

            # 按出价降序排列
            sorted_bids = sorted(self._bids, key=lambda b: b["value"], reverse=True)

            winner = sorted_bids[0]
            second_price = sorted_bids[1]["value"] if len(sorted_bids) > 1 else winner["value"]

            return {
                "winner_id": winner["agent_id"],
                "winning_bid": winner["value"],
                "second_price": second_price,
                "payment": second_price,  # Vickrey: 支付第二价格
                "total_bids": len(self._bids),
                "metadata": winner["metadata"],
            }

    def close(self) -> None:
        """关闭拍卖"""
        self._closed = True

    def reset(self) -> None:
        """重置拍卖"""
        with self._lock:
            self._bids.clear()
            self._closed = False

    def get_bids(self) -> List[Dict[str, Any]]:
        """获取所有出价"""
        with self._lock:
            return list(self._bids)


class CombinatorialAuction:
    """组合拍卖

    允许竞标者对多个物品的组合进行出价。
    使用贪心算法确定赢家集合（近似最优）。
    """

    def __init__(self, items: Optional[List[str]] = None):
        """
        Args:
            items: 可拍卖物品列表
        """
        self._items = list(items) if items else []
        self._bids: List[Dict[str, Any]] = []
        self._lock = threading.Lock()
        self._closed = False

    def bid(
        self,
        agent_id: str,
        item_bundle: List[str],
        value: float,
        metadata: Optional[Dict] = None,
    ) -> bool:
        """提交组合出价

        Args:
            agent_id: 竞标者ID
            item_bundle: 物品组合列表
            value: 出价
            metadata: 附加信息

        Returns:
            是否成功出价
        """
        if self._closed:
            return False
        if value < 0:
            return False
        for item in item_bundle:
            if item not in self._items:
                return False

        with self._lock:
            self._bids.append({
                "agent_id": agent_id,
                "items": set(item_bundle),
                "value": value,
                "metadata": metadata or {},
                "timestamp": time.time(),
            })
            return True

    def determine_winner(self) -> Dict[str, Any]:
        """确定赢家集合（贪心算法）

        贪心策略：每次选择单位物品价值最高的出价，
        直到所有物品被分配或没有更多有效出价。

        Returns:
            拍卖结果字典
        """
        with self._lock:
            if not self._bids:
                return {
                    "winners": [],
                    "total_revenue": 0.0,
                    "allocated_items": set(),
                }

            allocated = set()
            winners = []
            remaining_bids = sorted(
                self._bids, key=lambda b: b["value"] / max(len(b["items"]), 1),
                reverse=True,
            )

            for bid in remaining_bids:
                bundle = bid["items"]
                # 检查bundle中是否有已分配的物品
                if bundle & allocated:
                    continue
                if not bundle:
                    continue

                winners.append({
                    "agent_id": bid["agent_id"],
                    "items": list(bundle),
                    "value": bid["value"],
                    "payment": bid["value"],
                    "metadata": bid["metadata"],
                })
                allocated |= bundle

                if allocated >= set(self._items):
                    break

            total_revenue = sum(w["payment"] for w in winners)

            return {
                "winners": winners,
                "total_revenue": total_revenue,
                "allocated_items": allocated,
                "unallocated_items": set(self._items) - allocated,
            }

    def close(self) -> None:
        """关闭拍卖"""
        self._closed = True

    def reset(self) -> None:
        """重置拍卖"""
        with self._lock:
            self._bids.clear()
            self._closed = False


class AuctionMechanism:
    """拍卖机制

    统一的拍卖机制入口，支持Vickrey拍卖和组合拍卖。
    """

    def __init__(self):
        self._vickrey = VickreyAuction()
        self._combinatorial = CombinatorialAuction()

    def create_vickrey(self) -> VickreyAuction:
        """创建Vickrey拍卖实例"""
        return VickreyAuction()

    def create_combinatorial(self, items: List[str]) -> CombinatorialAuction:
        """创建组合拍卖实例"""
        return CombinatorialAuction(items)


class TaskAllocation:
    """任务分配器

    提供多种任务分配策略：匈牙利算法最优分配、拍卖机制分配、
    贪心分配，以及Shapley值贡献度评估。
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._assignment_history: List[TaskAssignment] = []

    def hungarian_assignment(
        self,
        cost_matrix: List[List[float]],
        task_ids: Optional[List[str]] = None,
        agent_ids: Optional[List[str]] = None,
        maximize: bool = False,
    ) -> List[TaskAssignment]:
        """匈牙利算法最优分配

        Args:
            cost_matrix: 成本矩阵
            task_ids: 任务ID列表
            agent_ids: 智能体ID列表
            maximize: 是否最大化

        Returns:
            任务分配结果列表
        """
        solver = HungarianSolver(maximize=maximize)
        assignments, total_cost = solver.solve_with_cost(cost_matrix)

        results = []
        for row, col in assignments:
            task_id = task_ids[row] if task_ids and row < len(task_ids) else f"task_{row}"
            agent_id = agent_ids[col] if agent_ids and col < len(agent_ids) else f"agent_{col}"
            cost = cost_matrix[row][col] if row < len(cost_matrix) and col < len(cost_matrix[0]) else 0.0

            results.append(TaskAssignment(
                task_id=task_id,
                agent_id=agent_id,
                cost=cost,
                confidence=1.0,
                metadata={"method": "hungarian", "total_cost": total_cost},
            ))

        with self._lock:
            self._assignment_history.extend(results)

        return results

    def auction_allocation(
        self,
        task_id: str,
        agent_bids: Dict[str, float],
        auction_type: str = "vickrey",
    ) -> Optional[TaskAssignment]:
        """拍卖机制分配

        Args:
            task_id: 任务ID
            agent_bids: 智能体出价字典 {agent_id: bid_value}
            auction_type: 拍卖类型 ("vickrey" 或 "combinatorial")

        Returns:
            任务分配结果
        """
        if not agent_bids:
            return None

        if auction_type == "vickrey":
            auction = VickreyAuction()
            for agent_id, value in agent_bids.items():
                auction.bid(agent_id, value)

            result = auction.determine_winner()
            if result is None:
                return None

            assignment = TaskAssignment(
                task_id=task_id,
                agent_id=result["winner_id"],
                cost=result["payment"],
                confidence=1.0,
                metadata={
                    "method": "vickrey_auction",
                    "winning_bid": result["winning_bid"],
                    "second_price": result["second_price"],
                    "total_bids": result["total_bids"],
                },
            )
        else:
            # 简单拍卖：最高出价者获胜
            sorted_bids = sorted(agent_bids.items(), key=lambda x: x[1], reverse=True)
            winner_id, winning_bid = sorted_bids[0]
            second_bid = sorted_bids[1][1] if len(sorted_bids) > 1 else winning_bid

            assignment = TaskAssignment(
                task_id=task_id,
                agent_id=winner_id,
                cost=second_bid,
                confidence=1.0,
                metadata={
                    "method": "simple_auction",
                    "winning_bid": winning_bid,
                    "second_bid": second_bid,
                },
            )

        with self._lock:
            self._assignment_history.append(assignment)

        return assignment

    def greedy_allocation(
        self,
        tasks: List[Dict[str, Any]],
        agents: List[Dict[str, Any]],
        cost_fn: Optional[callable] = None,
    ) -> List[TaskAssignment]:
        """贪心分配

        每次为当前任务选择成本最低的可用智能体。

        Args:
            tasks: 任务列表，每个任务为字典，必须包含"id"字段
            agents: 智能体列表，每个智能体为字典，必须包含"id"和"capacity"字段
            cost_fn: 成本函数 (task, agent) -> float，默认为随机成本

        Returns:
            任务分配结果列表
        """
        if cost_fn is None:
            cost_fn = lambda t, a: 1.0  # 默认均匀成本

        # 跟踪每个智能体的剩余容量
        agent_capacity = {}
        for agent in agents:
            agent_capacity[agent["id"]] = agent.get("capacity", 10)
        agent_current = {a["id"]: 0 for a in agents}

        results = []
        for task in tasks:
            task_id = task.get("id", str(uuid.uuid4()))
            best_agent = None
            best_cost = float("inf")

            for agent in agents:
                agent_id = agent["id"]
                if agent_current[agent_id] >= agent_capacity[agent_id]:
                    continue  # 智能体已满载

                cost = cost_fn(task, agent)
                if cost < best_cost:
                    best_cost = cost
                    best_agent = agent_id

            if best_agent is not None:
                agent_current[best_agent] += 1
                assignment = TaskAssignment(
                    task_id=task_id,
                    agent_id=best_agent,
                    cost=best_cost,
                    confidence=1.0,
                    metadata={"method": "greedy"},
                )
                results.append(assignment)
            else:
                # 所有智能体都满载，选择负载最低的
                min_load_agent = min(agent_current, key=agent_current.get)
                agent_current[min_load_agent] += 1
                assignment = TaskAssignment(
                    task_id=task_id,
                    agent_id=min_load_agent,
                    cost=float("inf"),
                    confidence=0.5,
                    metadata={"method": "greedy_overflow"},
                )
                results.append(assignment)

        with self._lock:
            self._assignment_history.extend(results)

        return results

    def shapley_value_contribution(
        self,
        agent_id: str,
        coalition: List[str],
        value_fn: callable,
    ) -> float:
        """Shapley值贡献度评估

        计算指定智能体在联盟中的Shapley值，衡量其边际贡献。

        Shapley值公式:
        phi_i = (1/n!) * sum_{S subset N\\{i}} |S|!(n-|S|-1)! * [v(S U {i}) - v(S)]

        对于大联盟使用采样近似。

        Args:
            agent_id: 要评估的智能体ID
            coalition: 联盟成员ID列表
            value_fn: 联盟价值函数 (set of agent_ids) -> float

        Returns:
            Shapley值
        """
        n = len(coalition)
        if n == 0:
            return 0.0
        if agent_id not in coalition:
            return 0.0

        others = [a for a in coalition if a != agent_id]
        m = len(others)

        if m == 0:
            # 联盟只有该智能体
            return value_fn({agent_id})

        # 对于小联盟使用精确计算
        if m <= 10:
            return self._exact_shapley(agent_id, others, value_fn, n)
        else:
            # 对于大联盟使用蒙特卡洛采样
            return self._sampled_shapley(agent_id, others, value_fn, n)

    def _exact_shapley(
        self,
        agent_id: str,
        others: List[str],
        value_fn: callable,
        n: int,
    ) -> float:
        """精确计算Shapley值"""
        from math import factorial

        shapley = 0.0
        m = len(others)

        # 遍历others的所有子集
        for size in range(m + 1):
            # 生成所有大小为size的子集
            for subset in self._combinations(others, size):
                s = set(subset)
                # v(S U {i}) - v(S)
                marginal = value_fn(s | {agent_id}) - value_fn(s)
                # 权重: |S|! * (n - |S| - 1)! / n!
                weight = (factorial(size) * factorial(n - size - 1)) / factorial(n)
                shapley += weight * marginal

        return shapley

    def _sampled_shapley(
        self,
        agent_id: str,
        others: List[str],
        value_fn: callable,
        n: int,
        num_samples: int = 1000,
    ) -> float:
        """蒙特卡洛采样近似Shapley值"""
        import random

        shapley = 0.0
        m = len(others)

        for _ in range(num_samples):
            # 随机排列
            perm = list(others)
            random.shuffle(perm)

            # 随机选择一个位置插入agent
            pos = random.randint(0, m)
            s = set(perm[:pos])
            marginal = value_fn(s | {agent_id}) - value_fn(s)
            shapley += marginal

        return shapley / num_samples

    @staticmethod
    def _combinations(lst: List, k: int) -> List[List]:
        """生成列表的所有k大小组合"""
        if k == 0:
            return [[]]
        if k > len(lst):
            return []
        if k == len(lst):
            return [list(lst)]

        result = []
        # 递归生成组合
        def _combine(start, current):
            if len(current) == k:
                result.append(list(current))
                return
            for i in range(start, len(lst)):
                current.append(lst[i])
                _combine(i + 1, current)
                current.pop()

        _combine(0, [])
        return result

    def get_assignment_history(self) -> List[TaskAssignment]:
        """获取分配历史"""
        with self._lock:
            return list(self._assignment_history)

    def clear_history(self) -> None:
        """清空分配历史"""
        with self._lock:
            self._assignment_history.clear()
