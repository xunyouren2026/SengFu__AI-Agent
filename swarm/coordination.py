"""
协调器模块

提供多智能体协商、冲突解决、共识投票、分布式资源锁
和共享黑板等协调机制。线程安全实现。
"""

import threading
import time
import uuid
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from collections import defaultdict


class Coordinator:
    """协调器

    提供多智能体间的协商、冲突解决和共识投票机制。

    Attributes:
        negotiation_timeout: 协商超时时间（秒）
    """

    def __init__(self, negotiation_timeout: float = 60.0):
        self._lock = threading.RLock()
        self._negotiation_timeout = negotiation_timeout
        # 活跃的协商会话 {session_id: session_info}
        self._negotiations: Dict[str, Dict[str, Any]] = {}
        # 投票会话 {vote_id: vote_info}
        self._votes: Dict[str, Dict[str, Any]] = {}

    def negotiate(
        self,
        agent_ids: List[str],
        task: Dict[str, Any],
        strategy: str = "majority",
        rounds: int = 3,
    ) -> Dict[str, Any]:
        """多Agent协商

        智能体通过多轮协商达成任务分配协议。

        Args:
            agent_ids: 参与协商的智能体ID列表
            task: 任务描述
            strategy: 协商策略 ("majority", "unanimous", "weighted")
            rounds: 最大协商轮数

        Returns:
            协商结果字典
        """
        if not agent_ids:
            return {"success": False, "reason": "无参与智能体"}

        session_id = str(uuid.uuid4())
        start_time = time.time()

        session = {
            "id": session_id,
            "task": task,
            "agents": list(agent_ids),
            "strategy": strategy,
            "round": 0,
            "max_rounds": rounds,
            "proposals": [],
            "status": "negotiating",
            "start_time": start_time,
        }

        with self._lock:
            self._negotiations[session_id] = session

        # 模拟协商过程
        result = self._run_negotiation(session)

        with self._lock:
            session["status"] = result.get("status", "completed")
            session["result"] = result

        return {
            "session_id": session_id,
            "success": result.get("success", False),
            "agreement": result.get("agreement"),
            "rounds_used": session["round"],
            "elapsed": time.time() - start_time,
        }

    def _run_negotiation(self, session: Dict[str, Any]) -> Dict[str, Any]:
        """执行协商过程

        使用基于提议-反提议的协商模型：
        1. 初始提议由任务需求决定
        2. 每轮智能体可以接受、拒绝或反提议
        3. 根据策略判断是否达成协议
        """
        agents = session["agents"]
        strategy = session["strategy"]
        max_rounds = session["max_rounds"]
        task = session["task"]

        # 初始提议：均匀分配
        current_proposal = self._generate_initial_proposal(task, agents)

        for round_num in range(1, max_rounds + 1):
            session["round"] = round_num

            # 收集各智能体的反馈
            accept_count = 0
            reject_count = 0
            counter_proposals = []

            for agent_id in agents:
                # 模拟智能体评估（基于负载和能力的接受概率）
                feedback = self._simulate_agent_feedback(
                    agent_id, current_proposal, task
                )
                if feedback["action"] == "accept":
                    accept_count += 1
                elif feedback["action"] == "reject":
                    reject_count += 1
                else:
                    counter_proposals.append(feedback.get("counter_proposal"))

            session["proposals"].append({
                "round": round_num,
                "proposal": current_proposal,
                "accept_count": accept_count,
                "reject_count": reject_count,
            })

            # 判断是否达成协议
            total = len(agents)
            if strategy == "majority":
                if accept_count > total / 2:
                    return {
                        "success": True,
                        "status": "agreed",
                        "agreement": current_proposal,
                        "accept_count": accept_count,
                    }
            elif strategy == "unanimous":
                if accept_count == total:
                    return {
                        "success": True,
                        "status": "agreed",
                        "agreement": current_proposal,
                        "accept_count": accept_count,
                    }
            elif strategy == "weighted":
                # 加权投票：accept权重 > reject权重
                accept_weight = accept_count * 1.0
                reject_weight = reject_count * 1.0
                if accept_weight > reject_weight:
                    return {
                        "success": True,
                        "status": "agreed",
                        "agreement": current_proposal,
                        "accept_count": accept_count,
                    }

            # 如果有反提议，合并为新的提议
            if counter_proposals:
                current_proposal = self._merge_proposals(counter_proposals, agents)
            else:
                # 无反提议且未达成协议，调整提议
                current_proposal = self._adjust_proposal(current_proposal, agents)

        # 超过最大轮数未达成协议
        return {
            "success": False,
            "status": "timeout",
            "agreement": None,
            "reason": f"超过{max_rounds}轮协商未达成协议",
        }

    def _generate_initial_proposal(
        self, task: Dict[str, Any], agents: List[str]
    ) -> Dict[str, Any]:
        """生成初始提议"""
        n = len(agents)
        task_items = task.get("subtasks", [task.get("name", "default_task")])
        if isinstance(task_items, str):
            task_items = [task_items]

        # 均匀分配子任务
        assignment = {}
        for i, agent_id in enumerate(agents):
            start = (i * len(task_items)) // n
            end = ((i + 1) * len(task_items)) // n
            assignment[agent_id] = task_items[start:end]

        return {
            "type": "task_assignment",
            "assignment": assignment,
            "deadline": task.get("deadline", time.time() + 3600),
            "priority": task.get("priority", "normal"),
        }

    def _simulate_agent_feedback(
        self, agent_id: str, proposal: Dict[str, Any], task: Dict[str, Any]
    ) -> Dict[str, Any]:
        """模拟智能体反馈

        基于任务分配的公平性和智能体负载决定接受/拒绝。
        """
        import random

        assignment = proposal.get("assignment", {})
        assigned_tasks = assignment.get(agent_id, [])

        # 基于分配的任务量决定接受概率
        total_tasks = sum(len(v) for v in assignment.values())
        my_tasks = len(assigned_tasks)
        fairness = my_tasks / max(total_tasks, 1)

        # 公平性越高越可能接受
        accept_prob = max(0.1, 1.0 - abs(fairness - 1.0 / max(len(assignment), 1)))

        rand = random.random()
        if rand < accept_prob:
            return {"action": "accept", "agent_id": agent_id}
        elif rand < accept_prob + 0.2:
            return {
                "action": "counter",
                "agent_id": agent_id,
                "counter_proposal": {
                    "agent_id": agent_id,
                    "preferred_tasks": assigned_tasks[:max(1, len(assigned_tasks) // 2)],
                },
            }
        else:
            return {"action": "reject", "agent_id": agent_id}

    def _merge_proposals(
        self, counter_proposals: List[Dict], agents: List[str]
    ) -> Dict[str, Any]:
        """合并反提议"""
        # 简单合并：取各智能体偏好的交集
        all_preferred = []
        for cp in counter_proposals:
            all_preferred.extend(cp.get("preferred_tasks", []))

        # 去重并重新分配
        unique_tasks = list(dict.fromkeys(all_preferred))
        n = len(agents)
        assignment = {}
        for i, agent_id in enumerate(agents):
            start = (i * len(unique_tasks)) // max(n, 1)
            end = ((i + 1) * len(unique_tasks)) // max(n, 1)
            assignment[agent_id] = unique_tasks[start:end]

        return {
            "type": "task_assignment",
            "assignment": assignment,
            "deadline": time.time() + 3600,
            "priority": "normal",
        }

    def _adjust_proposal(
        self, proposal: Dict[str, Any], agents: List[str]
    ) -> Dict[str, Any]:
        """调整提议（降低要求以增加接受概率）"""
        assignment = proposal.get("assignment", {})
        # 将任务重新均匀分配
        all_tasks = []
        for tasks in assignment.values():
            all_tasks.extend(tasks)

        n = len(agents)
        new_assignment = {}
        for i, agent_id in enumerate(agents):
            start = (i * len(all_tasks)) // max(n, 1)
            end = ((i + 1) * len(all_tasks)) // max(n, 1)
            new_assignment[agent_id] = all_tasks[start:end]

        return {
            "type": "task_assignment",
            "assignment": new_assignment,
            "deadline": proposal.get("deadline", time.time() + 3600),
            "priority": proposal.get("priority", "normal"),
        }

    def resolve_conflict(
        self,
        agent_ids: List[str],
        resource: str,
        strategy: str = "priority",
    ) -> Dict[str, Any]:
        """冲突解决

        当多个智能体竞争同一资源时，使用指定策略解决冲突。

        Args:
            agent_ids: 冲突智能体ID列表
            resource: 冲突资源标识
            strategy: 解决策略 ("priority", "round_robin", "random", "oldest")

        Returns:
            冲突解决结果
        """
        if not agent_ids:
            return {"success": False, "reason": "无冲突智能体"}

        if len(agent_ids) == 1:
            return {
                "success": True,
                "winner": agent_ids[0],
                "resource": resource,
                "strategy": strategy,
            }

        if strategy == "priority":
            # 按ID字典序（模拟优先级）
            winner = min(agent_ids)
        elif strategy == "random":
            import random
            winner = random.choice(agent_ids)
        elif strategy == "oldest":
            # 第一个请求的获胜（列表第一个）
            winner = agent_ids[0]
        elif strategy == "round_robin":
            # 轮询：基于资源名的哈希选择
            idx = hash(resource) % len(agent_ids)
            winner = agent_ids[idx]
        else:
            winner = agent_ids[0]

        return {
            "success": True,
            "winner": winner,
            "resource": resource,
            "strategy": strategy,
            "candidates": agent_ids,
        }

    def consensus_vote(
        self,
        agent_ids: List[str],
        proposal: str,
        quorum: float = 0.5,
    ) -> Dict[str, Any]:
        """共识投票

        智能体对提案进行投票，达到法定人数后得出结果。

        Args:
            agent_ids: 投票智能体ID列表
            proposal: 提案描述
            quorum: 通过所需的赞成比例 (0.0 ~ 1.0)

        Returns:
            投票结果
        """
        if not agent_ids:
            return {"success": False, "reason": "无投票智能体"}

        vote_id = str(uuid.uuid4())

        # 模拟投票（每个智能体随机投票）
        import random
        votes = {}
        for agent_id in agent_ids:
            # 模拟：70%概率赞成
            votes[agent_id] = "for" if random.random() < 0.7 else "against"

        for_count = sum(1 for v in votes.values() if v == "for")
        against_count = sum(1 for v in votes.values() if v == "against")
        total = len(votes)
        ratio = for_count / max(total, 1)

        passed = ratio >= quorum

        result = {
            "vote_id": vote_id,
            "proposal": proposal,
            "votes": votes,
            "for_count": for_count,
            "against_count": against_count,
            "total": total,
            "ratio": round(ratio, 4),
            "quorum": quorum,
            "passed": passed,
        }

        with self._lock:
            self._votes[vote_id] = result

        return result


class ResourceLock:
    """分布式资源锁

    基于threading实现的分布式资源锁，支持：
    - 获取/释放锁
    - 超时机制
    - 死锁检测（等待图分析）
    - 锁持有者跟踪

    线程安全实现。
    """

    def __init__(self):
        self._lock = threading.RLock()
        # 资源锁状态 {resource_id: {"owner": agent_id, "lock_time": float, "thread_lock": Lock}}
        self._resource_locks: Dict[str, Dict[str, Any]] = {}
        # 等待图 {waiting_agent: {resource_id: holding_agent}}
        self._wait_graph: Dict[str, Dict[str, str]] = defaultdict(dict)
        # 锁条件变量用于等待
        self._conditions: Dict[str, threading.Condition] = {}

    def acquire(
        self,
        resource_id: str,
        agent_id: str,
        timeout: float = 30.0,
    ) -> bool:
        """获取资源锁

        如果资源已被锁定，则等待直到超时或锁被释放。

        Args:
            resource_id: 资源ID
            agent_id: 请求者ID
            timeout: 超时时间（秒）

        Returns:
            是否成功获取锁
        """
        deadline = time.time() + timeout

        while True:
            with self._lock:
                resource_info = self._resource_locks.get(resource_id)

                if resource_info is None or resource_info["owner"] == agent_id:
                    # 资源未被锁定或已被自己持有
                    self._resource_locks[resource_id] = {
                        "owner": agent_id,
                        "lock_time": time.time(),
                    }
                    # 从等待图中移除
                    self._remove_from_wait_graph(agent_id, resource_id)
                    return True

                # 资源被其他智能体持有，加入等待图
                current_owner = resource_info["owner"]
                self._wait_graph[agent_id][resource_id] = current_owner

                # 死锁检测
                if self._detect_deadlock(agent_id):
                    # 检测到死锁，放弃获取
                    self._remove_from_wait_graph(agent_id, resource_id)
                    return False

            # 等待锁释放
            remaining = deadline - time.time()
            if remaining <= 0:
                with self._lock:
                    self._remove_from_wait_graph(agent_id, resource_id)
                return False

            # 使用条件变量等待
            with self._lock:
                if resource_id not in self._conditions:
                    self._conditions[resource_id] = threading.Condition(self._lock)
                cond = self._conditions[resource_id]

            cond.wait(timeout=min(remaining, 1.0))

    def release(self, resource_id: str, agent_id: str) -> bool:
        """释放资源锁

        Args:
            resource_id: 资源ID
            agent_id: 持有者ID

        Returns:
            是否成功释放
        """
        with self._lock:
            resource_info = self._resource_locks.get(resource_id)
            if resource_info is None:
                return False
            if resource_info["owner"] != agent_id:
                return False

            del self._resource_locks[resource_id]

            # 通知等待者
            if resource_id in self._conditions:
                self._conditions[resource_id].notify_all()

            return True

    def is_locked(self, resource_id: str) -> bool:
        """检查资源是否被锁定"""
        with self._lock:
            return resource_id in self._resource_locks

    def get_owner(self, resource_id: str) -> Optional[str]:
        """获取资源锁持有者"""
        with self._lock:
            info = self._resource_locks.get(resource_id)
            return info["owner"] if info else None

    def _detect_deadlock(self, start_agent: str) -> bool:
        """死锁检测（等待图环路检测）

        使用DFS检测等待图中是否存在环路。

        Args:
            start_agent: 起始智能体ID

        Returns:
            是否存在死锁
        """
        visited = set()
        path = set()

        def _dfs(agent: str) -> bool:
            if agent in path:
                return True  # 发现环路
            if agent in visited:
                return False

            visited.add(agent)
            path.add(agent)

            # 查看该智能体等待的资源
            for resource_id, holding_agent in self._wait_graph.get(agent, {}).items():
                if _dfs(holding_agent):
                    return True

            path.remove(agent)
            return False

        return _dfs(start_agent)

    def _remove_from_wait_graph(self, agent_id: str, resource_id: str) -> None:
        """从等待图中移除条目"""
        if agent_id in self._wait_graph:
            self._wait_graph[agent_id].pop(resource_id, None)
            if not self._wait_graph[agent_id]:
                del self._wait_graph[agent_id]

    def get_wait_graph(self) -> Dict[str, Dict[str, str]]:
        """获取当前等待图"""
        with self._lock:
            return dict(self._wait_graph)

    def get_all_locks(self) -> Dict[str, str]:
        """获取所有锁及其持有者"""
        with self._lock:
            return {
                rid: info["owner"]
                for rid, info in self._resource_locks.items()
            }

    def force_release(self, resource_id: str) -> bool:
        """强制释放资源锁（管理员操作）"""
        with self._lock:
            if resource_id in self._resource_locks:
                del self._resource_locks[resource_id]
                if resource_id in self._conditions:
                    self._conditions[resource_id].notify_all()
                return True
            return False


class SharedBlackboard:
    """共享黑板

    多智能体共享的信息黑板，支持：
    - 键值对读写
    - 变更订阅和通知
    - 历史记录
    - 命名空间隔离

    线程安全实现。
    """

    def __init__(self, max_history: int = 1000):
        """
        Args:
            max_history: 每个key的最大历史记录数
        """
        self._lock = threading.RLock()
        # 数据存储 {key: value}
        self._data: Dict[str, Any] = {}
        # 变更历史 {key: [(value, agent_id, timestamp), ...]}
        self._history: Dict[str, List[Tuple[Any, str, float]]] = {}
        # 订阅者 {key: [callback, ...]}
        self._subscribers: Dict[str, List[Callable]] = defaultdict(list)
        # 全局订阅者
        self._global_subscribers: List[Callable] = []
        self._max_history = max_history

    def write(self, key: str, value: Any, agent_id: str = "system") -> bool:
        """写入数据

        Args:
            key: 键名
            value: 值
            agent_id: 写入者ID

        Returns:
            是否成功写入
        """
        timestamp = time.time()
        old_value = None

        with self._lock:
            old_value = self._data.get(key)
            self._data[key] = value

            # 记录历史
            if key not in self._history:
                self._history[key] = []
            self._history[key].append((value, agent_id, timestamp))
            if len(self._history[key]) > self._max_history:
                self._history[key] = self._history[key][-self._max_history:]

        # 通知订阅者（在锁外执行，避免死锁）
        self._notify_subscribers(key, value, old_value, agent_id, timestamp)

        return True

    def read(self, key: str, default: Any = None) -> Any:
        """读取数据

        Args:
            key: 键名
            default: 默认值

        Returns:
            键对应的值，不存在则返回默认值
        """
        with self._lock:
            return self._data.get(key, default)

    def delete(self, key: str) -> bool:
        """删除数据"""
        with self._lock:
            if key in self._data:
                del self._data[key]
                return True
            return False

    def exists(self, key: str) -> bool:
        """检查键是否存在"""
        with self._lock:
            return key in self._data

    def keys(self) -> List[str]:
        """获取所有键"""
        with self._lock:
            return list(self._data.keys())

    def get_history(self, key: str, limit: int = 50) -> List[Dict[str, Any]]:
        """获取键的变更历史

        Args:
            key: 键名
            limit: 返回记录数限制

        Returns:
            历史记录列表
        """
        with self._lock:
            records = self._history.get(key, [])
            result = []
            for value, agent_id, timestamp in records[-limit:]:
                result.append({
                    "value": value,
                    "agent_id": agent_id,
                    "timestamp": timestamp,
                })
            return result

    def subscribe(self, key: str, callback: Callable) -> bool:
        """订阅键的变更

        回调签名: callback(key, new_value, old_value, agent_id, timestamp)

        Args:
            key: 键名
            callback: 回调函数

        Returns:
            是否成功订阅
        """
        if not callable(callback):
            raise TypeError("回调必须是可调用对象")

        with self._lock:
            self._subscribers[key].append(callback)
            return True

    def subscribe_all(self, callback: Callable) -> bool:
        """订阅所有键的变更"""
        if not callable(callback):
            raise TypeError("回调必须是可调用对象")

        with self._lock:
            self._global_subscribers.append(callback)
            return True

    def unsubscribe(self, key: str, callback: Callable) -> bool:
        """取消订阅"""
        with self._lock:
            handlers = self._subscribers.get(key, [])
            try:
                handlers.remove(callback)
                return True
            except ValueError:
                return False

    def _notify_subscribers(
        self,
        key: str,
        new_value: Any,
        old_value: Any,
        agent_id: str,
        timestamp: float,
    ) -> None:
        """通知订阅者"""
        # 通知键级订阅者
        with self._lock:
            handlers = list(self._subscribers.get(key, []))
            global_handlers = list(self._global_subscribers)

        for handler in handlers:
            try:
                handler(key, new_value, old_value, agent_id, timestamp)
            except Exception:
                pass

        # 通知全局订阅者
        for handler in global_handlers:
            try:
                handler(key, new_value, old_value, agent_id, timestamp)
            except Exception:
                pass

    def clear(self) -> None:
        """清空黑板"""
        with self._lock:
            self._data.clear()
            self._history.clear()

    def get_snapshot(self) -> Dict[str, Any]:
        """获取黑板快照"""
        with self._lock:
            return dict(self._data)

    def __repr__(self) -> str:
        with self._lock:
            return f"SharedBlackboard(keys={len(self._data)})"
