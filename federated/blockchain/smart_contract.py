"""
Smart Contract - 智能合约接口
联邦学习中的智能合约抽象层实现

提供智能合约的定义、部署、调用和状态管理功能。
包含激励合约和信誉合约两个预置合约。

Author: AGI Unified Framework
"""

import hashlib
import json
import time
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Callable
from enum import Enum
from collections import deque


# ============== 合约ABI ==============

class ParamType(Enum):
    """参数类型枚举"""
    ADDRESS = "address"
    UINT256 = "uint256"
    INT256 = "int256"
    BOOL = "bool"
    STRING = "string"
    BYTES = "bytes"
    FLOAT = "float"
    ARRAY = "array"
    MAP = "map"
    ANY = "any"


@dataclass
class ABIParameter:
    """
    ABI参数定义

    Attributes:
        name: 参数名称
        param_type: 参数类型
        is_optional: 是否可选
        default_value: 默认值
    """
    name: str
    param_type: ParamType
    is_optional: bool = False
    default_value: Any = None

    def validate(self, value: Any) -> bool:
        """验证参数值是否符合类型"""
        if value is None:
            return self.is_optional

        type_checks = {
            ParamType.ADDRESS: lambda v: isinstance(v, str) and len(v) == 42,
            ParamType.UINT256: lambda v: isinstance(v, int) and v >= 0,
            ParamType.INT256: lambda v: isinstance(v, int),
            ParamType.BOOL: lambda v: isinstance(v, bool),
            ParamType.STRING: lambda v: isinstance(v, str),
            ParamType.BYTES: lambda v: isinstance(v, (bytes, str)),
            ParamType.FLOAT: lambda v: isinstance(v, (int, float)),
            ParamType.ARRAY: lambda v: isinstance(v, (list, tuple)),
            ParamType.MAP: lambda v: isinstance(v, dict),
            ParamType.ANY: lambda v: True,
        }

        checker = type_checks.get(self.param_type)
        if checker:
            return checker(value)
        return True


@dataclass
class ContractABI:
    """
    合约ABI定义

    定义合约的接口规范，包括函数签名、参数类型和返回类型。
    类似Solidity中的ABI（Application Binary Interface）。

    Attributes:
        name: 合约名称
        version: 合约版本
        functions: 函数定义列表
        events: 事件定义列表
    """
    name: str
    version: str = "1.0.0"
    functions: List['ContractFunction'] = field(default_factory=list)
    events: List['ContractEvent'] = field(default_factory=list)

    def get_function(self, name: str) -> Optional['ContractFunction']:
        """获取函数定义"""
        for func in self.functions:
            if func.name == name:
                return func
        return None

    def get_event(self, name: str) -> Optional['ContractEvent']:
        """获取事件定义"""
        for event in self.events:
            if event.name == name:
                return event
        return None

    def to_dict(self) -> Dict[str, Any]:
        """序列化为字典"""
        return {
            'name': self.name,
            'version': self.version,
            'functions': [f.to_dict() for f in self.functions],
            'events': [e.to_dict() for e in self.events]
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ContractABI':
        """从字典反序列化"""
        abi = cls(name=data['name'], version=data.get('version', '1.0.0'))
        for func_data in data.get('functions', []):
            abi.functions.append(ContractFunction.from_dict(func_data))
        for event_data in data.get('events', []):
            abi.events.append(ContractEvent.from_dict(event_data))
        return abi


# ============== 合约函数 ==============

class FunctionVisibility(Enum):
    """函数可见性"""
    PUBLIC = "public"
    PRIVATE = "private"
    INTERNAL = "internal"
    EXTERNAL = "external"


class FunctionMutability(Enum):
    """函数可变性"""
    VIEW = "view"           # 只读，不修改状态
    PURE = "pure"           # 纯函数，不读取也不修改状态
    PAYABLE = "payable"     # 可接收支付
    NONPAYABLE = "nonpayable"  # 不可接收支付


@dataclass
class ContractFunction:
    """
    合约函数调用封装

    封装合约函数的定义和调用逻辑。

    Attributes:
        name: 函数名称
        inputs: 输入参数列表
        outputs: 输出参数列表
        visibility: 函数可见性
        mutability: 函数可变性
        handler: 函数处理逻辑
    """
    name: str
    inputs: List[ABIParameter] = field(default_factory=list)
    outputs: List[ABIParameter] = field(default_factory=list)
    visibility: FunctionVisibility = FunctionVisibility.PUBLIC
    mutability: FunctionMutability = FunctionMutability.NONPAYABLE
    handler: Optional[Callable] = None

    def validate_inputs(self, args: Dict[str, Any]) -> Tuple[bool, str]:
        """
        验证输入参数

        Args:
            args: 参数字典

        Returns:
            (是否有效, 错误信息)
        """
        for param in self.inputs:
            if param.name not in args:
                if not param.is_optional:
                    return False, f"Missing required parameter: {param.name}"
                continue

            value = args[param.name]
            if not param.validate(value):
                return False, f"Invalid type for parameter '{param.name}': expected {param.param_type.value}"

        return True, ""

    def call(self, contract_state: 'ContractState',
             args: Dict[str, Any], caller: str) -> Any:
        """
        调用合约函数

        Args:
            contract_state: 合约状态
            args: 调用参数
            caller: 调用者地址

        Returns:
            函数返回值

        Raises:
            ValueError: 参数验证失败
            RuntimeError: 函数执行失败
        """
        # 验证输入
        valid, error = self.validate_inputs(args)
        if not valid:
            raise ValueError(error)

        if self.handler is None:
            raise RuntimeError(f"Function '{self.name}' has no handler")

        # 执行函数
        try:
            result = self.handler(contract_state, args, caller)
            return result
        except Exception as e:
            raise RuntimeError(f"Function '{self.name}' execution failed: {e}")

    def to_dict(self) -> Dict[str, Any]:
        """序列化为字典"""
        return {
            'name': self.name,
            'inputs': [
                {'name': p.name, 'type': p.param_type.value, 'optional': p.is_optional}
                for p in self.inputs
            ],
            'outputs': [
                {'name': p.name, 'type': p.param_type.value}
                for p in self.outputs
            ],
            'visibility': self.visibility.value,
            'mutability': self.mutability.value
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ContractFunction':
        """从字典反序列化"""
        inputs = [
            ABIParameter(
                name=p['name'],
                param_type=ParamType(p['type']),
                is_optional=p.get('optional', False)
            )
            for p in data.get('inputs', [])
        ]
        outputs = [
            ABIParameter(name=p['name'], param_type=ParamType(p['type']))
            for p in data.get('outputs', [])
        ]
        return cls(
            name=data['name'],
            inputs=inputs,
            outputs=outputs,
            visibility=FunctionVisibility(data.get('visibility', 'public')),
            mutability=FunctionMutability(data.get('mutability', 'nonpayable'))
        )


# ============== 合约事件 ==============

@dataclass
class ContractEvent:
    """
    合约事件

    合约执行过程中产生的事件日志，用于外部监听和审计。

    Attributes:
        name: 事件名称
        parameters: 事件参数列表
    """
    name: str
    parameters: List[ABIParameter] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """序列化为字典"""
        return {
            'name': self.name,
            'parameters': [
                {'name': p.name, 'type': p.param_type.value}
                for p in self.parameters
            ]
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ContractEvent':
        """从字典反序列化"""
        params = [
            ABIParameter(name=p['name'], param_type=ParamType(p['type']))
            for p in data.get('parameters', [])
        ]
        return cls(name=data['name'], parameters=params)


@dataclass
class EventLog:
    """
    事件日志条目

    Attributes:
        event_name: 事件名称
        data: 事件数据
        block_number: 区块号
        tx_hash: 交易哈希
        timestamp: 时间戳
        contract_address: 合约地址
    """
    event_name: str
    data: Dict[str, Any]
    block_number: int
    tx_hash: str
    timestamp: float = field(default_factory=time.time)
    contract_address: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            'event_name': self.event_name,
            'data': self.data,
            'block_number': self.block_number,
            'tx_hash': self.tx_hash,
            'timestamp': self.timestamp,
            'contract_address': self.contract_address
        }


# ============== 合约状态 ==============

class ContractState:
    """
    合约状态

    管理合约的存储状态，提供键值存储和映射功能。
    类似Solidity中的storage。

    状态变更是原子性的：要么全部成功，要么全部回滚。
    """

    def __init__(self):
        self._storage: Dict[str, Any] = {}
        self._maps: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.RLock()
        self._version: int = 0

    def get(self, key: str, default: Any = None) -> Any:
        """获取存储值"""
        with self._lock:
            return self._storage.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """设置存储值"""
        with self._lock:
            self._storage[key] = value
            self._version += 1

    def get_map(self, map_name: str, key: str,
                default: Any = None) -> Any:
        """获取映射值"""
        with self._lock:
            map_data = self._maps.get(map_name, {})
            return map_data.get(key, default)

    def set_map(self, map_name: str, key: str, value: Any) -> None:
        """设置映射值"""
        with self._lock:
            if map_name not in self._maps:
                self._maps[map_name] = {}
            self._maps[map_name][key] = value
            self._version += 1

    def get_map_keys(self, map_name: str) -> List[str]:
        """获取映射的所有键"""
        with self._lock:
            return list(self._maps.get(map_name, {}).keys())

    def delete(self, key: str) -> bool:
        """删除存储值"""
        with self._lock:
            if key in self._storage:
                del self._storage[key]
                self._version += 1
                return True
        return False

    def delete_map(self, map_name: str, key: str) -> bool:
        """删除映射值"""
        with self._lock:
            if map_name in self._maps and key in self._maps[map_name]:
                del self._maps[map_name][key]
                self._version += 1
                return True
        return False

    def clear(self) -> None:
        """清空所有状态"""
        with self._lock:
            self._storage.clear()
            self._maps.clear()
            self._version += 1

    @property
    def version(self) -> int:
        """获取状态版本号"""
        return self._version

    def snapshot(self) -> Dict[str, Any]:
        """创建状态快照"""
        with self._lock:
            return {
                'storage': dict(self._storage),
                'maps': {k: dict(v) for k, v in self._maps.items()},
                'version': self._version
            }

    def restore(self, snapshot: Dict[str, Any]) -> None:
        """从快照恢复状态"""
        with self._lock:
            self._storage = dict(snapshot['storage'])
            self._maps = {k: dict(v) for k, v in snapshot['maps'].items()}
            self._version = snapshot['version']

    def get_all(self) -> Dict[str, Any]:
        """获取所有状态"""
        with self._lock:
            return {
                'storage': dict(self._storage),
                'maps': {k: dict(v) for k, v in self._maps.items()},
                'version': self._version
            }


# ============== 智能合约基类 ==============

class SmartContract:
    """
    智能合约基类

    提供合约的生命周期管理：部署、调用、查询和事件记录。

    子类需要：
    1. 定义ABI
    2. 注册函数处理器
    3. 实现业务逻辑

    Author: AGI Unified Framework
    """

    def __init__(self, contract_name: str, owner: str = ""):
        self._contract_name = contract_name
        self._owner = owner
        self._address = ""
        self._abi = ContractABI(name=contract_name)
        self._state = ContractState()
        self._event_logs: List[EventLog] = []
        self._deployed = False
        self._deploy_time: float = 0.0
        self._block_number: int = 0
        self._lock = threading.RLock()
        self._tx_counter: int = 0

    @property
    def contract_name(self) -> str:
        return self._contract_name

    @property
    def address(self) -> str:
        return self._address

    @property
    def owner(self) -> str:
        return self._owner

    @property
    def abi(self) -> ContractABI:
        return self._abi

    @property
    def state(self) -> ContractState:
        return self._state

    @property
    def is_deployed(self) -> bool:
        return self._deployed

    def _generate_address(self) -> str:
        """生成合约地址"""
        raw = f"{self._contract_name}:{self._owner}:{time.time()}"
        return "0x" + hashlib.sha256(raw.encode()).hexdigest()[:40]

    def _generate_tx_hash(self) -> str:
        """生成交易哈希"""
        self._tx_counter += 1
        raw = f"{self._address}:{self._tx_counter}:{time.time()}"
        return "0x" + hashlib.sha256(raw.encode()).hexdigest()

    def deploy(self, deployer: str = "",
               init_args: Optional[Dict[str, Any]] = None) -> str:
        """
        部署合约

        Args:
            deployer: 部署者地址
            init_args: 初始化参数

        Returns:
            合约地址
        """
        if self._deployed:
            raise RuntimeError("Contract already deployed")

        self._owner = deployer or self._owner
        self._address = self._generate_address()
        self._deployed = True
        self._deploy_time = time.time()

        # 调用初始化函数（如果存在）
        init_func = self._abi.get_function("init")
        if init_func and init_args:
            init_func.call(self._state, init_args, self._owner)

        # 记录部署事件
        self._emit_event("Deployed", {
            'contract_name': self._contract_name,
            'address': self._address,
            'owner': self._owner
        })

        return self._address

    def call(self, function_name: str, args: Optional[Dict[str, Any]] = None,
             caller: str = "") -> Any:
        """
        调用合约函数（可修改状态）

        Args:
            function_name: 函数名称
            args: 调用参数
            caller: 调用者地址

        Returns:
            函数返回值

        Raises:
            RuntimeError: 合约未部署或函数不存在
        """
        if not self._deployed:
            raise RuntimeError("Contract not deployed")

        args = args or {}
        caller = caller or self._owner

        func = self._abi.get_function(function_name)
        if func is None:
            raise RuntimeError(f"Function '{function_name}' not found")

        # 检查权限
        if func.visibility == FunctionVisibility.PRIVATE and caller != self._owner:
            raise RuntimeError("Unauthorized: private function")

        # 创建快照（用于回滚）
        snapshot = self._state.snapshot()

        try:
            result = func.call(self._state, args, caller)
            self._block_number += 1
            return result
        except Exception as e:
            # 回滚状态
            self._state.restore(snapshot)
            raise e

    def query(self, function_name: str,
              args: Optional[Dict[str, Any]] = None) -> Any:
        """
        查询合约状态（只读）

        Args:
            function_name: 函数名称
            args: 查询参数

        Returns:
            查询结果
        """
        args = args or {}
        return self.call(function_name, args, caller=self._owner)

    def get_events(self, event_name: Optional[str] = None,
                   limit: int = 100) -> List[EventLog]:
        """
        获取事件日志

        Args:
            event_name: 事件名称过滤（None表示所有事件）
            limit: 返回数量限制

        Returns:
            事件日志列表
        """
        with self._lock:
            logs = self._event_logs
            if event_name:
                logs = [l for l in logs if l.event_name == event_name]
            return logs[-limit:]

    def _emit_event(self, event_name: str,
                    data: Dict[str, Any]) -> None:
        """触发事件"""
        log = EventLog(
            event_name=event_name,
            data=data,
            block_number=self._block_number,
            tx_hash=self._generate_tx_hash(),
            contract_address=self._address
        )
        with self._lock:
            self._event_logs.append(log)

    def _register_function(self, func: ContractFunction) -> None:
        """注册合约函数"""
        self._abi.functions.append(func)

    def _register_event(self, event: ContractEvent) -> None:
        """注册合约事件"""
        self._abi.events.append(event)

    def get_info(self) -> Dict[str, Any]:
        """获取合约信息"""
        return {
            'name': self._contract_name,
            'address': self._address,
            'owner': self._owner,
            'deployed': self._deployed,
            'deploy_time': self._deploy_time,
            'block_number': self._block_number,
            'state_version': self._state.version,
            'event_count': len(self._event_logs)
        }


# ============== 激励合约 ==============

class IncentiveContract(SmartContract):
    """
    激励合约

    管理联邦学习网络中的激励分配：
    - 发放训练奖励
    - 质押管理
    - 惩罚机制
    - 奖励池管理

    Author: AGI Unified Framework
    """

    REWARD_PER_ROUND = 10.0
    MIN_STAKE = 100.0
    PENALTY_RATIO = 0.1

    def __init__(self, owner: str = ""):
        super().__init__("IncentiveContract", owner)
        self._setup_functions()
        self._setup_events()

    def _setup_functions(self) -> None:
        """注册合约函数"""
        # 发放奖励
        self._register_function(ContractFunction(
            name="reward",
            inputs=[
                ABIParameter("participant", ParamType.ADDRESS),
                ABIParameter("amount", ParamType.FLOAT),
                ABIParameter("reason", ParamType.STRING)
            ],
            outputs=[ABIParameter("success", ParamType.BOOL)],
            mutability=FunctionMutability.NONPAYABLE,
            handler=self._reward
        ))

        # 质押
        self._register_function(ContractFunction(
            name="stake",
            inputs=[
                ABIParameter("participant", ParamType.ADDRESS),
                ABIParameter("amount", ParamType.FLOAT)
            ],
            outputs=[ABIParameter("success", ParamType.BOOL)],
            mutability=FunctionMutability.NONPAYABLE,
            handler=self._stake
        ))

        # 解除质押
        self._register_function(ContractFunction(
            name="unstake",
            inputs=[
                ABIParameter("participant", ParamType.ADDRESS),
                ABIParameter("amount", ParamType.FLOAT)
            ],
            outputs=[ABIParameter("success", ParamType.BOOL)],
            mutability=FunctionMutability.NONPAYABLE,
            handler=self._unstake
        ))

        # 惩罚
        self._register_function(ContractFunction(
            name="penalize",
            inputs=[
                ABIParameter("participant", ParamType.ADDRESS),
                ABIParameter("amount", ParamType.FLOAT),
                ABIParameter("reason", ParamType.STRING)
            ],
            outputs=[ABIParameter("success", ParamType.BOOL)],
            mutability=FunctionMutability.NONPAYABLE,
            handler=self._penalize
        ))

        # 查询余额
        self._register_function(ContractFunction(
            name="get_balance",
            inputs=[ABIParameter("participant", ParamType.ADDRESS)],
            outputs=[ABIParameter("balance", ParamType.FLOAT)],
            mutability=FunctionMutability.VIEW,
            handler=self._get_balance
        ))

        # 查询质押
        self._register_function(ContractFunction(
            name="get_stake",
            inputs=[ABIParameter("participant", ParamType.ADDRESS)],
            outputs=[ABIParameter("stake", ParamType.FLOAT)],
            mutability=FunctionMutability.VIEW,
            handler=self._get_stake
        ))

        # 查询奖励池
        self._register_function(ContractFunction(
            name="get_reward_pool",
            inputs=[],
            outputs=[ABIParameter("pool", ParamType.FLOAT)],
            mutability=FunctionMutability.VIEW,
            handler=self._get_reward_pool
        ))

    def _setup_events(self) -> None:
        """注册合约事件"""
        self._register_event(ContractEvent(
            name="RewardIssued",
            parameters=[
                ABIParameter("participant", ParamType.ADDRESS),
                ABIParameter("amount", ParamType.FLOAT),
                ABIParameter("reason", ParamType.STRING)
            ]
        ))
        self._register_event(ContractEvent(
            name="Staked",
            parameters=[
                ABIParameter("participant", ParamType.ADDRESS),
                ABIParameter("amount", ParamType.FLOAT)
            ]
        ))
        self._register_event(ContractEvent(
            name="Penalized",
            parameters=[
                ABIParameter("participant", ParamType.ADDRESS),
                ABIParameter("amount", ParamType.FLOAT),
                ABIParameter("reason", ParamType.STRING)
            ]
        ))

    def _reward(self, state: ContractState, args: Dict[str, Any],
                caller: str) -> bool:
        """发放奖励"""
        participant = args['participant']
        amount = float(args['amount'])
        reason = args.get('reason', '')

        current = state.get_map("balances", participant, 0.0)
        state.set_map("balances", participant, current + amount)

        pool = state.get("reward_pool", 0.0)
        state.set("reward_pool", pool + amount)

        self._emit_event("RewardIssued", {
            'participant': participant,
            'amount': amount,
            'reason': reason
        })
        return True

    def _stake(self, state: ContractState, args: Dict[str, Any],
               caller: str) -> bool:
        """质押"""
        participant = args['participant']
        amount = float(args['amount'])

        if amount < self.MIN_STAKE:
            raise ValueError(f"Minimum stake is {self.MIN_STAKE}")

        current_stake = state.get_map("stakes", participant, 0.0)
        state.set_map("stakes", participant, current_stake + amount)

        # 从余额中扣除
        balance = state.get_map("balances", participant, 0.0)
        if balance < amount:
            raise ValueError("Insufficient balance")
        state.set_map("balances", participant, balance - amount)

        self._emit_event("Staked", {
            'participant': participant,
            'amount': amount
        })
        return True

    def _unstake(self, state: ContractState, args: Dict[str, Any],
                 caller: str) -> bool:
        """解除质押"""
        participant = args['participant']
        amount = float(args['amount'])

        current_stake = state.get_map("stakes", participant, 0.0)
        if current_stake < amount:
            raise ValueError("Insufficient stake")

        state.set_map("stakes", participant, current_stake - amount)

        # 返回余额
        balance = state.get_map("balances", participant, 0.0)
        state.set_map("balances", participant, balance + amount)

        return True

    def _penalize(self, state: ContractState, args: Dict[str, Any],
                  caller: str) -> bool:
        """惩罚"""
        participant = args['participant']
        amount = float(args['amount'])
        reason = args.get('reason', '')

        stake = state.get_map("stakes", participant, 0.0)
        penalty = min(amount, stake * self.PENALTY_RATIO)

        if penalty > 0:
            state.set_map("stakes", participant, stake - penalty)
            pool = state.get("reward_pool", 0.0)
            state.set("reward_pool", pool + penalty)

        self._emit_event("Penalized", {
            'participant': participant,
            'amount': penalty,
            'reason': reason
        })
        return True

    def _get_balance(self, state: ContractState, args: Dict[str, Any],
                     caller: str) -> float:
        """查询余额"""
        return state.get_map("balances", args['participant'], 0.0)

    def _get_stake(self, state: ContractState, args: Dict[str, Any],
                   caller: str) -> float:
        """查询质押"""
        return state.get_map("stakes", args['participant'], 0.0)

    def _get_reward_pool(self, state: ContractState, args: Dict[str, Any],
                         caller: str) -> float:
        """查询奖励池"""
        return state.get("reward_pool", 0.0)


# ============== 信誉合约 ==============

class ReputationContract(SmartContract):
    """
    信誉合约

    管理联邦学习网络中参与者的信誉评分：
    - 记录训练贡献
    - 更新信誉分
    - 查询信誉排名
    - 信誉衰减机制

    Author: AGI Unified Framework
    """

    DEFAULT_REPUTATION = 50.0
    MAX_REPUTATION = 100.0
    MIN_REPUTATION = 0.0
    DECAY_FACTOR = 0.99  # 每轮衰减因子
    CONTRIBUTION_WEIGHT = 0.1
    QUALITY_WEIGHT = 0.15

    def __init__(self, owner: str = ""):
        super().__init__("ReputationContract", owner)
        self._setup_functions()
        self._setup_events()

    def _setup_functions(self) -> None:
        """注册合约函数"""
        # 记录贡献
        self._register_function(ContractFunction(
            name="record_contribution",
            inputs=[
                ABIParameter("participant", ParamType.ADDRESS),
                ABIParameter("round", ParamType.UINT256),
                ABIParameter("data_samples", ParamType.UINT256),
                ABIParameter("model_quality", ParamType.FLOAT)
            ],
            outputs=[ABIParameter("success", ParamType.BOOL)],
            mutability=FunctionMutability.NONPAYABLE,
            handler=self._record_contribution
        ))

        # 查询信誉分
        self._register_function(ContractFunction(
            name="get_reputation",
            inputs=[ABIParameter("participant", ParamType.ADDRESS)],
            outputs=[ABIParameter("score", ParamType.FLOAT)],
            mutability=FunctionMutability.VIEW,
            handler=self._get_reputation
        ))

        # 查询排名
        self._register_function(ContractFunction(
            name="get_ranking",
            inputs=[
                ABIParameter("limit", ParamType.UINT256, is_optional=True)
            ],
            outputs=[ABIParameter("ranking", ParamType.ARRAY)],
            mutability=FunctionMutability.VIEW,
            handler=self._get_ranking
        ))

        # 获取贡献历史
        self._register_function(ContractFunction(
            name="get_contributions",
            inputs=[ABIParameter("participant", ParamType.ADDRESS)],
            outputs=[ABIParameter("contributions", ParamType.ARRAY)],
            mutability=FunctionMutability.VIEW,
            handler=self._get_contributions
        ))

        # 信誉衰减
        self._register_function(ContractFunction(
            name="decay_reputation",
            inputs=[],
            outputs=[ABIParameter("affected", ParamType.UINT256)],
            mutability=FunctionMutability.NONPAYABLE,
            handler=self._decay_reputation
        ))

    def _setup_events(self) -> None:
        """注册合约事件"""
        self._register_event(ContractEvent(
            name="ContributionRecorded",
            parameters=[
                ABIParameter("participant", ParamType.ADDRESS),
                ABIParameter("round", ParamType.UINT256),
                ABIParameter("reputation_change", ParamType.FLOAT)
            ]
        ))
        self._register_event(ContractEvent(
            name="ReputationUpdated",
            parameters=[
                ABIParameter("participant", ParamType.ADDRESS),
                ABIParameter("old_score", ParamType.FLOAT),
                ABIParameter("new_score", ParamType.FLOAT)
            ]
        ))

    def _record_contribution(self, state: ContractState, args: Dict[str, Any],
                             caller: str) -> bool:
        """记录贡献"""
        participant = args['participant']
        round_num = int(args['round'])
        data_samples = int(args['data_samples'])
        model_quality = float(args['model_quality'])

        # 计算信誉增量
        contribution_score = min(data_samples / 1000.0, 1.0) * self.CONTRIBUTION_WEIGHT
        quality_score = model_quality * self.QUALITY_WEIGHT
        reputation_delta = contribution_score + quality_score

        # 更新信誉分
        old_score = state.get_map("reputation", participant, self.DEFAULT_REPUTATION)
        new_score = min(self.MAX_REPUTATION, old_score + reputation_delta)
        state.set_map("reputation", participant, new_score)

        # 记录贡献
        contribution = {
            'round': round_num,
            'data_samples': data_samples,
            'model_quality': model_quality,
            'reputation_delta': reputation_delta,
            'timestamp': time.time()
        }

        contributions = state.get_map("contributions", participant, [])
        contributions.append(contribution)
        state.set_map("contributions", participant, contributions)

        # 更新总贡献数
        total = state.get("total_contributions", 0)
        state.set("total_contributions", total + 1)

        self._emit_event("ContributionRecorded", {
            'participant': participant,
            'round': round_num,
            'reputation_change': reputation_delta
        })
        self._emit_event("ReputationUpdated", {
            'participant': participant,
            'old_score': old_score,
            'new_score': new_score
        })
        return True

    def _get_reputation(self, state: ContractState, args: Dict[str, Any],
                        caller: str) -> float:
        """查询信誉分"""
        return state.get_map("reputation", args['participant'], self.DEFAULT_REPUTATION)

    def _get_ranking(self, state: ContractState, args: Dict[str, Any],
                     caller: str) -> List[Dict[str, Any]]:
        """查询信誉排名"""
        limit = int(args.get('limit', 20))

        # 收集所有信誉分
        all_scores: List[Dict[str, Any]] = []
        for participant in state.get_map_keys("reputation"):
            score = state.get_map("reputation", participant, 0.0)
            all_scores.append({
                'participant': participant,
                'score': score
            })

        # 按分数降序排列
        all_scores.sort(key=lambda x: x['score'], reverse=True)
        return all_scores[:limit]

    def _get_contributions(self, state: ContractState, args: Dict[str, Any],
                           caller: str) -> List[Dict[str, Any]]:
        """获取贡献历史"""
        return state.get_map("contributions", args['participant'], [])

    def _decay_reputation(self, state: ContractState, args: Dict[str, Any],
                          caller: str) -> int:
        """信誉衰减"""
        affected = 0
        for participant in state.get_map_keys("reputation"):
            score = state.get_map("reputation", participant, 0.0)
            if score > self.DEFAULT_REPUTATION:
                new_score = max(self.DEFAULT_REPUTATION, score * self.DECAY_FACTOR)
                state.set_map("reputation", participant, new_score)
                affected += 1
        return affected


# ============== 合约管理器 ==============

class ContractManager:
    """
    合约管理器

    统一管理多个智能合约的部署、调用和升级。

    功能：
    - 部署新合约
    - 调用合约函数
    - 查询合约状态
    - 升级合约
    - 合约生命周期管理

    Author: AGI Unified Framework
    """

    def __init__(self):
        self._contracts: Dict[str, SmartContract] = {}  # address -> contract
        self._contracts_by_name: Dict[str, SmartContract] = {}  # name -> contract
        self._lock = threading.RLock()
        self._deploy_history: List[Dict[str, Any]] = []

    def deploy(self, contract: SmartContract,
               deployer: str = "",
               init_args: Optional[Dict[str, Any]] = None) -> str:
        """
        部署合约

        Args:
            contract: 合约实例
            deployer: 部署者地址
            init_args: 初始化参数

        Returns:
            合约地址
        """
        address = contract.deploy(deployer, init_args)

        with self._lock:
            self._contracts[address] = contract
            self._contracts_by_name[contract.contract_name] = contract
            self._deploy_history.append({
                'name': contract.contract_name,
                'address': address,
                'deployer': deployer,
                'timestamp': time.time()
            })

        return address

    def call(self, address: str, function_name: str,
             args: Optional[Dict[str, Any]] = None,
             caller: str = "") -> Any:
        """
        调用合约函数

        Args:
            address: 合约地址
            function_name: 函数名称
            args: 调用参数
            caller: 调用者地址

        Returns:
            函数返回值
        """
        with self._lock:
            contract = self._contracts.get(address)
            if contract is None:
                raise RuntimeError(f"Contract not found: {address}")
        return contract.call(function_name, args, caller)

    def query(self, address: str, function_name: str,
              args: Optional[Dict[str, Any]] = None) -> Any:
        """
        查询合约状态

        Args:
            address: 合约地址
            function_name: 函数名称
            args: 查询参数

        Returns:
            查询结果
        """
        with self._lock:
            contract = self._contracts.get(address)
            if contract is None:
                raise RuntimeError(f"Contract not found: {address}")
        return contract.query(function_name, args)

    def get_contract(self, address: str) -> Optional[SmartContract]:
        """获取合约实例"""
        return self._contracts.get(address)

    def get_contract_by_name(self, name: str) -> Optional[SmartContract]:
        """按名称获取合约"""
        return self._contracts_by_name.get(name)

    def get_events(self, address: str,
                   event_name: Optional[str] = None) -> List[EventLog]:
        """获取合约事件"""
        with self._lock:
            contract = self._contracts.get(address)
            if contract is None:
                return []
        return contract.get_events(event_name)

    def upgrade(self, address: str, new_contract: SmartContract) -> str:
        """
        升级合约

        部署新合约并迁移状态。

        Args:
            address: 旧合约地址
            new_contract: 新合约实例

        Returns:
            新合约地址
        """
        with self._lock:
            old_contract = self._contracts.get(address)
            if old_contract is None:
                raise RuntimeError(f"Contract not found: {address}")

            # 迁移状态
            old_state = old_contract.state.snapshot()

            # 部署新合约
            new_address = new_contract.deploy(
                old_contract.owner,
                init_args=None
            )

            # 恢复状态
            new_contract.state.restore(old_state)

            # 更新注册
            self._contracts[new_address] = new_contract
            self._contracts_by_name[new_contract.contract_name] = new_contract

            # 标记旧合约
            del self._contracts[address]

            return new_address

    def list_contracts(self) -> List[Dict[str, Any]]:
        """列出所有已部署合约"""
        with self._lock:
            return [
                contract.get_info()
                for contract in self._contracts.values()
            ]

    def get_deploy_history(self) -> List[Dict[str, Any]]:
        """获取部署历史"""
        with self._lock:
            return list(self._deploy_history)


# ============== 主程序入口 ==============

if __name__ == "__main__":
    print("=== Smart Contract Demo ===\n")

    manager = ContractManager()

    # 部署激励合约
    incentive = IncentiveContract(owner="0xdeployer")
    addr = manager.deploy(incentive, deployer="0xdeployer")
    print(f"IncentiveContract deployed at: {addr}")

    # 发放奖励
    manager.call(addr, "reward", {
        'participant': '0xnode1',
        'amount': 100.0,
        'reason': 'training_round_1'
    })

    # 质押
    manager.call(addr, "stake", {
        'participant': '0xnode1',
        'amount': 200.0
    })

    # 查询余额
    balance = manager.query(addr, "get_balance", {'participant': '0xnode1'})
    print(f"Node1 balance: {balance}")

    stake = manager.query(addr, "get_stake", {'participant': '0xnode1'})
    print(f"Node1 stake: {stake}")

    # 部署信誉合约
    reputation = ReputationContract(owner="0xdeployer")
    rep_addr = manager.deploy(reputation, deployer="0xdeployer")
    print(f"\nReputationContract deployed at: {rep_addr}")

    # 记录贡献
    manager.call(rep_addr, "record_contribution", {
        'participant': '0xnode1',
        'round': 1,
        'data_samples': 5000,
        'model_quality': 0.85
    })

    # 查询信誉
    score = manager.query(rep_addr, "get_reputation", {'participant': '0xnode1'})
    print(f"Node1 reputation: {score}")

    # 查询排名
    ranking = manager.query(rep_addr, "get_ranking", {'limit': 10})
    print(f"Ranking: {ranking}")

    # 获取事件
    events = manager.get_events(addr, "RewardIssued")
    print(f"\nReward events: {len(events)}")

    # 列出合约
    contracts = manager.list_contracts()
    print(f"\nDeployed contracts: {len(contracts)}")

    print("\n=== Demo Complete ===")
