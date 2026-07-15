"""
网络策略 - 容器网络访问控制
"""
import json
import ipaddress
from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass, field
from enum import Enum


class NetworkAction(Enum):
    """网络动作"""
    ALLOW = "allow"
    DENY = "deny"
    LOG = "log"


class Protocol(Enum):
    """协议"""
    TCP = "tcp"
    UDP = "udp"
    ICMP = "icmp"
    ALL = "all"


@dataclass
class NetworkRule:
    """网络规则"""
    action: NetworkAction
    protocol: Protocol = Protocol.ALL
    direction: str = "both"  # ingress, egress, both
    source_ip: Optional[str] = None
    source_port: Optional[int] = None
    dest_ip: Optional[str] = None
    dest_port: Optional[int] = None
    description: str = ""


@dataclass
class NetworkPolicy:
    """网络策略"""
    name: str
    default_action: NetworkAction = NetworkAction.DENY
    rules: List[NetworkRule] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "name": self.name,
            "default_action": self.default_action.value,
            "rules": [
                {
                    "action": rule.action.value,
                    "protocol": rule.protocol.value,
                    "direction": rule.direction,
                    "source_ip": rule.source_ip,
                    "source_port": rule.source_port,
                    "dest_ip": rule.dest_ip,
                    "dest_port": rule.dest_port,
                    "description": rule.description
                }
                for rule in self.rules
            ]
        }


class NetworkPolicyBuilder:
    """网络策略构建器"""
    
    def __init__(self, name: str):
        self._name = name
        self._default_action = NetworkAction.DENY
        self._rules: List[NetworkRule] = []
    
    def with_default_action(self, action: NetworkAction) -> 'NetworkPolicyBuilder':
        self._default_action = action
        return self
    
    def allow_all(self) -> 'NetworkPolicyBuilder':
        """允许所有流量"""
        self._rules.append(NetworkRule(
            action=NetworkAction.ALLOW,
            protocol=Protocol.ALL,
            direction="both",
            description="允许所有流量"
        ))
        return self
    
    def deny_all(self) -> 'NetworkPolicyBuilder':
        """拒绝所有流量"""
        self._rules.append(NetworkRule(
            action=NetworkAction.DENY,
            protocol=Protocol.ALL,
            direction="both",
            description="拒绝所有流量"
        ))
        return self
    
    def allow_egress(
        self,
        dest_ip: Optional[str] = None,
        dest_port: Optional[int] = None,
        protocol: Protocol = Protocol.TCP
    ) -> 'NetworkPolicyBuilder':
        """允许出站"""
        self._rules.append(NetworkRule(
            action=NetworkAction.ALLOW,
            protocol=protocol,
            direction="egress",
            dest_ip=dest_ip,
            dest_port=dest_port,
            description=f"允许出站 {protocol.value}"
        ))
        return self
    
    def deny_egress(
        self,
        dest_ip: Optional[str] = None,
        dest_port: Optional[int] = None,
        protocol: Protocol = Protocol.TCP
    ) -> 'NetworkPolicyBuilder':
        """拒绝出站"""
        self._rules.append(NetworkRule(
            action=NetworkAction.DENY,
            protocol=protocol,
            direction="egress",
            dest_ip=dest_ip,
            dest_port=dest_port,
            description=f"拒绝出站 {protocol.value}"
        ))
        return self
    
    def allow_ingress(
        self,
        source_ip: Optional[str] = None,
        source_port: Optional[int] = None,
        protocol: Protocol = Protocol.TCP
    ) -> 'NetworkPolicyBuilder':
        """允许入站"""
        self._rules.append(NetworkRule(
            action=NetworkAction.ALLOW,
            protocol=protocol,
            direction="ingress",
            source_ip=source_ip,
            source_port=source_port,
            description=f"允许入站 {protocol.value}"
        ))
        return self
    
    def allow_dns(self) -> 'NetworkPolicyBuilder':
        """允许DNS查询"""
        self._rules.append(NetworkRule(
            action=NetworkAction.ALLOW,
            protocol=Protocol.UDP,
            direction="egress",
            dest_port=53,
            description="允许DNS查询"
        ))
        return self
    
    def allow_http(self) -> 'NetworkPolicyBuilder':
        """允许HTTP"""
        self.allow_egress(dest_port=80, protocol=Protocol.TCP)
        return self
    
    def allow_https(self) -> 'NetworkPolicyBuilder':
        """允许HTTPS"""
        self.allow_egress(dest_port=443, protocol=Protocol.TCP)
        return self
    
    def allow_api_endpoints(self, endpoints: List[str]) -> 'NetworkPolicyBuilder':
        """允许API端点"""
        for endpoint in endpoints:
            self.allow_egress(dest_ip=endpoint, protocol=Protocol.TCP)
        return self
    
    def build(self) -> NetworkPolicy:
        """构建策略"""
        return NetworkPolicy(
            name=self._name,
            default_action=self._default_action,
            rules=self._rules
        )


class NetworkPolicyManager:
    """网络策略管理器"""
    
    def __init__(self):
        self._policies: Dict[str, NetworkPolicy] = {}
        self._load_default_policies()
    
    def _load_default_policies(self) -> None:
        """加载默认策略"""
        # 无网络策略
        self._policies["none"] = NetworkPolicyBuilder("none") \
            .with_default_action(NetworkAction.DENY) \
            .deny_all() \
            .build()
        
        # 仅DNS策略
        self._policies["dns_only"] = NetworkPolicyBuilder("dns_only") \
            .with_default_action(NetworkAction.DENY) \
            .allow_dns() \
            .build()
        
        # 仅HTTP/HTTPS策略
        self._policies["web_only"] = NetworkPolicyBuilder("web_only") \
            .with_default_action(NetworkAction.DENY) \
            .allow_dns() \
            .allow_http() \
            .allow_https() \
            .build()
        
        # 开放策略
        self._policies["open"] = NetworkPolicyBuilder("open") \
            .with_default_action(NetworkAction.ALLOW) \
            .allow_all() \
            .build()
    
    def get_policy(self, name: str) -> Optional[NetworkPolicy]:
        """获取策略"""
        return self._policies.get(name)
    
    def add_policy(self, policy: NetworkPolicy) -> None:
        """添加策略"""
        self._policies[policy.name] = policy
    
    def remove_policy(self, name: str) -> bool:
        """移除策略"""
        if name in self._policies and name not in ["none", "dns_only", "web_only", "open"]:
            del self._policies[name]
            return True
        return False
    
    def list_policies(self) -> List[str]:
        """列出所有策略"""
        return list(self._policies.keys())
    
    def check_connection(
        self,
        policy_name: str,
        source_ip: str,
        dest_ip: str,
        dest_port: int,
        protocol: Protocol,
        direction: str
    ) -> NetworkAction:
        """检查连接是否允许"""
        policy = self._policies.get(policy_name)
        if not policy:
            return NetworkAction.DENY
        
        # 检查规则
        for rule in policy.rules:
            if self._rule_matches(rule, source_ip, dest_ip, dest_port, protocol, direction):
                return rule.action
        
        # 返回默认动作
        return policy.default_action
    
    def _rule_matches(
        self,
        rule: NetworkRule,
        source_ip: str,
        dest_ip: str,
        dest_port: int,
        protocol: Protocol,
        direction: str
    ) -> bool:
        """检查规则是否匹配"""
        # 检查方向
        if rule.direction != "both" and rule.direction != direction:
            return False
        
        # 检查协议
        if rule.protocol != Protocol.ALL and rule.protocol != protocol:
            return False
        
        # 检查源IP
        if rule.source_ip and not self._ip_matches(source_ip, rule.source_ip):
            return False
        
        # 检查目标IP
        if rule.dest_ip and not self._ip_matches(dest_ip, rule.dest_ip):
            return False
        
        # 检查目标端口
        if rule.dest_port and rule.dest_port != dest_port:
            return False
        
        return True
    
    def _ip_matches(self, ip: str, pattern: str) -> bool:
        """检查IP是否匹配"""
        try:
            # CIDR表示法
            if '/' in pattern:
                network = ipaddress.ip_network(pattern, strict=False)
                return ipaddress.ip_address(ip) in network
            # 单个IP
            return ip == pattern
        except ValueError:
            return False
    
    def to_docker_network_args(self, policy_name: str) -> List[str]:
        """转换为Docker网络参数"""
        policy = self._policies.get(policy_name)
        if not policy:
            return ["--network=none"]
        
        if policy_name == "none":
            return ["--network=none"]
        elif policy_name == "open":
            return []  # 默认网络
        
        # 其他策略需要更复杂的配置
        args = []
        
        # 收集允许的端口
        allowed_ports = set()
        for rule in policy.rules:
            if rule.action == NetworkAction.ALLOW and rule.dest_port:
                allowed_ports.add(rule.dest_port)
        
        return args


class IPRange:
    """IP范围"""
    
    def __init__(self, start: str, end: str = None):
        if end is None:
            # CIDR或单个IP
            if '/' in start:
                self._network = ipaddress.ip_network(start, strict=False)
                self._start = int(self._network.network_address)
                self._end = int(self._network.broadcast_address)
            else:
                self._start = int(ipaddress.ip_address(start))
                self._end = self._start
        else:
            self._start = int(ipaddress.ip_address(start))
            self._end = int(ipaddress.ip_address(end))
    
    def contains(self, ip: str) -> bool:
        """检查IP是否在范围内"""
        ip_int = int(ipaddress.ip_address(ip))
        return self._start <= ip_int <= self._end
    
    def __contains__(self, ip: str) -> bool:
        return self.contains(ip)


# 预定义IP范围
class IPRanges:
    """预定义IP范围"""
    
    LOCALHOST = IPRange("127.0.0.0/8")
    PRIVATE_A = IPRange("10.0.0.0/8")
    PRIVATE_B = IPRange("172.16.0.0/12")
    PRIVATE_C = IPRange("192.168.0.0/16")
    LINK_LOCAL = IPRange("169.254.0.0/16")
    MULTICAST = IPRange("224.0.0.0/4")
    
    @classmethod
    def is_private(cls, ip: str) -> bool:
        """检查是否为私有IP"""
        return (
            ip in cls.LOCALHOST or
            ip in cls.PRIVATE_A or
            ip in cls.PRIVATE_B or
            ip in cls.PRIVATE_C
        )
    
    @classmethod
    def is_public(cls, ip: str) -> bool:
        """检查是否为公网IP"""
        return not cls.is_private(ip) and ip not in cls.LINK_LOCAL
