"""
网络隔离策略
定义网络规则、DNS限制等网络隔离配置
"""

import json
from typing import Any, Dict, List, Optional, Set, Union
from dataclasses import dataclass, field
from enum import Enum
from ipaddress import IPv4Address, IPv4Network, IPv6Address, IPv6Network, ip_network


class NetworkAction(Enum):
    """网络动作枚举"""
    ALLOW = "allow"
    DENY = "deny"
    DROP = "drop"
    REJECT = "reject"
    LOG = "log"


class Protocol(Enum):
    """网络协议枚举"""
    TCP = "tcp"
    UDP = "udp"
    ICMP = "icmp"
    ALL = "all"


class NetworkDirection(Enum):
    """网络方向枚举"""
    INGRESS = "ingress"
    EGRESS = "egress"
    BOTH = "both"


@dataclass
class PortRange:
    """端口范围"""
    start: int
    end: int
    
    def __post_init__(self):
        if self.start < 0 or self.start > 65535:
            raise ValueError(f"Invalid start port: {self.start}")
        if self.end < 0 or self.end > 65535:
            raise ValueError(f"Invalid end port: {self.end}")
        if self.start > self.end:
            raise ValueError(f"Start port {self.start} > end port {self.end}")
    
    def contains(self, port: int) -> bool:
        """检查端口是否在范围内"""
        return self.start <= port <= self.end
    
    def to_dict(self) -> Dict[str, int]:
        """转换为字典"""
        return {'start': self.start, 'end': self.end}
    
    @classmethod
    def single(cls, port: int) -> 'PortRange':
        """创建单端口范围"""
        return cls(start=port, end=port)
    
    @classmethod
    def well_known(cls) -> 'PortRange':
        """知名端口范围 (0-1023)"""
        return cls(start=0, end=1023)
    
    @classmethod
    def registered(cls) -> 'PortRange':
        """注册端口范围 (1024-49151)"""
        return cls(start=1024, end=49151)
    
    @classmethod
    def dynamic(cls) -> 'PortRange':
        """动态端口范围 (49152-65535)"""
        return cls(start=49152, end=65535)


@dataclass
class NetworkRule:
    """网络规则"""
    action: NetworkAction
    direction: NetworkDirection
    protocol: Protocol = Protocol.ALL
    port: Optional[PortRange] = None
    source: Optional[str] = None       # 源IP/CIDR
    destination: Optional[str] = None  # 目标IP/CIDR
    interface: Optional[str] = None    # 网络接口
    comment: Optional[str] = None      # 注释
    priority: int = 100                # 优先级（数字越小优先级越高）
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        result = {
            'action': self.action.value,
            'direction': self.direction.value,
            'protocol': self.protocol.value,
            'priority': self.priority
        }
        if self.port:
            result['port'] = self.port.to_dict()
        if self.source:
            result['source'] = self.source
        if self.destination:
            result['destination'] = self.destination
        if self.interface:
            result['interface'] = self.interface
        if self.comment:
            result['comment'] = self.comment
        return result
    
    def matches(
        self,
        direction: NetworkDirection,
        protocol: Protocol,
        port: Optional[int] = None,
        source: Optional[str] = None,
        destination: Optional[str] = None
    ) -> bool:
        """
        检查是否匹配
        
        Args:
            direction: 方向
            protocol: 协议
            port: 端口
            source: 源地址
            destination: 目标地址
            
        Returns:
            是否匹配
        """
        # 检查方向
        if self.direction != NetworkDirection.BOTH and self.direction != direction:
            return False
        
        # 检查协议
        if self.protocol != Protocol.ALL and self.protocol != protocol:
            return False
        
        # 检查端口
        if self.port and port is not None:
            if not self.port.contains(port):
                return False
        
        return True


@dataclass
class DNSConfig:
    """DNS配置"""
    enabled: bool = False
    allowed_servers: List[str] = field(default_factory=list)  # 允许的DNS服务器
    blocked_domains: List[str] = field(default_factory=list)   # 禁止解析的域名
    allowed_domains: List[str] = field(default_factory=list)   # 允许解析的域名
    dns_over_https: bool = False                               # 是否使用DoH
    dns_over_tls: bool = False                                 # 是否使用DoT
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'enabled': self.enabled,
            'allowed_servers': self.allowed_servers,
            'blocked_domains': self.blocked_domains,
            'allowed_domains': self.allowed_domains,
            'dns_over_https': self.dns_over_https,
            'dns_over_tls': self.dns_over_tls
        }
    
    @classmethod
    def disabled(cls) -> 'DNSConfig':
        """创建禁用DNS的配置"""
        return cls(enabled=False)
    
    @classmethod
    def restricted(cls, allowed_domains: List[str]) -> 'DNSConfig':
        """创建受限DNS配置"""
        return cls(
            enabled=True,
            allowed_domains=allowed_domains,
            allowed_servers=['8.8.8.8', '1.1.1.1']  # Google和Cloudflare DNS
        )


class NetworkPolicy:
    """
    网络隔离策略
    定义沙箱的网络访问规则
    """
    
    def __init__(
        self,
        name: str = "default_policy",
        default_ingress: NetworkAction = NetworkAction.DENY,
        default_egress: NetworkAction = NetworkAction.DENY
    ):
        """
        初始化网络策略
        
        Args:
            name: 策略名称
            default_ingress: 默认入站动作
            default_egress: 默认出站动作
        """
        self.name = name
        self.default_ingress = default_ingress
        self.default_egress = default_egress
        self._rules: List[NetworkRule] = []
        self._dns_config = DNSConfig()
        self._allowed_hosts: Set[str] = set()
        self._blocked_hosts: Set[str] = set()
    
    @property
    def dns_config(self) -> DNSConfig:
        """获取DNS配置"""
        return self._dns_config
    
    @dns_config.setter
    def dns_config(self, config: DNSConfig) -> None:
        """设置DNS配置"""
        self._dns_config = config
    
    def add_rule(self, rule: NetworkRule) -> 'NetworkPolicy':
        """
        添加规则
        
        Args:
            rule: 网络规则
            
        Returns:
            self，支持链式调用
        """
        self._rules.append(rule)
        self._rules.sort(key=lambda r: r.priority)
        return self
    
    def allow_ingress(
        self,
        port: Union[int, PortRange],
        protocol: Protocol = Protocol.TCP,
        source: Optional[str] = None,
        interface: Optional[str] = None
    ) -> 'NetworkPolicy':
        """
        允许入站流量
        
        Args:
            port: 端口或端口范围
            protocol: 协议
            source: 源地址
            interface: 网络接口
            
        Returns:
            self
        """
        if isinstance(port, int):
            port = PortRange.single(port)
        
        rule = NetworkRule(
            action=NetworkAction.ALLOW,
            direction=NetworkDirection.INGRESS,
            protocol=protocol,
            port=port,
            source=source,
            interface=interface
        )
        return self.add_rule(rule)
    
    def allow_egress(
        self,
        port: Union[int, PortRange],
        protocol: Protocol = Protocol.TCP,
        destination: Optional[str] = None,
        interface: Optional[str] = None
    ) -> 'NetworkPolicy':
        """
        允许出站流量
        
        Args:
            port: 端口或端口范围
            protocol: 协议
            destination: 目标地址
            interface: 网络接口
            
        Returns:
            self
        """
        if isinstance(port, int):
            port = PortRange.single(port)
        
        rule = NetworkRule(
            action=NetworkAction.ALLOW,
            direction=NetworkDirection.EGRESS,
            protocol=protocol,
            port=port,
            destination=destination,
            interface=interface
        )
        return self.add_rule(rule)
    
    def deny_ingress(
        self,
        port: Union[int, PortRange],
        protocol: Protocol = Protocol.TCP,
        source: Optional[str] = None
    ) -> 'NetworkPolicy':
        """拒绝入站流量"""
        if isinstance(port, int):
            port = PortRange.single(port)
        
        rule = NetworkRule(
            action=NetworkAction.DENY,
            direction=NetworkDirection.INGRESS,
            protocol=protocol,
            port=port,
            source=source
        )
        return self.add_rule(rule)
    
    def deny_egress(
        self,
        port: Union[int, PortRange],
        protocol: Protocol = Protocol.TCP,
        destination: Optional[str] = None
    ) -> 'NetworkPolicy':
        """拒绝出站流量"""
        if isinstance(port, int):
            port = PortRange.single(port)
        
        rule = NetworkRule(
            action=NetworkAction.DENY,
            direction=NetworkDirection.EGRESS,
            protocol=protocol,
            port=port,
            destination=destination
        )
        return self.add_rule(rule)
    
    def allow_host(self, host: str, port: Optional[int] = None) -> 'NetworkPolicy':
        """
        允许访问指定主机
        
        Args:
            host: 主机名或IP
            port: 端口（可选）
            
        Returns:
            self
        """
        self._allowed_hosts.add(host)
        if port:
            self.allow_egress(port, destination=host)
        else:
            self.allow_egress(PortRange(0, 65535), destination=host)
        return self
    
    def block_host(self, host: str) -> 'NetworkPolicy':
        """
        禁止访问指定主机
        
        Args:
            host: 主机名或IP
            
        Returns:
            self
        """
        self._blocked_hosts.add(host)
        self.deny_egress(PortRange(0, 65535), destination=host)
        return self
    
    def allow_http(self) -> 'NetworkPolicy':
        """允许HTTP (端口80)"""
        return self.allow_egress(80, Protocol.TCP)
    
    def allow_https(self) -> 'NetworkPolicy':
        """允许HTTPS (端口443)"""
        return self.allow_egress(443, Protocol.TCP)
    
    def allow_dns(self, servers: Optional[List[str]] = None) -> 'NetworkPolicy':
        """
        允许DNS查询
        
        Args:
            servers: DNS服务器列表
            
        Returns:
            self
        """
        # 允许UDP 53端口
        self.allow_egress(53, Protocol.UDP)
        self.allow_egress(53, Protocol.TCP)  # TCP DNS
        
        if servers:
            self._dns_config.allowed_servers = servers
        self._dns_config.enabled = True
        
        return self
    
    def disable_network(self) -> 'NetworkPolicy':
        """
        完全禁用网络
        
        Returns:
            self
        """
        self.default_ingress = NetworkAction.DENY
        self.default_egress = NetworkAction.DENY
        self._rules.clear()
        self._dns_config = DNSConfig.disabled()
        return self
    
    def check_connection(
        self,
        direction: NetworkDirection,
        protocol: Protocol,
        port: int,
        destination: Optional[str] = None
    ) -> NetworkAction:
        """
        检查连接是否允许
        
        Args:
            direction: 方向
            protocol: 协议
            port: 端口
            destination: 目标地址
            
        Returns:
            动作
        """
        # 检查规则
        for rule in self._rules:
            if rule.matches(direction, protocol, port, destination=destination):
                return rule.action
        
        # 返回默认动作
        if direction == NetworkDirection.INGRESS:
            return self.default_ingress
        return self.default_egress
    
    def get_rules(self) -> List[NetworkRule]:
        """获取所有规则"""
        return list(self._rules)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'name': self.name,
            'default_ingress': self.default_ingress.value,
            'default_egress': self.default_egress.value,
            'rules': [rule.to_dict() for rule in self._rules],
            'dns': self._dns_config.to_dict(),
            'allowed_hosts': list(self._allowed_hosts),
            'blocked_hosts': list(self._blocked_hosts)
        }
    
    def to_json(self, indent: int = 2) -> str:
        """转换为JSON"""
        return json.dumps(self.to_dict(), indent=indent)
    
    def save(self, filepath: str) -> None:
        """保存到文件"""
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(self.to_json())
    
    @classmethod
    def load(cls, filepath: str) -> 'NetworkPolicy':
        """从文件加载"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls.from_dict(data)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'NetworkPolicy':
        """从字典创建"""
        policy = cls(
            name=data.get('name', 'default_policy'),
            default_ingress=NetworkAction(data.get('default_ingress', 'deny')),
            default_egress=NetworkAction(data.get('default_egress', 'deny'))
        )
        
        for rule_data in data.get('rules', []):
            port = None
            if 'port' in rule_data:
                port = PortRange(**rule_data['port'])
            
            rule = NetworkRule(
                action=NetworkAction(rule_data['action']),
                direction=NetworkDirection(rule_data['direction']),
                protocol=Protocol(rule_data.get('protocol', 'all')),
                port=port,
                source=rule_data.get('source'),
                destination=rule_data.get('destination'),
                interface=rule_data.get('interface'),
                comment=rule_data.get('comment'),
                priority=rule_data.get('priority', 100)
            )
            policy._rules.append(rule)
        
        if 'dns' in data:
            dns_data = data['dns']
            policy._dns_config = DNSConfig(
                enabled=dns_data.get('enabled', False),
                allowed_servers=dns_data.get('allowed_servers', []),
                blocked_domains=dns_data.get('blocked_domains', []),
                allowed_domains=dns_data.get('allowed_domains', []),
                dns_over_https=dns_data.get('dns_over_https', False),
                dns_over_tls=dns_data.get('dns_over_tls', False)
            )
        
        policy._allowed_hosts = set(data.get('allowed_hosts', []))
        policy._blocked_hosts = set(data.get('blocked_hosts', []))
        
        return policy


class NetworkPolicyBuilder:
    """网络策略构建器"""
    
    @staticmethod
    def create_isolated() -> NetworkPolicy:
        """创建完全隔离的网络策略"""
        policy = NetworkPolicy(
            name="isolated",
            default_ingress=NetworkAction.DENY,
            default_egress=NetworkAction.DENY
        )
        policy._dns_config = DNSConfig.disabled()
        return policy
    
    @staticmethod
    def create_web_only() -> NetworkPolicy:
        """创建仅允许Web访问的策略"""
        policy = NetworkPolicy(
            name="web_only",
            default_ingress=NetworkAction.DENY,
            default_egress=NetworkAction.DENY
        )
        policy.allow_http()
        policy.allow_https()
        policy.allow_dns()
        return policy
    
    @staticmethod
    def create_api_access(allowed_hosts: List[str]) -> NetworkPolicy:
        """创建API访问策略"""
        policy = NetworkPolicy(
            name="api_access",
            default_ingress=NetworkAction.DENY,
            default_egress=NetworkAction.DENY
        )
        
        for host in allowed_hosts:
            policy.allow_host(host, 443)  # HTTPS
        
        policy.allow_dns()
        return policy
    
    @staticmethod
    def create_custom(
        allowed_ports: List[int],
        allowed_hosts: Optional[List[str]] = None
    ) -> NetworkPolicy:
        """创建自定义策略"""
        policy = NetworkPolicy(
            name="custom",
            default_ingress=NetworkAction.DENY,
            default_egress=NetworkAction.DENY
        )
        
        for port in allowed_ports:
            policy.allow_egress(port)
        
        if allowed_hosts:
            for host in allowed_hosts:
                policy.allow_host(host)
        
        policy.allow_dns()
        return policy


# 预定义策略
ISOLATED_POLICY = NetworkPolicyBuilder.create_isolated()
WEB_ONLY_POLICY = NetworkPolicyBuilder.create_web_only()
