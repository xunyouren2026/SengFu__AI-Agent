"""
网络工具模块
提供端口扫描、连接测试、防火墙规则功能
"""

import os
import socket
import struct
import time
import threading
import subprocess
import re
from typing import Optional, Union, List, Dict, Any, Tuple, Callable, Set
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

logger = logging.getLogger(__name__)


class PortState(Enum):
    """端口状态枚举"""
    OPEN = "open"
    CLOSED = "closed"
    FILTERED = "filtered"
    UNKNOWN = "unknown"


class Protocol(Enum):
    """协议枚举"""
    TCP = "tcp"
    UDP = "udp"


@dataclass
class PortInfo:
    """端口信息数据类"""
    port: int
    state: PortState
    protocol: Protocol
    service: Optional[str] = None
    banner: Optional[str] = None
    latency: Optional[float] = None


@dataclass
class ConnectionInfo:
    """连接信息数据类"""
    local_addr: str
    local_port: int
    remote_addr: str
    remote_port: int
    protocol: str
    state: str
    pid: Optional[int] = None
    process_name: Optional[str] = None


@dataclass
class NetworkInterface:
    """网络接口数据类"""
    name: str
    addresses: List[Dict[str, str]]
    is_up: bool
    speed: Optional[int] = None
    mtu: Optional[int] = None
    mac: Optional[str] = None


@dataclass
class FirewallRule:
    """防火墙规则数据类"""
    name: str
    direction: str  # 'in' or 'out'
    action: str  # 'allow' or 'deny'
    protocol: Optional[str] = None
    port: Optional[int] = None
    port_range: Optional[Tuple[int, int]] = None
    source: Optional[str] = None
    destination: Optional[str] = None
    enabled: bool = True


@dataclass
class ScanResult:
    """扫描结果数据类"""
    target: str
    ports: List[PortInfo]
    start_time: datetime
    end_time: datetime
    total_time: float
    hosts_scanned: int = 1


class PortScanner:
    """端口扫描器"""
    
    # 常用端口服务映射
    COMMON_PORTS = {
        20: 'ftp-data', 21: 'ftp', 22: 'ssh', 23: 'telnet',
        25: 'smtp', 53: 'dns', 80: 'http', 110: 'pop3',
        143: 'imap', 443: 'https', 465: 'smtps', 587: 'smtp',
        993: 'imaps', 995: 'pop3s', 3306: 'mysql', 3389: 'rdp',
        5432: 'postgresql', 5900: 'vnc', 6379: 'redis',
        8080: 'http-proxy', 8443: 'https-alt', 27017: 'mongodb',
    }
    
    def __init__(self, timeout: float = 1.0, max_threads: int = 100):
        """
        初始化端口扫描器
        
        Args:
            timeout: 连接超时时间
            max_threads: 最大线程数
        """
        self.timeout = timeout
        self.max_threads = max_threads
    
    def scan_port(self, host: str, port: int,
                  protocol: Protocol = Protocol.TCP,
                  grab_banner: bool = False) -> PortInfo:
        """
        扫描单个端口
        
        Args:
            host: 目标主机
            port: 端口号
            protocol: 协议类型
            grab_banner: 是否抓取banner
            
        Returns:
            端口信息
        """
        start_time = time.time()
        state = PortState.UNKNOWN
        banner = None
        
        if protocol == Protocol.TCP:
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(self.timeout)
                
                result = sock.connect_ex((host, port))
                
                if result == 0:
                    state = PortState.OPEN
                    
                    if grab_banner:
                        try:
                            sock.sendall(b'\r\n')
                            banner = sock.recv(1024).decode('utf-8', errors='replace').strip()
                        except Exception:
                            pass
                else:
                    state = PortState.CLOSED
                
                sock.close()
                
            except socket.timeout:
                state = PortState.FILTERED
            except Exception as e:
                logger.debug(f"扫描端口 {port} 失败: {e}")
                state = PortState.UNKNOWN
        
        else:  # UDP
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                sock.settimeout(self.timeout)
                
                sock.sendto(b'\x00', (host, port))
                sock.recv(1024)
                state = PortState.OPEN
                sock.close()
                
            except socket.timeout:
                state = PortState.OPEN  # UDP无响应可能是开放
            except ConnectionRefusedError:
                state = PortState.CLOSED
            except Exception:
                state = PortState.UNKNOWN
        
        latency = time.time() - start_time
        
        return PortInfo(
            port=port,
            state=state,
            protocol=protocol,
            service=self.COMMON_PORTS.get(port),
            banner=banner,
            latency=latency
        )
    
    def scan_ports(self, host: str, ports: List[int],
                   protocol: Protocol = Protocol.TCP,
                   grab_banner: bool = False,
                   progress_callback: Optional[Callable[[int, int], None]] = None) -> List[PortInfo]:
        """
        扫描多个端口
        
        Args:
            host: 目标主机
            ports: 端口列表
            protocol: 协议类型
            grab_banner: 是否抓取banner
            progress_callback: 进度回调函数
            
        Returns:
            端口信息列表
        """
        results = []
        completed = 0
        
        with ThreadPoolExecutor(max_workers=self.max_threads) as executor:
            futures = {
                executor.submit(self.scan_port, host, port, protocol, grab_banner): port
                for port in ports
            }
            
            for future in as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    port = futures[future]
                    results.append(PortInfo(
                        port=port,
                        state=PortState.UNKNOWN,
                        protocol=protocol
                    ))
                
                completed += 1
                if progress_callback:
                    progress_callback(completed, len(ports))
        
        # 按端口号排序
        results.sort(key=lambda x: x.port)
        return results
    
    def scan_range(self, host: str, start_port: int, end_port: int,
                   protocol: Protocol = Protocol.TCP,
                   grab_banner: bool = False,
                   progress_callback: Optional[Callable[[int, int], None]] = None) -> ScanResult:
        """
        扫描端口范围
        
        Args:
            host: 目标主机
            start_port: 起始端口
            end_port: 结束端口
            protocol: 协议类型
            grab_banner: 是否抓取banner
            progress_callback: 进度回调函数
            
        Returns:
            扫描结果
        """
        start_time = datetime.now()
        
        ports = list(range(start_port, end_port + 1))
        port_results = self.scan_ports(host, ports, protocol, grab_banner, progress_callback)
        
        end_time = datetime.now()
        total_time = (end_time - start_time).total_seconds()
        
        return ScanResult(
            target=host,
            ports=port_results,
            start_time=start_time,
            end_time=end_time,
            total_time=total_time
        )
    
    def scan_common_ports(self, host: str,
                          protocol: Protocol = Protocol.TCP,
                          grab_banner: bool = False) -> ScanResult:
        """
        扫描常用端口
        
        Args:
            host: 目标主机
            protocol: 协议类型
            grab_banner: 是否抓取banner
            
        Returns:
            扫描结果
        """
        return self.scan_range(host, 1, 1024, protocol, grab_banner)
    
    def quick_scan(self, host: str) -> ScanResult:
        """快速扫描（仅扫描常用端口）"""
        ports = list(self.COMMON_PORTS.keys())
        start_time = datetime.now()
        
        port_results = self.scan_ports(host, ports)
        
        end_time = datetime.now()
        total_time = (end_time - start_time).total_seconds()
        
        return ScanResult(
            target=host,
            ports=port_results,
            start_time=start_time,
            end_time=end_time,
            total_time=total_time
        )


class ConnectionTester:
    """连接测试器"""
    
    def __init__(self, timeout: float = 5.0):
        """
        初始化连接测试器
        
        Args:
            timeout: 连接超时时间
        """
        self.timeout = timeout
    
    def test_tcp_connection(self, host: str, port: int) -> Tuple[bool, float, str]:
        """
        测试TCP连接
        
        Args:
            host: 目标主机
            port: 端口号
            
        Returns:
            (是否成功, 延迟, 消息)
        """
        start_time = time.time()
        
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(self.timeout)
            
            result = sock.connect_ex((host, port))
            latency = time.time() - start_time
            
            sock.close()
            
            if result == 0:
                return True, latency, f"连接成功，延迟: {latency*1000:.2f}ms"
            else:
                return False, latency, f"连接失败，错误码: {result}"
                
        except socket.timeout:
            latency = time.time() - start_time
            return False, latency, "连接超时"
        except socket.gaierror as e:
            return False, 0, f"域名解析失败: {e}"
        except Exception as e:
            return False, 0, f"连接错误: {e}"
    
    def test_udp_connection(self, host: str, port: int) -> Tuple[bool, float, str]:
        """
        测试UDP连接
        
        Args:
            host: 目标主机
            port: 端口号
            
        Returns:
            (是否成功, 延迟, 消息)
        """
        start_time = time.time()
        
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.settimeout(self.timeout)
            
            sock.sendto(b'\x00', (host, port))
            sock.recv(1024)
            
            latency = time.time() - start_time
            sock.close()
            
            return True, latency, f"UDP响应成功，延迟: {latency*1000:.2f}ms"
            
        except socket.timeout:
            latency = time.time() - start_time
            return True, latency, "UDP无响应（可能正常）"
        except ConnectionRefusedError:
            latency = time.time() - start_time
            return False, latency, "UDP连接被拒绝"
        except Exception as e:
            return False, 0, f"UDP连接错误: {e}"
    
    def ping(self, host: str, count: int = 4,
             interval: float = 1.0) -> Dict[str, Any]:
        """
        Ping测试
        
        Args:
            host: 目标主机
            count: ping次数
            interval: 间隔时间
            
        Returns:
            ping结果
        """
        results = {
            'host': host,
            'packets_sent': 0,
            'packets_received': 0,
            'latencies': [],
            'avg_latency': None,
            'min_latency': None,
            'max_latency': None,
            'packet_loss': None
        }
        
        for i in range(count):
            try:
                # 使用ICMP ping
                if os.name == 'nt':  # Windows
                    cmd = ['ping', '-n', '1', '-w', str(int(self.timeout * 1000)), host]
                else:  # Linux/Unix
                    cmd = ['ping', '-c', '1', '-W', str(int(self.timeout)), host]
                
                start_time = time.time()
                proc = subprocess.run(cmd, capture_output=True, timeout=self.timeout + 1)
                latency = time.time() - start_time
                
                results['packets_sent'] += 1
                
                if proc.returncode == 0:
                    results['packets_received'] += 1
                    results['latencies'].append(latency)
                else:
                    results['latencies'].append(None)
                
            except subprocess.TimeoutExpired:
                results['packets_sent'] += 1
                results['latencies'].append(None)
            except Exception as e:
                logger.debug(f"Ping失败: {e}")
                results['packets_sent'] += 1
                results['latencies'].append(None)
            
            if i < count - 1:
                time.sleep(interval)
        
        # 计算统计信息
        valid_latencies = [l for l in results['latencies'] if l is not None]
        
        if valid_latencies:
            results['avg_latency'] = sum(valid_latencies) / len(valid_latencies)
            results['min_latency'] = min(valid_latencies)
            results['max_latency'] = max(valid_latencies)
        
        if results['packets_sent'] > 0:
            results['packet_loss'] = (results['packets_sent'] - results['packets_received']) / results['packets_sent'] * 100
        
        return results
    
    def check_port_reachable(self, host: str, port: int,
                              protocol: Protocol = Protocol.TCP) -> bool:
        """检查端口是否可达"""
        if protocol == Protocol.TCP:
            success, _, _ = self.test_tcp_connection(host, port)
            return success
        else:
            success, _, _ = self.test_udp_connection(host, port)
            return success
    
    def resolve_hostname(self, hostname: str) -> Tuple[bool, List[str], str]:
        """
        解析主机名
        
        Args:
            hostname: 主机名
            
        Returns:
            (是否成功, IP地址列表, 消息)
        """
        try:
            addrinfo = socket.getaddrinfo(hostname, None)
            ips = list(set(addr[4][0] for addr in addrinfo))
            return True, ips, f"解析成功，共{len(ips)}个IP"
        except socket.gaierror as e:
            return False, [], f"解析失败: {e}"
        except Exception as e:
            return False, [], f"解析错误: {e}"
    
    def reverse_dns(self, ip: str) -> Tuple[bool, str, str]:
        """
        反向DNS查询
        
        Args:
            ip: IP地址
            
        Returns:
            (是否成功, 主机名, 消息)
        """
        try:
            hostname, _, _ = socket.gethostbyaddr(ip)
            return True, hostname, f"反向解析成功"
        except socket.herror as e:
            return False, "", f"反向解析失败: {e}"
        except Exception as e:
            return False, "", f"反向解析错误: {e}"


class NetworkInfo:
    """网络信息收集器"""
    
    def get_interfaces(self) -> List[NetworkInterface]:
        """获取网络接口列表"""
        interfaces = []
        
        if HAS_PSUTIL:
            stats = psutil.net_if_stats()
            addrs = psutil.net_if_addrs()
            
            for name, stat in stats.items():
                interface = NetworkInterface(
                    name=name,
                    addresses=[],
                    is_up=stat.isup,
                    speed=stat.speed if hasattr(stat, 'speed') else None,
                    mtu=stat.mtu if hasattr(stat, 'mtu') else None,
                    mac=None
                )
                
                if name in addrs:
                    for addr in addrs[name]:
                        addr_info = {
                            'family': str(addr.family),
                            'address': addr.address,
                        }
                        if addr.netmask:
                            addr_info['netmask'] = addr.netmask
                        if addr.broadcast:
                            addr_info['broadcast'] = addr.broadcast
                        
                        interface.addresses.append(addr_info)
                        
                        # 获取MAC地址
                        if 'AF_LINK' in str(addr.family) or 'AF_PACKET' in str(addr.family):
                            interface.mac = addr.address
                
                interfaces.append(interface)
        
        return interfaces
    
    def get_connections(self, kind: str = 'inet') -> List[ConnectionInfo]:
        """
        获取网络连接
        
        Args:
            kind: 连接类型 ('inet', 'inet4', 'inet6', 'tcp', 'udp')
            
        Returns:
            连接信息列表
        """
        connections = []
        
        if HAS_PSUTIL:
            try:
                for conn in psutil.net_connections(kind=kind):
                    info = ConnectionInfo(
                        local_addr=conn.laddr.ip if conn.laddr else '',
                        local_port=conn.laddr.port if conn.laddr else 0,
                        remote_addr=conn.raddr.ip if conn.raddr else '',
                        remote_port=conn.raddr.port if conn.raddr else 0,
                        protocol='tcp' if conn.type == socket.SOCK_STREAM else 'udp',
                        state=conn.status,
                        pid=conn.pid
                    )
                    
                    # 获取进程名
                    if conn.pid:
                        try:
                            info.process_name = psutil.Process(conn.pid).name()
                        except (psutil.NoSuchProcess, psutil.AccessDenied):
                            pass
                    
                    connections.append(info)
                    
            except (psutil.AccessDenied, PermissionError):
                logger.warning("无权限获取网络连接")
        
        return connections
    
    def get_listening_ports(self) -> List[Dict[str, Any]]:
        """获取监听端口列表"""
        listening = []
        
        for conn in self.get_connections():
            if conn.state == 'LISTEN':
                listening.append({
                    'port': conn.local_port,
                    'address': conn.local_addr,
                    'protocol': conn.protocol,
                    'pid': conn.pid,
                    'process': conn.process_name
                })
        
        return listening
    
    def get_route_table(self) -> List[Dict[str, str]]:
        """获取路由表"""
        routes = []
        
        try:
            if os.name == 'nt':  # Windows
                result = subprocess.run(['route', 'print'], capture_output=True, text=True)
                # 简单解析
                lines = result.stdout.split('\n')
                for line in lines:
                    if line.strip() and not line.startswith('='):
                        parts = line.split()
                        if len(parts) >= 4 and parts[0].replace('.', '').isdigit():
                            routes.append({
                                'destination': parts[0],
                                'gateway': parts[2],
                                'interface': parts[3] if len(parts) > 3 else '',
                            })
            else:  # Linux
                result = subprocess.run(['ip', 'route'], capture_output=True, text=True)
                for line in result.stdout.strip().split('\n'):
                    if line:
                        parts = line.split()
                        route = {'raw': line}
                        if parts:
                            route['destination'] = parts[0]
                        for i, part in enumerate(parts):
                            if part == 'via':
                                route['gateway'] = parts[i + 1] if i + 1 < len(parts) else ''
                            elif part == 'dev':
                                route['interface'] = parts[i + 1] if i + 1 < len(parts) else ''
                        routes.append(route)
                        
        except Exception as e:
            logger.error(f"获取路由表失败: {e}")
        
        return routes
    
    def get_dns_servers(self) -> List[str]:
        """获取DNS服务器列表"""
        dns_servers = []
        
        try:
            if os.name == 'nt':  # Windows
                result = subprocess.run(['ipconfig', '/all'], capture_output=True, text=True)
                for line in result.stdout.split('\n'):
                    if 'DNS' in line and ':' in line:
                        match = re.search(r'(\d+\.\d+\.\d+\.\d+)', line)
                        if match:
                            dns_servers.append(match.group(1))
            else:  # Linux
                try:
                    with open('/etc/resolv.conf', 'r') as f:
                        for line in f:
                            if line.startswith('nameserver'):
                                parts = line.split()
                                if len(parts) > 1:
                                    dns_servers.append(parts[1])
                except FileNotFoundError:
                    pass
                    
        except Exception as e:
            logger.error(f"获取DNS服务器失败: {e}")
        
        return dns_servers


class FirewallManager:
    """防火墙管理器"""
    
    def __init__(self):
        """初始化防火墙管理器"""
        self._rules: List[FirewallRule] = []
        self._load_existing_rules()
    
    def _load_existing_rules(self) -> None:
        """加载现有规则"""
        try:
            if os.name == 'nt':  # Windows
                result = subprocess.run(
                    ['netsh', 'advfirewall', 'firewall', 'show', 'rule', 'name=all'],
                    capture_output=True, text=True
                )
                # 解析Windows防火墙规则
                # 简化处理，实际需要更复杂的解析
            else:  # Linux
                # 尝试读取iptables规则
                result = subprocess.run(
                    ['iptables', '-L', '-n'],
                    capture_output=True, text=True
                )
                # 解析iptables规则
        except Exception as e:
            logger.debug(f"加载防火墙规则失败: {e}")
    
    def add_rule(self, rule: FirewallRule) -> Tuple[bool, str]:
        """
        添加防火墙规则
        
        Args:
            rule: 防火墙规则
            
        Returns:
            (是否成功, 消息)
        """
        try:
            if os.name == 'nt':  # Windows
                cmd = ['netsh', 'advfirewall', 'firewall', 'add', 'rule']
                cmd.extend([f'name={rule.name}'])
                cmd.extend([f'direction={rule.direction}'])
                cmd.extend([f'action={rule.action}'])
                
                if rule.protocol:
                    cmd.extend([f'protocol={rule.protocol}'])
                if rule.port:
                    cmd.extend([f'localport={rule.port}'])
                elif rule.port_range:
                    cmd.extend([f'localport={rule.port_range[0]}-{rule.port_range[1]}'])
                
                if not rule.enabled:
                    cmd.extend(['enable=no'])
                
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                if result.returncode == 0:
                    self._rules.append(rule)
                    return True, "规则添加成功"
                else:
                    return False, f"添加失败: {result.stderr}"
                    
            else:  # Linux
                cmd = ['iptables']
                
                if rule.direction == 'in':
                    cmd.append('-A INPUT')
                else:
                    cmd.append('-A OUTPUT')
                
                if rule.protocol:
                    cmd.extend(['-p', rule.protocol])
                
                if rule.port:
                    cmd.extend(['--dport', str(rule.port)])
                elif rule.port_range:
                    cmd.extend(['--dport', f'{rule.port_range[0]}:{rule.port_range[1]}'])
                
                if rule.action == 'allow':
                    cmd.append('-j ACCEPT')
                else:
                    cmd.append('-j DROP')
                
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                if result.returncode == 0:
                    self._rules.append(rule)
                    return True, "规则添加成功"
                else:
                    return False, f"添加失败: {result.stderr}"
                    
        except Exception as e:
            return False, f"添加规则错误: {e}"
    
    def remove_rule(self, name: str) -> Tuple[bool, str]:
        """
        删除防火墙规则
        
        Args:
            name: 规则名称
            
        Returns:
            (是否成功, 消息)
        """
        try:
            if os.name == 'nt':  # Windows
                cmd = ['netsh', 'advfirewall', 'firewall', 'delete', 'rule', f'name={name}']
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                if result.returncode == 0:
                    self._rules = [r for r in self._rules if r.name != name]
                    return True, "规则删除成功"
                else:
                    return False, f"删除失败: {result.stderr}"
            else:  # Linux
                # 需要知道规则的详细信息才能删除
                return False, "Linux下删除规则需要指定完整规则"
                
        except Exception as e:
            return False, f"删除规则错误: {e}"
    
    def list_rules(self) -> List[FirewallRule]:
        """列出所有规则"""
        return self._rules.copy()
    
    def enable_firewall(self) -> Tuple[bool, str]:
        """启用防火墙"""
        try:
            if os.name == 'nt':  # Windows
                result = subprocess.run(
                    ['netsh', 'advfirewall', 'set', 'allprofiles', 'state', 'on'],
                    capture_output=True, text=True
                )
                return result.returncode == 0, "防火墙已启用" if result.returncode == 0 else "启用失败"
            else:  # Linux
                return True, "Linux防火墙需要手动配置iptables"
        except Exception as e:
            return False, f"启用防火墙错误: {e}"
    
    def disable_firewall(self) -> Tuple[bool, str]:
        """禁用防火墙"""
        try:
            if os.name == 'nt':  # Windows
                result = subprocess.run(
                    ['netsh', 'advfirewall', 'set', 'allprofiles', 'state', 'off'],
                    capture_output=True, text=True
                )
                return result.returncode == 0, "防火墙已禁用" if result.returncode == 0 else "禁用失败"
            else:  # Linux
                return True, "Linux防火墙需要手动配置iptables"
        except Exception as e:
            return False, f"禁用防火墙错误: {e}"
    
    def block_port(self, port: int, protocol: str = 'tcp',
                   direction: str = 'in') -> Tuple[bool, str]:
        """阻止端口"""
        rule = FirewallRule(
            name=f"block_{port}_{protocol}",
            direction=direction,
            action='deny',
            protocol=protocol,
            port=port
        )
        return self.add_rule(rule)
    
    def allow_port(self, port: int, protocol: str = 'tcp',
                   direction: str = 'in') -> Tuple[bool, str]:
        """允许端口"""
        rule = FirewallRule(
            name=f"allow_{port}_{protocol}",
            direction=direction,
            action='allow',
            protocol=protocol,
            port=port
        )
        return self.add_rule(rule)


class NetworkTools:
    """网络工具集合类"""
    
    def __init__(self, timeout: float = 5.0):
        """
        初始化网络工具
        
        Args:
            timeout: 默认超时时间
        """
        self.port_scanner = PortScanner(timeout=timeout)
        self.connection_tester = ConnectionTester(timeout=timeout)
        self.network_info = NetworkInfo()
        self.firewall = FirewallManager()
    
    def scan_host(self, host: str, ports: Optional[List[int]] = None,
                  quick: bool = False) -> ScanResult:
        """
        扫描主机
        
        Args:
            host: 目标主机
            ports: 端口列表
            quick: 是否快速扫描
            
        Returns:
            扫描结果
        """
        if quick:
            return self.port_scanner.quick_scan(host)
        elif ports:
            start_time = datetime.now()
            port_results = self.port_scanner.scan_ports(host, ports)
            end_time = datetime.now()
            return ScanResult(
                target=host,
                ports=port_results,
                start_time=start_time,
                end_time=end_time,
                total_time=(end_time - start_time).total_seconds()
            )
        else:
            return self.port_scanner.scan_common_ports(host)
    
    def check_connection(self, host: str, port: int,
                         protocol: Protocol = Protocol.TCP) -> Tuple[bool, float, str]:
        """检查连接"""
        if protocol == Protocol.TCP:
            return self.connection_tester.test_tcp_connection(host, port)
        else:
            return self.connection_tester.test_udp_connection(host, port)
    
    def get_network_status(self) -> Dict[str, Any]:
        """获取网络状态"""
        return {
            'interfaces': self.network_info.get_interfaces(),
            'connections': self.network_info.get_connections(),
            'listening_ports': self.network_info.get_listening_ports(),
            'dns_servers': self.network_info.get_dns_servers(),
        }
