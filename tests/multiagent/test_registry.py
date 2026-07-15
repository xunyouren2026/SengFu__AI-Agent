"""
注册中心服务测试模块

测试Agent注册、心跳、租约过期、服务发现、版本兼容性和ACL访问控制。
"""

import unittest
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Set, Optional, Any
import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from multiagent.registry.schema import (
    AgentMetadata,
    AgentRegistration,
    AgentStatus,
    AgentRole,
    AgentAddress,
    DiscoveryQuery,
    DiscoveryResult,
    HealthStatus,
    ServiceEndpoint
)
from multiagent.registry.service import (
    RegistryService,
    AgentNotFoundError,
    DuplicateAgentError,
    RegistryError
)


class MockAgentMetadata:
    """用于测试的模拟Agent元数据"""

    @staticmethod
    def create_agent(
        agent_id: str,
        name: str = "TestAgent",
        version: str = "1.0.0",
        capabilities: Optional[Set[str]] = None,
        role: AgentRole = AgentRole.WORKER,
        status: AgentStatus = AgentStatus.STARTING,
        ttl_seconds: int = 30
    ) -> AgentMetadata:
        """创建测试用Agent元数据"""
        address = AgentAddress(
            host="127.0.0.1",
            port=8000,
            protocol="http"
        )
        caps = capabilities or {"text_processing", "data_analysis"}
        metadata = AgentMetadata(
            agent_id=agent_id,
            name=name,
            version=version,
            address=address,
            capabilities=caps,
            role=role,
            status=status,
            ttl_seconds=ttl_seconds,
            labels={"env": "test", "region": "us-east"}
        )
        return metadata

    @staticmethod
    def create_registration(
        agent_id: str,
        name: str = "TestAgent",
        version: str = "1.0.0",
        capabilities: Optional[Set[str]] = None,
        role: AgentRole = AgentRole.WORKER
    ) -> AgentRegistration:
        """创建测试用Agent注册请求"""
        metadata = MockAgentMetadata.create_agent(
            agent_id=agent_id,
            name=name,
            version=version,
            capabilities=capabilities,
            role=role
        )
        endpoints = [
            ServiceEndpoint(
                name="process",
                path="/api/v1/process",
                method="POST",
                description="处理任务"
            )
        ]
        return AgentRegistration(
            metadata=metadata,
            endpoints=endpoints,
            api_version="1.0.0"
        )


class TestRegistry(unittest.TestCase):
    """测试注册中心核心功能"""

    def setUp(self):
        """测试初始化"""
        self.registry = RegistryService(default_ttl=30, cleanup_interval=1.0)
        self.registry.start()

    def tearDown(self):
        """测试清理"""
        self.registry.stop()

    def test_registry_initialization(self):
        """测试注册中心初始化"""
        self.assertIsNotNone(self.registry)
        self.assertEqual(self.registry._default_ttl, 30)
        self.assertEqual(self.registry._cleanup_interval, 1.0)
        self.assertEqual(len(self.registry._agents), 0)

    def test_register_single_agent(self):
        """测试注册单个Agent"""
        registration = MockAgentMetadata.create_registration(
            agent_id="agent_001",
            name="WorkerAgent1"
        )
        result = self.registry.register(registration)
        self.assertTrue(result)

        # 验证Agent已存储
        agent = self.registry.get_agent("agent_001")
        self.assertIsNotNone(agent)
        self.assertEqual(agent.agent_id, "agent_001")
        self.assertEqual(agent.name, "WorkerAgent1")

    def test_register_multiple_agents(self):
        """测试注册多个Agent"""
        agents = []
        for i in range(5):
            registration = MockAgentMetadata.create_registration(
                agent_id=f"agent_{i:03d}",
                name=f"WorkerAgent{i}"
            )
            result = self.registry.register(registration)
            self.assertTrue(result)
            agents.append(f"agent_{i:03d}")

        # 验证所有Agent已存储
        all_agents = self.registry.get_all_agents()
        self.assertEqual(len(all_agents), 5)

        for agent_id in agents:
            agent = self.registry.get_agent(agent_id)
            self.assertIsNotNone(agent)

    def test_register_duplicate_agent_raises_error(self):
        """测试重复注册Agent抛出异常"""
        registration = MockAgentMetadata.create_registration(
            agent_id="duplicate_agent"
        )
        self.registry.register(registration)

        # 尝试重复注册应抛出异常
        with self.assertRaises(DuplicateAgentError):
            self.registry.register(registration)

    def test_deregister_existing_agent(self):
        """测试注销已存在的Agent"""
        registration = MockAgentMetadata.create_registration(
            agent_id="agent_to_remove"
        )
        self.registry.register(registration)

        # 注销Agent
        result = self.registry.deregister("agent_to_remove")
        self.assertTrue(result)

        # 验证Agent已不存在
        agent = self.registry.get_agent("agent_to_remove")
        self.assertIsNone(agent)

    def test_deregister_nonexistent_agent(self):
        """测试注销不存在的Agent"""
        result = self.registry.deregister("nonexistent_agent")
        self.assertFalse(result)

    def test_get_all_agents(self):
        """测试获取所有Agent"""
        for i in range(3):
            registration = MockAgentMetadata.create_registration(
                agent_id=f"agent_{i}"
            )
            self.registry.register(registration)

        agents = self.registry.get_all_agents()
        self.assertEqual(len(agents), 3)

    def test_get_healthy_agents(self):
        """测试获取健康的Agent"""
        # 注册不同状态的Agent
        healthy_agent = MockAgentMetadata.create_agent(
            agent_id="healthy_agent",
            status=AgentStatus.HEALTHY
        )
        unhealthy_agent = MockAgentMetadata.create_agent(
            agent_id="unhealthy_agent",
            status=AgentStatus.UNHEALTHY
        )

        self.registry.register(AgentRegistration(metadata=healthy_agent))
        self.registry.register(AgentRegistration(metadata=unhealthy_agent))

        healthy_agents = self.registry.get_healthy_agents()
        self.assertEqual(len(healthy_agents), 1)
        self.assertEqual(healthy_agents[0].agent_id, "healthy_agent")


class TestAgent(unittest.TestCase):
    """测试Agent相关功能"""

    def setUp(self):
        """测试初始化"""
        self.registry = RegistryService(default_ttl=30)
        self.registry.start()

    def tearDown(self):
        """测试清理"""
        self.registry.stop()

    def test_agent_heartbeat_success(self):
        """测试Agent心跳成功"""
        registration = MockAgentMetadata.create_registration(
            agent_id="heartbeat_agent"
        )
        self.registry.register(registration)

        # 发送心跳
        result = self.registry.heartbeat("heartbeat_agent")
        self.assertTrue(result)

        # 验证状态已更新
        agent = self.registry.get_agent("heartbeat_agent")
        self.assertEqual(agent.status, AgentStatus.HEALTHY)

    def test_agent_heartbeat_nonexistent(self):
        """测试对不存在的Agent发送心跳"""
        result = self.registry.heartbeat("nonexistent")
        self.assertFalse(result)

    def test_agent_heartbeat_updates_timestamp(self):
        """测试心跳更新时间戳"""
        registration = MockAgentMetadata.create_registration(
            agent_id="timestamp_agent"
        )
        self.registry.register(registration)

        original_heartbeat = self.registry.get_agent("timestamp_agent").last_heartbeat
        time.sleep(0.1)
        self.registry.heartbeat("timestamp_agent")
        updated_heartbeat = self.registry.get_agent("timestamp_agent").last_heartbeat

        self.assertGreater(updated_heartbeat, original_heartbeat)

    def test_agent_default_ttl(self):
        """测试Agent默认TTL"""
        registration = MockAgentMetadata.create_registration(
            agent_id="ttl_agent"
        )
        registration.metadata.ttl_seconds = 0  # 设置为0应使用默认值
        self.registry.register(registration)

        agent = self.registry.get_agent("ttl_agent")
        self.assertEqual(agent.ttl_seconds, 30)

    def test_agent_custom_ttl(self):
        """测试Agent自定义TTL"""
        registration = MockAgentMetadata.create_registration(
            agent_id="custom_ttl_agent"
        )
        registration.metadata.ttl_seconds = 60
        self.registry.register(registration)

        agent = self.registry.get_agent("custom_ttl_agent")
        self.assertEqual(agent.ttl_seconds, 60)

    def test_agent_capabilities(self):
        """测试Agent能力管理"""
        capabilities = {"python", "data_processing", "ml"}
        registration = MockAgentMetadata.create_registration(
            agent_id="cap_agent",
            capabilities=capabilities
        )
        self.registry.register(registration)

        agent = self.registry.get_agent("cap_agent")
        self.assertEqual(agent.capabilities, capabilities)
        self.assertTrue(agent.has_capability("python"))
        self.assertFalse(agent.has_capability("java"))

    def test_agent_role_assignment(self):
        """测试Agent角色分配"""
        registration = MockAgentMetadata.create_registration(
            agent_id="role_agent",
            role=AgentRole.COORDINATOR
        )
        self.registry.register(registration)

        agent = self.registry.get_agent("role_agent")
        self.assertEqual(agent.role, AgentRole.COORDINATOR)

    def test_agent_labels(self):
        """测试Agent标签"""
        registration = MockAgentMetadata.create_registration(
            agent_id="label_agent"
        )
        self.registry.register(registration)

        agent = self.registry.get_agent("label_agent")
        self.assertEqual(agent.labels.get("env"), "test")
        self.assertEqual(agent.labels.get("region"), "us-east")


class TestHeartbeat(unittest.TestCase):
    """测试心跳机制"""

    def setUp(self):
        """测试初始化"""
        self.registry = RegistryService(default_ttl=2, cleanup_interval=0.5)
        self.registry.start()

    def tearDown(self):
        """测试清理"""
        self.registry.stop()

    def test_heartbeat_updates_status(self):
        """测试心跳更新状态"""
        registration = MockAgentMetadata.create_registration(
            agent_id="status_update_agent"
        )
        self.registry.register(registration)

        # 初始状态
        agent = self.registry.get_agent("status_update_agent")
        self.assertEqual(agent.status, AgentStatus.STARTING)

        # 发送心跳后状态变为HEALTHY
        self.registry.heartbeat("status_update_agent")
        agent = self.registry.get_agent("status_update_agent")
        self.assertEqual(agent.status, AgentStatus.HEALTHY)

    def test_multiple_heartbeats(self):
        """测试多次心跳"""
        registration = MockAgentMetadata.create_registration(
            agent_id="multi_heartbeat_agent"
        )
        self.registry.register(registration)

        for _ in range(10):
            result = self.registry.heartbeat("multi_heartbeat_agent")
            self.assertTrue(result)

        agent = self.registry.get_agent("multi_heartbeat_agent")
        self.assertEqual(agent.status, AgentStatus.HEALTHY)

    def test_heartbeat_statistics(self):
        """测试心跳统计"""
        registration = MockAgentMetadata.create_registration(
            agent_id="stats_agent"
        )
        self.registry.register(registration)

        for _ in range(5):
            self.registry.heartbeat("stats_agent")

        stats = self.registry.get_stats()
        self.assertGreaterEqual(stats["total_heartbeats"], 5)

    def test_lease_expiry(self):
        """测试租约过期"""
        registration = MockAgentMetadata.create_registration(
            agent_id="expiry_agent",
            ttl_seconds=1  # 1秒租约
        )
        self.registry.register(registration)
        self.registry.heartbeat("expiry_agent")

        # 等待租约过期
        time.sleep(1.5)

        # Agent应该已过期
        agent = self.registry.get_agent("expiry_agent")
        self.assertTrue(agent.is_expired())

    def test_heartbeat_renews_lease(self):
        """测试心跳续租"""
        registration = MockAgentMetadata.create_registration(
            agent_id="renew_agent",
            ttl_seconds=2
        )
        self.registry.register(registration)

        # 等待1秒
        time.sleep(1)

        # 心跳续租
        self.registry.heartbeat("renew_agent")

        # 再等待1秒，Agent不应过期
        time.sleep(1)
        agent = self.registry.get_agent("renew_agent")
        self.assertFalse(agent.is_expired())


class TestDiscovery(unittest.TestCase):
    """测试服务发现功能"""

    def setUp(self):
        """测试初始化"""
        self.registry = RegistryService(default_ttl=30)
        self.registry.start()

    def tearDown(self):
        """测试清理"""
        self.registry.stop()

    def test_discovery_by_capability(self):
        """测试按能力发现"""
        # 注册具有不同能力的Agent
        agent1 = MockAgentMetadata.create_agent(
            agent_id="agent_nlp",
            capabilities={"nlp", "text_processing"}
        )
        agent2 = MockAgentMetadata.create_agent(
            agent_id="agent_cv",
            capabilities={"computer_vision", "image_processing"}
        )
        agent3 = MockAgentMetadata.create_agent(
            agent_id="agent_both",
            capabilities={"nlp", "computer_vision"}
        )

        self.registry.register(AgentRegistration(metadata=agent1))
        self.registry.register(AgentRegistration(metadata=agent2))
        self.registry.register(AgentRegistration(metadata=agent3))

        # 按NLP能力查询
        query = DiscoveryQuery(required_capabilities={"nlp"})
        result = self.registry.discover(query)

        self.assertEqual(result.total_count, 2)
        agent_ids = {a.agent_id for a in result.agents}
        self.assertIn("agent_nlp", agent_ids)
        self.assertIn("agent_both", agent_ids)

    def test_discovery_by_multiple_capabilities(self):
        """测试按多个能力发现"""
        agent1 = MockAgentMetadata.create_agent(
            agent_id="agent_multi",
            capabilities={"python", "data_analysis", "ml"}
        )

        self.registry.register(AgentRegistration(metadata=agent1))

        # 按多个能力查询
        query = DiscoveryQuery(
            required_capabilities={"python", "data_analysis"}
        )
        result = self.registry.discover(query)

        self.assertEqual(result.total_count, 1)
        self.assertEqual(result.agents[0].agent_id, "agent_multi")

    def test_discovery_by_role(self):
        """测试按角色发现"""
        coordinator = MockAgentMetadata.create_agent(
            agent_id="coordinator",
            role=AgentRole.COORDINATOR
        )
        worker = MockAgentMetadata.create_agent(
            agent_id="worker",
            role=AgentRole.WORKER
        )

        self.registry.register(AgentRegistration(metadata=coordinator))
        self.registry.register(AgentRegistration(metadata=worker))

        # 按角色查询
        query = DiscoveryQuery(role=AgentRole.COORDINATOR)
        result = self.registry.discover(query)

        self.assertEqual(result.total_count, 1)
        self.assertEqual(result.agents[0].role, AgentRole.COORDINATOR)

    def test_discovery_by_status(self):
        """测试按状态发现"""
        healthy = MockAgentMetadata.create_agent(
            agent_id="healthy",
            status=AgentStatus.HEALTHY
        )
        unhealthy = MockAgentMetadata.create_agent(
            agent_id="unhealthy",
            status=AgentStatus.UNHEALTHY
        )

        self.registry.register(AgentRegistration(metadata=healthy))
        self.registry.register(AgentRegistration(metadata=unhealthy))

        # 按状态查询
        query = DiscoveryQuery(status=AgentStatus.HEALTHY)
        result = self.registry.discover(query)

        self.assertEqual(result.total_count, 1)
        self.assertEqual(result.agents[0].status, AgentStatus.HEALTHY)

    def test_discovery_by_labels(self):
        """测试按标签发现"""
        agent1 = MockAgentMetadata.create_agent(
            agent_id="labeled_agent",
            capabilities=set()
        )
        agent1.labels["env"] = "production"

        self.registry.register(AgentRegistration(metadata=agent1))

        # 按标签查询
        query = DiscoveryQuery(labels={"env": "production"})
        result = self.registry.discover(query)

        self.assertEqual(result.total_count, 1)

    def test_discovery_no_matches(self):
        """测试无匹配结果"""
        query = DiscoveryQuery(
            required_capabilities={"nonexistent_capability"}
        )
        result = self.registry.discover(query)

        self.assertEqual(result.total_count, 0)
        self.assertEqual(len(result.agents), 0)

    def test_discovery_with_version_filter(self):
        """测试带版本过滤的发现"""
        agent1 = MockAgentMetadata.create_agent(
            agent_id="v1_agent",
            version="1.0.0"
        )
        agent2 = MockAgentMetadata.create_agent(
            agent_id="v2_agent",
            version="2.0.0"
        )

        self.registry.register(AgentRegistration(metadata=agent1))
        self.registry.register(AgentRegistration(metadata=agent2))

        # 版本范围查询
        query = DiscoveryQuery(
            min_version="1.5.0",
            max_version="2.5.0"
        )
        result = self.registry.discover(query)

        self.assertEqual(result.total_count, 1)
        self.assertEqual(result.agents[0].agent_id, "v2_agent")

    def test_discovery_performance(self):
        """测试发现性能"""
        # 注册大量Agent
        for i in range(100):
            agent = MockAgentMetadata.create_agent(
                agent_id=f"perf_agent_{i}",
                capabilities={"capability_a", f"capability_{i % 10}"}
            )
            self.registry.register(AgentRegistration(metadata=agent))

        # 执行查询
        query = DiscoveryQuery(
            required_capabilities={"capability_a"}
        )

        start_time = time.time()
        result = self.registry.discover(query)
        elapsed_ms = (time.time() - start_time) * 1000

        self.assertEqual(result.total_count, 100)
        self.assertLess(elapsed_ms, 100)  # 应在100ms内完成


class TestVersionCompatibility(unittest.TestCase):
    """测试版本兼容性"""

    def setUp(self):
        """测试初始化"""
        self.registry = RegistryService(default_ttl=30)
        self.registry.start()

    def tearDown(self):
        """测试清理"""
        self.registry.stop()

    def test_version_comparison(self):
        """测试版本比较"""
        agent = MockAgentMetadata.create_agent(
            agent_id="version_test",
            version="1.2.3"
        )
        self.registry.register(AgentRegistration(metadata=agent))

        # 获取版本
        retrieved = self.registry.get_agent("version_test")
        self.assertEqual(retrieved.version, "1.2.3")

    def test_min_version_filter(self):
        """测试最小版本过滤"""
        agents = [
            ("v1_agent", "1.0.0"),
            ("v2_agent", "2.0.0"),
            ("v3_agent", "3.0.0")
        ]

        for agent_id, version in agents:
            agent = MockAgentMetadata.create_agent(
                agent_id=agent_id,
                version=version
            )
            self.registry.register(AgentRegistration(metadata=agent))

        query = DiscoveryQuery(min_version="2.0.0")
        result = self.registry.discover(query)

        self.assertEqual(result.total_count, 2)
        versions = {a.version for a in result.agents}
        self.assertIn("2.0.0", versions)
        self.assertIn("3.0.0", versions)
        self.assertNotIn("1.0.0", versions)

    def test_max_version_filter(self):
        """测试最大版本过滤"""
        agents = [
            ("v1_agent", "1.0.0"),
            ("v2_agent", "2.0.0"),
            ("v3_agent", "3.0.0")
        ]

        for agent_id, version in agents:
            agent = MockAgentMetadata.create_agent(
                agent_id=agent_id,
                version=version
            )
            self.registry.register(AgentRegistration(metadata=agent))

        query = DiscoveryQuery(max_version="2.0.0")
        result = self.registry.discover(query)

        self.assertEqual(result.total_count, 2)
        versions = {a.version for a in result.agents}
        self.assertIn("1.0.0", versions)
        self.assertIn("2.0.0", versions)

    def test_version_range_filter(self):
        """测试版本范围过滤"""
        agents = [
            ("v1_agent", "1.0.0"),
            ("v2_agent", "2.0.0"),
            ("v3_agent", "3.0.0"),
            ("v4_agent", "4.0.0")
        ]

        for agent_id, version in agents:
            agent = MockAgentMetadata.create_agent(
                agent_id=agent_id,
                version=version
            )
            self.registry.register(AgentRegistration(metadata=agent))

        query = DiscoveryQuery(
            min_version="1.5.0",
            max_version="3.5.0"
        )
        result = self.registry.discover(query)

        self.assertEqual(result.total_count, 2)
        versions = {a.version for a in result.agents}
        self.assertIn("2.0.0", versions)
        self.assertIn("3.0.0", versions)


class TestACL(unittest.TestCase):
    """测试访问控制列表"""

    def setUp(self):
        """测试初始化"""
        self.registry = RegistryService(default_ttl=30)
        self.registry.start()

    def tearDown(self):
        """测试清理"""
        self.registry.stop()

    def test_event_listeners(self):
        """测试事件监听器"""
        events_received = []

        def event_listener(event_type: str, metadata: AgentMetadata):
            events_received.append((event_type, metadata.agent_id))

        # 添加监听器
        self.registry.add_event_listener(event_listener)

        # 注册Agent应触发事件
        registration = MockAgentMetadata.create_registration(
            agent_id="listener_test"
        )
        self.registry.register(registration)

        self.assertEqual(len(events_received), 1)
        self.assertEqual(events_received[0][0], "registered")
        self.assertEqual(events_received[0][1], "listener_test")

    def test_remove_event_listener(self):
        """测试移除事件监听器"""
        events_received = []

        def event_listener(event_type: str, metadata: AgentMetadata):
            events_received.append(event_type)

        self.registry.add_event_listener(event_listener)
        self.registry.remove_event_listener(event_listener)

        registration = MockAgentMetadata.create_registration(
            agent_id="removed_listener_test"
        )
        self.registry.register(registration)

        self.assertEqual(len(events_received), 0)

    def test_deregister_event(self):
        """测试注销事件"""
        events_received = []

        def event_listener(event_type: str, metadata: AgentMetadata):
            events_received.append(event_type)

        self.registry.add_event_listener(event_listener)

        registration = MockAgentMetadata.create_registration(
            agent_id="deregister_event_test"
        )
        self.registry.register(registration)
        self.registry.deregister("deregister_event_test")

        self.assertEqual(len(events_received), 2)
        self.assertIn("registered", events_received)
        self.assertIn("deregistered", events_received)

    def test_stats_tracking(self):
        """测试统计信息追踪"""
        # 注册统计
        for i in range(5):
            registration = MockAgentMetadata.create_registration(
                agent_id=f"stats_agent_{i}"
            )
            self.registry.register(registration)

        # 注销统计
        self.registry.deregister("stats_agent_0")

        # 心跳统计
        for i in range(5):
            self.registry.heartbeat(f"stats_agent_{i}")

        stats = self.registry.get_stats()
        self.assertGreaterEqual(stats["total_registered"], 5)
        self.assertGreaterEqual(stats["total_deregistered"], 1)
        self.assertGreaterEqual(stats["total_heartbeats"], 5)
        self.assertEqual(stats["current_agents"], 4)


class TestConcurrentAccess(unittest.TestCase):
    """测试并发访问"""

    def setUp(self):
        """测试初始化"""
        self.registry = RegistryService(default_ttl=30)
        self.registry.start()

    def tearDown(self):
        """测试清理"""
        self.registry.stop()

    def test_concurrent_registration(self):
        """测试并发注册"""
        errors = []
        success_count = [0]

        def register_agents(start_id: int, count: int):
            try:
                for i in range(count):
                    registration = MockAgentMetadata.create_registration(
                        agent_id=f"concurrent_{start_id}_{i}"
                    )
                    self.registry.register(registration)
                    success_count[0] += 1
            except Exception as e:
                errors.append(e)

        # 启动多个线程
        threads = []
        for i in range(5):
            thread = threading.Thread(target=register_agents, args=(i, 10))
            threads.append(thread)
            thread.start()

        # 等待所有线程完成
        for thread in threads:
            thread.join()

        self.assertEqual(len(errors), 0)
        self.assertEqual(success_count[0], 50)
        self.assertEqual(len(self.registry.get_all_agents()), 50)

    def test_concurrent_heartbeat(self):
        """测试并发心跳"""
        registration = MockAgentMetadata.create_registration(
            agent_id="concurrent_heartbeat_agent"
        )
        self.registry.register(registration)

        def send_heartbeat():
            for _ in range(100):
                self.registry.heartbeat("concurrent_heartbeat_agent")

        threads = []
        for _ in range(5):
            thread = threading.Thread(target=send_heartbeat)
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        stats = self.registry.get_stats()
        self.assertGreaterEqual(stats["total_heartbeats"], 500)

    def test_concurrent_discovery(self):
        """测试并发发现"""
        # 注册一些Agent
        for i in range(20):
            agent = MockAgentMetadata.create_agent(
                agent_id=f"discovery_agent_{i}",
                capabilities={"test_capability"}
            )
            self.registry.register(AgentRegistration(metadata=agent))

        results = []

        def perform_discovery():
            for _ in range(50):
                query = DiscoveryQuery(
                    required_capabilities={"test_capability"}
                )
                result = self.registry.discover(query)
                results.append(result.total_count)

        threads = []
        for _ in range(5):
            thread = threading.Thread(target=perform_discovery)
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        self.assertEqual(len(results), 250)
        for count in results:
            self.assertEqual(count, 20)


class TestEdgeCases(unittest.TestCase):
    """测试边界情况"""

    def setUp(self):
        """测试初始化"""
        self.registry = RegistryService(default_ttl=30)
        self.registry.start()

    def tearDown(self):
        """测试清理"""
        self.registry.stop()

    def test_empty_capabilities(self):
        """测试空能力集"""
        agent = MockAgentMetadata.create_agent(
            agent_id="empty_cap_agent",
            capabilities=set()
        )
        self.registry.register(AgentRegistration(metadata=agent))

        query = DiscoveryQuery(required_capabilities=set())
        result = self.registry.discover(query)

        self.assertEqual(result.total_count, 1)

    def test_special_characters_in_id(self):
        """测试ID中的特殊字符"""
        registration = MockAgentMetadata.create_registration(
            agent_id="agent-with-special.id_123"
        )
        result = self.registry.register(registration)
        self.assertTrue(result)

        agent = self.registry.get_agent("agent-with-special.id_123")
        self.assertIsNotNone(agent)

    def test_case_sensitive_capabilities(self):
        """测试能力区分大小写"""
        agent = MockAgentMetadata.create_agent(
            agent_id="case_test",
            capabilities={"Python", "JAVA", "JavaScript"}
        )
        self.registry.register(AgentRegistration(metadata=agent))

        query_lower = DiscoveryQuery(required_capabilities={"python"})
        result_lower = self.registry.discover(query_lower)
        self.assertEqual(result_lower.total_count, 0)

        query_exact = DiscoveryQuery(required_capabilities={"Python"})
        result_exact = self.registry.discover(query_exact)
        self.assertEqual(result_exact.total_count, 1)

    def test_zero_ttl_agent(self):
        """测试零TTL的Agent"""
        registration = MockAgentMetadata.create_registration(
            agent_id="zero_ttl_agent"
        )
        registration.metadata.ttl_seconds = 0
        self.registry.register(registration)

        agent = self.registry.get_agent("zero_ttl_agent")
        self.assertEqual(agent.ttl_seconds, 30)

    def test_large_number_of_labels(self):
        """测试大量标签"""
        registration = MockAgentMetadata.create_registration(
            agent_id="many_labels_agent"
        )
        for i in range(100):
            registration.metadata.labels[f"label_{i}"] = f"value_{i}"

        self.registry.register(registration)
        agent = self.registry.get_agent("many_labels_agent")
        self.assertEqual(len(agent.labels), 102)  # 100个新标签 + 2个默认标签


if __name__ == "__main__":
    unittest.main()
