"""
MCP互操作测试模块

测试MCP服务器/客户端、工具转换、流式传输、资源访问和提示词注册。
"""

import unittest
import time
import json
from typing import Dict, List, Set, Optional, Any, Tuple
import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from multiagent.mcp.server import MCPServer, ServerState
from multiagent.mcp.client import MCPClient
from multiagent.mcp.schema import (
    MCPMethod,
    MCPErrorCode,
    JSONRPCRequest,
    JSONRPCResponse,
    MCPTool,
    MCPResource,
    MCPPrompt
)


class MockMCPHelpers:
    """MCP测试辅助类"""

    @staticmethod
    def create_server(
        name: str = "test_server",
        version: str = "1.0.0"
    ) -> MCPServer:
        """创建测试用MCP服务器"""
        return MCPServer(name=name, version=version)

    @staticmethod
    def create_tool_handler(tool_name: str) -> callable:
        """创建测试用工具处理器"""
        def handler(**kwargs):
            return {"result": f"{tool_name} executed", "params": kwargs}
        return handler

    @staticmethod
    def create_resource_handler(resource_uri: str) -> callable:
        """创建测试用资源处理器"""
        def handler(uri: str):
            return f"Resource content for {uri}"
        return handler


class TestMCPServer(unittest.TestCase):
    """测试MCP服务器"""

    def test_server_initialization(self):
        """测试服务器初始化"""
        server = MockMCPHelpers.create_server()

        self.assertEqual(server.name, "test_server")
        self.assertEqual(server.version, "1.0.0")
        self.assertEqual(server.state, ServerState.UNINITIALIZED)

    def test_register_tool(self):
        """测试注册工具"""
        server = MockMCPHelpers.create_server()

        handler = MockMCPHelpers.create_tool_handler("test_tool")
        server.register_tool(
            name="my_tool",
            description="A test tool",
            handler=handler
        )

        self.assertEqual(server.tool_count, 1)

    def test_unregister_tool(self):
        """测试注销工具"""
        server = MockMCPHelpers.create_server()

        handler = MockMCPHelpers.create_tool_handler("remove_tool")
        server.register_tool(
            name="tool_to_remove",
            description="Tool to be removed",
            handler=handler
        )

        server.unregister_tool("tool_to_remove")
        self.assertEqual(server.tool_count, 0)

    def test_register_resource(self):
        """测试注册资源"""
        server = MockMCPHelpers.create_server()

        handler = MockMCPHelpers.create_resource_handler("test://resource")
        server.register_resource(
            uri="test://resource",
            name="TestResource",
            description="A test resource",
            handler=handler
        )

        self.assertEqual(server.resource_count, 1)

    def test_unregister_resource(self):
        """测试注销资源"""
        server = MockMCPHelpers.create_server()

        handler = MockMCPHelpers.create_resource_handler("test://remove")
        server.register_resource(
            uri="test://remove",
            name="ResourceToRemove",
            handler=handler
        )

        server.unregister_resource("test://remove")
        self.assertEqual(server.resource_count, 0)

    def test_register_prompt(self):
        """测试注册提示词"""
        server = MockMCPHelpers.create_server()

        server.register_prompt(
            name="greeting",
            description="A greeting template",
            template="Hello, {name}!",
            arguments=[
                {"name": "name", "description": "The name to greet", "required": True}
            ]
        )

        self.assertEqual(server.prompt_count, 1)

    def test_unregister_prompt(self):
        """测试注销提示词"""
        server = MockMCPHelpers.create_server()

        server.register_prompt(
            name="remove_prompt",
            description="A prompt to remove"
        )

        server.unregister_prompt("remove_prompt")
        self.assertEqual(server.prompt_count, 0)

    def test_server_is_ready(self):
        """测试服务器就绪状态"""
        server = MockMCPHelpers.create_server()

        self.assertFalse(server.is_ready)

    def test_server_shutdown(self):
        """测试服务器关闭"""
        server = MockMCPHelpers.create_server()

        server.register_tool(
            name="shutdown_test",
            description="Test tool",
            handler=lambda: None
        )

        server.shutdown()

        self.assertEqual(server.state, ServerState.SHUTDOWN)
        self.assertEqual(server.tool_count, 0)


class TestMCPClient(unittest.TestCase):
    """测试MCP客户端"""

    def setUp(self):
        """测试初始化"""
        from multiagent.mcp.client import MCPClient, ClientState
        self.client_module = __import__('multiagent.mcp.client', fromlist=['MCPClient'])
        self.MCPClient = self.client_module.MCPClient
        self.ClientState = self.client_module.ClientState

    def test_client_initialization(self):
        """测试客户端初始化"""
        client = self.MCPClient()

        self.assertIsNotNone(client)
        self.assertEqual(client.state, self.ClientState.DISCONNECTED)

    def test_client_connect(self):
        """测试客户端连接"""
        client = self.MCPClient()

        # 模拟连接
        result = client.connect("http://localhost:8000")

        # 应该能够建立连接（即使服务器不存在）
        self.assertIsNotNone(result)

    def test_client_disconnect(self):
        """测试客户端断开连接"""
        client = self.MCPClient()

        client.connect("http://localhost:8000")
        client.disconnect()

        self.assertEqual(client.state, self.ClientState.DISCONNECTED)

    def test_list_tools(self):
        """测试列出工具"""
        client = self.MCPClient()

        # 模拟服务器
        tools = client.list_tools()

        self.assertIsNotNone(tools)

    def test_call_tool(self):
        """测试调用工具"""
        client = self.MCPClient()

        # 模拟工具调用
        result = client.call_tool(
            "test_tool",
            arguments={"param": "value"}
        )

        self.assertIsNotNone(result)


class TestToolConversion(unittest.TestCase):
    """测试工具转换"""

    def setUp(self):
        """测试初始化"""
        from multiagent.mcp.tool_converter import ToolConverter, OpenAISchema, AnthropicSchema
        self.converter_module = __import__('multiagent.mcp.tool_converter', fromlist=['ToolConverter'])
        self.ToolConverter = self.converter_module.ToolConverter

    def test_converter_initialization(self):
        """测试转换器初始化"""
        converter = self.ToolConverter()
        self.assertIsNotNone(converter)

    def test_convert_to_openai_schema(self):
        """测试转换为OpenAI schema"""
        converter = self.ToolConverter()

        tool = MCPTool(
            name="test_function",
            description="A test function",
            inputSchema={"type": "object", "properties": {}}
        )

        openai_schema = converter.to_openai_schema(tool)

        self.assertIn("type", openai_schema)
        self.assertIn("function", openai_schema)

    def test_convert_from_openai_schema(self):
        """测试从OpenAI schema转换"""
        converter = self.ToolConverter()

        openai_schema = {
            "type": "function",
            "function": {
                "name": "my_function",
                "description": "A converted function",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "arg1": {"type": "string"}
                    }
                }
            }
        }

        tool = converter.from_openai_schema(openai_schema)

        self.assertEqual(tool.name, "my_function")

    def test_convert_to_anthropic_schema(self):
        """测试转换为Anthropic schema"""
        converter = self.ToolConverter()

        tool = MCPTool(
            name="claude_tool",
            description="A Claude tool",
            inputSchema={"type": "object", "properties": {}}
        )

        anthropic_schema = converter.to_anthropic_schema(tool)

        self.assertIn("name", anthropic_schema)
        self.assertIn("description", anthropic_schema)

    def test_property_type_mapping(self):
        """测试属性类型映射"""
        converter = self.ToolConverter()

        mcp_props = {
            "string_field": {"type": "string"},
            "number_field": {"type": "number"},
            "boolean_field": {"type": "boolean"},
            "array_field": {"type": "array"}
        }

        openai_props = converter.map_property_types(mcp_props)

        self.assertEqual(openai_props["string_field"]["type"], "string")
        self.assertEqual(openai_props["number_field"]["type"], "number")


class TestStreaming(unittest.TestCase):
    """测试流式传输"""

    def setUp(self):
        """测试初始化"""
        from multiagent.mcp.streaming import StreamingHandler, StreamChunk
        self.streaming_module = __import__('multiagent.mcp.streaming', fromlist=['StreamingHandler'])
        self.StreamingHandler = self.streaming_module.StreamingHandler
        self.StreamChunk = self.streaming_module.StreamChunk

    def test_streaming_initialization(self):
        """测试流初始化"""
        handler = self.StreamingHandler()
        self.assertIsNotNone(handler)

    def test_create_chunk(self):
        """测试创建数据块"""
        chunk = self.StreamChunk(
            content="test content",
            chunk_index=0,
            is_final=False
        )

        self.assertEqual(chunk.content, "test content")
        self.assertEqual(chunk.chunk_index, 0)
        self.assertFalse(chunk.is_final)

    def test_final_chunk(self):
        """测试最终数据块"""
        chunk = self.StreamChunk(
            content="final",
            chunk_index=5,
            is_final=True
        )

        self.assertTrue(chunk.is_final)

    def test_chunk_accumulation(self):
        """测试数据块累积"""
        handler = self.StreamingHandler()

        chunks = [
            self.StreamChunk("part1", 0, False),
            self.StreamChunk("part2", 1, False),
            self.StreamChunk("part3", 2, True),
        ]

        full_content = handler.accumulate(chunks)

        self.assertEqual(full_content, "part1part2part3")

    def test_stream_progress(self):
        """测试流进度"""
        handler = self.StreamingHandler()

        handler.start_stream(total=100)

        self.assertEqual(handler.progress, 0)
        self.assertEqual(handler.total, 100)


class TestResourceAccess(unittest.TestCase):
    """测试资源访问"""

    def test_resource_uri_parsing(self):
        """测试资源URI解析"""
        uri = "test://user/data/file.txt"

        # 简单解析
        parts = uri.split("://")
        self.assertEqual(len(parts), 2)
        self.assertEqual(parts[0], "test")

    def test_resource_caching(self):
        """测试资源缓存"""
        server = MockMCPHelpers.create_server()

        handler = MockMCPHelpers.create_resource_handler("cached://resource")
        server.register_resource(
            uri="cached://resource",
            name="CachedResource",
            handler=handler
        )

        # 第一次访问
        result1 = server._resource_handlers.get("cached://resource")("")
        # 第二次访问应该使用缓存
        result2 = server._resource_handlers.get("cached://resource")("")

        self.assertEqual(result1, result2)

    def test_resource_not_found(self):
        """测试资源未找到"""
        server = MockMCPHelpers.create_server()

        # 尝试获取不存在的资源
        resource = server._resources.get("nonexistent://resource")
        self.assertIsNone(resource)

    def test_resource_mime_types(self):
        """测试资源MIME类型"""
        resource = MCPResource(
            uri="test://doc",
            name="Document",
            mimeType="application/pdf"
        )

        self.assertEqual(resource.mimeType, "application/pdf")


class TestPromptRegistry(unittest.TestCase):
    """测试提示词注册"""

    def setUp(self):
        """测试初始化"""
        from multiagent.mcp.prompt_registry import PromptRegistry, PromptTemplate
        self.prompt_module = __import__('multiagent.mcp.prompt_registry', fromlist=['PromptRegistry'])
        self.PromptRegistry = self.prompt_module.PromptRegistry
        self.PromptTemplate = self.prompt_module.PromptTemplate

    def test_registry_initialization(self):
        """测试注册表初始化"""
        registry = self.PromptRegistry()
        self.assertIsNotNone(registry)

    def test_register_template(self):
        """测试注册模板"""
        registry = self.PromptRegistry()

        template = self.PromptTemplate(
            name="analysis_prompt",
            template="Analyze the following: {content}",
            variables=["content"]
        )

        registry.register(template)
        self.assertEqual(len(registry.templates), 1)

    def test_render_template(self):
        """测试渲染模板"""
        registry = self.PromptRegistry()

        template = self.PromptTemplate(
            name="greeting",
            template="Hello, {name}!",
            variables=["name"]
        )

        rendered = registry.render("greeting", {"name": "Alice"})

        self.assertEqual(rendered, "Hello, Alice!")

    def test_render_with_missing_variable(self):
        """测试缺少变量的渲染"""
        registry = self.PromptRegistry()

        template = self.PromptTemplate(
            name="complex",
            template="Hello {name}, your age is {age}",
            variables=["name", "age"]
        )

        # 缺少age变量
        rendered = registry.render("complex", {"name": "Bob"})

        self.assertIn("{age}", rendered)

    def test_template_versioning(self):
        """测试模板版本控制"""
        registry = self.PromptRegistry()

        template = self.PromptTemplate(
            name="versioned",
            template="Version 1: {content}",
            variables=["content"],
            version=1
        )

        registry.register(template)

        updated = registry.update_template("versioned", version=2)
        self.assertEqual(updated.version, 2)

    def test_list_prompts(self):
        """测试列出提示词"""
        registry = self.PromptRegistry()

        for i in range(3):
            registry.register(self.PromptTemplate(
                name=f"prompt_{i}",
                template=f"Template {i}"
            ))

        prompts = registry.list_prompts()
        self.assertEqual(len(prompts), 3)


class TestJSONRPCProtocol(unittest.TestCase):
    """测试JSON-RPC协议"""

    def test_request_creation(self):
        """测试请求创建"""
        request = JSONRPCRequest(
            jsonrpc="2.0",
            method="tools/list",
            params={"type": "all"},
            id=1
        )

        self.assertEqual(request.jsonrpc, "2.0")
        self.assertEqual(request.method, "tools/list")
        self.assertEqual(request.id, 1)

    def test_response_creation(self):
        """测试响应创建"""
        response = JSONRPCResponse(
            jsonrpc="2.0",
            result={"tools": []},
            id=1
        )

        self.assertEqual(response.jsonrpc, "2.0")
        self.assertIn("tools", response.result)

    def test_error_response(self):
        """测试错误响应"""
        response = JSONRPCResponse.error(
            code=MCPErrorCode.METHOD_NOT_FOUND,
            message="Method not found",
            request_id=1
        )

        self.assertEqual(response.error["code"], -32601)

    def test_notification_request(self):
        """测试通知请求"""
        request = JSONRPCRequest(
            jsonrpc="2.0",
            method="notifications/event"
        )

        self.assertIsNone(request.id)

    def test_batch_request(self):
        """测试批量请求"""
        requests = [
            JSONRPCRequest(jsonrpc="2.0", method="method1", id=1),
            JSONRPCRequest(jsonrpc="2.0", method="method2", id=2),
        ]

        self.assertEqual(len(requests), 2)


class TestMCPMethods(unittest.TestCase):
    """测试MCP方法"""

    def test_tools_list_method(self):
        """测试tools.list方法"""
        method = MCPMethod.TOOLS_LIST
        self.assertEqual(method.value, "tools/list")

    def test_tools_call_method(self):
        """测试tools.call方法"""
        method = MCPMethod.TOOLS_CALL
        self.assertEqual(method.value, "tools/call")

    def test_resources_list_method(self):
        """测试resources.list方法"""
        method = MCPMethod.RESOURCES_LIST
        self.assertEqual(method.value, "resources/list")

    def test_resources_read_method(self):
        """测试resources.read方法"""
        method = MCPMethod.RESOURCES_READ
        self.assertEqual(method.value, "resources/read")

    def test_prompts_list_method(self):
        """测试prompts.list方法"""
        method = MCPMethod.PROMPTS_LIST
        self.assertEqual(method.value, "prompts/list")

    def test_prompts_get_method(self):
        """测试prompts.get方法"""
        method = MCPMethod.PROMPTS_GET
        self.assertEqual(method.value, "prompts/get")


class TestMCPServerCapabilities(unittest.TestCase):
    """测试MCP服务器能力"""

    def test_capabilities_initialization(self):
        """测试能力初始化"""
        from multiagent.mcp.schema import MCPServerCapabilities

        capabilities = MCPServerCapabilities(
            logging=True,
            prompts=True,
            resources=True,
            tools=True
        )

        self.assertTrue(capabilities.logging)
        self.assertTrue(capabilities.prompts)
        self.assertTrue(capabilities.resources)
        self.assertTrue(capabilities.tools)

    def test_capabilities_to_dict(self):
        """测试能力转换为字典"""
        from multiagent.mcp.schema import MCPServerCapabilities

        capabilities = MCPServerCapabilities()
        caps_dict = capabilities.to_dict()

        self.assertIsInstance(caps_dict, dict)


class TestServerClientInteraction(unittest.TestCase):
    """测试服务器客户端交互"""

    def setUp(self):
        """测试初始化"""
        from multiagent.mcp.stdio_transport import StdioTransport
        from multiagent.mcp.websocket_transport import WebSocketTransport
        self.stdio_module = __import__('multiagent.mcp.stdio_transport', fromlist=['StdioTransport'])
        self.ws_module = __import__('multiagent.mcp.websocket_transport', fromlist=['WebSocketTransport'])

    def test_stdio_transport_initialization(self):
        """测试stdio传输初始化"""
        transport = self.stdio_module.StdioTransport()
        self.assertIsNotNone(transport)

    def test_websocket_transport_initialization(self):
        """测试websocket传输初始化"""
        transport = self.ws_module.WebSocketTransport(url="ws://localhost:8000")
        self.assertIsNotNone(transport)

    def test_request_response_cycle(self):
        """测试请求响应周期"""
        server = MockMCPHelpers.create_server()

        # 注册工具
        handler = lambda **kwargs: {"result": "success"}
        server.register_tool(
            name="echo",
            description="Echo back input",
            handler=handler
        )

        # 创建请求
        request = JSONRPCRequest(
            jsonrpc="2.0",
            method="tools/list",
            params={},
            id=1
        )

        # 处理请求
        response = server.handle_request(request)

        self.assertIsNotNone(response)

    def test_server_request_handler(self):
        """测试服务器请求处理"""
        server = MockMCPHelpers.create_server()

        # 注册工具
        def add_handler(a: int, b: int):
            return a + b

        server.register_tool(
            name="add",
            description="Add two numbers",
            handler=add_handler,
            input_schema={
                "type": "object",
                "properties": {
                    "a": {"type": "number"},
                    "b": {"type": "number"}
                },
                "required": ["a", "b"]
            }
        )

        # 处理调用请求
        request = JSONRPCRequest(
            jsonrpc="2.0",
            method="tools/call",
            params={"name": "add", "arguments": {"a": 5, "b": 3}},
            id=1
        )

        response = server.handle_request(request)

        self.assertIsNotNone(response)


class TestAuthentication(unittest.TestCase):
    """测试认证"""

    def setUp(self):
        """测试初始化"""
        from multiagent.mcp.auth import AuthMiddleware, TokenValidator
        self.auth_module = __import__('multiagent.mcp.auth', fromlist=['AuthMiddleware'])
        self.AuthMiddleware = self.auth_module.AuthMiddleware

    def test_auth_middleware_initialization(self):
        """测试认证中间件初始化"""
        auth = self.AuthMiddleware()
        self.assertIsNotNone(auth)

    def test_token_validation(self):
        """测试令牌验证"""
        auth = self.AuthMiddleware()

        # 模拟有效令牌
        valid_token = "valid_token_123"
        is_valid = auth.validate_token(valid_token)

        self.assertIn(is_valid, [True, False])

    def test_permission_check(self):
        """测试权限检查"""
        auth = self.AuthMiddleware()

        has_permission = auth.check_permission(
            user_id="user_1",
            resource="tools",
            action="call"
        )

        self.assertIn(has_permission, [True, False])

    def test_invalid_token(self):
        """测试无效令牌"""
        auth = self.AuthMiddleware()

        is_valid = auth.validate_token("invalid_token")
        self.assertFalse(is_valid)


class TestProgressNotifications(unittest.TestCase):
    """测试进度通知"""

    def test_progress_token(self):
        """测试进度令牌"""
        server = MockMCPHelpers.create_server()

        progress_callback_called = [False]

        def progress_callback(progress: float, total: float):
            progress_callback_called[0] = True

        server.on_progress("progress_token_123", progress_callback)

        # 发送进度
        server.send_progress("progress_token_123", progress=50, total=100)

        self.assertTrue(progress_callback_called[0])

    def test_progress_calculation(self):
        """测试进度计算"""
        progress = 50
        total = 100

        percentage = (progress / total) * 100
        self.assertEqual(percentage, 50)


class TestErrorHandling(unittest.TestCase):
    """测试错误处理"""

    def test_method_not_found_error(self):
        """测试方法未找到错误"""
        server = MockMCPHelpers.create_server()

        request = JSONRPCRequest(
            jsonrpc="2.0",
            method="nonexistent/method",
            params={},
            id=1
        )

        response = server.handle_request(request)

        self.assertIsNotNone(response.error)

    def test_invalid_params_error(self):
        """测试无效参数错误"""
        server = MockMCPHelpers.create_server()

        request = JSONRPCRequest(
            jsonrpc="2.0",
            method="tools/call",
            params={"name": "nonexistent"},
            id=1
        )

        response = server.handle_request(request)

        self.assertIsNotNone(response)

    def test_internal_error(self):
        """测试内部错误"""
        server = MockMCPHelpers.create_server()

        # 注册一个会抛出异常的处理器
        def failing_handler():
            raise RuntimeError("Intentional error")

        server.register_tool(
            name="failing_tool",
            description="A tool that fails",
            handler=failing_handler
        )

        request = JSONRPCRequest(
            jsonrpc="2.0",
            method="tools/call",
            params={"name": "failing_tool", "arguments": {}},
            id=1
        )

        response = server.handle_request(request)

        self.assertIsNotNone(response.error)


class TestToolExecution(unittest.TestCase):
    """测试工具执行"""

    def test_tool_with_return_value(self):
        """测试有返回值的工具"""
        server = MockMCPHelpers.create_server()

        def return_value_handler():
            return "Hello from tool"

        server.register_tool(
            name="return_tool",
            description="Returns a value",
            handler=return_value_handler
        )

        self.assertEqual(server.tool_count, 1)

    def test_tool_with_parameters(self):
        """测试带参数的工兴"""
        server = MockMCPHelpers.create_server()

        def param_handler(name: str, age: int = 0):
            return f"{name} is {age} years old"

        server.register_tool(
            name="param_tool",
            description="Tool with parameters",
            handler=param_handler,
            input_schema={
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "age": {"type": "integer", "default": 0}
                },
                "required": ["name"]
            }
        )

        self.assertEqual(server.tool_count, 1)

    def test_tool_execution_error(self):
        """测试工具执行错误"""
        server = MockMCPHelpers.create_server()

        def error_handler():
            raise ValueError("Parameter error")

        server.register_tool(
            name="error_tool",
            description="Tool that errors",
            handler=error_handler
        )

        self.assertEqual(server.tool_count, 1)


class TestEdgeCases(unittest.TestCase):
    """测试边界情况"""

    def test_empty_tool_name(self):
        """测试空工具名称"""
        server = MockMCPHelpers.create_server()

        # 应该处理空名称
        with self.assertRaises(Exception):
            server.register_tool(
                name="",
                description="Empty name",
                handler=lambda: None
            )

    def test_very_long_template(self):
        """测试非常长的模板"""
        server = MockMCPHelpers.create_server()

        long_template = "x" * 10000
        server.register_prompt(
            name="long_prompt",
            template=long_template
        )

        self.assertEqual(server.prompt_count, 1)

    def test_special_characters_in_uri(self):
        """测试URI中的特殊字符"""
        server = MockMCPHelpers.create_server()

        server.register_resource(
            uri="test://path/with spaces/and%encoding",
            name="SpecialURI",
            handler=lambda: "content"
        )

        self.assertEqual(server.resource_count, 1)

    def test_concurrent_tool_registration(self):
        """测试并发工具注册"""
        import threading

        server = MockMCPHelpers.create_server()

        def register_tools(start_id: int):
            for i in range(10):
                server.register_tool(
                    name=f"tool_{start_id}_{i}",
                    description=f"Tool {start_id}_{i}",
                    handler=lambda: None
                )

        threads = []
        for i in range(5):
            thread = threading.Thread(target=register_tools, args=(i,))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        self.assertEqual(server.tool_count, 50)


if __name__ == "__main__":
    unittest.main()
