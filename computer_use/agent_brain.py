#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AGI Unified Framework - Agent Brain Module
大模型Agent决策大脑 - 多模型LLM集成

Production-ready implementation for Claw-like intelligent agent system.
Integrates multiple LLM providers for computer use automation.

Author: AGI Framework Team
Version: 1.0.0
Lines: ~1200
"""

import os
import json
import base64
import asyncio
import aiohttp
import requests
from typing import Dict, List, Optional, Union, Callable, Any, AsyncGenerator
from dataclasses import dataclass, field, asdict
from enum import Enum, auto
from abc import ABC, abstractmethod
import logging
from datetime import datetime
import time
import re
from pathlib import Path
import threading
from collections import deque
import hashlib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class LLMProvider(Enum):
    """Supported LLM providers"""
    OPENAI = "openai"
    DEEPSEEK = "deepseek"
    MOONSHOT = "moonshot"  # 月之暗面 Kimi
    ZHIPU = "zhipu"  # 智谱AI
    BAIDU = "baidu"  # 百度文心
    ALIBABA = "alibaba"  # 阿里通义
    ANTHROPIC = "anthropic"  # Claude
    GOOGLE = "google"  # Gemini
    LOCAL = "local"  # Local LLM via Ollama/vLLM


class ActionType(Enum):
    """Agent action types"""
    CLICK = "click"
    DOUBLE_CLICK = "double_click"
    RIGHT_CLICK = "right_click"
    DRAG = "drag"
    SCROLL = "scroll"
    TYPE = "type"
    KEYPRESS = "keypress"
    HOTKEY = "hotkey"
    WAIT = "wait"
    SCREENSHOT = "screenshot"
    FIND_ELEMENT = "find_element"
    LAUNCH_APP = "launch_app"
    CLOSE_APP = "close_app"
    SWITCH_WINDOW = "switch_window"
    COPY = "copy"
    PASTE = "paste"
    TERMINATE = "terminate"


@dataclass
class AgentAction:
    """Represents a single agent action"""
    action_type: ActionType
    params: Dict[str, Any] = field(default_factory=dict)
    description: str = ""
    confidence: float = 1.0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict:
        return {
            "action_type": self.action_type.value,
            "params": self.params,
            "description": self.description,
            "confidence": self.confidence,
            "timestamp": self.timestamp
        }


@dataclass
class ActionResult:
    """Result of executing an action"""
    success: bool
    action: AgentAction
    message: str = ""
    screenshot_path: Optional[str] = None
    data: Dict[str, Any] = field(default_factory=dict)
    execution_time_ms: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class ConversationMessage:
    """Single message in conversation history"""
    role: str  # system, user, assistant, tool
    content: Union[str, List[Dict]]
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentState:
    """Current state of the agent"""
    task: str = ""
    step_count: int = 0
    max_steps: int = 50
    conversation_history: List[ConversationMessage] = field(default_factory=list)
    action_history: List[ActionResult] = field(default_factory=list)
    current_screenshot: Optional[str] = None
    context_variables: Dict[str, Any] = field(default_factory=dict)
    is_running: bool = False
    last_error: Optional[str] = None


class LLMClient(ABC):
    """Abstract base class for LLM clients"""
    
    def __init__(self, api_key: str, base_url: Optional[str] = None, **kwargs):
        self.api_key = api_key
        self.base_url = base_url
        self.config = kwargs
        self.request_count = 0
        self.token_count = 0
        self.last_request_time = 0
        self.rate_limit_delay = 0.1  # seconds between requests
        
    @abstractmethod
    async def chat_completion(
        self,
        messages: List[Dict],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        **kwargs
    ) -> str:
        """Send chat completion request"""
        pass
    
    @abstractmethod
    async def vision_completion(
        self,
        messages: List[Dict],
        image_base64: str,
        model: Optional[str] = None,
        **kwargs
    ) -> str:
        """Send vision completion request with image"""
        pass
    
    def _rate_limit(self):
        """Apply rate limiting"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - time_since_last)
        self.last_request_time = time.time()
    
    def count_tokens(self, text: str) -> int:
        """Estimate token count (rough approximation)"""
        # Rough estimate: 1 token ≈ 4 characters for English, 1 token ≈ 1 character for Chinese
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
        other_chars = len(text) - chinese_chars
        return chinese_chars + other_chars // 4


class OpenAIClient(LLMClient):
    """OpenAI GPT client"""
    
    DEFAULT_MODEL = "gpt-4o"
    VISION_MODEL = "gpt-4o"
    
    def __init__(self, api_key: str, base_url: Optional[str] = None, **kwargs):
        super().__init__(api_key, base_url or "https://api.openai.com/v1", **kwargs)
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
    
    async def chat_completion(
        self,
        messages: List[Dict],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        **kwargs
    ) -> str:
        self._rate_limit()
        model = model or self.DEFAULT_MODEL
        
        url = f"{self.base_url}/chat/completions"
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            **kwargs
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=self.headers, json=payload) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        result = data["choices"][0]["message"]["content"]
                        self.request_count += 1
                        self.token_count += self.count_tokens(result)
                        return result
                    else:
                        error_text = await resp.text()
                        raise Exception(f"OpenAI API error {resp.status}: {error_text}")
        except Exception as e:
            logger.error(f"OpenAI request failed: {e}")
            raise
    
    async def vision_completion(
        self,
        messages: List[Dict],
        image_base64: str,
        model: Optional[str] = None,
        **kwargs
    ) -> str:
        model = model or self.VISION_MODEL
        
        # Add image to the last user message
        vision_messages = messages.copy()
        if vision_messages and vision_messages[-1]["role"] == "user":
            content = vision_messages[-1]["content"]
            if isinstance(content, str):
                vision_messages[-1]["content"] = [
                    {"type": "text", "text": content},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_base64}"
                        }
                    }
                ]
        
        return await self.chat_completion(vision_messages, model=model, **kwargs)


class DeepSeekClient(LLMClient):
    """DeepSeek client"""
    
    DEFAULT_MODEL = "deepseek-chat"
    VISION_MODEL = "deepseek-vision"
    
    def __init__(self, api_key: str, base_url: Optional[str] = None, **kwargs):
        super().__init__(api_key, base_url or "https://api.deepseek.com/v1", **kwargs)
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
    
    async def chat_completion(
        self,
        messages: List[Dict],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        **kwargs
    ) -> str:
        self._rate_limit()
        model = model or self.DEFAULT_MODEL
        
        url = f"{self.base_url}/chat/completions"
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            **kwargs
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=self.headers, json=payload) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        result = data["choices"][0]["message"]["content"]
                        self.request_count += 1
                        self.token_count += self.count_tokens(result)
                        return result
                    else:
                        error_text = await resp.text()
                        raise Exception(f"DeepSeek API error {resp.status}: {error_text}")
        except Exception as e:
            logger.error(f"DeepSeek request failed: {e}")
            raise
    
    async def vision_completion(
        self,
        messages: List[Dict],
        image_base64: str,
        model: Optional[str] = None,
        **kwargs
    ) -> str:
        model = model or self.VISION_MODEL
        
        vision_messages = messages.copy()
        if vision_messages and vision_messages[-1]["role"] == "user":
            content = vision_messages[-1]["content"]
            if isinstance(content, str):
                vision_messages[-1]["content"] = [
                    {"type": "text", "text": content},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_base64}"
                        }
                    }
                ]
        
        return await self.chat_completion(vision_messages, model=model, **kwargs)


class MoonshotClient(LLMClient):
    """Moonshot AI (Kimi) client - 月之暗面"""
    
    DEFAULT_MODEL = "moonshot-v1-8k"
    VISION_MODEL = "moonshot-v1-8k-vision-preview"
    
    def __init__(self, api_key: str, base_url: Optional[str] = None, **kwargs):
        super().__init__(api_key, base_url or "https://api.moonshot.cn/v1", **kwargs)
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
    
    async def chat_completion(
        self,
        messages: List[Dict],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        **kwargs
    ) -> str:
        self._rate_limit()
        model = model or self.DEFAULT_MODEL
        
        url = f"{self.base_url}/chat/completions"
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            **kwargs
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=self.headers, json=payload) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        result = data["choices"][0]["message"]["content"]
                        self.request_count += 1
                        self.token_count += self.count_tokens(result)
                        return result
                    else:
                        error_text = await resp.text()
                        raise Exception(f"Moonshot API error {resp.status}: {error_text}")
        except Exception as e:
            logger.error(f"Moonshot request failed: {e}")
            raise
    
    async def vision_completion(
        self,
        messages: List[Dict],
        image_base64: str,
        model: Optional[str] = None,
        **kwargs
    ) -> str:
        model = model or self.VISION_MODEL
        
        vision_messages = messages.copy()
        if vision_messages and vision_messages[-1]["role"] == "user":
            content = vision_messages[-1]["content"]
            if isinstance(content, str):
                vision_messages[-1]["content"] = [
                    {"type": "text", "text": content},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_base64}"
                        }
                    }
                ]
        
        return await self.chat_completion(vision_messages, model=model, **kwargs)


class ZhipuClient(LLMClient):
    """Zhipu AI (GLM) client - 智谱AI"""
    
    DEFAULT_MODEL = "glm-4"
    VISION_MODEL = "glm-4v"
    
    def __init__(self, api_key: str, base_url: Optional[str] = None, **kwargs):
        super().__init__(api_key, base_url or "https://open.bigmodel.cn/api/paas/v4", **kwargs)
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
    
    async def chat_completion(
        self,
        messages: List[Dict],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        **kwargs
    ) -> str:
        self._rate_limit()
        model = model or self.DEFAULT_MODEL
        
        url = f"{self.base_url}/chat/completions"
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            **kwargs
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=self.headers, json=payload) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        result = data["choices"][0]["message"]["content"]
                        self.request_count += 1
                        self.token_count += self.count_tokens(result)
                        return result
                    else:
                        error_text = await resp.text()
                        raise Exception(f"Zhipu API error {resp.status}: {error_text}")
        except Exception as e:
            logger.error(f"Zhipu request failed: {e}")
            raise
    
    async def vision_completion(
        self,
        messages: List[Dict],
        image_base64: str,
        model: Optional[str] = None,
        **kwargs
    ) -> str:
        model = model or self.VISION_MODEL
        
        vision_messages = messages.copy()
        if vision_messages and vision_messages[-1]["role"] == "user":
            content = vision_messages[-1]["content"]
            if isinstance(content, str):
                vision_messages[-1]["content"] = [
                    {"type": "text", "text": content},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_base64}"
                        }
                    }
                ]
        
        return await self.chat_completion(vision_messages, model=model, **kwargs)


class BaiduClient(LLMClient):
    """Baidu Wenxin client - 百度文心"""
    
    DEFAULT_MODEL = "ernie-bot-4"
    
    def __init__(self, api_key: str, secret_key: str, **kwargs):
        super().__init__(api_key, "https://aip.baidubce.com", **kwargs)
        self.secret_key = secret_key
        self.access_token = None
        self.token_expires = 0
    
    async def _get_access_token(self) -> str:
        """Get Baidu access token"""
        if self.access_token and time.time() < self.token_expires:
            return self.access_token
        
        url = f"{self.base_url}/oauth/2.0/token"
        params = {
            "grant_type": "client_credentials",
            "client_id": self.api_key,
            "client_secret": self.secret_key
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, params=params) as resp:
                data = await resp.json()
                self.access_token = data["access_token"]
                self.token_expires = time.time() + data.get("expires_in", 3600) - 300
                return self.access_token
    
    async def chat_completion(
        self,
        messages: List[Dict],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        **kwargs
    ) -> str:
        self._rate_limit()
        model = model or self.DEFAULT_MODEL
        token = await self._get_access_token()
        
        url = f"{self.base_url}/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/{model}"
        headers = {"Content-Type": "application/json"}
        payload = {
            "messages": messages,
            "temperature": temperature,
            "max_output_tokens": max_tokens,
            **kwargs
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, json=payload, params={"access_token": token}) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        result = data.get("result", "")
                        self.request_count += 1
                        self.token_count += self.count_tokens(result)
                        return result
                    else:
                        error_text = await resp.text()
                        raise Exception(f"Baidu API error {resp.status}: {error_text}")
        except Exception as e:
            logger.error(f"Baidu request failed: {e}")
            raise
    
    async def vision_completion(
        self,
        messages: List[Dict],
        image_base64: str,
        model: Optional[str] = None,
        **kwargs
    ) -> str:
        # Baidu has separate vision API
        token = await self._get_access_token()
        url = f"{self.base_url}/rest/2.0/image-classify/v2/advanced_general"
        # For now, fall back to text-only
        return await self.chat_completion(messages, model=model, **kwargs)


class LLMClientFactory:
    """Factory for creating LLM clients"""
    
    _clients: Dict[LLMProvider, type] = {
        LLMProvider.OPENAI: OpenAIClient,
        LLMProvider.DEEPSEEK: DeepSeekClient,
        LLMProvider.MOONSHOT: MoonshotClient,
        LLMProvider.ZHIPU: ZhipuClient,
        LLMProvider.BAIDU: BaiduClient,
    }
    
    @classmethod
    def create_client(
        cls,
        provider: LLMProvider,
        api_key: str,
        **kwargs
    ) -> LLMClient:
        """Create LLM client for specified provider"""
        client_class = cls._clients.get(provider)
        if not client_class:
            raise ValueError(f"Unsupported provider: {provider}")
        return client_class(api_key, **kwargs)
    
    @classmethod
    def register_client(cls, provider: LLMProvider, client_class: type):
        """Register a new client type"""
        cls._clients[provider] = client_class


class ActionParser:
    """Parse LLM output into structured actions"""
    
    @staticmethod
    def parse_json_action(text: str) -> Optional[AgentAction]:
        """Parse JSON-formatted action from LLM output"""
        try:
            # Try to find JSON in the text
            json_match = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                json_match = re.search(r'\{.*\}', text, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                else:
                    return None
            
            data = json.loads(json_str)
            
            action_type = ActionType(data.get("action", "wait"))
            params = data.get("params", {})
            description = data.get("description", "")
            confidence = data.get("confidence", 1.0)
            
            return AgentAction(
                action_type=action_type,
                params=params,
                description=description,
                confidence=confidence
            )
        except Exception as e:
            logger.warning(f"Failed to parse JSON action: {e}")
            return None
    
    @staticmethod
    def parse_natural_language(text: str) -> Optional[AgentAction]:
        """Parse natural language into action (fallback)"""
        text_lower = text.lower()
        
        # Click patterns
        if any(word in text_lower for word in ["点击", "click", "按下", "press"]):
            # Try to extract coordinates
            coord_match = re.search(r'(\d+)\s*,\s*(\d+)', text)
            if coord_match:
                x, y = int(coord_match.group(1)), int(coord_match.group(2))
                return AgentAction(
                    action_type=ActionType.CLICK,
                    params={"x": x, "y": y},
                    description=text[:100]
                )
        
        # Type patterns
        if any(word in text_lower for word in ["输入", "type", "填写", "enter"]):
            text_match = re.search(r'["\'](.+)["\']', text)
            if text_match:
                return AgentAction(
                    action_type=ActionType.TYPE,
                    params={"text": text_match.group(1)},
                    description=text[:100]
                )
        
        # Wait patterns
        if any(word in text_lower for word in ["等待", "wait", "pause"]):
            seconds_match = re.search(r'(\d+)\s*(秒|s|seconds?)', text_lower)
            seconds = int(seconds_match.group(1)) if seconds_match else 1
            return AgentAction(
                action_type=ActionType.WAIT,
                params={"seconds": seconds},
                description=text[:100]
            )
        
        # Screenshot patterns
        if any(word in text_lower for word in ["截图", "screenshot", "拍照", "capture"]):
            return AgentAction(
                action_type=ActionType.SCREENSHOT,
                description=text[:100]
            )
        
        # Terminate patterns
        if any(word in text_lower for word in ["完成", "结束", "done", "finish", "terminate"]):
            return AgentAction(
                action_type=ActionType.TERMINATE,
                params={"status": "success"},
                description=text[:100]
            )
        
        return None
    
    @classmethod
    def parse(cls, text: str) -> Optional[AgentAction]:
        """Try to parse action from text using multiple methods"""
        # Try JSON first
        action = cls.parse_json_action(text)
        if action:
            return action
        
        # Fall back to natural language
        action = cls.parse_natural_language(text)
        if action:
            return action
        
        # Default: wait
        return AgentAction(
            action_type=ActionType.WAIT,
            params={"seconds": 1},
            description="Could not parse action, waiting",
            confidence=0.5
        )


class AgentBrain:
    """
    Main agent brain that coordinates LLM reasoning and action execution.
    Similar to Claw/Crayfish intelligent agent.
    """
    
    SYSTEM_PROMPT = """You are an intelligent computer automation agent. Your task is to control the computer by analyzing screenshots and generating precise actions.

Available Actions (respond in JSON format):
{
    "action": "click|double_click|right_click|drag|scroll|type|keypress|hotkey|wait|screenshot|find_element|launch_app|close_app|switch_window|copy|paste|terminate",
    "params": {
        // For click/double_click/right_click:
        "x": integer, "y": integer,
        
        // For drag:
        "start_x": integer, "start_y": integer, "end_x": integer, "end_y": integer,
        
        // For scroll:
        "direction": "up|down|left|right", "amount": integer,
        
        // For type:
        "text": string,
        
        // For keypress:
        "key": string (e.g., "enter", "tab", "escape"),
        
        // For hotkey:
        "keys": ["ctrl", "c"] // array of keys to press together,
        
        // For wait:
        "seconds": number,
        
        // For find_element:
        "description": string (e.g., "Chrome icon", "Submit button"),
        
        // For launch_app:
        "app_name": string,
        
        // For terminate:
        "status": "success|failure", "message": string
    },
    "description": "Brief description of what this action does",
    "confidence": 0.95 // confidence score 0-1
}

Guidelines:
1. Always analyze the current screenshot carefully before deciding
2. Use precise coordinates based on screen resolution
3. Prefer keyboard shortcuts over mouse when appropriate
4. Wait for UI to stabilize after actions (1-2 seconds)
5. If an action fails, try an alternative approach
6. Think step by step for complex tasks
7. When task is complete, use terminate action

Response Format:
Always respond with a valid JSON object containing the action details."""

    def __init__(
        self,
        llm_client: LLMClient,
        vision_enabled: bool = True,
        max_steps: int = 50,
        callback: Optional[Callable[[ActionResult], None]] = None
    ):
        self.llm = llm_client
        self.vision_enabled = vision_enabled
        self.max_steps = max_steps
        self.callback = callback
        self.state = AgentState(max_steps=max_steps)
        self.parser = ActionParser()
        self._lock = threading.Lock()
        
        # Initialize conversation with system prompt
        self.state.conversation_history.append(
            ConversationMessage(role="system", content=self.SYSTEM_PROMPT)
        )
    
    async def think(
        self,
        task: str,
        screenshot_base64: Optional[str] = None,
        ocr_text: Optional[str] = ""
    ) -> AgentAction:
        """
        Generate next action based on task and current state.
        This is the core reasoning loop of the agent.
        """
        with self._lock:
            # Build user message
            user_content = f"Task: {task}\n\n"
            
            if ocr_text:
                user_content += f"OCR detected text on screen:\n{ocr_text}\n\n"
            
            user_content += f"Step {self.state.step_count + 1}/{self.state.max_steps}\n"
            
            if self.state.last_error:
                user_content += f"Previous error: {self.state.last_error}\n"
            
            user_content += "What action should I take next? Respond with JSON."
            
            # Add to conversation history
            self.state.conversation_history.append(
                ConversationMessage(role="user", content=user_content)
            )
            
            # Prepare messages for LLM
            messages = [
                {"role": msg.role, "content": msg.content}
                for msg in self.state.conversation_history[-10:]  # Keep last 10 messages
            ]
            
            try:
                # Use vision if enabled and screenshot available
                if self.vision_enabled and screenshot_base64:
                    response = await self.llm.vision_completion(
                        messages=messages,
                        image_base64=screenshot_base64,
                        temperature=0.3  # Lower temperature for more deterministic actions
                    )
                else:
                    response = await self.llm.chat_completion(
                        messages=messages,
                        temperature=0.3
                    )
                
                # Add assistant response to history
                self.state.conversation_history.append(
                    ConversationMessage(role="assistant", content=response)
                )
                
                # Parse action from response
                action = self.parser.parse(response)
                return action
                
            except Exception as e:
                logger.error(f"Think failed: {e}")
                self.state.last_error = str(e)
                return AgentAction(
                    action_type=ActionType.WAIT,
                    params={"seconds": 2},
                    description=f"Error during thinking: {e}",
                    confidence=0.0
                )
    
    async def execute_action(
        self,
        action: AgentAction,
        executor: Optional[Any] = None
    ) -> ActionResult:
        """
        Execute the action using provided executor.
        Executor should have methods matching action types.
        """
        start_time = time.time()
        
        try:
            if executor is None:
                # No executor provided, return simulated result
                result = ActionResult(
                    success=True,
                    action=action,
                    message=f"Simulated execution of {action.action_type.value}",
                    execution_time_ms=(time.time() - start_time) * 1000
                )
            else:
                # Execute via executor
                method_name = f"execute_{action.action_type.value}"
                method = getattr(executor, method_name, None)
                
                if method:
                    execution_result = await method(**action.params)
                    result = ActionResult(
                        success=execution_result.get("success", True),
                        action=action,
                        message=execution_result.get("message", ""),
                        data=execution_result,
                        execution_time_ms=(time.time() - start_time) * 1000
                    )
                else:
                    result = ActionResult(
                        success=False,
                        action=action,
                        message=f"No executor method for {action.action_type.value}",
                        execution_time_ms=(time.time() - start_time) * 1000
                    )
            
            # Update state
            self.state.action_history.append(result)
            self.state.step_count += 1
            
            # Clear error on success
            if result.success:
                self.state.last_error = None
            else:
                self.state.last_error = result.message
            
            # Call callback if provided
            if self.callback:
                self.callback(result)
            
            return result
            
        except Exception as e:
            logger.error(f"Action execution failed: {e}")
            result = ActionResult(
                success=False,
                action=action,
                message=str(e),
                execution_time_ms=(time.time() - start_time) * 1000
            )
            self.state.last_error = str(e)
            return result
    
    async def run_task(
        self,
        task: str,
        executor: Optional[Any] = None,
        get_screenshot: Optional[Callable[[], str]] = None,
        get_ocr: Optional[Callable[[], str]] = None
    ) -> List[ActionResult]:
        """
        Run a complete task autonomously.
        
        Args:
            task: Natural language description of the task
            executor: Object with execute_* methods for actions
            get_screenshot: Function that returns base64 screenshot
            get_ocr: Function that returns OCR text
        
        Returns:
            List of action results
        """
        self.state.task = task
        self.state.is_running = True
        self.state.step_count = 0
        
        logger.info(f"Starting task: {task}")
        
        try:
            while self.state.is_running and self.state.step_count < self.state.max_steps:
                # Get current screenshot and OCR
                screenshot = get_screenshot() if get_screenshot else None
                ocr_text = get_ocr() if get_ocr else ""
                
                # Think about next action
                action = await self.think(task, screenshot, ocr_text)
                
                # Execute action
                result = await self.execute_action(action, executor)
                
                # Check for termination
                if action.action_type == ActionType.TERMINATE:
                    logger.info(f"Task terminated: {result.message}")
                    break
                
                # Small delay between actions
                await asyncio.sleep(0.5)
            
            if self.state.step_count >= self.state.max_steps:
                logger.warning("Max steps reached, stopping task")
        
        except Exception as e:
            logger.error(f"Task execution failed: {e}")
        
        finally:
            self.state.is_running = False
        
        return self.state.action_history
    
    def stop(self):
        """Stop the current task"""
        self.state.is_running = False
    
    def get_state(self) -> Dict:
        """Get current agent state as dictionary"""
        return {
            "task": self.state.task,
            "step_count": self.state.step_count,
            "max_steps": self.state.max_steps,
            "is_running": self.state.is_running,
            "last_error": self.state.last_error,
            "action_count": len(self.state.action_history),
            "conversation_length": len(self.state.conversation_history)
        }
    
    def export_history(self, filepath: str):
        """Export action history to JSON file"""
        history = {
            "task": self.state.task,
            "start_time": self.state.conversation_history[0].timestamp if self.state.conversation_history else None,
            "end_time": datetime.now().isoformat(),
            "actions": [asdict(result) for result in self.state.action_history]
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
        
        logger.info(f"History exported to {filepath}")


class MultiModelAgent:
    """
    Agent that can switch between multiple LLM providers
    for optimal performance and cost.
    """
    
    def __init__(self, clients: Dict[LLMProvider, LLMClient], default_provider: LLMProvider):
        self.clients = clients
        self.default_provider = default_provider
        self.current_provider = default_provider
        self.fallback_chain = [
            LLMProvider.OPENAI,
            LLMProvider.MOONSHOT,
            LLMProvider.DEEPSEEK,
            LLMProvider.ZHIPU
        ]
    
    async def chat_with_fallback(
        self,
        messages: List[Dict],
        preferred_provider: Optional[LLMProvider] = None,
        **kwargs
    ) -> str:
        """Chat with automatic fallback on failure"""
        providers_to_try = [preferred_provider] if preferred_provider else []
        providers_to_try.extend(self.fallback_chain)
        
        for provider in providers_to_try:
            if provider not in self.clients:
                continue
            
            try:
                client = self.clients[provider]
                result = await client.chat_completion(messages, **kwargs)
                self.current_provider = provider
                return result
            except Exception as e:
                logger.warning(f"{provider.value} failed: {e}, trying fallback...")
                continue
        
        raise Exception("All LLM providers failed")


# Example usage and testing
if __name__ == "__main__":
    async def test_agent():
        """Test the agent brain"""
        # Create a mock LLM client for testing
        class MockLLMClient(LLMClient):
            async def chat_completion(self, messages, **kwargs):
                return '{"action": "click", "params": {"x": 100, "y": 200}, "description": "Click on button", "confidence": 0.95}'
            
            async def vision_completion(self, messages, image_base64, **kwargs):
                return await self.chat_completion(messages, **kwargs)
        
        # Create agent
        client = MockLLMClient("mock_key")
        agent = AgentBrain(client, vision_enabled=False)
        
        # Test thinking
        action = await agent.think("Click the submit button")
        print(f"Generated action: {action}")
        
        # Test execution
        result = await agent.execute_action(action)
        print(f"Execution result: {result}")
        
        # Print state
        print(f"Agent state: {agent.get_state()}")
    
    # Run test
    asyncio.run(test_agent())
