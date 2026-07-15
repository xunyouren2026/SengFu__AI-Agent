#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AGI Unified Framework - Computer Use Module
计算机自动化控制模块 - 小龙虾智能体系统

This module provides production-ready computer automation capabilities:
- Real GUI control (mouse, keyboard, clipboard)
- Screen capture and OCR
- Multi-model LLM integration for agent decision making
- Cross-platform window management

Usage:
    from computer_use import AgentBrain, RealMouseController, RealScreenCapture
    
    # Create agent with LLM
    client = LLMClientFactory.create_client(LLMProvider.OPENAI, api_key)
    agent = AgentBrain(client)
    
    # Run task
    results = await agent.run_task(
        task="Open Chrome and search for AGI",
        executor=RealInputExecutor(),
        get_screenshot=screen_capture.capture_to_base64
    )

Author: AGI Framework Team
Version: 1.0.0
"""

from .real_input import (
    RealMouseController,
    RealKeyboardController,
    RealClipboardManager,
    InputExecutor,
    HumanBehaviorSimulator
)

from .real_screen import (
    RealScreenCapture,
    RealOCR,
    VisionAgent,
    ImagePreprocessor
)

from .real_window import (
    WindowManager,
    WindowManagerFactory,
    WindowAutomation,
    WindowInfo,
    WindowState,
    Platform
)

from .agent_brain import (
    AgentBrain,
    AgentAction,
    ActionResult,
    ActionType,
    LLMClient,
    LLMClientFactory,
    LLMProvider,
    OpenAIClient,
    DeepSeekClient,
    MoonshotClient,
    ZhipuClient,
    BaiduClient,
    MultiModelAgent,
    ActionParser
)

__version__ = "1.0.0"
__author__ = "AGI Framework Team"

__all__ = [
    # Input Control
    "RealMouseController",
    "RealKeyboardController", 
    "RealClipboardManager",
    "InputExecutor",
    "HumanBehaviorSimulator",
    
    # Screen & Vision
    "RealScreenCapture",
    "RealOCR",
    "VisionAgent",
    "ImagePreprocessor",
    
    # Window Management
    "WindowManager",
    "WindowManagerFactory",
    "WindowAutomation",
    "WindowInfo",
    "WindowState",
    "Platform",
    
    # Agent Brain
    "AgentBrain",
    "AgentAction",
    "ActionResult",
    "ActionType",
    "LLMClient",
    "LLMClientFactory",
    "LLMProvider",
    "OpenAIClient",
    "DeepSeekClient",
    "MoonshotClient",
    "ZhipuClient",
    "BaiduClient",
    "MultiModelAgent",
    "ActionParser"
]
