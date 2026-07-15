"""
Computer Use Module
电脑操作模块

提供屏幕操控、键盘鼠标模拟、OCR识别、自动化操作等功能
"""

import asyncio
import base64
import hashlib
import io
import json
import logging
import os
import tempfile
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, Callable

logger = logging.getLogger(__name__)


class ActionType(Enum):
    """操作类型"""
    CLICK = "click"
    DOUBLE_CLICK = "double_click"
    RIGHT_CLICK = "right_click"
    TYPE = "type"
    PRESS = "press"
    HOTKEY = "hotkey"
    SCROLL = "scroll"
    DRAG = "drag"
    MOVE = "move"
    SCREENSHOT = "screenshot"
    OCR = "ocr"
    FIND_IMAGE = "find_image"
    WAIT = "wait"
    EXECUTE = "execute"


class MouseButton(Enum):
    """鼠标按钮"""
    LEFT = "left"
    RIGHT = "right"
    MIDDLE = "middle"


@dataclass
class Point:
    """坐标点"""
    x: int
    y: int
    
    def to_dict(self) -> Dict[str, int]:
        return {"x": self.x, "y": self.y}


@dataclass
class Region:
    """屏幕区域"""
    x: int
    y: int
    width: int
    height: int
    
    def to_dict(self) -> Dict[str, int]:
        return {
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
        }


@dataclass
class ActionResult:
    """操作结果"""
    success: bool
    action_type: ActionType
    message: str = ""
    data: Optional[Any] = None
    screenshot: Optional[str] = None  # base64
    error: Optional[str] = None
    duration: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "action_type": self.action_type.value,
            "message": self.message,
            "data": self.data,
            "screenshot": self.screenshot[:100] + "..." if self.screenshot and len(self.screenshot) > 100 else self.screenshot,
            "error": self.error,
            "duration": self.duration,
            "metadata": self.metadata,
        }


class ComputerUseEngine:
    """
    电脑操作引擎
    
    功能：
    - 屏幕截图
    - 鼠标操作（点击、移动、拖拽、滚动）
    - 键盘操作（输入、按键、快捷键）
    - OCR文字识别
    - 图像查找
    - 自动化脚本执行
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self._mouse = None
        self._keyboard = None
        self._screen = None
        self._ocr = None
        self._initialized = False
        
        # 安全限制
        self._allowed_actions = set(ActionType)
        self._blocked_keys = set(self.config.get("blocked_keys", ["win", "cmd"]))
        self._action_delay = self.config.get("action_delay", 0.1)
        
    async def _ensure_initialized(self):
        """确保引擎已初始化"""
        if self._initialized:
            return
        
        try:
            import pyautogui
            self._mouse = pyautogui
            self._keyboard = pyautogui
            self._screen = pyautogui
            
            # 设置安全措施
            pyautogui.FAILSAFE = True
            pyautogui.PAUSE = self._action_delay
            
            logger.info("PyAutoGUI initialized")
            
        except ImportError:
            logger.warning("pyautogui not installed, using mock mode")
        
        # 初始化OCR
        await self._init_ocr()
        
        self._initialized = True
        logger.info("Computer use engine initialized")
    
    async def _init_ocr(self):
        """初始化OCR"""
        ocr_backend = self.config.get("ocr_backend", "easyocr")
        
        if ocr_backend == "easyocr":
            try:
                import easyocr
                languages = self.config.get("ocr_languages", ["ch_sim", "en"])
                self._ocr = easyocr.Reader(languages)
                logger.info(f"EasyOCR initialized with languages: {languages}")
            except ImportError:
                logger.warning("easyocr not installed")
        elif ocr_backend == "paddleocr":
            try:
                from paddleocr import PaddleOCR
                self._ocr = PaddleOCR(use_angle_cls=True, lang="ch")
                logger.info("PaddleOCR initialized")
            except ImportError:
                logger.warning("paddleocr not installed")
    
    async def screenshot(
        self,
        region: Optional[Region] = None,
        save_path: Optional[str] = None,
    ) -> ActionResult:
        """
        截取屏幕
        
        Args:
            region: 截取区域，None为全屏
            save_path: 保存路径
        """
        await self._ensure_initialized()
        
        start_time = time.time()
        
        try:
            if self._screen:
                if region:
                    screenshot = self._screen.screenshot(region=(
                        region.x, region.y, region.width, region.height
                    ))
                else:
                    screenshot = self._screen.screenshot()
                
                # 转换为base64
                buffer = io.BytesIO()
                screenshot.save(buffer, format="PNG")
                screenshot_base64 = base64.b64encode(buffer.getvalue()).decode()
                
                # 保存文件
                if save_path:
                    screenshot.save(save_path)
                
                return ActionResult(
                    success=True,
                    action_type=ActionType.SCREENSHOT,
                    message="Screenshot captured successfully",
                    data={
                        "width": screenshot.width,
                        "height": screenshot.height,
                    },
                    screenshot=screenshot_base64,
                    duration=time.time() - start_time,
                )
            else:
                # 模拟模式
                return ActionResult(
                    success=True,
                    action_type=ActionType.SCREENSHOT,
                    message="[Mock] Screenshot captured",
                    data={"width": 1920, "height": 1080},
                    duration=time.time() - start_time,
                )
                
        except Exception as e:
            logger.error(f"Screenshot failed: {e}")
            return ActionResult(
                success=False,
                action_type=ActionType.SCREENSHOT,
                error=str(e),
                duration=time.time() - start_time,
            )
    
    async def click(
        self,
        x: int,
        y: int,
        button: MouseButton = MouseButton.LEFT,
        clicks: int = 1,
    ) -> ActionResult:
        """
        点击屏幕
        
        Args:
            x: X坐标
            y: Y坐标
            button: 鼠标按钮
            clicks: 点击次数
        """
        await self._ensure_initialized()
        
        start_time = time.time()
        
        try:
            if self._mouse:
                self._mouse.click(x, y, clicks=clicks, button=button.value)
                
                return ActionResult(
                    success=True,
                    action_type=ActionType.CLICK if clicks == 1 else ActionType.DOUBLE_CLICK,
                    message=f"Clicked at ({x}, {y}) with {button.value} button",
                    data={"x": x, "y": y, "button": button.value, "clicks": clicks},
                    duration=time.time() - start_time,
                )
            else:
                return ActionResult(
                    success=True,
                    action_type=ActionType.CLICK,
                    message=f"[Mock] Clicked at ({x}, {y})",
                    data={"x": x, "y": y},
                    duration=time.time() - start_time,
                )
                
        except Exception as e:
            logger.error(f"Click failed: {e}")
            return ActionResult(
                success=False,
                action_type=ActionType.CLICK,
                error=str(e),
                duration=time.time() - start_time,
            )
    
    async def move(
        self,
        x: int,
        y: int,
        duration: float = 0.5,
    ) -> ActionResult:
        """
        移动鼠标
        
        Args:
            x: 目标X坐标
            y: 目标Y坐标
            duration: 移动时间
        """
        await self._ensure_initialized()
        
        start_time = time.time()
        
        try:
            if self._mouse:
                self._mouse.moveTo(x, y, duration=duration)
                
                return ActionResult(
                    success=True,
                    action_type=ActionType.MOVE,
                    message=f"Moved to ({x}, {y})",
                    data={"x": x, "y": y, "duration": duration},
                    duration=time.time() - start_time,
                )
            else:
                return ActionResult(
                    success=True,
                    action_type=ActionType.MOVE,
                    message=f"[Mock] Moved to ({x}, {y})",
                    duration=time.time() - start_time,
                )
                
        except Exception as e:
            logger.error(f"Move failed: {e}")
            return ActionResult(
                success=False,
                action_type=ActionType.MOVE,
                error=str(e),
                duration=time.time() - start_time,
            )
    
    async def drag(
        self,
        start_x: int,
        start_y: int,
        end_x: int,
        end_y: int,
        duration: float = 1.0,
    ) -> ActionResult:
        """
        拖拽操作
        
        Args:
            start_x: 起始X坐标
            start_y: 起始Y坐标
            end_x: 结束X坐标
            end_y: 结束Y坐标
            duration: 拖拽时间
        """
        await self._ensure_initialized()
        
        start_time = time.time()
        
        try:
            if self._mouse:
                self._mouse.dragTo(start_x, start_y, end_x, end_y, duration=duration)
                
                return ActionResult(
                    success=True,
                    action_type=ActionType.DRAG,
                    message=f"Dragged from ({start_x}, {start_y}) to ({end_x}, {end_y})",
                    data={
                        "start": {"x": start_x, "y": start_y},
                        "end": {"x": end_x, "y": end_y},
                        "duration": duration,
                    },
                    duration=time.time() - start_time,
                )
            else:
                return ActionResult(
                    success=True,
                    action_type=ActionType.DRAG,
                    message=f"[Mock] Dragged from ({start_x}, {start_y}) to ({end_x}, {end_y})",
                    duration=time.time() - start_time,
                )
                
        except Exception as e:
            logger.error(f"Drag failed: {e}")
            return ActionResult(
                success=False,
                action_type=ActionType.DRAG,
                error=str(e),
                duration=time.time() - start_time,
            )
    
    async def scroll(
        self,
        amount: int,
        x: Optional[int] = None,
        y: Optional[int] = None,
    ) -> ActionResult:
        """
        滚动操作
        
        Args:
            amount: 滚动量（正数向上，负数向下）
            x: X坐标（可选）
            y: Y坐标（可选）
        """
        await self._ensure_initialized()
        
        start_time = time.time()
        
        try:
            if self._mouse:
                if x is not None and y is not None:
                    self._mouse.scroll(amount, x, y)
                else:
                    self._mouse.scroll(amount)
                
                return ActionResult(
                    success=True,
                    action_type=ActionType.SCROLL,
                    message=f"Scrolled by {amount}",
                    data={"amount": amount, "x": x, "y": y},
                    duration=time.time() - start_time,
                )
            else:
                return ActionResult(
                    success=True,
                    action_type=ActionType.SCROLL,
                    message=f"[Mock] Scrolled by {amount}",
                    duration=time.time() - start_time,
                )
                
        except Exception as e:
            logger.error(f"Scroll failed: {e}")
            return ActionResult(
                success=False,
                action_type=ActionType.SCROLL,
                error=str(e),
                duration=time.time() - start_time,
            )
    
    async def type_text(
        self,
        text: str,
        interval: float = 0.05,
    ) -> ActionResult:
        """
        输入文本
        
        Args:
            text: 要输入的文本
            interval: 字符间隔
        """
        await self._ensure_initialized()
        
        start_time = time.time()
        
        try:
            if self._keyboard:
                self._keyboard.typewrite(text, interval=interval)
                
                return ActionResult(
                    success=True,
                    action_type=ActionType.TYPE,
                    message=f"Typed: {text[:50]}{'...' if len(text) > 50 else ''}",
                    data={"text": text, "interval": interval},
                    duration=time.time() - start_time,
                )
            else:
                return ActionResult(
                    success=True,
                    action_type=ActionType.TYPE,
                    message=f"[Mock] Typed: {text[:50]}",
                    duration=time.time() - start_time,
                )
                
        except Exception as e:
            logger.error(f"Type failed: {e}")
            return ActionResult(
                success=False,
                action_type=ActionType.TYPE,
                error=str(e),
                duration=time.time() - start_time,
            )
    
    async def press(
        self,
        key: str,
        presses: int = 1,
    ) -> ActionResult:
        """
        按键
        
        Args:
            key: 按键名称
            presses: 按键次数
        """
        await self._ensure_initialized()
        
        if key.lower() in self._blocked_keys:
            return ActionResult(
                success=False,
                action_type=ActionType.PRESS,
                error=f"Key '{key}' is blocked for safety",
            )
        
        start_time = time.time()
        
        try:
            if self._keyboard:
                self._keyboard.press(key, presses=presses)
                
                return ActionResult(
                    success=True,
                    action_type=ActionType.PRESS,
                    message=f"Pressed: {key}",
                    data={"key": key, "presses": presses},
                    duration=time.time() - start_time,
                )
            else:
                return ActionResult(
                    success=True,
                    action_type=ActionType.PRESS,
                    message=f"[Mock] Pressed: {key}",
                    duration=time.time() - start_time,
                )
                
        except Exception as e:
            logger.error(f"Press failed: {e}")
            return ActionResult(
                success=False,
                action_type=ActionType.PRESS,
                error=str(e),
                duration=time.time() - start_time,
            )
    
    async def hotkey(
        self,
        *keys: str,
    ) -> ActionResult:
        """
        快捷键
        
        Args:
            keys: 按键组合
        """
        await self._ensure_initialized()
        
        # 检查被阻止的按键
        for key in keys:
            if key.lower() in self._blocked_keys:
                return ActionResult(
                    success=False,
                    action_type=ActionType.HOTKEY,
                    error=f"Key '{key}' is blocked for safety",
                )
        
        start_time = time.time()
        
        try:
            if self._keyboard:
                self._keyboard.hotkey(*keys)
                
                return ActionResult(
                    success=True,
                    action_type=ActionType.HOTKEY,
                    message=f"Hotkey: {'+'.join(keys)}",
                    data={"keys": list(keys)},
                    duration=time.time() - start_time,
                )
            else:
                return ActionResult(
                    success=True,
                    action_type=ActionType.HOTKEY,
                    message=f"[Mock] Hotkey: {'+'.join(keys)}",
                    duration=time.time() - start_time,
                )
                
        except Exception as e:
            logger.error(f"Hotkey failed: {e}")
            return ActionResult(
                success=False,
                action_type=ActionType.HOTKEY,
                error=str(e),
                duration=time.time() - start_time,
            )
    
    async def ocr(
        self,
        image: Optional[Union[str, bytes]] = None,
        region: Optional[Region] = None,
        languages: Optional[List[str]] = None,
    ) -> ActionResult:
        """
        OCR文字识别
        
        Args:
            image: 图像路径或数据，None则截屏
            region: 截屏区域
            languages: 语言列表
        """
        await self._ensure_initialized()
        
        start_time = time.time()
        
        try:
            # 获取图像
            if image is None:
                # 截屏
                screenshot_result = await self.screenshot(region)
                if not screenshot_result.success:
                    return screenshot_result
                
                image_data = base64.b64decode(screenshot_result.screenshot)
            elif isinstance(image, str):
                with open(image, "rb") as f:
                    image_data = f.read()
            else:
                image_data = image
            
            # OCR识别
            if self._ocr:
                import numpy as np
                from PIL import Image
                
                # 转换图像
                pil_image = Image.open(io.BytesIO(image_data))
                np_image = np.array(pil_image)
                
                # 识别
                results = self._ocr.readtext(np_image)
                
                # 整理结果
                text_results = []
                for detection in results:
                    bbox, text, confidence = detection
                    text_results.append({
                        "text": text,
                        "confidence": float(confidence),
                        "bbox": [[int(p[0]), int(p[1])] for p in bbox],
                    })
                
                # 合并文本
                full_text = " ".join([r["text"] for r in text_results])
                
                return ActionResult(
                    success=True,
                    action_type=ActionType.OCR,
                    message=f"OCR recognized {len(text_results)} text regions",
                    data={
                        "text": full_text,
                        "regions": text_results,
                    },
                    duration=time.time() - start_time,
                )
            else:
                return ActionResult(
                    success=False,
                    action_type=ActionType.OCR,
                    error="OCR engine not initialized",
                    duration=time.time() - start_time,
                )
                
        except Exception as e:
            logger.error(f"OCR failed: {e}")
            return ActionResult(
                success=False,
                action_type=ActionType.OCR,
                error=str(e),
                duration=time.time() - start_time,
            )
    
    async def find_image(
        self,
        template: Union[str, bytes],
        region: Optional[Region] = None,
        confidence: float = 0.9,
    ) -> ActionResult:
        """
        在屏幕上查找图像
        
        Args:
            template: 模板图像
            region: 搜索区域
            confidence: 匹配置信度
        """
        await self._ensure_initialized()
        
        start_time = time.time()
        
        try:
            if self._screen:
                # 保存模板图像
                if isinstance(template, bytes):
                    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
                        f.write(template)
                        template_path = f.name
                else:
                    template_path = template
                
                # 查找图像
                if region:
                    location = self._screen.locateOnScreen(
                        template_path,
                        region=(region.x, region.y, region.width, region.height),
                        confidence=confidence,
                    )
                else:
                    location = self._screen.locateOnScreen(
                        template_path,
                        confidence=confidence,
                    )
                
                # 清理临时文件
                if isinstance(template, bytes):
                    os.unlink(template_path)
                
                if location:
                    center = self._screen.center(location)
                    
                    return ActionResult(
                        success=True,
                        action_type=ActionType.FIND_IMAGE,
                        message=f"Image found at ({center.x}, {center.y})",
                        data={
                            "x": center.x,
                            "y": center.y,
                            "width": location.width,
                            "height": location.height,
                        },
                        duration=time.time() - start_time,
                    )
                else:
                    return ActionResult(
                        success=False,
                        action_type=ActionType.FIND_IMAGE,
                        message="Image not found on screen",
                        duration=time.time() - start_time,
                    )
            else:
                return ActionResult(
                    success=True,
                    action_type=ActionType.FIND_IMAGE,
                    message="[Mock] Image found at (100, 100)",
                    data={"x": 100, "y": 100},
                    duration=time.time() - start_time,
                )
                
        except Exception as e:
            logger.error(f"Find image failed: {e}")
            return ActionResult(
                success=False,
                action_type=ActionType.FIND_IMAGE,
                error=str(e),
                duration=time.time() - start_time,
            )
    
    async def wait(
        self,
        seconds: float,
    ) -> ActionResult:
        """等待"""
        start_time = time.time()
        
        await asyncio.sleep(seconds)
        
        return ActionResult(
            success=True,
            action_type=ActionType.WAIT,
            message=f"Waited {seconds} seconds",
            data={"duration": seconds},
            duration=time.time() - start_time,
        )
    
    async def execute_script(
        self,
        script: str,
        language: str = "python",
    ) -> ActionResult:
        """
        执行脚本
        
        Args:
            script: 脚本内容
            language: 脚本语言
        """
        await self._ensure_initialized()
        
        start_time = time.time()
        
        try:
            if language == "python":
                # 安全执行Python脚本
                allowed_globals = {
                    "__builtins__": {
                        "print": print,
                        "len": len,
                        "range": range,
                        "str": str,
                        "int": int,
                        "float": float,
                        "list": list,
                        "dict": dict,
                        "True": True,
                        "False": False,
                        "None": None,
                    }
                }
                
                # 执行脚本
                output = []
                def capture_print(*args):
                    output.append(" ".join(str(a) for a in args))
                
                allowed_globals["__builtins__"]["print"] = capture_print
                
                exec(script, allowed_globals)
                
                return ActionResult(
                    success=True,
                    action_type=ActionType.EXECUTE,
                    message="Script executed successfully",
                    data={"output": "\n".join(output)},
                    duration=time.time() - start_time,
                )
            else:
                return ActionResult(
                    success=False,
                    action_type=ActionType.EXECUTE,
                    error=f"Unsupported language: {language}",
                    duration=time.time() - start_time,
                )
                
        except Exception as e:
            logger.error(f"Script execution failed: {e}")
            return ActionResult(
                success=False,
                action_type=ActionType.EXECUTE,
                error=str(e),
                duration=time.time() - start_time,
            )
    
    async def get_screen_size(self) -> Tuple[int, int]:
        """获取屏幕尺寸"""
        await self._ensure_initialized()
        
        if self._screen:
            return self._screen.size()
        return (1920, 1080)
    
    async def get_mouse_position(self) -> Point:
        """获取鼠标位置"""
        await self._ensure_initialized()
        
        if self._mouse:
            x, y = self._mouse.position()
            return Point(x=x, y=y)
        return Point(x=0, y=0)


# 全局实例
_computer_engine: Optional[ComputerUseEngine] = None


def get_computer_engine() -> ComputerUseEngine:
    """获取全局电脑操作引擎"""
    global _computer_engine
    if _computer_engine is None:
        _computer_engine = ComputerUseEngine()
    return _computer_engine


async def init_computer_engine(config: Optional[Dict[str, Any]] = None):
    """初始化全局电脑操作引擎"""
    global _computer_engine
    _computer_engine = ComputerUseEngine(config)
    await _computer_engine._ensure_initialized()
    return _computer_engine
