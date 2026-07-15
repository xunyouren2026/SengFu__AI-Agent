"""
Computer Use API Routes
电脑操作API路由

提供屏幕操控、键盘鼠标模拟、OCR识别等功能
"""

import asyncio
import base64
import logging
import os
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, UploadFile
from pydantic import BaseModel

# 导入核心模块
from core.multimodal import (
    ComputerUseEngine,
    ActionType,
    MouseButton,
    Point,
    Region,
    ActionResult,
    get_computer_engine,
    init_computer_engine,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/computer", tags=["Computer Use"])


# ============== 请求模型 ==============

class ClickRequest(BaseModel):
    """点击请求"""
    x: int
    y: int
    button: str = "left"
    clicks: int = 1


class MoveRequest(BaseModel):
    """移动请求"""
    x: int
    y: int
    duration: float = 0.5


class DragRequest(BaseModel):
    """拖拽请求"""
    start_x: int
    start_y: int
    end_x: int
    end_y: int
    duration: float = 1.0


class ScrollRequest(BaseModel):
    """滚动请求"""
    amount: int
    x: Optional[int] = None
    y: Optional[int] = None


class TypeRequest(BaseModel):
    """输入文本请求"""
    text: str
    interval: float = 0.05


class PressRequest(BaseModel):
    """按键请求"""
    key: str
    presses: int = 1


class HotkeyRequest(BaseModel):
    """快捷键请求"""
    keys: List[str]


class ScreenshotRequest(BaseModel):
    """截图请求"""
    region: Optional[Dict[str, int]] = None
    save_path: Optional[str] = None


class OCRRequest(BaseModel):
    """OCR请求"""
    region: Optional[Dict[str, int]] = None
    languages: Optional[List[str]] = None


class FindImageRequest(BaseModel):
    """查找图像请求"""
    confidence: float = 0.9
    region: Optional[Dict[str, int]] = None


class ExecuteScriptRequest(BaseModel):
    """执行脚本请求"""
    script: str
    language: str = "python"


class AutomationSequenceRequest(BaseModel):
    """自动化序列请求"""
    name: str
    actions: List[Dict[str, Any]]
    repeat: int = 1
    delay: float = 0.5


# ============== 屏幕操作 API ==============

@router.get("/screen/size", summary="Get screen size")
async def get_screen_size():
    """获取屏幕尺寸"""
    try:
        engine = get_computer_engine()
        width, height = await engine.get_screen_size()
        
        return {
            "width": width,
            "height": height,
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/mouse/position", summary="Get mouse position")
async def get_mouse_position():
    """获取鼠标位置"""
    try:
        engine = get_computer_engine()
        position = await engine.get_mouse_position()
        
        return position.to_dict()
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/screenshot", summary="Take screenshot")
async def take_screenshot(request: ScreenshotRequest = None):
    """
    截取屏幕
    
    - 不传region参数则截取全屏
    - 返回base64编码的PNG图像
    """
    try:
        engine = get_computer_engine()
        
        region = None
        if request and request.region:
            region = Region(**request.region)
        
        result = await engine.screenshot(region, request.save_path if request else None)
        
        return result.to_dict()
        
    except Exception as e:
        logger.error(f"Screenshot failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/ocr", summary="OCR text recognition")
async def ocr_recognition(request: OCRRequest = None, file: UploadFile = None):
    """
    OCR文字识别
    
    - 可以上传图像文件
    - 或者截取屏幕区域
    - 支持中英文识别
    """
    try:
        engine = get_computer_engine()
        
        region = None
        if request and request.region:
            region = Region(**request.region)
        
        image_data = None
        if file:
            image_data = await file.read()
        
        result = await engine.ocr(
            image=image_data,
            region=region,
            languages=request.languages if request else None,
        )
        
        return result.to_dict()
        
    except Exception as e:
        logger.error(f"OCR failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============== 鼠标操作 API ==============

@router.post("/mouse/click", summary="Mouse click")
async def mouse_click(request: ClickRequest):
    """
    鼠标点击
    
    - 支持左键、右键、中键
    - 支持单击、双击
    """
    try:
        engine = get_computer_engine()
        
        result = await engine.click(
            x=request.x,
            y=request.y,
            button=MouseButton(request.button),
            clicks=request.clicks,
        )
        
        return result.to_dict()
        
    except Exception as e:
        logger.error(f"Click failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/mouse/move", summary="Mouse move")
async def mouse_move(request: MoveRequest):
    """
    移动鼠标
    
    - duration参数控制移动动画时间
    """
    try:
        engine = get_computer_engine()
        
        result = await engine.move(
            x=request.x,
            y=request.y,
            duration=request.duration,
        )
        
        return result.to_dict()
        
    except Exception as e:
        logger.error(f"Move failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/mouse/drag", summary="Mouse drag")
async def mouse_drag(request: DragRequest):
    """
    鼠标拖拽
    
    - 从起点拖拽到终点
    """
    try:
        engine = get_computer_engine()
        
        result = await engine.drag(
            start_x=request.start_x,
            start_y=request.start_y,
            end_x=request.end_x,
            end_y=request.end_y,
            duration=request.duration,
        )
        
        return result.to_dict()
        
    except Exception as e:
        logger.error(f"Drag failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/mouse/scroll", summary="Mouse scroll")
async def mouse_scroll(request: ScrollRequest):
    """
    鼠标滚动
    
    - amount正数向上滚动，负数向下滚动
    """
    try:
        engine = get_computer_engine()
        
        result = await engine.scroll(
            amount=request.amount,
            x=request.x,
            y=request.y,
        )
        
        return result.to_dict()
        
    except Exception as e:
        logger.error(f"Scroll failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============== 键盘操作 API ==============

@router.post("/keyboard/type", summary="Type text")
async def keyboard_type(request: TypeRequest):
    """
    输入文本
    
    - interval参数控制字符输入间隔
    - 支持中英文输入
    """
    try:
        engine = get_computer_engine()
        
        result = await engine.type_text(
            text=request.text,
            interval=request.interval,
        )
        
        return result.to_dict()
        
    except Exception as e:
        logger.error(f"Type failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/keyboard/press", summary="Press key")
async def keyboard_press(request: PressRequest):
    """
    按键
    
    - 支持所有键盘按键
    - 某些系统按键可能被阻止
    """
    try:
        engine = get_computer_engine()
        
        result = await engine.press(
            key=request.key,
            presses=request.presses,
        )
        
        return result.to_dict()
        
    except Exception as e:
        logger.error(f"Press failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/keyboard/hotkey", summary="Press hotkey")
async def keyboard_hotkey(request: HotkeyRequest):
    """
    快捷键
    
    - 支持组合键，如 Ctrl+C, Alt+Tab
    - 某些系统快捷键可能被阻止
    """
    try:
        engine = get_computer_engine()
        
        result = await engine.hotkey(*request.keys)
        
        return result.to_dict()
        
    except Exception as e:
        logger.error(f"Hotkey failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============== 图像查找 API ==============

@router.post("/find-image", summary="Find image on screen")
async def find_image(
    file: UploadFile = File(...),
    confidence: float = 0.9,
    region_x: Optional[int] = None,
    region_y: Optional[int] = None,
    region_width: Optional[int] = None,
    region_height: Optional[int] = None,
):
    """
    在屏幕上查找图像
    
    - 上传模板图像
    - 返回匹配位置
    - confidence参数控制匹配置信度阈值
    """
    try:
        engine = get_computer_engine()
        
        # 读取模板图像
        template_data = await file.read()
        
        # 构建区域
        region = None
        if all(v is not None for v in [region_x, region_y, region_width, region_height]):
            region = Region(
                x=region_x,
                y=region_y,
                width=region_width,
                height=region_height,
            )
        
        result = await engine.find_image(
            template=template_data,
            region=region,
            confidence=confidence,
        )
        
        return result.to_dict()
        
    except Exception as e:
        logger.error(f"Find image failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============== 自动化 API ==============

@router.post("/automation/sequence", summary="Execute automation sequence")
async def execute_automation_sequence(request: AutomationSequenceRequest):
    """
    执行自动化序列
    
    - 按顺序执行多个操作
    - 支持重复执行
    - 支持操作间延迟
    """
    try:
        engine = get_computer_engine()
        
        results = []
        
        for _ in range(request.repeat):
            for action in request.actions:
                action_type = ActionType(action.get("type"))
                
                if action_type == ActionType.CLICK:
                    result = await engine.click(
                        x=action.get("x", 0),
                        y=action.get("y", 0),
                        button=MouseButton(action.get("button", "left")),
                        clicks=action.get("clicks", 1),
                    )
                elif action_type == ActionType.MOVE:
                    result = await engine.move(
                        x=action.get("x", 0),
                        y=action.get("y", 0),
                        duration=action.get("duration", 0.5),
                    )
                elif action_type == ActionType.TYPE:
                    result = await engine.type_text(
                        text=action.get("text", ""),
                        interval=action.get("interval", 0.05),
                    )
                elif action_type == ActionType.PRESS:
                    result = await engine.press(
                        key=action.get("key", ""),
                        presses=action.get("presses", 1),
                    )
                elif action_type == ActionType.HOTKEY:
                    result = await engine.hotkey(*action.get("keys", []))
                elif action_type == ActionType.SCROLL:
                    result = await engine.scroll(
                        amount=action.get("amount", 0),
                    )
                elif action_type == ActionType.WAIT:
                    await engine.wait(action.get("seconds", 1.0))
                    result = ActionResult(
                        success=True,
                        action_type=ActionType.WAIT,
                        message=f"Waited {action.get('seconds', 1.0)} seconds",
                    )
                elif action_type == ActionType.SCREENSHOT:
                    result = await engine.screenshot()
                else:
                    result = ActionResult(
                        success=False,
                        action_type=action_type,
                        error=f"Unknown action type: {action_type}",
                    )
                
                results.append(result.to_dict())
                
                # 操作间延迟
                if request.delay > 0:
                    await asyncio.sleep(request.delay)
        
        return {
            "success": True,
            "name": request.name,
            "total_actions": len(results),
            "results": results,
        }
        
    except Exception as e:
        logger.error(f"Automation sequence failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/automation/script", summary="Execute script")
async def execute_script(request: ExecuteScriptRequest):
    """
    执行脚本
    
    - 支持Python脚本
    - 安全沙箱环境
    - 受限的内置函数
    """
    try:
        engine = get_computer_engine()
        
        result = await engine.execute_script(
            script=request.script,
            language=request.language,
        )
        
        return result.to_dict()
        
    except Exception as e:
        logger.error(f"Script execution failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============== AI辅助操作 API ==============

@router.post("/ai/understand-screen", summary="AI understand screen")
async def ai_understand_screen(
    prompt: str = "描述屏幕上的内容",
    region: Optional[Dict[str, int]] = None,
):
    """
    AI理解屏幕内容
    
    - 截取屏幕并使用AI分析
    - 支持指定区域
    """
    try:
        engine = get_computer_engine()
        
        # 截图
        region_obj = Region(**region) if region else None
        screenshot_result = await engine.screenshot(region_obj)
        
        if not screenshot_result.success:
            raise HTTPException(status_code=500, detail=screenshot_result.error)
        
        # 使用AI分析
        # TODO: 集成视觉模型
        
        return {
            "success": True,
            "prompt": prompt,
            "screenshot": screenshot_result.screenshot[:100] + "...",
            "analysis": "AI analysis not implemented yet",
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/ai/execute-command", summary="AI execute natural language command")
async def ai_execute_command(command: str):
    """
    AI执行自然语言命令
    
    - 解析自然语言命令
    - 自动转换为操作序列
    - 执行操作
    """
    try:
        # TODO: 实现AI命令解析和执行
        
        return {
            "success": True,
            "command": command,
            "interpretation": "Command interpretation not implemented yet",
            "actions": [],
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# 导出路由
__all__ = ["router"]
