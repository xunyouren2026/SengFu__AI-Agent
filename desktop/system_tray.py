#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AGI Unified Framework - System Tray Application
系统托盘应用 - 桌面常驻智能体

Production-ready system tray application with:
- Tray icon with context menu
- Quick task dialog
- Settings panel
- Web dashboard launcher
- Auto-start support
- Status monitoring

Author: AGI Framework Team
Version: 1.0.0
"""

import os
import sys
import json
import time
import signal
import logging
import threading
import subprocess
from pathlib import Path
from typing import Optional, Callable, Dict, List, Any
from dataclasses import dataclass, field
from enum import Enum, auto
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(os.path.expanduser("~"), ".agi_framework", "tray.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Constants
APP_NAME = "AGI 智能体"
APP_VERSION = "1.0.0"
CONFIG_DIR = os.path.join(os.path.expanduser("~"), ".agi_framework")
CONFIG_FILE = os.path.join(CONFIG_DIR, "config.json")
DATA_DIR = os.path.join(CONFIG_DIR, "data")
LOG_DIR = os.path.join(CONFIG_DIR, "logs")

# Ensure directories exist
for d in [CONFIG_DIR, DATA_DIR, LOG_DIR]:
    os.makedirs(d, exist_ok=True)


class TrayStatus(Enum):
    """System tray status indicators"""
    IDLE = "idle"
    WORKING = "working"
    ERROR = "error"
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"


@dataclass
class TrayConfig:
    """Application configuration"""
    llm_provider: str = "deepseek"
    llm_api_key: str = ""
    llm_model: str = "deepseek-chat"
    web_port: int = 8080
    auto_start: bool = False
    start_web_server: bool = True
    minimize_to_tray: bool = True
    show_notifications: bool = True
    language: str = "zh-CN"
    theme: str = "system"
    hotkey_enabled: bool = True
    hotkey_toggle: str = "ctrl+shift+a"
    hotkey_quick_task: str = "ctrl+shift+q"
    max_steps: int = 50
    vision_enabled: bool = True
    ocr_enabled: bool = True
    
    @classmethod
    def load(cls) -> 'TrayConfig':
        """Load config from file"""
        try:
            if os.path.exists(CONFIG_FILE):
                with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
        except Exception as e:
            logger.warning(f"Failed to load config: {e}")
        return cls()
    
    def save(self):
        """Save config to file"""
        try:
            with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
                json.dump(self.__dict__, f, ensure_ascii=False, indent=2)
            logger.info("Config saved")
        except Exception as e:
            logger.error(f"Failed to save config: {e}")


class TaskHistory:
    """Task execution history"""
    
    def __init__(self, max_items: int = 100):
        self.max_items = max_items
        self.history_file = os.path.join(DATA_DIR, "task_history.json")
        self._items: List[Dict] = []
        self._load()
    
    def _load(self):
        """Load history from file"""
        try:
            if os.path.exists(self.history_file):
                with open(self.history_file, 'r', encoding='utf-8') as f:
                    self._items = json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load history: {e}")
    
    def _save(self):
        """Save history to file"""
        try:
            with open(self.history_file, 'w', encoding='utf-8') as f:
                json.dump(self._items[-self.max_items:], f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Failed to save history: {e}")
    
    def add(self, task: str, status: str, result: str = "", duration_ms: float = 0):
        """Add a task to history"""
        item = {
            "task": task,
            "status": status,
            "result": result[:200] if result else "",
            "duration_ms": duration_ms,
            "timestamp": datetime.now().isoformat()
        }
        self._items.append(item)
        if len(self._items) > self.max_items:
            self._items = self._items[-self.max_items:]
        self._save()
    
    def get_recent(self, count: int = 10) -> List[Dict]:
        """Get recent tasks"""
        return list(reversed(self._items[-count:]))
    
    def clear(self):
        """Clear all history"""
        self._items = []
        self._save()


class NotificationManager:
    """Cross-platform notification manager"""
    
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self._platform = sys.platform
    
    def notify(self, title: str, message: str, icon_path: Optional[str] = None):
        """Send desktop notification"""
        if not self.enabled:
            return
        
        try:
            if self._platform == "win32":
                self._notify_windows(title, message)
            elif self._platform == "darwin":
                self._notify_macos(title, message)
            elif self._platform.startswith("linux"):
                self._notify_linux(title, message)
        except Exception as e:
            logger.debug(f"Notification failed: {e}")
    
    def _notify_windows(self, title: str, message: str):
        """Windows notification via PowerShell"""
        try:
            ps_script = f'''
            [Windows.UI.Notifications.ToastNotificationManager, Windows.UI.Notifications, ContentType = WindowsRuntime] > $null
            [Windows.Data.Xml.Dom.XmlDocument, Windows.Data.Xml.Dom.XmlDocument, ContentType = WindowsRuntime] > $null
            $template = @"
            <toast>
                <visual>
                    <binding template="ToastGeneric">
                        <text>{title}</text>
                        <text>{message}</text>
                    </binding>
                </visual>
            </toast>
"@
            $xml = New-Object Windows.Data.Xml.Dom.XmlDocument
            $xml.LoadXml($template)
            $toast = [Windows.UI.Notifications.ToastNotification]::new($xml)
            [Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier("AGI Agent").Show($toast)
            '''
            subprocess.run(
                ["powershell", "-Command", ps_script],
                capture_output=True, timeout=5
            )
        except Exception:
            # Fallback: balloon tip
            try:
                import ctypes
                ctypes.windll.user32.MessageBoxTimeoutW(0, message, title, 0, 0, 3000)
            except Exception:
                pass
    
    def _notify_macos(self, title: str, message: str):
        """macOS notification via osascript"""
        subprocess.run([
            "osascript", "-e",
            f'display notification "{message}" with title "{title}"'
        ], capture_output=True, timeout=5)
    
    def _notify_linux(self, title: str, message: str):
        """Linux notification via notify-send"""
        try:
            subprocess.run(
                ["notify-send", title, message, "-t", "3000"],
                capture_output=True, timeout=5
            )
        except FileNotFoundError:
            logger.debug("notify-send not available")


class WebServerManager:
    """Manage the web dashboard server"""
    
    def __init__(self, port: int = 8080):
        self.port = port
        self.process: Optional[subprocess.Popen] = None
        self._running = False
    
    def start(self) -> bool:
        """Start web server"""
        if self._running:
            return True
        
        try:
            # Find the web directory
            web_dir = Path(__file__).parent.parent / "web"
            if not web_dir.exists():
                # Try alternative paths
                for candidate in [
                    Path(os.getcwd()) / "web",
                    Path(os.getcwd()) / "agi_unified_framework" / "web"
                ]:
                    if candidate.exists():
                        web_dir = candidate
                        break
            
            if not web_dir.exists():
                logger.error(f"Web directory not found")
                return False
            
            server_script = web_dir / "start_server.py"
            if server_script.exists():
                self.process = subprocess.Popen(
                    [sys.executable, str(server_script)],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
                )
            else:
                # Use built-in HTTP server
                self.process = subprocess.Popen(
                    [sys.executable, "-m", "http.server", str(self.port)],
                    cwd=str(web_dir),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
                )
            
            self._running = True
            logger.info(f"Web server started on port {self.port}")
            return True
        except Exception as e:
            logger.error(f"Failed to start web server: {e}")
            return False
    
    def stop(self):
        """Stop web server"""
        if self.process:
            try:
                self.process.terminate()
                self.process.wait(timeout=5)
            except Exception:
                self.process.kill()
            self.process = None
            self._running = False
            logger.info("Web server stopped")
    
    def is_running(self) -> bool:
        """Check if server is running"""
        if self.process:
            self.process.poll()
            return self.process.returncode is None
        return False
    
    def open_browser(self):
        """Open web dashboard in browser"""
        import webbrowser
        webbrowser.open(f"http://localhost:{self.port}/index.html")


class AutoStartManager:
    """Manage auto-start on system boot"""
    
    @staticmethod
    def is_enabled() -> bool:
        """Check if auto-start is enabled"""
        if sys.platform == "win32":
            return AutoStartManager._check_windows()
        elif sys.platform == "darwin":
            return AutoStartManager._check_macos()
        elif sys.platform.startswith("linux"):
            return AutoStartManager._check_linux()
        return False
    
    @staticmethod
    def enable():
        """Enable auto-start"""
        if sys.platform == "win32":
            AutoStartManager._enable_windows()
        elif sys.platform == "darwin":
            AutoStartManager._enable_macos()
        elif sys.platform.startswith("linux"):
            AutoStartManager._enable_linux()
    
    @staticmethod
    def disable():
        """Disable auto-start"""
        if sys.platform == "win32":
            AutoStartManager._disable_windows()
        elif sys.platform == "darwin":
            AutoStartManager._disable_macos()
        elif sys.platform.startswith("linux"):
            AutoStartManager._disable_linux()
    
    @staticmethod
    def _get_app_path() -> str:
        """Get the path to the application"""
        if getattr(sys, 'frozen', False):
            return sys.executable
        return os.path.abspath(sys.argv[0])
    
    @staticmethod
    def _check_windows() -> bool:
        """Check Windows registry for auto-start"""
        try:
            import winreg
            key = winreg.OpenKey(
                winreg.HKEY_CURRENT_USER,
                r"Software\Microsoft\Windows\CurrentVersion\Run",
                0, winreg.KEY_READ
            )
            try:
                value, _ = winreg.QueryValueEx(key, "AGIFramework")
                winreg.CloseKey(key)
                return True
            except FileNotFoundError:
                winreg.CloseKey(key)
                return False
        except Exception:
            return False
    
    @staticmethod
    def _enable_windows():
        """Enable auto-start on Windows"""
        try:
            import winreg
            key = winreg.OpenKey(
                winreg.HKEY_CURRENT_USER,
                r"Software\Microsoft\Windows\CurrentVersion\Run",
                0, winreg.KEY_SET_VALUE
            )
            winreg.SetValueEx(key, "AGIFramework", 0, winreg.REG_SZ, 
                             f'"{AutoStartManager._get_app_path()}" --minimized')
            winreg.CloseKey(key)
            logger.info("Auto-start enabled on Windows")
        except Exception as e:
            logger.error(f"Failed to enable auto-start: {e}")
    
    @staticmethod
    def _disable_windows():
        """Disable auto-start on Windows"""
        try:
            import winreg
            key = winreg.OpenKey(
                winreg.HKEY_CURRENT_USER,
                r"Software\Microsoft\Windows\CurrentVersion\Run",
                0, winreg.KEY_SET_VALUE
            )
            winreg.DeleteValue(key, "AGIFramework")
            winreg.CloseKey(key)
            logger.info("Auto-start disabled on Windows")
        except Exception as e:
            logger.error(f"Failed to disable auto-start: {e}")
    
    @staticmethod
    def _check_macos() -> bool:
        """Check macOS LaunchAgents"""
        plist_path = os.path.expanduser(
            "~/Library/LaunchAgents/com.agi.framework.plist"
        )
        return os.path.exists(plist_path)
    
    @staticmethod
    def _enable_macos():
        """Enable auto-start on macOS"""
        plist_content = f'''<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.agi.framework</string>
    <key>ProgramArguments</key>
    <array>
        <string>{AutoStartManager._get_app_path()}</string>
        <string>--minimized</string>
    </array>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <false/>
</dict>
</plist>'''
        plist_path = os.path.expanduser(
            "~/Library/LaunchAgents/com.agi.framework.plist"
        )
        os.makedirs(os.path.dirname(plist_path), exist_ok=True)
        with open(plist_path, 'w') as f:
            f.write(plist_content)
        logger.info("Auto-start enabled on macOS")
    
    @staticmethod
    def _disable_macos():
        """Disable auto-start on macOS"""
        plist_path = os.path.expanduser(
            "~/Library/LaunchAgents/com.agi.framework.plist"
        )
        if os.path.exists(plist_path):
            os.remove(plist_path)
            logger.info("Auto-start disabled on macOS")
    
    @staticmethod
    def _check_linux() -> bool:
        """Check Linux autostart desktop file"""
        desktop_path = os.path.expanduser(
            "~/.config/autostart/agi-framework.desktop"
        )
        return os.path.exists(desktop_path)
    
    @staticmethod
    def _enable_linux():
        """Enable auto-start on Linux"""
        desktop_content = f'''[Desktop Entry]
Type=Application
Name=AGI Framework
Exec={AutoStartManager._get_app_path()} --minimized
Icon=agi-framework
Terminal=false
Hidden=false
X-GNOME-Autostart-enabled=true
'''
        desktop_dir = os.path.expanduser("~/.config/autostart")
        os.makedirs(desktop_dir, exist_ok=True)
        desktop_path = os.path.join(desktop_dir, "agi-framework.desktop")
        with open(desktop_path, 'w') as f:
            f.write(desktop_content)
        logger.info("Auto-start enabled on Linux")
    
    @staticmethod
    def _disable_linux():
        """Disable auto-start on Linux"""
        desktop_path = os.path.expanduser(
            "~/.config/autostart/agi-framework.desktop"
        )
        if os.path.exists(desktop_path):
            os.remove(desktop_path)
            logger.info("Auto-start disabled on Linux")


class SettingsDialog:
    """Settings dialog using tkinter (cross-platform)"""
    
    def __init__(self, config: TrayConfig, on_save: Optional[Callable] = None):
        self.config = config
        self.on_save = on_save
        self._window = None
    
    def show(self):
        """Show settings dialog"""
        try:
            self._show_tkinter()
        except Exception as e:
            logger.error(f"Failed to show settings: {e}")
    
    def _show_tkinter(self):
        """Show settings using tkinter"""
        import tkinter as tk
        from tkinter import ttk, messagebox, filedialog
        
        self._window = tk.Tk()
        self._window.title(f"{APP_NAME} - 设置")
        self._window.geometry("600x500")
        self._window.resizable(False, False)
        
        # Create notebook (tabs)
        notebook = ttk.Notebook(self._window)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Tab 1: LLM Settings
        llm_frame = ttk.Frame(notebook, padding=10)
        notebook.add(llm_frame, text="🤖 大模型设置")
        
        ttk.Label(llm_frame, text="LLM提供商:").grid(row=0, column=0, sticky="w", pady=5)
        self._llm_provider = tk.StringVar(value=self.config.llm_provider)
        providers = ["openai", "deepseek", "moonshot", "zhipu", "baidu", "alibaba", "anthropic"]
        ttk.Combobox(llm_frame, textvariable=self._llm_provider, values=providers, 
                     state="readonly", width=30).grid(row=0, column=1, pady=5)
        
        ttk.Label(llm_frame, text="API Key:").grid(row=1, column=0, sticky="w", pady=5)
        self._api_key = tk.StringVar(value=self.config.llm_api_key)
        ttk.Entry(llm_frame, textvariable=self._api_key, width=35, show="*").grid(row=1, column=1, pady=5)
        
        ttk.Label(llm_frame, text="模型名称:").grid(row=2, column=0, sticky="w", pady=5)
        self._model_name = tk.StringVar(value=self.config.llm_model)
        ttk.Entry(llm_frame, textvariable=self._model_name, width=35).grid(row=2, column=1, pady=5)
        
        self._vision_var = tk.BooleanVar(value=self.config.vision_enabled)
        ttk.Checkbutton(llm_frame, text="启用视觉理解 (截图分析)", 
                        variable=self._vision_var).grid(row=3, column=0, columnspan=2, sticky="w", pady=5)
        
        self._ocr_var = tk.BooleanVar(value=self.config.ocr_enabled)
        ttk.Checkbutton(llm_frame, text="启用OCR文字识别", 
                        variable=self._ocr_var).grid(row=4, column=0, columnspan=2, sticky="w", pady=5)
        
        ttk.Label(llm_frame, text="最大执行步数:").grid(row=5, column=0, sticky="w", pady=5)
        self._max_steps = tk.IntVar(value=self.config.max_steps)
        ttk.Spinbox(llm_frame, from_=5, to=200, textvariable=self._max_steps, 
                    width=10).grid(row=5, column=1, sticky="w", pady=5)
        
        # Tab 2: General Settings
        general_frame = ttk.Frame(notebook, padding=10)
        notebook.add(general_frame, text="⚙️ 通用设置")
        
        ttk.Label(general_frame, text="Web端口:").grid(row=0, column=0, sticky="w", pady=5)
        self._web_port = tk.IntVar(value=self.config.web_port)
        ttk.Spinbox(general_frame, from_=1024, to=65535, textvariable=self._web_port,
                    width=10).grid(row=0, column=1, sticky="w", pady=5)
        
        self._auto_start_var = tk.BooleanVar(value=self.config.auto_start)
        ttk.Checkbutton(general_frame, text="开机自动启动", 
                        variable=self._auto_start_var).grid(row=1, column=0, columnspan=2, sticky="w", pady=5)
        
        self._start_web_var = tk.BooleanVar(value=self.config.start_web_server)
        ttk.Checkbutton(general_frame, text="启动时开启Web服务", 
                        variable=self._start_web_var).grid(row=2, column=0, columnspan=2, sticky="w", pady=5)
        
        self._notify_var = tk.BooleanVar(value=self.config.show_notifications)
        ttk.Checkbutton(general_frame, text="显示桌面通知", 
                        variable=self._notify_var).grid(row=3, column=0, columnspan=2, sticky="w", pady=5)
        
        ttk.Label(general_frame, text="语言:").grid(row=4, column=0, sticky="w", pady=5)
        self._language = tk.StringVar(value=self.config.language)
        ttk.Combobox(general_frame, textvariable=self._language, 
                     values=["zh-CN", "en-US"], state="readonly", width=10).grid(row=4, column=1, sticky="w", pady=5)
        
        # Tab 3: About
        about_frame = ttk.Frame(notebook, padding=10)
        notebook.add(about_frame, text="ℹ️ 关于")
        
        ttk.Label(about_frame, text=APP_NAME, font=("Arial", 16, "bold")).pack(pady=10)
        ttk.Label(about_frame, text=f"版本: {APP_VERSION}").pack(pady=5)
        ttk.Label(about_frame, text="AGI Unified Framework").pack(pady=5)
        ttk.Label(about_frame, text="开源智能体框架").pack(pady=5)
        ttk.Label(about_frame, text="支持多模型LLM · GUI自动化 · 多智能体协作").pack(pady=10)
        
        # Buttons
        btn_frame = ttk.Frame(self._window)
        btn_frame.pack(fill=tk.X, padx=10, pady=10)
        
        def save_and_close():
            self.config.llm_provider = self._llm_provider.get()
            self.config.llm_api_key = self._api_key.get()
            self.config.llm_model = self._model_name.get()
            self.config.vision_enabled = self._vision_var.get()
            self.config.ocr_enabled = self._ocr_var.get()
            self.config.max_steps = self._max_steps.get()
            self.config.web_port = self._web_port.get()
            self.config.auto_start = self._auto_start_var.get()
            self.config.start_web_server = self._start_web_var.get()
            self.config.show_notifications = self._notify_var.get()
            self.config.language = self._language.get()
            self.config.save()
            
            # Handle auto-start
            if self.config.auto_start:
                AutoStartManager.enable()
            else:
                AutoStartManager.disable()
            
            if self.on_save:
                self.on_save(self.config)
            
            messagebox.showinfo("设置", "设置已保存！")
            self._window.destroy()
        
        ttk.Button(btn_frame, text="保存", command=save_and_close).pack(side=tk.RIGHT, padx=5)
        ttk.Button(btn_frame, text="取消", command=self._window.destroy).pack(side=tk.RIGHT, padx=5)
        
        self._window.mainloop()


class QuickTaskDialog:
    """Quick task input dialog"""
    
    def __init__(self, on_submit: Optional[Callable] = None):
        self.on_submit = on_submit
    
    def show(self):
        """Show quick task dialog"""
        try:
            import tkinter as tk
            from tkinter import ttk
            
            window = tk.Tk()
            window.title(f"{APP_NAME} - 快速任务")
            window.geometry("500x200")
            window.resizable(False, False)
            
            ttk.Label(window, text="输入任务描述:", font=("Arial", 11)).pack(pady=(15, 5))
            
            task_var = tk.StringVar()
            entry = ttk.Entry(window, textvariable=task_var, width=50, font=("Arial", 11))
            entry.pack(pady=5, padx=15)
            entry.focus_set()
            
            # Recent tasks
            history = TaskHistory()
            recent = history.get_recent(5)
            if recent:
                ttk.Label(window, text="最近任务:").pack(pady=(10, 2))
                for item in recent:
                    task_text = item["task"][:40] + "..." if len(item["task"]) > 40 else item["task"]
                    status_icon = "✅" if item["status"] == "success" else "❌"
                    btn = ttk.Button(window, text=f"{status_icon} {task_text}",
                                    command=lambda t=item["task"]: task_var.set(t))
                    btn.pack(pady=1)
            
            def submit():
                task = task_var.get().strip()
                if task and self.on_submit:
                    self.on_submit(task)
                window.destroy()
            
            def on_enter(event):
                submit()
            
            entry.bind("<Return>", on_enter)
            
            btn_frame = ttk.Frame(window)
            btn_frame.pack(pady=10)
            ttk.Button(btn_frame, text="执行", command=submit).pack(side=tk.LEFT, padx=5)
            ttk.Button(btn_frame, text="取消", command=window.destroy).pack(side=tk.LEFT, padx=5)
            
            window.mainloop()
        except Exception as e:
            logger.error(f"Failed to show quick task dialog: {e}")


class SystemTrayApp:
    """
    Main system tray application.
    Cross-platform tray icon with menu and functionality.
    """
    
    def __init__(self, config: Optional[TrayConfig] = None):
        self.config = config or TrayConfig.load()
        self.status = TrayStatus.IDLE
        self.notifications = NotificationManager(self.config.show_notifications)
        self.task_history = TaskHistory()
        self.web_server = WebServerManager(self.config.web_port)
        self._running = False
        self._tray_icon = None
        self._agent = None
    
    def _create_icon_image(self):
        """Create tray icon image"""
        try:
            from PIL import Image, ImageDraw
            
            # Create a simple AGI icon
            size = 64
            img = Image.new('RGBA', (size, size), (0, 0, 0, 0))
            draw = ImageDraw.Draw(img)
            
            # Draw circle background
            draw.ellipse([4, 4, 60, 60], fill=(66, 133, 244, 255))
            
            # Draw "A" letter
            draw.text((18, 12), "A", fill=(255, 255, 255, 255))
            draw.text((22, 28), "G", fill=(255, 255, 255, 255))
            draw.text((26, 44), "I", fill=(255, 255, 255, 255))
            
            return img
        except ImportError:
            logger.warning("PIL not available, using default icon")
            return None
    
    def _get_status_icon(self):
        """Get icon based on status"""
        try:
            from PIL import Image, ImageDraw
            
            size = 64
            img = Image.new('RGBA', (size, size), (0, 0, 0, 0))
            draw = ImageDraw.Draw(img)
            
            colors = {
                TrayStatus.IDLE: (66, 133, 244),
                TrayStatus.WORKING: (255, 193, 7),
                TrayStatus.ERROR: (234, 67, 53),
                TrayStatus.CONNECTED: (52, 168, 83),
                TrayStatus.DISCONNECTED: (158, 158, 158),
            }
            color = colors.get(self.status, (66, 133, 244))
            draw.ellipse([4, 4, 60, 60], fill=color + (255,))
            draw.text((18, 12), "A", fill=(255, 255, 255, 255))
            draw.text((22, 28), "G", fill=(255, 255, 255, 255))
            draw.text((26, 44), "I", fill=(255, 255, 255, 255))
            
            return img
        except ImportError:
            return self._create_icon_image()
    
    def _show_dashboard(self, icon=None, item=None):
        """Open web dashboard"""
        if not self.web_server.is_running():
            if self.web_server.start():
                time.sleep(1)
        self.web_server.open_browser()
    
    def _quick_task(self, icon=None, item=None):
        """Show quick task dialog"""
        def on_submit(task: str):
            threading.Thread(target=self._execute_task, args=(task,), daemon=True).start()
        
        dialog = QuickTaskDialog(on_submit=on_submit)
        dialog.show()
    
    def _open_settings(self, icon=None, item=None):
        """Open settings dialog"""
        def on_save(config):
            self.config = config
            self.notifications.enabled = config.show_notifications
        
        dialog = SettingsDialog(self.config, on_save=on_save)
        dialog.show()
    
    def _show_history(self, icon=None, item=None):
        """Show task history"""
        try:
            import tkinter as tk
            from tkinter import ttk
            
            window = tk.Tk()
            window.title(f"{APP_NAME} - 任务历史")
            window.geometry("600x400")
            
            tree = ttk.Treeview(window, columns=("status", "task", "time", "duration"), show="headings")
            tree.heading("status", text="状态")
            tree.heading("task", text="任务")
            tree.heading("time", text="时间")
            tree.heading("duration", text="耗时")
            tree.column("status", width=60)
            tree.column("task", width=300)
            tree.column("time", width=140)
            tree.column("duration", width=80)
            
            for item in self.task_history.get_recent(50):
                status = "✅" if item["status"] == "success" else "❌"
                time_str = item["timestamp"][:19]
                duration = f"{item['duration_ms']:.0f}ms"
                tree.insert("", tk.END, values=(status, item["task"], time_str, duration))
            
            tree.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            
            ttk.Button(window, text="关闭", command=window.destroy).pack(pady=5)
            window.mainloop()
        except Exception as e:
            logger.error(f"Failed to show history: {e}")
    
    def _execute_task(self, task: str):
        """Execute a task using the agent brain"""
        start_time = time.time()
        self.status = TrayStatus.WORKING
        self.notifications.notify(APP_NAME, f"开始执行: {task[:50]}")
        
        try:
            # Import and create agent
            from computer_use.agent_brain import LLMClientFactory, LLMProvider, AgentBrain
            from computer_use.real_screen import RealScreenCapture
            from computer_use.real_input import InputExecutor
            
            provider_map = {
                "openai": LLMProvider.OPENAI,
                "deepseek": LLMProvider.DEEPSEEK,
                "moonshot": LLMProvider.MOONSHOT,
                "zhipu": LLMProvider.ZHIPU,
                "baidu": LLMProvider.BAIDU,
                "alibaba": LLMProvider.ALIBABA,
                "anthropic": LLMProvider.ANTHROPIC,
            }
            
            provider = provider_map.get(self.config.llm_provider, LLMProvider.DEEPSEEK)
            client = LLMClientFactory.create_client(provider, self.config.llm_api_key)
            
            agent = AgentBrain(
                client,
                vision_enabled=self.config.vision_enabled,
                max_steps=self.config.max_steps
            )
            
            # Get screenshot and OCR callbacks
            def get_screenshot():
                try:
                    screen = RealScreenCapture()
                    return screen.capture_to_base64()
                except Exception:
                    return None
            
            def get_ocr():
                try:
                    from computer_use.real_screen import RealOCR
                    screen = RealScreenCapture()
                    ocr = RealOCR()
                    img = screen.capture_screen()
                    result = ocr.recognize(img)
                    return "\n".join([f"{r['text']} ({r['confidence']:.0%})" for r in result])
                except Exception:
                    return ""
            
            # Run task
            results = asyncio.run(agent.run_task(
                task=task,
                get_screenshot=get_screenshot,
                get_ocr=get_ocr
            ))
            
            duration = (time.time() - start_time) * 1000
            self.task_history.add(task, "success", f"完成 {len(results)} 步", duration)
            self.status = TrayStatus.CONNECTED
            self.notifications.notify(APP_NAME, f"任务完成: {task[:30]}")
            
        except Exception as e:
            duration = (time.time() - start_time) * 1000
            self.task_history.add(task, "error", str(e), duration)
            self.status = TrayStatus.ERROR
            self.notifications.notify(APP_NAME, f"任务失败: {str(e)[:50]}")
            logger.error(f"Task execution failed: {e}")
    
    def _quit(self, icon=None, item=None):
        """Quit application"""
        self._running = False
        self.web_server.stop()
        if self._tray_icon:
            self._tray_icon.stop()
    
    def run(self):
        """Run the system tray application"""
        self._running = True
        
        # Start web server if configured
        if self.config.start_web_server:
            self.web_server.start()
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, lambda s, f: self._quit())
        signal.signal(signal.SIGTERM, lambda s, f: self._quit())
        
        try:
            import pystray
            
            icon_image = self._create_icon_image()
            if icon_image is None:
                # Create a simple icon without PIL
                icon_image = self._create_text_icon()
            
            menu = pystray.Menu(
                pystray.MenuItem("🖥️ 打开控制台", self._show_dashboard, default=True),
                pystray.MenuItem("⚡ 快速任务", self._quick_task),
                pystray.MenuItem("📋 任务历史", self._show_history),
                pystray.Menu.SEPARATOR,
                pystray.MenuItem("⚙️ 设置", self._open_settings),
                pystray.Menu.SEPARATOR,
                pystray.MenuItem("退出", self._quit),
            )
            
            self._tray_icon = pystray.Icon(
                "agi_framework",
                icon_image,
                APP_NAME,
                menu
            )
            
            logger.info(f"{APP_NAME} v{APP_VERSION} started")
            self.notifications.notify(APP_NAME, "智能体已启动")
            
            self._tray_icon.run()
            
        except ImportError:
            logger.warning("pystray not available, running in CLI mode")
            self._run_cli_mode()
        
        finally:
            self.web_server.stop()
            logger.info("Application stopped")
    
    def _create_text_icon(self):
        """Create icon without PIL"""
        try:
            import pystray
            return pystray.Icon("agi", "AGI", APP_NAME)
        except Exception:
            return None
    
    def _run_cli_mode(self):
        """Run in CLI mode when GUI is not available"""
        print(f"\n{APP_NAME} v{APP_VERSION}")
        print("=" * 40)
        print("系统托盘不可用，运行CLI模式")
        print()
        
        while self._running:
            try:
                print("\n选项:")
                print("1. 执行任务")
                print("2. 查看历史")
                print("3. 打开Web控制台")
                print("4. 设置")
                print("5. 退出")
                
                choice = input("\n请选择: ").strip()
                
                if choice == "1":
                    task = input("输入任务: ").strip()
                    if task:
                        threading.Thread(target=self._execute_task, args=(task,), daemon=True).start()
                elif choice == "2":
                    for item in self.task_history.get_recent(10):
                        status = "✅" if item["status"] == "success" else "❌"
                        print(f"  {status} [{item['timestamp'][:19]}] {item['task']}")
                elif choice == "3":
                    self._show_dashboard()
                elif choice == "4":
                    SettingsDialog(self.config).show()
                elif choice == "5":
                    self._running = False
            except (KeyboardInterrupt, EOFError):
                self._running = False


# Need asyncio for agent execution
import asyncio


def main():
    """Main entry point"""
    import argparse
    parser = argparse.ArgumentParser(description=APP_NAME)
    parser.add_argument("--minimized", action="store_true", help="Start minimized")
    parser.add_argument("--config", type=str, help="Config file path")
    parser.add_argument("--port", type=int, help="Web server port")
    parser.add_argument("--version", action="store_true", help="Show version")
    args = parser.parse_args()
    
    if args.version:
        print(f"{APP_NAME} v{APP_VERSION}")
        return
    
    config = TrayConfig.load()
    if args.port:
        config.web_port = args.port
    if args.config:
        try:
            with open(args.config, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for k, v in data.items():
                    if hasattr(config, k):
                        setattr(config, k, v)
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
    
    app = SystemTrayApp(config)
    app.run()


if __name__ == "__main__":
    main()
