"""
AGI Unified Framework CLI - 交互模式模块

提供实时对话、命令补全、历史记录和快捷键功能。
支持交互式聊天和命令执行。

使用示例:
    agi chat                           # 启动交互模式
"""

import os
import sys
import atexit
from typing import Any, Dict, List, Optional, Callable
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime

import click
from click import Context


@dataclass
class ChatMessage:
    """聊天消息"""
    role: str  # user, assistant, system
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


class ChatHistory:
    """聊天历史管理"""

    def __init__(self, history_file: Optional[str] = None):
        if history_file:
            self._history_file = Path(history_file).expanduser()
        else:
            home = Path.home()
            self._history_file = home / ".agi_framework" / "chat_history.txt"

        self._messages: List[ChatMessage] = []
        self._load_history()

    def _load_history(self) -> None:
        """加载历史记录"""
        if self._history_file.exists():
            try:
                lines = self._history_file.read_text(encoding='utf-8').strip().split('\n')
                for line in lines[-100:]:  # 只加载最近100条
                    if line.strip():
                        # 简单解析格式: [timestamp] role: content
                        self._messages.append(ChatMessage(
                            role="user",
                            content=line,
                            timestamp=datetime.now()
                        ))
            except Exception:
                pass

    def add(self, role: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """添加消息"""
        message = ChatMessage(
            role=role,
            content=content,
            metadata=metadata or {}
        )
        self._messages.append(message)

        # 追加到文件
        try:
            self._history_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self._history_file, 'a', encoding='utf-8') as f:
                f.write(f"[{message.timestamp}] {role}: {content}\n")
        except Exception:
            pass

    def get_recent(self, count: int = 10) -> List[ChatMessage]:
        """获取最近的消息"""
        return self._messages[-count:]

    def clear(self) -> None:
        """清空历史"""
        self._messages.clear()
        if self._history_file.exists():
            self._history_file.unlink()


class InteractiveSession:
    """
    交互式会话

    提供实时对话、命令补全和快捷键支持。
    """

    # 可用命令
    COMMANDS = {
        "/help": "显示帮助信息",
        "/clear": "清空屏幕",
        "/history": "显示聊天历史",
        "/personality": "切换人格",
        "/model": "切换模型",
        "/status": "显示系统状态",
        "/quit": "退出交互模式",
        "/exit": "退出交互模式",
    }

    def __init__(self):
        self._history = ChatHistory()
        self._running = False
        self._current_personality = "assistant"
        self._current_model = "gpt-4"
        self._session_start = datetime.now()

    def start(self) -> None:
        """启动交互会话"""
        self._running = True

        # 显示欢迎信息
        self._show_welcome()

        # 注册退出处理
        atexit.register(self._on_exit)

        # 主循环
        while self._running:
            try:
                # 获取用户输入
                user_input = self._get_input()

                if not user_input.strip():
                    continue

                # 处理命令
                if user_input.startswith("/"):
                    if self._handle_command(user_input):
                        continue

                # 处理普通消息
                self._process_message(user_input)

            except KeyboardInterrupt:
                click.echo("\n")
                if click.confirm("是否退出?", default=True):
                    break
            except EOFError:
                break
            except Exception as e:
                click.echo(click.style(f"错误: {str(e)}", fg="red"), err=True)

        self._running = False
        click.echo(click.style("\n再见!", fg="green"))

    def _show_welcome(self) -> None:
        """显示欢迎信息"""
        click.echo()
        click.echo(click.style("=" * 60, fg="cyan"))
        click.echo(click.style("   AGI 交互模式", fg="cyan", bold=True))
        click.echo(click.style("=" * 60, fg="cyan"))
        click.echo()
        click.echo("输入消息开始对话，或使用 /help 查看可用命令。")
        click.echo()
        click.echo(f"当前人格: {click.style(self._current_personality, fg='green')}")
        click.echo(f"当前模型: {click.style(self._current_model, fg='green')}")
        click.echo()

    def _get_input(self) -> str:
        """获取用户输入"""
        # 使用Click的prompt功能
        prompt_text = click.style(f"[{self._current_personality}] > ", fg="cyan")

        try:
            # 尝试使用readline进行历史记录
            import readline
            user_input = input(prompt_text)
        except (ImportError, NameError):
            user_input = click.prompt(prompt_text, prompt_suffix="")

        return user_input

    def _handle_command(self, command_line: str) -> bool:
        """处理命令，返回是否已处理"""
        parts = command_line.strip().split()
        if not parts:
            return False

        command = parts[0].lower()
        args = parts[1:]

        if command == "/help":
            self._show_help()
        elif command == "/clear":
            self._clear_screen()
        elif command == "/history":
            self._show_history()
        elif command == "/personality":
            self._switch_personality(args)
        elif command == "/model":
            self._switch_model(args)
        elif command == "/status":
            self._show_status()
        elif command in ["/quit", "/exit"]:
            self._running = False
        else:
            click.echo(click.style(f"未知命令: {command}", fg="red"))
            click.echo("使用 /help 查看可用命令")

        return True

    def _show_help(self) -> None:
        """显示帮助信息"""
        click.echo()
        click.echo(click.style("可用命令:", fg="cyan", bold=True))
        click.echo()

        for cmd, desc in self.COMMANDS.items():
            click.echo(f"  {click.style(cmd, fg='green'):<15} {desc}")

        click.echo()
        click.echo("快捷操作:")
        click.echo("  Ctrl+C        中断当前操作")
        click.echo("  Ctrl+D        退出交互模式")
        click.echo("  上/下箭头     浏览历史记录")
        click.echo()

    def _clear_screen(self) -> None:
        """清空屏幕"""
        os.system('cls' if os.name == 'nt' else 'clear')
        self._show_welcome()

    def _show_history(self) -> None:
        """显示聊天历史"""
        messages = self._history.get_recent(20)

        if not messages:
            click.echo("暂无聊天记录")
            return

        click.echo()
        click.echo(click.style("最近聊天记录:", fg="cyan", bold=True))
        click.echo()

        for msg in messages:
            time_str = msg.timestamp.strftime("%H:%M:%S")
            if msg.role == "user":
                click.echo(f"[{time_str}] {click.style('你:', fg='green')} {msg.content}")
            elif msg.role == "assistant":
                click.echo(f"[{time_str}] {click.style('AI:', fg='blue')} {msg.content}")

        click.echo()

    def _switch_personality(self, args: List[str]) -> None:
        """切换人格"""
        if not args:
            click.echo(f"当前人格: {click.style(self._current_personality, fg='green')}")
            click.echo("可用人格: assistant, coder, teacher, creative")
            click.echo("用法: /personality <name>")
            return

        personality = args[0]
        valid_personalities = ["assistant", "coder", "teacher", "creative"]

        if personality not in valid_personalities:
            click.echo(click.style(f"无效的人格: {personality}", fg="red"))
            click.echo(f"可用人格: {', '.join(valid_personalities)}")
            return

        self._current_personality = personality
        click.echo(click.style(f"已切换到 {personality} 人格", fg="green"))

    def _switch_model(self, args: List[str]) -> None:
        """切换模型"""
        if not args:
            click.echo(f"当前模型: {click.style(self._current_model, fg='green')}")
            click.echo("用法: /model <name>")
            return

        model = args[0]
        self._current_model = model
        click.echo(click.style(f"已切换到 {model} 模型", fg="green"))

    def _show_status(self) -> None:
        """显示系统状态"""
        session_duration = datetime.now() - self._session_start

        click.echo()
        click.echo(click.style("系统状态", fg="cyan", bold=True))
        click.echo()
        click.echo(f"会话时长:    {session_duration}")
        click.echo(f"当前人格:    {self._current_personality}")
        click.echo(f"当前模型:    {self._current_model}")
        click.echo(f"历史记录:    {len(self._history._messages)} 条消息")
        click.echo()

    def _process_message(self, message: str) -> None:
        """处理用户消息"""
        # 记录用户消息
        self._history.add("user", message)

        # 模拟处理（实际应调用LLM）
        click.echo()
        click.echo(click.style("思考中...", fg="yellow"), nl=False)

        import time
        time.sleep(0.5)

        click.echo("\r" + " " * 20 + "\r", nl=False)

        # 生成响应（模拟）
        response = self._generate_response(message)

        # 显示响应
        click.echo(click.style("AI: ", fg="blue", bold=True) + response)
        click.echo()

        # 记录响应
        self._history.add("assistant", response)

    def _generate_response(self, message: str) -> str:
        """生成响应（模拟）"""
        # 简单的关键词响应
        lower_msg = message.lower()

        if any(word in lower_msg for word in ["hello", "hi", "你好"]):
            return f"你好！我是基于 {self._current_personality} 人格的AI助手。有什么可以帮助你的吗？"

        if any(word in lower_msg for word in ["help", "帮助"]):
            return "我可以帮助你回答问题、编写代码、分析数据等。请告诉我你需要什么帮助。"

        if any(word in lower_msg for word in ["code", "编程", "代码"]):
            return "我可以帮你编写和调试代码。请告诉我你使用的编程语言和具体需求。"

        if any(word in lower_msg for word in ["weather", "天气"]):
            return "我无法获取实时天气信息。建议你查看天气预报网站或使用天气应用。"

        if any(word in lower_msg for word in ["time", "时间", "几点"]):
            return f"当前时间是 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

        # 默认响应
        default_responses = [
            "我理解你的问题。让我思考一下...",
            "这是一个有趣的话题。",
            "我可以帮你处理这个请求。",
            "请提供更多细节，以便我更好地帮助你。",
            "收到，我正在处理你的请求。",
        ]

        import random
        return random.choice(default_responses)

    def _on_exit(self) -> None:
        """退出处理"""
        if hasattr(self, '_history'):
            # 保存会话摘要
            pass


# Click命令定义
@click.command(name="chat", help="启动交互模式")
@click.option("--personality", "-p", help="指定人格")
@click.option("--model", "-m", help="指定模型")
@click.option("--no-history", is_flag=True, help="不加载历史记录")
@click.pass_context
def chat(ctx: Context, personality: Optional[str], model: Optional[str], no_history: bool) -> None:
    """启动交互式聊天模式"""
    session = InteractiveSession()

    if personality:
        session._current_personality = personality

    if model:
        session._current_model = model

    if no_history:
        session._history.clear()

    try:
        session.start()
    except Exception as e:
        click.echo(click.style(f"启动失败: {str(e)}", fg="red"), err=True)
        ctx.exit(1)
