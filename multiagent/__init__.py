"""
多智能体协作层 (Multi-Agent Orchestration)

实现完整的分布式多智能体系统，包括：
- 智能体注册与发现 (registry/)
- 联盟形成与任务分配 (alliance/)
- 辩论与共识引擎 (debate/)
- 信誉与激励系统 (reputation/, incentive/)
- 知识迁移与社会学习 (learning/)
- 多智能体仿真与沙盘 (simulator/)
- MCP协议与外部集成 (mcp/)
- 智能体市场与生态 (market/)
- 元智能体与自组织 (meta/)
"""

__version__ = "1.0.0"

# 子模块导入
from . import registry
from . import alliance
from . import debate
from . import reputation
from . import incentive
from . import learning
from . import simulator
from . import mcp
from . import market
from . import meta

__all__ = [
    "registry",
    "alliance", 
    "debate",
    "reputation",
    "incentive",
    "learning",
    "simulator",
    "mcp",
    "market",
    "meta",
]
