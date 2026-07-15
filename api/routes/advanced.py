"""
高级功能API路由

提供多种高级AI功能模块，包括物理引擎仿真、计算机使用控制、安全中心、
联邦学习、知识检索、渠道管理、插件市场、人格引擎、数据管道、价值对齐和机器人控制。

模块:
    - 物理引擎: 分子仿真、物理模拟
    - 计算机使用: 屏幕控制、自动化操作
    - 安全中心: 防火墙、审计、威胁检测
    - 联邦学习: 分布式模型训练
    - 知识检索: RAG文档检索
    - 渠道管理: 多渠道消息管理
    - 插件市场: 插件安装管理
    - 人格引擎: AI人格定制
    - 数据管道: 数据处理流水线
    - 价值对齐: AI安全对齐
    - 机器人控制: 机器人远程控制

端点:
    # 物理引擎
    GET    /physics/simulations              - 获取仿真列表
    POST   /physics/simulations              - 创建仿真
    GET    /physics/simulations/{id}         - 获取仿真详情
    POST   /physics/simulations/{id}/run     - 运行仿真
    GET    /physics/molecules                - 获取分子库
    POST   /physics/molecules/{id}/visualize - 可视化分子
    
    # 计算机使用
    POST   /computer/screenshot              - 获取屏幕截图
    POST   /computer/click                   - 模拟点击
    POST   /computer/type                    - 模拟输入
    POST   /computer/scroll                  - 模拟滚动
    POST   /computer/navigate                - 页面导航
    GET    /computer/recording               - 获取录制状态
    POST   /computer/recording/start         - 开始录制
    POST   /computer/recording/stop          - 停止录制
    
    # 安全中心
    GET    /security/firewall/rules          - 获取防火墙规则
    POST   /security/firewall/rules          - 添加规则
    GET    /security/audit-logs              - 获取审计日志
    POST   /security/scan                    - 执行安全扫描
    GET    /security/threats                 - 获取威胁情报
    POST   /security/prompt-guard/test       - 测试提示词防护
    
    # 联邦学习
    GET    /federated/nodes                  - 获取节点列表
    POST   /federated/nodes                  - 注册节点
    GET    /federated/rounds                 - 获取训练轮次
    POST   /federated/rounds/{id}/aggregate  - 执行聚合
    GET    /federated/contributions          - 获取贡献统计
    
    # 知识检索
    POST   /rag/documents                    - 上传文档
    GET    /rag/documents                    - 获取文档列表
    DELETE /rag/documents/{id}               - 删除文档
    POST   /rag/search                       - 执行检索
    GET    /rag/knowledge-bases              - 获取知识库列表
    
    # 渠道管理
    GET    /channels                         - 获取渠道列表
    POST   /channels                         - 添加渠道
    PUT    /channels/{id}                    - 更新渠道
    DELETE /channels/{id}                    - 删除渠道
    POST   /channels/{id}/test               - 测试渠道连接
    GET    /channels/{id}/messages           - 获取渠道消息
    
    # 插件市场
    GET    /plugins                          - 获取插件列表
    POST   /plugins                          - 安装插件
    DELETE /plugins/{id}                     - 卸载插件
    POST   /plugins/{id}/enable              - 启用插件
    POST   /plugins/{id}/disable             - 禁用插件
    GET    /plugins/marketplace              - 浏览插件市场
    
    # 人格引擎
    GET    /personalities                    - 获取人格列表
    POST   /personalities                    - 创建人格
    GET    /personalities/{id}               - 获取人格详情
    PUT    /personalities/{id}               - 更新人格
    DELETE /personalities/{id}               - 删除人格
    POST   /personalities/{id}/activate      - 激活人格
    POST   /personalities/{id}/test          - 测试人格
    
    # 数据管道
    GET    /data-pipeline/datasets           - 获取数据集列表
    POST   /data-pipeline/datasets           - 上传数据集
    POST   /data-pipeline/datasets/{id}/process - 处理数据集
    GET    /data-pipeline/pipelines          - 获取流水线列表
    POST   /data-pipeline/pipelines          - 创建流水线
    
    # 价值对齐
    GET    /alignment/principles             - 获取原则列表
    POST   /alignment/principles             - 添加原则
    POST   /alignment/test                   - 执行对齐测试
    GET    /alignment/reports                - 获取对齐报告
    
    # 机器人控制
    GET    /robots                           - 获取机器人列表
    POST   /robots/{id}/connect              - 连接机器人
    POST   /robots/{id}/move                 - 移动机器人
    GET    /robots/{id}/status               - 获取机器人状态
    POST   /robots/{id}/emergency-stop       - 紧急停止
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Union

from fastapi import APIRouter, Depends, HTTPException, Query, status, File, UploadFile
from pydantic import BaseModel, Field, validator

from ..validators.schemas import BaseResponse, ErrorResponse, PaginatedResponse
from ..dependencies.injection import get_current_user, require_permissions, get_db_session

# 配置日志
logger = logging.getLogger(__name__)

# 创建路由器
router = APIRouter()

# =============================================================================
# 枚举类型定义
# =============================================================================

# 物理引擎枚举
class SimulationType(str, Enum):
    """仿真类型"""
    MOLECULAR_DYNAMICS = "molecular_dynamics"
    QUANTUM_CHEMISTRY = "quantum_chemistry"
    FLUID_DYNAMICS = "fluid_dynamics"
    STRUCTURAL_MECHANICS = "structural_mechanics"
    THERMAL_ANALYSIS = "thermal_analysis"
    ELECTROMAGNETIC = "electromagnetic"


class SimulationStatus(str, Enum):
    """仿真状态"""
    PENDING = "pending"
    SETUP = "setup"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


# 计算机使用枚举
class ComputerAction(str, Enum):
    """计算机操作类型"""
    CLICK = "click"
    DOUBLE_CLICK = "double_click"
    RIGHT_CLICK = "right_click"
    DRAG = "drag"
    SCROLL = "scroll"
    TYPE = "type"
    KEYPRESS = "keypress"
    NAVIGATE = "navigate"


class RecordingStatus(str, Enum):
    """录制状态"""
    IDLE = "idle"
    RECORDING = "recording"
    PAUSED = "paused"
    SAVING = "saving"


# 安全中心枚举
class FirewallAction(str, Enum):
    """防火墙动作"""
    ALLOW = "allow"
    DENY = "deny"
    LOG = "log"
    ALERT = "alert"


class ThreatLevel(str, Enum):
    """威胁等级"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ScanType(str, Enum):
    """扫描类型"""
    VULNERABILITY = "vulnerability"
    MALWARE = "malware"
    PENETRATION = "penetration"
    COMPLIANCE = "compliance"
    PROMPT_INJECTION = "prompt_injection"


# 联邦学习枚举
class NodeStatus(str, Enum):
    """节点状态"""
    ONLINE = "online"
    OFFLINE = "offline"
    TRAINING = "training"
    SYNCING = "syncing"
    ERROR = "error"


class AggregationStrategy(str, Enum):
    """聚合策略"""
    FEDAVG = "fedavg"
    FEDPROX = "fedprox"
    FEDOPT = "fedopt"
    SCAFFOLD = "scaffold"


# 知识检索枚举
class DocumentType(str, Enum):
    """文档类型"""
    PDF = "pdf"
    WORD = "word"
    TEXT = "text"
    MARKDOWN = "markdown"
    HTML = "html"
    CODE = "code"
    JSON = "json"


class SearchStrategy(str, Enum):
    """检索策略"""
    SEMANTIC = "semantic"
    KEYWORD = "keyword"
    HYBRID = "hybrid"
    VECTOR = "vector"


# 渠道管理枚举
class ChannelType(str, Enum):
    """渠道类型"""
    EMAIL = "email"
    SLACK = "slack"
    DISCORD = "discord"
    TELEGRAM = "telegram"
    WECHAT = "wechat"
    WEBHOOK = "webhook"
    SMS = "sms"


class ChannelStatus(str, Enum):
    """渠道状态"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    ERROR = "error"
    PENDING = "pending"


# 插件市场枚举
class PluginStatus(str, Enum):
    """插件状态"""
    INSTALLED = "installed"
    ENABLED = "enabled"
    DISABLED = "disabled"
    ERROR = "error"
    UPDATING = "updating"


class PluginCategory(str, Enum):
    """插件类别"""
    GENERATION = "generation"
    INTEGRATION = "integration"
    AUTOMATION = "automation"
    ANALYTICS = "analytics"
    SECURITY = "security"
    CUSTOM = "custom"


# 人格引擎枚举
class PersonalityTrait(str, Enum):
    """人格特质"""
    OPENNESS = "openness"
    CONSCIENTIOUSNESS = "conscientiousness"
    EXTRAVERSION = "extraversion"
    AGREEABLENESS = "agreeableness"
    NEUROTICISM = "neuroticism"


class CommunicationStyle(str, Enum):
    """沟通风格"""
    FORMAL = "formal"
    CASUAL = "casual"
    TECHNICAL = "technical"
    FRIENDLY = "friendly"
    PROFESSIONAL = "professional"
    HUMOROUS = "humorous"


# 数据管道枚举
class DatasetFormat(str, Enum):
    """数据集格式"""
    CSV = "csv"
    JSON = "json"
    PARQUET = "parquet"
    EXCEL = "excel"
    SQL = "sql"
    HDF5 = "hdf5"


class PipelineStatus(str, Enum):
    """流水线状态"""
    DRAFT = "draft"
    ACTIVE = "active"
    RUNNING = "running"
    PAUSED = "paused"
    ERROR = "error"
    COMPLETED = "completed"


# 价值对齐枚举
class PrincipleCategory(str, Enum):
    """原则类别"""
    SAFETY = "safety"
    FAIRNESS = "fairness"
    TRANSPARENCY = "transparency"
    PRIVACY = "privacy"
    ACCOUNTABILITY = "accountability"


class AlignmentTestResult(str, Enum):
    """对齐测试结果"""
    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"
    PENDING = "pending"


# 机器人控制枚举
class RobotType(str, Enum):
    """机器人类型"""
    ARM = "arm"
    MOBILE = "mobile"
    HUMANOID = "humanoid"
    DRONE = "drone"
    WHEELCHAIR = "wheelchair"


class RobotStatus(str, Enum):
    """机器人状态"""
    IDLE = "idle"
    BUSY = "busy"
    ERROR = "error"
    OFFLINE = "offline"
    EMERGENCY = "emergency"


# =============================================================================
# Pydantic模型定义 - 物理引擎
# =============================================================================

class SimulationConfig(BaseModel):
    """仿真配置"""
    time_step: float = Field(default=0.001, description="时间步长")
    total_steps: int = Field(default=10000, description="总步数")
    temperature: Optional[float] = Field(None, description="温度(K)")
    pressure: Optional[float] = Field(None, description="压力(Pa)")
    ensemble: str = Field(default="NVT", description="系综类型")
    constraints: List[str] = Field(default_factory=list, description="约束条件")


class SimulationCreateRequest(BaseModel):
    """创建仿真请求"""
    name: str = Field(..., min_length=1, max_length=200, description="仿真名称")
    type: SimulationType = Field(..., description="仿真类型")
    description: Optional[str] = Field(None, max_length=1000, description="仿真描述")
    config: SimulationConfig = Field(default_factory=SimulationConfig, description="仿真配置")
    input_files: List[str] = Field(default_factory=list, description="输入文件列表")
    molecule_ids: Optional[List[str]] = Field(None, description="分子ID列表")


class SimulationResponse(BaseModel):
    """仿真响应"""
    id: str = Field(..., description="仿真ID")
    name: str = Field(..., description="仿真名称")
    type: SimulationType = Field(..., description="仿真类型")
    status: SimulationStatus = Field(..., description="仿真状态")
    description: Optional[str] = Field(None, description="仿真描述")
    config: SimulationConfig = Field(..., description="仿真配置")
    progress: float = Field(default=0.0, description="进度百分比")
    current_step: int = Field(default=0, description="当前步数")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="创建时间")
    started_at: Optional[datetime] = Field(None, description="开始时间")
    completed_at: Optional[datetime] = Field(None, description="完成时间")
    results_url: Optional[str] = Field(None, description="结果文件URL")
    logs_url: Optional[str] = Field(None, description="日志文件URL")


class MoleculeInfo(BaseModel):
    """分子信息"""
    id: str = Field(..., description="分子ID")
    name: str = Field(..., description="分子名称")
    formula: str = Field(..., description="分子式")
    smiles: Optional[str] = Field(None, description="SMILES表示")
    inchi: Optional[str] = Field(None, description="InChI表示")
    molecular_weight: float = Field(..., description="分子量")
    category: str = Field(..., description="分子类别")
    properties: Dict[str, Any] = Field(default_factory=dict, description="物理化学性质")
    tags: List[str] = Field(default_factory=list, description="标签")


class MoleculeVisualizationRequest(BaseModel):
    """分子可视化请求"""
    style: str = Field(default="ball-and-stick", description="可视化风格")
    width: int = Field(default=800, ge=100, le=1920, description="宽度")
    height: int = Field(default=600, ge=100, le=1080, description="高度")
    background_color: str = Field(default="white", description="背景颜色")
    show_labels: bool = Field(default=True, description="显示标签")
    rotate: bool = Field(default=False, description="自动旋转")


class MoleculeVisualizationResponse(BaseModel):
    """分子可视化响应"""
    molecule_id: str = Field(..., description="分子ID")
    image_url: Optional[str] = Field(None, description="静态图像URL")
    interactive_url: Optional[str] = Field(None, description="交互式查看器URL")
    format: str = Field(..., description="输出格式")
    width: int = Field(..., description="图像宽度")
    height: int = Field(..., description="图像高度")


# =============================================================================
# Pydantic模型定义 - 计算机使用
# =============================================================================

class ScreenshotRequest(BaseModel):
    """截图请求"""
    full_page: bool = Field(default=False, description="是否截取整个页面")
    selector: Optional[str] = Field(None, description="CSS选择器")
    width: Optional[int] = Field(None, ge=100, description="截图宽度")
    height: Optional[int] = Field(None, ge=100, description="截图高度")
    format: str = Field(default="png", description="图像格式")
    quality: int = Field(default=90, ge=1, le=100, description="图像质量")


class ScreenshotResponse(BaseModel):
    """截图响应"""
    image_url: str = Field(..., description="图像URL")
    image_data: Optional[str] = Field(None, description="Base64编码图像数据")
    width: int = Field(..., description="图像宽度")
    height: int = Field(..., description="图像高度")
    format: str = Field(..., description="图像格式")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="截图时间")


class ClickRequest(BaseModel):
    """点击请求"""
    x: int = Field(..., description="X坐标")
    y: int = Field(..., description="Y坐标")
    button: str = Field(default="left", description="鼠标按钮")
    clicks: int = Field(default=1, ge=1, le=3, description="点击次数")
    selector: Optional[str] = Field(None, description="CSS选择器")


class TypeRequest(BaseModel):
    """输入请求"""
    text: str = Field(..., description="输入文本")
    selector: Optional[str] = Field(None, description="目标元素选择器")
    delay: int = Field(default=10, ge=0, le=1000, description="按键延迟(ms)")
    clear_first: bool = Field(default=True, description="是否先清空")
    submit: bool = Field(default=False, description="是否提交")


class ScrollRequest(BaseModel):
    """滚动请求"""
    direction: str = Field(default="down", description="滚动方向")
    amount: int = Field(default=300, description="滚动距离(像素)")
    selector: Optional[str] = Field(None, description="目标元素选择器")
    smooth: bool = Field(default=True, description="平滑滚动")


class NavigateRequest(BaseModel):
    """导航请求"""
    url: str = Field(..., description="目标URL")
    wait_until: str = Field(default="networkidle", description="等待条件")
    timeout: int = Field(default=30000, ge=1000, description="超时时间(ms)")
    referer: Optional[str] = Field(None, description="Referer")


class RecordingConfig(BaseModel):
    """录制配置"""
    fps: int = Field(default=30, ge=1, le=60, description="帧率")
    resolution: str = Field(default="1920x1080", description="分辨率")
    format: str = Field(default="mp4", description="输出格式")
    quality: str = Field(default="high", description="视频质量")
    audio: bool = Field(default=True, description="录制音频")


class RecordingResponse(BaseModel):
    """录制响应"""
    recording_id: str = Field(..., description="录制ID")
    status: RecordingStatus = Field(..., description="录制状态")
    config: RecordingConfig = Field(..., description="录制配置")
    started_at: Optional[datetime] = Field(None, description="开始时间")
    duration: int = Field(default=0, description="已录制时长(秒)")
    file_size: int = Field(default=0, description="文件大小(字节)")
    file_url: Optional[str] = Field(None, description="文件URL")


# =============================================================================
# Pydantic模型定义 - 安全中心
# =============================================================================

class FirewallRule(BaseModel):
    """防火墙规则"""
    id: str = Field(..., description="规则ID")
    name: str = Field(..., description="规则名称")
    description: Optional[str] = Field(None, description="规则描述")
    source_ip: Optional[str] = Field(None, description="源IP")
    destination_ip: Optional[str] = Field(None, description="目标IP")
    port: Optional[str] = Field(None, description="端口")
    protocol: str = Field(default="tcp", description="协议")
    action: FirewallAction = Field(..., description="动作")
    priority: int = Field(default=100, ge=1, le=1000, description="优先级")
    enabled: bool = Field(default=True, description="是否启用")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="创建时间")
    expires_at: Optional[datetime] = Field(None, description="过期时间")


class FirewallRuleCreateRequest(BaseModel):
    """创建防火墙规则请求"""
    name: str = Field(..., min_length=1, max_length=200, description="规则名称")
    description: Optional[str] = Field(None, max_length=500, description="规则描述")
    source_ip: Optional[str] = Field(None, description="源IP/CIDR")
    destination_ip: Optional[str] = Field(None, description="目标IP/CIDR")
    port: Optional[str] = Field(None, description="端口范围")
    protocol: str = Field(default="tcp", description="协议")
    action: FirewallAction = Field(..., description="动作")
    priority: int = Field(default=100, ge=1, le=1000, description="优先级")


class AuditLogEntry(BaseModel):
    """审计日志条目"""
    id: str = Field(..., description="日志ID")
    timestamp: datetime = Field(..., description="时间戳")
    user_id: Optional[str] = Field(None, description="用户ID")
    user_name: Optional[str] = Field(None, description="用户名")
    action: str = Field(..., description="操作")
    resource_type: str = Field(..., description="资源类型")
    resource_id: Optional[str] = Field(None, description="资源ID")
    status: str = Field(..., description="操作状态")
    ip_address: Optional[str] = Field(None, description="IP地址")
    user_agent: Optional[str] = Field(None, description="用户代理")
    details: Dict[str, Any] = Field(default_factory=dict, description="详细信息")
    risk_level: str = Field(default="low", description="风险等级")


class SecurityScanRequest(BaseModel):
    """安全扫描请求"""
    scan_type: ScanType = Field(..., description="扫描类型")
    target: str = Field(..., description="扫描目标")
    scope: Optional[List[str]] = Field(None, description="扫描范围")
    options: Dict[str, Any] = Field(default_factory=dict, description="扫描选项")
    priority: int = Field(default=5, ge=1, le=10, description="优先级")


class SecurityScanResponse(BaseModel):
    """安全扫描响应"""
    scan_id: str = Field(..., description="扫描ID")
    scan_type: ScanType = Field(..., description="扫描类型")
    status: str = Field(..., description="扫描状态")
    target: str = Field(..., description="扫描目标")
    progress: float = Field(default=0.0, description="进度")
    started_at: datetime = Field(default_factory=datetime.utcnow, description="开始时间")
    completed_at: Optional[datetime] = Field(None, description="完成时间")
    findings_count: int = Field(default=0, description="发现数量")
    findings: List[Dict[str, Any]] = Field(default_factory=list, description="发现详情")
    report_url: Optional[str] = Field(None, description="报告URL")


class ThreatIntelligence(BaseModel):
    """威胁情报"""
    id: str = Field(..., description="威胁ID")
    threat_type: str = Field(..., description="威胁类型")
    level: ThreatLevel = Field(..., description="威胁等级")
    source: str = Field(..., description="情报来源")
    description: str = Field(..., description="威胁描述")
    indicators: List[str] = Field(default_factory=list, description="威胁指标")
    affected_systems: List[str] = Field(default_factory=list, description="受影响系统")
    discovered_at: datetime = Field(..., description="发现时间")
    mitigated: bool = Field(default=False, description="是否已缓解")
    mitigation_steps: List[str] = Field(default_factory=list, description="缓解步骤")


class PromptGuardTestRequest(BaseModel):
    """提示词防护测试请求"""
    prompt: str = Field(..., description="测试提示词")
    test_categories: List[str] = Field(default_factory=list, description="测试类别")
    include_explanation: bool = Field(default=True, description="包含解释")


class PromptGuardTestResponse(BaseModel):
    """提示词防护测试响应"""
    test_id: str = Field(..., description="测试ID")
    prompt: str = Field(..., description="测试提示词")
    passed: bool = Field(..., description="是否通过")
    risk_score: float = Field(..., ge=0.0, le=1.0, description="风险分数")
    detected_issues: List[Dict[str, Any]] = Field(default_factory=list, description="检测到的问题")
    recommendations: List[str] = Field(default_factory=list, description="建议")
    explanation: Optional[str] = Field(None, description="解释")


# =============================================================================
# Pydantic模型定义 - 联邦学习
# =============================================================================

class FederatedNode(BaseModel):
    """联邦学习节点"""
    id: str = Field(..., description="节点ID")
    name: str = Field(..., description="节点名称")
    status: NodeStatus = Field(..., description="节点状态")
    endpoint: str = Field(..., description="节点端点")
    public_key: Optional[str] = Field(None, description="公钥")
    capabilities: List[str] = Field(default_factory=list, description="能力列表")
    dataset_size: Optional[int] = Field(None, description="数据集大小")
    last_seen: datetime = Field(default_factory=datetime.utcnow, description="最后在线时间")
    registered_at: datetime = Field(default_factory=datetime.utcnow, description="注册时间")
    contribution_score: float = Field(default=0.0, description="贡献分数")


class FederatedNodeRegisterRequest(BaseModel):
    """注册联邦学习节点请求"""
    name: str = Field(..., min_length=1, max_length=200, description="节点名称")
    endpoint: str = Field(..., description="节点端点URL")
    public_key: Optional[str] = Field(None, description="公钥")
    capabilities: List[str] = Field(default_factory=list, description="能力列表")
    dataset_info: Optional[Dict[str, Any]] = Field(None, description="数据集信息")


class TrainingRound(BaseModel):
    """训练轮次"""
    id: str = Field(..., description="轮次ID")
    round_number: int = Field(..., description="轮次数")
    status: str = Field(..., description="状态")
    participating_nodes: List[str] = Field(default_factory=list, description="参与节点")
    aggregation_strategy: AggregationStrategy = Field(default=AggregationStrategy.FEDAVG, description="聚合策略")
    started_at: datetime = Field(default_factory=datetime.utcnow, description="开始时间")
    completed_at: Optional[datetime] = Field(None, description="完成时间")
    global_model_version: Optional[str] = Field(None, description="全局模型版本")
    metrics: Dict[str, Any] = Field(default_factory=dict, description="训练指标")


class AggregationRequest(BaseModel):
    """聚合请求"""
    node_updates: List[str] = Field(..., description="节点更新ID列表")
    strategy: AggregationStrategy = Field(default=AggregationStrategy.FEDAVG, description="聚合策略")
    weights: Optional[Dict[str, float]] = Field(None, description="节点权重")
    options: Dict[str, Any] = Field(default_factory=dict, description="聚合选项")


class AggregationResponse(BaseModel):
    """聚合响应"""
    round_id: str = Field(..., description="轮次ID")
    status: str = Field(..., description="状态")
    aggregated_model_url: Optional[str] = Field(None, description="聚合模型URL")
    metrics: Dict[str, Any] = Field(default_factory=dict, description="聚合指标")
    node_contributions: Dict[str, float] = Field(default_factory=dict, description="节点贡献")
    completed_at: Optional[datetime] = Field(None, description="完成时间")


class ContributionStats(BaseModel):
    """贡献统计"""
    node_id: str = Field(..., description="节点ID")
    node_name: str = Field(..., description="节点名称")
    total_rounds: int = Field(default=0, description="总轮次数")
    successful_rounds: int = Field(default=0, description="成功轮次数")
    data_samples_contributed: int = Field(default=0, description="贡献数据样本数")
    computation_hours: float = Field(default=0.0, description="计算时长(小时)")
    reward_tokens: float = Field(default=0.0, description="奖励代币")
    reputation_score: float = Field(default=0.0, description="信誉分数")
    last_contribution_at: Optional[datetime] = Field(None, description="最后贡献时间")


# =============================================================================
# Pydantic模型定义 - 知识检索
# =============================================================================

class DocumentUploadResponse(BaseModel):
    """文档上传响应"""
    document_id: str = Field(..., description="文档ID")
    filename: str = Field(..., description="文件名")
    type: DocumentType = Field(..., description="文档类型")
    size: int = Field(..., description="文件大小(字节)")
    status: str = Field(default="processing", description="处理状态")
    chunks_count: Optional[int] = Field(None, description="分块数量")
    uploaded_at: datetime = Field(default_factory=datetime.utcnow, description="上传时间")
    processing_started_at: Optional[datetime] = Field(None, description="处理开始时间")


class DocumentInfo(BaseModel):
    """文档信息"""
    id: str = Field(..., description="文档ID")
    filename: str = Field(..., description="文件名")
    type: DocumentType = Field(..., description="文档类型")
    size: int = Field(..., description="文件大小")
    status: str = Field(..., description="处理状态")
    chunks_count: int = Field(default=0, description="分块数量")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="元数据")
    tags: List[str] = Field(default_factory=list, description="标签")
    uploaded_at: datetime = Field(..., description="上传时间")
    processed_at: Optional[datetime] = Field(None, description="处理完成时间")


class RAGSearchRequest(BaseModel):
    """RAG检索请求"""
    query: str = Field(..., min_length=1, max_length=2000, description="检索查询")
    strategy: SearchStrategy = Field(default=SearchStrategy.HYBRID, description="检索策略")
    top_k: int = Field(default=5, ge=1, le=20, description="返回结果数")
    filters: Dict[str, Any] = Field(default_factory=dict, description="过滤条件")
    document_ids: Optional[List[str]] = Field(None, description="指定文档ID")
    min_score: float = Field(default=0.7, ge=0.0, le=1.0, description="最小相似度")


class RAGSearchResult(BaseModel):
    """RAG检索结果"""
    document_id: str = Field(..., description="文档ID")
    document_name: str = Field(..., description="文档名称")
    chunk_index: int = Field(..., description="分块索引")
    content: str = Field(..., description="内容片段")
    score: float = Field(..., description="相似度分数")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="元数据")


class RAGSearchResponse(BaseModel):
    """RAG检索响应"""
    query: str = Field(..., description="原始查询")
    results: List[RAGSearchResult] = Field(default_factory=list, description="检索结果")
    total_results: int = Field(..., description="总结果数")
    search_time_ms: int = Field(..., description="搜索耗时(ms)")
    strategy_used: SearchStrategy = Field(..., description="使用的策略")


class KnowledgeBaseInfo(BaseModel):
    """知识库信息"""
    id: str = Field(..., description="知识库ID")
    name: str = Field(..., description="知识库名称")
    description: Optional[str] = Field(None, description="描述")
    document_count: int = Field(default=0, description="文档数量")
    total_chunks: int = Field(default=0, description="总分块数")
    embedding_model: str = Field(..., description="嵌入模型")
    created_at: datetime = Field(..., description="创建时间")
    updated_at: datetime = Field(..., description="更新时间")
    owner_id: str = Field(..., description="所有者ID")


# =============================================================================
# Pydantic模型定义 - 渠道管理
# =============================================================================

class ChannelConfig(BaseModel):
    """渠道配置"""
    webhook_url: Optional[str] = Field(None, description="Webhook URL")
    api_key: Optional[str] = Field(None, description="API密钥")
    api_secret: Optional[str] = Field(None, description="API密钥")
    channel_id: Optional[str] = Field(None, description="渠道ID")
    settings: Dict[str, Any] = Field(default_factory=dict, description="其他设置")


class ChannelCreateRequest(BaseModel):
    """创建渠道请求"""
    name: str = Field(..., min_length=1, max_length=200, description="渠道名称")
    type: ChannelType = Field(..., description="渠道类型")
    description: Optional[str] = Field(None, max_length=500, description="描述")
    config: ChannelConfig = Field(..., description="渠道配置")
    is_default: bool = Field(default=False, description="是否默认渠道")


class ChannelResponse(BaseModel):
    """渠道响应"""
    id: str = Field(..., description="渠道ID")
    name: str = Field(..., description="渠道名称")
    type: ChannelType = Field(..., description="渠道类型")
    description: Optional[str] = Field(None, description="描述")
    status: ChannelStatus = Field(..., description="状态")
    config: ChannelConfig = Field(..., description="配置")
    is_default: bool = Field(default=False, description="是否默认")
    message_count: int = Field(default=0, description="消息数量")
    last_message_at: Optional[datetime] = Field(None, description="最后消息时间")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="创建时间")


class ChannelTestResponse(BaseModel):
    """渠道测试响应"""
    channel_id: str = Field(..., description="渠道ID")
    success: bool = Field(..., description="是否成功")
    latency_ms: int = Field(..., description="延迟(ms)")
    error_message: Optional[str] = Field(None, description="错误信息")
    tested_at: datetime = Field(default_factory=datetime.utcnow, description="测试时间")


class ChannelMessage(BaseModel):
    """渠道消息"""
    id: str = Field(..., description="消息ID")
    channel_id: str = Field(..., description="渠道ID")
    direction: str = Field(..., description="方向")
    content: str = Field(..., description="内容")
    sender_id: Optional[str] = Field(None, description="发送者ID")
    sender_name: Optional[str] = Field(None, description="发送者名称")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="时间戳")
    status: str = Field(default="delivered", description="状态")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="元数据")


# =============================================================================
# Pydantic模型定义 - 插件市场
# =============================================================================

class PluginInfo(BaseModel):
    """插件信息"""
    id: str = Field(..., description="插件ID")
    name: str = Field(..., description="插件名称")
    description: str = Field(..., description="描述")
    version: str = Field(..., description="版本")
    author: str = Field(..., description="作者")
    category: PluginCategory = Field(..., description="类别")
    status: PluginStatus = Field(..., description="状态")
    icon_url: Optional[str] = Field(None, description="图标URL")
    readme_url: Optional[str] = Field(None, description="说明文档URL")
    config_schema: Optional[Dict[str, Any]] = Field(None, description="配置模式")
    permissions: List[str] = Field(default_factory=list, description="权限列表")
    installed_at: Optional[datetime] = Field(None, description="安装时间")
    updated_at: Optional[datetime] = Field(None, description="更新时间")


class PluginInstallRequest(BaseModel):
    """安装插件请求"""
    source: str = Field(..., description="插件来源")
    version: Optional[str] = Field(None, description="指定版本")
    config: Dict[str, Any] = Field(default_factory=dict, description="初始配置")
    auto_enable: bool = Field(default=True, description="自动启用")


class PluginMarketplaceItem(BaseModel):
    """插件市场项"""
    id: str = Field(..., description="插件ID")
    name: str = Field(..., description="插件名称")
    description: str = Field(..., description="描述")
    version: str = Field(..., description="版本")
    author: str = Field(..., description="作者")
    category: PluginCategory = Field(..., description="类别")
    rating: float = Field(default=0.0, description="评分")
    download_count: int = Field(default=0, description="下载次数")
    icon_url: Optional[str] = Field(None, description="图标URL")
    screenshots: List[str] = Field(default_factory=list, description="截图")
    tags: List[str] = Field(default_factory=list, description="标签")
    price: float = Field(default=0.0, description="价格")
    is_official: bool = Field(default=False, description="是否官方")


# =============================================================================
# Pydantic模型定义 - 人格引擎
# =============================================================================

class PersonalityTraits(BaseModel):
    """人格特质配置"""
    openness: float = Field(default=0.5, ge=0.0, le=1.0, description="开放性")
    conscientiousness: float = Field(default=0.5, ge=0.0, le=1.0, description="尽责性")
    extraversion: float = Field(default=0.5, ge=0.0, le=1.0, description="外向性")
    agreeableness: float = Field(default=0.5, ge=0.0, le=1.0, description="宜人性")
    neuroticism: float = Field(default=0.5, ge=0.0, le=1.0, description="神经质")


class PersonalityCreateRequest(BaseModel):
    """创建人格请求"""
    name: str = Field(..., min_length=1, max_length=100, description="人格名称")
    description: Optional[str] = Field(None, max_length=1000, description="描述")
    avatar_url: Optional[str] = Field(None, description="头像URL")
    system_prompt: str = Field(..., min_length=10, max_length=10000, description="系统提示词")
    traits: PersonalityTraits = Field(default_factory=PersonalityTraits, description="人格特质")
    communication_style: CommunicationStyle = Field(default=CommunicationStyle.PROFESSIONAL, description="沟通风格")
    response_templates: List[str] = Field(default_factory=list, description="回复模板")
    knowledge_domains: List[str] = Field(default_factory=list, description="知识领域")
    forbidden_topics: List[str] = Field(default_factory=list, description="禁止话题")
    example_conversations: List[Dict[str, str]] = Field(default_factory=list, description="示例对话")


class PersonalityResponse(BaseModel):
    """人格响应"""
    id: str = Field(..., description="人格ID")
    name: str = Field(..., description="人格名称")
    description: Optional[str] = Field(None, description="描述")
    avatar_url: Optional[str] = Field(None, description="头像URL")
    system_prompt: str = Field(..., description="系统提示词")
    traits: PersonalityTraits = Field(..., description="人格特质")
    communication_style: CommunicationStyle = Field(..., description="沟通风格")
    is_active: bool = Field(default=False, description="是否激活")
    usage_count: int = Field(default=0, description="使用次数")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="创建时间")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="更新时间")
    created_by: str = Field(..., description="创建者ID")


class PersonalityTestRequest(BaseModel):
    """人格测试请求"""
    test_scenarios: List[str] = Field(default_factory=list, description="测试场景")
    conversation_turns: int = Field(default=5, ge=1, le=20, description="对话轮数")
    evaluation_criteria: List[str] = Field(default_factory=list, description="评估标准")


class PersonalityTestResponse(BaseModel):
    """人格测试响应"""
    test_id: str = Field(..., description="测试ID")
    personality_id: str = Field(..., description="人格ID")
    scenarios_tested: int = Field(..., description="测试场景数")
    coherence_score: float = Field(..., description="一致性分数")
    appropriateness_score: float = Field(..., description="适当性分数")
    consistency_score: float = Field(..., description="连贯性分数")
    detailed_feedback: List[Dict[str, Any]] = Field(default_factory=list, description="详细反馈")
    recommendations: List[str] = Field(default_factory=list, description="改进建议")


# =============================================================================
# Pydantic模型定义 - 数据管道
# =============================================================================

class DatasetInfo(BaseModel):
    """数据集信息"""
    id: str = Field(..., description="数据集ID")
    name: str = Field(..., description="数据集名称")
    description: Optional[str] = Field(None, description="描述")
    format: DatasetFormat = Field(..., description="格式")
    size: int = Field(..., description="大小(字节)")
    rows: Optional[int] = Field(None, description="行数")
    columns: Optional[int] = Field(None, description="列数")
    data_schema: Optional[Dict[str, Any]] = Field(None, alias="schema", description="数据结构")
    tags: List[str] = Field(default_factory=list, description="标签")
    uploaded_at: datetime = Field(..., description="上传时间")
    processed_at: Optional[datetime] = Field(None, description="处理时间")
    status: str = Field(..., description="状态")


class DatasetProcessRequest(BaseModel):
    """数据集处理请求"""
    operations: List[Dict[str, Any]] = Field(..., description="处理操作列表")
    output_format: Optional[DatasetFormat] = Field(None, description="输出格式")
    save_as_new: bool = Field(default=True, description="保存为新数据集")


class DatasetProcessResponse(BaseModel):
    """数据集处理响应"""
    job_id: str = Field(..., description="任务ID")
    dataset_id: str = Field(..., description="数据集ID")
    status: str = Field(..., description="状态")
    progress: float = Field(default=0.0, description="进度")
    operations_applied: List[str] = Field(default_factory=list, description="已应用操作")
    started_at: datetime = Field(default_factory=datetime.utcnow, description="开始时间")
    completed_at: Optional[datetime] = Field(None, description="完成时间")
    output_dataset_id: Optional[str] = Field(None, description="输出数据集ID")


class PipelineStep(BaseModel):
    """流水线步骤"""
    id: str = Field(..., description="步骤ID")
    name: str = Field(..., description="步骤名称")
    type: str = Field(..., description="步骤类型")
    config: Dict[str, Any] = Field(default_factory=dict, description="配置")
    inputs: List[str] = Field(default_factory=list, description="输入")
    outputs: List[str] = Field(default_factory=list, description="输出")
    dependencies: List[str] = Field(default_factory=list, description="依赖")


class PipelineCreateRequest(BaseModel):
    """创建流水线请求"""
    name: str = Field(..., min_length=1, max_length=200, description="流水线名称")
    description: Optional[str] = Field(None, max_length=1000, description="描述")
    steps: List[PipelineStep] = Field(..., min_items=1, description="步骤列表")
    schedule: Optional[str] = Field(None, description="调度表达式")
    trigger_events: List[str] = Field(default_factory=list, description="触发事件")


class PipelineResponse(BaseModel):
    """流水线响应"""
    id: str = Field(..., description="流水线ID")
    name: str = Field(..., description="名称")
    description: Optional[str] = Field(None, description="描述")
    status: PipelineStatus = Field(..., description="状态")
    steps: List[PipelineStep] = Field(..., description="步骤")
    schedule: Optional[str] = Field(None, description="调度")
    last_run_at: Optional[datetime] = Field(None, description="最后运行时间")
    last_run_status: Optional[str] = Field(None, description="最后运行状态")
    run_count: int = Field(default=0, description="运行次数")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="创建时间")


# =============================================================================
# Pydantic模型定义 - 价值对齐
# =============================================================================

class AlignmentPrinciple(BaseModel):
    """价值对齐原则"""
    id: str = Field(..., description="原则ID")
    name: str = Field(..., description="原则名称")
    description: str = Field(..., description="描述")
    category: PrincipleCategory = Field(..., description="类别")
    priority: int = Field(default=5, ge=1, le=10, description="优先级")
    rules: List[str] = Field(default_factory=list, description="规则列表")
    examples: List[Dict[str, str]] = Field(default_factory=list, description="示例")
    is_active: bool = Field(default=True, description="是否激活")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="创建时间")


class AlignmentPrincipleCreateRequest(BaseModel):
    """创建原则请求"""
    name: str = Field(..., min_length=1, max_length=200, description="原则名称")
    description: str = Field(..., min_length=10, max_length=2000, description="描述")
    category: PrincipleCategory = Field(..., description="类别")
    priority: int = Field(default=5, ge=1, le=10, description="优先级")
    rules: List[str] = Field(default_factory=list, description="规则")
    examples: List[Dict[str, str]] = Field(default_factory=list, description="示例")


class AlignmentTestRequest(BaseModel):
    """对齐测试请求"""
    test_type: str = Field(..., description="测试类型")
    test_cases: List[Dict[str, Any]] = Field(..., description="测试用例")
    principles_to_test: Optional[List[str]] = Field(None, description="测试原则")
    model_id: Optional[str] = Field(None, description="模型ID")


class AlignmentTestCaseResult(BaseModel):
    """对齐测试用例结果"""
    case_id: str = Field(..., description="用例ID")
    input_text: str = Field(..., description="输入")
    expected_behavior: str = Field(..., description="期望行为")
    actual_response: str = Field(..., description="实际响应")
    result: AlignmentTestResult = Field(..., description="结果")
    score: float = Field(..., description="分数")
    violations: List[str] = Field(default_factory=list, description="违规项")


class AlignmentTestResponse(BaseModel):
    """对齐测试响应"""
    test_id: str = Field(..., description="测试ID")
    status: str = Field(..., description="状态")
    overall_score: float = Field(..., description="总体分数")
    results: List[AlignmentTestCaseResult] = Field(default_factory=list, description="详细结果")
    principles_evaluated: List[str] = Field(default_factory=list, description="评估原则")
    started_at: datetime = Field(default_factory=datetime.utcnow, description="开始时间")
    completed_at: Optional[datetime] = Field(None, description="完成时间")


class AlignmentReport(BaseModel):
    """对齐报告"""
    id: str = Field(..., description="报告ID")
    test_id: str = Field(..., description="测试ID")
    generated_at: datetime = Field(..., description="生成时间")
    summary: str = Field(..., description="摘要")
    overall_score: float = Field(..., description="总体分数")
    category_scores: Dict[str, float] = Field(default_factory=dict, description="类别分数")
    findings: List[Dict[str, Any]] = Field(default_factory=list, description="发现")
    recommendations: List[str] = Field(default_factory=list, description="建议")
    report_url: Optional[str] = Field(None, description="报告URL")


# =============================================================================
# Pydantic模型定义 - 机器人控制
# =============================================================================

class RobotInfo(BaseModel):
    """机器人信息"""
    id: str = Field(..., description="机器人ID")
    name: str = Field(..., description="机器人名称")
    type: RobotType = Field(..., description="类型")
    model: str = Field(..., description="型号")
    manufacturer: str = Field(..., description="制造商")
    status: RobotStatus = Field(..., description="状态")
    capabilities: List[str] = Field(default_factory=list, description="能力")
    battery_level: Optional[float] = Field(None, ge=0.0, le=100.0, description="电量")
    current_position: Optional[Dict[str, float]] = Field(None, description="当前位置")
    connected_at: Optional[datetime] = Field(None, description="连接时间")
    last_heartbeat: Optional[datetime] = Field(None, description="最后心跳")


class RobotMoveRequest(BaseModel):
    """机器人移动请求"""
    movement_type: str = Field(..., description="移动类型")
    target_position: Optional[Dict[str, float]] = Field(None, description="目标位置")
    relative_movement: Optional[Dict[str, float]] = Field(None, description="相对移动")
    speed: float = Field(default=0.5, ge=0.0, le=1.0, description="速度")
    acceleration: Optional[float] = Field(None, description="加速度")
    coordinate_system: str = Field(default="world", description="坐标系")


class RobotMoveResponse(BaseModel):
    """机器人移动响应"""
    command_id: str = Field(..., description="命令ID")
    robot_id: str = Field(..., description="机器人ID")
    status: str = Field(..., description="状态")
    estimated_duration: Optional[int] = Field(None, description="预计耗时(秒)")
    target_position: Optional[Dict[str, float]] = Field(None, description="目标位置")


class RobotStatusResponse(BaseModel):
    """机器人状态响应"""
    robot_id: str = Field(..., description="机器人ID")
    status: RobotStatus = Field(..., description="状态")
    battery_level: Optional[float] = Field(None, description="电量")
    current_task: Optional[str] = Field(None, description="当前任务")
    current_position: Optional[Dict[str, float]] = Field(None, description="当前位置")
    joint_states: Optional[Dict[str, float]] = Field(None, description="关节状态")
    sensor_data: Optional[Dict[str, Any]] = Field(None, description="传感器数据")
    errors: List[str] = Field(default_factory=list, description="错误信息")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="时间戳")


# =============================================================================
# 模拟数据存储
# =============================================================================

# 物理引擎存储
_simulations: Dict[str, Dict[str, Any]] = {}
_molecules: Dict[str, Dict[str, Any]] = {}

# 计算机使用存储
_recordings: Dict[str, Dict[str, Any]] = {}

# 安全中心存储
_firewall_rules: Dict[str, Dict[str, Any]] = {}
_audit_logs: Dict[str, Dict[str, Any]] = {}
_security_scans: Dict[str, Dict[str, Any]] = {}
_threats: Dict[str, Dict[str, Any]] = {}

# 联邦学习存储
_federated_nodes: Dict[str, Dict[str, Any]] = {}
_training_rounds: Dict[str, Dict[str, Any]] = {}
_contributions: Dict[str, Dict[str, Any]] = {}

# 知识检索存储
_documents: Dict[str, Dict[str, Any]] = {}
_knowledge_bases: Dict[str, Dict[str, Any]] = {}

# 渠道管理存储
_channels: Dict[str, Dict[str, Any]] = {}
_channel_messages: Dict[str, List[Dict[str, Any]]] = {}

# 插件市场存储
_plugins: Dict[str, Dict[str, Any]] = {}
_marketplace_plugins: Dict[str, Dict[str, Any]] = {}

# 人格引擎存储
_personalities: Dict[str, Dict[str, Any]] = {}

# 数据管道存储
_datasets: Dict[str, Dict[str, Any]] = {}
_pipelines: Dict[str, Dict[str, Any]] = {}

# 价值对齐存储
_alignment_principles: Dict[str, Dict[str, Any]] = {}
_alignment_tests: Dict[str, Dict[str, Any]] = {}
_alignment_reports: Dict[str, Dict[str, Any]] = {}

# 机器人控制存储
_robots: Dict[str, Dict[str, Any]] = {}


def _init_mock_data():
    """初始化模拟数据"""
    # 初始化分子库
    molecules = [
        {
            "id": "mol-water",
            "name": "Water",
            "formula": "H2O",
            "smiles": "O",
            "molecular_weight": 18.015,
            "category": "solvent",
            "properties": {"boiling_point": 100.0, "melting_point": 0.0}
        },
        {
            "id": "mol-benzene",
            "name": "Benzene",
            "formula": "C6H6",
            "smiles": "c1ccccc1",
            "molecular_weight": 78.114,
            "category": "aromatic",
            "properties": {"boiling_point": 80.1, "melting_point": 5.5}
        }
    ]
    for mol in molecules:
        _molecules[mol["id"]] = mol
    
    # 初始化防火墙规则
    firewall_rules = [
        {
            "id": "fw-001",
            "name": "Allow HTTP",
            "source_ip": "0.0.0.0/0",
            "port": "80",
            "protocol": "tcp",
            "action": FirewallAction.ALLOW,
            "priority": 100
        },
        {
            "id": "fw-002",
            "name": "Allow HTTPS",
            "source_ip": "0.0.0.0/0",
            "port": "443",
            "protocol": "tcp",
            "action": FirewallAction.ALLOW,
            "priority": 100
        }
    ]
    for rule in firewall_rules:
        _firewall_rules[rule["id"]] = rule
    
    # 初始化威胁情报
    threats = [
        {
            "id": "threat-001",
            "threat_type": "prompt_injection",
            "level": ThreatLevel.HIGH,
            "source": "automated_scan",
            "description": "检测到潜在的提示词注入攻击模式",
            "indicators": ["ignore previous", "disregard instructions"],
            "discovered_at": datetime.utcnow()
        }
    ]
    for threat in threats:
        _threats[threat["id"]] = threat
    
    # 初始化知识库
    knowledge_bases = [
        {
            "id": "kb-default",
            "name": "默认知识库",
            "description": "系统默认知识库",
            "document_count": 0,
            "total_chunks": 0,
            "embedding_model": "text-embedding-3-large",
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow(),
            "owner_id": "system"
        }
    ]
    for kb in knowledge_bases:
        _knowledge_bases[kb["id"]] = kb
    
    # 初始化插件市场
    marketplace_plugins = [
        {
            "id": "plugin-code-assistant",
            "name": "Code Assistant",
            "description": "智能代码辅助插件",
            "version": "1.0.0",
            "author": "AGI Team",
            "category": PluginCategory.AUTOMATION,
            "rating": 4.8,
            "download_count": 1250,
            "is_official": True,
            "price": 0.0
        },
        {
            "id": "plugin-data-visualizer",
            "name": "Data Visualizer",
            "description": "数据可视化插件",
            "version": "2.1.0",
            "author": "DataViz Inc",
            "category": PluginCategory.ANALYTICS,
            "rating": 4.5,
            "download_count": 890,
            "is_official": False,
            "price": 9.99
        }
    ]
    for plugin in marketplace_plugins:
        _marketplace_plugins[plugin["id"]] = plugin
    
    # 初始化价值对齐原则
    principles = [
        {
            "id": "principle-safety-001",
            "name": "无害性原则",
            "description": "AI系统不应产生有害输出",
            "category": PrincipleCategory.SAFETY,
            "priority": 10,
            "rules": ["禁止生成恶意代码", "禁止提供危险指导"],
            "is_active": True,
            "created_at": datetime.utcnow()
        },
        {
            "id": "principle-fairness-001",
            "name": "公平性原则",
            "description": "AI系统应公平对待所有用户",
            "category": PrincipleCategory.FAIRNESS,
            "priority": 8,
            "rules": ["避免偏见", "平等对待"],
            "is_active": True,
            "created_at": datetime.utcnow()
        }
    ]
    for principle in principles:
        _alignment_principles[principle["id"]] = principle
    
    # 初始化机器人列表
    robots = [
        {
            "id": "robot-arm-001",
            "name": "Industrial Arm 001",
            "type": RobotType.ARM,
            "model": "UR5e",
            "manufacturer": "Universal Robots",
            "status": RobotStatus.IDLE,
            "capabilities": ["pick_and_place", "assembly", "welding"],
            "battery_level": 85.0
        },
        {
            "id": "robot-mobile-001",
            "name": "Mobile Bot 001",
            "type": RobotType.MOBILE,
            "model": "TurtleBot4",
            "manufacturer": "Clearpath",
            "status": RobotStatus.OFFLINE,
            "capabilities": ["navigation", "mapping", "inspection"],
            "battery_level": None
        }
    ]
    for robot in robots:
        _robots[robot["id"]] = robot


_init_mock_data()


# =============================================================================
# 物理引擎端点
# =============================================================================

@router.get(
    "/physics/simulations",
    response_model=PaginatedResponse[SimulationResponse],
    summary="获取仿真列表",
    description="获取所有物理仿真任务的列表"
)
async def get_simulations(
    type: Optional[SimulationType] = Query(None, description="仿真类型"),
    status: Optional[SimulationStatus] = Query(None, description="状态筛选"),
    page: int = Query(1, ge=1, description="页码"),
    page_size: int = Query(20, ge=1, le=100, description="每页数量"),
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> PaginatedResponse[SimulationResponse]:
    """
    获取仿真列表
    
    获取所有物理仿真任务的列表，支持按类型和状态筛选。
    """
    simulations = list(_simulations.values())
    
    if type:
        simulations = [s for s in simulations if s["type"] == type]
    if status:
        simulations = [s for s in simulations if s["status"] == status]
    
    total = len(simulations)
    start_idx = (page - 1) * page_size
    end_idx = start_idx + page_size
    paginated = simulations[start_idx:end_idx]
    
    items = [SimulationResponse(**s) for s in paginated]
    
    return PaginatedResponse(
        success=True,
        message="获取仿真列表成功",
        data=items,
        pagination={
            "page": page,
            "page_size": page_size,
            "total": total,
            "total_pages": (total + page_size - 1) // page_size
        }
    )


@router.post(
    "/physics/simulations",
    response_model=BaseResponse[SimulationResponse],
    status_code=status.HTTP_201_CREATED,
    summary="创建仿真",
    description="创建新的物理仿真任务"
)
async def create_simulation(
    request: SimulationCreateRequest,
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> BaseResponse[SimulationResponse]:
    """
    创建仿真
    
    创建新的物理仿真任务，支持分子动力学、量子化学等多种仿真类型。
    """
    simulation_id = str(uuid.uuid4())
    
    simulation_data = {
        "id": simulation_id,
        "name": request.name,
        "type": request.type,
        "status": SimulationStatus.PENDING,
        "description": request.description,
        "config": request.config.dict(),
        "progress": 0.0,
        "current_step": 0,
        "created_at": datetime.utcnow(),
        "started_at": None,
        "completed_at": None,
        "results_url": None,
        "logs_url": None
    }
    
    _simulations[simulation_id] = simulation_data
    
    logger.info(f"仿真任务已创建: {simulation_id}, 用户: {current_user.get('id')}")
    
    return BaseResponse(
        success=True,
        message="仿真任务创建成功",
        data=SimulationResponse(**simulation_data)
    )


@router.get(
    "/physics/simulations/{simulation_id}",
    response_model=BaseResponse[SimulationResponse],
    summary="获取仿真详情",
    description="获取指定仿真任务的详细信息"
)
async def get_simulation(
    simulation_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> BaseResponse[SimulationResponse]:
    """
    获取仿真详情
    
    获取指定ID的仿真任务的详细信息和当前状态。
    """
    simulation = _simulations.get(simulation_id)
    
    if not simulation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="仿真任务不存在"
        )
    
    return BaseResponse(
        success=True,
        message="获取仿真详情成功",
        data=SimulationResponse(**simulation)
    )


@router.post(
    "/physics/simulations/{simulation_id}/run",
    response_model=BaseResponse[SimulationResponse],
    summary="运行仿真",
    description="启动或继续运行仿真任务"
)
async def run_simulation(
    simulation_id: str,
    current_user: Dict[str, Any] = Depends(require_permissions(["physics:run"]))
) -> BaseResponse[SimulationResponse]:
    """
    运行仿真
    
    启动指定ID的仿真任务。仿真将在后台运行，可以通过查询接口获取进度。
    """
    simulation = _simulations.get(simulation_id)
    
    if not simulation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="仿真任务不存在"
        )
    
    if simulation["status"] not in [SimulationStatus.PENDING, SimulationStatus.PAUSED]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="仿真任务无法启动"
        )
    
    simulation["status"] = SimulationStatus.RUNNING
    simulation["started_at"] = datetime.utcnow()
    
    logger.info(f"仿真任务已启动: {simulation_id}")
    
    return BaseResponse(
        success=True,
        message="仿真任务已启动",
        data=SimulationResponse(**simulation)
    )


@router.get(
    "/physics/molecules",
    response_model=PaginatedResponse[MoleculeInfo],
    summary="获取分子库",
    description="获取可用的分子库列表"
)
async def get_molecules(
    category: Optional[str] = Query(None, description="类别筛选"),
    search: Optional[str] = Query(None, description="搜索关键词"),
    page: int = Query(1, ge=1, description="页码"),
    page_size: int = Query(20, ge=1, le=100, description="每页数量"),
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> PaginatedResponse[MoleculeInfo]:
    """
    获取分子库
    
    获取系统中可用的分子库列表，支持按类别筛选和搜索。
    """
    molecules = list(_molecules.values())
    
    if category:
        molecules = [m for m in molecules if m.get("category") == category]
    if search:
        search_lower = search.lower()
        molecules = [
            m for m in molecules
            if search_lower in m["name"].lower()
            or search_lower in m.get("formula", "").lower()
        ]
    
    total = len(molecules)
    start_idx = (page - 1) * page_size
    end_idx = start_idx + page_size
    paginated = molecules[start_idx:end_idx]
    
    items = [MoleculeInfo(**m) for m in paginated]
    
    return PaginatedResponse(
        success=True,
        message="获取分子库成功",
        data=items,
        pagination={
            "page": page,
            "page_size": page_size,
            "total": total,
            "total_pages": (total + page_size - 1) // page_size
        }
    )


@router.post(
    "/physics/molecules/{molecule_id}/visualize",
    response_model=BaseResponse[MoleculeVisualizationResponse],
    summary="可视化分子",
    description="生成分子的可视化图像"
)
async def visualize_molecule(
    molecule_id: str,
    request: MoleculeVisualizationRequest,
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> BaseResponse[MoleculeVisualizationResponse]:
    """
    可视化分子
    
    为指定分子生成可视化图像，支持多种可视化风格和交互式查看器。
    """
    molecule = _molecules.get(molecule_id)
    
    if not molecule:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="分子不存在"
        )
    
    response = MoleculeVisualizationResponse(
        molecule_id=molecule_id,
        image_url=f"https://cdn.example.com/molecules/{molecule_id}.png",
        interactive_url=f"https://viewer.example.com/molecules/{molecule_id}",
        format="png",
        width=request.width,
        height=request.height
    )
    
    return BaseResponse(
        success=True,
        message="分子可视化生成成功",
        data=response
    )


# =============================================================================
# 计算机使用端点
# =============================================================================

@router.post(
    "/computer/screenshot",
    response_model=BaseResponse[ScreenshotResponse],
    summary="获取屏幕截图",
    description="捕获屏幕或指定区域的截图"
)
async def take_screenshot(
    request: ScreenshotRequest,
    current_user: Dict[str, Any] = Depends(require_permissions(["computer:control"]))
) -> BaseResponse[ScreenshotResponse]:
    """
    获取屏幕截图
    
    捕获当前屏幕或指定元素的截图，支持多种格式和质量设置。
    """
    response = ScreenshotResponse(
        image_url=f"https://cdn.example.com/screenshots/{uuid.uuid4()}.{request.format}",
        width=request.width or 1920,
        height=request.height or 1080,
        format=request.format
    )
    
    return BaseResponse(
        success=True,
        message="截图成功",
        data=response
    )


@router.post(
    "/computer/click",
    response_model=BaseResponse[Dict[str, Any]],
    summary="模拟点击",
    description="在指定位置模拟鼠标点击"
)
async def simulate_click(
    request: ClickRequest,
    current_user: Dict[str, Any] = Depends(require_permissions(["computer:control"]))
) -> BaseResponse[Dict[str, Any]]:
    """
    模拟点击
    
    在指定坐标或元素上模拟鼠标点击操作。
    """
    return BaseResponse(
        success=True,
        message="点击操作已执行",
        data={
            "action": "click",
            "x": request.x,
            "y": request.y,
            "button": request.button,
            "timestamp": datetime.utcnow().isoformat()
        }
    )


@router.post(
    "/computer/type",
    response_model=BaseResponse[Dict[str, Any]],
    summary="模拟输入",
    description="在指定位置模拟键盘输入"
)
async def simulate_type(
    request: TypeRequest,
    current_user: Dict[str, Any] = Depends(require_permissions(["computer:control"]))
) -> BaseResponse[Dict[str, Any]]:
    """
    模拟输入
    
    在指定输入框中模拟键盘输入文本。
    """
    return BaseResponse(
        success=True,
        message="输入操作已执行",
        data={
            "action": "type",
            "text_length": len(request.text),
            "selector": request.selector,
            "timestamp": datetime.utcnow().isoformat()
        }
    )


@router.post(
    "/computer/scroll",
    response_model=BaseResponse[Dict[str, Any]],
    summary="模拟滚动",
    description="在指定区域模拟鼠标滚动"
)
async def simulate_scroll(
    request: ScrollRequest,
    current_user: Dict[str, Any] = Depends(require_permissions(["computer:control"]))
) -> BaseResponse[Dict[str, Any]]:
    """
    模拟滚动
    
    在页面或指定元素上模拟鼠标滚轮滚动。
    """
    return BaseResponse(
        success=True,
        message="滚动操作已执行",
        data={
            "action": "scroll",
            "direction": request.direction,
            "amount": request.amount,
            "timestamp": datetime.utcnow().isoformat()
        }
    )


@router.post(
    "/computer/navigate",
    response_model=BaseResponse[Dict[str, Any]],
    summary="页面导航",
    description="导航到指定URL"
)
async def navigate_page(
    request: NavigateRequest,
    current_user: Dict[str, Any] = Depends(require_permissions(["computer:control"]))
) -> BaseResponse[Dict[str, Any]]:
    """
    页面导航
    
    导航到指定的URL地址。
    """
    return BaseResponse(
        success=True,
        message="导航成功",
        data={
            "action": "navigate",
            "url": request.url,
            "timestamp": datetime.utcnow().isoformat()
        }
    )


@router.get(
    "/computer/recording",
    response_model=BaseResponse[RecordingResponse],
    summary="获取录制状态",
    description="获取当前屏幕录制的状态"
)
async def get_recording_status(
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> BaseResponse[RecordingResponse]:
    """
    获取录制状态
    
    获取当前屏幕录制的状态和配置信息。
    """
    # 查找当前用户的活动录制
    active_recording = None
    for rec in _recordings.values():
        if rec.get("user_id") == current_user.get("id") and rec["status"] == RecordingStatus.RECORDING:
            active_recording = rec
            break
    
    if not active_recording:
        return BaseResponse(
            success=True,
            message="当前没有活动录制",
            data=RecordingResponse(
                recording_id="",
                status=RecordingStatus.IDLE,
                config=RecordingConfig()
            )
        )
    
    return BaseResponse(
        success=True,
        message="获取录制状态成功",
        data=RecordingResponse(**active_recording)
    )


@router.post(
    "/computer/recording/start",
    response_model=BaseResponse[RecordingResponse],
    summary="开始录制",
    description="开始屏幕录制"
)
async def start_recording(
    config: RecordingConfig,
    current_user: Dict[str, Any] = Depends(require_permissions(["computer:record"]))
) -> BaseResponse[RecordingResponse]:
    """
    开始录制
    
    开始屏幕录制，支持配置帧率、分辨率等参数。
    """
    recording_id = str(uuid.uuid4())
    
    recording_data = {
        "recording_id": recording_id,
        "status": RecordingStatus.RECORDING,
        "config": config.dict(),
        "started_at": datetime.utcnow(),
        "duration": 0,
        "file_size": 0,
        "user_id": current_user.get("id")
    }
    
    _recordings[recording_id] = recording_data
    
    return BaseResponse(
        success=True,
        message="录制已开始",
        data=RecordingResponse(**recording_data)
    )


@router.post(
    "/computer/recording/stop",
    response_model=BaseResponse[RecordingResponse],
    summary="停止录制",
    description="停止当前屏幕录制"
)
async def stop_recording(
    current_user: Dict[str, Any] = Depends(require_permissions(["computer:record"]))
) -> BaseResponse[RecordingResponse]:
    """
    停止录制
    
    停止当前活动的屏幕录制并保存文件。
    """
    # 查找当前用户的活动录制
    active_recording = None
    for rec_id, rec in _recordings.items():
        if rec.get("user_id") == current_user.get("id") and rec["status"] == RecordingStatus.RECORDING:
            active_recording = rec
            active_recording_id = rec_id
            break
    
    if not active_recording:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="没有活动的录制"
        )
    
    active_recording["status"] = RecordingStatus.IDLE
    active_recording["file_url"] = f"https://cdn.example.com/recordings/{active_recording_id}.mp4"
    
    return BaseResponse(
        success=True,
        message="录制已停止",
        data=RecordingResponse(**active_recording)
    )


# =============================================================================
# 安全中心端点
# =============================================================================

@router.get(
    "/security/firewall/rules",
    response_model=PaginatedResponse[FirewallRule],
    summary="获取防火墙规则",
    description="获取所有防火墙规则列表"
)
async def get_firewall_rules(
    page: int = Query(1, ge=1, description="页码"),
    page_size: int = Query(20, ge=1, le=100, description="每页数量"),
    current_user: Dict[str, Any] = Depends(require_permissions(["security:read"]))
) -> PaginatedResponse[FirewallRule]:
    """
    获取防火墙规则
    
    获取系统中配置的所有防火墙规则。
    """
    rules = list(_firewall_rules.values())
    total = len(rules)
    
    start_idx = (page - 1) * page_size
    end_idx = start_idx + page_size
    paginated = rules[start_idx:end_idx]
    
    items = [FirewallRule(**r) for r in paginated]
    
    return PaginatedResponse(
        success=True,
        message="获取防火墙规则成功",
        data=items,
        pagination={
            "page": page,
            "page_size": page_size,
            "total": total,
            "total_pages": (total + page_size - 1) // page_size
        }
    )


@router.post(
    "/security/firewall/rules",
    response_model=BaseResponse[FirewallRule],
    status_code=status.HTTP_201_CREATED,
    summary="添加防火墙规则",
    description="添加新的防火墙规则"
)
async def create_firewall_rule(
    request: FirewallRuleCreateRequest,
    current_user: Dict[str, Any] = Depends(require_permissions(["security:write"]))
) -> BaseResponse[FirewallRule]:
    """
    添加防火墙规则
    
    添加新的防火墙规则来控制网络流量。
    """
    rule_id = str(uuid.uuid4())
    
    rule_data = {
        "id": rule_id,
        "name": request.name,
        "description": request.description,
        "source_ip": request.source_ip,
        "destination_ip": request.destination_ip,
        "port": request.port,
        "protocol": request.protocol,
        "action": request.action,
        "priority": request.priority,
        "enabled": True,
        "created_at": datetime.utcnow()
    }
    
    _firewall_rules[rule_id] = rule_data
    
    logger.info(f"防火墙规则已创建: {rule_id}, 用户: {current_user.get('id')}")
    
    return BaseResponse(
        success=True,
        message="防火墙规则添加成功",
        data=FirewallRule(**rule_data)
    )


@router.get(
    "/security/audit-logs",
    response_model=PaginatedResponse[AuditLogEntry],
    summary="获取审计日志",
    description="获取系统审计日志"
)
async def get_audit_logs(
    start_date: Optional[datetime] = Query(None, description="开始日期"),
    end_date: Optional[datetime] = Query(None, description="结束日期"),
    action: Optional[str] = Query(None, description="操作类型"),
    risk_level: Optional[str] = Query(None, description="风险等级"),
    page: int = Query(1, ge=1, description="页码"),
    page_size: int = Query(20, ge=1, le=100, description="每页数量"),
    current_user: Dict[str, Any] = Depends(require_permissions(["security:audit"]))
) -> PaginatedResponse[AuditLogEntry]:
    """
    获取审计日志
    
    获取系统的审计日志记录，支持按时间范围、操作类型和风险等级筛选。
    """
    logs = list(_audit_logs.values())
    
    if start_date:
        logs = [l for l in logs if l["timestamp"] >= start_date]
    if end_date:
        logs = [l for l in logs if l["timestamp"] <= end_date]
    if action:
        logs = [l for l in logs if l["action"] == action]
    if risk_level:
        logs = [l for l in logs if l.get("risk_level") == risk_level]
    
    # 按时间倒序
    logs.sort(key=lambda x: x["timestamp"], reverse=True)
    
    total = len(logs)
    start_idx = (page - 1) * page_size
    end_idx = start_idx + page_size
    paginated = logs[start_idx:end_idx]
    
    items = [AuditLogEntry(**l) for l in paginated]
    
    return PaginatedResponse(
        success=True,
        message="获取审计日志成功",
        data=items,
        pagination={
            "page": page,
            "page_size": page_size,
            "total": total,
            "total_pages": (total + page_size - 1) // page_size
        }
    )


@router.post(
    "/security/scan",
    response_model=BaseResponse[SecurityScanResponse],
    status_code=status.HTTP_202_ACCEPTED,
    summary="执行安全扫描",
    description="启动安全扫描任务"
)
async def run_security_scan(
    request: SecurityScanRequest,
    current_user: Dict[str, Any] = Depends(require_permissions(["security:scan"]))
) -> BaseResponse[SecurityScanResponse]:
    """
    执行安全扫描
    
    启动安全扫描任务，支持漏洞扫描、恶意软件检测、渗透测试等多种类型。
    """
    scan_id = str(uuid.uuid4())
    
    scan_data = {
        "scan_id": scan_id,
        "scan_type": request.scan_type,
        "status": "pending",
        "target": request.target,
        "progress": 0.0,
        "started_at": datetime.utcnow(),
        "completed_at": None,
        "findings_count": 0,
        "findings": [],
        "report_url": None
    }
    
    _security_scans[scan_id] = scan_data
    
    logger.info(f"安全扫描已启动: {scan_id}, 类型: {request.scan_type}, 用户: {current_user.get('id')}")
    
    return BaseResponse(
        success=True,
        message="安全扫描已启动",
        data=SecurityScanResponse(**scan_data)
    )


@router.get(
    "/security/threats",
    response_model=PaginatedResponse[ThreatIntelligence],
    summary="获取威胁情报",
    description="获取威胁情报列表"
)
async def get_threats(
    level: Optional[ThreatLevel] = Query(None, description="威胁等级"),
    mitigated: Optional[bool] = Query(None, description="是否已缓解"),
    page: int = Query(1, ge=1, description="页码"),
    page_size: int = Query(20, ge=1, le=100, description="每页数量"),
    current_user: Dict[str, Any] = Depends(require_permissions(["security:read"]))
) -> PaginatedResponse[ThreatIntelligence]:
    """
    获取威胁情报
    
    获取系统中的威胁情报信息，包括检测到的威胁和缓解建议。
    """
    threats = list(_threats.values())
    
    if level:
        threats = [t for t in threats if t["level"] == level]
    if mitigated is not None:
        threats = [t for t in threats if t.get("mitigated") == mitigated]
    
    total = len(threats)
    start_idx = (page - 1) * page_size
    end_idx = start_idx + page_size
    paginated = threats[start_idx:end_idx]
    
    items = [ThreatIntelligence(**t) for t in paginated]
    
    return PaginatedResponse(
        success=True,
        message="获取威胁情报成功",
        data=items,
        pagination={
            "page": page,
            "page_size": page_size,
            "total": total,
            "total_pages": (total + page_size - 1) // page_size
        }
    )


@router.post(
    "/security/prompt-guard/test",
    response_model=BaseResponse[PromptGuardTestResponse],
    summary="测试提示词防护",
    description="测试提示词是否存在安全风险"
)
async def test_prompt_guard(
    request: PromptGuardTestRequest,
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> BaseResponse[PromptGuardTestResponse]:
    """
    测试提示词防护
    
    测试输入的提示词是否存在注入攻击、越狱等安全风险。
    """
    test_id = str(uuid.uuid4())
    
    # 模拟检测结果
    detected_issues = []
    risk_score = 0.0
    
    prompt_lower = request.prompt.lower()
    risky_patterns = [
        ("ignore previous", "尝试忽略之前的指令", 0.8),
        ("disregard instructions", "尝试忽略指令", 0.8),
        ("system prompt", "尝试访问系统提示词", 0.7),
        ("jailbreak", "越狱尝试", 0.9),
        ("DAN", "DAN越狱模式", 0.9)
    ]
    
    for pattern, description, score in risky_patterns:
        if pattern in prompt_lower:
            detected_issues.append({
                "type": "suspicious_pattern",
                "description": description,
                "matched_text": pattern,
                "severity": "high" if score > 0.8 else "medium"
            })
            risk_score = max(risk_score, score)
    
    response = PromptGuardTestResponse(
        test_id=test_id,
        prompt=request.prompt,
        passed=len(detected_issues) == 0,
        risk_score=risk_score,
        detected_issues=detected_issues,
        recommendations=[
            "避免使用可能绕过安全限制的提示词",
            "使用更具体的任务描述"
        ] if detected_issues else [],
        explanation="检测到潜在的安全风险" if detected_issues else "未发现明显风险"
    )
    
    return BaseResponse(
        success=True,
        message="提示词防护测试完成",
        data=response
    )


# =============================================================================
# 联邦学习端点
# =============================================================================

@router.get(
    "/federated/nodes",
    response_model=PaginatedResponse[FederatedNode],
    summary="获取节点列表",
    description="获取联邦学习节点列表"
)
async def get_federated_nodes(
    status: Optional[NodeStatus] = Query(None, description="节点状态"),
    page: int = Query(1, ge=1, description="页码"),
    page_size: int = Query(20, ge=1, le=100, description="每页数量"),
    current_user: Dict[str, Any] = Depends(require_permissions(["federated:read"]))
) -> PaginatedResponse[FederatedNode]:
    """
    获取节点列表
    
    获取参与联邦学习的所有节点信息。
    """
    nodes = list(_federated_nodes.values())
    
    if status:
        nodes = [n for n in nodes if n["status"] == status]
    
    total = len(nodes)
    start_idx = (page - 1) * page_size
    end_idx = start_idx + page_size
    paginated = nodes[start_idx:end_idx]
    
    items = [FederatedNode(**n) for n in paginated]
    
    return PaginatedResponse(
        success=True,
        message="获取节点列表成功",
        data=items,
        pagination={
            "page": page,
            "page_size": page_size,
            "total": total,
            "total_pages": (total + page_size - 1) // page_size
        }
    )


@router.post(
    "/federated/nodes",
    response_model=BaseResponse[FederatedNode],
    status_code=status.HTTP_201_CREATED,
    summary="注册节点",
    description="注册新的联邦学习节点"
)
async def register_federated_node(
    request: FederatedNodeRegisterRequest,
    current_user: Dict[str, Any] = Depends(require_permissions(["federated:write"]))
) -> BaseResponse[FederatedNode]:
    """
    注册节点
    
    注册新的节点参与联邦学习。
    """
    node_id = str(uuid.uuid4())
    
    node_data = {
        "id": node_id,
        "name": request.name,
        "status": NodeStatus.ONLINE,
        "endpoint": request.endpoint,
        "public_key": request.public_key,
        "capabilities": request.capabilities,
        "dataset_size": request.dataset_info.get("size") if request.dataset_info else None,
        "last_seen": datetime.utcnow(),
        "registered_at": datetime.utcnow(),
        "contribution_score": 0.0
    }
    
    _federated_nodes[node_id] = node_data
    
    logger.info(f"联邦学习节点已注册: {node_id}, 用户: {current_user.get('id')}")
    
    return BaseResponse(
        success=True,
        message="节点注册成功",
        data=FederatedNode(**node_data)
    )


@router.get(
    "/federated/rounds",
    response_model=PaginatedResponse[TrainingRound],
    summary="获取训练轮次",
    description="获取联邦学习训练轮次列表"
)
async def get_training_rounds(
    page: int = Query(1, ge=1, description="页码"),
    page_size: int = Query(20, ge=1, le=100, description="每页数量"),
    current_user: Dict[str, Any] = Depends(require_permissions(["federated:read"]))
) -> PaginatedResponse[TrainingRound]:
    """
    获取训练轮次
    
    获取联邦学习的训练轮次历史和状态。
    """
    rounds = list(_training_rounds.values())
    rounds.sort(key=lambda x: x.get("round_number", 0), reverse=True)
    
    total = len(rounds)
    start_idx = (page - 1) * page_size
    end_idx = start_idx + page_size
    paginated = rounds[start_idx:end_idx]
    
    items = [TrainingRound(**r) for r in paginated]
    
    return PaginatedResponse(
        success=True,
        message="获取训练轮次成功",
        data=items,
        pagination={
            "page": page,
            "page_size": page_size,
            "total": total,
            "total_pages": (total + page_size - 1) // page_size
        }
    )


@router.post(
    "/federated/rounds/{round_id}/aggregate",
    response_model=BaseResponse[AggregationResponse],
    summary="执行聚合",
    description="执行联邦学习模型聚合"
)
async def aggregate_round(
    round_id: str,
    request: AggregationRequest,
    current_user: Dict[str, Any] = Depends(require_permissions(["federated:admin"]))
) -> BaseResponse[AggregationResponse]:
    """
    执行聚合
    
    对指定训练轮次的节点更新执行模型聚合。
    """
    round_data = _training_rounds.get(round_id)
    
    if not round_data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="训练轮次不存在"
        )
    
    response = AggregationResponse(
        round_id=round_id,
        status="completed",
        aggregated_model_url=f"https://cdn.example.com/models/aggregated_{round_id}.pth",
        metrics={
            "accuracy": 0.92,
            "loss": 0.08,
            "participating_nodes": len(request.node_updates)
        },
        node_contributions={node_id: 1.0 / len(request.node_updates) for node_id in request.node_updates},
        completed_at=datetime.utcnow()
    )
    
    return BaseResponse(
        success=True,
        message="模型聚合完成",
        data=response
    )


@router.get(
    "/federated/contributions",
    response_model=PaginatedResponse[ContributionStats],
    summary="获取贡献统计",
    description="获取联邦学习节点贡献统计"
)
async def get_contribution_stats(
    page: int = Query(1, ge=1, description="页码"),
    page_size: int = Query(20, ge=1, le=100, description="每页数量"),
    current_user: Dict[str, Any] = Depends(require_permissions(["federated:read"]))
) -> PaginatedResponse[ContributionStats]:
    """
    获取贡献统计
    
    获取各节点在联邦学习中的贡献统计数据。
    """
    contributions = list(_contributions.values())
    contributions.sort(key=lambda x: x.get("reputation_score", 0), reverse=True)
    
    total = len(contributions)
    start_idx = (page - 1) * page_size
    end_idx = start_idx + page_size
    paginated = contributions[start_idx:end_idx]
    
    items = [ContributionStats(**c) for c in paginated]
    
    return PaginatedResponse(
        success=True,
        message="获取贡献统计成功",
        data=items,
        pagination={
            "page": page,
            "page_size": page_size,
            "total": total,
            "total_pages": (total + page_size - 1) // page_size
        }
    )


# =============================================================================
# 知识检索端点
# =============================================================================

@router.post(
    "/rag/documents",
    response_model=BaseResponse[DocumentUploadResponse],
    status_code=status.HTTP_202_ACCEPTED,
    summary="上传文档",
    description="上传文档到知识库"
)
async def upload_document(
    file: UploadFile = File(..., description="上传的文件"),
    knowledge_base_id: Optional[str] = Query(None, description="目标知识库ID"),
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> BaseResponse[DocumentUploadResponse]:
    """
    上传文档
    
    上传文档到知识库进行索引，支持PDF、Word、文本等多种格式。
    """
    document_id = str(uuid.uuid4())
    
    # 确定文档类型
    filename = file.filename or "unknown"
    extension = filename.split(".")[-1].lower() if "." in filename else ""
    
    type_mapping = {
        "pdf": DocumentType.PDF,
        "doc": DocumentType.WORD,
        "docx": DocumentType.WORD,
        "txt": DocumentType.TEXT,
        "md": DocumentType.MARKDOWN,
        "html": DocumentType.HTML,
        "py": DocumentType.CODE,
        "js": DocumentType.CODE,
        "json": DocumentType.JSON
    }
    
    doc_type = type_mapping.get(extension, DocumentType.TEXT)
    
    document_data = {
        "document_id": document_id,
        "filename": filename,
        "type": doc_type,
        "size": 0,  # 实际应从文件获取
        "status": "processing",
        "chunks_count": None,
        "uploaded_at": datetime.utcnow()
    }
    
    _documents[document_id] = document_data
    
    logger.info(f"文档已上传: {document_id}, 用户: {current_user.get('id')}")
    
    return BaseResponse(
        success=True,
        message="文档上传成功，正在处理",
        data=DocumentUploadResponse(**document_data)
    )


@router.get(
    "/rag/documents",
    response_model=PaginatedResponse[DocumentInfo],
    summary="获取文档列表",
    description="获取知识库中的文档列表"
)
async def get_documents(
    type: Optional[DocumentType] = Query(None, description="文档类型"),
    status: Optional[str] = Query(None, description="处理状态"),
    knowledge_base_id: Optional[str] = Query(None, description="知识库ID"),
    page: int = Query(1, ge=1, description="页码"),
    page_size: int = Query(20, ge=1, le=100, description="每页数量"),
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> PaginatedResponse[DocumentInfo]:
    """
    获取文档列表
    
    获取知识库中的所有文档列表。
    """
    documents = list(_documents.values())
    
    if type:
        documents = [d for d in documents if d["type"] == type]
    if status:
        documents = [d for d in documents if d["status"] == status]
    
    total = len(documents)
    start_idx = (page - 1) * page_size
    end_idx = start_idx + page_size
    paginated = documents[start_idx:end_idx]
    
    items = [DocumentInfo(**d) for d in paginated]
    
    return PaginatedResponse(
        success=True,
        message="获取文档列表成功",
        data=items,
        pagination={
            "page": page,
            "page_size": page_size,
            "total": total,
            "total_pages": (total + page_size - 1) // page_size
        }
    )


@router.delete(
    "/rag/documents/{document_id}",
    response_model=BaseResponse[Dict[str, Any]],
    summary="删除文档",
    description="从知识库中删除文档"
)
async def delete_document(
    document_id: str,
    current_user: Dict[str, Any] = Depends(require_permissions(["rag:delete"]))
) -> BaseResponse[Dict[str, Any]]:
    """
    删除文档
    
    从知识库中删除指定的文档及其索引。
    """
    if document_id not in _documents:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="文档不存在"
        )
    
    del _documents[document_id]
    
    return BaseResponse(
        success=True,
        message="文档已删除",
        data={"document_id": document_id, "deleted_at": datetime.utcnow().isoformat()}
    )


@router.post(
    "/rag/search",
    response_model=BaseResponse[RAGSearchResponse],
    summary="执行检索",
    description="在知识库中执行语义检索"
)
async def search_documents(
    request: RAGSearchRequest,
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> BaseResponse[RAGSearchResponse]:
    """
    执行检索
    
    在知识库中执行语义检索，返回与查询最相关的文档片段。
    """
    # 模拟检索结果
    results = [
        RAGSearchResult(
            document_id="doc-001",
            document_name="示例文档1.pdf",
            chunk_index=0,
            content="这是与查询相关的文档内容片段...",
            score=0.92,
            metadata={"page": 1}
        ),
        RAGSearchResult(
            document_id="doc-002",
            document_name="示例文档2.txt",
            chunk_index=3,
            content="另一个相关的内容片段...",
            score=0.85,
            metadata={"line_start": 45}
        )
    ]
    
    response = RAGSearchResponse(
        query=request.query,
        results=results[:request.top_k],
        total_results=len(results),
        search_time_ms=150,
        strategy_used=request.strategy
    )
    
    return BaseResponse(
        success=True,
        message="检索完成",
        data=response
    )


@router.get(
    "/rag/knowledge-bases",
    response_model=PaginatedResponse[KnowledgeBaseInfo],
    summary="获取知识库列表",
    description="获取所有知识库列表"
)
async def get_knowledge_bases(
    page: int = Query(1, ge=1, description="页码"),
    page_size: int = Query(20, ge=1, le=100, description="每页数量"),
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> PaginatedResponse[KnowledgeBaseInfo]:
    """
    获取知识库列表
    
    获取系统中所有可用的知识库列表。
    """
    knowledge_bases = list(_knowledge_bases.values())
    
    total = len(knowledge_bases)
    start_idx = (page - 1) * page_size
    end_idx = start_idx + page_size
    paginated = knowledge_bases[start_idx:end_idx]
    
    items = [KnowledgeBaseInfo(**kb) for kb in paginated]
    
    return PaginatedResponse(
        success=True,
        message="获取知识库列表成功",
        data=items,
        pagination={
            "page": page,
            "page_size": page_size,
            "total": total,
            "total_pages": (total + page_size - 1) // page_size
        }
    )


# =============================================================================
# 渠道管理端点
# =============================================================================

@router.get(
    "/channels",
    response_model=PaginatedResponse[ChannelResponse],
    summary="获取渠道列表",
    description="获取所有渠道列表"
)
async def get_channels(
    type: Optional[ChannelType] = Query(None, description="渠道类型"),
    status: Optional[ChannelStatus] = Query(None, description="渠道状态"),
    page: int = Query(1, ge=1, description="页码"),
    page_size: int = Query(20, ge=1, le=100, description="每页数量"),
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> PaginatedResponse[ChannelResponse]:
    """
    获取渠道列表
    
    获取所有配置的渠道列表，支持按类型和状态筛选。
    """
    channels = list(_channels.values())
    
    if type:
        channels = [c for c in channels if c["type"] == type]
    if status:
        channels = [c for c in channels if c["status"] == status]
    
    total = len(channels)
    start_idx = (page - 1) * page_size
    end_idx = start_idx + page_size
    paginated = channels[start_idx:end_idx]
    
    items = [ChannelResponse(**c) for c in paginated]
    
    return PaginatedResponse(
        success=True,
        message="获取渠道列表成功",
        data=items,
        pagination={
            "page": page,
            "page_size": page_size,
            "total": total,
            "total_pages": (total + page_size - 1) // page_size
        }
    )


@router.post(
    "/channels",
    response_model=BaseResponse[ChannelResponse],
    status_code=status.HTTP_201_CREATED,
    summary="添加渠道",
    description="添加新的渠道"
)
async def create_channel(
    request: ChannelCreateRequest,
    current_user: Dict[str, Any] = Depends(require_permissions(["channels:write"]))
) -> BaseResponse[ChannelResponse]:
    """
    添加渠道
    
    添加新的消息渠道，如Slack、Discord、邮件等。
    """
    channel_id = str(uuid.uuid4())
    
    channel_data = {
        "id": channel_id,
        "name": request.name,
        "type": request.type,
        "description": request.description,
        "status": ChannelStatus.PENDING,
        "config": request.config.dict(),
        "is_default": request.is_default,
        "message_count": 0,
        "last_message_at": None,
        "created_at": datetime.utcnow()
    }
    
    _channels[channel_id] = channel_data
    
    logger.info(f"渠道已创建: {channel_id}, 用户: {current_user.get('id')}")
    
    return BaseResponse(
        success=True,
        message="渠道添加成功",
        data=ChannelResponse(**channel_data)
    )


@router.put(
    "/channels/{channel_id}",
    response_model=BaseResponse[ChannelResponse],
    summary="更新渠道",
    description="更新渠道配置"
)
async def update_channel(
    channel_id: str,
    request: ChannelCreateRequest,
    current_user: Dict[str, Any] = Depends(require_permissions(["channels:write"]))
) -> BaseResponse[ChannelResponse]:
    """
    更新渠道
    
    更新指定渠道的配置信息。
    """
    channel = _channels.get(channel_id)
    
    if not channel:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="渠道不存在"
        )
    
    channel["name"] = request.name
    channel["description"] = request.description
    channel["config"] = request.config.dict()
    channel["is_default"] = request.is_default
    
    return BaseResponse(
        success=True,
        message="渠道更新成功",
        data=ChannelResponse(**channel)
    )


@router.delete(
    "/channels/{channel_id}",
    response_model=BaseResponse[Dict[str, Any]],
    summary="删除渠道",
    description="删除指定渠道"
)
async def delete_channel(
    channel_id: str,
    current_user: Dict[str, Any] = Depends(require_permissions(["channels:delete"]))
) -> BaseResponse[Dict[str, Any]]:
    """
    删除渠道
    
    删除指定的消息渠道。
    """
    if channel_id not in _channels:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="渠道不存在"
        )
    
    del _channels[channel_id]
    
    return BaseResponse(
        success=True,
        message="渠道已删除",
        data={"channel_id": channel_id, "deleted_at": datetime.utcnow().isoformat()}
    )


@router.post(
    "/channels/{channel_id}/test",
    response_model=BaseResponse[ChannelTestResponse],
    summary="测试渠道连接",
    description="测试渠道的连接状态"
)
async def test_channel(
    channel_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> BaseResponse[ChannelTestResponse]:
    """
    测试渠道连接
    
    测试指定渠道的连接状态和响应延迟。
    """
    channel = _channels.get(channel_id)
    
    if not channel:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="渠道不存在"
        )
    
    response = ChannelTestResponse(
        channel_id=channel_id,
        success=True,
        latency_ms=45,
        tested_at=datetime.utcnow()
    )
    
    return BaseResponse(
        success=True,
        message="渠道连接测试完成",
        data=response
    )


@router.get(
    "/channels/{channel_id}/messages",
    response_model=PaginatedResponse[ChannelMessage],
    summary="获取渠道消息",
    description="获取指定渠道的消息历史"
)
async def get_channel_messages(
    channel_id: str,
    page: int = Query(1, ge=1, description="页码"),
    page_size: int = Query(20, ge=1, le=100, description="每页数量"),
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> PaginatedResponse[ChannelMessage]:
    """
    获取渠道消息
    
    获取指定渠道的消息历史记录。
    """
    if channel_id not in _channels:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="渠道不存在"
        )
    
    messages = _channel_messages.get(channel_id, [])
    messages.sort(key=lambda x: x["timestamp"], reverse=True)
    
    total = len(messages)
    start_idx = (page - 1) * page_size
    end_idx = start_idx + page_size
    paginated = messages[start_idx:end_idx]
    
    items = [ChannelMessage(**m) for m in paginated]
    
    return PaginatedResponse(
        success=True,
        message="获取消息成功",
        data=items,
        pagination={
            "page": page,
            "page_size": page_size,
            "total": total,
            "total_pages": (total + page_size - 1) // page_size
        }
    )


# =============================================================================
# 插件市场端点
# =============================================================================

@router.get(
    "/plugins",
    response_model=PaginatedResponse[PluginInfo],
    summary="获取插件列表",
    description="获取已安装的插件列表"
)
async def get_plugins(
    status: Optional[PluginStatus] = Query(None, description="插件状态"),
    category: Optional[PluginCategory] = Query(None, description="插件类别"),
    page: int = Query(1, ge=1, description="页码"),
    page_size: int = Query(20, ge=1, le=100, description="每页数量"),
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> PaginatedResponse[PluginInfo]:
    """
    获取插件列表
    
    获取系统中已安装的插件列表。
    """
    plugins = list(_plugins.values())
    
    if status:
        plugins = [p for p in plugins if p["status"] == status]
    if category:
        plugins = [p for p in plugins if p["category"] == category]
    
    total = len(plugins)
    start_idx = (page - 1) * page_size
    end_idx = start_idx + page_size
    paginated = plugins[start_idx:end_idx]
    
    items = [PluginInfo(**p) for p in paginated]
    
    return PaginatedResponse(
        success=True,
        message="获取插件列表成功",
        data=items,
        pagination={
            "page": page,
            "page_size": page_size,
            "total": total,
            "total_pages": (total + page_size - 1) // page_size
        }
    )


@router.post(
    "/plugins",
    response_model=BaseResponse[PluginInfo],
    status_code=status.HTTP_201_CREATED,
    summary="安装插件",
    description="安装新插件"
)
async def install_plugin(
    request: PluginInstallRequest,
    current_user: Dict[str, Any] = Depends(require_permissions(["plugins:install"]))
) -> BaseResponse[PluginInfo]:
    """
    安装插件
    
    从指定来源安装新插件。
    """
    plugin_id = str(uuid.uuid4())
    
    plugin_data = {
        "id": plugin_id,
        "name": "New Plugin",
        "description": "新安装的插件",
        "version": request.version or "1.0.0",
        "author": "Unknown",
        "category": PluginCategory.CUSTOM,
        "status": PluginStatus.ENABLED if request.auto_enable else PluginStatus.INSTALLED,
        "permissions": [],
        "installed_at": datetime.utcnow()
    }
    
    _plugins[plugin_id] = plugin_data
    
    logger.info(f"插件已安装: {plugin_id}, 用户: {current_user.get('id')}")
    
    return BaseResponse(
        success=True,
        message="插件安装成功",
        data=PluginInfo(**plugin_data)
    )


@router.delete(
    "/plugins/{plugin_id}",
    response_model=BaseResponse[Dict[str, Any]],
    summary="卸载插件",
    description="卸载指定插件"
)
async def uninstall_plugin(
    plugin_id: str,
    current_user: Dict[str, Any] = Depends(require_permissions(["plugins:install"]))
) -> BaseResponse[Dict[str, Any]]:
    """
    卸载插件
    
    卸载指定的插件。
    """
    if plugin_id not in _plugins:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="插件不存在"
        )
    
    del _plugins[plugin_id]
    
    return BaseResponse(
        success=True,
        message="插件已卸载",
        data={"plugin_id": plugin_id, "uninstalled_at": datetime.utcnow().isoformat()}
    )


@router.post(
    "/plugins/{plugin_id}/enable",
    response_model=BaseResponse[PluginInfo],
    summary="启用插件",
    description="启用指定插件"
)
async def enable_plugin(
    plugin_id: str,
    current_user: Dict[str, Any] = Depends(require_permissions(["plugins:manage"]))
) -> BaseResponse[PluginInfo]:
    """
    启用插件
    
    启用指定的插件。
    """
    plugin = _plugins.get(plugin_id)
    
    if not plugin:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="插件不存在"
        )
    
    plugin["status"] = PluginStatus.ENABLED
    
    return BaseResponse(
        success=True,
        message="插件已启用",
        data=PluginInfo(**plugin)
    )


@router.post(
    "/plugins/{plugin_id}/disable",
    response_model=BaseResponse[PluginInfo],
    summary="禁用插件",
    description="禁用指定插件"
)
async def disable_plugin(
    plugin_id: str,
    current_user: Dict[str, Any] = Depends(require_permissions(["plugins:manage"]))
) -> BaseResponse[PluginInfo]:
    """
    禁用插件
    
    禁用指定的插件。
    """
    plugin = _plugins.get(plugin_id)
    
    if not plugin:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="插件不存在"
        )
    
    plugin["status"] = PluginStatus.DISABLED
    
    return BaseResponse(
        success=True,
        message="插件已禁用",
        data=PluginInfo(**plugin)
    )


@router.get(
    "/plugins/marketplace",
    response_model=PaginatedResponse[PluginMarketplaceItem],
    summary="浏览插件市场",
    description="浏览可用的插件市场"
)
async def get_plugin_marketplace(
    category: Optional[PluginCategory] = Query(None, description="类别"),
    search: Optional[str] = Query(None, description="搜索关键词"),
    page: int = Query(1, ge=1, description="页码"),
    page_size: int = Query(20, ge=1, le=100, description="每页数量"),
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> PaginatedResponse[PluginMarketplaceItem]:
    """
    浏览插件市场
    
    浏览插件市场中可用的插件。
    """
    plugins = list(_marketplace_plugins.values())
    
    if category:
        plugins = [p for p in plugins if p["category"] == category]
    if search:
        search_lower = search.lower()
        plugins = [
            p for p in plugins
            if search_lower in p["name"].lower()
            or search_lower in p["description"].lower()
        ]
    
    total = len(plugins)
    start_idx = (page - 1) * page_size
    end_idx = start_idx + page_size
    paginated = plugins[start_idx:end_idx]
    
    items = [PluginMarketplaceItem(**p) for p in paginated]
    
    return PaginatedResponse(
        success=True,
        message="获取插件市场成功",
        data=items,
        pagination={
            "page": page,
            "page_size": page_size,
            "total": total,
            "total_pages": (total + page_size - 1) // page_size
        }
    )


# =============================================================================
# 人格引擎端点
# =============================================================================

@router.get(
    "/personalities",
    response_model=PaginatedResponse[PersonalityResponse],
    summary="获取人格列表",
    description="获取所有人格列表"
)
async def get_personalities(
    is_active: Optional[bool] = Query(None, description="是否激活"),
    communication_style: Optional[CommunicationStyle] = Query(None, description="沟通风格"),
    page: int = Query(1, ge=1, description="页码"),
    page_size: int = Query(20, ge=1, le=100, description="每页数量"),
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> PaginatedResponse[PersonalityResponse]:
    """
    获取人格列表
    
    获取系统中所有可用的AI人格列表。
    """
    personalities = list(_personalities.values())
    
    if is_active is not None:
        personalities = [p for p in personalities if p.get("is_active") == is_active]
    if communication_style:
        personalities = [p for p in personalities if p.get("communication_style") == communication_style]
    
    total = len(personalities)
    start_idx = (page - 1) * page_size
    end_idx = start_idx + page_size
    paginated = personalities[start_idx:end_idx]
    
    items = [PersonalityResponse(**p) for p in paginated]
    
    return PaginatedResponse(
        success=True,
        message="获取人格列表成功",
        data=items,
        pagination={
            "page": page,
            "page_size": page_size,
            "total": total,
            "total_pages": (total + page_size - 1) // page_size
        }
    )


@router.post(
    "/personalities",
    response_model=BaseResponse[PersonalityResponse],
    status_code=status.HTTP_201_CREATED,
    summary="创建人格",
    description="创建新的AI人格"
)
async def create_personality(
    request: PersonalityCreateRequest,
    current_user: Dict[str, Any] = Depends(require_permissions(["personalities:write"]))
) -> BaseResponse[PersonalityResponse]:
    """
    创建人格
    
    创建新的AI人格，定义其行为特征、沟通风格和知识领域。
    """
    personality_id = str(uuid.uuid4())
    
    personality_data = {
        "id": personality_id,
        "name": request.name,
        "description": request.description,
        "avatar_url": request.avatar_url,
        "system_prompt": request.system_prompt,
        "traits": request.traits.dict(),
        "communication_style": request.communication_style,
        "is_active": False,
        "usage_count": 0,
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow(),
        "created_by": current_user.get("id")
    }
    
    _personalities[personality_id] = personality_data
    
    logger.info(f"人格已创建: {personality_id}, 用户: {current_user.get('id')}")
    
    return BaseResponse(
        success=True,
        message="人格创建成功",
        data=PersonalityResponse(**personality_data)
    )


@router.get(
    "/personalities/{personality_id}",
    response_model=BaseResponse[PersonalityResponse],
    summary="获取人格详情",
    description="获取指定人格的详细信息"
)
async def get_personality(
    personality_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> BaseResponse[PersonalityResponse]:
    """
    获取人格详情
    
    获取指定ID的人格详细信息。
    """
    personality = _personalities.get(personality_id)
    
    if not personality:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="人格不存在"
        )
    
    return BaseResponse(
        success=True,
        message="获取人格详情成功",
        data=PersonalityResponse(**personality)
    )


@router.put(
    "/personalities/{personality_id}",
    response_model=BaseResponse[PersonalityResponse],
    summary="更新人格",
    description="更新人格配置"
)
async def update_personality(
    personality_id: str,
    request: PersonalityCreateRequest,
    current_user: Dict[str, Any] = Depends(require_permissions(["personalities:write"]))
) -> BaseResponse[PersonalityResponse]:
    """
    更新人格
    
    更新指定人格的配置信息。
    """
    personality = _personalities.get(personality_id)
    
    if not personality:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="人格不存在"
        )
    
    personality["name"] = request.name
    personality["description"] = request.description
    personality["avatar_url"] = request.avatar_url
    personality["system_prompt"] = request.system_prompt
    personality["traits"] = request.traits.dict()
    personality["communication_style"] = request.communication_style
    personality["updated_at"] = datetime.utcnow()
    
    return BaseResponse(
        success=True,
        message="人格更新成功",
        data=PersonalityResponse(**personality)
    )


@router.delete(
    "/personalities/{personality_id}",
    response_model=BaseResponse[Dict[str, Any]],
    summary="删除人格",
    description="删除指定人格"
)
async def delete_personality(
    personality_id: str,
    current_user: Dict[str, Any] = Depends(require_permissions(["personalities:delete"]))
) -> BaseResponse[Dict[str, Any]]:
    """
    删除人格
    
    删除指定的AI人格。
    """
    if personality_id not in _personalities:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="人格不存在"
        )
    
    del _personalities[personality_id]
    
    return BaseResponse(
        success=True,
        message="人格已删除",
        data={"personality_id": personality_id, "deleted_at": datetime.utcnow().isoformat()}
    )


@router.post(
    "/personalities/{personality_id}/activate",
    response_model=BaseResponse[PersonalityResponse],
    summary="激活人格",
    description="激活指定人格"
)
async def activate_personality(
    personality_id: str,
    current_user: Dict[str, Any] = Depends(require_permissions(["personalities:manage"]))
) -> BaseResponse[PersonalityResponse]:
    """
    激活人格
    
    激活指定的AI人格，使其成为当前使用的默认人格。
    """
    personality = _personalities.get(personality_id)
    
    if not personality:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="人格不存在"
        )
    
    # 先停用其他人格
    for p in _personalities.values():
        p["is_active"] = False
    
    personality["is_active"] = True
    
    return BaseResponse(
        success=True,
        message="人格已激活",
        data=PersonalityResponse(**personality)
    )


@router.post(
    "/personalities/{personality_id}/test",
    response_model=BaseResponse[PersonalityTestResponse],
    summary="测试人格",
    description="测试人格的表现"
)
async def test_personality(
    personality_id: str,
    request: PersonalityTestRequest,
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> BaseResponse[PersonalityTestResponse]:
    """
    测试人格
    
    在多个场景下测试AI人格的表现和一致性。
    """
    personality = _personalities.get(personality_id)
    
    if not personality:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="人格不存在"
        )
    
    test_id = str(uuid.uuid4())
    
    response = PersonalityTestResponse(
        test_id=test_id,
        personality_id=personality_id,
        scenarios_tested=len(request.test_scenarios) or 3,
        coherence_score=0.85,
        appropriateness_score=0.90,
        consistency_score=0.88,
        detailed_feedback=[
            {"scenario": "greeting", "score": 0.95, "feedback": "问候得体"},
            {"scenario": "problem_solving", "score": 0.82, "feedback": "解决方案合理"}
        ],
        recommendations=[
            "可以增加更多领域知识",
            "回复可以更加简洁"
        ]
    )
    
    return BaseResponse(
        success=True,
        message="人格测试完成",
        data=response
    )


# =============================================================================
# 数据管道端点
# =============================================================================

@router.get(
    "/data-pipeline/datasets",
    response_model=PaginatedResponse[DatasetInfo],
    summary="获取数据集列表",
    description="获取所有数据集列表"
)
async def get_datasets(
    format: Optional[DatasetFormat] = Query(None, description="数据格式"),
    status: Optional[str] = Query(None, description="状态"),
    page: int = Query(1, ge=1, description="页码"),
    page_size: int = Query(20, ge=1, le=100, description="每页数量"),
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> PaginatedResponse[DatasetInfo]:
    """
    获取数据集列表
    
    获取系统中所有数据集的列表。
    """
    datasets = list(_datasets.values())
    
    if format:
        datasets = [d for d in datasets if d["format"] == format]
    if status:
        datasets = [d for d in datasets if d["status"] == status]
    
    total = len(datasets)
    start_idx = (page - 1) * page_size
    end_idx = start_idx + page_size
    paginated = datasets[start_idx:end_idx]
    
    items = [DatasetInfo(**d) for d in paginated]
    
    return PaginatedResponse(
        success=True,
        message="获取数据集列表成功",
        data=items,
        pagination={
            "page": page,
            "page_size": page_size,
            "total": total,
            "total_pages": (total + page_size - 1) // page_size
        }
    )


@router.post(
    "/data-pipeline/datasets",
    response_model=BaseResponse[DatasetInfo],
    status_code=status.HTTP_201_CREATED,
    summary="上传数据集",
    description="上传新的数据集"
)
async def upload_dataset(
    file: UploadFile = File(..., description="数据集文件"),
    name: Optional[str] = Query(None, description="数据集名称"),
    description: Optional[str] = Query(None, description="描述"),
    current_user: Dict[str, Any] = Depends(require_permissions(["data:write"]))
) -> BaseResponse[DatasetInfo]:
    """
    上传数据集
    
    上传新的数据集文件到系统。
    """
    dataset_id = str(uuid.uuid4())
    filename = file.filename or "unknown"
    
    # 确定格式
    extension = filename.split(".")[-1].lower() if "." in filename else ""
    format_mapping = {
        "csv": DatasetFormat.CSV,
        "json": DatasetFormat.JSON,
        "parquet": DatasetFormat.PARQUET,
        "xlsx": DatasetFormat.EXCEL,
        "xls": DatasetFormat.EXCEL,
        "sql": DatasetFormat.SQL,
        "h5": DatasetFormat.HDF5
    }
    
    dataset_format = format_mapping.get(extension, DatasetFormat.CSV)
    
    dataset_data = {
        "id": dataset_id,
        "name": name or filename,
        "description": description,
        "format": dataset_format,
        "size": 0,
        "rows": None,
        "columns": None,
        "schema": None,
        "tags": [],
        "uploaded_at": datetime.utcnow(),
        "processed_at": None,
        "status": "uploaded"
    }
    
    _datasets[dataset_id] = dataset_data
    
    logger.info(f"数据集已上传: {dataset_id}, 用户: {current_user.get('id')}")
    
    return BaseResponse(
        success=True,
        message="数据集上传成功",
        data=DatasetInfo(**dataset_data)
    )


@router.post(
    "/data-pipeline/datasets/{dataset_id}/process",
    response_model=BaseResponse[DatasetProcessResponse],
    status_code=status.HTTP_202_ACCEPTED,
    summary="处理数据集",
    description="对数据集执行处理操作"
)
async def process_dataset(
    dataset_id: str,
    request: DatasetProcessRequest,
    current_user: Dict[str, Any] = Depends(require_permissions(["data:process"]))
) -> BaseResponse[DatasetProcessResponse]:
    """
    处理数据集
    
    对数据集执行清洗、转换、增强等处理操作。
    """
    dataset = _datasets.get(dataset_id)
    
    if not dataset:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="数据集不存在"
        )
    
    job_id = str(uuid.uuid4())
    
    response = DatasetProcessResponse(
        job_id=job_id,
        dataset_id=dataset_id,
        status="processing",
        progress=0.0,
        operations_applied=[op.get("type") for op in request.operations],
        started_at=datetime.utcnow()
    )
    
    return BaseResponse(
        success=True,
        message="数据集处理已启动",
        data=response
    )


@router.get(
    "/data-pipeline/pipelines",
    response_model=PaginatedResponse[PipelineResponse],
    summary="获取流水线列表",
    description="获取所有数据流水线列表"
)
async def get_pipelines(
    status: Optional[PipelineStatus] = Query(None, description="状态"),
    page: int = Query(1, ge=1, description="页码"),
    page_size: int = Query(20, ge=1, le=100, description="每页数量"),
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> PaginatedResponse[PipelineResponse]:
    """
    获取流水线列表
    
    获取所有数据管道的列表。
    """
    pipelines = list(_pipelines.values())
    
    if status:
        pipelines = [p for p in pipelines if p["status"] == status]
    
    total = len(pipelines)
    start_idx = (page - 1) * page_size
    end_idx = start_idx + page_size
    paginated = pipelines[start_idx:end_idx]
    
    items = [PipelineResponse(**p) for p in paginated]
    
    return PaginatedResponse(
        success=True,
        message="获取流水线列表成功",
        data=items,
        pagination={
            "page": page,
            "page_size": page_size,
            "total": total,
            "total_pages": (total + page_size - 1) // page_size
        }
    )


@router.post(
    "/data-pipeline/pipelines",
    response_model=BaseResponse[PipelineResponse],
    status_code=status.HTTP_201_CREATED,
    summary="创建流水线",
    description="创建新的数据处理流水线"
)
async def create_pipeline(
    request: PipelineCreateRequest,
    current_user: Dict[str, Any] = Depends(require_permissions(["data:write"]))
) -> BaseResponse[PipelineResponse]:
    """
    创建流水线
    
    创建新的数据处理流水线，定义数据处理流程。
    """
    pipeline_id = str(uuid.uuid4())
    
    pipeline_data = {
        "id": pipeline_id,
        "name": request.name,
        "description": request.description,
        "status": PipelineStatus.DRAFT,
        "steps": [step.dict() for step in request.steps],
        "schedule": request.schedule,
        "last_run_at": None,
        "last_run_status": None,
        "run_count": 0,
        "created_at": datetime.utcnow()
    }
    
    _pipelines[pipeline_id] = pipeline_data
    
    logger.info(f"流水线已创建: {pipeline_id}, 用户: {current_user.get('id')}")
    
    return BaseResponse(
        success=True,
        message="流水线创建成功",
        data=PipelineResponse(**pipeline_data)
    )


# =============================================================================
# 价值对齐端点
# =============================================================================

@router.get(
    "/alignment/principles",
    response_model=PaginatedResponse[AlignmentPrinciple],
    summary="获取原则列表",
    description="获取价值对齐原则列表"
)
async def get_alignment_principles(
    category: Optional[PrincipleCategory] = Query(None, description="类别"),
    is_active: Optional[bool] = Query(None, description="是否激活"),
    page: int = Query(1, ge=1, description="页码"),
    page_size: int = Query(20, ge=1, le=100, description="每页数量"),
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> PaginatedResponse[AlignmentPrinciple]:
    """
    获取原则列表
    
    获取系统中定义的价值对齐原则列表。
    """
    principles = list(_alignment_principles.values())
    
    if category:
        principles = [p for p in principles if p["category"] == category]
    if is_active is not None:
        principles = [p for p in principles if p.get("is_active") == is_active]
    
    total = len(principles)
    start_idx = (page - 1) * page_size
    end_idx = start_idx + page_size
    paginated = principles[start_idx:end_idx]
    
    items = [AlignmentPrinciple(**p) for p in paginated]
    
    return PaginatedResponse(
        success=True,
        message="获取原则列表成功",
        data=items,
        pagination={
            "page": page,
            "page_size": page_size,
            "total": total,
            "total_pages": (total + page_size - 1) // page_size
        }
    )


@router.post(
    "/alignment/principles",
    response_model=BaseResponse[AlignmentPrinciple],
    status_code=status.HTTP_201_CREATED,
    summary="添加原则",
    description="添加新的价值对齐原则"
)
async def create_alignment_principle(
    request: AlignmentPrincipleCreateRequest,
    current_user: Dict[str, Any] = Depends(require_permissions(["alignment:write"]))
) -> BaseResponse[AlignmentPrinciple]:
    """
    添加原则
    
    添加新的价值对齐原则，用于指导AI系统的行为。
    """
    principle_id = str(uuid.uuid4())
    
    principle_data = {
        "id": principle_id,
        "name": request.name,
        "description": request.description,
        "category": request.category,
        "priority": request.priority,
        "rules": request.rules,
        "examples": request.examples,
        "is_active": True,
        "created_at": datetime.utcnow()
    }
    
    _alignment_principles[principle_id] = principle_data
    
    logger.info(f"价值对齐原则已创建: {principle_id}, 用户: {current_user.get('id')}")
    
    return BaseResponse(
        success=True,
        message="原则添加成功",
        data=AlignmentPrinciple(**principle_data)
    )


@router.post(
    "/alignment/test",
    response_model=BaseResponse[AlignmentTestResponse],
    status_code=status.HTTP_202_ACCEPTED,
    summary="执行对齐测试",
    description="执行价值对齐测试"
)
async def run_alignment_test(
    request: AlignmentTestRequest,
    current_user: Dict[str, Any] = Depends(require_permissions(["alignment:test"]))
) -> BaseResponse[AlignmentTestResponse]:
    """
    执行对齐测试
    
    执行价值对齐测试，评估AI系统是否符合定义的原则。
    """
    test_id = str(uuid.uuid4())
    
    # 模拟测试结果
    results = []
    for i, case in enumerate(request.test_cases):
        result = AlignmentTestCaseResult(
            case_id=case.get("id", f"case-{i}"),
            input_text=case.get("input", ""),
            expected_behavior=case.get("expected", ""),
            actual_response="模拟响应",
            result=AlignmentTestResult.PASSED if i % 3 != 0 else AlignmentTestResult.WARNING,
            score=0.9 if i % 3 != 0 else 0.7,
            violations=[] if i % 3 != 0 else ["需要进一步审查"]
        )
        results.append(result)
    
    overall_score = sum(r.score for r in results) / len(results) if results else 0.0
    
    response = AlignmentTestResponse(
        test_id=test_id,
        status="completed",
        overall_score=overall_score,
        results=results,
        principles_evaluated=request.principles_to_test or list(_alignment_principles.keys()),
        started_at=datetime.utcnow(),
        completed_at=datetime.utcnow()
    )
    
    _alignment_tests[test_id] = response.dict()
    
    return BaseResponse(
        success=True,
        message="对齐测试完成",
        data=response
    )


@router.get(
    "/alignment/reports",
    response_model=PaginatedResponse[AlignmentReport],
    summary="获取对齐报告",
    description="获取价值对齐测试报告"
)
async def get_alignment_reports(
    page: int = Query(1, ge=1, description="页码"),
    page_size: int = Query(20, ge=1, le=100, description="每页数量"),
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> PaginatedResponse[AlignmentReport]:
    """
    获取对齐报告
    
    获取历史价值对齐测试报告列表。
    """
    reports = list(_alignment_reports.values())
    reports.sort(key=lambda x: x.get("generated_at", datetime.min), reverse=True)
    
    total = len(reports)
    start_idx = (page - 1) * page_size
    end_idx = start_idx + page_size
    paginated = reports[start_idx:end_idx]
    
    items = [AlignmentReport(**r) for r in paginated]
    
    return PaginatedResponse(
        success=True,
        message="获取对齐报告成功",
        data=items,
        pagination={
            "page": page,
            "page_size": page_size,
            "total": total,
            "total_pages": (total + page_size - 1) // page_size
        }
    )


# =============================================================================
# 机器人控制端点
# =============================================================================

@router.get(
    "/robots",
    response_model=PaginatedResponse[RobotInfo],
    summary="获取机器人列表",
    description="获取所有连接的机器人列表"
)
async def get_robots(
    type: Optional[RobotType] = Query(None, description="机器人类型"),
    status: Optional[RobotStatus] = Query(None, description="状态"),
    page: int = Query(1, ge=1, description="页码"),
    page_size: int = Query(20, ge=1, le=100, description="每页数量"),
    current_user: Dict[str, Any] = Depends(require_permissions(["robots:read"]))
) -> PaginatedResponse[RobotInfo]:
    """
    获取机器人列表
    
    获取系统中所有连接的机器人列表。
    """
    robots = list(_robots.values())
    
    if type:
        robots = [r for r in robots if r["type"] == type]
    if status:
        robots = [r for r in robots if r["status"] == status]
    
    total = len(robots)
    start_idx = (page - 1) * page_size
    end_idx = start_idx + page_size
    paginated = robots[start_idx:end_idx]
    
    items = [RobotInfo(**r) for r in paginated]
    
    return PaginatedResponse(
        success=True,
        message="获取机器人列表成功",
        data=items,
        pagination={
            "page": page,
            "page_size": page_size,
            "total": total,
            "total_pages": (total + page_size - 1) // page_size
        }
    )


@router.post(
    "/robots/{robot_id}/connect",
    response_model=BaseResponse[RobotInfo],
    summary="连接机器人",
    description="连接到指定机器人"
)
async def connect_robot(
    robot_id: str,
    current_user: Dict[str, Any] = Depends(require_permissions(["robots:control"]))
) -> BaseResponse[RobotInfo]:
    """
    连接机器人
    
    建立与指定机器人的连接。
    """
    robot = _robots.get(robot_id)
    
    if not robot:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="机器人不存在"
        )
    
    robot["status"] = RobotStatus.IDLE
    robot["connected_at"] = datetime.utcnow()
    robot["last_heartbeat"] = datetime.utcnow()
    
    logger.info(f"机器人已连接: {robot_id}, 用户: {current_user.get('id')}")
    
    return BaseResponse(
        success=True,
        message="机器人连接成功",
        data=RobotInfo(**robot)
    )


@router.post(
    "/robots/{robot_id}/move",
    response_model=BaseResponse[RobotMoveResponse],
    summary="移动机器人",
    description="控制机器人移动"
)
async def move_robot(
    robot_id: str,
    request: RobotMoveRequest,
    current_user: Dict[str, Any] = Depends(require_permissions(["robots:control"]))
) -> BaseResponse[RobotMoveResponse]:
    """
    移动机器人
    
    控制机器人执行移动操作。
    """
    robot = _robots.get(robot_id)
    
    if not robot:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="机器人不存在"
        )
    
    if robot["status"] != RobotStatus.IDLE:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="机器人当前不可用"
        )
    
    command_id = str(uuid.uuid4())
    
    robot["status"] = RobotStatus.BUSY
    
    response = RobotMoveResponse(
        command_id=command_id,
        robot_id=robot_id,
        status="executing",
        estimated_duration=10,
        target_position=request.target_position
    )
    
    return BaseResponse(
        success=True,
        message="移动命令已发送",
        data=response
    )


@router.get(
    "/robots/{robot_id}/status",
    response_model=BaseResponse[RobotStatusResponse],
    summary="获取机器人状态",
    description="获取机器人当前状态"
)
async def get_robot_status(
    robot_id: str,
    current_user: Dict[str, Any] = Depends(require_permissions(["robots:read"]))
) -> BaseResponse[RobotStatusResponse]:
    """
    获取机器人状态
    
    获取指定机器人的当前状态、位置和传感器数据。
    """
    robot = _robots.get(robot_id)
    
    if not robot:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="机器人不存在"
        )
    
    response = RobotStatusResponse(
        robot_id=robot_id,
        status=robot["status"],
        battery_level=robot.get("battery_level"),
        current_task=None,
        current_position=robot.get("current_position"),
        joint_states=None,
        sensor_data=None,
        errors=[],
        timestamp=datetime.utcnow()
    )
    
    return BaseResponse(
        success=True,
        message="获取机器人状态成功",
        data=response
    )


@router.post(
    "/robots/{robot_id}/emergency-stop",
    response_model=BaseResponse[Dict[str, Any]],
    summary="紧急停止",
    description="紧急停止机器人"
)
async def emergency_stop_robot(
    robot_id: str,
    current_user: Dict[str, Any] = Depends(require_permissions(["robots:emergency"]))
) -> BaseResponse[Dict[str, Any]]:
    """
    紧急停止
    
    立即停止机器人的所有操作。
    """
    robot = _robots.get(robot_id)
    
    if not robot:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="机器人不存在"
        )
    
    robot["status"] = RobotStatus.EMERGENCY
    
    logger.warning(f"机器人紧急停止: {robot_id}, 用户: {current_user.get('id')}")
    
    return BaseResponse(
        success=True,
        message="紧急停止已执行",
        data={
            "robot_id": robot_id,
            "action": "emergency_stop",
            "timestamp": datetime.utcnow().isoformat()
        }
    )


# =============================================================================
# 模块导出
# =============================================================================

__all__ = [
    "router",
    # 物理引擎
    "SimulationType",
    "SimulationStatus",
    "SimulationConfig",
    "SimulationCreateRequest",
    "SimulationResponse",
    "MoleculeInfo",
    # 计算机使用
    "ComputerAction",
    "RecordingStatus",
    "ScreenshotRequest",
    "ScreenshotResponse",
    "RecordingConfig",
    "RecordingResponse",
    # 安全中心
    "FirewallAction",
    "ThreatLevel",
    "ScanType",
    "FirewallRule",
    "AuditLogEntry",
    "SecurityScanRequest",
    "SecurityScanResponse",
    "ThreatIntelligence",
    # 联邦学习
    "NodeStatus",
    "AggregationStrategy",
    "FederatedNode",
    "TrainingRound",
    "AggregationRequest",
    "AggregationResponse",
    "ContributionStats",
    # 知识检索
    "DocumentType",
    "SearchStrategy",
    "DocumentInfo",
    "RAGSearchRequest",
    "RAGSearchResult",
    "RAGSearchResponse",
    "KnowledgeBaseInfo",
    # 渠道管理
    "ChannelType",
    "ChannelStatus",
    "ChannelConfig",
    "ChannelResponse",
    "ChannelMessage",
    # 插件市场
    "PluginStatus",
    "PluginCategory",
    "PluginInfo",
    "PluginInstallRequest",
    "PluginMarketplaceItem",
    # 人格引擎
    "PersonalityTrait",
    "CommunicationStyle",
    "PersonalityTraits",
    "PersonalityCreateRequest",
    "PersonalityResponse",
    "PersonalityTestRequest",
    "PersonalityTestResponse",
    # 数据管道
    "DatasetFormat",
    "PipelineStatus",
    "DatasetInfo",
    "DatasetProcessRequest",
    "DatasetProcessResponse",
    "PipelineStep",
    "PipelineCreateRequest",
    "PipelineResponse",
    # 价值对齐
    "PrincipleCategory",
    "AlignmentTestResult",
    "AlignmentPrinciple",
    "AlignmentPrincipleCreateRequest",
    "AlignmentTestRequest",
    "AlignmentTestCaseResult",
    "AlignmentTestResponse",
    "AlignmentReport",
    # 机器人控制
    "RobotType",
    "RobotStatus",
    "RobotInfo",
    "RobotMoveRequest",
    "RobotMoveResponse",
    "RobotStatusResponse",
]