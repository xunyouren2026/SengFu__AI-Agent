"""
公共工作流库模块

提供工作流模板的保存、加载、搜索和管理功能。
包含内置模板。
仅使用Python标准库实现。
"""

import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from .editor import Workflow, Step


@dataclass
class WorkflowTemplate:
    """
    工作流模板数据类。
    
    Attributes:
        id: 模板ID
        name: 模板名称
        description: 模板描述
        tags: 标签列表
        workflow: 工作流对象
        usage_count: 使用次数
        rating: 评分
        created_at: 创建时间
        updated_at: 更新时间
        author: 作者
        version: 版本
    """
    id: str
    name: str
    description: str
    tags: List[str] = field(default_factory=list)
    workflow: Optional[Workflow] = None
    workflow_data: Optional[Dict[str, Any]] = None
    usage_count: int = 0
    rating: float = 0.0
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    author: str = ""
    version: str = "1.0"
    
    def __post_init__(self):
        """初始化后处理。"""
        if not self.id:
            self.id = str(uuid.uuid4())
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典。"""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "tags": self.tags,
            "workflow": self.workflow.to_dict() if self.workflow else self.workflow_data,
            "usage_count": self.usage_count,
            "rating": self.rating,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "author": self.author,
            "version": self.version,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WorkflowTemplate":
        """从字典创建。"""
        workflow_data = data.get("workflow")
        workflow = None
        
        if isinstance(workflow_data, dict):
            workflow = Workflow.from_dict(workflow_data)
        
        return cls(
            id=data.get("id", str(uuid.uuid4())),
            name=data["name"],
            description=data.get("description", ""),
            tags=data.get("tags", []),
            workflow=workflow,
            workflow_data=workflow_data if isinstance(workflow_data, dict) else None,
            usage_count=data.get("usage_count", 0),
            rating=data.get("rating", 0.0),
            created_at=data.get("created_at", time.time()),
            updated_at=data.get("updated_at", time.time()),
            author=data.get("author", ""),
            version=data.get("version", "1.0"),
        )
    
    def increment_usage(self) -> None:
        """增加使用次数。"""
        self.usage_count += 1
        self.updated_at = time.time()
    
    def update_rating(self, new_rating: float) -> None:
        """
        更新评分。
        
        Args:
            new_rating: 新评分
        """
        if self.usage_count == 0:
            self.rating = new_rating
        else:
            # 计算新的平均评分
            total = self.rating * self.usage_count
            self.usage_count += 1
            self.rating = (total + new_rating) / self.usage_count
        self.updated_at = time.time()
    
    def get_workflow(self) -> Optional[Workflow]:
        """
        获取工作流对象。
        
        Returns:
            工作流对象
        """
        if self.workflow:
            return self.workflow
        if self.workflow_data:
            return Workflow.from_dict(self.workflow_data)
        return None


class WorkflowLibrary:
    """
    公共工作流库。
    
    管理工作流模板的存储、搜索和加载。
    """
    
    def __init__(self):
        """初始化工作流库。"""
        self._templates: Dict[str, WorkflowTemplate] = {}
        self._tags_index: Dict[str, List[str]] = {}  # tag -> template_ids
        self._builtin_templates_initialized = False
    
    def _ensure_builtin_templates(self) -> None:
        """确保内置模板已初始化。"""
        if self._builtin_templates_initialized:
            return
        
        self._builtin_templates_initialized = True
        
        # 内置模板：打开浏览器并登录
        browser_login_template = self._create_browser_login_template()
        self._templates[browser_login_template.id] = browser_login_template
        self._index_tags(browser_login_template)
        
        # 内置模板：填写表单
        form_fill_template = self._create_form_fill_template()
        self._templates[form_fill_template.id] = form_fill_template
        self._index_tags(form_fill_template)
        
        # 内置模板：网页数据采集
        web_scraping_template = self._create_web_scraping_template()
        self._templates[web_scraping_template.id] = web_scraping_template
        self._index_tags(web_scraping_template)
        
        # 内置模板：批量文件下载
        batch_download_template = self._create_batch_download_template()
        self._templates[batch_download_template.id] = batch_download_template
        self._index_tags(batch_download_template)
    
    def _create_browser_login_template(self) -> WorkflowTemplate:
        """创建浏览器登录模板。"""
        steps = [
            Step(
                id=str(uuid.uuid4()),
                type="window_focus",
                params={"title": "浏览器"},
                description="打开浏览器",
                enabled=True,
            ),
            Step(
                id=str(uuid.uuid4()),
                type="mouse_click",
                params={"x": 100, "y": 50},
                description="点击地址栏",
                enabled=True,
            ),
            Step(
                id=str(uuid.uuid4()),
                type="key_type",
                params={"text": "https://example.com/login"},
                description="输入登录网址",
                enabled=True,
            ),
            Step(
                id=str(uuid.uuid4()),
                type="key_press",
                params={"key": "Enter"},
                description="按回车键",
                enabled=True,
            ),
            Step(
                id=str(uuid.uuid4()),
                type="wait",
                params={"duration": 2.0},
                description="等待页面加载",
                enabled=True,
            ),
        ]
        
        workflow = Workflow(
            id=str(uuid.uuid4()),
            name="浏览器登录流程",
            steps=steps,
        )
        
        return WorkflowTemplate(
            id="builtin_browser_login",
            name="打开浏览器并登录",
            description="打开浏览器并登录网站的完整流程",
            tags=["浏览器", "登录", "自动化"],
            workflow=workflow,
            author="系统",
            version="1.0",
        )
    
    def _create_form_fill_template(self) -> WorkflowTemplate:
        """创建表单填写模板。"""
        steps = [
            Step(
                id=str(uuid.uuid4()),
                type="mouse_click",
                params={"x": 200, "y": 300},
                description="点击用户名输入框",
                enabled=True,
            ),
            Step(
                id=str(uuid.uuid4()),
                type="key_type",
                params={"text": "${username}"},
                description="输入用户名",
                enabled=True,
            ),
            Step(
                id=str(uuid.uuid4()),
                type="key_press",
                params={"key": "Tab"},
                description="切换到密码框",
                enabled=True,
            ),
            Step(
                id=str(uuid.uuid4()),
                type="key_type",
                params={"text": "${password}"},
                description="输入密码",
                enabled=True,
            ),
            Step(
                id=str(uuid.uuid4()),
                type="mouse_click",
                params={"x": 300, "y": 400},
                description="点击提交按钮",
                enabled=True,
            ),
        ]
        
        workflow = Workflow(
            id=str(uuid.uuid4()),
            name="表单填写流程",
            steps=steps,
        )
        
        return WorkflowTemplate(
            id="builtin_form_fill",
            name="填写表单",
            description="通用表单填写流程，支持参数化输入",
            tags=["表单", "输入", "自动化"],
            workflow=workflow,
            author="系统",
            version="1.0",
        )
    
    def _create_web_scraping_template(self) -> WorkflowTemplate:
        """创建网页数据采集模板。"""
        steps = [
            Step(
                id=str(uuid.uuid4()),
                type="window_focus",
                params={"title": "浏览器"},
                description="打开目标网页",
                enabled=True,
            ),
            Step(
                id=str(uuid.uuid4()),
                type="mouse_click",
                params={"x": 100, "y": 50},
                description="点击地址栏",
                enabled=True,
            ),
            Step(
                id=str(uuid.uuid4()),
                type="key_type",
                params={"text": "${url}"},
                description="输入目标网址",
                enabled=True,
            ),
            Step(
                id=str(uuid.uuid4()),
                type="key_press",
                params={"key": "Enter"},
                description="访问网址",
                enabled=True,
            ),
            Step(
                id=str(uuid.uuid4()),
                type="wait",
                params={"duration": 3.0},
                description="等待页面加载",
                enabled=True,
            ),
            Step(
                id=str(uuid.uuid4()),
                type="screenshot",
                params={},
                description="截图保存页面",
                enabled=True,
            ),
        ]
        
        workflow = Workflow(
            id=str(uuid.uuid4()),
            name="网页数据采集流程",
            steps=steps,
        )
        
        return WorkflowTemplate(
            id="builtin_web_scraping",
            name="网页数据采集",
            description="访问网页并采集数据的流程",
            tags=["网页", "采集", "数据"],
            workflow=workflow,
            author="系统",
            version="1.0",
        )
    
    def _create_batch_download_template(self) -> WorkflowTemplate:
        """创建批量文件下载模板。"""
        steps = [
            Step(
                id=str(uuid.uuid4()),
                type="loop",
                params={"items": "${file_list}", "max_iterations": 100},
                description="循环下载文件",
                enabled=True,
            ),
            Step(
                id=str(uuid.uuid4()),
                type="mouse_click",
                params={"x": 500, "y": 300},
                description="点击下载按钮",
                enabled=True,
            ),
            Step(
                id=str(uuid.uuid4()),
                type="wait",
                params={"duration": 2.0},
                description="等待下载",
                enabled=True,
            ),
        ]
        
        workflow = Workflow(
            id=str(uuid.uuid4()),
            name="批量文件下载流程",
            steps=steps,
        )
        
        return WorkflowTemplate(
            id="builtin_batch_download",
            name="批量文件下载",
            description="批量下载多个文件的流程",
            tags=["下载", "批量", "文件"],
            workflow=workflow,
            author="系统",
            version="1.0",
        )
    
    def _index_tags(self, template: WorkflowTemplate) -> None:
        """索引模板标签。"""
        for tag in template.tags:
            if tag not in self._tags_index:
                self._tags_index[tag] = []
            if template.id not in self._tags_index[tag]:
                self._tags_index[tag].append(template.id)
    
    def save_as_template(
        self,
        workflow_id: str,
        workflow: Workflow,
        name: str,
        description: str = "",
        tags: Optional[List[str]] = None,
        author: str = "",
    ) -> WorkflowTemplate:
        """
        保存为模板。
        
        Args:
            workflow_id: 工作流ID
            workflow: 工作流对象
            name: 模板名称
            description: 模板描述
            tags: 标签列表
            author: 作者
            
        Returns:
            创建的模板对象
        """
        template_id = f"template_{workflow_id}_{int(time.time())}"
        
        template = WorkflowTemplate(
            id=template_id,
            name=name,
            description=description,
            tags=tags or [],
            workflow=workflow,
            author=author,
        )
        
        self._templates[template.id] = template
        self._index_tags(template)
        
        return template
    
    def load_template(self, template_id: str) -> Optional[Workflow]:
        """
        加载模板。
        
        Args:
            template_id: 模板ID
            
        Returns:
            工作流对象，不存在则返回None
        """
        template = self._templates.get(template_id)
        if template is None:
            self._ensure_builtin_templates()
            template = self._templates.get(template_id)
        
        if template:
            template.increment_usage()
            return template.get_workflow()
        
        return None
    
    def search_templates(
        self,
        query: Optional[str] = None,
        tags: Optional[List[str]] = None,
        limit: int = 10,
    ) -> List[WorkflowTemplate]:
        """
        搜索模板。
        
        Args:
            query: 搜索关键词（搜索名称和描述）
            tags: 标签过滤
            limit: 返回数量限制
            
        Returns:
            匹配的模板列表
        """
        self._ensure_builtin_templates()
        
        results = []
        
        for template in self._templates.values():
            # 标签过滤
            if tags:
                if not any(tag in template.tags for tag in tags):
                    continue
            
            # 关键词搜索
            if query:
                query_lower = query.lower()
                name_match = query_lower in template.name.lower()
                desc_match = query_lower in template.description.lower()
                tag_match = any(query_lower in tag.lower() for tag in template.tags)
                
                if not (name_match or desc_match or tag_match):
                    continue
            
            results.append(template)
        
        # 按使用次数和评分排序
        results.sort(key=lambda t: (t.usage_count, t.rating), reverse=True)
        
        return results[:limit]
    
    def list_templates(
        self,
        include_builtin: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        列出所有模板。
        
        Args:
            include_builtin: 是否包含内置模板
            
        Returns:
            模板信息列表
        """
        self._ensure_builtin_templates()
        
        templates = []
        for template in self._templates.values():
            if not include_builtin and template.id.startswith("builtin_"):
                continue
            
            templates.append({
                "id": template.id,
                "name": template.name,
                "description": template.description,
                "tags": template.tags,
                "usage_count": template.usage_count,
                "rating": template.rating,
                "is_builtin": template.id.startswith("builtin_"),
                "author": template.author,
                "created_at": template.created_at,
            })
        
        return templates
    
    def update_template(
        self,
        template_id: str,
        updates: Dict[str, Any],
    ) -> Optional[WorkflowTemplate]:
        """
        更新模板。
        
        Args:
            template_id: 模板ID
            updates: 更新字段
            
        Returns:
            更新后的模板，不存在则返回None
        """
        template = self._templates.get(template_id)
        if template is None:
            return None
        
        # 更新字段
        if "name" in updates:
            template.name = updates["name"]
        if "description" in updates:
            template.description = updates["description"]
        if "tags" in updates:
            # 重新索引标签
            template.tags = updates["tags"]
            self._index_tags(template)
        if "rating" in updates:
            template.update_rating(updates["rating"])
        
        template.updated_at = time.time()
        
        return template
    
    def delete_template(self, template_id: str) -> bool:
        """
        删除模板。
        
        Args:
            template_id: 模板ID
            
        Returns:
            是否成功删除（内置模板不可删除）
        """
        if template_id.startswith("builtin_"):
            return False
        
        if template_id in self._templates:
            # 从标签索引中移除
            template = self._templates[template_id]
            for tag in template.tags:
                if tag in self._tags_index and template_id in self._tags_index[tag]:
                    self._tags_index[tag].remove(template_id)
            
            del self._templates[template_id]
            return True
        
        return False
    
    def get_template(self, template_id: str) -> Optional[WorkflowTemplate]:
        """
        获取模板对象。
        
        Args:
            template_id: 模板ID
            
        Returns:
            模板对象，不存在则返回None
        """
        template = self._templates.get(template_id)
        if template is None:
            self._ensure_builtin_templates()
            template = self._templates.get(template_id)
        return template
    
    def rate_template(self, template_id: str, rating: float) -> bool:
        """
        评分模板。
        
        Args:
            template_id: 模板ID
            rating: 评分（1-5）
            
        Returns:
            是否成功评分
        """
        template = self.get_template(template_id)
        if template is None:
            return False
        
        template.update_rating(max(1.0, min(5.0, rating)))
        return True
    
    def export_template(self, template_id: str) -> Optional[str]:
        """
        导出模板为JSON字符串。
        
        Args:
            template_id: 模板ID
            
        Returns:
            JSON字符串，不存在则返回None
        """
        template = self.get_template(template_id)
        if template is None:
            return None
        
        return json.dumps(template.to_dict(), ensure_ascii=False, indent=2)
    
    def import_template(self, json_str: str) -> WorkflowTemplate:
        """
        从JSON字符串导入模板。
        
        Args:
            json_str: JSON字符串
            
        Returns:
            导入的模板
        """
        data = json.loads(json_str)
        
        # 确保ID唯一
        data["id"] = f"imported_{data.get('id', str(uuid.uuid4()))}"
        
        template = WorkflowTemplate.from_dict(data)
        
        self._templates[template.id] = template
        self._index_tags(template)
        
        return template
    
    def get_templates_by_tag(self, tag: str) -> List[WorkflowTemplate]:
        """
        按标签获取模板。
        
        Args:
            tag: 标签
            
        Returns:
            模板列表
        """
        self._ensure_builtin_templates()
        
        template_ids = self._tags_index.get(tag, [])
        return [self._templates[tid] for tid in template_ids if tid in self._templates]
    
    def get_popular_templates(self, limit: int = 5) -> List[WorkflowTemplate]:
        """
        获取热门模板。
        
        Args:
            limit: 返回数量
            
        Returns:
            热门模板列表
        """
        self._ensure_builtin_templates()
        
        sorted_templates = sorted(
            self._templates.values(),
            key=lambda t: t.usage_count,
            reverse=True,
        )
        
        return sorted_templates[:limit]
    
    def get_top_rated_templates(self, limit: int = 5) -> List[WorkflowTemplate]:
        """
        获取高评分模板。
        
        Args:
            limit: 返回数量
            
        Returns:
            高评分模板列表
        """
        self._ensure_builtin_templates()
        
        sorted_templates = sorted(
            self._templates.values(),
            key=lambda t: t.rating,
            reverse=True,
        )
        
        return sorted_templates[:limit]
    
    def get_all_tags(self) -> List[str]:
        """
        获取所有标签。
        
        Returns:
            标签列表
        """
        self._ensure_builtin_templates()
        return list(self._tags_index.keys())


# 全局单例实例
_default_library: Optional[WorkflowLibrary] = None


def get_default_library() -> WorkflowLibrary:
    """获取默认工作流库实例。"""
    global _default_library
    if _default_library is None:
        _default_library = WorkflowLibrary()
    return _default_library
