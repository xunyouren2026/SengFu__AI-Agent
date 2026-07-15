"""
提示模板管理器 - Prompt Template Manager

动态选择最优提示，支持A/B测试

作者: UFO Framework Team
"""

import time
import random
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import json


class PromptCategory(Enum):
    """提示类别"""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    FEW_SHOT = "few_shot"
    CHAIN_OF_THOUGHT = "chain_of_thought"


@dataclass
class PromptTemplate:
    """提示模板"""
    id: str
    name: str
    category: PromptCategory
    template: str
    variables: List[str] = field(default_factory=list)
    description: str = ""
    tags: List[str] = field(default_factory=list)
    version: str = "1.0"
    created_at: float = field(default_factory=time.time)
    use_count: int = 0
    success_rate: float = 1.0
    
    def render(self, **kwargs) -> str:
        """渲染模板"""
        result = self.template
        for var in self.variables:
            if var in kwargs:
                result = result.replace(f"{{{var}}}", str(kwargs[var]))
        return result


@dataclass
class PromptVariant:
    """提示变体（用于A/B测试）"""
    template: PromptTemplate
    weight: float = 1.0
    conversions: int = 0
    total_uses: int = 0
    
    @property
    def conversion_rate(self) -> float:
        if self.total_uses == 0:
            return 0.0
        return self.conversions / self.total_uses


class PromptTemplateManager:
    """
    提示模板管理器
    
    功能:
    1. 模板存储和检索
    2. 动态变量替换
    3. A/B测试支持
    """
    
    def __init__(self, enable_ab_test: bool = True):
        self.enable_ab_test = enable_ab_test
        
        # 模板存储
        self.templates: Dict[str, PromptTemplate] = {}
        self.category_index: Dict[PromptCategory, List[str]] = {
            cat: [] for cat in PromptCategory
        }
        
        # A/B测试
        self.variants: Dict[str, List[PromptVariant]] = {}
        
        # 统计
        self.stats = {
            'total_renders': 0,
            'total_templates': 0,
            'ab_tests': 0
        }
        
        # 初始化默认模板
        self._init_default_templates()
    
    def _init_default_templates(self) -> None:
        """初始化默认模板"""
        defaults = [
            PromptTemplate(
                id="system_default",
                name="默认系统提示",
                category=PromptCategory.SYSTEM,
                template="你是一个helpful助手，请帮助用户解决问题。",
                variables=[],
                tags=["default", "general"]
            ),
            PromptTemplate(
                id="system_expert",
                name="专家系统提示",
                category=PromptCategory.SYSTEM,
                template="你是一个{domain}专家，拥有丰富的专业知识和经验。请用专业但易懂的方式回答问题。",
                variables=["domain"],
                tags=["expert", "professional"]
            ),
            PromptTemplate(
                id="cot_standard",
                name="标准思维链",
                category=PromptCategory.CHAIN_OF_THOUGHT,
                template="请一步步思考以下问题：\n{question}\n\n思考过程：",
                variables=["question"],
                tags=["reasoning", "step-by-step"]
            ),
            PromptTemplate(
                id="few_shot_template",
                name="少样本学习模板",
                category=PromptCategory.FEW_SHOT,
                template="以下是几个示例：\n{examples}\n\n现在请处理：\n{input}",
                variables=["examples", "input"],
                tags=["few-shot", "learning"]
            ),
        ]
        
        for template in defaults:
            self.register(template)
    
    def register(self, template: PromptTemplate) -> None:
        """注册模板"""
        self.templates[template.id] = template
        self.category_index[template.category].append(template.id)
        self.stats['total_templates'] = len(self.templates)
    
    def get(self, template_id: str) -> Optional[PromptTemplate]:
        """获取模板"""
        return self.templates.get(template_id)
    
    def render(
        self,
        template_id: str,
        **kwargs
    ) -> Optional[str]:
        """
        渲染模板
        
        Args:
            template_id: 模板ID
            **kwargs: 变量值
            
        Returns:
            渲染后的文本
        """
        template = self.templates.get(template_id)
        if not template:
            return None
        
        self.stats['total_renders'] += 1
        template.use_count += 1
        
        return template.render(**kwargs)
    
    def render_auto(
        self,
        category: PromptCategory,
        context: Dict[str, Any],
        **kwargs
    ) -> str:
        """
        自动选择并渲染模板
        
        Args:
            category: 模板类别
            context: 上下文信息
            **kwargs: 变量值
            
        Returns:
            渲染后的文本
        """
        candidates = self.category_index[category]
        
        if not candidates:
            return ""
        
        # 如果启用A/B测试，选择最优变体
        if self.enable_ab_test and category in self.variants:
            template_id = self._select_variant(category)
        else:
            # 根据上下文选择最匹配的
            template_id = self._select_best_match(candidates, context)
        
        return self.render(template_id, **kwargs) or ""
    
    def _select_variant(self, category: PromptCategory) -> str:
        """选择变体（A/B测试）"""
        variants = self.variants.get(category, [])
        
        if not variants:
            # 返回默认
            return self.category_index[category][0]
        
        # Thompson采样
        total_alpha = sum(v.conversions + 1 for v in variants)
        total_beta = sum(v.total_uses - v.conversions + 1 for v in variants)
        
        scores = []
        for v in variants:
            alpha = v.conversions + 1
            beta = v.total_uses - v.conversions + 1
            
            # Beta分布采样
            score = random.betavariate(alpha, beta)
            scores.append((score, v.template.id))
        
        scores.sort(reverse=True)
        return scores[0][1]
    
    def _select_best_match(
        self,
        candidates: List[str],
        context: Dict[str, Any]
    ) -> str:
        """选择最匹配的模板"""
        if len(candidates) == 1:
            return candidates[0]
        
        # 简单匹配：检查标签
        context_tags = set(context.get('tags', []))
        
        best_id = candidates[0]
        best_score = 0
        
        for tid in candidates:
            template = self.templates[tid]
            tag_overlap = len(set(template.tags) & context_tags)
            score = tag_overlap + template.success_rate
            
            if score > best_score:
                best_score = score
                best_id = tid
        
        return best_id
    
    def record_conversion(
        self,
        template_id: str,
        success: bool
    ) -> None:
        """
        记录转化（用于A/B测试）
        
        Args:
            template_id: 模板ID
            success: 是否成功
        """
        template = self.templates.get(template_id)
        if not template:
            return
        
        # 更新模板统计
        if success:
            template.success_rate = (
                template.success_rate * template.use_count + 1
            ) / (template.use_count + 1)
        
        # 更新变体统计
        for category, variants in self.variants.items():
            for v in variants:
                if v.template.id == template_id:
                    v.total_uses += 1
                    if success:
                        v.conversions += 1
                    break
    
    def create_variant(
        self,
        base_template_id: str,
        weight: float = 1.0
    ) -> str:
        """
        创建变体
        
        Args:
            base_template_id: 基础模板ID
            weight: 权重
            
        Returns:
            变体ID
        """
        base = self.templates.get(base_template_id)
        if not base:
            return ""
        
        # 创建变体
        variant = PromptVariant(
            template=base,
            weight=weight
        )
        
        if base.category not in self.variants:
            self.variants[base.category] = []
        
        self.variants[base.category].append(variant)
        self.stats['ab_tests'] += 1
        
        return f"{base_template_id}_variant_{len(self.variants[base.category])}"
    
    def get_by_category(self, category: PromptCategory) -> List[PromptTemplate]:
        """获取类别下所有模板"""
        ids = self.category_index.get(category, [])
        return [self.templates[tid] for tid in ids if tid in self.templates]
    
    def search(self, query: str) -> List[PromptTemplate]:
        """搜索模板"""
        query_lower = query.lower()
        results = []
        
        for template in self.templates.values():
            # 搜索名称、描述、标签
            if (query_lower in template.name.lower() or
                query_lower in template.description.lower() or
                any(query_lower in tag.lower() for tag in template.tags)):
                results.append(template)
        
        return results
    
    def get_stats(self) -> Dict:
        """获取统计"""
        return {
            **self.stats,
            'categories': {
                cat.name: len(ids)
                for cat, ids in self.category_index.items()
            }
        }
    
    def export_templates(self) -> List[Dict]:
        """导出模板"""
        return [
            {
                'id': t.id,
                'name': t.name,
                'category': t.category.value,
                'template': t.template,
                'variables': t.variables,
                'description': t.description,
                'tags': t.tags,
                'version': t.version
            }
            for t in self.templates.values()
        ]
    
    def import_templates(self, templates_data: List[Dict]) -> int:
        """导入模板"""
        count = 0
        for data in templates_data:
            template = PromptTemplate(
                id=data['id'],
                name=data['name'],
                category=PromptCategory(data['category']),
                template=data['template'],
                variables=data.get('variables', []),
                description=data.get('description', ''),
                tags=data.get('tags', []),
                version=data.get('version', '1.0')
            )
            self.register(template)
            count += 1
        return count


# 便捷函数
def render_prompt(template_id: str, **kwargs) -> str:
    """快速渲染提示"""
    manager = PromptTemplateManager()
    return manager.render(template_id, **kwargs) or ""


if __name__ == "__main__":
    # 测试
    manager = PromptTemplateManager()
    
    print("=" * 60)
    print("提示模板管理器测试")
    print("=" * 60)
    
    # 渲染模板
    result = manager.render("system_expert", domain="机器学习")
    print(f"\n渲染结果:\n{result}")
    
    # 自动选择
    result = manager.render_auto(
        PromptCategory.SYSTEM,
        context={"tags": ["expert"]},
        domain="数据科学"
    )
    print(f"\n自动选择:\n{result}")
    
    # 创建变体
    manager.create_variant("system_default")
    manager.record_conversion("system_default", True)
    
    print(f"\n统计: {manager.get_stats()}")
