"""
Data Classifier Module - 数据分类器

提供数据敏感度自动分类能力，支持：
- 基于关键词、模式、上下文的分类
- 机器学习特征提取（TF-IDF + 关键词匹配）
- 分类级别：PUBLIC/INTERNAL/CONFIDENTIAL/RESTRICTED/SECRET
"""

import re
import json
import math
from enum import Enum
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Callable, Any, Set, Tuple, Union
from pathlib import Path
from collections import Counter


class ClassificationLevel(Enum):
    """数据分类级别"""
    PUBLIC = "public"               # 公开
    INTERNAL = "internal"           # 内部
    CONFIDENTIAL = "confidential"   # 机密
    RESTRICTED = "restricted"       # 受限
    SECRET = "secret"               # 绝密


@dataclass
class ClassificationRule:
    """分类规则"""
    name: str
    pattern: Union[str, re.Pattern]
    weight: float
    category: ClassificationLevel
    conditions: Dict[str, Any] = field(default_factory=dict)
    description: str = ""
    
    def __post_init__(self):
        if isinstance(self.pattern, str):
            self.pattern = re.compile(self.pattern, re.IGNORECASE)
    
    def match(self, text: str) -> bool:
        """检查文本是否匹配此规则"""
        if isinstance(self.pattern, re.Pattern):
            return bool(self.pattern.search(text))
        return self.pattern in text.lower()
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "name": self.name,
            "weight": self.weight,
            "category": self.category.value,
            "description": self.description
        }


@dataclass
class ClassificationResult:
    """分类结果"""
    level: ClassificationLevel
    confidence: float
    matched_rules: List[ClassificationRule]
    recommendations: List[str]
    scores: Dict[ClassificationLevel, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "level": self.level.value,
            "confidence": self.confidence,
            "matched_rules": [r.to_dict() for r in self.matched_rules],
            "recommendations": self.recommendations,
            "scores": {k.value: v for k, v in self.scores.items()}
        }


@dataclass
class DataLabel:
    """数据标签"""
    classification: ClassificationLevel
    owner: str
    retention_period: int  # 保留天数
    encryption_required: bool
    created_at: Optional[str] = None
    expires_at: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        import datetime
        if self.created_at is None:
            self.created_at = datetime.datetime.now().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "classification": self.classification.value,
            "owner": self.owner,
            "retention_period": self.retention_period,
            "encryption_required": self.encryption_required,
            "created_at": self.created_at,
            "expires_at": self.expires_at,
            "tags": self.tags,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DataLabel':
        """从字典创建"""
        return cls(
            classification=ClassificationLevel(data["classification"]),
            owner=data["owner"],
            retention_period=data["retention_period"],
            encryption_required=data["encryption_required"],
            created_at=data.get("created_at"),
            expires_at=data.get("expires_at"),
            tags=data.get("tags", []),
            metadata=data.get("metadata", {})
        )


class LabelManager:
    """标签管理器"""
    
    def __init__(self):
        self.labels: Dict[str, DataLabel] = {}
    
    def add_label(self, data_id: str, label: DataLabel):
        """添加标签"""
        self.labels[data_id] = label
    
    def remove_label(self, data_id: str) -> bool:
        """移除标签"""
        if data_id in self.labels:
            del self.labels[data_id]
            return True
        return False
    
    def get_label(self, data_id: str) -> Optional[DataLabel]:
        """获取标签"""
        return self.labels.get(data_id)
    
    def query_labels(self, classification: Optional[ClassificationLevel] = None,
                     owner: Optional[str] = None) -> Dict[str, DataLabel]:
        """查询标签"""
        results = {}
        for data_id, label in self.labels.items():
            if classification and label.classification != classification:
                continue
            if owner and label.owner != owner:
                continue
            results[data_id] = label
        return results
    
    def update_label(self, data_id: str, **kwargs) -> bool:
        """更新标签"""
        if data_id not in self.labels:
            return False
        
        label = self.labels[data_id]
        for key, value in kwargs.items():
            if hasattr(label, key):
                setattr(label, key, value)
        return True
    
    def export_labels(self, file_path: Union[str, Path]):
        """导出标签到文件"""
        data = {k: v.to_dict() for k, v in self.labels.items()}
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def import_labels(self, file_path: Union[str, Path]):
        """从文件导入标签"""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        for data_id, label_data in data.items():
            self.labels[data_id] = DataLabel.from_dict(label_data)


class DataClassifier:
    """数据分类器 - 自动分类数据敏感度"""
    
    def __init__(self):
        self.rules: List[ClassificationRule] = []
        self.label_manager = LabelManager()
        self.document_frequency: Dict[str, int] = {}
        self.total_documents: int = 0
        self._init_default_rules()
    
    def _init_default_rules(self):
        """初始化默认分类规则"""
        # 财务数据规则
        self.add_rule(ClassificationRule(
            name="financial_data",
            pattern=r'(?:银行|bank|account|balance|transaction|payment|invoice|revenue|profit|loss)',
            weight=0.8,
            category=ClassificationLevel.CONFIDENTIAL,
            description="财务相关数据"
        ))
        
        # 个人身份信息规则
        self.add_rule(ClassificationRule(
            name="personal_identity",
            pattern=r'(?:身份证|id.?card|passport|ssn|social.?security)',
            weight=0.9,
            category=ClassificationLevel.RESTRICTED,
            description="个人身份信息"
        ))
        
        # 商业机密规则
        self.add_rule(ClassificationRule(
            name="trade_secret",
            pattern=r'(?:机密|confidential|proprietary|trade.?secret|专利|patent|algorithm|formula)',
            weight=0.95,
            category=ClassificationLevel.SECRET,
            description="商业机密"
        ))
        
        # 医疗数据规则
        self.add_rule(ClassificationRule(
            name="medical_data",
            pattern=r'(?:病历|medical|health|diagnosis|treatment|patient|doctor|hospital)',
            weight=0.85,
            category=ClassificationLevel.RESTRICTED,
            description="医疗相关数据"
        ))
        
        # 法律数据规则
        self.add_rule(ClassificationRule(
            name="legal_data",
            pattern=r'(?:法律|legal|contract|agreement|lawsuit|court|attorney|litigation)',
            weight=0.75,
            category=ClassificationLevel.CONFIDENTIAL,
            description="法律相关数据"
        ))
        
        # 密码/密钥规则
        self.add_rule(ClassificationRule(
            name="credentials",
            pattern=r'(?:password|pwd|secret|key|token|api.?key|credential)',
            weight=1.0,
            category=ClassificationLevel.SECRET,
            description="凭证和密钥"
        ))
        
        # 内部信息规则
        self.add_rule(ClassificationRule(
            name="internal_info",
            pattern=r'(?:内部|internal|employee|staff|meeting|memo|memo)',
            weight=0.4,
            category=ClassificationLevel.INTERNAL,
            description="内部信息"
        ))
        
        # 公开信息规则
        self.add_rule(ClassificationRule(
            name="public_info",
            pattern=r'(?:公开|public|news|press|announcement|blog)',
            weight=0.1,
            category=ClassificationLevel.PUBLIC,
            description="公开信息"
        ))
    
    def add_rule(self, rule: ClassificationRule):
        """添加分类规则"""
        self.rules.append(rule)
    
    def remove_rule(self, name: str) -> bool:
        """移除分类规则"""
        for i, rule in enumerate(self.rules):
            if rule.name == name:
                del self.rules[i]
                return True
        return False
    
    def classify(self, text: Union[str, Dict]) -> ClassificationResult:
        """
        自动分类数据敏感度
        
        Args:
            text: 要分类的文本或字典
            
        Returns:
            分类结果
        """
        if isinstance(text, dict):
            text = json.dumps(text, ensure_ascii=False)
        
        text_lower = text.lower()
        
        # 计算各分类级别的得分
        scores: Dict[ClassificationLevel, float] = {level: 0.0 for level in ClassificationLevel}
        matched_rules: List[ClassificationRule] = []
        
        for rule in self.rules:
            if rule.match(text_lower):
                scores[rule.category] += rule.weight
                matched_rules.append(rule)
        
        # TF-IDF特征提取增强
        tfidf_scores = self._calculate_tfidf_features(text)
        for level, score in tfidf_scores.items():
            scores[level] += score * 0.3  # TF-IDF权重为0.3
        
        # 上下文分析
        context_score = self._analyze_context(text)
        for level in scores:
            scores[level] += context_score.get(level, 0) * 0.2  # 上下文权重为0.2
        
        # 确定最终分类
        if not scores or max(scores.values()) == 0:
            final_level = ClassificationLevel.PUBLIC
            confidence = 0.5
        else:
            final_level = max(scores, key=scores.get)
            max_score = scores[final_level]
            total_score = sum(scores.values())
            confidence = max_score / total_score if total_score > 0 else 0.5
        
        # 生成建议
        recommendations = self._generate_recommendations(final_level, matched_rules)
        
        return ClassificationResult(
            level=final_level,
            confidence=min(confidence, 1.0),
            matched_rules=matched_rules,
            recommendations=recommendations,
            scores=scores
        )
    
    def _calculate_tfidf_features(self, text: str) -> Dict[ClassificationLevel, float]:
        """计算TF-IDF特征"""
        # 分词（简单实现）
        words = re.findall(r'\b\w+\b', text.lower())
        word_count = Counter(words)
        total_words = len(words)
        
        if total_words == 0:
            return {level: 0.0 for level in ClassificationLevel}
        
        # 计算TF
        tf = {word: count / total_words for word, count in word_count.items()}
        
        # 简化的IDF（使用预定义的关键词）
        keywords_by_level = {
            ClassificationLevel.SECRET: ['secret', 'confidential', 'proprietary', 'password', 'key', 'token'],
            ClassificationLevel.RESTRICTED: ['ssn', 'passport', 'medical', 'health', 'patient'],
            ClassificationLevel.CONFIDENTIAL: ['financial', 'bank', 'revenue', 'contract', 'legal'],
            ClassificationLevel.INTERNAL: ['internal', 'employee', 'meeting'],
            ClassificationLevel.PUBLIC: ['public', 'news', 'announcement']
        }
        
        scores = {level: 0.0 for level in ClassificationLevel}
        
        for level, keywords in keywords_by_level.items():
            for keyword in keywords:
                if keyword in tf:
                    # 简化的TF-IDF计算
                    idf = math.log(100 / (1 + self.document_frequency.get(keyword, 1)))
                    scores[level] += tf[keyword] * idf
        
        return scores
    
    def _analyze_context(self, text: str) -> Dict[ClassificationLevel, float]:
        """分析上下文"""
        scores = {level: 0.0 for level in ClassificationLevel}
        
        # 检查敏感上下文指示词
        sensitive_contexts = {
            ClassificationLevel.SECRET: [
                r'(?:绝密|top.?secret|classified)',
                r'(?:不要分享|do.?not.?share)',
                r'(?:仅限内部|internal.?only)'
            ],
            ClassificationLevel.RESTRICTED: [
                r'(?:受限|restricted)',
                r'(?:需要授权|authorization.?required)'
            ],
            ClassificationLevel.CONFIDENTIAL: [
                r'(?:机密|confidential)',
                r'(?:保密|private)'
            ]
        }
        
        for level, patterns in sensitive_contexts.items():
            for pattern in patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    scores[level] += 0.3
        
        return scores
    
    def _generate_recommendations(self, level: ClassificationLevel, 
                                   matched_rules: List[ClassificationRule]) -> List[str]:
        """生成分类建议"""
        recommendations = []
        
        if level == ClassificationLevel.SECRET:
            recommendations.extend([
                "必须使用强加密存储",
                "限制访问权限，仅授权人员可访问",
                "启用审计日志",
                "定期轮换密钥",
                "禁止通过邮件传输"
            ])
        elif level == ClassificationLevel.RESTRICTED:
            recommendations.extend([
                "建议使用加密存储",
                "实施访问控制",
                "记录访问日志",
                "传输时使用加密通道"
            ])
        elif level == ClassificationLevel.CONFIDENTIAL:
            recommendations.extend([
                "建议加密存储",
                "限制部门内访问",
                "避免在公共场合讨论"
            ])
        elif level == ClassificationLevel.INTERNAL:
            recommendations.extend([
                "仅限公司内部使用",
                "不要对外分享"
            ])
        else:
            recommendations.append("可以公开分享")
        
        # 基于匹配的规则添加特定建议
        rule_names = {r.name for r in matched_rules}
        
        if "credentials" in rule_names:
            recommendations.append("立即更换暴露的凭证")
        if "personal_identity" in rule_names:
            recommendations.append("遵守数据保护法规（如GDPR、个人信息保护法）")
        if "medical_data" in rule_names:
            recommendations.append("遵守医疗数据保护法规（如HIPAA）")
        
        return recommendations
    
    def batch_classify(self, texts: List[str]) -> List[ClassificationResult]:
        """批量分类"""
        return [self.classify(text) for text in texts]
    
    def classify_file(self, file_path: Union[str, Path], encoding: str = 'utf-8') -> ClassificationResult:
        """分类文件"""
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")
        
        with open(file_path, 'r', encoding=encoding, errors='ignore') as f:
            content = f.read()
        
        return self.classify(content)
    
    def update_document_frequency(self, documents: List[str]):
        """更新文档频率（用于TF-IDF）"""
        self.total_documents = len(documents)
        
        word_doc_count: Dict[str, int] = {}
        for doc in documents:
            words = set(re.findall(r'\b\w+\b', doc.lower()))
            for word in words:
                word_doc_count[word] = word_doc_count.get(word, 0) + 1
        
        self.document_frequency = word_doc_count
    
    def get_classification_stats(self, texts: List[str]) -> Dict[str, Any]:
        """获取分类统计"""
        results = self.batch_classify(texts)
        
        stats = {
            "total": len(results),
            "by_level": {level.value: 0 for level in ClassificationLevel},
            "avg_confidence": 0.0,
            "high_confidence_count": 0  # confidence > 0.8
        }
        
        total_confidence = 0.0
        for result in results:
            stats["by_level"][result.level.value] += 1
            total_confidence += result.confidence
            if result.confidence > 0.8:
                stats["high_confidence_count"] += 1
        
        if results:
            stats["avg_confidence"] = total_confidence / len(results)
        
        return stats


# 便捷函数
def classify_text(text: str) -> ClassificationResult:
    """便捷函数：分类文本"""
    classifier = DataClassifier()
    return classifier.classify(text)


def classify_file(file_path: Union[str, Path]) -> ClassificationResult:
    """便捷函数：分类文件"""
    classifier = DataClassifier()
    return classifier.classify_file(file_path)


# 示例用法
if __name__ == "__main__":
    classifier = DataClassifier()
    
    # 测试文本
    test_texts = [
        "这是一份公开的新闻稿，宣布公司新产品发布。",
        "内部会议纪要：讨论Q4季度销售策略",
        "财务报表：本季度收入1000万元，利润200万元",
        "员工信息：姓名张三，身份证号110101199001011234",
        "绝密：产品核心算法实现细节，专利申请材料"
    ]
    
    print("数据分类测试结果：")
    print("=" * 60)
    
    for text in test_texts:
        result = classifier.classify(text)
        print(f"\n文本: {text[:50]}...")
        print(f"分类级别: {result.level.value}")
        print(f"置信度: {result.confidence:.2f}")
        print(f"匹配规则: {[r.name for r in result.matched_rules]}")
        print(f"建议: {result.recommendations}")
        print("-" * 60)
    
    # 统计
    print("\n分类统计：")
    stats = classifier.get_classification_stats(test_texts)
    print(json.dumps(stats, ensure_ascii=False, indent=2))
