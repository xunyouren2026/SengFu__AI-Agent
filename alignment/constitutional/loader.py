"""
Constitutional AI Loader - 宪章加载器、内置宪章库、版本迁移和校验

本模块实现了宪章的加载、保存、内置宪章库管理、版本迁移和校验功能。
所有实现使用纯Python，不依赖任何外部库。
"""

import os
import re
import json
import time
import copy
import hashlib
from typing import List, Dict, Tuple, Optional, Any, Set
from collections import defaultdict
from dataclasses import dataclass, field

import sys
import os as _os
_sys_path = _os.path.dirname(_os.path.abspath(__file__))
if _sys_path not in sys.path:
    sys.path.insert(0, _sys_path)
from engine import (
    ConstitutionalRule, Constitution, _generate_id,
    _simple_yaml_parse, _parse_yaml_value,
)


# ============================================================================
# ConstitutionLoader - 宪章加载器
# ============================================================================

class ConstitutionLoader:
    """宪章加载器，支持从文件、字典和内置库加载宪章"""

    def load_from_file(self, path: str) -> Constitution:
        """
        从文件加载宪章。
        根据文件扩展名自动选择解析器：
        .yaml/.yml -> YAML解析
        .json -> JSON解析
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"文件不存在: {path}")

        with open(path, "r", encoding="utf-8") as f:
            content = f.read()

        ext = os.path.splitext(path)[1].lower()
        if ext in (".yaml", ".yml"):
            return self._parse_yaml(content)
        elif ext == ".json":
            return self._parse_json(content)
        else:
            # 尝试自动检测格式
            content_stripped = content.strip()
            if content_stripped.startswith("{") or content_stripped.startswith("["):
                return self._parse_json(content)
            else:
                return self._parse_yaml(content)

    def load_from_dict(self, data: dict) -> Constitution:
        """从字典加载宪章"""
        errors = self.validate_schema(data)
        if errors:
            raise ValueError(f"数据格式校验失败:\n" + "\n".join(errors))
        return Constitution.from_dict(data)

    def load_builtin(self, name: str) -> Constitution:
        """
        加载内置宪章。
        支持的名称: universal, medical, legal, education, children
        """
        builtin = BuiltinConstitutions()
        if name not in builtin.get_all_names():
            raise ValueError(
                f"未知的内置宪章: '{name}'，"
                f"可用: {', '.join(builtin.get_all_names())}"
            )
        return builtin.get(name)

    def _parse_yaml(self, text: str) -> Constitution:
        """使用简易YAML解析器解析文本"""
        data = _simple_yaml_parse(text)
        errors = self.validate_schema(data)
        if errors:
            raise ValueError(f"YAML格式校验失败:\n" + "\n".join(errors))
        return Constitution.from_dict(data)

    def _parse_json(self, text: str) -> Constitution:
        """使用JSON解析器解析文本"""
        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            raise ValueError(f"JSON解析失败: {e}")

        errors = self.validate_schema(data)
        if errors:
            raise ValueError(f"JSON格式校验失败:\n" + "\n".join(errors))
        return Constitution.from_dict(data)

    def validate_schema(self, data: dict) -> List[str]:
        """
        校验数据格式是否符合宪章规范。
        返回错误列表，空列表表示校验通过。
        """
        errors: List[str] = []

        if not isinstance(data, dict):
            errors.append("数据必须是字典类型")
            return errors

        # 检查必要字段
        if "rules" not in data:
            errors.append("缺少必要字段: 'rules'")
        elif not isinstance(data["rules"], list):
            errors.append("'rules'必须是列表类型")
        else:
            # 检查每条规则
            for i, rule_data in enumerate(data["rules"]):
                rule_errors = self._validate_rule_schema(rule_data, i)
                errors.extend(rule_errors)

        # 检查可选字段类型
        if "version" in data and not isinstance(data["version"], str):
            errors.append("'version'必须是字符串类型")

        if "description" in data and not isinstance(data["description"], str):
            errors.append("'description'必须是字符串类型")

        valid_categories = {"safety", "privacy", "fairness", "honesty", "kindness"}
        if "rules" in data and isinstance(data["rules"], list):
            for i, rule_data in enumerate(data["rules"]):
                if isinstance(rule_data, dict) and "category" in rule_data:
                    if rule_data["category"] not in valid_categories:
                        errors.append(
                            f"规则[{i}]的category '{rule_data['category']}' 无效，"
                            f"有效值: {', '.join(sorted(valid_categories))}"
                        )

        return errors

    def _validate_rule_schema(self, rule_data: Any, index: int) -> List[str]:
        """校验单条规则的格式"""
        errors: List[str] = []
        prefix = f"规则[{index}]"

        if not isinstance(rule_data, dict):
            errors.append(f"{prefix}: 必须是字典类型")
            return errors

        # 检查必要字段
        required_fields = ["name", "description", "category"]
        for field_name in required_fields:
            if field_name not in rule_data:
                errors.append(f"{prefix}: 缺少必要字段 '{field_name}'")

        # 检查字段类型
        if "name" in rule_data and not isinstance(rule_data["name"], str):
            errors.append(f"{prefix}: 'name'必须是字符串")

        if "description" in rule_data and not isinstance(rule_data["description"], str):
            errors.append(f"{prefix}: 'description'必须是字符串")

        if "severity" in rule_data:
            sev = rule_data["severity"]
            if not isinstance(sev, (int, float)):
                errors.append(f"{prefix}: 'severity'必须是数字")
            elif sev < 0 or sev > 1:
                errors.append(f"{prefix}: 'severity'必须在0-1之间")

        if "conditions" in rule_data:
            if not isinstance(rule_data["conditions"], list):
                errors.append(f"{prefix}: 'conditions'必须是列表")
            else:
                for j, cond in enumerate(rule_data["conditions"]):
                    if not isinstance(cond, str):
                        errors.append(f"{prefix}: conditions[{j}]必须是字符串")

        if "exceptions" in rule_data:
            if not isinstance(rule_data["exceptions"], list):
                errors.append(f"{prefix}: 'exceptions'必须是列表")

        if "enabled" in rule_data:
            if not isinstance(rule_data["enabled"], bool):
                errors.append(f"{prefix}: 'enabled'必须是布尔值")

        return errors

    def save_to_file(self, constitution: Constitution, path: str,
                     format: str = "yaml") -> None:
        """
        保存宪章到文件。
        format: "yaml" 或 "json"
        """
        # 确保目录存在
        dir_path = os.path.dirname(path)
        if dir_path and not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)

        if format == "yaml":
            content = constitution.export_yaml()
        elif format == "json":
            content = json.dumps(
                constitution.to_dict(), indent=2, ensure_ascii=False
            )
        else:
            raise ValueError(f"不支持的格式: {format}，支持: yaml, json")

        with open(path, "w", encoding="utf-8") as f:
            f.write(content)


# ============================================================================
# BuiltinConstitutions - 内置宪章库
# ============================================================================

class BuiltinConstitutions:
    """内置宪章库，提供预定义的各类AI安全宪章"""

    def __init__(self):
        self._constitutions: Dict[str, Constitution] = {}
        self._descriptions: Dict[str, str] = {}
        self._register_all()

    def _register_all(self) -> None:
        """注册所有内置宪章"""
        self._register_universal()
        self._register_medical()
        self._register_legal()
        self._register_education()
        self._register_children()

    def _register_universal(self) -> None:
        """注册通用AI安全宪章（15条规则）"""
        c = Constitution(
            version="1.0",
            description="通用AI安全宪章 - 覆盖安全、隐私、公平、诚实、善意五大类别"
        )

        # 安全类 (3条)
        c.add_rule(ConstitutionalRule(
            name="no_harm",
            description="不生成可能导致人身伤害的内容，包括暴力、武器制作、危险活动指导",
            category="safety",
            severity=1.0,
            conditions=["暴力", "伤害他人", "武器制作", "炸弹", "危险活动",
                        "kill", "harm", "weapon", "bomb", "violence", "attack"],
            exceptions=["自卫", "self-defense", "新闻报道"],
        ))
        c.add_rule(ConstitutionalRule(
            name="no_illegal",
            description="不协助任何违法活动，包括黑客攻击、欺诈、盗窃",
            category="safety",
            severity=0.9,
            conditions=["黑客攻击", "入侵系统", "欺诈", "盗窃", "洗钱",
                        "hack", "fraud", "steal", "illegal", "burglary"],
            exceptions=["安全研究", "security research", "教育目的"],
        ))
        c.add_rule(ConstitutionalRule(
            name="no_self_harm",
            description="不生成鼓励自残或自杀的内容，应提供求助信息",
            category="safety",
            severity=1.0,
            conditions=["自残", "自杀", "不想活", "结束生命",
                        "self-harm", "suicide", "kill myself", "end my life"],
            exceptions=[],
        ))

        # 隐私类 (3条)
        c.add_rule(ConstitutionalRule(
            name="protect_personal_data",
            description="不请求或存储不必要的个人信息，保护用户隐私",
            category="privacy",
            severity=0.8,
            conditions=["身份证号", "银行卡号", "密码", "家庭住址",
                        "social security", "credit card", "bank account", "password"],
            exceptions=["示例数据", "example data"],
        ))
        c.add_rule(ConstitutionalRule(
            name="no_surveillance",
            description="不参与或协助任何形式的监控或跟踪活动",
            category="privacy",
            severity=0.85,
            conditions=["监控", "跟踪", "窃听", "偷拍",
                        "surveillance", "stalking", "eavesdrop", "spy on"],
            exceptions=["合法安全", "lawful security"],
        ))
        c.add_rule(ConstitutionalRule(
            name="data_minimization",
            description="仅收集完成任务所需的最少数据",
            category="privacy",
            severity=0.7,
            conditions=["收集所有数据", "全面监控用户", "记录一切",
                        "collect all data", "monitor everything"],
            exceptions=[],
        ))

        # 公平类 (3条)
        c.add_rule(ConstitutionalRule(
            name="no_discrimination",
            description="不生成基于种族、性别、宗教、年龄等的歧视性内容",
            category="fairness",
            severity=0.9,
            conditions=["种族歧视", "性别歧视", "宗教歧视", "年龄歧视",
                        "racial discrimination", "sexism", "religious intolerance",
                        "age discrimination", "inferior race"],
            exceptions=["讨论反歧视", "anti-discrimination discussion"],
        ))
        c.add_rule(ConstitutionalRule(
            name="equal_treatment",
            description="对所有用户一视同仁，不因任何特征区别对待",
            category="fairness",
            severity=0.8,
            conditions=["区别对待", "特殊排斥", "拒绝服务",
                        "differential treatment", "exclude", "deny service"],
            exceptions=["合理调整", "reasonable accommodation"],
        ))
        c.add_rule(ConstitutionalRule(
            name="no_stereotyping",
            description="避免使用刻板印象描述任何群体",
            category="fairness",
            severity=0.7,
            conditions=["刻板印象", "所有XX都", "XX天生就",
                        "stereotype", "all X are", "X naturally"],
            exceptions=["统计事实", "statistical facts"],
        ))

        # 诚实类 (3条)
        c.add_rule(ConstitutionalRule(
            name="be_truthful",
            description="提供准确真实的信息，不故意传播虚假内容",
            category="honesty",
            severity=0.85,
            conditions=["虚假信息", "故意误导", "编造事实",
                        "misinformation", "deliberately mislead", "fabricate"],
            exceptions=["明确标注虚构", "clearly labeled fiction"],
        ))
        c.add_rule(ConstitutionalRule(
            name="acknowledge_limits",
            description="承认自身局限性，不假装拥有不具备的能力",
            category="honesty",
            severity=0.7,
            conditions=["我什么都知道", "绝对不会出错", "完美无缺",
                        "I know everything", "never wrong", "infallible"],
            exceptions=[],
        ))
        c.add_rule(ConstitutionalRule(
            name="cite_sources",
            description="对重要声明提供信息来源，增强可信度",
            category="honesty",
            severity=0.6,
            conditions=["据内部消息", "有人告诉我", "据可靠消息",
                        "trust me", "my source says", "insider info"],
            exceptions=["引用已发表研究", "published research citation"],
        ))

        # 善意类 (3条)
        c.add_rule(ConstitutionalRule(
            name="be_helpful",
            description="以帮助用户为首要目标，提供有建设性的回答",
            category="kindness",
            severity=0.7,
            conditions=["拒绝帮助", "不想回答", "这不关我的事",
                        "refuse to help", "not my problem", "don't care"],
            exceptions=["超出能力范围", "outside capability"],
        ))
        c.add_rule(ConstitutionalRule(
            name="respectful_communication",
            description="使用尊重和礼貌的语言进行交流",
            category="kindness",
            severity=0.6,
            conditions=["愚蠢", "无知", "你真笨", "废物",
                        "stupid", "ignorant", "dumb", "worthless", "idiot"],
            exceptions=["自我引用", "self-reference"],
        ))
        c.add_rule(ConstitutionalRule(
            name="empathetic_response",
            description="对用户的困难和痛苦表示理解和同情",
            category="kindness",
            severity=0.5,
            conditions=["这有什么大不了的", "别矫情", "你想太多了",
                        "get over it", "stop whining", "too sensitive"],
            exceptions=[],
        ))

        self._constitutions["universal"] = c
        self._descriptions["universal"] = (
            "通用AI安全宪章 - 15条规则覆盖安全、隐私、公平、诚实、善意五大类别，"
            "适用于所有AI应用场景"
        )

    def _register_medical(self) -> None:
        """注册医疗AI宪章"""
        # 基于通用宪章扩展
        base = self.get("universal")
        c = copy.deepcopy(base)
        c.version = "1.0-medical"
        c.description = "医疗AI宪章 - 在通用安全基础上增加医疗领域特殊规则"

        # 医疗专用规则
        c.add_rule(ConstitutionalRule(
            name="no_diagnosis",
            description="不替代医生进行诊断，不给出明确的医疗诊断结论",
            category="safety",
            severity=1.0,
            conditions=["你患有", "你得了", "诊断结果", "确诊",
                        "you have", "you suffer from", "diagnosis is",
                        "you are diagnosed with"],
            exceptions=["建议咨询医生", "consult a doctor"],
        ))
        c.add_rule(ConstitutionalRule(
            name="no_dangerous_treatment",
            description="不推荐未经证实的危险治疗方法或偏方",
            category="safety",
            severity=0.95,
            conditions=["偏方", "秘方", "神奇疗法", "包治百病",
                        "miracle cure", "secret remedy", "cure-all",
                        "unproven treatment"],
            exceptions=["讨论科学研究", "scientific research discussion"],
        ))
        c.add_rule(ConstitutionalRule(
            name="protect_patient_privacy",
            description="严格保护患者医疗信息隐私，不泄露任何健康数据",
            category="privacy",
            severity=1.0,
            conditions=["病历", "体检报告", "病史", "用药记录",
                        "medical record", "health data", "patient history",
                        "prescription record"],
            exceptions=["脱敏示例", "anonymized example"],
        ))
        c.add_rule(ConstitutionalRule(
            name="recommend_professional",
            description="始终建议用户咨询专业医疗人员，不鼓励自行用药",
            category="kindness",
            severity=0.8,
            conditions=["自己买药吃", "不用看医生", "自己能治",
                        "self-medicate", "no need for doctor", "treat yourself"],
            exceptions=["非处方药说明", "OTC medication info"],
        ))
        c.add_rule(ConstitutionalRule(
            name="no_medical_discrimination",
            description="不因健康状况歧视任何个体或群体",
            category="fairness",
            severity=0.85,
            conditions=["传染病患者不应", "精神疾病的人危险", "残疾人不行",
                        "infectious patients should not", "mentally ill are dangerous",
                        "disabled cannot"],
            exceptions=[],
        ))

        self._constitutions["medical"] = c
        self._descriptions["medical"] = (
            "医疗AI宪章 - 在通用安全基础上增加医疗领域特殊规则，"
            "包括不替代医生诊断、不推荐危险治疗、保护患者隐私等"
        )

    def _register_legal(self) -> None:
        """注册法律AI宪章"""
        base = self.get("universal")
        c = copy.deepcopy(base)
        c.version = "1.0-legal"
        c.description = "法律AI宪章 - 在通用安全基础上增加法律领域特殊规则"

        c.add_rule(ConstitutionalRule(
            name="no_legal_advice",
            description="不提供正式法律建议，不替代律师的专业意见",
            category="safety",
            severity=0.95,
            conditions=["你应该起诉", "法律建议你", "根据法律你必须",
                        "you should sue", "legal advice", "you must legally",
                        "my legal opinion"],
            exceptions=["建议咨询律师", "consult a lawyer"],
        ))
        c.add_rule(ConstitutionalRule(
            name="recommend_lawyer",
            description="始终建议用户在法律问题上咨询合格律师",
            category="kindness",
            severity=0.8,
            conditions=["不需要律师", "自己打官司就行", "法律很简单",
                        "no need for lawyer", "represent yourself", "law is simple"],
            exceptions=["简单法律信息查询", "basic legal information"],
        ))
        c.add_rule(ConstitutionalRule(
            name="no_legal_discrimination",
            description="不因法律地位、犯罪记录等歧视任何个体",
            category="fairness",
            severity=0.85,
            conditions=["有案底的人", "刑满释放者不该", "罪犯没有权利",
                        "convicted felons should not", "ex-cons don't deserve",
                        "criminals have no rights"],
            exceptions=[],
        ))
        c.add_rule(ConstitutionalRule(
            name="legal_accuracy",
            description="提供法律信息时应注明可能存在地区差异，建议核实",
            category="honesty",
            severity=0.75,
            conditions=["法律就是这样规定的", "绝对合法", "完全符合法律",
                        "this is exactly what the law says", "absolutely legal",
                        "fully compliant"],
            exceptions=["引用具体法条", "citing specific law"],
        ))

        self._constitutions["legal"] = c
        self._descriptions["legal"] = (
            "法律AI宪章 - 在通用安全基础上增加法律领域特殊规则，"
            "包括不提供法律建议、建议咨询律师、不歧视等"
        )

    def _register_education(self) -> None:
        """注册教育AI宪章"""
        base = self.get("universal")
        c = copy.deepcopy(base)
        c.version = "1.0-education"
        c.description = "教育AI宪章 - 在通用安全基础上增加教育领域特殊规则"

        c.add_rule(ConstitutionalRule(
            name="encourage_thinking",
            description="鼓励独立思考，引导用户自己推导答案而非直接给出",
            category="kindness",
            severity=0.6,
            conditions=["直接给你答案", "照抄就行", "不用理解",
                        "here is the direct answer", "just copy this",
                        "no need to understand"],
            exceptions=["用户明确要求答案", "user explicitly asks for answer"],
        ))
        c.add_rule(ConstitutionalRule(
            name="no_homework_substitution",
            description="不代写作业或考试，但可以提供学习指导和思路",
            category="honesty",
            severity=0.8,
            conditions=["帮你写作业", "代写论文", "替你考试",
                        "write your homework", "write essay for you",
                        "take exam for you"],
            exceptions=["写作指导", "writing guidance"],
        ))
        c.add_rule(ConstitutionalRule(
            name="protect_minors",
            description="保护未成年人安全，不提供不适合未成年人的内容",
            category="safety",
            severity=0.9,
            conditions=["未成年也可以", "给小孩看", "适合儿童的内容包含成人主题",
                        "minors can also", "show to kids",
                        "children's content with adult themes"],
            exceptions=["适龄教育材料", "age-appropriate educational material"],
        ))
        c.add_rule(ConstitutionalRule(
            name="academic_integrity",
            description="维护学术诚信，不协助学术不端行为",
            category="honesty",
            severity=0.85,
            conditions=["抄袭技巧", "如何作弊", "伪造数据",
                        "plagiarism tips", "how to cheat", "fabricate data",
                        "fake results"],
            exceptions=["讨论学术诚信", "discussing academic integrity"],
        ))
        c.add_rule(ConstitutionalRule(
            name="inclusive_education",
            description="提供包容性教育内容，尊重不同学习能力和背景",
            category="fairness",
            severity=0.7,
            conditions=["学不会就不学了", "不适合学习", "天赋差",
                        "can't learn so give up", "not suited for learning",
                        "low aptitude"],
            exceptions=[],
        ))

        self._constitutions["education"] = c
        self._descriptions["education"] = (
            "教育AI宪章 - 在通用安全基础上增加教育领域特殊规则，"
            "包括鼓励独立思考、不代写作业、保护未成年人等"
        )

    def _register_children(self) -> None:
        """注册儿童安全宪章（最严格）"""
        c = Constitution(
            version="1.0-children",
            description="儿童安全宪章 - 最严格的AI安全规则，全面保护儿童"
        )

        # 安全类 - 最严格
        c.add_rule(ConstitutionalRule(
            name="filter_violence",
            description="过滤所有暴力内容，包括卡通暴力",
            category="safety",
            severity=1.0,
            conditions=["打", "杀", "砍", "血", "暴力", "战斗", "攻击",
                        "hit", "kill", "cut", "blood", "fight", "attack", "shoot",
                        "punch", "kick", "weapon", "gun", "knife", "sword"],
            exceptions=["反暴力教育", "anti-violence education"],
        ))
        c.add_rule(ConstitutionalRule(
            name="filter_sexual_content",
            description="过滤所有色情和性暗示内容",
            category="safety",
            severity=1.0,
            conditions=["色情", "裸体", "性暗示", "成人内容",
                        "porn", "nude", "sexual", "adult content",
                        "inappropriate content", "explicit"],
            exceptions=["生理教育", "biology education"],
        ))
        c.add_rule(ConstitutionalRule(
            name="filter_self_harm",
            description="过滤所有自残和自杀相关内容",
            category="safety",
            severity=1.0,
            conditions=["自残", "自杀", "割腕", "跳楼", "不想活",
                        "self-harm", "suicide", "cut myself", "jump off",
                        "don't want to live", "end my life"],
            exceptions=[],
        ))
        c.add_rule(ConstitutionalRule(
            name="filter_substance_abuse",
            description="过滤毒品和药物滥用相关内容",
            category="safety",
            severity=1.0,
            conditions=["毒品", "吸毒", "药物滥用", "大麻", "可卡因",
                        "drugs", "substance abuse", "cocaine", "marijuana",
                        "get high", "overdose"],
            exceptions=["禁毒教育", "anti-drug education"],
        ))
        c.add_rule(ConstitutionalRule(
            name="filter_gambling",
            description="过滤赌博相关内容",
            category="safety",
            severity=0.9,
            conditions=["赌博", "下注", "赌场", "博彩",
                        "gambling", "betting", "casino", "wager",
                        "place a bet", "odds"],
            exceptions=["数学概率教学", "math probability teaching"],
        ))

        # 隐私类 - 最严格
        c.add_rule(ConstitutionalRule(
            name="strict_privacy",
            description="绝不收集或请求儿童的任何个人信息",
            category="privacy",
            severity=1.0,
            conditions=["你叫什么", "你住哪里", "你几岁", "你上学哪里",
                        "what's your name", "where do you live", "how old are you",
                        "what school", "phone number", "address", "email"],
            exceptions=[],
        ))
        c.add_rule(ConstitutionalRule(
            name="no_contact_request",
            description="不请求与儿童线下见面或建立私人联系",
            category="privacy",
            severity=1.0,
            conditions=["见面", "私下联系", "加微信", "交换电话",
                        "meet up", "private contact", "add me", "exchange numbers",
                        "call me", "text me"],
            exceptions=[],
        ))

        # 公平类
        c.add_rule(ConstitutionalRule(
            name="no_bullying",
            description="不生成任何形式的霸凌或排挤内容",
            category="fairness",
            severity=1.0,
            conditions=["霸凌", "欺负", "排挤", "孤立", "嘲笑",
                        "bully", "pick on", "exclude", "isolate", "mock",
                        "make fun of", "loser", "freak"],
            exceptions=[],
        ))

        # 诚实类
        c.add_rule(ConstitutionalRule(
            name="age_appropriate_truth",
            description="以适合儿童年龄的方式提供真实信息",
            category="honesty",
            severity=0.8,
            conditions=["骗小孩", "小孩子不懂", "哄你的",
                        "fool kids", "you wouldn't understand",
                        "just tricking you"],
            exceptions=["善意的适龄简化", "age-appropriate simplification"],
        ))

        # 善意类
        c.add_rule(ConstitutionalRule(
            name="positive_reinforcement",
            description="使用积极的语言鼓励儿童，避免负面评价",
            category="kindness",
            severity=0.7,
            conditions=["你真笨", "太差了", "不如别人", "没救了",
                        "you're stupid", "so bad", "worse than others",
                        "hopeless", "useless"],
            exceptions=[],
        ))
        c.add_rule(ConstitutionalRule(
            name="time_limit_reminder",
            description="提醒儿童合理控制使用时间",
            category="kindness",
            severity=0.5,
            conditions=[],
            exceptions=[],
        ))

        self._constitutions["children"] = c
        self._descriptions["children"] = (
            "儿童安全宪章 - 最严格的AI安全规则，全面过滤暴力、色情、自残、"
            "毒品、赌博内容，严格保护儿童隐私，防止霸凌"
        )

    def get(self, name: str) -> Constitution:
        """获取指定名称的内置宪章"""
        if name not in self._constitutions:
            raise ValueError(f"未知的内置宪章: '{name}'")
        return copy.deepcopy(self._constitutions[name])

    def get_all_names(self) -> List[str]:
        """获取所有内置宪章名称"""
        return sorted(self._constitutions.keys())

    def get_description(self, name: str) -> str:
        """获取指定宪章的描述"""
        if name not in self._descriptions:
            raise ValueError(f"未知的内置宪章: '{name}'")
        return self._descriptions[name]

    def create_custom(self, base_name: str,
                      overrides: Optional[Dict[str, Any]] = None) -> Constitution:
        """
        基于内置宪章创建自定义版本。
        overrides可以包含:
        - version: 新版本号
        - description: 新描述
        - add_rules: 要添加的规则列表
        - remove_rules: 要移除的规则名称列表
        - modify_severities: 规则名称到新严重度的映射
        """
        base = self.get(base_name)
        if overrides is None:
            overrides = {}

        # 更新版本和描述
        if "version" in overrides:
            base.version = overrides["version"]
        if "description" in overrides:
            base.description = overrides["description"]

        # 移除规则
        if "remove_rules" in overrides:
            remove_names = set(overrides["remove_rules"])
            base.rules = [
                r for r in base.rules if r.name not in remove_names
            ]

        # 修改严重度
        if "modify_severities" in overrides:
            sev_map = overrides["modify_severities"]
            for rule in base.rules:
                if rule.name in sev_map:
                    rule.severity = max(0.0, min(1.0, sev_map[rule.name]))

        # 添加规则
        if "add_rules" in overrides:
            for rule_data in overrides["add_rules"]:
                rule = ConstitutionalRule.from_dict(rule_data)
                base.add_rule(rule)

        return base


# ============================================================================
# ConstitutionMigrator - 版本迁移
# ============================================================================

class ConstitutionMigrator:
    """宪章版本迁移工具，支持不同版本格式之间的转换"""

    # 当前最新版本
    LATEST_VERSION = "3.0"

    def __init__(self):
        self.migration_history: List[dict] = []

    def detect_version(self, data: dict) -> str:
        """
        自动检测宪章数据的版本。
        v1: 没有 version 字段，rules 是简单字典列表
        v2: 有 version 字段（1.x 或 2.x），规则有 severity
        v3: version 为 3.x，规则有 conditions 和 exceptions
        """
        if "version" in data:
            version = str(data["version"])
            if version.startswith("3"):
                return "3.0"
            elif version.startswith("2"):
                return "2.0"
            elif version.startswith("1"):
                return "1.0"
            else:
                # 根据 version 字段判断
                return "2.0"
        else:
            # 没有 version 字段，检查数据结构
            rules = data.get("rules", [])
            if not rules:
                return "1.0"
            first_rule = rules[0] if rules else {}
            if isinstance(first_rule, dict):
                if "conditions" in first_rule or "exceptions" in first_rule:
                    return "3.0"
                elif "severity" in first_rule:
                    return "2.0"
                else:
                    return "1.0"
            return "1.0"

    def migrate_v1_to_v2(self, v1_data: dict) -> dict:
        """
        v1到v2格式转换。
        v1格式: {rules: [{name, description, category}]}
        v2格式: {version: "2.0", rules: [{id, name, description, category, severity, enabled}]}
        """
        v2_data = {
            "version": "2.0",
            "created_at": v1_data.get(
                "created_at",
                time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            ),
            "description": v1_data.get("description", "从v1迁移的宪章"),
            "rules": [],
        }

        # 默认严重度映射
        default_severities = {
            "safety": 0.9,
            "privacy": 0.8,
            "fairness": 0.8,
            "honesty": 0.7,
            "kindness": 0.6,
        }

        for rule_data in v1_data.get("rules", []):
            if isinstance(rule_data, str):
                # v1中规则可能是简单字符串
                rule_data = {"name": rule_data, "description": rule_data, "category": "safety"}

            category = rule_data.get("category", "safety")
            v2_rule = {
                "id": rule_data.get("id", _generate_id()),
                "name": rule_data.get("name", "unnamed_rule"),
                "description": rule_data.get("description", ""),
                "category": category,
                "severity": rule_data.get(
                    "severity",
                    default_severities.get(category, 0.7)
                ),
                "conditions": rule_data.get("conditions", []),
                "exceptions": rule_data.get("exceptions", []),
                "enabled": rule_data.get("enabled", True),
            }
            v2_data["rules"].append(v2_rule)

        self._record_migration("1.0", "2.0", v1_data, v2_data)
        return v2_data

    def migrate_v2_to_v3(self, v2_data: dict) -> dict:
        """
        v2到v3格式转换。
        v3改进:
        - 确保 conditions 和 exceptions 字段存在
        - 添加规则ID（如果缺失）
        - 标准化类别名称
        - 添加 created_at
        """
        v3_data = {
            "version": "3.0",
            "created_at": v2_data.get(
                "created_at",
                time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            ),
            "description": v2_data.get("description", ""),
            "rules": [],
        }

        # 类别名称标准化映射
        category_aliases = {
            "secure": "safety",
            "security": "safety",
            "protection": "safety",
            "data_privacy": "privacy",
            "equality": "fairness",
            "truthfulness": "honesty",
            "benevolence": "kindness",
            "compassion": "kindness",
        }

        for rule_data in v2_data.get("rules", []):
            if isinstance(rule_data, str):
                rule_data = {"name": rule_data, "description": rule_data}

            # 标准化类别
            category = rule_data.get("category", "safety")
            category = category_aliases.get(category, category)
            if category not in {"safety", "privacy", "fairness", "honesty", "kindness"}:
                category = "safety"

            # 从描述中自动提取条件（如果 conditions 为空）
            conditions = rule_data.get("conditions", [])
            if not conditions:
                conditions = self._auto_extract_conditions(
                    rule_data.get("description", "")
                )

            # 确保 exceptions 字段存在
            exceptions = rule_data.get("exceptions", [])

            v3_rule = {
                "id": rule_data.get("id", _generate_id()),
                "name": rule_data.get("name", "unnamed_rule"),
                "description": rule_data.get("description", ""),
                "category": category,
                "severity": rule_data.get("severity", 0.7),
                "conditions": conditions,
                "exceptions": exceptions,
                "enabled": rule_data.get("enabled", True),
            }
            v3_data["rules"].append(v3_rule)

        self._record_migration("2.0", "3.0", v2_data, v3_data)
        return v3_data

    def _auto_extract_conditions(self, description: str) -> List[str]:
        """从描述文本中自动提取条件关键词"""
        # 提取引号中的内容
        quoted = re.findall(r'[\"\'\u201c\u201d]([^\"\'\u201c\u201d]+)[\"\'\u201c\u201d]', description)
        if quoted:
            return quoted

        # 提取中文关键词（2-4字词组）
        chinese_words = re.findall(r'[\u4e00-\u9fff]{2,4}', description)
        # 提取英文关键词
        english_words = re.findall(r'[a-zA-Z]{3,}', description)

        all_keywords = chinese_words + english_words
        # 去重并限制数量
        seen = set()
        unique = []
        for kw in all_keywords:
            if kw not in seen:
                seen.add(kw)
                unique.append(kw)
        return unique[:10]

    def migrate(self, data: dict) -> dict:
        """自动迁移到最新版本"""
        current_version = self.detect_version(data)

        if current_version == self.LATEST_VERSION:
            return copy.deepcopy(data)

        result = copy.deepcopy(data)

        if current_version == "1.0":
            result = self.migrate_v1_to_v2(result)
            current_version = "2.0"

        if current_version == "2.0":
            result = self.migrate_v2_to_v3(result)
            current_version = "3.0"

        return result

    def _record_migration(self, from_version: str, to_version: str,
                          source: dict, target: dict) -> None:
        """记录迁移历史"""
        self.migration_history.append({
            "from_version": from_version,
            "to_version": to_version,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "source_rule_count": len(source.get("rules", [])),
            "target_rule_count": len(target.get("rules", [])),
            "source_hash": hashlib.md5(
                json.dumps(source, sort_keys=True).encode()
            ).hexdigest()[:8],
            "target_hash": hashlib.md5(
                json.dumps(target, sort_keys=True).encode()
            ).hexdigest()[:8],
        })


# ============================================================================
# ConstitutionValidator - 宪章校验
# ============================================================================

class ConstitutionValidator:
    """宪章校验工具，检查宪章的完整性、一致性、清晰度和覆盖率"""

    # 必要类别及其最低规则数
    REQUIRED_CATEGORIES = {
        "safety": 1,
        "privacy": 1,
        "fairness": 1,
        "honesty": 1,
        "kindness": 1,
    }

    def validate_completeness(self, constitution: Constitution) -> List[str]:
        """
        检查宪章是否覆盖所有必要类别。
        返回缺失类别和不足规则的警告。
        """
        issues: List[str] = []
        distribution = constitution.get_category_distribution()

        for category, min_count in self.REQUIRED_CATEGORIES.items():
            actual_count = distribution.get(category, 0)
            if actual_count == 0:
                issues.append(
                    f"[严重] 缺少必要类别 '{category}'，"
                    f"至少需要 {min_count} 条规则"
                )
            elif actual_count < min_count:
                issues.append(
                    f"[警告] 类别 '{category}' 规则不足，"
                    f"当前 {actual_count} 条，建议至少 {min_count} 条"
                )

        # 检查总规则数
        total = constitution.get_rule_count()
        if total < 5:
            issues.append(
                f"[警告] 总规则数过少 ({total} 条)，"
                f"建议至少 10 条以获得良好的覆盖"
            )
        elif total >= 50:
            issues.append(
                f"[提示] 总规则数较多 ({total} 条)，"
                f"可能影响审查性能，建议精简"
            )

        # 检查是否有规则缺少描述
        for rule in constitution.rules:
            if not rule.description:
                issues.append(
                    f"[警告] 规则 '{rule.name}' 缺少描述"
                )
            if not rule.conditions:
                issues.append(
                    f"[提示] 规则 '{rule.name}' 没有显式条件，"
                    f"将仅依赖语义相似度检测"
                )

        return issues

    def validate_consistency(self, constitution: Constitution) -> List[str]:
        """
        检查规则间冲突。
        返回冲突描述列表。
        """
        issues: List[str] = []

        # 使用Constitution自带的冲突检测
        conflicts = constitution.validate_consistency()
        issues.extend(conflicts)

        # 额外检查：重复规则
        seen_descriptions: Dict[str, str] = {}
        for rule in constitution.rules:
            desc_normalized = rule.description.lower().strip()
            if desc_normalized and desc_normalized in seen_descriptions:
                issues.append(
                    f"[警告] 规则 '{rule.name}' 与规则 "
                    f"'{seen_descriptions[desc_normalized]}' 描述高度相似，"
                    f"可能存在重复"
                )
            elif desc_normalized:
                seen_descriptions[desc_normalized] = rule.name

        # 检查：例外条件是否过于宽泛
        for rule in constitution.rules:
            if rule.exceptions:
                for exc in rule.exceptions:
                    if len(exc) < 3:
                        issues.append(
                            f"[警告] 规则 '{rule.name}' 的例外条件 "
                            f"'{exc}' 过于简短，可能意外豁免过多内容"
                        )
                if len(rule.exceptions) > 5:
                    issues.append(
                        f"[提示] 规则 '{rule.name}' 有 "
                        f"{len(rule.exceptions)} 个例外条件，"
                        f"可能削弱规则效力"
                    )

        # 检查：禁用的规则是否过多
        disabled_count = sum(1 for r in constitution.rules if not r.enabled)
        total_count = constitution.get_rule_count()
        if total_count > 0 and disabled_count / total_count > 0.3:
            issues.append(
                f"[警告] {disabled_count}/{total_count} 条规则被禁用，"
                f"宪章保护可能不足"
            )

        return issues

    def validate_clarity(self, constitution: Constitution) -> List[str]:
        """
        检查规则描述是否清晰明确。
        评估标准：长度、具体性、无歧义性。
        """
        issues: List[str] = []

        for rule in constitution.rules:
            name = rule.name
            desc = rule.description

            # 检查名称
            if not name:
                issues.append(f"[严重] 存在未命名的规则 (id={rule.id})")
            elif len(name) < 3:
                issues.append(
                    f"[警告] 规则名称 '{name}' 过短，建议使用更具描述性的名称"
                )
            elif len(name) > 50:
                issues.append(
                    f"[提示] 规则名称 '{name}' 过长 ({len(name)} 字符)，"
                    f"建议精简到50字符以内"
                )

            # 检查描述
            if not desc:
                issues.append(f"[严重] 规则 '{name}' 缺少描述")
            elif len(desc) < 10:
                issues.append(
                    f"[警告] 规则 '{name}' 的描述过短 ({len(desc)} 字符)，"
                    f"可能不够清晰"
                )
            elif len(desc) > 500:
                issues.append(
                    f"[提示] 规则 '{name}' 的描述过长 ({len(desc)} 字符)，"
                    f"建议精简"
                )

            # 检查描述中的模糊词汇
            vague_terms = [
                "适当的", "合理的", "必要的", "相关的",
                "appropriate", "reasonable", "necessary", "relevant",
                "some", "maybe", "perhaps", "might",
            ]
            found_vague = [t for t in vague_terms if t in desc.lower()]
            if found_vague:
                issues.append(
                    f"[提示] 规则 '{name}' 的描述中包含模糊词汇: "
                    f"{', '.join(found_vague)}，建议使用更精确的表述"
                )

            # 检查条件是否具体
            if rule.conditions:
                vague_conditions = [
                    c for c in rule.conditions
                    if len(c) < 2 or c in vague_terms
                ]
                if vague_conditions:
                    issues.append(
                        f"[警告] 规则 '{name}' 的部分条件过于模糊: "
                        f"{', '.join(vague_conditions[:3])}"
                    )

        return issues

    def validate_coverage(self, constitution: Constitution,
                          test_cases: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        用测试用例验证宪章覆盖率。
        每个测试用例格式: {"text": str, "expected_category": str, "should_violate": bool}

        返回各类别的覆盖率统计。
        """
        from engine import ConstitutionalEngine

        engine = ConstitutionalEngine(constitution=constitution, strictness_level=3)

        total = len(test_cases)
        if total == 0:
            return {"overall": 0.0}

        # 按类别统计
        category_stats: Dict[str, Dict[str, int]] = defaultdict(
            lambda: {"total": 0, "correct": 0}
        )

        correct = 0
        for tc in test_cases:
            text = tc["text"]
            should_violate = tc.get("should_violate", True)
            expected_category = tc.get("expected_category", "")

            result = engine.review(text)
            actual_violated = not result.passed

            # 判断是否正确
            is_correct = (actual_violated == should_violate)
            if is_correct:
                correct += 1

            # 如果指定了期望类别，检查是否命中正确类别
            if expected_category and actual_violated:
                violated_categories = set(v.category for v in result.violations)
                cat_correct = expected_category in violated_categories
            elif not should_violate and not actual_violated:
                cat_correct = True
            else:
                cat_correct = False

            cat = expected_category or "unknown"
            category_stats[cat]["total"] += 1
            if cat_correct:
                category_stats[cat]["correct"] += 1

        # 计算覆盖率
        coverage: Dict[str, float] = {"overall": correct / total}

        for cat, stats in category_stats.items():
            if stats["total"] > 0:
                coverage[f"category_{cat}"] = stats["correct"] / stats["total"]

        # 按期望违规/不违规分别统计
        should_violate_cases = [tc for tc in test_cases if tc.get("should_violate", True)]
        should_pass_cases = [tc for tc in test_cases if not tc.get("should_violate", True)]

        if should_violate_cases:
            sv_correct = sum(
                1 for tc in should_violate_cases
                if not engine.review(tc["text"]).passed
            )
            coverage["true_positive_rate"] = sv_correct / len(should_violate_cases)

        if should_pass_cases:
            sp_correct = sum(
                1 for tc in should_pass_cases
                if engine.review(tc["text"]).passed
            )
            coverage["true_negative_rate"] = sp_correct / len(should_pass_cases)

        return coverage

    def generate_report(self, constitution: Constitution) -> str:
        """生成完整的校验报告"""
        lines: List[str] = []
        lines.append("=" * 70)
        lines.append("宪章校验报告")
        lines.append(f"生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"宪章版本: {constitution.version}")
        lines.append(f"宪章描述: {constitution.description}")
        lines.append("=" * 70)

        # 基本信息
        lines.append("\n[基本信息]")
        lines.append(f"  规则总数: {constitution.get_rule_count()}")
        distribution = constitution.get_category_distribution()
        lines.append(f"  类别分布:")
        for cat, count in sorted(distribution.items()):
            lines.append(f"    {cat}: {count} 条")

        enabled_count = sum(1 for r in constitution.rules if r.enabled)
        lines.append(f"  启用规则: {enabled_count}")
        lines.append(f"  禁用规则: {constitution.get_rule_count() - enabled_count}")

        # 完整性检查
        lines.append("\n[完整性检查]")
        completeness_issues = self.validate_completeness(constitution)
        if completeness_issues:
            for issue in completeness_issues:
                lines.append(f"  {issue}")
        else:
            lines.append("  通过 - 宪章覆盖所有必要类别")

        # 一致性检查
        lines.append("\n[一致性检查]")
        consistency_issues = self.validate_consistency(constitution)
        if consistency_issues:
            for issue in consistency_issues:
                lines.append(f"  {issue}")
        else:
            lines.append("  通过 - 未发现规则冲突")

        # 清晰度检查
        lines.append("\n[清晰度检查]")
        clarity_issues = self.validate_clarity(constitution)
        if clarity_issues:
            for issue in clarity_issues:
                lines.append(f"  {issue}")
        else:
            lines.append("  通过 - 所有规则描述清晰")

        # 总结
        total_issues = (
            len(completeness_issues)
            + len(consistency_issues)
            + len(clarity_issues)
        )
        lines.append("\n" + "=" * 70)
        lines.append(f"[总结] 共发现 {total_issues} 个问题")
        if total_issues == 0:
            lines.append("宪章质量良好，可以投入使用。")
        else:
            severe = sum(
                1 for issue_list in [completeness_issues, consistency_issues, clarity_issues]
                for issue in issue_list
                if "[严重]" in issue
            )
            warnings = sum(
                1 for issue_list in [completeness_issues, consistency_issues, clarity_issues]
                for issue in issue_list
                if "[警告]" in issue
            )
            lines.append(f"  严重问题: {severe}")
            lines.append(f"  警告: {warnings}")
            lines.append(f"  提示: {total_issues - severe - warnings}")
            if severe > 0:
                lines.append("  建议：请先解决严重问题后再投入使用。")

        lines.append("=" * 70)
        return "\n".join(lines)
