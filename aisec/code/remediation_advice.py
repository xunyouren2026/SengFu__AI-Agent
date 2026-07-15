"""
修复建议 - 针对安全问题提供修复建议
"""
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum


class FixDifficulty(Enum):
    """修复难度"""
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"
    COMPLEX = "complex"


@dataclass
class Remediation:
    """修复建议"""
    issue_type: str
    title: str
    description: str
    difficulty: FixDifficulty
    steps: List[str]
    code_example: Optional[str] = None
    references: List[str] = None
    priority: int = 1  # 1-5, 1最高


class RemediationAdvice:
    """修复建议生成器"""
    
    def __init__(self):
        self._remediation_db = self._load_remediation_db()
    
    def _load_remediation_db(self) -> Dict[str, Remediation]:
        """加载修复建议数据库"""
        return {
            "sql_injection": Remediation(
                issue_type="sql_injection",
                title="修复SQL注入漏洞",
                description="使用参数化查询替代字符串拼接，确保用户输入不会被解释为SQL代码",
                difficulty=FixDifficulty.EASY,
                steps=[
                    "识别所有使用字符串拼接构建SQL查询的位置",
                    "将字符串拼接替换为参数化查询",
                    "使用占位符（?或%s）代替直接拼接的值",
                    "确保所有用户输入都通过参数传递"
                ],
                code_example="""
# 不安全的代码
query = "SELECT * FROM users WHERE id = " + user_id

# 安全的代码（参数化查询）
query = "SELECT * FROM users WHERE id = ?"
cursor.execute(query, (user_id,))

# 使用ORM（推荐）
user = User.query.filter_by(id=user_id).first()
""",
                references=[
                    "https://owasp.org/www-community/attacks/SQL_Injection",
                    "https://bobby-tables.com/"
                ],
                priority=1
            ),
            
            "command_injection": Remediation(
                issue_type="command_injection",
                title="修复命令注入漏洞",
                description="避免使用shell=True，验证并转义所有用户输入",
                difficulty=FixDifficulty.MEDIUM,
                steps=[
                    "识别所有调用外部命令的位置",
                    "避免使用shell=True参数",
                    "将命令和参数作为列表传递",
                    "验证用户输入是否符合预期格式",
                    "使用白名单限制允许的命令和参数"
                ],
                code_example="""
# 不安全的代码
import os
os.system("ls " + user_input)

# 安全的代码
import subprocess
result = subprocess.run(
    ["ls", user_input],
    shell=False,
    capture_output=True,
    text=True
)

# 带输入验证
import re
if re.match(r'^[a-zA-Z0-9_-]+$', user_input):
    result = subprocess.run(["ls", user_input], shell=False)
""",
                references=[
                    "https://owasp.org/www-community/attacks/Command_Injection"
                ],
                priority=1
            ),
            
            "code_injection": Remediation(
                issue_type="code_injection",
                title="修复代码注入漏洞",
                description="移除eval/exec调用，使用安全的替代方案",
                difficulty=FixDifficulty.HARD,
                steps=[
                    "识别所有使用eval/exec的位置",
                    "分析代码执行的目的是什么",
                    "选择安全的替代方案：",
                    "  - 数学表达式：使用ast.literal_eval或numexpr",
                    "  - JSON解析：使用json.loads",
                    "  - 配置解析：使用configparser或yaml.safe_load",
                    "如果必须使用，严格限制输入范围"
                ],
                code_example="""
# 不安全的代码
result = eval(user_expression)

# 安全的替代方案1：数学表达式
import ast
import operator

allowed_ops = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
}

def safe_eval(expr):
    tree = ast.parse(expr, mode='eval')
    # 验证AST节点类型...
    return eval(compile(tree, '<string>', 'eval'), {"__builtins__": {}}, allowed_ops)

# 安全的替代方案2：JSON
import json
data = json.loads(user_input)  # 安全的解析
""",
                references=[
                    "https://nedbatchelder.com/blog/201206/eval_really_is_dangerous.html"
                ],
                priority=1
            ),
            
            "path_traversal": Remediation(
                issue_type="path_traversal",
                title="修复路径遍历漏洞",
                description="验证并规范化文件路径，确保用户无法访问预期目录之外的文件",
                difficulty=FixDifficulty.MEDIUM,
                steps=[
                    "使用os.path.abspath获取绝对路径",
                    "使用os.path.realpath解析符号链接",
                    "验证最终路径是否在允许的基目录内",
                    "使用白名单限制可访问的文件",
                    "避免直接使用用户输入构建路径"
                ],
                code_example="""
import os

def safe_path(base_dir, user_path):
    # 规范化路径
    base_dir = os.path.abspath(base_dir)
    target_path = os.path.abspath(os.path.join(base_dir, user_path))
    
    # 验证路径在基目录内
    if not target_path.startswith(base_dir + os.sep):
        raise ValueError("路径遍历攻击检测")
    
    return target_path

# 使用示例
safe_file = safe_path("/var/www/uploads", user_filename)
""",
                references=[
                    "https://owasp.org/www-community/attacks/Path_Traversal"
                ],
                priority=2
            ),
            
            "hardcoded_secret": Remediation(
                issue_type="hardcoded_secret",
                title="移除硬编码密钥",
                description="使用环境变量或密钥管理服务存储敏感信息",
                difficulty=FixDifficulty.EASY,
                steps=[
                    "识别代码中所有硬编码的密钥和密码",
                    "将密钥移至环境变量或配置文件",
                    "使用密钥管理服务（如AWS Secrets Manager）",
                    "确保配置文件不被提交到版本控制",
                    "轮换已暴露的密钥"
                ],
                code_example="""
# 不安全的代码
API_KEY = "sk-1234567890abcdef"
password = "my_password"

# 安全的代码：环境变量
import os
API_KEY = os.environ.get("API_KEY")
password = os.environ.get("DB_PASSWORD")

# 安全的代码：密钥管理服务
import boto3
client = boto3.client('secretsmanager')
response = client.get_secret_value(SecretId='my-secret')
secret = response['SecretString']

# 使用.env文件（开发环境）
from dotenv import load_dotenv
load_dotenv()
API_KEY = os.environ.get("API_KEY")
""",
                references=[
                    "https://owasp.org/www-project-web-security-testing-guide/"
                ],
                priority=1
            ),
            
            "insecure_deserialize": Remediation(
                issue_type="insecure_deserialize",
                title="修复不安全的反序列化",
                description="使用JSON替代pickle，或限制反序列化的对象类型",
                difficulty=FixDifficulty.MEDIUM,
                steps=[
                    "识别所有使用pickle/marshal/yaml.load的位置",
                    "评估是否可以使用JSON替代",
                    "如果必须使用pickle，限制允许的对象类型",
                    "使用yaml.safe_load替代yaml.load",
                    "验证反序列化数据的来源"
                ],
                code_example="""
# 不安全的代码
import pickle
data = pickle.loads(untrusted_data)

# 安全的替代方案1：JSON
import json
data = json.loads(untrusted_data)

# 安全的替代方案2：限制pickle
import pickle

class RestrictedUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        # 只允许特定模块和类
        if module == "collections" and name == "OrderedDict":
            return getattr(__import__(module), name)
        raise pickle.UnpicklingError(f"禁止反序列化 {module}.{name}")

def safe_loads(data):
    return RestrictedUnpickler(io.BytesIO(data)).load()

# 安全的YAML
import yaml
data = yaml.safe_load(untrusted_data)  # 使用safe_load
""",
                references=[
                    "https://owasp.org/www-community/vulnerabilities/Deserialization_of_untrusted_data"
                ],
                priority=1
            ),
            
            "xss": Remediation(
                issue_type="xss",
                title="修复跨站脚本攻击(XSS)",
                description="对用户输入进行适当的转义和编码",
                difficulty=FixDifficulty.EASY,
                steps=[
                    "识别所有输出用户输入的位置",
                    "使用模板引擎的自动转义功能",
                    "对HTML上下文使用HTML实体编码",
                    "对JavaScript上下文使用Unicode转义",
                    "设置适当的Content Security Policy (CSP)"
                ],
                code_example="""
# Python后端转义
from html import escape
safe_output = escape(user_input)

# Flask/Jinja2（自动转义）
{{ user_input }}  # 自动转义
{{ user_input|safe }}  # 不转义（谨慎使用）

# 设置CSP头
from flask import Flask, make_response
app = Flask(__name__)

@app.after_request
def set_csp(response):
    response.headers['Content-Security-Policy'] = "default-src 'self'; script-src 'self'"
    return response
""",
                references=[
                    "https://owasp.org/www-community/attacks/xss/"
                ],
                priority=2
            ),
            
            "weak_crypto": Remediation(
                issue_type="weak_crypto",
                title="使用强加密算法",
                description="替换弱加密算法和哈希函数",
                difficulty=FixDifficulty.EASY,
                steps=[
                    "识别使用MD5、SHA1的位置",
                    "使用SHA-256或更强的哈希算法",
                    "密码存储使用bcrypt或argon2",
                    "加密使用AES-256-GCM",
                    "确保使用安全的随机数生成器"
                ],
                code_example="""
# 不安全的代码
import hashlib
hash = hashlib.md5(password.encode()).hexdigest()

# 安全的代码：密码哈希
import bcrypt
hashed = bcrypt.hashpw(password.encode(), bcrypt.gensalt())
if bcrypt.checkpw(password.encode(), hashed):
    print("密码正确")

# 安全的代码：文件哈希
import hashlib
sha256_hash = hashlib.sha256(data).hexdigest()

# 安全的代码：随机数
import secrets
token = secrets.token_urlsafe(32)
""",
                references=[
                    "https://owasp.org/www-project-web-security-testing-guide/"
                ],
                priority=2
            ),
            
            "dangerous_import": Remediation(
                issue_type="dangerous_import",
                title="审查危险导入",
                description="评估并替换危险的模块导入",
                difficulty=FixDifficulty.MEDIUM,
                steps=[
                    "审查每个危险导入的使用场景",
                    "评估是否有安全的替代方案",
                    "如果必须使用，添加安全封装",
                    "记录使用原因和安全措施",
                    "定期审查和更新"
                ],
                code_example="""
# 危险导入的安全封装示例
import subprocess

def safe_subprocess_run(cmd, *args, **kwargs):
    '''安全的subprocess封装'''
    # 验证命令
    if not isinstance(cmd, list):
        raise TypeError("命令必须是列表")
    
    # 禁止shell=True
    kwargs['shell'] = False
    
    # 设置超时
    if 'timeout' not in kwargs:
        kwargs['timeout'] = 30
    
    return subprocess.run(cmd, *args, **kwargs)
""",
                priority=3
            ),
        }
    
    def get_remediation(self, issue_type: str) -> Optional[Remediation]:
        """获取修复建议"""
        return self._remediation_db.get(issue_type)
    
    def get_all_remediations(self) -> List[Remediation]:
        """获取所有修复建议"""
        return list(self._remediation_db.values())
    
    def add_remediation(self, remediation: Remediation) -> None:
        """添加自定义修复建议"""
        self._remediation_db[remediation.issue_type] = remediation
    
    def get_remediations_for_issues(self, issue_types: List[str]) -> List[Remediation]:
        """根据问题类型获取修复建议"""
        remediations = []
        for issue_type in issue_types:
            if issue_type in self._remediation_db:
                remediations.append(self._remediation_db[issue_type])
        
        # 按优先级排序
        return sorted(remediations, key=lambda r: r.priority)
    
    def to_markdown(self, remediation: Remediation) -> str:
        """转换为Markdown格式"""
        lines = [
            f"## {remediation.title}",
            "",
            f"**问题类型**: {remediation.issue_type}",
            f"**难度**: {remediation.difficulty.value}",
            f"**优先级**: {remediation.priority}",
            "",
            f"**描述**: {remediation.description}",
            "",
            "### 修复步骤",
            ""
        ]
        
        for i, step in enumerate(remediation.steps, 1):
            lines.append(f"{i}. {step}")
        
        if remediation.code_example:
            lines.extend([
                "",
                "### 代码示例",
                "",
                "```python",
                remediation.code_example.strip(),
                "```"
            ])
        
        if remediation.references:
            lines.extend([
                "",
                "### 参考资料",
                ""
            ])
            for ref in remediation.references:
                lines.append(f"- {ref}")
        
        return "\n".join(lines)
    
    def generate_fix_plan(self, issues: List[Dict[str, Any]]) -> Dict[str, Any]:
        """生成修复计划"""
        plan = {
            "total_issues": len(issues),
            "by_priority": {},
            "by_difficulty": {},
            "remediations": [],
            "estimated_effort": ""
        }
        
        issue_types = set()
        for issue in issues:
            issue_types.add(issue.get("type", "unknown"))
        
        for issue_type in issue_types:
            remediation = self.get_remediation(issue_type)
            if remediation:
                plan["remediations"].append({
                    "issue_type": issue_type,
                    "title": remediation.title,
                    "difficulty": remediation.difficulty.value,
                    "priority": remediation.priority
                })
                
                # 统计
                diff = remediation.difficulty.value
                plan["by_difficulty"][diff] = plan["by_difficulty"].get(diff, 0) + 1
                
                pri = remediation.priority
                plan["by_priority"][pri] = plan["by_priority"].get(pri, 0) + 1
        
        # 估算工作量
        total = len(plan["remediations"])
        easy = plan["by_difficulty"].get("easy", 0)
        medium = plan["by_difficulty"].get("medium", 0)
        hard = plan["by_difficulty"].get("hard", 0)
        
        hours = easy * 0.5 + medium * 2 + hard * 4
        plan["estimated_effort"] = f"约 {hours:.1f} 小时"
        
        return plan
