"""
Visual Question Answering Tool - 视觉问答工具
基于图像内容回答问题
"""

import json
import base64
import urllib.request
import urllib.error
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class VQATaskType(Enum):
    """VQA任务类型枚举"""
    GENERAL = "general"
    COUNTING = "counting"
    SPATIAL = "spatial"
    ATTRIBUTE = "attribute"
    COMPARISON = "comparison"
    BOOLEAN = "boolean"


@dataclass
class BoundingBox:
    """边界框"""
    x: float
    y: float
    width: float
    height: float
    
    def to_dict(self) -> Dict[str, float]:
        return {"x": self.x, "y": self.y, "width": self.width, "height": self.height}
    
    def area(self) -> float:
        return self.width * self.height
    
    def intersection(self, other: "BoundingBox") -> Optional["BoundingBox"]:
        """计算交集"""
        x1 = max(self.x, other.x)
        y1 = max(self.y, other.y)
        x2 = min(self.x + self.width, other.x + other.width)
        y2 = min(self.y + self.height, other.y + other.height)
        
        if x2 > x1 and y2 > y1:
            return BoundingBox(x1, y1, x2 - x1, y2 - y1)
        return None
    
    def iou(self, other: "BoundingBox") -> float:
        """计算IoU"""
        intersection = self.intersection(other)
        if intersection is None:
            return 0.0
        
        intersection_area = intersection.area()
        union_area = self.area() + other.area() - intersection_area
        
        return intersection_area / union_area if union_area > 0 else 0.0


@dataclass
class VQAQuestion:
    """VQA问题"""
    question: str
    question_type: VQATaskType = VQATaskType.GENERAL
    options: List[str] = field(default_factory=list)
    context: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "question": self.question,
            "type": self.question_type.value,
            "options": self.options,
            "context": self.context
        }


@dataclass
class VQAAnswer:
    """VQA答案"""
    answer: str
    confidence: float
    question_type: VQATaskType
    reasoning: Optional[str] = None
    supporting_regions: List[BoundingBox] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "answer": self.answer,
            "confidence": self.confidence,
            "question_type": self.question_type.value,
            "reasoning": self.reasoning,
            "supporting_regions": [r.to_dict() for r in self.supporting_regions],
            "metadata": self.metadata
        }


@dataclass
class VQAConfig:
    """VQA配置"""
    model_name: str = "default"
    max_tokens: int = 512
    temperature: float = 0.1
    top_p: float = 0.9
    use_chain_of_thought: bool = False
    api_endpoint: Optional[str] = None
    api_key: Optional[str] = None
    timeout: int = 60


class VQATool:
    """视觉问答工具"""
    
    def __init__(self, config: Optional[VQAConfig] = None):
        self.config = config or VQAConfig()
        self._question_templates: Dict[VQATaskType, List[str]] = self._init_templates()
    
    def _init_templates(self) -> Dict[VQATaskType, List[str]]:
        """初始化问题模板"""
        return {
            VQATaskType.COUNTING: [
                "How many {objects} are in the image?",
                "Count the number of {objects}.",
                "What is the total count of {objects}?"
            ],
            VQATaskType.SPATIAL: [
                "Where is the {object} located?",
                "What is the position of {object}?",
                "Is the {object1} to the left/right of {object2}?",
                "Is the {object1} above/below {object2}?"
            ],
            VQATaskType.ATTRIBUTE: [
                "What color is the {object}?",
                "What is the shape of {object}?",
                "What size is the {object}?",
                "Describe the {object}."
            ],
            VQATaskType.COMPARISON: [
                "Which {object} is larger/smaller?",
                "Are there more {object1} or {object2}?",
                "Compare {object1} and {object2}."
            ],
            VQATaskType.BOOLEAN: [
                "Is there a {object} in the image?",
                "Does the image contain {object}?",
                "Are there any {objects}?"
            ]
        }
    
    def load_image(self, image_path: str) -> bytes:
        """加载图像"""
        with open(image_path, 'rb') as f:
            return f.read()
    
    def load_image_base64(self, image_path: str) -> str:
        """加载图像并转换为base64"""
        image_data = self.load_image(image_path)
        return base64.b64encode(image_data).decode('utf-8')
    
    def encode_image(self, image_data: bytes) -> str:
        """编码图像为base64"""
        return base64.b64encode(image_data).decode('utf-8')
    
    def ask(self, image: Union[str, bytes], question: Union[str, VQAQuestion],
            **kwargs) -> VQAAnswer:
        """对图像提问"""
        # 处理输入
        if isinstance(image, str):
            image_base64 = self.load_image_base64(image)
        else:
            image_base64 = self.encode_image(image)
        
        if isinstance(question, str):
            question_obj = VQAQuestion(question=question)
        else:
            question_obj = question
        
        # 构建请求
        prompt = self._build_prompt(question_obj)
        
        # 调用模型（这里提供框架，实际实现需要连接模型）
        result = self._call_model(image_base64, prompt, **kwargs)
        
        return result
    
    def ask_batch(self, image: Union[str, bytes],
                  questions: List[Union[str, VQAQuestion]],
                  **kwargs) -> List[VQAAnswer]:
        """批量提问"""
        results = []
        for question in questions:
            answer = self.ask(image, question, **kwargs)
            results.append(answer)
        return results
    
    def _build_prompt(self, question: VQAQuestion) -> str:
        """构建提示"""
        if self.config.use_chain_of_thought:
            prompt = f"""Analyze the image and answer the following question step by step.

Question: {question.question}

Please provide:
1. Initial observations about the image
2. Relevant details for answering the question
3. Reasoning process
4. Final answer

Format your response as JSON with keys: "observations", "reasoning", "answer", "confidence"."""
        else:
            prompt = question.question
            
            if question.options:
                options_str = ", ".join(question.options)
                prompt += f"\n\nOptions: {options_str}"
            
            if question.context:
                prompt += f"\n\nContext: {question.context}"
        
        return prompt
    
    def _call_model(self, image_base64: str, prompt: str,
                    **kwargs) -> VQAAnswer:
        """调用模型"""
        # 如果配置了API端点
        if self.config.api_endpoint and self.config.api_key:
            return self._call_api(image_base64, prompt)
        
        # 否则返回模拟结果
        return VQAAnswer(
            answer="[Model response placeholder]",
            confidence=0.0,
            question_type=VQATaskType.GENERAL,
            reasoning="No model configured. Please set api_endpoint and api_key.",
            metadata={"prompt": prompt}
        )
    
    def _call_api(self, image_base64: str, prompt: str) -> VQAAnswer:
        """调用API"""
        payload = {
            "image": image_base64,
            "prompt": prompt,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
            "top_p": self.config.top_p
        }
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.config.api_key}"
        }
        
        try:
            data = json.dumps(payload).encode()
            request = urllib.request.Request(
                self.config.api_endpoint,
                data=data,
                headers=headers,
                method="POST"
            )
            
            with urllib.request.urlopen(request, timeout=self.config.timeout) as response:
                result = json.loads(response.read())
                
                return VQAAnswer(
                    answer=result.get("answer", ""),
                    confidence=result.get("confidence", 0.0),
                    question_type=VQATaskType.GENERAL,
                    reasoning=result.get("reasoning"),
                    metadata={"raw_response": result}
                )
        except Exception as e:
            logger.error(f"API call failed: {e}")
            return VQAAnswer(
                answer="",
                confidence=0.0,
                question_type=VQATaskType.GENERAL,
                reasoning=f"Error: {str(e)}"
            )
    
    def generate_question(self, task_type: VQATaskType,
                          **placeholders) -> str:
        """生成问题模板"""
        templates = self._question_templates.get(task_type, [])
        
        if not templates:
            return ""
        
        template = templates[0]
        return template.format(**placeholders)
    
    def classify_question(self, question: str) -> VQATaskType:
        """分类问题类型"""
        question_lower = question.lower()
        
        # 计数问题
        if any(kw in question_lower for kw in ["how many", "count", "number of", "total"]):
            return VQATaskType.COUNTING
        
        # 空间问题
        if any(kw in question_lower for kw in ["where", "position", "left", "right", "above", "below", "location"]):
            return VQATaskType.SPATIAL
        
        # 属性问题
        if any(kw in question_lower for kw in ["what color", "what shape", "what size", "describe", "what is the"]):
            return VQATaskType.ATTRIBUTE
        
        # 比较问题
        if any(kw in question_lower for kw in ["which", "compare", "more", "larger", "smaller", "difference"]):
            return VQATaskType.COMPARISON
        
        # 布尔问题
        if any(kw in question_lower for kw in ["is there", "does", "are there", "has"]):
            return VQATaskType.BOOLEAN
        
        return VQATaskType.GENERAL
    
    def get_supported_question_types(self) -> List[str]:
        """获取支持的问题类型"""
        return [t.value for t in VQATaskType]
    
    def analyze_image(self, image: Union[str, bytes]) -> Dict[str, Any]:
        """综合分析图像"""
        questions = [
            VQAQuestion("What is the main subject of this image?", VQATaskType.GENERAL),
            VQAQuestion("Describe the overall scene.", VQATaskType.GENERAL),
            VQAQuestion("What objects are visible in the image?", VQATaskType.GENERAL),
            VQAQuestion("What is the setting or background?", VQATaskType.SPATIAL)
        ]
        
        results = self.ask_batch(image, questions)
        
        return {
            "main_subject": results[0].answer,
            "scene_description": results[1].answer,
            "objects": results[2].answer,
            "setting": results[3].answer,
            "details": [r.to_dict() for r in results]
        }
