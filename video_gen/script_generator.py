"""
脚本生成器模块 - LLM驱动的镜头脚本生成
支持本地模型和API模式
"""

import json
import re
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict


@dataclass
class Shot:
    """镜头定义"""
    id: int
    description: str
    duration: float
    camera_movement: str
    prompt: str
    negative_prompt: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Script:
    """镜头脚本"""
    title: str
    total_duration: float
    shots: List[Shot]
    metadata: Dict[str, Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "total_duration": self.total_duration,
            "shots": [s.to_dict() for s in self.shots],
            "metadata": self.metadata or {}
        }


class ScriptGenerator:
    """
    镜头脚本生成器
    基于LLM将故事文本转换为结构化的镜头脚本
    """
    
    def __init__(self, use_local: bool = True, model_name: str = "default", api_key: Optional[str] = None):
        """
        初始化脚本生成器
        
        Args:
            use_local: 是否使用本地模型
            model_name: 模型名称
            api_key: API密钥（API模式下使用）
        """
        self.use_local = use_local
        self.model_name = model_name
        self.api_key = api_key
        self.local_model = None
        
        if use_local:
            self._init_local_model()
    
    def _init_local_model(self):
        """初始化本地模型"""
        # 模拟本地模型加载
        print(f"[ScriptGenerator] 初始化本地模型: {self.model_name}")
        self.local_model = True  # 标记模型已加载
    
    def generate(self, story: str, max_shots: int = 8, total_duration: float = 60.0) -> Dict[str, Any]:
        """
        根据故事生成镜头脚本
        
        Args:
            story: 故事文本
            max_shots: 最大镜头数量
            total_duration: 总时长（秒）
        
        Returns:
            镜头脚本字典
        """
        if self.use_local:
            return self._generate_local(story, max_shots, total_duration)
        else:
            return self._generate_api(story, max_shots, total_duration)
    
    def _generate_local(self, story: str, max_shots: int, total_duration: float) -> Dict[str, Any]:
        """使用本地模型生成脚本"""
        print(f"[ScriptGenerator] 使用本地模型生成脚本")
        
        # 分析故事结构
        segments = self._analyze_story(story)
        
        # 生成镜头
        shots = []
        avg_duration = total_duration / min(len(segments), max_shots)
        
        for i, segment in enumerate(segments[:max_shots], 1):
            shot = Shot(
                id=i,
                description=segment['description'],
                duration=avg_duration,
                camera_movement=segment.get('camera', 'static'),
                prompt=segment['prompt'],
                negative_prompt=segment.get('negative', '')
            )
            shots.append(shot)
        
        # 创建脚本
        script = Script(
            title=self._extract_title(story),
            total_duration=sum(s.duration for s in shots),
            shots=shots,
            metadata={
                "source": "local_model",
                "model_name": self.model_name,
                "story_length": len(story)
            }
        )
        
        return script.to_dict()
    
    def _generate_api(self, story: str, max_shots: int, total_duration: float) -> Dict[str, Any]:
        """使用API生成脚本"""
        print(f"[ScriptGenerator] 使用API生成脚本")
        
        # 构建提示词
        prompt = self._build_prompt(story, max_shots, total_duration)
        
        # 调用API（模拟）
        # 实际实现中这里会调用OpenAI或其他API
        response = self._call_api(prompt)
        
        # 解析响应
        return self._parse_response(response)
    
    def _analyze_story(self, story: str) -> List[Dict[str, str]]:
        """
        分析故事结构，提取场景片段
        
        Args:
            story: 故事文本
        
        Returns:
            场景片段列表
        """
        segments = []
        
        # 简单的场景分割逻辑
        # 按句号、感叹号、问号分割
        sentences = re.split(r'[。！？\.\!\?]+', story)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # 合并短句形成场景
        current_segment = []
        for sentence in sentences:
            current_segment.append(sentence)
            
            # 每2-3句形成一个场景
            if len(current_segment) >= 2:
                scene_text = '。'.join(current_segment)
                
                # 提取场景信息
                segment = {
                    'description': scene_text,
                    'prompt': self._convert_to_prompt(scene_text),
                    'negative': self._extract_negative_prompt(scene_text),
                    'camera': self._suggest_camera_movement(scene_text)
                }
                segments.append(segment)
                current_segment = []
        
        # 处理剩余句子
        if current_segment:
            scene_text = '。'.join(current_segment)
            segments.append({
                'description': scene_text,
                'prompt': self._convert_to_prompt(scene_text),
                'negative': self._extract_negative_prompt(scene_text),
                'camera': self._suggest_camera_movement(scene_text)
            })
        
        return segments
    
    def _convert_to_prompt(self, text: str) -> str:
        """将描述文本转换为生成提示词"""
        # 提取关键词和描述
        # 移除停用词，保留关键描述
        
        # 简单的关键词提取
        keywords = []
        
        # 提取名词短语（简化实现）
        # 实际可以使用NLP库
        common_descriptors = [
            '美丽的', '壮观的', '神秘的', '宁静的', '激烈的',
            '阳光明媚的', '阴暗的', '繁华的', '荒凉的'
        ]
        
        # 检查描述词
        for desc in common_descriptors:
            if desc in text:
                keywords.append(desc)
        
        # 构建提示词
        prompt = f" cinematic shot, {text}, high quality, detailed, 4k"
        
        return prompt.strip()
    
    def _extract_negative_prompt(self, text: str) -> str:
        """提取负面提示词"""
        # 常见负面元素
        negatives = [
            "blurry", "low quality", "distorted", "deformed",
            "bad anatomy", "watermark", "signature", "text"
        ]
        
        return ", ".join(negatives)
    
    def _suggest_camera_movement(self, text: str) -> str:
        """建议运镜方式"""
        # 根据文本内容推荐运镜
        
        # 检测动作词
        movement_keywords = {
            'push_in': ['靠近', '走进', '特写', '聚焦'],
            'pull_out': ['远离', '后退', '全景', '拉远'],
            'pan': ['环顾', '扫视', '环顾四周'],
            'tilt': ['仰视', '俯视', '抬头', '低头'],
            'track': ['跟随', '跟踪', '追逐'],
            'dolly': ['推进', '拉远', '移动'],
            'static': ['静止', '固定', '稳定']
        }
        
        for movement, keywords in movement_keywords.items():
            for keyword in keywords:
                if keyword in text:
                    return movement
        
        return "static"
    
    def _extract_title(self, story: str) -> str:
        """从故事中提取标题"""
        # 取前20个字符作为标题
        title = story[:30].strip()
        if len(story) > 30:
            title += "..."
        return title
    
    def _build_prompt(self, story: str, max_shots: int, total_duration: float) -> str:
        """构建API提示词"""
        prompt = f"""请将以下故事转换为视频镜头脚本，要求：
- 总时长约{total_duration}秒
- 分为{max_shots}个镜头
- 每个镜头包含：描述、时长、运镜方式、生成提示词

故事内容：
{story}

请按以下JSON格式输出：
{{
    "title": "脚本标题",
    "shots": [
        {{
            "id": 1,
            "description": "镜头描述",
            "duration": 8.0,
            "camera_movement": "static/push_in/pull_out/pan/tilt/track",
            "prompt": "生成提示词",
            "negative_prompt": "负面提示词"
        }}
    ]
}}"""
        return prompt
    
    def _call_api(self, prompt: str) -> str:
        """调用API（模拟）"""
        # 实际实现中这里会调用OpenAI API或其他LLM API
        print(f"[ScriptGenerator] 调用API (模拟)")
        
        # 模拟API响应
        mock_response = """{
    "title": "Generated Script",
    "shots": [
        {
            "id": 1,
            "description": "Opening scene",
            "duration": 8.0,
            "camera_movement": "static",
            "prompt": "cinematic opening shot, high quality",
            "negative_prompt": "blurry, low quality"
        }
    ]
}"""
        return mock_response
    
    def _parse_response(self, response: str) -> Dict[str, Any]:
        """解析API响应"""
        try:
            data = json.loads(response)
            
            # 验证和补充字段
            shots = []
            for i, shot_data in enumerate(data.get('shots', []), 1):
                shot = Shot(
                    id=shot_data.get('id', i),
                    description=shot_data.get('description', ''),
                    duration=shot_data.get('duration', 5.0),
                    camera_movement=shot_data.get('camera_movement', 'static'),
                    prompt=shot_data.get('prompt', ''),
                    negative_prompt=shot_data.get('negative_prompt', '')
                )
                shots.append(shot)
            
            script = Script(
                title=data.get('title', 'Untitled'),
                total_duration=sum(s.duration for s in shots),
                shots=shots,
                metadata={"source": "api", "model_name": self.model_name}
            )
            
            return script.to_dict()
            
        except json.JSONDecodeError as e:
            print(f"[ScriptGenerator] JSON解析错误: {e}")
            # 返回默认脚本
            return self._generate_default_script()
    
    def _generate_default_script(self) -> Dict[str, Any]:
        """生成默认脚本"""
        script = Script(
            title="Default Script",
            total_duration=10.0,
            shots=[
                Shot(
                    id=1,
                    description="Default opening shot",
                    duration=10.0,
                    camera_movement="static",
                    prompt="cinematic shot, high quality, detailed",
                    negative_prompt="blurry, low quality"
                )
            ],
            metadata={"source": "default"}
        )
        return script.to_dict()
    
    def save_script(self, script: Dict[str, Any], filepath: str):
        """保存脚本到文件"""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(script, f, ensure_ascii=False, indent=2)
        print(f"[ScriptGenerator] 脚本已保存: {filepath}")
    
    def load_script(self, filepath: str) -> Dict[str, Any]:
        """从文件加载脚本"""
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)


class LensScriptParser:
    """镜头脚本解析器"""
    
    @staticmethod
    def parse_script(script_path: str) -> List[Dict[str, Any]]:
        """
        解析镜头脚本文件
        
        Args:
            script_path: 脚本文件路径
        
        Returns:
            镜头列表
        """
        with open(script_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return data.get('shots', [])
    
    @staticmethod
    def validate_script(script: Dict[str, Any]) -> bool:
        """
        验证脚本格式
        
        Args:
            script: 脚本字典
        
        Returns:
            是否有效
        """
        required_fields = ['title', 'shots']
        for field in required_fields:
            if field not in script:
                print(f"[LensScriptParser] 缺少必需字段: {field}")
                return False
        
        shots = script.get('shots', [])
        if not shots:
            print("[LensScriptParser] 脚本为空")
            return False
        
        for i, shot in enumerate(shots):
            if 'description' not in shot:
                print(f"[LensScriptParser] 镜头 {i+1} 缺少描述")
                return False
        
        return True


# 便捷函数
def generate_script_from_story(
    story: str,
    use_local: bool = True,
    model_name: str = "default",
    max_shots: int = 8,
    total_duration: float = 60.0
) -> Dict[str, Any]:
    """
    从故事生成脚本的便捷函数
    
    Args:
        story: 故事文本
        use_local: 是否使用本地模型
        model_name: 模型名称
        max_shots: 最大镜头数
        total_duration: 总时长
    
    Returns:
        镜头脚本
    """
    generator = ScriptGenerator(use_local=use_local, model_name=model_name)
    return generator.generate(story, max_shots, total_duration)


if __name__ == "__main__":
    # 测试脚本生成器
    test_story = """
    一个年轻的探险家走进神秘的森林。阳光透过树叶洒下斑驳的光影。
    他发现了一座古老的遗迹，石壁上刻满了神秘的符号。
    突然，地面开始震动，一道光芒从遗迹中心射出。
    探险家小心翼翼地靠近，发现了一块发光的水晶。
    """
    
    generator = ScriptGenerator(use_local=True)
    script = generator.generate(test_story, max_shots=5, total_duration=30.0)
    
    print("\n生成的脚本:")
    print(json.dumps(script, ensure_ascii=False, indent=2))