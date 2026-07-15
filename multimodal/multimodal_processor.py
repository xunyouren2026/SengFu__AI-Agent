"""
多模态处理器统一接口
提供统一的多模态数据处理和特征提取接口
"""
from typing import Optional, List, Dict, Any, Tuple, Union
import math
import random


class ModalityType:
    """模态类型枚举"""
    IMAGE = 'image'
    TEXT = 'text'
    AUDIO = 'audio'
    VIDEO = 'video'


class MultimodalProcessor:
    """多模态处理器
    
    统一的多模态数据处理和特征提取接口
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        config = config or {}
        
        # 模态配置
        self.enabled_modalities = config.get('enabled_modalities', 
                                             ['image', 'text', 'audio', 'video'])
        
        # 各模态的嵌入维度
        self.embed_dims = config.get('embed_dims', {
            'image': 768,
            'text': 512,
            'audio': 512,
            'video': 768
        })
        
        # 统一输出维度
        self.output_dim = config.get('output_dim', 512)
        
        # 初始化各模态处理器
        self.processors: Dict[str, Any] = {}
        self._init_processors(config)
        
        # 模态权重
        self.modality_weights = config.get('modality_weights', {
            'image': 1.0,
            'text': 1.0,
            'audio': 0.8,
            'video': 1.0
        })
    
    def _init_processors(self, config: Dict[str, Any]):
        """初始化各模态处理器"""
        # 图像处理器
        if 'image' in self.enabled_modalities:
            self.processors['image'] = ImageProcessor(
                config.get('image_config', {})
            )
        
        # 文本处理器
        if 'text' in self.enabled_modalities:
            self.processors['text'] = TextProcessor(
                config.get('text_config', {})
            )
        
        # 音频处理器
        if 'audio' in self.enabled_modalities:
            self.processors['audio'] = AudioProcessor(
                config.get('audio_config', {})
            )
        
        # 视频处理器
        if 'video' in self.enabled_modalities:
            self.processors['video'] = VideoProcessor(
                config.get('video_config', {})
            )
    
    def process(self, inputs: Dict[str, Any]) -> Dict[str, List[List[float]]]:
        """
        处理多模态输入
        
        Args:
            inputs: 各模态的输入数据
                   {'image': image_data, 'text': text_data, ...}
        
        Returns:
            各模态的特征表示
        """
        features = {}
        
        for modality, data in inputs.items():
            if modality in self.processors:
                try:
                    feat = self.processors[modality].process(data)
                    features[modality] = feat
                except Exception as e:
                    print(f"Error processing {modality}: {e}")
        
        return features
    
    def process_single(self, modality: str, data: Any) -> List[List[float]]:
        """处理单个模态"""
        if modality not in self.processors:
            raise ValueError(f"Unknown modality: {modality}")
        return self.processors[modality].process(data)
    
    def get_unified_features(self, inputs: Dict[str, Any]) -> List[float]:
        """
        获取统一的多模态特征
        
        Args:
            inputs: 各模态输入
        
        Returns:
            融合后的特征向量
        """
        features = self.process(inputs)
        
        # 加权融合
        unified = [0.0] * self.output_dim
        total_weight = 0.0
        
        for modality, feat_list in features.items():
            if feat_list:
                weight = self.modality_weights.get(modality, 1.0)
                
                # 池化
                if isinstance(feat_list[0], list):
                    pooled = [sum(f[i] for f in feat_list) / len(feat_list) 
                              for i in range(len(feat_list[0]))]
                else:
                    pooled = feat_list
                
                # 投影到统一维度
                if len(pooled) != self.output_dim:
                    pooled = self._project(pooled, self.output_dim)
                
                for i in range(len(unified)):
                    unified[i] += weight * pooled[i]
                total_weight += weight
        
        # 归一化
        if total_weight > 0:
            unified = [u / total_weight for u in unified]
        
        return unified
    
    def _project(self, features: List[float], target_dim: int) -> List[float]:
        """简单投影"""
        if len(features) == target_dim:
            return features
        
        if len(features) > target_dim:
            # 平均池化
            ratio = len(features) // target_dim
            projected = []
            for i in range(target_dim):
                start = i * ratio
                end = min(start + ratio, len(features))
                val = sum(features[start:end]) / (end - start) if end > start else 0.0
                projected.append(val)
            return projected
        else:
            # 重复填充
            projected = []
            ratio = target_dim // len(features)
            for f in features:
                projected.extend([f] * ratio)
            while len(projected) < target_dim:
                projected.append(0.0)
            return projected[:target_dim]
    
    def get_config(self) -> Dict[str, Any]:
        """获取配置"""
        return {
            'enabled_modalities': self.enabled_modalities,
            'embed_dims': self.embed_dims,
            'output_dim': self.output_dim,
            'modality_weights': self.modality_weights
        }


class ImageProcessor:
    """图像处理器"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        config = config or {}
        
        self.image_size = config.get('image_size', 224)
        self.patch_size = config.get('patch_size', 16)
        self.embed_dim = config.get('embed_dim', 768)
        
        # 简化的图像处理参数
        scale = 0.02
        self.proj = [[random.gauss(0, scale) for _ in range(self.patch_size ** 2 * 3)] 
                     for _ in range(self.embed_dim)]
    
    def process(self, image: List[List[List[List[float]]]]) -> List[List[float]]:
        """
        处理图像
        
        Args:
            image: 图像数据 [batch, channels, height, width]
        
        Returns:
            图像特征 [batch, embed_dim]
        """
        if not image:
            return []
        
        batch_size = len(image)
        features = []
        
        for b in range(batch_size):
            # 简化处理：全局平均池化 + 投影
            channels = image[b]
            
            # 全局平均池化
            pooled = []
            for c in range(len(channels)):
                for h in range(0, len(channels[c]), self.patch_size):
                    for w in range(0, len(channels[c][h]), self.patch_size):
                        patch_vals = []
                        for ph in range(self.patch_size):
                            for pw in range(self.patch_size):
                                hh, ww = h + ph, w + pw
                                if hh < len(channels[c]) and ww < len(channels[c][hh]):
                                    patch_vals.append(channels[c][hh][ww])
                        if patch_vals:
                            pooled.append(sum(patch_vals) / len(patch_vals))
            
            # 投影
            if pooled:
                feat = [sum(pooled[k] * self.proj[j][k % len(self.proj[j])] 
                       for k in range(min(len(pooled), len(self.proj[j])))) 
                       for j in range(len(self.proj))]
            else:
                feat = [0.0] * self.embed_dim
            
            features.append(feat)
        
        return features


class TextProcessor:
    """文本处理器"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        config = config or {}
        
        self.vocab_size = config.get('vocab_size', 30522)
        self.embed_dim = config.get('embed_dim', 512)
        self.max_length = config.get('max_length', 512)
        
        # 词嵌入
        scale = 0.02
        self.embeddings = [[random.gauss(0, scale) for _ in range(self.embed_dim)] 
                           for _ in range(self.vocab_size)]
        
        # 位置嵌入
        self.pos_embeddings = [[random.gauss(0, scale) for _ in range(self.embed_dim)] 
                               for _ in range(self.max_length)]
    
    def tokenize(self, text: str) -> List[int]:
        """简单的字符级分词"""
        tokens = []
        for char in text[:self.max_length]:
            token_id = ord(char) % self.vocab_size
            tokens.append(token_id)
        return tokens
    
    def process(self, text: Union[str, List[str]]) -> List[List[float]]:
        """
        处理文本
        
        Args:
            text: 文本字符串或字符串列表
        
        Returns:
            文本特征
        """
        if isinstance(text, str):
            texts = [text]
        else:
            texts = text
        
        features = []
        for t in texts:
            tokens = self.tokenize(t)
            
            if not tokens:
                features.append([0.0] * self.embed_dim)
                continue
            
            # 嵌入查找 + 位置编码
            embedded = []
            for i, token in enumerate(tokens):
                emb = self.embeddings[token].copy()
                if i < len(self.pos_embeddings):
                    emb = [emb[j] + self.pos_embeddings[i][j] for j in range(len(emb))]
                embedded.append(emb)
            
            # 平均池化
            pooled = [sum(e[i] for e in embedded) / len(embedded) 
                      for i in range(len(embedded[0]))]
            features.append(pooled)
        
        return features


class AudioProcessor:
    """音频处理器"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        config = config or {}
        
        self.sample_rate = config.get('sample_rate', 16000)
        self.n_mels = config.get('n_mels', 80)
        self.embed_dim = config.get('embed_dim', 512)
        
        scale = 0.02
        self.proj = [[random.gauss(0, scale) for _ in range(self.n_mels)] 
                     for _ in range(self.embed_dim)]
    
    def extract_mel(self, audio: List[float]) -> List[List[float]]:
        """简化的Mel频谱提取"""
        # 简化处理：分段能量
        segment_size = len(audio) // 100 if len(audio) > 100 else 1
        mel_features = []
        
        for m in range(self.n_mels):
            row = []
            for i in range(0, len(audio), segment_size):
                segment = audio[i:i + segment_size]
                energy = sum(x ** 2 for x in segment) / len(segment) if segment else 0.0
                row.append(math.log(energy + 1e-10))
            mel_features.append(row)
        
        return mel_features
    
    def process(self, audio: Union[List[float], List[List[float]]]) -> List[List[float]]:
        """
        处理音频
        
        Args:
            audio: 音频波形数据
        
        Returns:
            音频特征
        """
        if isinstance(audio[0], list):
            audios = audio
        else:
            audios = [audio]
        
        features = []
        for a in audios:
            mel = self.extract_mel(a)
            
            # 平均池化
            pooled = [sum(row) / len(row) if row else 0.0 for row in mel]
            
            # 投影
            feat = [sum(pooled[k] * self.proj[j][k] for k in range(len(pooled))) 
                    for j in range(len(self.proj))]
            features.append(feat)
        
        return features


class VideoProcessor:
    """视频处理器"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        config = config or {}
        
        self.num_frames = config.get('num_frames', 16)
        self.frame_size = config.get('frame_size', 224)
        self.embed_dim = config.get('embed_dim', 768)
        
        self.image_processor = ImageProcessor({
            'image_size': self.frame_size,
            'embed_dim': self.embed_dim
        })
        
        # 时序编码
        scale = 0.02
        self.temporal_proj = [[random.gauss(0, scale) for _ in range(self.embed_dim)] 
                              for _ in range(self.embed_dim)]
    
    def sample_frames(self, frames: List[Any], num_frames: int) -> List[Any]:
        """均匀采样帧"""
        if len(frames) <= num_frames:
            return frames
        
        interval = len(frames) / num_frames
        sampled = []
        for i in range(num_frames):
            idx = int(i * interval)
            sampled.append(frames[idx])
        
        return sampled
    
    def process(self, video: List[Any]) -> List[List[float]]:
        """
        处理视频
        
        Args:
            video: 视频帧列表
        
        Returns:
            视频特征
        """
        if not video:
            return [[0.0] * self.embed_dim]
        
        # 采样帧
        frames = self.sample_frames(video, self.num_frames)
        
        # 处理每帧
        frame_features = []
        for frame in frames:
            feat = self.image_processor.process([frame])
            if feat:
                frame_features.append(feat[0])
        
        if not frame_features:
            return [[0.0] * self.embed_dim]
        
        # 时序聚合
        # 加权平均 (近期帧权重更高)
        weights = [math.exp(i / len(frame_features)) for i in range(len(frame_features))]
        total_weight = sum(weights)
        
        aggregated = [0.0] * self.embed_dim
        for i, feat in enumerate(frame_features):
            w = weights[i] / total_weight
            for j in range(len(aggregated)):
                aggregated[j] += w * feat[j]
        
        return [aggregated]


class MultimodalPipeline:
    """多模态处理流水线"""
    
    def __init__(self, processor: MultimodalProcessor):
        self.processor = processor
        self.preprocessing_steps: List[callable] = []
        self.postprocessing_steps: List[callable] = []
    
    def add_preprocessing(self, step: callable):
        """添加预处理步骤"""
        self.preprocessing_steps.append(step)
    
    def add_postprocessing(self, step: callable):
        """添加后处理步骤"""
        self.postprocessing_steps.append(step)
    
    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """运行流水线"""
        # 预处理
        processed_inputs = inputs.copy()
        for step in self.preprocessing_steps:
            processed_inputs = step(processed_inputs)
        
        # 处理
        features = self.processor.process(processed_inputs)
        
        # 后处理
        result = {'features': features}
        for step in self.postprocessing_steps:
            result = step(result)
        
        return result


def create_multimodal_processor(modalities: List[str] = None,
                                output_dim: int = 512) -> MultimodalProcessor:
    """创建多模态处理器"""
    modalities = modalities or ['image', 'text', 'audio', 'video']
    return MultimodalProcessor({
        'enabled_modalities': modalities,
        'output_dim': output_dim
    })
