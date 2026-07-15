"""
音频处理模块
支持TTS语音合成、背景音乐添加、音频处理
"""

import os
import tempfile
import wave
import struct
import math
from typing import Optional, List, Tuple
from pathlib import Path


class AudioProcessor:
    """音频处理器"""
    
    def __init__(self, sample_rate: int = 24000):
        self.sample_rate = sample_rate
    
    def generate_silence(self, duration: float, output_path: str):
        """生成静音音频"""
        num_samples = int(self.sample_rate * duration)
        
        with wave.open(output_path, 'wb') as wf:
            wf.setnchannels(1)  # 单声道
            wf.setsampwidth(2)  # 16位
            wf.setframerate(self.sample_rate)
            wf.writeframes(b'\x00' * (num_samples * 2))
        
        return output_path
    
    def mix_audio(self, audio_paths: List[str], output_path: str, weights: Optional[List[float]] = None):
        """
        混合多个音频文件
        
        Args:
            audio_paths: 音频文件路径列表
            output_path: 输出路径
            weights: 混合权重
        """
        if not audio_paths:
            return None
        
        if weights is None:
            weights = [1.0 / len(audio_paths)] * len(audio_paths)
        
        # 读取所有音频
        audio_data = []
        max_length = 0
        
        for path in audio_paths:
            if not os.path.exists(path):
                continue
            
            with wave.open(path, 'rb') as wf:
                frames = wf.readframes(wf.getnframes())
                samples = struct.unpack(f'{len(frames)//2}h', frames)
                audio_data.append(samples)
                max_length = max(max_length, len(samples))
        
        if not audio_data:
            return None
        
        # 混合音频
        mixed = [0] * max_length
        for samples, weight in zip(audio_data, weights):
            for i, sample in enumerate(samples):
                mixed[i] += int(sample * weight)
        
        # 裁剪到有效范围
        mixed = [max(-32768, min(32767, s)) for s in mixed]
        
        # 保存
        with wave.open(output_path, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self.sample_rate)
            wf.writeframes(struct.pack(f'{len(mixed)}h', *mixed))
        
        return output_path
    
    def adjust_volume(self, input_path: str, output_path: str, volume_db: float):
        """
        调整音量
        
        Args:
            input_path: 输入音频路径
            output_path: 输出音频路径
            volume_db: 音量调整（分贝）
        """
        if not os.path.exists(input_path):
            return None
        
        # 计算增益
        gain = 10 ** (volume_db / 20)
        
        with wave.open(input_path, 'rb') as wf_in:
            with wave.open(output_path, 'wb') as wf_out:
                wf_out.setnchannels(wf_in.getnchannels())
                wf_out.setsampwidth(wf_in.getsampwidth())
                wf_out.setframerate(wf_in.getframerate())
                
                # 读取并调整
                frames = wf_in.readframes(wf_in.getnframes())
                samples = struct.unpack(f'{len(frames)//2}h', frames)
                adjusted = [max(-32768, min(32767, int(s * gain))) for s in samples]
                wf_out.writeframes(struct.pack(f'{len(adjusted)}h', *adjusted))
        
        return output_path
    
    def fade_in_out(self, input_path: str, output_path: str, fade_in: float = 0.5, fade_out: float = 0.5):
        """
        添加淡入淡出效果
        
        Args:
            input_path: 输入音频路径
            output_path: 输出音频路径
            fade_in: 淡入时长（秒）
            fade_out: 淡出时长（秒）
        """
        if not os.path.exists(input_path):
            return None
        
        with wave.open(input_path, 'rb') as wf_in:
            nchannels = wf_in.getnchannels()
            sampwidth = wf_in.getsampwidth()
            framerate = wf_in.getframerate()
            nframes = wf_in.getnframes()
            
            fade_in_samples = int(fade_in * framerate)
            fade_out_samples = int(fade_out * framerate)
            
            with wave.open(output_path, 'wb') as wf_out:
                wf_out.setnchannels(nchannels)
                wf_out.setsampwidth(sampwidth)
                wf_out.setframerate(framerate)
                
                frames = wf_in.readframes(nframes)
                samples = struct.unpack(f'{len(frames)//2}h', frames)
                
                # 应用淡入淡出
                faded = []
                for i, sample in enumerate(samples):
                    if i < fade_in_samples:
                        factor = i / fade_in_samples
                    elif i >= nframes - fade_out_samples:
                        factor = (nframes - i) / fade_out_samples
                    else:
                        factor = 1.0
                    
                    faded.append(int(sample * factor))
                
                wf_out.writeframes(struct.pack(f'{len(faded)}h', *faded))
        
        return output_path


def generate_tts(text: str, output_path: str, voice: str = "zh-CN-XiaoxiaoNeural") -> str:
    """
    文本转语音
    
    Args:
        text: 文本内容
        output_path: 输出路径
        voice: 语音类型
    
    Returns:
        输出文件路径
    """
    try:
        # 尝试使用edge-tts
        import subprocess
        subprocess.run([
            'edge-tts', '--text', text, '--voice', voice,
            '--write-media', output_path
        ], check=True, capture_output=True)
        return output_path
    except:
        # 失败时生成静音
        print(f"[Audio] TTS失败，生成静音文件")
        processor = AudioProcessor()
        processor.generate_silence(2.0, output_path)
        return output_path


def add_background_music(video_path: str, music_path: str, output_path: str, 
                         music_volume: float = -10.0) -> str:
    """
    添加背景音乐到视频
    
    Args:
        video_path: 视频文件路径
        music_path: 音乐文件路径
        output_path: 输出路径
        music_volume: 音乐音量（分贝）
    
    Returns:
        输出文件路径
    """
    try:
        # 尝试使用ffmpeg
        import subprocess
        
        # 调整音乐音量并混合
        cmd = [
            'ffmpeg', '-y',
            '-i', video_path,
            '-i', music_path,
            '-filter_complex', f'[1:a]volume={music_volume}dB[music];[0:a][music]amix=inputs=2:duration=first',
            '-c:v', 'copy',
            output_path
        ]
        
        subprocess.run(cmd, check=True, capture_output=True)
        return output_path
        
    except Exception as e:
        print(f"[Audio] 添加背景音乐失败: {e}")
        # 失败时复制原视频
        import shutil
        shutil.copy(video_path, output_path)
        return output_path


def extract_audio(video_path: str, output_path: str) -> str:
    """
    从视频中提取音频
    
    Args:
        video_path: 视频文件路径
        output_path: 输出音频路径
    
    Returns:
        输出文件路径
    """
    try:
        import subprocess
        cmd = [
            'ffmpeg', '-y',
            '-i', video_path,
            '-vn',  # 不处理视频
            '-acodec', 'pcm_s16le',  # PCM 16位小端
            '-ar', '24000',  # 采样率
            '-ac', '1',  # 单声道
            output_path
        ]
        
        subprocess.run(cmd, check=True, capture_output=True)
        return output_path
        
    except Exception as e:
        print(f"[Audio] 提取音频失败: {e}")
        # 生成静音
        processor = AudioProcessor()
        processor.generate_silence(1.0, output_path)
        return output_path


def merge_audio_video(video_path: str, audio_path: str, output_path: str) -> str:
    """
    合并音频和视频
    
    Args:
        video_path: 视频文件路径
        audio_path: 音频文件路径
        output_path: 输出路径
    
    Returns:
        输出文件路径
    """
    try:
        import subprocess
        cmd = [
            'ffmpeg', '-y',
            '-i', video_path,
            '-i', audio_path,
            '-c:v', 'copy',
            '-c:a', 'aac',
            '-shortest',  # 以最短的为准
            output_path
        ]
        
        subprocess.run(cmd, check=True, capture_output=True)
        return output_path
        
    except Exception as e:
        print(f"[Audio] 合并音视频失败: {e}")
        import shutil
        shutil.copy(video_path, output_path)
        return output_path


class AudioEncoder:
    """音频编码器（用于多模态输入）"""
    
    def __init__(self, model_path: str = "./models/wav2vec2-base", device: str = "cpu"):
        self.model_path = model_path
        self.device = device
        self.sample_rate = 16000
    
    def load_audio(self, audio_path: str) -> Tuple[List[float], int]:
        """
        加载音频文件
        
        Args:
            audio_path: 音频文件路径
        
        Returns:
            (音频数据, 采样率)
        """
        if not os.path.exists(audio_path):
            # 返回静音
            return [0.0] * self.sample_rate, self.sample_rate
        
        try:
            import wave
            with wave.open(audio_path, 'rb') as wf:
                nchannels = wf.getnchannels()
                sampwidth = wf.getsampwidth()
                framerate = wf.getframerate()
                nframes = wf.getnframes()
                
                frames = wf.readframes(nframes)
                
                # 转换为浮点数
                if sampwidth == 2:
                    import struct
                    samples = struct.unpack(f'{nframes * nchannels}h', frames)
                    # 转换为单声道
                    if nchannels == 2:
                        samples = [(samples[i] + samples[i+1]) / 2 for i in range(0, len(samples), 2)]
                    # 归一化到[-1, 1]
                    samples = [s / 32768.0 for s in samples]
                else:
                    samples = [0.0] * nframes
                
                return samples, framerate
                
        except Exception as e:
            print(f"[AudioEncoder] 加载音频失败: {e}")
            return [0.0] * self.sample_rate, self.sample_rate
    
    def __call__(self, audio_data: Tuple[List[float], int]) -> 'AudioEncoderOutput':
        """
        编码音频
        
        Args:
            audio_data: (音频数据, 采样率)
        
        Returns:
            编码后的特征
        """
        samples, sr = audio_data
        
        # 简化实现：返回统计特征
        # 实际应该使用Wav2Vec等模型
        
        import random
        # 模拟特征 (768维，与CLIP对齐)
        features = [random.gauss(0, 0.1) for _ in range(768)]
        
        return AudioEncoderOutput(features)


class AudioEncoderOutput:
    """音频编码输出"""
    
    def __init__(self, features: List[float]):
        self.features = features
        self.last_hidden_state = features
    
    def mean(self, dim: int = 0, keepdim: bool = False):
        """计算均值（模拟）"""
        import random
        return [[random.gauss(0, 0.1) for _ in range(768)]]


if __name__ == "__main__":
    print("音频处理模块已加载")
    
    # 测试
    processor = AudioProcessor()
    
    # 生成测试音频
    test_path = "/tmp/test_audio.wav"
    processor.generate_silence(2.0, test_path)
    print(f"生成测试音频: {test_path}")
    
    # 测试音量调整
    adjusted_path = "/tmp/test_adjusted.wav"
    processor.adjust_volume(test_path, adjusted_path, -6.0)
    print(f"调整音量: {adjusted_path}")
    
    # 测试淡入淡出
    faded_path = "/tmp/test_faded.wav"
    processor.fade_in_out(test_path, faded_path, 0.5, 0.5)
    print(f"淡入淡出: {faded_path}")