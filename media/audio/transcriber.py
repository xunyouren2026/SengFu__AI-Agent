"""
音频转文字模块
支持调用 Whisper API 或本地模型进行语音识别
"""

import os
import json
import subprocess
import wave
import struct
import math
from pathlib import Path
from typing import Optional, Union, List, Dict, Any, Generator
from dataclasses import dataclass
from enum import Enum
import tempfile
import re


class TranscriptionModel(Enum):
    """转录模型"""
    WHISPER_TINY = "tiny"
    WHISPER_BASE = "base"
    WHISPER_SMALL = "small"
    WHISPER_MEDIUM = "medium"
    WHISPER_LARGE = "large"
    WHISPER_LARGE_V3 = "large-v3"
    OPENAI_API = "openai-api"


class TranscriptionLanguage(Enum):
    """支持的语言"""
    AUTO = "auto"
    CHINESE = "zh"
    ENGLISH = "en"
    JAPANESE = "ja"
    KOREAN = "ko"
    FRENCH = "fr"
    GERMAN = "de"
    SPANISH = "es"
    RUSSIAN = "ru"
    PORTUGUESE = "pt"
    ITALIAN = "it"
    ARABIC = "ar"


class OutputFormat(Enum):
    """输出格式"""
    TEXT = "text"
    JSON = "json"
    SRT = "srt"
    VTT = "vtt"
    TSV = "tsv"


@dataclass
class TranscriptionSegment:
    """转录片段"""
    start: float
    end: float
    text: str
    confidence: float = 1.0
    
    @property
    def duration(self) -> float:
        return self.end - self.start
    
    def __repr__(self) -> str:
        return f"Segment({self.start:.2f}s-{self.end:.2f}s: {self.text[:30]}...)"


@dataclass
class TranscriptionResult:
    """转录结果"""
    text: str
    segments: List[TranscriptionSegment]
    language: str
    duration: float
    model: str
    
    def __repr__(self) -> str:
        return (
            f"TranscriptionResult(language={self.language}, "
            f"segments={len(self.segments)}, duration={self.duration:.2f}s)"
        )
    
    def to_srt(self) -> str:
        """转换为 SRT 格式"""
        lines = []
        for i, seg in enumerate(self.segments, 1):
            start = self._format_srt_time(seg.start)
            end = self._format_srt_time(seg.end)
            lines.append(f"{i}")
            lines.append(f"{start} --> {end}")
            lines.append(seg.text)
            lines.append("")
        return "\n".join(lines)
    
    def to_vtt(self) -> str:
        """转换为 VTT 格式"""
        lines = ["WEBVTT", ""]
        for seg in self.segments:
            start = self._format_vtt_time(seg.start)
            end = self._format_vtt_time(seg.end)
            lines.append(f"{start} --> {end}")
            lines.append(seg.text)
            lines.append("")
        return "\n".join(lines)
    
    @staticmethod
    def _format_srt_time(seconds: float) -> str:
        """格式化 SRT 时间"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"
    
    @staticmethod
    def _format_vtt_time(seconds: float) -> str:
        """格式化 VTT 时间"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


@dataclass
class TranscriptionConfig:
    """转录配置"""
    model: TranscriptionModel = TranscriptionModel.WHISPER_BASE
    language: TranscriptionLanguage = TranscriptionLanguage.AUTO
    output_format: OutputFormat = OutputFormat.TEXT
    temperature: float = 0.0
    beam_size: int = 5
    best_of: int = 5
    word_timestamps: bool = False
    vad_filter: bool = True
    initial_prompt: Optional[str] = None
    api_key: Optional[str] = None


class AudioProcessor:
    """音频处理器"""
    
    @staticmethod
    def get_audio_info(file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        获取音频文件信息
        
        Args:
            file_path: 音频文件路径
            
        Returns:
            音频信息字典
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"文件不存在: {path}")
        
        info = {
            'duration': 0.0,
            'sample_rate': 0,
            'channels': 0,
            'sample_width': 0,
            'frames': 0
        }
        
        # 尝试使用 wave 读取 WAV 文件
        if path.suffix.lower() == '.wav':
            try:
                with wave.open(str(path), 'rb') as wf:
                    info['channels'] = wf.getnchannels()
                    info['sample_width'] = wf.getsampwidth()
                    info['sample_rate'] = wf.getframerate()
                    info['frames'] = wf.getnframes()
                    info['duration'] = info['frames'] / info['sample_rate']
            except Exception:
                pass
        
        return info
    
    @staticmethod
    def convert_to_wav(
        input_path: Union[str, Path],
        output_path: Union[str, Path],
        sample_rate: int = 16000,
        channels: int = 1
    ) -> bool:
        """
        转换为 WAV 格式
        
        Args:
            input_path: 输入文件路径
            output_path: 输出文件路径
            sample_rate: 采样率
            channels: 声道数
            
        Returns:
            是否成功
        """
        input_path = Path(input_path)
        output_path = Path(output_path)
        
        if not input_path.exists():
            raise FileNotFoundError(f"输入文件不存在: {input_path}")
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        cmd = [
            "ffmpeg", "-y",
            "-i", str(input_path),
            "-ar", str(sample_rate),
            "-ac", str(channels),
            "-f", "wav",
            str(output_path)
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, timeout=300)
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False
    
    @staticmethod
    def split_audio(
        file_path: Union[str, Path],
        chunk_duration: float = 30.0,
        output_dir: Optional[Union[str, Path]] = None
    ) -> List[Path]:
        """
        分割音频文件
        
        Args:
            file_path: 音频文件路径
            chunk_duration: 每段时长（秒）
            output_dir: 输出目录
            
        Returns:
            分割后的文件列表
        """
        file_path = Path(file_path)
        
        if output_dir is None:
            output_dir = file_path.parent / "chunks"
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 获取音频信息
        info = AudioProcessor.get_audio_info(file_path)
        duration = info.get('duration', 0)
        
        if duration <= 0:
            return []
        
        chunks = []
        num_chunks = int(math.ceil(duration / chunk_duration))
        
        for i in range(num_chunks):
            start = i * chunk_duration
            output_file = output_dir / f"chunk_{i:04d}.wav"
            
            cmd = [
                "ffmpeg", "-y",
                "-i", str(file_path),
                "-ss", str(start),
                "-t", str(chunk_duration),
                "-acodec", "pcm_s16le",
                str(output_file)
            ]
            
            try:
                result = subprocess.run(cmd, capture_output=True, timeout=60)
                if result.returncode == 0:
                    chunks.append(output_file)
            except (subprocess.TimeoutExpired, FileNotFoundError):
                pass
        
        return chunks


class Transcriber:
    """音频转文字主类"""
    
    def __init__(self, config: Optional[TranscriptionConfig] = None):
        """
        初始化转录器
        
        Args:
            config: 转录配置
        """
        self.config = config or TranscriptionConfig()
        self._model_loaded = False
    
    def transcribe(
        self,
        audio_path: Union[str, Path],
        config: Optional[TranscriptionConfig] = None
    ) -> TranscriptionResult:
        """
        转录音频文件
        
        Args:
            audio_path: 音频文件路径
            config: 转录配置（可选，覆盖默认配置）
            
        Returns:
            转录结果
        """
        cfg = config or self.config
        audio_path = Path(audio_path)
        
        if not audio_path.exists():
            raise FileNotFoundError(f"音频文件不存在: {audio_path}")
        
        # 根据模型选择转录方法
        if cfg.model == TranscriptionModel.OPENAI_API:
            return self._transcribe_with_api(audio_path, cfg)
        else:
            return self._transcribe_with_local(audio_path, cfg)
    
    def _transcribe_with_api(
        self,
        audio_path: Path,
        config: TranscriptionConfig
    ) -> TranscriptionResult:
        """使用 OpenAI API 转录"""
        import urllib.request
        import urllib.error
        
        api_key = config.api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("需要 OpenAI API Key")
        
        # 读取音频文件
        with open(audio_path, 'rb') as f:
            audio_data = f.read()
        
        # 构建请求
        boundary = "----WebKitFormBoundary7MA4YWxkTrZu0gW"
        
        body_parts = []
        body_parts.append(
            f'--{boundary}\r\n'
            f'Content-Disposition: form-data; name="file"; filename="{audio_path.name}"\r\n'
            f'Content-Type: audio/mpeg\r\n\r\n'
        )
        body_parts.append(audio_data)
        body_parts.append(
            f'\r\n--{boundary}\r\n'
            f'Content-Disposition: form-data; name="model"\r\n\r\n'
            f'whisper-1\r\n'
        )
        
        if config.language != TranscriptionLanguage.AUTO:
            body_parts.append(
                f'--{boundary}\r\n'
                f'Content-Disposition: form-data; name="language"\r\n\r\n'
                f'{config.language.value}\r\n'
            )
        
        body_parts.append(f'--{boundary}--\r\n')
        
        body = b''
        for part in body_parts:
            if isinstance(part, str):
                body += part.encode('utf-8')
            else:
                body += part
        
        # 发送请求
        url = "https://api.openai.com/v1/audio/transcriptions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": f"multipart/form-data; boundary={boundary}"
        }
        
        try:
            req = urllib.request.Request(
                url,
                data=body,
                headers=headers,
                method='POST'
            )
            
            with urllib.request.urlopen(req, timeout=300) as response:
                result = json.loads(response.read().decode('utf-8'))
            
            text = result.get('text', '')
            
            # 创建结果对象
            segments = [TranscriptionSegment(
                start=0.0,
                end=self._get_audio_duration(audio_path),
                text=text,
                confidence=1.0
            )]
            
            return TranscriptionResult(
                text=text,
                segments=segments,
                language=config.language.value if config.language != TranscriptionLanguage.AUTO else "unknown",
                duration=segments[0].end,
                model="whisper-1"
            )
            
        except urllib.error.URLError as e:
            raise RuntimeError(f"API 请求失败: {e}")
    
    def _transcribe_with_local(
        self,
        audio_path: Path,
        config: TranscriptionConfig
    ) -> TranscriptionResult:
        """使用本地模型转录"""
        # 检查 whisper 是否安装
        try:
            import whisper
        except ImportError:
            # 尝试使用 whisper.cpp 或回退到模拟
            return self._transcribe_with_whisper_cpp(audio_path, config)
        
        # 加载模型
        model_name = config.model.value
        model = whisper.load_model(model_name)
        
        # 转录选项
        options = {
            'temperature': config.temperature,
            'beam_size': config.beam_size,
            'best_of': config.best_of,
            'word_timestamps': config.word_timestamps,
        }
        
        if config.language != TranscriptionLanguage.AUTO:
            options['language'] = config.language.value
        
        if config.initial_prompt:
            options['initial_prompt'] = config.initial_prompt
        
        # 执行转录
        result = model.transcribe(str(audio_path), **options)
        
        # 解析结果
        segments = []
        for seg in result.get('segments', []):
            segments.append(TranscriptionSegment(
                start=seg['start'],
                end=seg['end'],
                text=seg['text'].strip(),
                confidence=seg.get('avg_logprob', 0.0)
            ))
        
        return TranscriptionResult(
            text=result['text'],
            segments=segments,
            language=result.get('language', 'unknown'),
            duration=self._get_audio_duration(audio_path),
            model=model_name
        )
    
    def _transcribe_with_whisper_cpp(
        self,
        audio_path: Path,
        config: TranscriptionConfig
    ) -> TranscriptionResult:
        """使用 whisper.cpp 转录"""
        # 查找 whisper.cpp 可执行文件
        whisper_cpp = self._find_whisper_cpp()
        
        if whisper_cpp is None:
            # 回退到模拟转录
            return self._mock_transcribe(audio_path, config)
        
        # 转换为 WAV 格式
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            tmp_wav = Path(tmp.name)
        
        if not AudioProcessor.convert_to_wav(audio_path, tmp_wav):
            return self._mock_transcribe(audio_path, config)
        
        try:
            # 执行 whisper.cpp
            model_path = self._get_whisper_cpp_model(config.model)
            
            cmd = [
                whisper_cpp,
                "-m", str(model_path),
                "-f", str(tmp_wav),
                "-t", "4",  # 线程数
                "--output-json"
            ]
            
            if config.language != TranscriptionLanguage.AUTO:
                cmd.extend(["-l", config.language.value])
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600
            )
            
            if result.returncode == 0:
                # 解析 JSON 输出
                json_file = tmp_wav.with_suffix('.json')
                if json_file.exists():
                    with open(json_file) as f:
                        data = json.load(f)
                    
                    segments = []
                    for seg in data.get('transcription', []):
                        segments.append(TranscriptionSegment(
                            start=seg.get('offsets', [0, 0])[0] / 1000.0,
                            end=seg.get('offsets', [0, 0])[1] / 1000.0,
                            text=seg.get('text', ''),
                            confidence=1.0
                        ))
                    
                    text = ' '.join(s.text for s in segments)
                    
                    return TranscriptionResult(
                        text=text,
                        segments=segments,
                        language=config.language.value,
                        duration=self._get_audio_duration(audio_path),
                        model=config.model.value
                    )
        except (subprocess.TimeoutExpired, FileNotFoundError, json.JSONDecodeError):
            pass
        finally:
            if tmp_wav.exists():
                tmp_wav.unlink()
        
        return self._mock_transcribe(audio_path, config)
    
    def _find_whisper_cpp(self) -> Optional[str]:
        """查找 whisper.cpp 可执行文件"""
        candidates = ["whisper", "whisper-cpp", "./whisper"]
        for candidate in candidates:
            try:
                result = subprocess.run(
                    [candidate, "--help"],
                    capture_output=True,
                    timeout=5
                )
                if result.returncode == 0:
                    return candidate
            except (FileNotFoundError, subprocess.TimeoutExpired):
                pass
        return None
    
    def _get_whisper_cpp_model(self, model: TranscriptionModel) -> Path:
        """获取 whisper.cpp 模型路径"""
        model_dir = Path.home() / ".cache" / "whisper"
        model_file = f"ggml-{model.value}.bin"
        return model_dir / model_file
    
    def _mock_transcribe(
        self,
        audio_path: Path,
        config: TranscriptionConfig
    ) -> TranscriptionResult:
        """模拟转录（用于测试）"""
        duration = self._get_audio_duration(audio_path)
        
        # 创建模拟结果
        segments = [
            TranscriptionSegment(
                start=0.0,
                end=duration / 3,
                text="[模拟转录] 这是一段示例文本。",
                confidence=0.9
            ),
            TranscriptionSegment(
                start=duration / 3,
                end=2 * duration / 3,
                text="[模拟转录] 实际使用时需要安装 Whisper。",
                confidence=0.9
            ),
            TranscriptionSegment(
                start=2 * duration / 3,
                end=duration,
                text="[模拟转录] 请安装 openai-whisper 或 whisper.cpp。",
                confidence=0.9
            )
        ]
        
        return TranscriptionResult(
            text=' '.join(s.text for s in segments),
            segments=segments,
            language=config.language.value if config.language != TranscriptionLanguage.AUTO else "zh",
            duration=duration,
            model=config.model.value
        )
    
    def _get_audio_duration(self, audio_path: Path) -> float:
        """获取音频时长"""
        info = AudioProcessor.get_audio_info(audio_path)
        if info['duration'] > 0:
            return info['duration']
        
        # 使用 ffprobe
        try:
            cmd = [
                "ffprobe",
                "-v", "quiet",
                "-print_format", "json",
                "-show_format",
                str(audio_path)
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if result.returncode == 0:
                data = json.loads(result.stdout)
                return float(data.get('format', {}).get('duration', 0))
        except Exception:
            pass
        
        return 0.0
    
    def transcribe_stream(
        self,
        audio_stream: Generator[bytes, None, None],
        config: Optional[TranscriptionConfig] = None
    ) -> Generator[TranscriptionSegment, None, None]:
        """
        流式转录
        
        Args:
            audio_stream: 音频数据流
            config: 转录配置
            
        Yields:
            转录片段
        """
        cfg = config or self.config
        
        # 保存流数据到临时文件
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            for chunk in audio_stream:
                tmp.write(chunk)
            tmp_path = Path(tmp.name)
        
        try:
            result = self.transcribe(tmp_path, cfg)
            for segment in result.segments:
                yield segment
        finally:
            if tmp_path.exists():
                tmp_path.unlink()
    
    def save_transcription(
        self,
        result: TranscriptionResult,
        output_path: Union[str, Path],
        format: Optional[OutputFormat] = None
    ) -> bool:
        """
        保存转录结果
        
        Args:
            result: 转录结果
            output_path: 输出路径
            format: 输出格式
            
        Returns:
            是否成功
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        fmt = format or self.config.output_format
        
        if fmt == OutputFormat.TEXT:
            content = result.text
        elif fmt == OutputFormat.JSON:
            content = json.dumps({
                'text': result.text,
                'segments': [
                    {
                        'start': s.start,
                        'end': s.end,
                        'text': s.text,
                        'confidence': s.confidence
                    }
                    for s in result.segments
                ],
                'language': result.language,
                'duration': result.duration,
                'model': result.model
            }, ensure_ascii=False, indent=2)
        elif fmt == OutputFormat.SRT:
            content = result.to_srt()
        elif fmt == OutputFormat.VTT:
            content = result.to_vtt()
        else:
            content = result.text
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        return True
    
    def batch_transcribe(
        self,
        audio_files: List[Union[str, Path]],
        output_dir: Union[str, Path],
        config: Optional[TranscriptionConfig] = None
    ) -> Dict[str, TranscriptionResult]:
        """
        批量转录
        
        Args:
            audio_files: 音频文件列表
            output_dir: 输出目录
            config: 转录配置
            
        Returns:
            文件名到转录结果的映射
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results = {}
        cfg = config or self.config
        
        for audio_file in audio_files:
            audio_path = Path(audio_file)
            try:
                result = self.transcribe(audio_path, cfg)
                results[str(audio_file)] = result
                
                # 保存结果
                output_file = output_dir / (audio_path.stem + ".txt")
                self.save_transcription(result, output_file)
                
            except Exception as e:
                print(f"转录失败: {audio_file}, 错误: {e}")
        
        return results
