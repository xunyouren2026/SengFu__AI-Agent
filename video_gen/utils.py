"""
工具函数模块
提供视频处理、随机种子设置等通用功能
"""

import os
import random
from typing import Optional, Tuple, List


def set_seed(seed: int = 42):
    """
    设置随机种子，保证可复现性
    
    Args:
        seed: 随机种子
    """
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
    
    print(f"[Utils] 随机种子已设置为: {seed}")


def get_resolution(resolution: str) -> Tuple[int, int]:
    """
    获取分辨率对应的宽高
    
    Args:
        resolution: 分辨率字符串 (如 "1080p", "4k")
    
    Returns:
        (width, height) 元组
    """
    resolution_map = {
        "256p": (256, 256),
        "360p": (640, 360),
        "480p": (854, 480),
        "720p": (1280, 720),
        "1080p": (1920, 1080),
        "4k": (3840, 2160),
        "8k": (7680, 4320),
    }
    
    if resolution in resolution_map:
        return resolution_map[resolution]
    else:
        # 尝试解析自定义分辨率
        if "x" in resolution:
            parts = resolution.split("x")
            return int(parts[0]), int(parts[1])
        return (256, 256)


def save_video(frames: List, output_path: str, fps: int = 8, format: str = "mp4") -> str:
    """
    保存视频帧为视频文件
    
    Args:
        frames: 视频帧列表
        output_path: 输出路径
        fps: 帧率
        format: 视频格式
    
    Returns:
        输出文件路径
    """
    try:
        import cv2
        import numpy as np
        
        if not frames:
            print("[Utils] 警告: 空帧列表")
            return output_path
        
        # 获取帧尺寸
        h, w = frames[0].shape[:2]
        
        # 创建视频写入器
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
        
        for frame in frames:
            # 确保是uint8格式
            if frame.dtype != 'uint8':
                frame = (frame * 255).astype('uint8')
            
            # 转换为BGR格式
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            out.write(frame)
        
        out.release()
        print(f"[Utils] 视频已保存: {output_path}")
        return output_path
        
    except ImportError:
        print("[Utils] 警告: cv2未安装，无法保存视频")
        return output_path
    except Exception as e:
        print(f"[Utils] 保存视频失败: {e}")
        return output_path


def load_video(video_path: str, max_frames: Optional[int] = None) -> List:
    """
    加载视频文件
    
    Args:
        video_path: 视频路径
        max_frames: 最大帧数
    
    Returns:
        帧列表
    """
    try:
        import cv2
        import numpy as np
        
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # 转换为RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
            
            if max_frames and len(frames) >= max_frames:
                break
        
        cap.release()
        return frames
        
    except ImportError:
        print("[Utils] 警告: cv2未安装，无法加载视频")
        return []
    except Exception as e:
        print(f"[Utils] 加载视频失败: {e}")
        return []


def resize_video(frames: List, target_size: Tuple[int, int]) -> List:
    """
    调整视频帧大小
    
    Args:
        frames: 视频帧列表
        target_size: 目标尺寸 (width, height)
    
    Returns:
        调整后的帧列表
    """
    try:
        import cv2
        
        resized = []
        for frame in frames:
            resized_frame = cv2.resize(frame, target_size, interpolation=cv2.INTER_LANCZOS4)
            resized.append(resized_frame)
        
        return resized
        
    except ImportError:
        print("[Utils] 警告: cv2未安装，无法调整视频大小")
        return frames
    except Exception as e:
        print(f"[Utils] 调整视频大小失败: {e}")
        return frames


def extract_frames(video_path: str, output_dir: str, fps: Optional[int] = None) -> List[str]:
    """
    从视频中提取帧
    
    Args:
        video_path: 视频路径
        output_dir: 输出目录
        fps: 提取帧率 (None表示提取所有帧)
    
    Returns:
        提取的帧文件路径列表
    """
    try:
        import cv2
        import os
        
        os.makedirs(output_dir, exist_ok=True)
        
        cap = cv2.VideoCapture(video_path)
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        
        frame_interval = 1
        if fps is not None:
            frame_interval = int(original_fps / fps)
        
        frame_paths = []
        frame_count = 0
        saved_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % frame_interval == 0:
                frame_path = os.path.join(output_dir, f"frame_{saved_count:06d}.png")
                cv2.imwrite(frame_path, frame)
                frame_paths.append(frame_path)
                saved_count += 1
            
            frame_count += 1
        
        cap.release()
        print(f"[Utils] 已提取 {saved_count} 帧到 {output_dir}")
        return frame_paths
        
    except ImportError:
        print("[Utils] 警告: cv2未安装，无法提取帧")
        return []
    except Exception as e:
        print(f"[Utils] 提取帧失败: {e}")
        return []


class VideoProcessor:
    """视频处理器"""
    
    def __init__(self):
        self.temp_dir = "/tmp/video_gen"
        os.makedirs(self.temp_dir, exist_ok=True)
    
    def normalize_frames(self, frames: List) -> List:
        """归一化帧到[0, 1]范围"""
        normalized = []
        for frame in frames:
            if frame.max() > 1.0:
                frame = frame / 255.0
            normalized.append(frame)
        return normalized
    
    def denormalize_frames(self, frames: List) -> List:
        """反归一化帧到[0, 255]范围"""
        denormalized = []
        for frame in frames:
            if frame.max() <= 1.0:
                frame = (frame * 255).astype('uint8')
            denormalized.append(frame)
        return denormalized
    
    def apply_watermark(self, frames: List, watermark_text: str, position: str = "bottom-right") -> List:
        """
        添加水印
        
        Args:
            frames: 视频帧
            watermark_text: 水印文本
            position: 位置 (top-left, top-right, bottom-left, bottom-right)
        
        Returns:
            添加水印后的帧
        """
        try:
            import cv2
            
            watermarked = []
            for frame in frames:
                frame_copy = frame.copy()
                h, w = frame_copy.shape[:2]
                
                # 计算位置
                if position == "bottom-right":
                    org = (w - 200, h - 20)
                elif position == "bottom-left":
                    org = (10, h - 20)
                elif position == "top-right":
                    org = (w - 200, 30)
                else:  # top-left
                    org = (10, 30)
                
                cv2.putText(frame_copy, watermark_text, org, 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                watermarked.append(frame_copy)
            
            return watermarked
            
        except ImportError:
            return frames
    
    def concatenate_videos(self, video_paths: List[str], output_path: str, transition_frames: int = 5) -> str:
        """
        拼接多个视频
        
        Args:
            video_paths: 视频路径列表
            output_path: 输出路径
            transition_frames: 过渡帧数
        
        Returns:
            输出路径
        """
        all_frames = []
        
        for i, path in enumerate(video_paths):
            frames = load_video(path)
            
            if i > 0 and transition_frames > 0:
                # 添加过渡效果（简单淡入淡出）
                prev_frames = all_frames[-transition_frames:]
                curr_frames = frames[:transition_frames]
                
                # 混合过渡帧
                for j in range(transition_frames):
                    alpha = j / transition_frames
                    blended = (1 - alpha) * prev_frames[j] + alpha * curr_frames[j]
                    all_frames[-(transition_frames - j)] = blended.astype('uint8')
                
                all_frames.extend(frames[transition_frames:])
            else:
                all_frames.extend(frames)
        
        return save_video(all_frames, output_path)


# 便捷函数
def create_video_from_images(image_paths: List[str], output_path: str, fps: int = 8) -> str:
    """
    从图片序列创建视频
    
    Args:
        image_paths: 图片路径列表
        output_path: 输出路径
        fps: 帧率
    
    Returns:
        输出路径
    """
    try:
        import cv2
        
        if not image_paths:
            return output_path
        
        # 读取第一帧获取尺寸
        first_frame = cv2.imread(image_paths[0])
        h, w = first_frame.shape[:2]
        
        # 创建视频写入器
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
        
        for path in image_paths:
            frame = cv2.imread(path)
            out.write(frame)
        
        out.release()
        print(f"[Utils] 视频已创建: {output_path}")
        return output_path
        
    except Exception as e:
        print(f"[Utils] 创建视频失败: {e}")
        return output_path


def get_video_info(video_path: str) -> dict:
    """
    获取视频信息
    
    Args:
        video_path: 视频路径
    
    Returns:
        视频信息字典
    """
    try:
        import cv2
        
        cap = cv2.VideoCapture(video_path)
        
        info = {
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps": cap.get(cv2.CAP_PROP_FPS),
            "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            "duration": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / cap.get(cv2.CAP_PROP_FPS)
        }
        
        cap.release()
        return info
        
    except Exception as e:
        print(f"[Utils] 获取视频信息失败: {e}")
        return {}


if __name__ == "__main__":
    print("工具函数模块已加载")
    
    # 测试
    set_seed(42)
    print(f"分辨率 1080p: {get_resolution('1080p')}")
    print(f"分辨率 4k: {get_resolution('4k')}")
