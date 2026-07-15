"""
模型加载器模块
支持从本地路径或HuggingFace Hub加载预训练模型
"""

import os
from typing import Optional


def load_pretrained_model(config, device, pretrained_name: Optional[str] = None, cache_dir: Optional[str] = None):
    """
    加载预训练模型
    
    Args:
        config: 配置对象
        device: 计算设备
        pretrained_name: 预训练模型名称或路径
        cache_dir: 缓存目录
    
    Returns:
        加载好的模型实例
    """
    # 导入模型类
    from .models.dit import SpatialTemporalUNet
    from .models.unet import UNetModel
    from .models.simple_dit import SimpleDiT
    
    # 无预训练权重时随机初始化
    if pretrained_name is None:
        print("[ModelLoader] 未指定预训练模型，使用随机初始化")
        
        # 根据模式选择模型
        if config.model.mode == 'creative':
            model = SimpleDiT(config.model)
        else:
            if config.model.model_type == "dit":
                model = SpatialTemporalUNet(config.model)
            else:
                model = UNetModel(config.model)
        
        model.to(device)
        return model
    
    # 检查是否为本地文件路径
    if os.path.isfile(pretrained_name):
        weight_path = pretrained_name
        print(f"[ModelLoader] 从本地文件加载: {weight_path}")
    else:
        # 从HuggingFace下载
        weight_path = download_from_hub(pretrained_name, cache_dir)
    
    # 构建模型结构
    if config.model.mode == 'creative':
        model = SimpleDiT(config.model)
    else:
        if config.model.model_type == "dit":
            model = SpatialTemporalUNet(config.model)
        else:
            model = UNetModel(config.model)
    
    # 加载权重
    try:
        import torch
        state_dict = torch.load(weight_path, map_location=device)
        model.load_state_dict(state_dict, strict=False)
        print(f"[ModelLoader] 成功加载权重: {weight_path}")
    except Exception as e:
        print(f"[ModelLoader] 权重加载失败: {e}")
        print("[ModelLoader] 使用随机初始化继续")
    
    model.to(device)
    return model


def download_from_hub(repo_id: str, cache_dir: Optional[str] = None, filename: str = "model.safetensors") -> str:
    """
    从HuggingFace Hub下载模型
    
    Args:
        repo_id: 仓库ID
        cache_dir: 缓存目录
        filename: 文件名
    
    Returns:
        下载后的本地路径
    """
    if cache_dir is None:
        cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
    
    try:
        # 尝试使用huggingface_hub
        from huggingface_hub import hf_hub_download
        
        weight_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            cache_dir=cache_dir,
            resume_download=True
        )
        
        print(f"[ModelLoader] 从Hub下载: {repo_id}")
        return weight_path
        
    except ImportError:
        print("[ModelLoader] huggingface_hub未安装，无法下载模型")
        raise
    except Exception as e:
        print(f"[ModelLoader] 下载失败: {e}")
        raise


def save_checkpoint(model, vae, optimizer, epoch: int, save_path: str):
    """
    保存训练检查点
    
    Args:
        model: 模型
        vae: VAE模型
        optimizer: 优化器
        epoch: 当前轮数
        save_path: 保存路径
    """
    import torch
    
    checkpoint = {
        'epoch': epoch,
        'model': model.state_dict(),
        'vae': vae.state_dict() if vae else None,
        'optimizer': optimizer.state_dict() if optimizer else None,
    }
    
    # 创建目录
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # 保存
    torch.save(checkpoint, save_path)
    print(f"[ModelLoader] 检查点已保存: {save_path}")


def load_checkpoint(model, vae, optimizer, checkpoint_path: str):
    """
    加载训练检查点
    
    Args:
        model: 模型
        vae: VAE模型
        optimizer: 优化器
        checkpoint_path: 检查点路径
    
    Returns:
        加载的轮数
    """
    import torch
    
    if not os.path.exists(checkpoint_path):
        print(f"[ModelLoader] 检查点不存在: {checkpoint_path}")
        return 0
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        model.load_state_dict(checkpoint['model'], strict=False)
        
        if vae and checkpoint.get('vae'):
            vae.load_state_dict(checkpoint['vae'], strict=False)
        
        if optimizer and checkpoint.get('optimizer'):
            optimizer.load_state_dict(checkpoint['optimizer'])
        
        epoch = checkpoint.get('epoch', 0)
        print(f"[ModelLoader] 检查点已加载: {checkpoint_path}, epoch={epoch}")
        
        return epoch
        
    except Exception as e:
        print(f"[ModelLoader] 检查点加载失败: {e}")
        return 0


class ModelManager:
    """模型管理器"""
    
    def __init__(self, cache_dir: Optional[str] = None):
        self.cache_dir = cache_dir or os.path.expanduser("~/.cache/video_gen/models")
        self.loaded_models = {}
    
    def register_model(self, name: str, model_path: str):
        """注册模型路径"""
        self.loaded_models[name] = model_path
    
    def get_model_path(self, name: str) -> Optional[str]:
        """获取模型路径"""
        return self.loaded_models.get(name)
    
    def list_models(self):
        """列出所有已注册模型"""
        return list(self.loaded_models.keys())
    
    def clear_cache(self):
        """清理缓存"""
        import shutil
        if os.path.exists(self.cache_dir):
            shutil.rmtree(self.cache_dir)
            print(f"[ModelManager] 缓存已清理: {self.cache_dir}")


if __name__ == "__main__":
    print("模型加载器模块已加载")
    
    # 测试
    manager = ModelManager()
    manager.register_model("test", "/path/to/test")
    print(f"已注册模型: {manager.list_models()}")
