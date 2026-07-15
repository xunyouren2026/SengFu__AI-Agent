"""
智能配置适配器 - 根据GPU显存自动优化配置
"""

import os
from typing import Dict, Any


class AutoConfigAdapter:
    """
    智能配置适配器
    根据当前GPU显存和架构，动态调整config中的关键性能参数，
    在保证不OOM的前提下最大化生成速度。
    """
    
    def __init__(self):
        self.device = self._get_device()
        self.vram_total_gb = 0.0
        self.vram_reserved_gb = 2.5  # 基础预留
        self.gpu_name = "CPU"
        self.arch_name = "Unknown"
        self.compute_capability = (0, 0)
        self.supports_fp8 = False
        
        self._detect_gpu()
    
    def _get_device(self):
        """获取计算设备"""
        try:
            # 尝试检测PyTorch
            import torch
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        except:
            return None
    
    def _detect_gpu(self):
        """检测GPU信息"""
        try:
            import torch
            if torch.cuda.is_available():
                props = torch.cuda.get_device_properties(0)
                self.vram_total_gb = props.total_memory / (1024 ** 3)
                self.gpu_name = props.name
                self.compute_capability = (props.major, props.minor)
                
                # 获取当前已分配显存
                allocated_mem = torch.cuda.memory_allocated(0) / (1024 ** 3)
                reserved_extra = float(os.environ.get('VRAM_RESERVED_EXTRA', 1.0))
                self.vram_reserved_gb = 2.5 + allocated_mem + reserved_extra
                
                # 识别架构
                if props.major == 9:
                    self.arch_name = "Hopper"
                elif props.major == 8 and props.minor >= 9:
                    self.arch_name = "Ada Lovelace"
                elif props.major == 8:
                    self.arch_name = "Ampere"
                elif props.major == 7:
                    self.arch_name = "Volta/Turing"
                else:
                    self.arch_name = "Legacy"
                
                # FP8支持检测 (需要Ada或Hopper)
                self.supports_fp8 = (props.major == 9) or (props.major == 8 and props.minor >= 9)
        except Exception as e:
            print(f"[AutoConfig] GPU检测失败: {e}")
    
    def detect_profile(self) -> str:
        """检测硬件档位"""
        if self.device is None or self.device.type == 'cpu':
            return "cpu_safe"
        
        usable_vram = self.vram_total_gb - self.vram_reserved_gb
        
        if usable_vram >= 20.0:
            return "flagship_ultra"   # 4090, A100, H100
        elif usable_vram >= 13.0:
            return "high_perf"        # 4080, 3090, 4070Ti
        elif usable_vram >= 9.0:
            return "balanced"         # 4070, 3060 12G, 4060Ti 16G
        elif usable_vram >= 6.0:
            return "entry_mid"        # 3060 8G, 2060, 3050
        else:
            return "entry_low"        # <6G
    
    def apply_optimizations(self, config) -> None:
        """应用优化到config对象"""
        profile = self.detect_profile()
        
        print("=" * 60)
        print("🚀 [AutoConfig] 硬件检测报告")
        print(f"   GPU: {self.gpu_name}")
        print(f"   架构: {self.arch_name} (Compute {self.compute_capability})")
        print(f"   总显存: {self.vram_total_gb:.1f} GB")
        print(f"   可用显存: {self.vram_total_gb - self.vram_reserved_gb:.1f} GB")
        print(f"   FP8支持: {self.supports_fp8}")
        print(f"   匹配策略: [{profile}]")
        print("-" * 60)
        
        # 默认值 (安全底线)
        adjustments = {
            'max_block_frames': 48,
            'memory_size': 256,
            'tile_batch_size': 1,
            'vae_type': 'image',
            'use_turboquant': False,
            'teacache_threshold': 0.25,
            'use_internal_memory': False,
            'use_multi_scale_memory': False
        }
        
        if profile == "flagship_ultra":
            adjustments.update({
                'max_block_frames': 192,
                'memory_size': 1024,
                'tile_batch_size': 8,
                'vae_type': 'lean',
                'use_turboquant': self.supports_fp8,
                'teacache_threshold': 0.15,
                'use_internal_memory': True,
                'use_multi_scale_memory': True
            })
            print("✅ 已启用旗舰级优化: LeanVAE + 192帧块 + 1024记忆容量 + FP8")
            
        elif profile == "high_perf":
            adjustments.update({
                'max_block_frames': 128,
                'memory_size': 512,
                'tile_batch_size': 4,
                'vae_type': 'lean',
                'use_turboquant': self.supports_fp8,
                'teacache_threshold': 0.18,
                'use_internal_memory': True,
                'use_multi_scale_memory': True
            })
            print("✅ 已启用高性能优化: LeanVAE + 128帧块 + 512记忆容量")
            
        elif profile == "balanced":
            adjustments.update({
                'max_block_frames': 96,
                'memory_size': 512,
                'tile_batch_size': 2,
                'vae_type': 'lean',
                'use_turboquant': False,
                'teacache_threshold': 0.20,
                'use_internal_memory': True,
                'use_multi_scale_memory': False
            })
            print("✅ 已启用均衡优化: LeanVAE + 96帧块 + 适度记忆")
            
        elif profile in ["entry_mid", "entry_low"]:
            adjustments.update({
                'max_block_frames': 48 if profile == "entry_low" else 64,
                'memory_size': 128 if profile == "entry_low" else 256,
                'tile_batch_size': 1,
                'vae_type': 'image',
                'use_turboquant': False,
                'teacache_threshold': 0.30,
                'use_internal_memory': False,
                'use_multi_scale_memory': False
            })
            print("⚠️ 显存紧张，已切换至安全模式: 普通VAE + 小块生成 + 关闭高级记忆")
        
        # 检查LeanVAE权重是否存在
        if adjustments['vae_type'] == 'lean':
            lean_path = getattr(config.model, 'lean_vae_path', './models/LeanVAE')
            if not os.path.exists(lean_path):
                print(f"[AutoConfig] LeanVAE未找到: {lean_path}，回退到image VAE")
                adjustments['vae_type'] = 'image'
        
        # 应用配置
        applied_count = 0
        for key, value in adjustments.items():
            if hasattr(config.model, key):
                old_val = getattr(config.model, key)
                if old_val != value:
                    setattr(config.model, key, value)
                    print(f"   调整 {key}: {old_val} -> {value}")
                    applied_count += 1
        
        # 显存很小时强制关闭重型功能
        if self.vram_total_gb < 12.0:
            config.model.use_hierarchical_memory = False
            config.model.use_learned_compressor = False
            config.model.use_token_routing = False
            print("   强制关闭分级记忆和复杂路由以节省显存")
        
        print(f"🎉 [AutoConfig] 成功应用 {applied_count} 项动态优化配置!")
        print("=" * 60)
    
    def get_recommended_batch_size(self) -> int:
        """获取推荐的批大小"""
        profile = self.detect_profile()
        batch_sizes = {
            "flagship_ultra": 4,
            "high_perf": 2,
            "balanced": 1,
            "entry_mid": 1,
            "entry_low": 1,
            "cpu_safe": 1
        }
        return batch_sizes.get(profile, 1)
    
    def get_recommended_resolution(self) -> str:
        """获取推荐的分辨率"""
        profile = self.detect_profile()
        resolutions = {
            "flagship_ultra": "1080p",
            "high_perf": "720p",
            "balanced": "720p",
            "entry_mid": "480p",
            "entry_low": "360p",
            "cpu_safe": "256p"
        }
        return resolutions.get(profile, "256p")


if __name__ == "__main__":
    print("智能配置适配器测试")
    
    # 创建模拟配置
    class MockConfig:
        class model:
            max_block_frames = 48
            memory_size = 256
            tile_batch_size = 1
            vae_type = 'image'
            use_turboquant = False
            teacache_threshold = 0.25
            use_internal_memory = False
            use_multi_scale_memory = False
            use_hierarchical_memory = True
            use_learned_compressor = True
            use_token_routing = True
            lean_vae_path = './models/LeanVAE'
    
    config = MockConfig()
    
    # 应用优化
    adapter = AutoConfigAdapter()
    adapter.apply_optimizations(config)
    
    print(f"\n推荐批大小: {adapter.get_recommended_batch_size()}")
    print(f"推荐分辨率: {adapter.get_recommended_resolution()}")
