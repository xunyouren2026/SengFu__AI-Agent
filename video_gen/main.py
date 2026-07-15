"""
主入口模块 - AI视频生成系统
支持训练、推理、API、WebUI四种模式
"""

import os
import sys
import argparse
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent))


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="AI视频生成系统 - Ultimate Optimized",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 训练模式
  python main.py --mode train --config config.yaml
  
  # 推理模式
  python main.py --mode infer --prompt "a cat playing" --duration 10
  
  # API服务模式
  python main.py --mode api --config config.yaml
  
  # WebUI模式
  python main.py --mode webui --config config.yaml
        """
    )
    
    # 基本参数
    parser.add_argument(
        "--mode",
        choices=["train", "infer", "api", "webui"],
        required=True,
        help="运行模式: train(训练), infer(推理), api(API服务), webui(Web界面)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="配置文件路径 (默认: config.yaml)"
    )
    
    # 模型参数
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="模型检查点路径"
    )
    parser.add_argument(
        "--pretrained",
        type=str,
        help="预训练模型标识 (HuggingFace ID或本地路径)"
    )
    
    # 生成参数
    parser.add_argument(
        "--prompt",
        type=str,
        default="a beautiful landscape",
        help="生成提示词"
    )
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default="",
        help="负面提示词"
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=2.0,
        help="视频时长 (秒)"
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=8,
        help="视频帧率"
    )
    parser.add_argument(
        "--resolution",
        type=str,
        default="256p",
        choices=["256p", "360p", "480p", "720p", "1080p", "4k", "8k"],
        help="输出分辨率"
    )
    parser.add_argument(
        "--width",
        type=int,
        help="自定义宽度 (像素)"
    )
    parser.add_argument(
        "--height",
        type=int,
        help="自定义高度 (像素)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output.mp4",
        help="输出文件路径"
    )
    
    # 高级参数
    parser.add_argument(
        "--cfg_scale",
        type=float,
        default=7.5,
        help="CFG引导强度"
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=50,
        help="推理步数"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子"
    )
    
    # 长视频参数
    parser.add_argument(
        "--long_script",
        type=str,
        help="镜头脚本路径 (用于长视频)"
    )
    parser.add_argument(
        "--use_parallel",
        action="store_true",
        help="启用多GPU并行生成"
    )
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=0,
        help="使用的GPU数量 (0=自动检测)"
    )
    
    # 系统参数
    parser.add_argument(
        "--no_auto_config",
        action="store_true",
        help="禁用智能显存适配"
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="分布式训练本地rank"
    )
    
    args = parser.parse_args()
    
    # 打印欢迎信息
    print("\n" + "=" * 70)
    print("🎬 AI视频生成系统 - Ultimate Optimized")
    print("=" * 70)
    print(f"运行模式: {args.mode}")
    print(f"配置文件: {args.config}")
    print("=" * 70 + "\n")
    
    # 根据模式执行
    if args.mode == "train":
        run_train(args)
    elif args.mode == "infer":
        run_infer(args)
    elif args.mode == "api":
        run_api(args)
    elif args.mode == "webui":
        run_webui(args)
    else:
        print(f"❌ 未知模式: {args.mode}")
        sys.exit(1)


def run_train(args):
    """运行训练模式"""
    print("🚀 启动训练模式\n")
    
    # 导入配置
    from .models.config import Config
    config = Config(args.config)
    
    # 分布式初始化
    try:
        import torch
        import torch.distributed as dist
        
        if args.local_rank != -1:
            torch.cuda.set_device(args.local_rank)
            dist.init_process_group(backend='nccl', init_method='env://')
            config.train.local_rank = args.local_rank
            config.train.distributed = True
            config.train.world_size = dist.get_world_size()
        else:
            config.train.distributed = False
            config.train.local_rank = 0
            config.train.world_size = 1
    except Exception as e:
        print(f"⚠️ 分布式初始化失败: {e}")
        config.train.distributed = False
    
    # 设置随机种子
    from .utils import set_seed
    set_seed(args.seed)
    
    # 创建设备
    try:
        import torch
        device = torch.device(f"cuda:{config.train.local_rank}" if torch.cuda.is_available() else "cpu")
    except:
        device = "cpu"
    
    print(f"设备: {device}")
    print(f"分布式: {config.train.distributed}")
    print(f"World Size: {config.train.world_size}\n")
    
    # 加载模型
    print("📦 加载模型组件...")
    from .model_loader import load_pretrained_model
    from .models.vae import VideoVAE
    from .models.diffusion import DiffusionScheduler
    from .multimodal.encoders import TextEncoder, ImageEncoder, AudioEncoder, VideoEncoder
    from .multimodal.lens_controller import LensController
    
    model = load_pretrained_model(config, device, args.pretrained)
    vae = VideoVAE(config.model, device=device)
    scheduler = DiffusionScheduler(config.model, device)
    text_encoder = TextEncoder(config.model.text_encoder_model, device)
    
    image_encoder = ImageEncoder(config.model.image_encoder_model, device) if config.model.use_image_encoder else None
    audio_encoder = AudioEncoder(config.model.audio_encoder_model, device) if config.model.use_audio_encoder else None
    video_encoder = VideoEncoder(vae, config.model) if config.model.use_video_encoder else None
    lens_controller = LensController(config.model)
    
    print("✅ 模型加载完成!\n")
    
    # 加载检查点
    if args.checkpoint:
        print(f"📂 加载检查点: {args.checkpoint}")
        try:
            import torch
            state = torch.load(args.checkpoint, map_location=device)
            model.load_state_dict(state['model'], strict=False)
            vae.load_state_dict(state['vae'], strict=False)
            print("✅ 检查点加载完成!\n")
        except Exception as e:
            print(f"⚠️ 检查点加载失败: {e}\n")
    
    # 创建训练器
    from .training.trainer import ProgressiveTrainer
    trainer = ProgressiveTrainer(
        config=config,
        model=model,
        vae=vae,
        scheduler=scheduler,
        text_encoder=text_encoder,
        image_encoder=image_encoder,
        audio_encoder=audio_encoder,
        video_encoder=video_encoder,
        lens_controller=lens_controller,
        device=device
    )
    
    # 加载数据
    print("📊 加载训练数据...")
    from .data.dataset import get_dataloader
    train_loader = get_dataloader(config.data, split="train")
    val_loader = get_dataloader(config.data, split="val")
    print(f"✅ 训练样本: {len(train_loader.dataset)}")
    print(f"✅ 验证样本: {len(val_loader.dataset)}\n")
    
    # 开始训练
    print("🏃 开始训练...\n")
    trainer.train(train_loader, val_loader)
    
    print("\n✅ 训练完成!")


def run_infer(args):
    """运行推理模式"""
    print("🚀 启动推理模式\n")
    
    # 导入配置
    from .models.config import Config
    from .auto_config_adapter import AutoConfigAdapter
    
    config = Config(args.config)
    
    # 智能显存适配
    if not args.no_auto_config:
        try:
            adapter = AutoConfigAdapter()
            adapter.apply_optimizations(config)
        except Exception as e:
            print(f"⚠️ 显存适配失败: {e}")
    
    # 设置随机种子
    from .utils import set_seed
    set_seed(args.seed)
    
    # 创建设备
    try:
        import torch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    except:
        device = "cpu"
    
    print(f"设备: {device}\n")
    
    # 加载模型
    print("📦 加载模型组件...")
    from .model_loader import load_pretrained_model
    from .models.vae import VideoVAE
    from .models.diffusion import DiffusionScheduler
    from .multimodal.encoders import TextEncoder, ImageEncoder, AudioEncoder, VideoEncoder
    from .multimodal.lens_controller import LensController
    from .inferencer import Inferencer
    
    model = load_pretrained_model(config, device, args.pretrained)
    vae = VideoVAE(config.model, device=device)
    scheduler = DiffusionScheduler(config.model, device)
    text_encoder = TextEncoder(config.model.text_encoder_model, device)
    
    image_encoder = ImageEncoder(config.model.image_encoder_model, device) if config.model.use_image_encoder else None
    audio_encoder = AudioEncoder(config.model.audio_encoder_model, device) if config.model.use_audio_encoder else None
    video_encoder = VideoEncoder(vae, config.model) if config.model.use_video_encoder else None
    lens_controller = LensController(config.model)
    
    print("✅ 模型加载完成!\n")
    
    # 加载检查点
    if args.checkpoint:
        print(f"📂 加载检查点: {args.checkpoint}")
        try:
            import torch
            state = torch.load(args.checkpoint, map_location=device)
            model.load_state_dict(state['model'], strict=False)
            vae.load_state_dict(state['vae'], strict=False)
            print("✅ 检查点加载完成!\n")
        except Exception as e:
            print(f"⚠️ 检查点加载失败: {e}\n")
    
    # 创建推理器
    inferencer = Inferencer(
        config=config,
        model=model,
        vae=vae,
        scheduler=scheduler,
        text_encoder=text_encoder,
        image_encoder=image_encoder,
        audio_encoder=audio_encoder,
        video_encoder=video_encoder,
        lens_controller=lens_controller,
        device=device
    )
    
    # 生成视频
    print(f"🎬 开始生成视频...")
    print(f"   提示词: {args.prompt}")
    print(f"   时长: {args.duration}s")
    print(f"   分辨率: {args.resolution}")
    print(f"   帧率: {args.fps}fps\n")
    
    try:
        if args.duration < 30:
            # 短视频：创意模式
            output_path = inferencer.generate(
                prompt=args.prompt,
                negative_prompt=args.negative_prompt,
                duration=args.duration,
                fps=args.fps,
                resolution=args.resolution,
                width=args.width,
                height=args.height,
                cfg_scale=args.cfg_scale,
                num_steps=args.steps,
                output_path=args.output
            )
        else:
            # 长视频
            optimization_level = 'full' if args.duration > 300 else 'light'
            output_path, _ = inferencer.generate_long(
                prompt=args.prompt,
                duration=args.duration,
                fps=args.fps,
                resolution=args.resolution,
                width=args.width,
                height=args.height,
                cfg_scale=args.cfg_scale,
                steps=args.steps,
                negative_prompt=args.negative_prompt,
                optimization_level=optimization_level,
                use_parallel=args.use_parallel,
                num_gpus=args.num_gpus
            )
        
        print(f"\n✅ 视频生成完成!")
        print(f"   输出路径: {output_path}")
        
    except Exception as e:
        print(f"\n❌ 生成失败: {e}")
        import traceback
        traceback.print_exc()


def run_api(args):
    """运行API服务模式"""
    print("🚀 启动API服务模式\n")
    
    # 导入配置
    from .models.config import Config
    from .auto_config_adapter import AutoConfigAdapter
    
    config = Config(args.config)
    
    # 智能显存适配
    if not args.no_auto_config:
        try:
            adapter = AutoConfigAdapter()
            adapter.apply_optimizations(config)
        except Exception as e:
            print(f"⚠️ 显存适配失败: {e}")
    
    # 设置随机种子
    from .utils import set_seed
    set_seed(args.seed)
    
    # 创建设备
    try:
        import torch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    except:
        device = "cpu"
    
    print(f"设备: {device}\n")
    
    # 加载模型
    print("📦 加载模型组件...")
    from .model_loader import load_pretrained_model
    from .models.vae import VideoVAE
    from .models.diffusion import DiffusionScheduler
    from .multimodal.encoders import TextEncoder, ImageEncoder, AudioEncoder, VideoEncoder
    from .multimodal.lens_controller import LensController
    from .inferencer import Inferencer
    from .api import app
    
    model = load_pretrained_model(config, device, args.pretrained)
    vae = VideoVAE(config.model, device=device)
    scheduler = DiffusionScheduler(config.model, device)
    text_encoder = TextEncoder(config.model.text_encoder_model, device)
    
    image_encoder = ImageEncoder(config.model.image_encoder_model, device) if config.model.use_image_encoder else None
    audio_encoder = AudioEncoder(config.model.audio_encoder_model, device) if config.model.use_audio_encoder else None
    video_encoder = VideoEncoder(vae, config.model) if config.model.use_video_encoder else None
    lens_controller = LensController(config.model)
    
    print("✅ 模型加载完成!\n")
    
    # 加载检查点
    if args.checkpoint:
        print(f"📂 加载检查点: {args.checkpoint}")
        try:
            import torch
            state = torch.load(args.checkpoint, map_location=device)
            model.load_state_dict(state['model'], strict=False)
            vae.load_state_dict(state['vae'], strict=False)
            print("✅ 检查点加载完成!\n")
        except Exception as e:
            print(f"⚠️ 检查点加载失败: {e}\n")
    
    # 创建推理器
    inferencer = Inferencer(
        config=config,
        model=model,
        vae=vae,
        scheduler=scheduler,
        text_encoder=text_encoder,
        image_encoder=image_encoder,
        audio_encoder=audio_encoder,
        video_encoder=video_encoder,
        lens_controller=lens_controller,
        device=device
    )
    
    # 设置应用状态
    app.state.inferencer = inferencer
    app.state.results = {}
    app.state.tasks = {}
    
    # 创建临时目录
    os.makedirs(config.api.temp_dir, exist_ok=True)
    
    print("\n" + "=" * 70)
    print(f"🌐 API服务已启动")
    print(f"   访问地址: http://{config.api.host}:{config.api.port}")
    print(f"   API文档: http://{config.api.host}:{config.api.port}/docs")
    print("=" * 70 + "\n")
    
    # 启动服务器
    try:
        import uvicorn
        uvicorn.run(app, host=config.api.host, port=config.api.port)
    except ImportError:
        print("⚠️ uvicorn未安装，无法启动API服务")
        print("   请安装: pip install uvicorn")


def run_webui(args):
    """运行WebUI模式"""
    print("🚀 启动WebUI模式\n")
    
    # 导入配置
    from .models.config import Config
    from .auto_config_adapter import AutoConfigAdapter
    
    config = Config(args.config)
    
    # 智能显存适配
    if not args.no_auto_config:
        try:
            adapter = AutoConfigAdapter()
            adapter.apply_optimizations(config)
        except Exception as e:
            print(f"⚠️ 显存适配失败: {e}")
    
    # 设置随机种子
    from .utils import set_seed
    set_seed(args.seed)
    
    # 创建设备
    try:
        import torch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    except:
        device = "cpu"
    
    print(f"设备: {device}\n")
    
    # 加载模型
    print("📦 加载模型组件...")
    from .model_loader import load_pretrained_model
    from .models.vae import VideoVAE
    from .models.diffusion import DiffusionScheduler
    from .multimodal.encoders import TextEncoder, ImageEncoder, AudioEncoder, VideoEncoder
    from .multimodal.lens_controller import LensController
    from .inferencer import Inferencer
    
    model = load_pretrained_model(config, device, args.pretrained)
    vae = VideoVAE(config.model, device=device)
    scheduler = DiffusionScheduler(config.model, device)
    text_encoder = TextEncoder(config.model.text_encoder_model, device)
    
    image_encoder = ImageEncoder(config.model.image_encoder_model, device) if config.model.use_image_encoder else None
    audio_encoder = AudioEncoder(config.model.audio_encoder_model, device) if config.model.use_audio_encoder else None
    video_encoder = VideoEncoder(vae, config.model) if config.model.use_video_encoder else None
    lens_controller = LensController(config.model)
    
    print("✅ 模型加载完成!\n")
    
    # 加载检查点
    if args.checkpoint:
        print(f"📂 加载检查点: {args.checkpoint}")
        try:
            import torch
            state = torch.load(args.checkpoint, map_location=device)
            model.load_state_dict(state['model'], strict=False)
            vae.load_state_dict(state['vae'], strict=False)
            print("✅ 检查点加载完成!\n")
        except Exception as e:
            print(f"⚠️ 检查点加载失败: {e}\n")
    
    # 创建推理器
    inferencer = Inferencer(
        config=config,
        model=model,
        vae=vae,
        scheduler=scheduler,
        text_encoder=text_encoder,
        image_encoder=image_encoder,
        audio_encoder=audio_encoder,
        video_encoder=video_encoder,
        lens_controller=lens_controller,
        device=device
    )
    
    # 启动WebUI
    from .webui import create_webui
    create_webui(inferencer)


if __name__ == "__main__":
    main()
