#!/usr/bin/env python3
"""
PWA图标生成脚本

生成UFO AGI框架的PWA图标：
- icon-192.png (192x192)
- icon-512.png (512x512)

图标设计：蓝色渐变背景，白色UFO图标
"""

from PIL import Image, ImageDraw, ImageFilter
import math


def create_gradient_background(size: int, color1: tuple, color2: tuple) -> Image.Image:
    """创建渐变背景"""
    img = Image.new('RGB', (size, size))
    draw = ImageDraw.Draw(img)
    
    for y in range(size):
        r = int(color1[0] + (color2[0] - color1[0]) * y / size)
        g = int(color1[1] + (color2[1] - color1[1]) * y / size)
        b = int(color1[2] + (color2[2] - color1[2]) * y / size)
        draw.line([(0, y), (size, y)], fill=(r, g, b))
    
    return img


def draw_ufo(draw: ImageDraw.Draw, center_x: int, center_y: int, scale: float):
    """绘制UFO图标"""
    # UFO主体 - 椭圆形的飞碟底部
    body_width = int(120 * scale)
    body_height = int(40 * scale)
    body_top = center_y - int(20 * scale)
    body_bottom = body_top + body_height
    
    # 绘制飞碟底部阴影
    shadow_offset = int(5 * scale)
    draw.ellipse(
        [center_x - body_width//2 + shadow_offset, body_top + shadow_offset,
         center_x + body_width//2 + shadow_offset, body_bottom + shadow_offset],
        fill=(0, 0, 0, 80)
    )
    
    # 绘制飞碟底部（主体）
    draw.ellipse(
        [center_x - body_width//2, body_top,
         center_x + body_width//2, body_bottom],
        fill=(255, 255, 255),
        outline=(200, 220, 255),
        width=int(2 * scale)
    )
    
    # 绘制飞碟顶部圆顶
    dome_width = int(60 * scale)
    dome_height = int(50 * scale)
    dome_top = center_y - int(60 * scale)
    
    # 圆顶高光
    draw.ellipse(
        [center_x - dome_width//2, dome_top,
         center_x + dome_width//2, dome_top + dome_height],
        fill=(240, 248, 255),
        outline=(255, 255, 255),
        width=int(2 * scale)
    )
    
    # 圆顶内部渐变效果
    inner_dome_width = int(40 * scale)
    inner_dome_height = int(30 * scale)
    inner_dome_top = dome_top + int(10 * scale)
    draw.ellipse(
        [center_x - inner_dome_width//2, inner_dome_top,
         center_x + inner_dome_width//2, inner_dome_top + inner_dome_height],
        fill=(200, 230, 255)
    )
    
    # 绘制飞碟底部灯光
    light_count = 5
    light_radius = int(6 * scale)
    light_y = body_top + body_height // 2
    light_spacing = body_width // (light_count + 1)
    
    for i in range(light_count):
        light_x = center_x - body_width//2 + light_spacing * (i + 1)
        # 灯光发光效果
        for r in range(light_radius + 2, light_radius - 1, -1):
            alpha = int(100 * (light_radius + 2 - r) / 3)
            glow_color = (150 + alpha, 200 + alpha//2, 255)
            draw.ellipse(
                [light_x - r, light_y - r, light_x + r, light_y + r],
                fill=glow_color
            )
        # 灯光核心
        draw.ellipse(
            [light_x - light_radius//2, light_y - light_radius//2,
             light_x + light_radius//2, light_y + light_radius//2],
            fill=(255, 255, 255)
        )
    
    # 绘制光束
    beam_width_top = int(30 * scale)
    beam_width_bottom = int(80 * scale)
    beam_height = int(60 * scale)
    beam_top = body_bottom
    beam_bottom = beam_top + beam_height
    
    # 使用多边形绘制光束
    beam_points = [
        (center_x - beam_width_top//2, beam_top),
        (center_x + beam_width_top//2, beam_top),
        (center_x + beam_width_bottom//2, beam_bottom),
        (center_x - beam_width_bottom//2, beam_bottom),
    ]
    
    # 创建半透明光束层
    beam_layer = Image.new('RGBA', (512, 512), (0, 0, 0, 0))
    beam_draw = ImageDraw.Draw(beam_layer)
    beam_draw.polygon(beam_points, fill=(200, 230, 255, 60))
    
    return beam_layer


def generate_icon(size: int, output_path: str):
    """生成指定尺寸的图标"""
    # 颜色定义 - 蓝色渐变
    color_top = (15, 23, 42)      # 深蓝灰色 #0f172a
    color_bottom = (59, 130, 246)  # 亮蓝色 #3b82f6
    
    # 创建基础图像（RGBA模式以支持透明）
    img = Image.new('RGBA', (size, size), (0, 0, 0, 0))
    
    # 创建渐变背景
    bg = create_gradient_background(size, color_top, color_bottom)
    bg = bg.convert('RGBA')
    
    # 计算缩放比例
    scale = size / 512
    
    # 创建UFO图层
    ufo_layer = Image.new('RGBA', (size, size), (0, 0, 0, 0))
    ufo_draw = ImageDraw.Draw(ufo_layer)
    
    center_x = size // 2
    center_y = size // 2 - int(20 * scale)
    
    # 绘制光束（在UFO下方）
    beam_layer = draw_ufo(ufo_draw, center_x, center_y, scale)
    
    # 重新绘制UFO主体（不带光束）
    ufo_layer = Image.new('RGBA', (size, size), (0, 0, 0, 0))
    ufo_draw = ImageDraw.Draw(ufo_layer)
    
    # 手动绘制UFO（不调用draw_ufo，避免光束）
    body_width = int(120 * scale)
    body_height = int(40 * scale)
    body_top = center_y - int(20 * scale)
    body_bottom = body_top + body_height
    
    # 飞碟底部阴影
    shadow_offset = int(5 * scale)
    ufo_draw.ellipse(
        [center_x - body_width//2 + shadow_offset, body_top + shadow_offset,
         center_x + body_width//2 + shadow_offset, body_bottom + shadow_offset],
        fill=(0, 0, 0, 80)
    )
    
    # 飞碟底部
    ufo_draw.ellipse(
        [center_x - body_width//2, body_top,
         center_x + body_width//2, body_bottom],
        fill=(255, 255, 255, 255),
        outline=(200, 220, 255, 255),
        width=int(2 * scale)
    )
    
    # 圆顶
    dome_width = int(60 * scale)
    dome_height = int(50 * scale)
    dome_top = center_y - int(60 * scale)
    
    ufo_draw.ellipse(
        [center_x - dome_width//2, dome_top,
         center_x + dome_width//2, dome_top + dome_height],
        fill=(240, 248, 255, 255),
        outline=(255, 255, 255, 255),
        width=int(2 * scale)
    )
    
    # 内部渐变
    inner_dome_width = int(40 * scale)
    inner_dome_height = int(30 * scale)
    inner_dome_top = dome_top + int(10 * scale)
    ufo_draw.ellipse(
        [center_x - inner_dome_width//2, inner_dome_top,
         center_x + inner_dome_width//2, inner_dome_top + inner_dome_height],
        fill=(200, 230, 255, 255)
    )
    
    # 灯光
    light_count = 5
    light_radius = int(6 * scale)
    light_y = body_top + body_height // 2
    light_spacing = body_width // (light_count + 1)
    
    for i in range(light_count):
        light_x = center_x - body_width//2 + light_spacing * (i + 1)
        for r in range(light_radius + 2, light_radius - 1, -1):
            alpha = int(100 * (light_radius + 2 - r) / 3)
            glow_color = (150 + alpha, 200 + alpha//2, 255, 150)
            ufo_draw.ellipse(
                [light_x - r, light_y - r, light_x + r, light_y + r],
                fill=glow_color
            )
        ufo_draw.ellipse(
            [light_x - light_radius//2, light_y - light_radius//2,
             light_x + light_radius//2, light_y + light_radius//2],
            fill=(255, 255, 255, 255)
        )
    
    # 光束
    beam_width_top = int(30 * scale)
    beam_width_bottom = int(80 * scale)
    beam_height = int(60 * scale)
    beam_top = body_bottom
    beam_bottom = beam_top + beam_height
    
    beam_points = [
        (center_x - beam_width_top//2, beam_top),
        (center_x + beam_width_top//2, beam_top),
        (center_x + beam_width_bottom//2, beam_bottom),
        (center_x - beam_width_bottom//2, beam_bottom),
    ]
    ufo_draw.polygon(beam_points, fill=(200, 230, 255, 80))
    
    # 合并图层
    img = Image.alpha_composite(bg, ufo_layer)
    
    # 转换为RGB并保存
    img_rgb = img.convert('RGB')
    img_rgb.save(output_path, 'PNG', quality=95)
    print(f"Generated: {output_path} ({size}x{size})")


def main():
    """主函数"""
    import os
    
    # 获取脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 生成192x192图标
    generate_icon(192, os.path.join(script_dir, 'icon-192.png'))
    
    # 生成512x512图标
    generate_icon(512, os.path.join(script_dir, 'icon-512.png'))
    
    print("All icons generated successfully!")


if __name__ == "__main__":
    main()
