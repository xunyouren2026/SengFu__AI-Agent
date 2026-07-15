#!/usr/bin/env python3
"""
生成PNG格式的图标文件
从SVG转换或创建新的PNG
"""

try:
    from PIL import Image, ImageDraw
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    print("PIL not available, skipping PNG generation")

import os


def create_avatar_png():
    """创建头像PNG"""
    if not HAS_PIL:
        return
    
    # 创建64x64的头像
    img = Image.new('RGBA', (64, 64), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    
    # 绘制圆形背景（蓝色渐变效果）
    for i in range(64):
        alpha = int(255 * (1 - i/64))
        draw.ellipse([0, 0, 64, 64], fill=(59, 130, 246, 255))
    
    # 绘制简单的用户图标（白色圆形）
    draw.ellipse([16, 12, 48, 44], fill=(255, 255, 255, 230))  # 头
    draw.ellipse([20, 36, 44, 60], fill=(255, 255, 255, 230))  # 身体
    
    img.save('avatar.png', 'PNG')
    print("Created avatar.png")


def create_logo_png():
    """创建logo PNG"""
    if not HAS_PIL:
        return
    
    # 创建128x128的logo
    img = Image.new('RGBA', (128, 128), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    
    # 绘制UFO形状
    # 主体（椭圆）
    draw.ellipse([14, 48, 114, 88], fill=(59, 130, 246, 255), outline=(37, 99, 235, 255), width=3)
    # 顶部圆顶
    draw.ellipse([44, 28, 84, 58], fill=(147, 197, 253, 255), outline=(59, 130, 246, 255), width=2)
    # 底部光束
    for i in range(5):
        x = 28 + i * 18
        draw.polygon([(x, 88), (x+8, 88), (x+4, 108)], fill=(251, 191, 36, 180))
    
    img.save('logo.png', 'PNG')
    print("Created logo.png")


def create_favicon_png():
    """创建favicon PNG"""
    if not HAS_PIL:
        return
    
    # 创建32x32的favicon
    img = Image.new('RGBA', (32, 32), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    
    # 绘制简化的UFO
    draw.ellipse([4, 14, 28, 24], fill=(59, 130, 246, 255))
    draw.ellipse([10, 8, 22, 16], fill=(147, 197, 253, 255))
    
    img.save('favicon.png', 'PNG')
    print("Created favicon.png")


if __name__ == "__main__":
    if HAS_PIL:
        create_avatar_png()
        create_logo_png()
        create_favicon_png()
        print("All PNG files created successfully!")
    else:
        print("Please install Pillow: pip install Pillow")
