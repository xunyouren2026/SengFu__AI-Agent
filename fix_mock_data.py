#!/usr/bin/env python3
"""
批量修复前端页面 - 删除Mock数据，添加空状态提示
"""

import os
import re

def fix_dashboard_html(filepath):
    """修复 dashboard.html"""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original = content
    
    # 1. 添加空状态提示到 body 开头
    empty_state_html = '''
    <!-- 空状态提示 -->
    <div id="empty-state" style="display:none; background:#fff3cd; border:1px solid #ffc107; 
         border-radius:8px; padding:15px; margin:20px; text-align:center; color:#856404;">
        ⚠️ 暂无数据，请确保后端服务已启动 (API: /api/v1/system/metrics)
    </div>
    '''
    
    # 找到 <div class="container"> 并在其前插入空状态
    if '<div class="container">' in content:
        content = content.replace('<div class="container">', empty_state_html + '\n    <div class="container">')
    
    # 2. 替换 displayMockData 函数
    old_mock_func = '''        // 显示模拟数据（当API不可用时）
        function displayMockData() {
            const now = new Date();
            const hour = now.getHours();
            
            // 模拟CPU使用率 (30-70%)
            const cpuValue = 30 + Math.random() * 40;
            document.getElementById('cpu-value').textContent = cpuValue.toFixed(1);
            updateProgressBar('cpu-bar', cpuValue);
            document.getElementById('cpu-percent').textContent = cpuValue.toFixed(1) + '%';
            
            // 模拟内存使用率 (40-60%)
            const memValue = 40 + Math.random() * 20;
            document.getElementById('memory-value').textContent = memValue.toFixed(1);
            updateProgressBar('memory-bar', memValue);
            document.getElementById('memory-percent').textContent = memValue.toFixed(1) + '%';
            
            // 模拟GPU使用率 (50-80%)
            const gpuValue = 50 + Math.random() * 30;
            document.getElementById('gpu-value').textContent = gpuValue.toFixed(1);
            updateProgressBar('gpu-bar', gpuValue);
            document.getElementById('gpu-percent').textContent = gpuValue.toFixed(1) + '%';
            
            // 模拟磁盘使用率 (30-50%)
            const diskValue = 30 + Math.random() * 20;
            document.getElementById('disk-value').textContent = diskValue.toFixed(1);
            updateProgressBar('disk-bar', diskValue);
            document.getElementById('disk-percent').textContent = diskValue.toFixed(1) + '%';
            
            // 显示空状态提示
            showEmptyState();
        }'''
    
    new_mock_func = '''        // 显示空状态（不再使用模拟数据）
        function showEmptyState() {
            document.getElementById('empty-state').style.display = 'block';
            // 数据显示为 --
            const placeholders = ['cpu', 'memory', 'gpu', 'disk'];
            placeholders.forEach(type => {
                document.getElementById(type + '-value').textContent = '--';
                updateProgressBar(type + '-bar', 0);
                const percentEl = document.getElementById(type + '-percent');
                if (percentEl) percentEl.textContent = '--';
            });
        }'''
    
    content = content.replace(old_mock_func, new_mock_func)
    
    # 3. 替换API失败时的处理
    old_error_handling = '''                    // 即使请求失败，也显示模拟数据
                    console.warn('API请求失败，使用模拟数据');
                    displayMockData();'''
    
    new_error_handling = '''                    // API失败时显示空状态
                    console.warn('API请求失败:', error);
                    showEmptyState();
                    document.getElementById('empty-state').style.display = 'block';'''
    
    content = content.replace(old_error_handling, new_error_handling)
    
    # 4. 添加全局空状态显示函数
    if 'function showEmptyState()' not in content:
        # 在 displayMockData 位置添加
        pass
    
    if content != original:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        return True
    return False


def fix_telemetry_html(filepath):
    """修复 telemetry.html"""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original = content
    
    # 添加空状态提示
    empty_state_html = '''
    <!-- 空状态提示 -->
    <div id="empty-state" style="display:none; background:#fff3cd; border:1px solid #ffc107; 
         border-radius:8px; padding:15px; margin:20px; text-align:center; color:#856404;">
        ⚠️ 暂无遥测数据，请确保后端服务已启动
    </div>
    '''
    
    if '<div class="container">' in content:
        content = content.replace('<div class="container">', empty_state_html + '\n    <div class="container">')
    
    # 替换 displayMockData 函数
    old_mock = '''        // 显示模拟数据（当API不可用时）
        function displayMockData() {
            const now = new Date();
            
            // 模拟请求数
            const requests = Math.floor(Math.random() * 1000);
            document.getElementById('requests-value').textContent = requests.toLocaleString();
            
            // 模拟错误率
            const errors = Math.random() * 5;
            document.getElementById('errors-value').textContent = errors.toFixed(2) + '%';
            
            // 模拟延迟
            const latency = Math.floor(Math.random() * 500) + 100;
            document.getElementById('latency-value').textContent = latency + 'ms';
            
            // 模拟CPU
            const cpu = Math.random() * 100;
            document.getElementById('cpu-value').textContent = cpu.toFixed(1) + '%';
            
            // 模拟内存
            const memory = Math.random() * 100;
            document.getElementById('memory-value').textContent = memory.toFixed(1) + '%';
            
            showEmptyState();
        }'''
    
    new_mock = '''        // 显示空状态（不再使用模拟数据）
        function showEmptyState() {
            document.getElementById('empty-state').style.display = 'block';
            const placeholders = ['requests', 'errors', 'latency', 'cpu', 'memory'];
            placeholders.forEach(id => {
                const el = document.getElementById(id + '-value');
                if (el) {
                    if (id === 'latency') el.textContent = '--ms';
                    else if (id === 'errors') el.textContent = '--%';
                    else el.textContent = '--';
                }
            });
        }'''
    
    content = content.replace(old_mock, new_mock)
    
    # 替换API失败处理
    content = content.replace(
        "// 请求失败，显示模拟数据\n                    console.warn('API请求失败，使用模拟数据');\n                    displayMockData();",
        "// API失败时显示空状态\n                    console.warn('API请求失败:', error);\n                    showEmptyState();"
    )
    
    content = content.replace(
        "// 出错时显示模拟数据，不阻塞页面\n                displayMockData();",
        "// 出错时显示空状态\n                showEmptyState();"
    )
    
    if content != original:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        return True
    return False


def fix_hardware_html(filepath):
    """修复 hardware.html"""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original = content
    
    # 添加空状态提示
    empty_state_html = '''
    <!-- 空状态提示 -->
    <div id="empty-state" style="display:none; background:#fff3cd; border:1px solid #ffc107; 
         border-radius:8px; padding:15px; margin:20px; text-align:center; color:#856404;">
        ⚠️ 暂无硬件数据，请确保后端服务已启动
    </div>
    '''
    
    if '<div class="container">' in content:
        content = content.replace('<div class="container">', empty_state_html + '\n    <div class="container">')
    
    # 替换 Math.random() 模拟数据
    # 替换 CPU 使用率模拟
    old_cpu_random = '''data: Array.from({length: 7}, () => Math.floor(Math.random() * 40) + 40),'''
    new_cpu_random = '''data: Array.from({length: 7}, () => null),'''
    content = content.replace(old_cpu_random, new_cpu_random)
    
    # 替换设备状态更新模拟
    old_device_update = '''                            device.utilization = Math.min(100, Math.max(0, device.utilization + Math.floor(Math.random() * 10) - 5));
                            device.temp = Math.min(90, Math.max(40, device.temp + Math.floor(Math.random() * 4) - 2));'''
    new_device_update = '''                            device.utilization = null;
                            device.temp = null;'''
    content = content.replace(old_device_update, new_device_update)
    
    if content != original:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        return True
    return False


def fix_other_html(filepath):
    """修复其他页面 - 添加通用空状态处理"""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original = content
    
    # 添加空状态提示
    if '<body' in content and '<div id="empty-state"' not in content:
        empty_state_html = '''
    <!-- 空状态提示 -->
    <div id="empty-state" style="display:none; background:#fff3cd; border:1px solid #ffc107; 
         border-radius:8px; padding:15px; margin:20px; text-align:center; color:#856404;">
        ⚠️ 暂无数据，请确保后端服务已启动
    </div>
    '''
        
        # 在 body 标签后第一个 div 前插入
        match = re.search(r'(<body[^>]*>)(.*?<div)', content, re.DOTALL)
        if match:
            content = content[:match.end(1)] + empty_state_html + content[match.end(1):]
    
    # 替换 displayMockData 调用
    content = re.sub(
        r"displayMockData\(\)",
        "showEmptyState()",
        content
    )
    
    # 替换 console.warn('API请求失败，使用模拟数据')
    content = re.sub(
        r"console\.warn\(['\"]API请求失败，使用模拟数据['\"]\)",
        "console.warn('API请求失败')",
        content
    )
    
    if content != original:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        return True
    return False


def main():
    pages_dir = '/workspace/AGI_FIX/web/pages'
    
    fixes = [
        (os.path.join(pages_dir, 'dashboard.html'), fix_dashboard_html),
        (os.path.join(pages_dir, 'telemetry.html'), fix_telemetry_html),
        (os.path.join(pages_dir, 'hardware.html'), fix_hardware_html),
    ]
    
    count = 0
    for filepath, fixer in fixes:
        if os.path.exists(filepath):
            if fixer(filepath):
                print(f"✅ 修复: {os.path.basename(filepath)}")
                count += 1
            else:
                print(f"⏭️  无需修复: {os.path.basename(filepath)}")
    
    # 修复其他HTML页面
    for filename in os.listdir(pages_dir):
        if filename.endswith('.html') and filename not in ['dashboard.html', 'telemetry.html', 'hardware.html']:
            filepath = os.path.join(pages_dir, filename)
            if fix_other_html(filepath):
                print(f"✅ 修复: {filename}")
                count += 1
    
    print(f"\n总计修复: {count} 个文件")


if __name__ == '__main__':
    main()
