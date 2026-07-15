"""
AGI Unified Framework Security Module Entry Point
===================================================

支持通过 python -m security 命令直接运行各子模块的CLI

Usage:
    python -m security double_auth [command]
    python -m security permission_boundary [command]
    python -m security capability_minimizer [command]
"""

import sys
import importlib

def main():
    """主入口函数"""
    if len(sys.argv) < 2:
        print("AGI Unified Framework Security Module")
        print("")
        print("可用模块:")
        print("  double_auth          - 双重授权框架")
        print("  permission_boundary  - 权限边界与RBAC")
        print("  capability_minimizer - 能力最小化控制")
        print("")
        print("Usage: python -m security.<module_name> [command]")
        print("示例: python -m security.double_auth check --action click --params '{}'")
        sys.exit(1)
    
    module_name = sys.argv[1]
    
    # 移除已处理的参数
    sys.argv = sys.argv[1:]
    
    try:
        # 动态导入模块
        if module_name == 'double_auth':
            from . import double_auth
            double_auth.main()
        elif module_name == 'permission_boundary':
            from . import permission_boundary
            permission_boundary.main()
        elif module_name == 'capability_minimizer':
            from . import capability_minimizer
            capability_minimizer.main()
        else:
            print(f"错误: 未知模块 '{module_name}'")
            print("运行 'python -m security' 查看可用模块列表")
            sys.exit(1)
    except ImportError as e:
        print(f"错误: 无法导入模块 '{module_name}'")
        print(f"详细信息: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"错误: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
