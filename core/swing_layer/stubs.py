"""
共享的 PyTorch fallback stub 定义。
当 torch 未安装时，提供 _TensorStub、torch、nn、F 等占位对象供类型注解使用。
"""

# PyTorch support
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False
    # Fallback stubs for type hints when torch is not installed
    class _TensorStub:
        pass
    torch = type('torch', (), {'Tensor': _TensorStub})()
    nn = type('nn', (), {'Module': object})()
    F = type('F', (), {})()
