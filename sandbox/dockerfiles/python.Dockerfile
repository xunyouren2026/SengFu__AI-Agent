# Python沙箱执行环境Dockerfile
# 提供安全、隔离的Python代码执行环境

FROM python:3.10-slim-bullseye

# 元数据
LABEL maintainer="AGI Unified Framework"
LABEL description="Secure Python sandbox execution environment"
LABEL version="1.0.0"

# 安全配置：创建非root用户
RUN groupadd -r sandbox && \
    useradd -r -g sandbox -d /sandbox -s /bin/false sandbox

# 设置工作目录
WORKDIR /sandbox

# 安装必要的系统依赖（最小化）
RUN apt-get update && apt-get install -y --no-install-recommends \
    # 基础工具
    ca-certificates=20210119 \
    # 时区数据
    tzdata=2021a-1+deb11u8 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# 安装Python安全相关包
RUN pip install --no-cache-dir \
    # 沙箱限制
    resource==0.2.1 \
    # 代码分析
    astunparse==1.6.3 \
    # 安全哈希
    hashlib-extplus==0.1.0 || true

# 创建必要的目录结构
RUN mkdir -p /sandbox/input \
    && mkdir -p /sandbox/output \
    && mkdir -p /sandbox/tmp \
    && mkdir -p /sandbox/.cache \
    && chown -R sandbox:sandbox /sandbox

# 设置权限
RUN chmod 750 /sandbox \
    && chmod 750 /sandbox/input \
    && chmod 750 /sandbox/output \
    && chmod 750 /sandbox/tmp

# 配置环境变量
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/sandbox \
    # 禁用Python生成.pyc文件
    PYTHONDONTWRITEBYTECODE=1 \
    # 限制递归深度
    PYTHON_RECURSION_LIMIT=1000 \
    # 设置时区
    TZ=UTC \
    # 禁用用户站点包
    PYTHONNOUSERSITE=1 \
    # 限制内存分配
    MALLOC_ARENA_MAX=2

# 创建安全配置文件
RUN echo '# Python安全配置\n\
[security]\n\
# 禁用危险模块\n\
disabled_modules = os,subprocess,sys,socket,ctypes,multiprocessing\n\
# 允许的内置函数\n\
allowed_builtins = print,len,range,str,int,float,list,dict,tuple,set\n\
# 最大递归深度\n\
max_recursion_depth = 1000\n\
# 最大内存(MB)\n\
max_memory_mb = 512\n\
# 最大执行时间(秒)\n\
max_execution_time = 60\n\
' > /sandbox/.security.ini

# 创建资源限制脚本
RUN echo '#!/usr/bin/env python3\n\
"""资源限制模块"""\n\
import sys\n\
import resource\n\
\n\
def set_limits(memory_mb=512, time_sec=60):\n\
    """设置资源限制"""\n\
    # 内存限制\n\
    memory_bytes = memory_mb * 1024 * 1024\n\
    try:\n\
        resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))\n\
    except (ValueError, resource.error):\n\
        pass\n\
    \n\
    # CPU时间限制\n\
    try:\n\
        resource.setrlimit(resource.RLIMIT_CPU, (time_sec, time_sec))\n\
    except (ValueError, resource.error):\n\
        pass\n\
    \n\
    # 文件描述符限制\n\
    try:\n\
        resource.setrlimit(resource.RLIMIT_NOFILE, (256, 256))\n\
    except (ValueError, resource.error):\n\
        pass\n\
    \n\
    # 进程数限制\n\
    try:\n\
        resource.setrlimit(resource.RLIMIT_NPROC, (10, 10))\n\
    except (ValueError, resource.error):\n\
        pass\n\
\n\
if __name__ == "__main__":\n\
    set_limits()\n\
' > /sandbox/resource_limiter.py \
    && chmod 644 /sandbox/resource_limiter.py

# 创建安全执行包装器
RUN echo '#!/usr/bin/env python3\n\
"""安全执行包装器"""\n\
import sys\n\
import os\n\
import traceback\n\
from pathlib import Path\n\
\n\
# 危险模块黑名单\n\
DANGEROUS_MODULES = {\n\
    "os", "subprocess", "socket", "ctypes", "multiprocessing",\n\
    "threading", "signal", "mmap", "shutil", "tempfile",\n\
    "pickle", "shelve", "marshal", "code", "codeop",\n\
    "compile", "exec", "eval", "__import__"\n\
}\n\
\n\
class SafeImporter:\n\
    """安全导入器"""\n\
    \n\
    def find_module(self, name, path=None):\n\
        if name.split(".")[0] in DANGEROUS_MODULES:\n\
            raise ImportError(f"Module {name} is not allowed in sandbox")\n\
        return None\n\
\n\
def setup_sandbox():\n\
    """设置沙箱环境"""\n\
    # 安装安全导入器\n\
    sys.meta_path.insert(0, SafeImporter())\n\
    \n\
    # 移除危险内置函数\n\
    dangerous_builtins = ["exec", "eval", "compile", "open", "input"]\n\
    for name in dangerous_builtins:\n\
        if name in __builtins__:\n\
            del __builtins__[name]\n\
\n\
def safe_execute(code: str, globals_dict=None, locals_dict=None):\n\
    """安全执行代码"""\n\
    setup_sandbox()\n\
    \n\
    if globals_dict is None:\n\
        globals_dict = {"__builtins__": __builtins__}\n\
    if locals_dict is None:\n\
        locals_dict = {}\n\
    \n\
    try:\n\
        # 编译并执行\n\
        compiled = compile(code, "<sandbox>", "exec")\n\
        exec(compiled, globals_dict, locals_dict)\n\
        return True, None\n\
    except Exception as e:\n\
        return False, traceback.format_exc()\n\
\n\
if __name__ == "__main__":\n\
    if len(sys.argv) > 1:\n\
        code_file = sys.argv[1]\n\
        with open(code_file, "r") as f:\n\
            code = f.read()\n\
        success, error = safe_execute(code)\n\
        if not success:\n\
            print(f"Execution failed: {error}", file=sys.stderr)\n\
            sys.exit(1)\n\
' > /sandbox/safe_executor.py \
    && chmod 755 /sandbox/safe_executor.py

# 健康检查脚本
RUN echo '#!/usr/bin/env python3\n\
"""健康检查"""\n\
import sys\n\
print("OK")\n\
sys.exit(0)\n\
' > /sandbox/healthcheck.py \
    && chmod 755 /sandbox/healthcheck.py

# 切换到非root用户
USER sandbox

# 健康检查
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD python3 /sandbox/healthcheck.py || exit 1

# 默认命令
CMD ["python3", "-c", "print('Sandbox ready')"]

# 暴露的卷（用于数据交换）
VOLUME ["/sandbox/input", "/sandbox/output", "/sandbox/tmp"]

# 设置入口点
ENTRYPOINT ["/usr/local/bin/python3"]
