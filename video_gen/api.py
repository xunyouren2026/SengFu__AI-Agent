"""
API服务模块 - FastAPI后端
提供视频生成REST API接口
"""

import os
import sys
import uuid
import shutil
import tempfile
from typing import Optional, List, Dict, Any
from pathlib import Path

# FastAPI相关导入（使用纯Python模拟）
class FastAPI:
    """FastAPI模拟类"""
    def __init__(self):
        self.routes = []
        self.state = type('State', (), {})()
        self.middlewares = []
    
    def get(self, path: str):
        def decorator(func):
            self.routes.append(('GET', path, func))
            return func
        return decorator
    
    def post(self, path: str):
        def decorator(func):
            self.routes.append(('POST', path, func))
            return func
        return decorator
    
    def add_middleware(self, middleware_class, **kwargs):
        self.middlewares.append((middleware_class, kwargs))


class CORSMiddleware:
    """CORS中间件模拟"""
    def __init__(self, allow_origins=None, allow_credentials=True, allow_methods=None, allow_headers=None):
        self.allow_origins = allow_origins or ["*"]
        self.allow_credentials = allow_credentials
        self.allow_methods = allow_methods or ["*"]
        self.allow_headers = allow_headers or ["*"]


class JSONResponse:
    """JSON响应模拟"""
    def __init__(self, content: Dict[str, Any], status_code: int = 200):
        self.content = content
        self.status_code = status_code


class HTMLResponse:
    """HTML响应模拟"""
    def __init__(self, content: str, status_code: int = 200):
        self.content = content
        self.status_code = status_code


class FileResponse:
    """文件响应模拟"""
    def __init__(self, path: str, media_type: str = "application/octet-stream"):
        self.path = path
        self.media_type = media_type


class HTTPException(Exception):
    """HTTP异常"""
    def __init__(self, status_code: int, detail: str):
        self.status_code = status_code
        self.detail = detail
        super().__init__(detail)


class BackgroundTasks:
    """后台任务"""
    def __init__(self):
        self.tasks = []
    
    def add_task(self, func, *args, **kwargs):
        self.tasks.append((func, args, kwargs))


class UploadFile:
    """上传文件模拟"""
    def __init__(self, filename: str = "", file=None):
        self.filename = filename
        self.file = file or tempfile.NamedTemporaryFile(delete=False)
    
    async def read(self):
        self.file.seek(0)
        return self.file.read()
    
    async def write(self, data):
        self.file.write(data)
    
    def close(self):
        if self.file:
            self.file.close()


class Form:
    """表单字段"""
    def __init__(self, default=..., description: str = ""):
        self.default = default
        self.description = description


class File:
    """文件字段"""
    def __init__(self, default=None):
        self.default = default


# 创建FastAPI应用
app = FastAPI()

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局脚本生成器实例（懒加载）
_script_gen_instance = None


def get_script_generator():
    """懒加载脚本生成器，根据配置初始化"""
    global _script_gen_instance
    if _script_gen_instance is None:
        if not hasattr(app.state, 'inferencer') or app.state.inferencer is None:
            raise RuntimeError("Inferencer not initialized")
        
        cfg = app.state.inferencer.config.script_gen
        if not cfg.enabled:
            raise HTTPException(status_code=501, detail="脚本生成功能已禁用，请检查配置。")
        
        from .script_generator import ScriptGenerator
        if cfg.mode == "local":
            _script_gen_instance = ScriptGenerator(
                use_local=True,
                model_name=cfg.model_name,
                api_key=None
            )
        elif cfg.mode == "api":
            if not cfg.api_key:
                raise HTTPException(status_code=500, detail="API 模式需要提供 api_key")
            _script_gen_instance = ScriptGenerator(
                use_local=False,
                model_name=cfg.model_name,
                api_key=cfg.api_key
            )
        else:
            raise HTTPException(status_code=500, detail=f"未知的脚本生成模式: {cfg.mode}")
    return _script_gen_instance


# 主界面 HTML 模板
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>AI视频生成系统</title>
    <meta charset="UTF-8">
    <style>
        body { font-family: Arial, sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; }
        h1 { color: #333; }
        .form-group { margin-bottom: 15px; }
        label { display: block; margin-bottom: 5px; font-weight: bold; }
        input, textarea, select { width: 100%; padding: 8px; border: 1px solid #ddd; border-radius: 4px; }
        button { background: #007bff; color: white; padding: 10px 20px; border: none; border-radius: 4px; cursor: pointer; }
        button:hover { background: #0056b3; }
        .status { margin-top: 20px; padding: 15px; border-radius: 4px; }
        .status.pending { background: #fff3cd; }
        .status.completed { background: #d4edda; }
        .status.error { background: #f8d7da; }
    </style>
</head>
<body>
    <h1>🎬 AI视频生成系统</h1>
    <form id="generateForm">
        <div class="form-group">
            <label>提示词 (Prompt)</label>
            <textarea name="prompt" rows="3" placeholder="描述你想要生成的视频..."></textarea>
        </div>
        <div class="form-group">
            <label>负面提示词 (Negative Prompt)</label>
            <input type="text" name="negative" placeholder="不想出现的内容...">
        </div>
        <div class="form-group">
            <label>时长 (秒)</label>
            <input type="number" name="duration" value="2.0" step="0.5">
        </div>
        <div class="form-group">
            <label>帧率 (FPS)</label>
            <input type="number" name="fps" value="8">
        </div>
        <div class="form-group">
            <label>分辨率</label>
            <select name="resolution">
                <option value="256p">256p</option>
                <option value="360p">360p</option>
                <option value="480p">480p</option>
                <option value="720p">720p</option>
                <option value="1080p" selected>1080p</option>
                <option value="4k">4K</option>
                <option value="8k">8K</option>
            </select>
        </div>
        <button type="submit">生成视频</button>
    </form>
    <div id="status" class="status" style="display:none;"></div>
    <script>
        document.getElementById('generateForm').onsubmit = async function(e) {
            e.preventDefault();
            const formData = new FormData(e.target);
            formData.append('async_mode', 'true');
            
            const statusDiv = document.getElementById('status');
            statusDiv.style.display = 'block';
            statusDiv.className = 'status pending';
            statusDiv.textContent = '提交任务中...';
            
            try {
                const response = await fetch('/generate', { method: 'POST', body: formData });
                const data = await response.json();
                
                if (data.task_id) {
                    statusDiv.textContent = '任务已提交，ID: ' + data.task_id;
                    checkStatus(data.task_id);
                } else {
                    statusDiv.className = 'status error';
                    statusDiv.textContent = '提交失败';
                }
            } catch (err) {
                statusDiv.className = 'status error';
                statusDiv.textContent = '错误: ' + err.message;
            }
        };
        
        async function checkStatus(taskId) {
            const statusDiv = document.getElementById('status');
            const check = async () => {
                const response = await fetch('/progress/' + taskId);
                const data = await response.json();
                
                if (data.status === 'completed') {
                    statusDiv.className = 'status completed';
                    statusDiv.innerHTML = '生成完成! <a href="/result/' + taskId + '">下载视频</a>';
                } else if (data.status === 'error') {
                    statusDiv.className = 'status error';
                    statusDiv.textContent = '生成失败: ' + (data.detail || '未知错误');
                } else {
                    statusDiv.textContent = '状态: ' + data.status + (data.progress ? ' (' + data.progress + '%)' : '');
                    setTimeout(check, 2000);
                }
            };
            check();
        }
    </script>
</body>
</html>
"""


def run_generation_task(task_id, inferencer, args):
    """执行单个生成任务（被任务队列调用）"""
    try:
        from .utils import save_video
        
        result_path = inferencer.generate(**args)
        if result_path and os.path.exists(result_path):
            shutil.move(result_path, args['output_path'])
        
        if hasattr(app.state, 'results'):
            app.state.results[task_id] = args['output_path']
        if hasattr(app.state, 'tasks') and task_id in app.state.tasks:
            del app.state.tasks[task_id]
        
        # 清理临时文件
        for p in args.get('audio_paths', []):
            try:
                if p and os.path.exists(p):
                    os.remove(p)
            except:
                pass
        if args.get('bgm_file_path') and os.path.exists(args['bgm_file_path']):
            try:
                os.remove(args['bgm_file_path'])
            except:
                pass
                
    except Exception as e:
        print(f"任务 {task_id}: 错误 - {str(e)}")
        import traceback
        traceback.print_exc()
        if hasattr(app.state, 'results'):
            app.state.results[task_id] = {"error": str(e)}
        if hasattr(app.state, 'tasks') and task_id in app.state.tasks:
            app.state.tasks[task_id] = {"status": "error", "detail": str(e)}


# API路由定义
@app.get("/")
async def serve_ui():
    """提供主界面 HTML"""
    return HTMLResponse(HTML_TEMPLATE)


@app.post("/generate")
async def generate_video(
    background_tasks: BackgroundTasks,
    prompt: str = Form(...),
    negative: str = Form(""),
    duration: float = Form(2.0),
    fps: int = Form(8),
    cfg_scale: float = Form(7.5),
    steps: int = Form(50),
    style: str = Form(""),
    camera: str = Form(""),
    output_format: str = Form("mp4"),
    watermark: str = Form(""),
    resolution: str = Form("256p"),
    async_mode: bool = Form(False),
    ar_mode: bool = Form(False),
    use_memory: bool = Form(True),
    distill_mode: bool = Form(False),
    use_raft: bool = Form(False),
    use_routing: bool = Form(False),
    use_compression: bool = Form(False),
    use_pyramid: bool = Form(False),
    use_learned_compressor: bool = Form(False),
    use_concatenated_history: bool = Form(False),
    temporal_smooth: bool = Form(False),
    interpolate: bool = Form(False),
    superres: bool = Form(False),
    use_parallel_tile: bool = Form(False),
    tile_batch_size: int = Form(4),
    use_tensorrt: bool = Form(False),
    tensorrt_engine_path: str = Form("model.trt"),
    physics_correct: bool = Form(False),
    use_pipeline: bool = Form(False),
    bgm_url: str = Form(""),
    bgm_file: Optional[UploadFile] = File(None),
    init_image: Optional[UploadFile] = File(None),
    reference_images: Optional[List[UploadFile]] = File(None),
    reference_videos: Optional[List[UploadFile]] = File(None),
    audio_files: Optional[List[UploadFile]] = File(None),
    lens_script: Optional[UploadFile] = File(None)
):
    """视频生成接口"""
    if not hasattr(app.state, 'inferencer') or app.state.inferencer is None:
        raise HTTPException(status_code=500, detail="Inferencer not initialized")
    
    task_id = str(uuid.uuid4())
    print(f"收到请求 - 任务ID: {task_id}, prompt: {prompt}")
    
    temp_dir = app.state.inferencer.config.api.temp_dir
    os.makedirs(temp_dir, exist_ok=True)
    output_path = os.path.join(temp_dir, f"{task_id}.{output_format}")
    
    # 处理多模态输入（简化版本）
    init_image_np = None
    ref_imgs = None
    ref_vids = None
    audio_paths = None
    lens_path = None
    bgm_file_path = None
    
    # 组装参数
    args = {
        'prompt': prompt,
        'negative_prompt': negative,
        'duration': duration,
        'fps': fps,
        'cfg_scale': cfg_scale,
        'num_steps': steps,
        'style': style if style != "无" else None,
        'camera': camera if camera != "无" else None,
        'output_format': output_format,
        'watermark': watermark,
        'init_image': init_image_np,
        'reference_images': ref_imgs,
        'reference_videos': ref_vids,
        'audio_paths': audio_paths,
        'lens_script_path': lens_path,
        'resolution': resolution,
        'ar_mode': ar_mode,
        'use_memory': use_memory,
        'distill_mode': distill_mode,
        'use_raft': use_raft,
        'use_routing': use_routing,
        'use_compression': use_compression,
        'use_pyramid': use_pyramid,
        'use_learned_compressor': use_learned_compressor,
        'use_concatenated_history': use_concatenated_history,
        'temporal_smooth': temporal_smooth,
        'interpolate': interpolate,
        'superres': superres,
        'use_parallel_tile': use_parallel_tile,
        'tile_batch_size': tile_batch_size,
        'use_tensorrt': use_tensorrt,
        'tensorrt_engine_path': tensorrt_engine_path,
        'physics_correct': physics_correct,
        'use_pipeline': use_pipeline,
        'bgm_url': bgm_url,
        'bgm_file_path': bgm_file_path,
        'output_path': output_path,
        'async_mode': async_mode,
    }
    
    if async_mode:
        if not hasattr(app.state, 'tasks'):
            app.state.tasks = {}
        if not hasattr(app.state, 'results'):
            app.state.results = {}
            
        app.state.tasks[task_id] = {"status": "pending", "progress": 0}
        
        # 提交到任务队列
        from .task_queue import get_global_task_queue
        task_queue = get_global_task_queue()
        task_queue.submit(task_id, run_generation_task, task_id, app.state.inferencer, args, priority=0)
        
        print(f"任务 {task_id}: 已添加到队列")
        return JSONResponse({"task_id": task_id, "status": "pending"})
    else:
        # 同步执行
        run_generation_task(task_id, app.state.inferencer, args)
        if isinstance(app.state.results.get(task_id), dict) and "error" in app.state.results[task_id]:
            raise HTTPException(status_code=500, detail=app.state.results[task_id]["error"])
        return FileResponse(output_path, media_type=f"video/{output_format}")


@app.post("/generate_batch")
async def generate_batch(
    prompts: List[str] = Form(...),
    negative: str = Form(""),
    duration: float = Form(2.0),
    fps: int = Form(8),
    cfg_scale: float = Form(7.5),
    steps: int = Form(50),
    style: str = Form(""),
    camera: str = Form(""),
    output_format: str = Form("mp4"),
    watermark: str = Form(""),
    resolution: str = Form("256p"),
    ar_mode: bool = Form(False),
    use_memory: bool = Form(True),
    distill_mode: bool = Form(False),
    use_raft: bool = Form(False),
    use_routing: bool = Form(False),
    use_compression: bool = Form(False),
    use_pyramid: bool = Form(False),
    use_learned_compressor: bool = Form(False),
    use_concatenated_history: bool = Form(False),
    temporal_smooth: bool = Form(False),
    interpolate: bool = Form(False),
    superres: bool = Form(False),
    use_parallel_tile: bool = Form(False),
    tile_batch_size: int = Form(4),
    use_tensorrt: bool = Form(False),
    tensorrt_engine_path: str = Form("model.trt"),
    physics_correct: bool = Form(False),
    use_pipeline: bool = Form(False),
    bgm_url: str = Form(""),
):
    """批量生成接口"""
    if not hasattr(app.state, 'inferencer') or app.state.inferencer is None:
        raise HTTPException(status_code=500, detail="Inferencer not initialized")
    
    task_ids = []
    temp_dir = app.state.inferencer.config.api.temp_dir
    os.makedirs(temp_dir, exist_ok=True)
    
    for prompt in prompts:
        task_id = str(uuid.uuid4())
        output_path = os.path.join(temp_dir, f"{task_id}.{output_format}")
        args = {
            'prompt': prompt,
            'negative_prompt': negative,
            'duration': duration,
            'fps': fps,
            'cfg_scale': cfg_scale,
            'num_steps': steps,
            'style': style if style != "无" else None,
            'camera': camera if camera != "无" else None,
            'output_format': output_format,
            'watermark': watermark,
            'resolution': resolution,
            'ar_mode': ar_mode,
            'use_memory': use_memory,
            'distill_mode': distill_mode,
            'use_raft': use_raft,
            'use_routing': use_routing,
            'use_compression': use_compression,
            'use_pyramid': use_pyramid,
            'use_learned_compressor': use_learned_compressor,
            'use_concatenated_history': use_concatenated_history,
            'temporal_smooth': temporal_smooth,
            'interpolate': interpolate,
            'superres': superres,
            'use_parallel_tile': use_parallel_tile,
            'tile_batch_size': tile_batch_size,
            'use_tensorrt': use_tensorrt,
            'tensorrt_engine_path': tensorrt_engine_path,
            'physics_correct': physics_correct,
            'use_pipeline': use_pipeline,
            'bgm_url': bgm_url,
            'bgm_file_path': None,
            'init_image': None,
            'reference_images': None,
            'reference_videos': None,
            'audio_paths': None,
            'lens_script_path': None,
            'output_path': output_path,
            'async_mode': True,
        }
        
        if not hasattr(app.state, 'tasks'):
            app.state.tasks = {}
        app.state.tasks[task_id] = {"status": "pending", "progress": 0}
        
        from .task_queue import get_global_task_queue
        task_queue = get_global_task_queue()
        task_queue.submit(task_id, run_generation_task, task_id, app.state.inferencer, args, priority=0)
        task_ids.append(task_id)
    
    return JSONResponse({"task_ids": task_ids, "status": "pending"})


@app.get("/result/{task_id}")
async def get_result(task_id: str):
    """获取生成结果"""
    if hasattr(app.state, 'results') and task_id in app.state.results:
        result = app.state.results.pop(task_id)
        if isinstance(result, dict) and "error" in result:
            return JSONResponse({"status": "error", "detail": result["error"]}, status_code=500)
        else:
            return FileResponse(result, media_type="video/mp4")
    else:
        if hasattr(app.state, 'tasks') and task_id in app.state.tasks:
            return JSONResponse({"status": "pending", "task_id": task_id})
        else:
            return JSONResponse({"status": "not_found"}, status_code=404)


@app.get("/progress/{task_id}")
async def get_progress(task_id: str):
    """获取任务进度"""
    from .task_queue import get_global_task_queue
    task_queue = get_global_task_queue()
    status = task_queue.get_status(task_id)
    
    if status:
        return JSONResponse(status)
    else:
        if hasattr(app.state, 'tasks') and task_id in app.state.tasks:
            return JSONResponse(app.state.tasks[task_id])
        return JSONResponse({"status": "not_found"}, status_code=404)


@app.get("/health")
async def health_check():
    """健康检查"""
    return JSONResponse({
        "status": "healthy",
        "inferencer_ready": hasattr(app.state, 'inferencer') and app.state.inferencer is not None,
        "tasks_count": len(getattr(app.state, 'tasks', {})),
        "results_count": len(getattr(app.state, 'results', {}))
    })


@app.post("/generate_script")
async def generate_script(story: str = Form(..., description="故事创意文本")):
    """根据故事创意生成镜头脚本（JSON格式）"""
    if not story or not story.strip():
        raise HTTPException(status_code=400, detail="故事创意不能为空")
    
    try:
        gen = get_script_generator()
        script = gen.generate(story)
        return JSONResponse(script)
    except Exception as e:
        print(f"脚本生成失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"脚本生成失败: {str(e)}")


# 用于直接运行的简单服务器
if __name__ == "__main__":
    print("API模块已加载")
    print("可用路由:")
    for method, path, func in app.routes:
        print(f"  {method} {path}")