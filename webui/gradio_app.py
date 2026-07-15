"""
AGI统一框架 - Gradio Web界面
实现完整的交互式Web界面，包括模型推理、训练监控、可视化等功能
"""

import gradio as gr
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass, field
import json
import time
import asyncio
import threading
import queue
from collections import deque
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from datetime import datetime
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')


# ==================== 配置类 ====================

@dataclass
class WebUIConfig:
    """Web界面配置"""
    server_name: str = "0.0.0.0"
    server_port: int = 7860
    share: bool = False
    auth_enabled: bool = True
    auth_username: str = "admin"
    auth_password: str = "agi2024"
    max_queue_size: int = 100
    inference_timeout: float = 30.0
    max_batch_size: int = 16
    theme: str = "dark"
    log_history_size: int = 1000
    metrics_history_size: int = 500
    enable_websocket: bool = True
    cache_dir: str = "/tmp/agi_cache"
    model_device: str = "cuda:0" if torch.cuda.is_available() else "cpu"


# ==================== 模拟模型类（用于演示） ====================

class MockAGIModel(nn.Module):
    """模拟AGI模型用于界面演示"""
    
    def __init__(self, hidden_dim: int = 512, num_layers: int = 6):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # 文本编码器
        self.text_encoder = nn.Sequential(
            nn.Linear(768, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        # 视觉编码器
        self.vision_encoder = nn.Sequential(
            nn.Linear(2048, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        # 跨模态融合
        self.fusion_layers = nn.ModuleList([
            nn.MultiheadAttention(hidden_dim, num_heads=8, batch_first=True)
            for _ in range(num_layers)
        ])
        
        # 输出头
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 768)
        )
        
    def forward(self, text_features: torch.Tensor, vision_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """前向传播"""
        # 编码
        text_encoded = self.text_encoder(text_features)
        vision_encoded = self.vision_encoder(vision_features)
        
        # 融合
        fused = text_encoded + vision_encoded
        for fusion_layer in self.fusion_layers:
            fused, _ = fusion_layer(fused, fused, fused)
        
        # 输出
        output = self.output_head(fused)
        
        return {
            "output": output,
            "text_features": text_encoded,
            "vision_features": vision_encoded,
            "fused_features": fused
        }


# ==================== 日志管理器 ====================

class LogManager:
    """日志管理器"""
    
    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self.logs: deque = deque(maxlen=max_history)
        self.log_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        self._lock = threading.Lock()
        
    def log(self, level: str, message: str, source: str = "system"):
        """添加日志"""
        with self._lock:
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "level": level,
                "message": message,
                "source": source
            }
            self.logs.append(log_entry)
            
    def get_logs(self, level_filter: Optional[str] = None, 
                 source_filter: Optional[str] = None,
                 limit: int = 100) -> List[Dict]:
        """获取日志"""
        with self._lock:
            logs = list(self.logs)
            
        if level_filter:
            logs = [l for l in logs if l["level"] == level_filter]
        if source_filter:
            logs = [l for l in logs if l["source"] == source_filter]
            
        return logs[-limit:]
    
    def clear(self):
        """清空日志"""
        with self._lock:
            self.logs.clear()


# ==================== 指标管理器 ====================

class MetricsManager:
    """训练/推理指标管理器"""
    
    def __init__(self, max_history: int = 500):
        self.max_history = max_history
        self.metrics: Dict[str, deque] = {}
        self._lock = threading.Lock()
        
    def record(self, metric_name: str, value: float, step: Optional[int] = None):
        """记录指标"""
        with self._lock:
            if metric_name not in self.metrics:
                self.metrics[metric_name] = deque(maxlen=self.max_history)
            
            entry = {
                "value": value,
                "step": step if step is not None else len(self.metrics[metric_name]),
                "timestamp": time.time()
            }
            self.metrics[metric_name].append(entry)
            
    def get_metric(self, metric_name: str) -> List[Dict]:
        """获取指标历史"""
        with self._lock:
            if metric_name in self.metrics:
                return list(self.metrics[metric_name])
        return []
    
    def get_all_metrics(self) -> Dict[str, List[Dict]]:
        """获取所有指标"""
        with self._lock:
            return {k: list(v) for k, v in self.metrics.items()}
    
    def get_latest(self, metric_name: str) -> Optional[float]:
        """获取最新值"""
        with self._lock:
            if metric_name in self.metrics and self.metrics[metric_name]:
                return self.metrics[metric_name][-1]["value"]
        return None


# ==================== 推理引擎 ====================

class InferenceEngine:
    """推理引擎"""
    
    def __init__(self, model: nn.Module, config: WebUIConfig):
        self.model = model
        self.config = config
        self.device = torch.device(config.model_device)
        self.model.to(self.device)
        self.model.eval()
        
        self.request_queue = queue.Queue(maxsize=config.max_queue_size)
        self.result_cache: Dict[str, Any] = {}
        self.is_running = True
        
        # 启动工作线程
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()
        
    def _worker_loop(self):
        """工作线程循环"""
        while self.is_running:
            try:
                request = self.request_queue.get(timeout=1.0)
                if request is None:
                    continue
                    
                request_id, text_features, vision_features, callback = request
                
                try:
                    # 执行推理
                    with torch.no_grad():
                        text_tensor = torch.tensor(text_features, device=self.device)
                        vision_tensor = torch.tensor(vision_features, device=self.device)
                        
                        start_time = time.time()
                        result = self.model(text_tensor, vision_tensor)
                        inference_time = time.time() - start_time
                        
                    # 缓存结果
                    self.result_cache[request_id] = {
                        "result": {k: v.cpu().numpy() for k, v in result.items()},
                        "inference_time": inference_time,
                        "status": "success"
                    }
                    
                    if callback:
                        callback(self.result_cache[request_id])
                        
                except Exception as e:
                    self.result_cache[request_id] = {
                        "error": str(e),
                        "status": "error"
                    }
                    
            except queue.Empty:
                continue
                
    def submit_request(self, request_id: str, text_features: np.ndarray, 
                       vision_features: np.ndarray, 
                       callback: Optional[Callable] = None) -> bool:
        """提交推理请求"""
        try:
            self.request_queue.put((request_id, text_features, vision_features, callback), timeout=1.0)
            return True
        except queue.Full:
            return False
            
    def get_result(self, request_id: str) -> Optional[Dict]:
        """获取推理结果"""
        return self.result_cache.pop(request_id, None)
    
    def shutdown(self):
        """关闭引擎"""
        self.is_running = False
        if self.worker_thread.is_alive():
            self.worker_thread.join(timeout=5.0)


# ==================== 可视化组件 ====================

class VisualizationComponents:
    """可视化组件集合"""
    
    @staticmethod
    def create_training_curves(metrics: Dict[str, List[Dict]]) -> go.Figure:
        """创建训练曲线图"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=("Loss", "Accuracy", "Learning Rate", "Gradient Norm"),
            vertical_spacing=0.15,
            horizontal_spacing=0.1
        )
        
        colors = px.colors.qualitative.Plotly
        
        # Loss曲线
        if "loss" in metrics:
            steps = [m["step"] for m in metrics["loss"]]
            values = [m["value"] for m in metrics["loss"]]
            fig.add_trace(go.Scatter(x=steps, y=values, mode='lines', 
                                     name='Loss', line=dict(color=colors[0])),
                          row=1, col=1)
        
        # Accuracy曲线
        if "accuracy" in metrics:
            steps = [m["step"] for m in metrics["accuracy"]]
            values = [m["value"] for m in metrics["accuracy"]]
            fig.add_trace(go.Scatter(x=steps, y=values, mode='lines',
                                     name='Accuracy', line=dict(color=colors[1])),
                          row=1, col=2)
        
        # Learning Rate曲线
        if "learning_rate" in metrics:
            steps = [m["step"] for m in metrics["learning_rate"]]
            values = [m["value"] for m in metrics["learning_rate"]]
            fig.add_trace(go.Scatter(x=steps, y=values, mode='lines',
                                     name='LR', line=dict(color=colors[2])),
                          row=2, col=1)
        
        # Gradient Norm曲线
        if "grad_norm" in metrics:
            steps = [m["step"] for m in metrics["grad_norm"]]
            values = [m["value"] for m in metrics["grad_norm"]]
            fig.add_trace(go.Scatter(x=steps, y=values, mode='lines',
                                     name='Grad Norm', line=dict(color=colors[3])),
                          row=2, col=2)
        
        fig.update_layout(height=600, showlegend=True, 
                          template="plotly_dark" if True else "plotly")
        return fig
    
    @staticmethod
    def create_attention_heatmap(attention_weights: np.ndarray, 
                                  title: str = "Attention Weights") -> go.Figure:
        """创建注意力热力图"""
        fig = go.Figure(data=go.Heatmap(
            z=attention_weights,
            colorscale='Viridis',
            showscale=True
        ))
        fig.update_layout(title=title, height=500, width=500,
                          xaxis_title="Key Position",
                          yaxis_title="Query Position")
        return fig
    
    @staticmethod
    def create_feature_distribution(features: np.ndarray, 
                                    title: str = "Feature Distribution") -> go.Figure:
        """创建特征分布图"""
        fig = make_subplots(rows=1, cols=2, 
                           subplot_titles=("Histogram", "Box Plot"))
        
        # 直方图
        fig.add_trace(go.Histogram(x=features.flatten(), nbinsx=50,
                                   name='Distribution'),
                      row=1, col=1)
        
        # 箱线图
        fig.add_trace(go.Box(y=features.flatten(), name='Box'),
                      row=1, col=2)
        
        fig.update_layout(height=400, showlegend=False, title=title)
        return fig
    
    @staticmethod
    def create_embedding_visualization(embeddings: np.ndarray, 
                                        labels: Optional[List[str]] = None,
                                        method: str = "pca") -> go.Figure:
        """创建嵌入可视化（2D/3D）"""
        from sklearn.decomposition import PCA
        from sklearn.manifold import TSNE
        
        # 降维
        if method == "pca":
            reducer = PCA(n_components=min(3, embeddings.shape[1]))
        else:
            reducer = TSNE(n_components=min(3, embeddings.shape[1]), 
                          perplexity=min(30, embeddings.shape[0] - 1))
        
        reduced = reducer.fit_transform(embeddings)
        
        if reduced.shape[1] == 2:
            fig = go.Figure(data=go.Scatter(
                x=reduced[:, 0], y=reduced[:, 1],
                mode='markers+text' if labels else 'markers',
                text=labels if labels else None,
                textposition="top center",
                marker=dict(size=10, opacity=0.8)
            ))
        else:
            fig = go.Figure(data=go.Scatter3d(
                x=reduced[:, 0], y=reduced[:, 1], z=reduced[:, 2],
                mode='markers+text' if labels else 'markers',
                text=labels if labels else None,
                marker=dict(size=5, opacity=0.8)
            ))
        
        fig.update_layout(title=f"Embedding Visualization ({method.upper()})",
                          height=600)
        return fig
    
    @staticmethod
    def create_confusion_matrix(predictions: np.ndarray, 
                                 targets: np.ndarray,
                                 class_names: Optional[List[str]] = None,
                                 title: str = "Confusion Matrix") -> go.Figure:
        """创建混淆矩阵"""
        num_classes = max(predictions.max(), targets.max()) + 1
        cm = np.zeros((num_classes, num_classes), dtype=int)
        
        for p, t in zip(predictions, targets):
            cm[int(t), int(p)] += 1
        
        # 归一化
        cm_normalized = cm.astype('float') / cm.sum(axis=1, keepdims=True)
        
        if class_names is None:
            class_names = [str(i) for i in range(num_classes)]
        
        fig = go.Figure(data=go.Heatmap(
            z=cm_normalized,
            x=class_names,
            y=class_names,
            colorscale='Blues',
            showscale=True,
            text=cm,
            texttemplate="%{text}",
            textfont={"size": 12}
        ))
        
        fig.update_layout(title=title, height=500, width=500,
                          xaxis_title="Predicted",
                          yaxis_title="Actual")
        return fig
    
    @staticmethod
    def create_performance_dashboard(metrics: Dict[str, float]) -> go.Figure:
        """创建性能仪表盘"""
        fig = make_subplots(
            rows=2, cols=3,
            specs=[[{"type": "indicator"}, {"type": "indicator"}, {"type": "indicator"}],
                   [{"type": "pie"}, {"type": "bar"}, {"type": "indicator"}]],
            subplot_titles=["", "", "", "Memory", "Throughput", ""]
        )
        
        # 指标卡片
        if "gpu_utilization" in metrics:
            fig.add_trace(go.Indicator(
                mode="gauge+number",
                value=metrics["gpu_utilization"],
                title={"text": "GPU Utilization"},
                gauge={'axis': {'range': [0, 100]},
                       'bar': {'color': "darkblue"}}
            ), row=1, col=1)
        
        if "cpu_utilization" in metrics:
            fig.add_trace(go.Indicator(
                mode="gauge+number",
                value=metrics["cpu_utilization"],
                title={"text": "CPU Utilization"},
                gauge={'axis': {'range': [0, 100]}}
            ), row=1, col=2)
        
        if "latency_ms" in metrics:
            fig.add_trace(go.Indicator(
                mode="number+delta",
                value=metrics["latency_ms"],
                title={"text": "Latency (ms)"},
                delta={'reference': 100}
            ), row=1, col=3)
        
        # 内存饼图
        if "memory_used" in metrics and "memory_total" in metrics:
            fig.add_trace(go.Pie(
                labels=["Used", "Free"],
                values=[metrics["memory_used"], 
                       metrics["memory_total"] - metrics["memory_used"]]
            ), row=2, col=1)
        
        # 吞吐量条形图
        if "requests_per_sec" in metrics:
            fig.add_trace(go.Bar(
                x=["Current", "Target"],
                y=[metrics["requests_per_sec"], 100]
            ), row=2, col=2)
        
        fig.update_layout(height=600, showlegend=False)
        return fig


# ==================== 主界面类 ====================

class AGIWebUI:
    """AGI统一框架Web界面"""
    
    def __init__(self, config: Optional[WebUIConfig] = None):
        self.config = config or WebUIConfig()
        
        # 初始化组件
        self.log_manager = LogManager(self.config.log_history_size)
        self.metrics_manager = MetricsManager(self.config.metrics_history_size)
        
        # 初始化模型
        self.model = MockAGIModel()
        self.inference_engine = InferenceEngine(self.model, self.config)
        
        # 状态管理
        self.current_session: Dict[str, Any] = {}
        self.training_status = "idle"
        self.inference_count = 0
        
        # 创建界面
        self.app = self._create_app()
        
    def _create_app(self) -> gr.Blocks:
        """创建Gradio应用"""
        
        # 自定义CSS
        custom_css = """
        .dark-theme {
            background-color: #1a1a2e;
            color: #eee;
        }
        .metric-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 10px;
            padding: 15px;
            color: white;
        }
        .status-running {
            color: #00ff00;
            font-weight: bold;
        }
        .status-idle {
            color: #ffcc00;
            font-weight: bold;
        }
        .status-error {
            color: #ff0000;
            font-weight: bold;
        }
        """
        
        with gr.Blocks(css=custom_css, title="AGI Unified Framework") as app:
            
            # ========== 顶部状态栏 ==========
            with gr.Row():
                with gr.Column(scale=1):
                    status_display = gr.Textbox(
                        value="🟢 系统就绪", 
                        label="系统状态",
                        interactive=False
                    )
                with gr.Column(scale=1):
                    model_info = gr.Textbox(
                        value=f"模型: MockAGIModel | 设备: {self.config.model_device}",
                        label="模型信息",
                        interactive=False
                    )
                with gr.Column(scale=1):
                    inference_count_display = gr.Number(
                        value=0,
                        label="推理次数",
                        interactive=False
                    )
            
            # ========== 主标签页 ==========
            with gr.Tabs() as main_tabs:
                
                # ---------- 推理标签页 ----------
                with gr.TabItem("🚀 推理"):
                    with gr.Row():
                        with gr.Column(scale=1):
                            # 输入区域
                            gr.Markdown("### 输入")
                            
                            input_text = gr.Textbox(
                                label="文本输入",
                                placeholder="输入文本进行推理...",
                                lines=3
                            )
                            
                            input_image = gr.Image(
                                label="图像输入",
                                type="numpy"
                            )
                            
                            with gr.Row():
                                inference_btn = gr.Button("执行推理", variant="primary")
                                clear_btn = gr.Button("清空输入")
                        
                        with gr.Column(scale=1):
                            # 输出区域
                            gr.Markdown("### 输出")
                            
                            output_text = gr.Textbox(
                                label="推理结果",
                                lines=5,
                                interactive=False
                            )
                            
                            output_features = gr.JSON(
                                label="特征向量"
                            )
                            
                            inference_time = gr.Number(
                                label="推理耗时 (ms)",
                                interactive=False
                            )
                    
                    # 推理历史
                    with gr.Accordion("推理历史", open=False):
                        inference_history = gr.Dataframe(
                            headers=["时间", "输入", "输出", "耗时"],
                            datatype=["str", "str", "str", "number"],
                            row_count=10,
                            col_count=(4, "fixed"),
                            interactive=False
                        )
                
                # ---------- 训练标签页 ----------
                with gr.TabItem("📊 训练监控"):
                    with gr.Row():
                        with gr.Column(scale=2):
                            # 训练曲线
                            training_curves = gr.Plot(label="训练曲线")
                        
                        with gr.Column(scale=1):
                            # 训练控制
                            gr.Markdown("### 训练控制")
                            
                            with gr.Row():
                                start_train_btn = gr.Button("开始训练", variant="primary")
                                stop_train_btn = gr.Button("停止训练")
                                pause_train_btn = gr.Button("暂停训练")
                            
                            current_epoch = gr.Slider(
                                minimum=0, maximum=100, value=0,
                                label="当前Epoch",
                                interactive=False
                            )
                            
                            current_loss = gr.Number(
                                value=0.0,
                                label="当前Loss",
                                interactive=False
                            )
                            
                            current_lr = gr.Number(
                                value=0.0,
                                label="当前学习率",
                                interactive=False
                            )
                            
                            # 训练配置
                            with gr.Accordion("训练配置", open=False):
                                epochs_input = gr.Slider(
                                    minimum=1, maximum=1000, value=100,
                                    label="总Epochs"
                                )
                                lr_input = gr.Number(
                                    value=0.001,
                                    label="学习率"
                                )
                                batch_size_input = gr.Slider(
                                    minimum=1, maximum=256, value=32,
                                    label="Batch Size"
                                )
                    
                    # 指标表格
                    with gr.Accordion("详细指标", open=False):
                        metrics_table = gr.Dataframe(
                            headers=["指标名", "当前值", "最小值", "最大值", "平均值"],
                            datatype=["str", "number", "number", "number", "number"],
                            row_count=20,
                            interactive=False
                        )
                
                # ---------- 可视化标签页 ----------
                with gr.TabItem("📈 可视化"):
                    with gr.Tabs() as viz_tabs:
                        
                        # 注意力可视化
                        with gr.TabItem("注意力"):
                            attention_plot = gr.Plot(label="注意力热力图")
                            with gr.Row():
                                layer_select = gr.Dropdown(
                                    choices=[f"Layer {i}" for i in range(6)],
                                    value="Layer 0",
                                    label="选择层"
                                )
                                head_select = gr.Dropdown(
                                    choices=[f"Head {i}" for i in range(8)],
                                    value="Head 0",
                                    label="选择注意力头"
                                )
                            refresh_attention_btn = gr.Button("刷新注意力图")
                        
                        # 特征可视化
                        with gr.TabItem("特征"):
                            feature_plot = gr.Plot(label="特征分布")
                            with gr.Row():
                                feature_type = gr.Radio(
                                    choices=["text", "vision", "fused"],
                                    value="fused",
                                    label="特征类型"
                                )
                                viz_method = gr.Radio(
                                    choices=["pca", "tsne"],
                                    value="pca",
                                    label="降维方法"
                                )
                            refresh_feature_btn = gr.Button("刷新特征图")
                        
                        # 嵌入可视化
                        with gr.TabItem("嵌入"):
                            embedding_plot = gr.Plot(label="嵌入空间")
                            embedding_samples = gr.Slider(
                                minimum=10, maximum=1000, value=100,
                                label="采样数量"
                            )
                            refresh_embedding_btn = gr.Button("刷新嵌入图")
                        
                        # 性能仪表盘
                        with gr.TabItem("性能"):
                            performance_plot = gr.Plot(label="性能仪表盘")
                            refresh_performance_btn = gr.Button("刷新性能数据")
                
                # ---------- 数据管理标签页 ----------
                with gr.TabItem("📁 数据管理"):
                    with gr.Row():
                        with gr.Column(scale=1):
                            gr.Markdown("### 数据集信息")
                            
                            dataset_info = gr.JSON(
                                label="数据集统计",
                                value={
                                    "name": "未加载数据集",
                                    "train_samples": 0,
                                    "val_samples": 0,
                                    "test_samples": 0,
                                    "num_classes": 0
                                }
                            )
                            
                            load_dataset_btn = gr.Button("加载数据集")
                            dataset_path = gr.Textbox(
                                label="数据集路径",
                                placeholder="/path/to/dataset"
                            )
                        
                        with gr.Column(scale=1):
                            gr.Markdown("### 数据预览")
                            
                            data_preview = gr.Dataframe(
                                headers=["ID", "输入", "标签"],
                                row_count=10,
                                interactive=False
                            )
                            
                            preview_page = gr.Slider(
                                minimum=0, maximum=100, value=0,
                                label="页码"
                            )
                    
                    # 数据统计图
                    with gr.Row():
                        data_distribution_plot = gr.Plot(label="数据分布")
                        data_statistics_plot = gr.Plot(label="统计信息")
                
                # ---------- 系统日志标签页 ----------
                with gr.TabItem("📝 系统日志"):
                    with gr.Row():
                        log_level_filter = gr.Dropdown(
                            choices=["ALL", "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                            value="ALL",
                            label="日志级别"
                        )
                        log_source_filter = gr.Textbox(
                            label="来源过滤",
                            placeholder="输入来源名称..."
                        )
                        refresh_log_btn = gr.Button("刷新日志")
                        clear_log_btn = gr.Button("清空日志")
                    
                    log_display = gr.Textbox(
                        label="日志输出",
                        lines=20,
                        interactive=False,
                        max_lines=100
                    )
                
                # ---------- 配置标签页 ----------
                with gr.TabItem("⚙️ 配置"):
                    with gr.Row():
                        with gr.Column(scale=1):
                            gr.Markdown("### 模型配置")
                            
                            model_config_json = gr.JSON(
                                label="模型配置",
                                value={
                                    "hidden_dim": 512,
                                    "num_layers": 6,
                                    "num_heads": 8,
                                    "dropout": 0.1
                                }
                            )
                            save_model_config_btn = gr.Button("保存模型配置")
                        
                        with gr.Column(scale=1):
                            gr.Markdown("### 训练配置")
                            
                            training_config_json = gr.JSON(
                                label="训练配置",
                                value={
                                    "optimizer": "adamw",
                                    "lr": 0.001,
                                    "weight_decay": 0.01,
                                    "warmup_steps": 1000,
                                    "max_grad_norm": 1.0
                                }
                            )
                            save_train_config_btn = gr.Button("保存训练配置")
                    
                    with gr.Accordion("高级配置", open=False):
                        advanced_config = gr.Code(
                            label="高级配置 (YAML)",
                            language="yaml",
                            value="""
system:
  device: cuda:0
  precision: fp16
  gradient_checkpointing: true
  
distributed:
  backend: nccl
  world_size: 1
  
logging:
  level: INFO
  save_dir: ./logs
  tensorboard: true
"""
                        )
                        save_advanced_config_btn = gr.Button("保存高级配置")
            
            # ========== 事件处理 ==========
            
            # 推理事件
            def run_inference(text, image):
                """执行推理"""
                start_time = time.time()
                
                # 模拟特征提取
                text_features = np.random.randn(1, 768).astype(np.float32)
                if image is not None:
                    vision_features = np.random.randn(1, 2048).astype(np.float32)
                else:
                    vision_features = np.zeros((1, 2048), dtype=np.float32)
                
                # 执行推理
                request_id = f"req_{int(time.time() * 1000)}"
                self.inference_engine.submit_request(
                    request_id, text_features, vision_features
                )
                
                # 等待结果
                result = None
                for _ in range(30):  # 最多等待3秒
                    result = self.inference_engine.get_result(request_id)
                    if result is not None:
                        break
                    time.sleep(0.1)
                
                inference_time_ms = (time.time() - start_time) * 1000
                self.inference_count += 1
                
                if result and result.get("status") == "success":
                    output = f"推理成功！\n输入文本: {text[:100]}...\n输出维度: {result['result']['output'].shape}"
                    features = {
                        "text_features_shape": list(result['result']['text_features'].shape),
                        "vision_features_shape": list(result['result']['vision_features'].shape),
                        "output_shape": list(result['result']['output'].shape)
                    }
                else:
                    output = "推理失败"
                    features = {}
                
                self.log_manager.log("INFO", f"推理完成: {text[:50]}...", "inference")
                
                return output, features, inference_time_ms, self.inference_count
            
            inference_btn.click(
                run_inference,
                inputs=[input_text, input_image],
                outputs=[output_text, output_features, inference_time, inference_count_display]
            )
            
            clear_btn.click(
                lambda: ("", None, "", {}, 0),
                outputs=[input_text, input_image, output_text, output_features, inference_time]
            )
            
            # 训练事件
            def start_training(epochs, lr, batch_size):
                """开始训练"""
                self.training_status = "running"
                self.log_manager.log("INFO", f"训练开始: epochs={epochs}, lr={lr}, batch_size={batch_size}", "training")
                
                # 模拟训练过程
                for epoch in range(int(epochs)):
                    if self.training_status != "running":
                        break
                    
                    # 模拟指标
                    loss = 1.0 * np.exp(-epoch / 20) + 0.1 * np.random.randn()
                    accuracy = 1 - np.exp(-epoch / 15) + 0.05 * np.random.randn()
                    grad_norm = 1.0 / (1 + epoch / 10) + 0.1 * np.random.randn()
                    
                    self.metrics_manager.record("loss", max(0, loss), epoch)
                    self.metrics_manager.record("accuracy", min(1, max(0, accuracy)), epoch)
                    self.metrics_manager.record("learning_rate", lr * (0.99 ** epoch), epoch)
                    self.metrics_manager.record("grad_norm", max(0, grad_norm), epoch)
                    
                    time.sleep(0.1)  # 模拟训练时间
                
                self.training_status = "idle"
                return self.training_status
            
            start_train_btn.click(
                start_training,
                inputs=[epochs_input, lr_input, batch_size_input],
                outputs=[status_display]
            )
            
            def stop_training():
                """停止训练"""
                self.training_status = "stopped"
                self.log_manager.log("INFO", "训练已停止", "training")
                return "🔴 训练已停止"
            
            stop_train_btn.click(
                stop_training,
                outputs=[status_display]
            )
            
            # 可视化事件
            def update_training_curves():
                """更新训练曲线"""
                metrics = self.metrics_manager.get_all_metrics()
                return VisualizationComponents.create_training_curves(metrics)
            
            refresh_attention_btn.click(
                lambda: VisualizationComponents.create_attention_heatmap(
                    np.random.rand(10, 10)
                ),
                outputs=[attention_plot]
            )
            
            refresh_feature_btn.click(
                lambda: VisualizationComponents.create_feature_distribution(
                    np.random.randn(100, 512)
                ),
                outputs=[feature_plot]
            )
            
            refresh_embedding_btn.click(
                lambda: VisualizationComponents.create_embedding_visualization(
                    np.random.randn(100, 512)
                ),
                outputs=[embedding_plot]
            )
            
            refresh_performance_btn.click(
                lambda: VisualizationComponents.create_performance_dashboard({
                    "gpu_utilization": np.random.rand() * 100,
                    "cpu_utilization": np.random.rand() * 100,
                    "latency_ms": np.random.rand() * 50 + 10,
                    "memory_used": np.random.rand() * 8,
                    "memory_total": 16,
                    "requests_per_sec": np.random.rand() * 100 + 50
                }),
                outputs=[performance_plot]
            )
            
            # 日志事件
            def update_logs(level_filter, source_filter):
                """更新日志显示"""
                logs = self.log_manager.get_logs(
                    level_filter=None if level_filter == "ALL" else level_filter,
                    source_filter=source_filter if source_filter else None
                )
                
                log_text = ""
                for log in logs[-100:]:
                    log_text += f"[{log['timestamp']}] [{log['level']}] [{log['source']}] {log['message']}\n"
                
                return log_text
            
            refresh_log_btn.click(
                update_logs,
                inputs=[log_level_filter, log_source_filter],
                outputs=[log_display]
            )
            
            clear_log_btn.click(
                lambda: (self.log_manager.clear(), "")[1],
                outputs=[log_display]
            )
            
            # 定时更新
            app.load(
                update_training_curves,
                outputs=[training_curves]
            )
        
        return app
    
    def launch(self):
        """启动Web界面"""
        self.log_manager.log("INFO", f"Web界面启动: {self.config.server_name}:{self.config.server_port}", "system")
        
        auth = None
        if self.config.auth_enabled:
            auth = (self.config.auth_username, self.config.auth_password)
        
        self.app.launch(
            server_name=self.config.server_name,
            server_port=self.config.server_port,
            share=self.config.share,
            auth=auth
        )
    
    def shutdown(self):
        """关闭Web界面"""
        self.inference_engine.shutdown()
        self.log_manager.log("INFO", "Web界面已关闭", "system")


# ==================== 命令行入口 ====================

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="AGI统一框架Web界面")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="服务器地址")
    parser.add_argument("--port", type=int, default=7860, help="服务器端口")
    parser.add_argument("--share", action="store_true", help="创建公共链接")
    parser.add_argument("--no-auth", action="store_true", help="禁用认证")
    parser.add_argument("--username", type=str, default="admin", help="用户名")
    parser.add_argument("--password", type=str, default="agi2024", help="密码")
    
    args = parser.parse_args()
    
    config = WebUIConfig(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        auth_enabled=not args.no_auth,
        auth_username=args.username,
        auth_password=args.password
    )
    
    webui = AGIWebUI(config)
    
    try:
        webui.launch()
    except KeyboardInterrupt:
        webui.shutdown()


if __name__ == "__main__":
    main()
