"""
Chart Reader Tool - 图表数据提取工具
从图表图像中提取数据
"""

import json
import base64
import urllib.request
import urllib.error
import re
import math
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class ChartType(Enum):
    """图表类型枚举"""
    LINE = "line"
    BAR = "bar"
    PIE = "pie"
    SCATTER = "scatter"
    AREA = "area"
    HISTOGRAM = "histogram"
    BOX_PLOT = "box_plot"
    RADAR = "radar"
    UNKNOWN = "unknown"


@dataclass
class AxisInfo:
    """坐标轴信息"""
    label: str = ""
    min_value: float = 0.0
    max_value: float = 100.0
    tick_values: List[float] = field(default_factory=list)
    tick_labels: List[str] = field(default_factory=list)
    is_logarithmic: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "label": self.label,
            "min_value": self.min_value,
            "max_value": self.max_value,
            "tick_values": self.tick_values,
            "tick_labels": self.tick_labels,
            "is_logarithmic": self.is_logarithmic
        }


@dataclass
class DataPoint:
    """数据点"""
    x: Union[float, str]
    y: float
    label: Optional[str] = None
    series: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "x": self.x,
            "y": self.y,
            "label": self.label,
            "series": self.series
        }


@dataclass
class DataSeries:
    """数据系列"""
    name: str
    points: List[DataPoint]
    color: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "points": [p.to_dict() for p in self.points],
            "color": self.color
        }


@dataclass
class ChartData:
    """图表数据"""
    chart_type: ChartType
    title: str = ""
    x_axis: Optional[AxisInfo] = None
    y_axis: Optional[AxisInfo] = None
    series: List[DataSeries] = field(default_factory=list)
    legend: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "chart_type": self.chart_type.value,
            "title": self.title,
            "x_axis": self.x_axis.to_dict() if self.x_axis else None,
            "y_axis": self.y_axis.to_dict() if self.y_axis else None,
            "series": [s.to_dict() for s in self.series],
            "legend": self.legend,
            "metadata": self.metadata
        }
    
    def to_dataframe(self) -> Dict[str, List[Any]]:
        """转换为数据框格式"""
        data = {"x": [], "y": [], "series": []}
        
        for series in self.series:
            for point in series.points:
                data["x"].append(point.x)
                data["y"].append(point.y)
                data["series"].append(series.name)
        
        return data
    
    def get_all_points(self) -> List[DataPoint]:
        """获取所有数据点"""
        points = []
        for series in self.series:
            points.extend(series.points)
        return points


@dataclass
class ChartReaderConfig:
    """图表读取配置"""
    detect_type: bool = True
    extract_axes: bool = True
    extract_legend: bool = True
    interpolate_missing: bool = False
    precision: int = 2
    api_endpoint: Optional[str] = None
    api_key: Optional[str] = None
    timeout: int = 60


class ChartReader:
    """图表数据提取工具"""
    
    def __init__(self, config: Optional[ChartReaderConfig] = None):
        self.config = config or ChartReaderConfig()
    
    def load_image(self, image_path: str) -> bytes:
        """加载图像"""
        with open(image_path, 'rb') as f:
            return f.read()
    
    def load_image_base64(self, image_path: str) -> str:
        """加载图像并转换为base64"""
        image_data = self.load_image(image_path)
        return base64.b64encode(image_data).decode('utf-8')
    
    def encode_image(self, image_data: bytes) -> str:
        """编码图像为base64"""
        return base64.b64encode(image_data).decode('utf-8')
    
    def read(self, image: Union[str, bytes],
             chart_type: Optional[ChartType] = None,
             **kwargs) -> ChartData:
        """读取图表数据"""
        if isinstance(image, str):
            image_base64 = self.load_image_base64(image)
        else:
            image_base64 = self.encode_image(image)
        
        result = self._call_model(image_base64, chart_type, **kwargs)
        
        return result
    
    def read_batch(self, images: List[Union[str, bytes]],
                   **kwargs) -> List[ChartData]:
        """批量读取"""
        results = []
        for image in images:
            result = self.read(image, **kwargs)
            results.append(result)
        return results
    
    def _call_model(self, image_base64: str,
                    chart_type: Optional[ChartType],
                    **kwargs) -> ChartData:
        """调用模型"""
        if self.config.api_endpoint and self.config.api_key:
            return self._call_api(image_base64, chart_type)
        
        return ChartData(chart_type=chart_type or ChartType.UNKNOWN)
    
    def _call_api(self, image_base64: str,
                  chart_type: Optional[ChartType]) -> ChartData:
        """调用API"""
        payload = {
            "image": image_base64,
            "detect_type": self.config.detect_type,
            "extract_axes": self.config.extract_axes,
            "extract_legend": self.config.extract_legend,
            "precision": self.config.precision
        }
        
        if chart_type:
            payload["chart_type"] = chart_type.value
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.config.api_key}"
        }
        
        try:
            data = json.dumps(payload).encode()
            request = urllib.request.Request(
                self.config.api_endpoint,
                data=data,
                headers=headers,
                method="POST"
            )
            
            with urllib.request.urlopen(request, timeout=self.config.timeout) as response:
                result = json.loads(response.read())
                
                return self._parse_response(result)
        except Exception as e:
            logger.error(f"API call failed: {e}")
            return ChartData(
                chart_type=ChartType.UNKNOWN,
                metadata={"error": str(e)}
            )
    
    def _parse_response(self, result: Dict[str, Any]) -> ChartData:
        """解析响应"""
        chart_type = ChartType(result.get("chart_type", "unknown"))
        
        x_axis = None
        if "x_axis" in result and result["x_axis"]:
            x_data = result["x_axis"]
            x_axis = AxisInfo(
                label=x_data.get("label", ""),
                min_value=x_data.get("min_value", 0),
                max_value=x_data.get("max_value", 100),
                tick_values=x_data.get("tick_values", []),
                tick_labels=x_data.get("tick_labels", [])
            )
        
        y_axis = None
        if "y_axis" in result and result["y_axis"]:
            y_data = result["y_axis"]
            y_axis = AxisInfo(
                label=y_data.get("label", ""),
                min_value=y_data.get("min_value", 0),
                max_value=y_data.get("max_value", 100),
                tick_values=y_data.get("tick_values", []),
                tick_labels=y_data.get("tick_labels", [])
            )
        
        series = []
        for series_data in result.get("series", []):
            points = []
            for pt_data in series_data.get("points", []):
                points.append(DataPoint(
                    x=pt_data.get("x", 0),
                    y=pt_data.get("y", 0),
                    label=pt_data.get("label"),
                    series=series_data.get("name")
                ))
            
            series.append(DataSeries(
                name=series_data.get("name", ""),
                points=points,
                color=series_data.get("color")
            ))
        
        return ChartData(
            chart_type=chart_type,
            title=result.get("title", ""),
            x_axis=x_axis,
            y_axis=y_axis,
            series=series,
            legend=result.get("legend", []),
            metadata={"raw_response": result}
        )
    
    def detect_chart_type(self, image: Union[str, bytes]) -> ChartType:
        """检测图表类型"""
        result = self.read(image)
        return result.chart_type
    
    def extract_values(self, image: Union[str, bytes]) -> List[Tuple[Any, Any]]:
        """提取数值对"""
        result = self.read(image)
        values = []
        
        for series in result.series:
            for point in series.points:
                values.append((point.x, point.y))
        
        return values
    
    def to_csv(self, chart_data: ChartData) -> str:
        """转换为CSV格式"""
        lines = []
        
        # 标题
        if chart_data.title:
            lines.append(f"# {chart_data.title}")
        
        # 表头
        lines.append("series,x,y")
        
        # 数据
        for series in chart_data.series:
            for point in series.points:
                lines.append(f"{series.name},{point.x},{point.y}")
        
        return "\n".join(lines)
    
    def to_json(self, chart_data: ChartData) -> str:
        """转换为JSON格式"""
        return json.dumps(chart_data.to_dict(), indent=2)
    
    def get_statistics(self, chart_data: ChartData) -> Dict[str, Any]:
        """计算统计信息"""
        all_points = chart_data.get_all_points()
        
        if not all_points:
            return {"count": 0}
        
        y_values = [p.y for p in all_points if isinstance(p.y, (int, float))]
        
        if not y_values:
            return {"count": len(all_points)}
        
        return {
            "count": len(all_points),
            "series_count": len(chart_data.series),
            "y_min": min(y_values),
            "y_max": max(y_values),
            "y_mean": sum(y_values) / len(y_values),
            "y_std": math.sqrt(sum((y - sum(y_values)/len(y_values))**2 for y in y_values) / len(y_values)) if len(y_values) > 1 else 0
        }
    
    def interpolate_points(self, chart_data: ChartData,
                           num_points: int = 100) -> ChartData:
        """插值数据点"""
        new_series = []
        
        for series in chart_data.series:
            if len(series.points) < 2:
                new_series.append(series)
                continue
            
            # 排序点
            sorted_points = sorted(series.points, key=lambda p: p.x if isinstance(p.x, (int, float)) else 0)
            
            # 线性插值
            x_min = sorted_points[0].x
            x_max = sorted_points[-1].x
            
            if not isinstance(x_min, (int, float)) or not isinstance(x_max, (int, float)):
                new_series.append(series)
                continue
            
            new_points = []
            step = (x_max - x_min) / (num_points - 1)
            
            for i in range(num_points):
                x = x_min + i * step
                
                # 找到插值区间
                for j in range(len(sorted_points) - 1):
                    p1 = sorted_points[j]
                    p2 = sorted_points[j + 1]
                    
                    if p1.x <= x <= p2.x:
                        if p2.x == p1.x:
                            y = p1.y
                        else:
                            t = (x - p1.x) / (p2.x - p1.x)
                            y = p1.y + t * (p2.y - p1.y)
                        
                        new_points.append(DataPoint(x=x, y=y, series=series.name))
                        break
            
            new_series.append(DataSeries(name=series.name, points=new_points, color=series.color))
        
        return ChartData(
            chart_type=chart_data.chart_type,
            title=chart_data.title,
            x_axis=chart_data.x_axis,
            y_axis=chart_data.y_axis,
            series=new_series,
            legend=chart_data.legend
        )
