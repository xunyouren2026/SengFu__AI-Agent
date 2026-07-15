"""
Image Search Tool - 以图搜图工具
使用图像搜索相似图像
"""

import json
import base64
import hashlib
import urllib.request
import urllib.error
import urllib.parse
import math
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class SearchEngine(Enum):
    """搜索引擎枚举"""
    GOOGLE = "google"
    BING = "bing"
    BAIDU = "baidu"
    YANDEX = "yandex"
    TINEYE = "tineye"
    CUSTOM = "custom"


class SearchType(Enum):
    """搜索类型枚举"""
    SIMILAR = "similar"
    EXACT = "exact"
    CONTAINS = "contains"
    LARGER = "larger"


@dataclass
class ImageFeature:
    """图像特征"""
    feature_vector: List[float]
    feature_type: str = "embedding"
    dimensions: int = 0
    
    def __post_init__(self):
        self.dimensions = len(self.feature_vector)
    
    def similarity(self, other: "ImageFeature") -> float:
        """计算余弦相似度"""
        if self.dimensions != other.dimensions:
            return 0.0
        
        dot_product = sum(a * b for a, b in zip(self.feature_vector, other.feature_vector))
        norm_a = math.sqrt(sum(a * a for a in self.feature_vector))
        norm_b = math.sqrt(sum(b * b for b in other.feature_vector))
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        return dot_product / (norm_a * norm_b)


@dataclass
class SearchResult:
    """搜索结果"""
    url: str
    title: str
    thumbnail_url: Optional[str] = None
    source_url: Optional[str] = None
    source_name: Optional[str] = None
    width: int = 0
    height: int = 0
    similarity_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "url": self.url,
            "title": self.title,
            "thumbnail_url": self.thumbnail_url,
            "source_url": self.source_url,
            "source_name": self.source_name,
            "width": self.width,
            "height": self.height,
            "similarity_score": self.similarity_score,
            "metadata": self.metadata
        }


@dataclass
class ImageSearchResult:
    """图像搜索结果集"""
    query_image: Optional[str] = None
    results: List[SearchResult] = field(default_factory=list)
    total_results: int = 0
    search_engine: str = ""
    processing_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "query_image": self.query_image,
            "results": [r.to_dict() for r in self.results],
            "total_results": self.total_results,
            "search_engine": self.search_engine,
            "processing_time": self.processing_time,
            "metadata": self.metadata
        }
    
    def filter_by_similarity(self, min_score: float) -> List[SearchResult]:
        """按相似度过滤"""
        return [r for r in self.results if r.similarity_score >= min_score]
    
    def filter_by_size(self, min_width: int = 0, min_height: int = 0) -> List[SearchResult]:
        """按尺寸过滤"""
        return [r for r in self.results if r.width >= min_width and r.height >= min_height]


@dataclass
class ImageSearchConfig:
    """图像搜索配置"""
    engine: SearchEngine = SearchEngine.GOOGLE
    max_results: int = 20
    search_type: SearchType = SearchType.SIMILAR
    safe_search: bool = True
    language: str = "en"
    region: str = "us"
    api_endpoint: Optional[str] = None
    api_key: Optional[str] = None
    timeout: int = 30


class ImageSearchTool:
    """以图搜图工具"""
    
    def __init__(self, config: Optional[ImageSearchConfig] = None):
        self.config = config or ImageSearchConfig()
        self._feature_cache: Dict[str, ImageFeature] = {}
    
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
    
    def compute_hash(self, image_data: bytes) -> str:
        """计算图像哈希"""
        return hashlib.md5(image_data).hexdigest()
    
    def search(self, image: Union[str, bytes],
               max_results: Optional[int] = None,
               search_type: Optional[SearchType] = None,
               **kwargs) -> ImageSearchResult:
        """以图搜图"""
        if isinstance(image, str):
            image_base64 = self.load_image_base64(image)
            image_hash = self.compute_hash(self.load_image(image))
        else:
            image_base64 = self.encode_image(image)
            image_hash = self.compute_hash(image)
        
        max_res = max_results or self.config.max_results
        s_type = search_type or self.config.search_type
        
        result = self._call_api(image_base64, max_res, s_type, **kwargs)
        result.query_image = image_hash
        
        return result
    
    def search_by_url(self, image_url: str,
                      max_results: Optional[int] = None,
                      **kwargs) -> ImageSearchResult:
        """通过URL搜索"""
        max_res = max_results or self.config.max_results
        
        result = self._call_api_url(image_url, max_res, **kwargs)
        result.query_image = image_url
        
        return result
    
    def search_batch(self, images: List[Union[str, bytes]],
                     **kwargs) -> List[ImageSearchResult]:
        """批量搜索"""
        results = []
        for image in images:
            result = self.search(image, **kwargs)
            results.append(result)
        return results
    
    def _call_api(self, image_base64: str,
                  max_results: int,
                  search_type: SearchType,
                  **kwargs) -> ImageSearchResult:
        """调用API"""
        if not self.config.api_endpoint or not self.config.api_key:
            return ImageSearchResult(
                search_engine=self.config.engine.value,
                metadata={"error": "API endpoint or key not configured"}
            )
        
        payload = {
            "image": image_base64,
            "engine": self.config.engine.value,
            "search_type": search_type.value,
            "max_results": max_results,
            "safe_search": self.config.safe_search,
            "language": self.config.language
        }
        
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
            return ImageSearchResult(
                search_engine=self.config.engine.value,
                metadata={"error": str(e)}
            )
    
    def _call_api_url(self, image_url: str,
                      max_results: int,
                      **kwargs) -> ImageSearchResult:
        """通过URL调用API"""
        if not self.config.api_endpoint or not self.config.api_key:
            return ImageSearchResult(
                search_engine=self.config.engine.value,
                metadata={"error": "API endpoint or key not configured"}
            )
        
        payload = {
            "image_url": image_url,
            "engine": self.config.engine.value,
            "max_results": max_results
        }
        
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
            return ImageSearchResult(
                search_engine=self.config.engine.value,
                metadata={"error": str(e)}
            )
    
    def _parse_response(self, result: Dict[str, Any]) -> ImageSearchResult:
        """解析响应"""
        results = []
        
        for item in result.get("results", []):
            results.append(SearchResult(
                url=item.get("url", ""),
                title=item.get("title", ""),
                thumbnail_url=item.get("thumbnail_url"),
                source_url=item.get("source_url"),
                source_name=item.get("source_name"),
                width=item.get("width", 0),
                height=item.get("height", 0),
                similarity_score=item.get("similarity", 0),
                metadata=item.get("metadata", {})
            ))
        
        return ImageSearchResult(
            results=results,
            total_results=result.get("total", len(results)),
            search_engine=result.get("engine", self.config.engine.value),
            processing_time=result.get("processing_time", 0),
            metadata={"raw_response": result}
        )
    
    def get_available_engines(self) -> List[str]:
        """获取可用搜索引擎"""
        return [e.value for e in SearchEngine]
    
    def get_search_types(self) -> List[str]:
        """获取搜索类型"""
        return [t.value for t in SearchType]
    
    def compare_images(self, image1: Union[str, bytes],
                       image2: Union[str, bytes]) -> float:
        """比较两张图片的相似度"""
        # 提取特征
        feature1 = self._extract_feature(image1)
        feature2 = self._extract_feature(image2)
        
        return feature1.similarity(feature2)
    
    def _extract_feature(self, image: Union[str, bytes]) -> ImageFeature:
        """提取图像特征"""
        if isinstance(image, str):
            image_data = self.load_image(image)
        else:
            image_data = image
        
        image_hash = self.compute_hash(image_data)
        
        # 检查缓存
        if image_hash in self._feature_cache:
            return self._feature_cache[image_hash]
        
        # 简单的特征提取（实际需要深度学习模型）
        # 这里使用哈希作为伪特征
        hash_bytes = hashlib.sha256(image_data).digest()
        feature_vector = [float(b) / 255.0 for b in hash_bytes]
        
        feature = ImageFeature(feature_vector=feature_vector)
        self._feature_cache[image_hash] = feature
        
        return feature
    
    def find_duplicates(self, images: List[Union[str, bytes]],
                        threshold: float = 0.9) -> List[Tuple[int, int, float]]:
        """查找重复图片"""
        duplicates = []
        
        for i in range(len(images)):
            for j in range(i + 1, len(images)):
                similarity = self.compare_images(images[i], images[j])
                if similarity >= threshold:
                    duplicates.append((i, j, similarity))
        
        return duplicates
    
    def cluster_images(self, images: List[Union[str, bytes]],
                       threshold: float = 0.8) -> List[List[int]]:
        """聚类相似图片"""
        n = len(images)
        clusters: List[List[int]] = []
        assigned = set()
        
        for i in range(n):
            if i in assigned:
                continue
            
            cluster = [i]
            assigned.add(i)
            
            for j in range(i + 1, n):
                if j in assigned:
                    continue
                
                similarity = self.compare_images(images[i], images[j])
                if similarity >= threshold:
                    cluster.append(j)
                    assigned.add(j)
            
            clusters.append(cluster)
        
        return clusters
