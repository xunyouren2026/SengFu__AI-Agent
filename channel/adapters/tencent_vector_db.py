"""
腾讯云向量数据库适配器模块

该模块提供腾讯云向量数据库(Tencent Vector DB)的通道适配器实现，支持：
- Collection管理（创建、删除、列表）
- 向量插入、更新、删除
- 相似度搜索（ANN搜索）
- 索引管理（HNSW、IVF等）
- 批量操作
- 元数据过滤
- RAG应用支持

API文档: https://cloud.tencent.com/document/product/1709
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import json
import logging
import struct
import time
import urllib.parse
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import aiohttp
from aiohttp import ClientTimeout

from ...base import (
    ChannelAdapter,
    ChannelCapability,
    ChannelConfig,
    ConnectionState,
    MessagePriority,
    ReceiveResult,
    RetryConfig,
    SendResult,
)
from ...universal_message import (
    Attachment,
    AttachmentType,
    ChannelIdentity,
    MessageContent,
    MessageDirection,
    MessageMetadata,
    MessageType,
    UniversalMessage,
    UserIdentity,
)

logger = logging.getLogger(__name__)


class IndexType(Enum):
    """索引类型"""
    HNSW = "HNSW"  # 分层导航小世界图
    IVF = "IVF"    # 倒排文件
    FLAT = "FLAT"  # 暴力搜索
    IVF_PQ = "IVF_PQ"  # 倒排文件+乘积量化


class MetricType(Enum):
    """距离度量类型"""
    COSINE = "COSINE"      # 余弦相似度
    L2 = "L2"              # 欧氏距离
    IP = "IP"              # 内积


class FieldType(Enum):
    """字段类型"""
    STRING = "string"
    INT64 = "int64"
    FLOAT = "float"
    BOOL = "bool"
    ARRAY = "array"
    JSON = "json"


@dataclass
class VectorDBConfig(ChannelConfig):
    """腾讯云向量数据库配置类
    
    Attributes:
        secret_id: 腾讯云SecretId
        secret_key: 腾讯云SecretKey
        endpoint: API端点，如 tdb.tencentcloudapi.com
        region: 地域ID，如 ap-guangzhou
        connect_timeout: 连接超时时间（秒）
        read_timeout: 读取超时时间（秒）
        max_retry: 最大重试次数
        default_collection: 默认Collection名称
        default_dimension: 默认向量维度
        default_index_type: 默认索引类型
        default_metric_type: 默认距离度量类型
    """
    secret_id: str = ""
    secret_key: str = ""
    endpoint: str = "tdb.tencentcloudapi.com"
    region: str = "ap-guangzhou"
    connect_timeout: int = 30
    read_timeout: int = 60
    max_retry: int = 3
    default_collection: Optional[str] = None
    default_dimension: int = 768
    default_index_type: str = "HNSW"
    default_metric_type: str = "COSINE"
    
    def __post_init__(self):
        if not self.secret_id or not self.secret_key:
            raise VectorDBConfigError("SecretId和SecretKey不能为空")


class VectorDBConfigError(Exception):
    """向量数据库配置错误"""
    pass


class VectorDBAPIError(Exception):
    """向量数据库API错误
    
    Attributes:
        code: 错误代码
        message: 错误信息
        request_id: 请求ID
    """
    
    def __init__(
        self,
        code: str,
        message: str,
        request_id: Optional[str] = None,
    ):
        self.code = code
        self.message = message
        self.request_id = request_id
        super().__init__(f"[{code}] {message} (RequestId: {request_id})")
    
    @property
    def is_retryable(self) -> bool:
        """判断错误是否可重试"""
        retryable_codes = [
            "InternalError",
            "RequestTimeout",
            "ServiceUnavailable",
        ]
        return self.code in retryable_codes


@dataclass
class VectorDocument:
    """向量文档类
    
    Attributes:
        id: 文档唯一标识
        vector: 向量数据
        metadata: 元数据
        score: 相似度分数（搜索时填充）
    """
    id: str
    vector: List[float]
    metadata: Dict[str, Any] = field(default_factory=dict)
    score: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "id": self.id,
            "vector": self.vector,
            "metadata": self.metadata,
            "score": self.score,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VectorDocument":
        """从字典创建"""
        return cls(
            id=data.get("id", ""),
            vector=data.get("vector", []),
            metadata=data.get("metadata", {}),
            score=data.get("score"),
        )


@dataclass
class CollectionSchema:
    """Collection模式定义
    
    Attributes:
        name: Collection名称
        dimension: 向量维度
        metric_type: 距离度量类型
        index_type: 索引类型
        fields: 元数据字段定义
        description: 描述
    """
    name: str
    dimension: int = 768
    metric_type: MetricType = MetricType.COSINE
    index_type: IndexType = IndexType.HNSW
    fields: List[Dict[str, Any]] = field(default_factory=list)
    description: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "name": self.name,
            "dimension": self.dimension,
            "metricType": self.metric_type.value,
            "indexType": self.index_type.value,
            "fields": self.fields,
            "description": self.description,
        }


@dataclass
class SearchResult:
    """搜索结果类
    
    Attributes:
        documents: 匹配的文档列表
        total: 总匹配数
        search_time: 搜索耗时（毫秒）
    """
    documents: List[VectorDocument]
    total: int = 0
    search_time: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "documents": [doc.to_dict() for doc in self.documents],
            "total": self.total,
            "search_time": self.search_time,
        }


@dataclass
class IndexConfig:
    """索引配置类
    
    Attributes:
        index_type: 索引类型
        params: 索引参数
    """
    index_type: IndexType
    params: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        result = {"indexType": self.index_type.value}
        if self.params:
            result["params"] = self.params
        return result
    
    @classmethod
    def hnsw(cls, m: int = 16, ef_construction: int = 200) -> "IndexConfig":
        """创建HNSW索引配置"""
        return cls(
            index_type=IndexType.HNSW,
            params={"M": m, "efConstruction": ef_construction},
        )
    
    @classmethod
    def ivf(cls, nlist: int = 100) -> "IndexConfig":
        """创建IVF索引配置"""
        return cls(
            index_type=IndexType.IVF,
            params={"nlist": nlist},
        )
    
    @classmethod
    def flat(cls) -> "IndexConfig":
        """创建FLAT索引配置"""
        return cls(index_type=IndexType.FLAT, params={})


class TencentVectorDBAdapter(ChannelAdapter):
    """腾讯云向量数据库适配器
    
    提供腾讯云向量数据库的统一接口，支持向量存储、相似度搜索、
    RAG应用等功能。
    
    Example:
        config = VectorDBConfig(
            secret_id="your-secret-id",
            secret_key="your-secret-key",
            endpoint="tdb.tencentcloudapi.com",
            region="ap-guangzhou",
        )
        adapter = TencentVectorDBAdapter(config)
        await adapter.connect()
        
        # 创建Collection
        await adapter.create_collection("my_collection", dimension=768)
        
        # 插入向量
        doc = VectorDocument(
            id="doc1",
            vector=[0.1, 0.2, 0.3, ...],
            metadata={"title": "示例文档"},
        )
        await adapter.upsert_documents("my_collection", [doc])
        
        # 相似度搜索
        results = await adapter.search(
            "my_collection",
            query_vector=[0.1, 0.2, 0.3, ...],
            top_k=10,
        )
    """
    
    SERVICE = "tdb"
    VERSION = "2023-11-30"
    
    def __init__(self, config: VectorDBConfig) -> None:
        super().__init__(config)
        self._cfg = config
        self._session: Optional[aiohttp.ClientSession] = None
        self._collection_cache: Dict[str, Any] = {}
    
    def _initialize_capabilities(self) -> None:
        """初始化适配器能力"""
        self._capabilities = {
            ChannelCapability.CHANNEL_INFO,
        }
    
    def _get_base_url(self) -> str:
        """获取基础URL"""
        return f"https://{self._cfg.endpoint}"
    
    def _sign_request(
        self,
        method: str,
        url_path: str,
        headers: Dict[str, str],
        payload: str = "",
    ) -> Dict[str, str]:
        """生成请求签名（腾讯云API V3签名）
        
        Args:
            method: HTTP方法
            url_path: URL路径
            headers: 请求头
            payload: 请求体
            
        Returns:
            包含签名的请求头
        """
        # 时间戳
        timestamp = str(int(time.time()))
        date = datetime.utcnow().strftime("%Y-%m-%d")
        
        # 构建规范请求
        http_method = method.upper()
        canonical_uri = url_path
        canonical_querystring = ""
        
        # 规范头部
        signed_headers = "content-type;host;x-tc-action;x-tc-timestamp;x-tc-version"
        canonical_headers = f"content-type:{headers.get('Content-Type', 'application/json')}\n"
        canonical_headers += f"host:{headers['Host']}\n"
        canonical_headers += f"x-tc-action:{headers['X-TC-Action']}\n"
        canonical_headers += f"x-tc-timestamp:{headers['X-TC-Timestamp']}\n"
        canonical_headers += f"x-tc-version:{headers['X-TC-Version']}\n"
        
        # 请求体哈希
        payload_hash = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        
        canonical_request = f"{http_method}\n{canonical_uri}\n{canonical_querystring}\n{canonical_headers}\n{signed_headers}\n{payload_hash}"
        
        # 构建待签名字符串
        credential_scope = f"{date}/{self.SERVICE}/tc3_request"
        string_to_sign = f"TC3-HMAC-SHA256\n{timestamp}\n{credential_scope}\n{hashlib.sha256(canonical_request.encode('utf-8')).hexdigest()}"
        
        # 计算签名
        secret_date = hmac.new(
            f"TC3{self._cfg.secret_key}".encode("utf-8"),
            date.encode("utf-8"),
            hashlib.sha256,
        ).digest()
        secret_service = hmac.new(secret_date, self.SERVICE.encode("utf-8"), hashlib.sha256).digest()
        secret_signing = hmac.new(secret_service, "tc3_request".encode("utf-8"), hashlib.sha256).digest()
        signature = hmac.new(secret_signing, string_to_sign.encode("utf-8"), hashlib.sha256).hexdigest()
        
        # 构建Authorization头
        authorization = f"TC3-HMAC-SHA256 Credential={self._cfg.secret_id}/{credential_scope}, SignedHeaders={signed_headers}, Signature={signature}"
        
        headers["Authorization"] = authorization
        return headers
    
    def _get_auth_headers(
        self,
        action: str,
        payload: str = "",
    ) -> Dict[str, str]:
        """获取认证头
        
        Args:
            action: API动作名称
            payload: 请求体
            
        Returns:
            包含认证信息的请求头
        """
        headers = {
            "Content-Type": "application/json",
            "Host": self._cfg.endpoint,
            "X-TC-Action": action,
            "X-TC-Version": self.VERSION,
            "X-TC-Timestamp": str(int(time.time())),
            "X-TC-Region": self._cfg.region,
        }
        
        return self._sign_request("POST", "/", headers, payload)
    
    async def _make_request(
        self,
        action: str,
        params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """发送API请求
        
        Args:
            action: API动作名称
            params: 请求参数
            
        Returns:
            API响应
        """
        if not self._session:
            raise VectorDBAPIError("NotConnected", "适配器未连接")
        
        payload = json.dumps(params)
        headers = self._get_auth_headers(action, payload)
        
        url = self._get_base_url()
        timeout = ClientTimeout(
            connect=self._cfg.connect_timeout,
            total=self._cfg.read_timeout,
        )
        
        try:
            async with self._session.post(
                url=url,
                headers=headers,
                data=payload,
                timeout=timeout,
            ) as response:
                body = await response.read()
                result = json.loads(body.decode("utf-8"))
                
                # 检查错误
                if "Response" in result:
                    response_data = result["Response"]
                    if "Error" in response_data:
                        error = response_data["Error"]
                        raise VectorDBAPIError(
                            error.get("Code", "UnknownError"),
                            error.get("Message", "Unknown error"),
                            response_data.get("RequestId"),
                        )
                    return response_data
                
                return result
                
        except aiohttp.ClientError as e:
            raise VectorDBAPIError("RequestError", str(e))
        except json.JSONDecodeError as e:
            raise VectorDBAPIError("ParseError", f"JSON解析失败: {e}")
    
    async def _connect_impl(self) -> bool:
        """实现连接逻辑"""
        try:
            timeout = ClientTimeout(
                connect=self._cfg.connect_timeout,
                total=self._cfg.read_timeout,
            )
            self._session = aiohttp.ClientSession(timeout=timeout)
            
            # 测试连接：列出Collections
            await self.list_collections()
            
            self._logger.info(f"已连接到腾讯云向量数据库: {self._cfg.endpoint}")
            return True
        except Exception as e:
            self._logger.error(f"连接腾讯云向量数据库失败: {e}")
            return False
    
    async def _disconnect_impl(self) -> None:
        """实现断开连接逻辑"""
        if self._session:
            await self._session.close()
            self._session = None
        self._collection_cache.clear()
        self._logger.info("已断开与腾讯云向量数据库的连接")
    
    async def _health_check_impl(self) -> bool:
        """实现健康检查逻辑"""
        try:
            await self.list_collections()
            return True
        except Exception:
            return False
    
    async def _send_impl(self, message: UniversalMessage, priority: MessagePriority) -> SendResult:
        """实现发送逻辑（用于插入向量）"""
        try:
            collection = message.get_context("collection") or self._cfg.default_collection
            operation = message.get_context("operation", "upsert")
            
            if not collection:
                return SendResult(
                    success=False,
                    error="缺少collection参数",
                    error_code="MISSING_PARAMS",
                )
            
            if operation == "upsert":
                # 从消息中提取向量文档
                documents = message.get_context("documents", [])
                if not documents:
                    # 尝试从消息内容创建文档
                    vector = message.get_context("vector")
                    doc_id = message.get_context("doc_id") or str(time.time())
                    metadata = message.get_context("metadata", {})
                    
                    if vector:
                        doc = VectorDocument(id=doc_id, vector=vector, metadata=metadata)
                        documents = [doc]
                
                if documents:
                    await self.upsert_documents(collection, documents)
                
                return SendResult(
                    success=True,
                    message_id=f"{collection}/{doc_id}",
                    timestamp=time.time(),
                )
            elif operation == "delete":
                doc_ids = message.get_context("doc_ids", [])
                if doc_ids:
                    await self.delete_documents(collection, doc_ids)
                return SendResult(success=True, timestamp=time.time())
            else:
                return SendResult(
                    success=False,
                    error=f"不支持的操作: {operation}",
                    error_code="UNSUPPORTED_OPERATION",
                )
                
        except VectorDBAPIError as e:
            return SendResult(
                success=False,
                error=e.message,
                error_code=e.code,
            )
        except Exception as e:
            return SendResult(
                success=False,
                error=str(e),
                error_code="UNKNOWN_ERROR",
            )
    
    async def _receive_impl(self, payload: Optional[Dict]) -> ReceiveResult:
        """实现接收逻辑（用于搜索向量）"""
        try:
            collection = payload.get("collection") if payload else self._cfg.default_collection
            operation = payload.get("operation", "search") if payload else "search"
            
            if not collection:
                return ReceiveResult(success=False, error="缺少collection参数")
            
            messages = []
            
            if operation == "search":
                query_vector = payload.get("query_vector") if payload else None
                top_k = payload.get("top_k", 10) if payload else 10
                
                if query_vector:
                    results = await self.search(collection, query_vector, top_k=top_k)
                    for doc in results.documents:
                        content = MessageContent(
                            text=f"[Vector Doc: {doc.id}, Score: {doc.score}]"
                        )
                        metadata = MessageMetadata(
                            message_id=doc.id,
                            channel_id="tencent_vector_db",
                            timestamp=time.time(),
                            direction=MessageDirection.INBOUND,
                            message_type=MessageType.DATA,
                        )
                        msg = UniversalMessage(content=content, metadata=metadata)
                        msg.set_context("document", doc.to_dict())
                        messages.append(msg)
            elif operation == "get":
                doc_ids = payload.get("doc_ids", []) if payload else []
                if doc_ids:
                    docs = await self.get_documents(collection, doc_ids)
                    for doc in docs:
                        content = MessageContent(text=f"[Vector Doc: {doc.id}]")
                        metadata = MessageMetadata(
                            message_id=doc.id,
                            channel_id="tencent_vector_db",
                            timestamp=time.time(),
                            direction=MessageDirection.INBOUND,
                            message_type=MessageType.DATA,
                        )
                        msg = UniversalMessage(content=content, metadata=metadata)
                        msg.set_context("document", doc.to_dict())
                        messages.append(msg)
            
            return ReceiveResult(success=True, messages=messages)
        except Exception as e:
            return ReceiveResult(success=False, error=str(e))
    
    # ========== Collection管理 ==========
    
    async def list_collections(self) -> List[str]:
        """列出所有Collection
        
        Returns:
            Collection名称列表
        """
        result = await self._make_request("DescribeCollections", {})
        collections = result.get("Collections", [])
        return [c.get("Name", "") for c in collections]
    
    async def create_collection(
        self,
        name: str,
        dimension: Optional[int] = None,
        metric_type: Optional[MetricType] = None,
        index_config: Optional[IndexConfig] = None,
        description: str = "",
        fields: Optional[List[Dict[str, Any]]] = None,
    ) -> bool:
        """创建Collection
        
        Args:
            name: Collection名称
            dimension: 向量维度
            metric_type: 距离度量类型
            index_config: 索引配置
            description: 描述
            fields: 元数据字段定义
            
        Returns:
            是否创建成功
        """
        dimension = dimension or self._cfg.default_dimension
        metric_type = metric_type or MetricType(self._cfg.default_metric_type)
        index_config = index_config or IndexConfig.hnsw()
        
        params = {
            "Name": name,
            "Dimension": dimension,
            "MetricType": metric_type.value,
            "IndexType": index_config.index_type.value,
            "Description": description,
        }
        
        if index_config.params:
            params["IndexParams"] = index_config.params
        
        if fields:
            params["Fields"] = fields
        
        await self._make_request("CreateCollection", params)
        return True
    
    async def delete_collection(self, name: str) -> bool:
        """删除Collection
        
        Args:
            name: Collection名称
            
        Returns:
            是否删除成功
        """
        await self._make_request("DeleteCollection", {"Name": name})
        return True
    
    async def get_collection_info(self, name: str) -> Dict[str, Any]:
        """获取Collection信息
        
        Args:
            name: Collection名称
            
        Returns:
            Collection信息
        """
        result = await self._make_request("DescribeCollection", {"Name": name})
        return result.get("Collection", {})
    
    async def collection_exists(self, name: str) -> bool:
        """检查Collection是否存在
        
        Args:
            name: Collection名称
            
        Returns:
            是否存在
        """
        try:
            await self.get_collection_info(name)
            return True
        except VectorDBAPIError as e:
            if e.code == "ResourceNotFound":
                return False
            raise
    
    # ========== 向量操作 ==========
    
    async def upsert_documents(
        self,
        collection: str,
        documents: List[VectorDocument],
    ) -> Dict[str, Any]:
        """插入或更新向量文档
        
        Args:
            collection: Collection名称
            documents: 文档列表
            
        Returns:
            操作结果
        """
        items = []
        for doc in documents:
            item = {
                "Id": doc.id,
                "Vector": doc.vector,
            }
            if doc.metadata:
                item["Metadata"] = doc.metadata
            items.append(item)
        
        params = {
            "Collection": collection,
            "Documents": items,
        }
        
        result = await self._make_request("UpsertDocuments", params)
        return {
            "success": True,
            "inserted": result.get("InsertedCount", 0),
            "updated": result.get("UpdatedCount", 0),
        }
    
    async def get_documents(
        self,
        collection: str,
        doc_ids: List[str],
    ) -> List[VectorDocument]:
        """获取指定ID的文档
        
        Args:
            collection: Collection名称
            doc_ids: 文档ID列表
            
        Returns:
            文档列表
        """
        params = {
            "Collection": collection,
            "Ids": doc_ids,
        }
        
        result = await self._make_request("GetDocuments", params)
        documents = result.get("Documents", [])
        
        return [
            VectorDocument(
                id=d.get("Id", ""),
                vector=d.get("Vector", []),
                metadata=d.get("Metadata", {}),
            )
            for d in documents
        ]
    
    async def delete_documents(
        self,
        collection: str,
        doc_ids: List[str],
    ) -> int:
        """删除文档
        
        Args:
            collection: Collection名称
            doc_ids: 文档ID列表
            
        Returns:
            删除数量
        """
        params = {
            "Collection": collection,
            "Ids": doc_ids,
        }
        
        result = await self._make_request("DeleteDocuments", params)
        return result.get("DeletedCount", 0)
    
    async def search(
        self,
        collection: str,
        query_vector: List[float],
        top_k: int = 10,
        filter_expr: Optional[str] = None,
        output_fields: Optional[List[str]] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> SearchResult:
        """相似度搜索
        
        Args:
            collection: Collection名称
            query_vector: 查询向量
            top_k: 返回结果数量
            filter_expr: 过滤表达式
            output_fields: 返回的字段列表
            params: 搜索参数
            
        Returns:
            搜索结果
        """
        search_params = {
            "Collection": collection,
            "Vector": query_vector,
            "TopK": top_k,
        }
        
        if filter_expr:
            search_params["Filter"] = filter_expr
        
        if output_fields:
            search_params["OutputFields"] = output_fields
        
        if params:
            search_params["Params"] = params
        
        start_time = time.time()
        result = await self._make_request("Search", search_params)
        search_time = (time.time() - start_time) * 1000
        
        documents = []
        for item in result.get("Results", []):
            doc = VectorDocument(
                id=item.get("Id", ""),
                vector=item.get("Vector", []),
                metadata=item.get("Metadata", {}),
                score=item.get("Score"),
            )
            documents.append(doc)
        
        return SearchResult(
            documents=documents,
            total=result.get("Total", 0),
            search_time=search_time,
        )
    
    async def batch_search(
        self,
        collection: str,
        query_vectors: List[List[float]],
        top_k: int = 10,
        filter_expr: Optional[str] = None,
    ) -> List[SearchResult]:
        """批量相似度搜索
        
        Args:
            collection: Collection名称
            query_vectors: 查询向量列表
            top_k: 每个查询返回结果数量
            filter_expr: 过滤表达式
            
        Returns:
            搜索结果列表
        """
        params = {
            "Collection": collection,
            "Vectors": query_vectors,
            "TopK": top_k,
        }
        
        if filter_expr:
            params["Filter"] = filter_expr
        
        result = await self._make_request("BatchSearch", params)
        
        results = []
        for batch in result.get("Results", []):
            documents = []
            for item in batch.get("Documents", []):
                doc = VectorDocument(
                    id=item.get("Id", ""),
                    vector=item.get("Vector", []),
                    metadata=item.get("Metadata", {}),
                    score=item.get("Score"),
                )
                documents.append(doc)
            
            results.append(SearchResult(
                documents=documents,
                total=batch.get("Total", 0),
            ))
        
        return results
    
    async def hybrid_search(
        self,
        collection: str,
        query_vector: List[float],
        query_text: str,
        top_k: int = 10,
        vector_weight: float = 0.7,
        text_weight: float = 0.3,
    ) -> SearchResult:
        """混合搜索（向量+文本）
        
        Args:
            collection: Collection名称
            query_vector: 查询向量
            query_text: 查询文本
            top_k: 返回结果数量
            vector_weight: 向量搜索权重
            text_weight: 文本搜索权重
            
        Returns:
            搜索结果
        """
        params = {
            "Collection": collection,
            "Vector": query_vector,
            "QueryText": query_text,
            "TopK": top_k,
            "VectorWeight": vector_weight,
            "TextWeight": text_weight,
        }
        
        result = await self._make_request("HybridSearch", params)
        
        documents = []
        for item in result.get("Results", []):
            doc = VectorDocument(
                id=item.get("Id", ""),
                vector=item.get("Vector", []),
                metadata=item.get("Metadata", {}),
                score=item.get("Score"),
            )
            documents.append(doc)
        
        return SearchResult(
            documents=documents,
            total=result.get("Total", 0),
        )
    
    # ========== 索引管理 ==========
    
    async def rebuild_index(self, collection: str) -> bool:
        """重建索引
        
        Args:
            collection: Collection名称
            
        Returns:
            是否成功
        """
        await self._make_request("RebuildIndex", {"Collection": collection})
        return True
    
    async def get_index_progress(self, collection: str) -> Dict[str, Any]:
        """获取索引构建进度
        
        Args:
            collection: Collection名称
            
        Returns:
            索引进度信息
        """
        result = await self._make_request("DescribeIndexProgress", {"Collection": collection})
        return {
            "status": result.get("Status", ""),
            "progress": result.get("Progress", 0),
            "total": result.get("Total", 0),
            "indexed": result.get("Indexed", 0),
        }
    
    # ========== RAG应用支持 ==========
    
    async def rag_retrieve(
        self,
        collection: str,
        query: str,
        embedding_func,
        top_k: int = 5,
        min_score: float = 0.7,
        filter_expr: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """RAG检索
        
        将查询文本转换为向量并执行相似度搜索
        
        Args:
            collection: Collection名称
            query: 查询文本
            embedding_func: 文本向量化函数
            top_k: 返回结果数量
            min_score: 最小相似度分数
            filter_expr: 过滤表达式
            
        Returns:
            检索结果列表
        """
        # 将查询文本转换为向量
        query_vector = await embedding_func(query)
        
        # 执行向量搜索
        results = await self.search(
            collection=collection,
            query_vector=query_vector,
            top_k=top_k,
            filter_expr=filter_expr,
        )
        
        # 过滤并格式化结果
        retrieved = []
        for doc in results.documents:
            if doc.score and doc.score >= min_score:
                retrieved.append({
                    "id": doc.id,
                    "content": doc.metadata.get("content", ""),
                    "metadata": doc.metadata,
                    "score": doc.score,
                })
        
        return retrieved
    
    async def rag_augment_query(
        self,
        collection: str,
        query: str,
        embedding_func,
        top_k: int = 3,
        template: Optional[str] = None,
    ) -> str:
        """RAG增强查询
        
        检索相关文档并增强原始查询
        
        Args:
            collection: Collection名称
            query: 原始查询
            embedding_func: 文本向量化函数
            top_k: 检索文档数量
            template: 提示词模板
            
        Returns:
            增强后的查询
        """
        # 检索相关文档
        docs = await self.rag_retrieve(collection, query, embedding_func, top_k=top_k)
        
        # 构建上下文
        context = "\n\n".join([
            f"[Document {i+1}]\n{doc['content']}"
            for i, doc in enumerate(docs)
        ])
        
        # 使用模板或默认格式
        if template:
            augmented = template.format(context=context, query=query)
        else:
            augmented = f"""基于以下参考信息回答问题：

参考信息：
{context}

问题：{query}

请根据参考信息回答问题，如果参考信息不足，请说明。"""
        
        return augmented
    
    # ========== 统计与监控 ==========
    
    async def get_collection_stats(self, collection: str) -> Dict[str, Any]:
        """获取Collection统计信息
        
        Args:
            collection: Collection名称
            
        Returns:
            统计信息
        """
        result = await self._make_request("DescribeCollectionStatistics", {"Collection": collection})
        return {
            "document_count": result.get("DocumentCount", 0),
            "storage_size": result.get("StorageSize", 0),
            "index_size": result.get("IndexSize", 0),
        }
    
    async def get_usage_stats(self) -> Dict[str, Any]:
        """获取使用统计
        
        Returns:
            使用统计信息
        """
        result = await self._make_request("DescribeUsage", {})
        return {
            "collection_count": result.get("CollectionCount", 0),
            "total_document_count": result.get("TotalDocumentCount", 0),
            "total_storage_size": result.get("TotalStorageSize", 0),
        }
    
    # ========== 工具方法 ==========
    
    def normalize_vector(self, vector: List[float]) -> List[float]:
        """归一化向量
        
        Args:
            vector: 输入向量
            
        Returns:
            归一化后的向量
        """
        import math
        norm = math.sqrt(sum(x * x for x in vector))
        if norm == 0:
            return vector
        return [x / norm for x in vector]
    
    def cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """计算余弦相似度
        
        Args:
            vec1: 向量1
            vec2: 向量2
            
        Returns:
            相似度分数
        """
        import math
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = math.sqrt(sum(x * x for x in vec1))
        norm2 = math.sqrt(sum(x * x for x in vec2))
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def get_capabilities(self):
        """获取适配器能力"""
        return self._capabilities
    
    def supports_capability(self, capability) -> bool:
        """检查是否支持特定能力"""
        return capability in self._capabilities
    
    def __repr__(self) -> str:
        return f"TencentVectorDBAdapter(endpoint={self._cfg.endpoint}, region={self._cfg.region})"


# ========== CLI测试代码 ==========

async def test_tencent_vector_db():
    """测试腾讯云向量数据库适配器"""
    import os
    
    # 从环境变量获取配置
    secret_id = os.environ.get("TENCENT_SECRET_ID", "test-secret-id")
    secret_key = os.environ.get("TENCENT_SECRET_KEY", "test-secret-key")
    endpoint = os.environ.get("TENCENT_VECTOR_DB_ENDPOINT", "tdb.tencentcloudapi.com")
    region = os.environ.get("TENCENT_REGION", "ap-guangzhou")
    
    config = VectorDBConfig(
        secret_id=secret_id,
        secret_key=secret_key,
        endpoint=endpoint,
        region=region,
        default_collection="test_collection",
    )
    
    adapter = TencentVectorDBAdapter(config)
    
    try:
        # 连接
        print("正在连接腾讯云向量数据库...")
        connected = await adapter.connect()
        print(f"连接结果: {connected}")
        
        if not connected:
            print("连接失败，请检查配置")
            return
        
        # 健康检查
        healthy = await adapter.health_check()
        print(f"健康检查: {healthy}")
        
        # 列出Collections
        print("\n列出所有Collections:")
        collections = await adapter.list_collections()
        for name in collections:
            print(f"  - {name}")
        
        # 创建测试Collection
        test_collection = "test_collection_demo"
        print(f"\n创建Collection: {test_collection}")
        try:
            await adapter.create_collection(
                name=test_collection,
                dimension=128,
                metric_type=MetricType.COSINE,
                index_config=IndexConfig.hnsw(m=16, ef_construction=200),
                description="测试Collection",
            )
            print("创建成功")
        except VectorDBAPIError as e:
            if e.code == "ResourceAlreadyExists":
                print("Collection已存在")
            else:
                raise
        
        # 插入测试文档
        print("\n插入测试文档:")
        import random
        docs = []
        for i in range(5):
            vector = [random.random() for _ in range(128)]
            # 归一化向量
            vector = adapter.normalize_vector(vector)
            doc = VectorDocument(
                id=f"doc_{i}",
                vector=vector,
                metadata={"title": f"文档{i}", "category": "test"},
            )
            docs.append(doc)
        
        result = await adapter.upsert_documents(test_collection, docs)
        print(f"插入结果: {result}")
        
        # 搜索测试
        print("\n执行相似度搜索:")
        query_vector = adapter.normalize_vector([random.random() for _ in range(128)])
        search_results = await adapter.search(test_collection, query_vector, top_k=3)
        print(f"搜索耗时: {search_results.search_time:.2f}ms")
        for doc in search_results.documents:
            print(f"  - {doc.id}: score={doc.score:.4f}")
        
        # 获取统计信息
        print("\nCollection统计信息:")
        stats = await adapter.get_collection_stats(test_collection)
        print(f"  文档数: {stats.get('document_count', 0)}")
        print(f"  存储大小: {stats.get('storage_size', 0)} bytes")
        
        print("\n测试完成!")
        
    except Exception as e:
        print(f"测试出错: {e}")
    finally:
        await adapter.disconnect()


if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    
    # 运行测试
    asyncio.run(test_tencent_vector_db())
