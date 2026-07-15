"""
AWS SDK Wrapper - AWS服务封装模块
提供S3、EC2、Lambda、DynamoDB等AWS服务的统一封装接口
"""

import json
import hashlib
import base64
import time
import urllib.request
import urllib.error
import urllib.parse
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional, Any, Union, Tuple, BinaryIO
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import hmac
import logging

logger = logging.getLogger(__name__)


class AWSService(Enum):
    """AWS服务类型枚举"""
    S3 = "s3"
    EC2 = "ec2"
    LAMBDA = "lambda"
    DYNAMODB = "dynamodb"
    SQS = "sqs"
    SNS = "sns"
    KINESIS = "kinesis"
    RDS = "rds"


class AWSRegion(Enum):
    """AWS区域枚举"""
    US_EAST_1 = "us-east-1"
    US_EAST_2 = "us-east-2"
    US_WEST_1 = "us-west-1"
    US_WEST_2 = "us-west-2"
    EU_WEST_1 = "eu-west-1"
    EU_CENTRAL_1 = "eu-central-1"
    AP_SOUTHEAST_1 = "ap-southeast-1"
    AP_NORTHEAST_1 = "ap-northeast-1"


@dataclass
class AWSConfig:
    """AWS配置类"""
    access_key_id: str
    secret_access_key: str
    region: AWSRegion = AWSRegion.US_EAST_1
    endpoint_url: Optional[str] = None
    max_retries: int = 3
    retry_delay: float = 1.0
    timeout: int = 30
    verify_ssl: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "access_key_id": self.access_key_id,
            "secret_access_key": "***",  # 敏感信息隐藏
            "region": self.region.value,
            "endpoint_url": self.endpoint_url,
            "max_retries": self.max_retries,
            "retry_delay": self.retry_delay,
            "timeout": self.timeout,
            "verify_ssl": self.verify_ssl
        }


@dataclass
class S3Object:
    """S3对象元数据"""
    key: str
    bucket: str
    size: int
    last_modified: datetime
    etag: str
    content_type: str = "application/octet-stream"
    metadata: Dict[str, str] = field(default_factory=dict)


@dataclass
class EC2Instance:
    """EC2实例信息"""
    instance_id: str
    instance_type: str
    state: str
    launch_time: datetime
    private_ip: Optional[str] = None
    public_ip: Optional[str] = None
    tags: Dict[str, str] = field(default_factory=dict)


@dataclass
class LambdaFunction:
    """Lambda函数信息"""
    function_name: str
    runtime: str
    handler: str
    code_size: int
    description: str = ""
    timeout: int = 3
    memory_size: int = 128
    last_modified: Optional[datetime] = None


@dataclass
class DynamoDBItem:
    """DynamoDB数据项"""
    table_name: str
    key: Dict[str, Any]
    attributes: Dict[str, Any] = field(default_factory=dict)


class AWSSignatureV4:
    """AWS Signature V4 签名算法"""
    
    ALGORITHM = "AWS4-HMAC-SHA256"
    
    def __init__(self, access_key: str, secret_key: str, region: str):
        self.access_key = access_key
        self.secret_key = secret_key
        self.region = region
    
    def sign(self, method: str, url: str, headers: Dict[str, str], 
             payload: bytes = b"", service: str = "s3") -> Dict[str, str]:
        """生成签名请求头"""
        now = datetime.utcnow()
        amz_date = now.strftime("%Y%m%dT%H%M%SZ")
        date_stamp = now.strftime("%Y%m%d")
        
        # 解析URL
        parsed = urllib.parse.urlparse(url)
        host = parsed.netloc
        canonical_uri = parsed.path or "/"
        canonical_querystring = parsed.query
        
        # 构建规范请求
        canonical_headers = f"host:{host}\nx-amz-date:{amz_date}\n"
        signed_headers = "host;x-amz-date"
        
        payload_hash = hashlib.sha256(payload).hexdigest()
        
        canonical_request = "\n".join([
            method,
            canonical_uri,
            canonical_querystring,
            canonical_headers,
            signed_headers,
            payload_hash
        ])
        
        # 构建待签字符串
        credential_scope = f"{date_stamp}/{self.region}/{service}/aws4_request"
        canonical_request_hash = hashlib.sha256(canonical_request.encode()).hexdigest()
        
        string_to_sign = "\n".join([
            self.ALGORITHM,
            amz_date,
            credential_scope,
            canonical_request_hash
        ])
        
        # 计算签名
        signing_key = self._get_signature_key(date_stamp, service)
        signature = hmac.new(signing_key, string_to_sign.encode(), hashlib.sha256).hexdigest()
        
        # 构建授权头
        authorization_header = (
            f"{self.ALGORITHM} Credential={self.access_key}/{credential_scope}, "
            f"SignedHeaders={signed_headers}, Signature={signature}"
        )
        
        return {
            "Authorization": authorization_header,
            "x-amz-date": amz_date,
            "Host": host
        }
    
    def _get_signature_key(self, date_stamp: str, service: str) -> bytes:
        """获取签名密钥"""
        k_date = hmac.new(f"AWS4{self.secret_key}".encode(), date_stamp.encode(), hashlib.sha256).digest()
        k_region = hmac.new(k_date, self.region.encode(), hashlib.sha256).digest()
        k_service = hmac.new(k_region, service.encode(), hashlib.sha256).digest()
        k_signing = hmac.new(k_service, b"aws4_request", hashlib.sha256).digest()
        return k_signing


class S3Client:
    """S3客户端封装"""
    
    def __init__(self, config: AWSConfig):
        self.config = config
        self.signer = AWSSignatureV4(
            config.access_key_id,
            config.secret_access_key,
            config.region.value
        )
        self._endpoint = config.endpoint_url or f"https://s3.{config.region.value}.amazonaws.com"
    
    def list_buckets(self) -> List[str]:
        """列出所有存储桶"""
        url = f"{self._endpoint}/"
        headers = self.signer.sign("GET", url, {})
        
        try:
            request = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(request, timeout=self.config.timeout) as response:
                root = ET.fromstring(response.read())
                buckets = []
                for bucket in root.findall(".//Bucket/Name"):
                    if bucket.text:
                        buckets.append(bucket.text)
                return buckets
        except Exception as e:
            logger.error(f"Failed to list buckets: {e}")
            return []
    
    def create_bucket(self, bucket_name: str) -> bool:
        """创建存储桶"""
        url = f"{self._endpoint}/{bucket_name}"
        headers = self.signer.sign("PUT", url, {})
        
        try:
            request = urllib.request.Request(url, headers=headers, method="PUT")
            with urllib.request.urlopen(request, timeout=self.config.timeout) as response:
                return response.status == 200
        except Exception as e:
            logger.error(f"Failed to create bucket {bucket_name}: {e}")
            return False
    
    def delete_bucket(self, bucket_name: str) -> bool:
        """删除存储桶"""
        url = f"{self._endpoint}/{bucket_name}"
        headers = self.signer.sign("DELETE", url, {})
        
        try:
            request = urllib.request.Request(url, headers=headers, method="DELETE")
            with urllib.request.urlopen(request, timeout=self.config.timeout) as response:
                return response.status == 204
        except Exception as e:
            logger.error(f"Failed to delete bucket {bucket_name}: {e}")
            return False
    
    def list_objects(self, bucket_name: str, prefix: str = "") -> List[S3Object]:
        """列出存储桶中的对象"""
        url = f"{self._endpoint}/{bucket_name}?list-type=2&prefix={urllib.parse.quote(prefix)}"
        headers = self.signer.sign("GET", url, {})
        
        objects = []
        try:
            request = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(request, timeout=self.config.timeout) as response:
                root = ET.fromstring(response.read())
                for content in root.findall(".//Contents"):
                    key_elem = content.find("Key")
                    size_elem = content.find("Size")
                    etag_elem = content.find("ETag")
                    last_modified_elem = content.find("LastModified")
                    
                    if key_elem is not None and key_elem.text:
                        objects.append(S3Object(
                            key=key_elem.text,
                            bucket=bucket_name,
                            size=int(size_elem.text or 0),
                            last_modified=datetime.fromisoformat(last_modified_elem.text.replace("Z", "+00:00")) if last_modified_elem is not None else datetime.now(),
                            etag=(etag_elem.text or "").strip('"')
                        ))
        except Exception as e:
            logger.error(f"Failed to list objects in {bucket_name}: {e}")
        
        return objects
    
    def get_object(self, bucket_name: str, key: str) -> Optional[bytes]:
        """获取对象内容"""
        url = f"{self._endpoint}/{bucket_name}/{urllib.parse.quote(key)}"
        headers = self.signer.sign("GET", url, {})
        
        try:
            request = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(request, timeout=self.config.timeout) as response:
                return response.read()
        except Exception as e:
            logger.error(f"Failed to get object {key} from {bucket_name}: {e}")
            return None
    
    def put_object(self, bucket_name: str, key: str, data: Union[bytes, str, BinaryIO],
                   content_type: str = "application/octet-stream",
                   metadata: Optional[Dict[str, str]] = None) -> bool:
        """上传对象"""
        url = f"{self._endpoint}/{bucket_name}/{urllib.parse.quote(key)}"
        
        if isinstance(data, str):
            data = data.encode("utf-8")
        elif hasattr(data, "read"):
            data = data.read()
        
        headers = {
            "Content-Type": content_type,
            "Content-Length": str(len(data))
        }
        
        if metadata:
            for k, v in metadata.items():
                headers[f"x-amz-meta-{k}"] = v
        
        signed_headers = self.signer.sign("PUT", url, headers, data)
        headers.update(signed_headers)
        
        try:
            request = urllib.request.Request(url, data=data, headers=headers, method="PUT")
            with urllib.request.urlopen(request, timeout=self.config.timeout) as response:
                return response.status == 200
        except Exception as e:
            logger.error(f"Failed to put object {key} to {bucket_name}: {e}")
            return False
    
    def delete_object(self, bucket_name: str, key: str) -> bool:
        """删除对象"""
        url = f"{self._endpoint}/{bucket_name}/{urllib.parse.quote(key)}"
        headers = self.signer.sign("DELETE", url, {})
        
        try:
            request = urllib.request.Request(url, headers=headers, method="DELETE")
            with urllib.request.urlopen(request, timeout=self.config.timeout) as response:
                return response.status == 204
        except Exception as e:
            logger.error(f"Failed to delete object {key} from {bucket_name}: {e}")
            return False
    
    def get_presigned_url(self, bucket_name: str, key: str, 
                          expiration: int = 3600, method: str = "GET") -> str:
        """生成预签名URL"""
        now = datetime.utcnow()
        amz_date = now.strftime("%Y%m%dT%H%M%SZ")
        date_stamp = now.strftime("%Y%m%d")
        
        expires = str(expiration)
        
        string_to_sign = "\n".join([
            method,
            "",
            "",
            f"x-amz-date:{amz_date}",
            f"x-amz-expires:{expires}",
            f"/{bucket_name}/{key}"
        ])
        
        signing_key = self.signer._get_signature_key(date_stamp, "s3")
        signature = hmac.new(signing_key, string_to_sign.encode(), hashlib.sha256).hexdigest()
        
        credential = f"{self.config.access_key_id}/{date_stamp}/{self.config.region.value}/s3/aws4_request"
        
        params = urllib.parse.urlencode({
            "X-Amz-Algorithm": "AWS4-HMAC-SHA256",
            "X-Amz-Credential": credential,
            "X-Amz-Date": amz_date,
            "X-Amz-Expires": expires,
            "X-Amz-Signature": signature
        })
        
        return f"{self._endpoint}/{bucket_name}/{key}?{params}"


class EC2Client:
    """EC2客户端封装"""
    
    def __init__(self, config: AWSConfig):
        self.config = config
        self.signer = AWSSignatureV4(
            config.access_key_id,
            config.secret_access_key,
            config.region.value
        )
        self._endpoint = config.endpoint_url or f"https://ec2.{config.region.value}.amazonaws.com"
    
    def _make_request(self, action: str, params: Optional[Dict[str, str]] = None) -> Optional[ET.Element]:
        """发送EC2 API请求"""
        query_params = {
            "Action": action,
            "Version": "2016-11-15"
        }
        if params:
            query_params.update(params)
        
        query_string = urllib.parse.urlencode(sorted(query_params.items()))
        url = f"{self._endpoint}/?{query_string}"
        
        headers = self.signer.sign("GET", url, {}, service="ec2")
        
        try:
            request = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(request, timeout=self.config.timeout) as response:
                return ET.fromstring(response.read())
        except Exception as e:
            logger.error(f"EC2 API request failed: {action} - {e}")
            return None
    
    def describe_instances(self, instance_ids: Optional[List[str]] = None) -> List[EC2Instance]:
        """描述实例"""
        params = {}
        if instance_ids:
            for i, instance_id in enumerate(instance_ids, 1):
                params[f"InstanceId.{i}"] = instance_id
        
        root = self._make_request("DescribeInstances", params)
        instances = []
        
        if root is not None:
            for reservation in root.findall(".//reservationSet/item"):
                for item in reservation.findall(".//instancesSet/item"):
                    instance_id_elem = item.find("instanceId")
                    instance_type_elem = item.find("instanceType")
                    state_elem = item.find(".//state/name")
                    launch_time_elem = item.find("launchTime")
                    private_ip_elem = item.find("privateIpAddress")
                    public_ip_elem = item.find("ipAddress")
                    
                    tags = {}
                    for tag in item.findall(".//tagSet/item"):
                        key = tag.find("key")
                        value = tag.find("value")
                        if key is not None and key.text and value is not None:
                            tags[key.text] = value.text or ""
                    
                    instances.append(EC2Instance(
                        instance_id=instance_id_elem.text or "" if instance_id_elem is not None else "",
                        instance_type=instance_type_elem.text or "" if instance_type_elem is not None else "",
                        state=state_elem.text or "" if state_elem is not None else "",
                        launch_time=datetime.fromisoformat(launch_time_elem.text.replace("Z", "+00:00")) if launch_time_elem is not None else datetime.now(),
                        private_ip=private_ip_elem.text if private_ip_elem is not None else None,
                        public_ip=public_ip_elem.text if public_ip_elem is not None else None,
                        tags=tags
                    ))
        
        return instances
    
    def start_instances(self, instance_ids: List[str]) -> Dict[str, str]:
        """启动实例"""
        params = {}
        for i, instance_id in enumerate(instance_ids, 1):
            params[f"InstanceId.{i}"] = instance_id
        
        root = self._make_request("StartInstances", params)
        result = {}
        
        if root is not None:
            for item in root.findall(".//instancesSet/item"):
                instance_id = item.find("instanceId")
                current_state = item.find(".//currentState/name")
                if instance_id is not None and instance_id.text:
                    result[instance_id.text] = current_state.text or "" if current_state is not None else ""
        
        return result
    
    def stop_instances(self, instance_ids: List[str], force: bool = False) -> Dict[str, str]:
        """停止实例"""
        params = {}
        for i, instance_id in enumerate(instance_ids, 1):
            params[f"InstanceId.{i}"] = instance_id
        if force:
            params["Force"] = "true"
        
        root = self._make_request("StopInstances", params)
        result = {}
        
        if root is not None:
            for item in root.findall(".//instancesSet/item"):
                instance_id = item.find("instanceId")
                current_state = item.find(".//currentState/name")
                if instance_id is not None and instance_id.text:
                    result[instance_id.text] = current_state.text or "" if current_state is not None else ""
        
        return result
    
    def terminate_instances(self, instance_ids: List[str]) -> Dict[str, str]:
        """终止实例"""
        params = {}
        for i, instance_id in enumerate(instance_ids, 1):
            params[f"InstanceId.{i}"] = instance_id
        
        root = self._make_request("TerminateInstances", params)
        result = {}
        
        if root is not None:
            for item in root.findall(".//instancesSet/item"):
                instance_id = item.find("instanceId")
                current_state = item.find(".//currentState/name")
                if instance_id is not None and instance_id.text:
                    result[instance_id.text] = current_state.text or "" if current_state is not None else ""
        
        return result
    
    def run_instances(self, image_id: str, instance_type: str, 
                      min_count: int = 1, max_count: int = 1,
                      key_name: Optional[str] = None,
                      security_group_ids: Optional[List[str]] = None,
                      user_data: Optional[str] = None) -> List[str]:
        """运行新实例"""
        params = {
            "ImageId": image_id,
            "InstanceType": instance_type,
            "MinCount": str(min_count),
            "MaxCount": str(max_count)
        }
        
        if key_name:
            params["KeyName"] = key_name
        if security_group_ids:
            for i, sg_id in enumerate(security_group_ids, 1):
                params[f"SecurityGroupId.{i}"] = sg_id
        if user_data:
            params["UserData"] = base64.b64encode(user_data.encode()).decode()
        
        root = self._make_request("RunInstances", params)
        instance_ids = []
        
        if root is not None:
            for item in root.findall(".//instancesSet/item"):
                instance_id = item.find("instanceId")
                if instance_id is not None and instance_id.text:
                    instance_ids.append(instance_id.text)
        
        return instance_ids


class LambdaClient:
    """Lambda客户端封装"""
    
    def __init__(self, config: AWSConfig):
        self.config = config
        self.signer = AWSSignatureV4(
            config.access_key_id,
            config.secret_access_key,
            config.region.value
        )
        self._endpoint = config.endpoint_url or f"https://lambda.{config.region.value}.amazonaws.com"
    
    def _make_request(self, method: str, path: str, 
                      payload: Optional[bytes] = None) -> Optional[Dict[str, Any]]:
        """发送Lambda API请求"""
        url = f"{self._endpoint}{path}"
        headers = {
            "Content-Type": "application/json"
        }
        
        if payload is None:
            payload = b""
        
        signed_headers = self.signer.sign(method, url, headers, payload, service="lambda")
        headers.update(signed_headers)
        
        try:
            request = urllib.request.Request(url, data=payload, headers=headers, method=method)
            with urllib.request.urlopen(request, timeout=self.config.timeout) as response:
                data = response.read()
                return json.loads(data) if data else {}
        except urllib.error.HTTPError as e:
            if e.code == 204:
                return {}
            error_body = e.read()
            logger.error(f"Lambda API error: {e.code} - {error_body}")
            return None
        except Exception as e:
            logger.error(f"Lambda API request failed: {e}")
            return None
    
    def list_functions(self) -> List[LambdaFunction]:
        """列出所有函数"""
        result = self._make_request("GET", "/2015-03-31/functions/")
        functions = []
        
        if result and "Functions" in result:
            for func_data in result["Functions"]:
                functions.append(LambdaFunction(
                    function_name=func_data.get("FunctionName", ""),
                    runtime=func_data.get("Runtime", ""),
                    handler=func_data.get("Handler", ""),
                    code_size=func_data.get("CodeSize", 0),
                    description=func_data.get("Description", ""),
                    timeout=func_data.get("Timeout", 3),
                    memory_size=func_data.get("MemorySize", 128),
                    last_modified=datetime.fromisoformat(func_data["LastModified"].replace("Z", "+00:00")) if "LastModified" in func_data else None
                ))
        
        return functions
    
    def get_function(self, function_name: str) -> Optional[LambdaFunction]:
        """获取函数详情"""
        result = self._make_request("GET", f"/2015-03-31/functions/{function_name}")
        
        if result and "Configuration" in result:
            config = result["Configuration"]
            return LambdaFunction(
                function_name=config.get("FunctionName", ""),
                runtime=config.get("Runtime", ""),
                handler=config.get("Handler", ""),
                code_size=config.get("CodeSize", 0),
                description=config.get("Description", ""),
                timeout=config.get("Timeout", 3),
                memory_size=config.get("MemorySize", 128),
                last_modified=datetime.fromisoformat(config["LastModified"].replace("Z", "+00:00")) if "LastModified" in config else None
            )
        return None
    
    def invoke(self, function_name: str, payload: Optional[Dict[str, Any]] = None,
               invocation_type: str = "RequestResponse") -> Optional[Dict[str, Any]]:
        """调用函数"""
        path = f"/2015-03-31/functions/{function_name}/invocations"
        
        headers_extra = {}
        if invocation_type == "Event":
            headers_extra["X-Amz-Invocation-Type"] = "Event"
        
        payload_bytes = json.dumps(payload or {}).encode()
        
        result = self._make_request("POST", path, payload_bytes)
        return result
    
    def create_function(self, function_name: str, runtime: str, handler: str,
                        code_zip_bytes: bytes, role: str,
                        description: str = "", timeout: int = 3,
                        memory_size: int = 128) -> Optional[str]:
        """创建函数"""
        # 注: 实际实现需要先上传代码到S3
        payload = json.dumps({
            "FunctionName": function_name,
            "Runtime": runtime,
            "Handler": handler,
            "Role": role,
            "Code": {"ZipFile": base64.b64encode(code_zip_bytes).decode()},
            "Description": description,
            "Timeout": timeout,
            "MemorySize": memory_size
        }).encode()
        
        result = self._make_request("POST", "/2015-03-31/functions", payload)
        
        if result:
            return result.get("FunctionName")
        return None
    
    def delete_function(self, function_name: str) -> bool:
        """删除函数"""
        result = self._make_request("DELETE", f"/2015-03-31/functions/{function_name}")
        return result is not None


class DynamoDBClient:
    """DynamoDB客户端封装"""
    
    def __init__(self, config: AWSConfig):
        self.config = config
        self.signer = AWSSignatureV4(
            config.access_key_id,
            config.secret_access_key,
            config.region.value
        )
        self._endpoint = config.endpoint_url or f"https://dynamodb.{config.region.value}.amazonaws.com"
    
    def _make_request(self, target: str, payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """发送DynamoDB API请求"""
        url = self._endpoint
        payload_bytes = json.dumps(payload).encode()
        
        headers = {
            "Content-Type": "application/x-amz-json-1.0",
            "X-Amz-Target": target
        }
        
        signed_headers = self.signer.sign("POST", url, headers, payload_bytes, service="dynamodb")
        headers.update(signed_headers)
        
        try:
            request = urllib.request.Request(url, data=payload_bytes, headers=headers, method="POST")
            with urllib.request.urlopen(request, timeout=self.config.timeout) as response:
                data = response.read()
                return json.loads(data) if data else {}
        except urllib.error.HTTPError as e:
            if e.code == 200:
                return {}
            error_body = e.read()
            logger.error(f"DynamoDB API error: {e.code} - {error_body}")
            return None
        except Exception as e:
            logger.error(f"DynamoDB API request failed: {e}")
            return None
    
    def _encode_value(self, value: Any) -> Dict[str, Any]:
        """编码DynamoDB属性值"""
        if value is None:
            return {"NULL": True}
        elif isinstance(value, bool):
            return {"BOOL": value}
        elif isinstance(value, int):
            return {"N": str(value)}
        elif isinstance(value, float):
            return {"N": str(value)}
        elif isinstance(value, str):
            return {"S": value}
        elif isinstance(value, bytes):
            return {"B": base64.b64encode(value).decode()}
        elif isinstance(value, list):
            return {"L": [self._encode_value(v) for v in value]}
        elif isinstance(value, dict):
            return {"M": {k: self._encode_value(v) for k, v in value.items()}}
        else:
            return {"S": str(value)}
    
    def _decode_value(self, value: Dict[str, Any]) -> Any:
        """解码DynamoDB属性值"""
        if "NULL" in value:
            return None
        elif "BOOL" in value:
            return value["BOOL"]
        elif "N" in value:
            num_str = value["N"]
            return int(num_str) if "." not in num_str else float(num_str)
        elif "S" in value:
            return value["S"]
        elif "B" in value:
            return base64.b64decode(value["B"])
        elif "L" in value:
            return [self._decode_value(v) for v in value["L"]]
        elif "M" in value:
            return {k: self._decode_value(v) for k, v in value["M"].items()}
        else:
            return None
    
    def list_tables(self) -> List[str]:
        """列出所有表"""
        result = self._make_request("DynamoDB_20120810.ListTables", {})
        if result and "TableNames" in result:
            return result["TableNames"]
        return []
    
    def create_table(self, table_name: str, key_schema: List[Dict[str, str]],
                     attribute_definitions: List[Dict[str, str]],
                     billing_mode: str = "PAY_PER_REQUEST") -> bool:
        """创建表"""
        payload = {
            "TableName": table_name,
            "KeySchema": key_schema,
            "AttributeDefinitions": attribute_definitions,
            "BillingMode": billing_mode
        }
        
        result = self._make_request("DynamoDB_20120810.CreateTable", payload)
        return result is not None
    
    def delete_table(self, table_name: str) -> bool:
        """删除表"""
        result = self._make_request("DynamoDB_20120810.DeleteTable", {"TableName": table_name})
        return result is not None
    
    def put_item(self, table_name: str, item: Dict[str, Any]) -> bool:
        """写入数据项"""
        payload = {
            "TableName": table_name,
            "Item": {k: self._encode_value(v) for k, v in item.items()}
        }
        
        result = self._make_request("DynamoDB_20120810.PutItem", payload)
        return result is not None
    
    def get_item(self, table_name: str, key: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """获取数据项"""
        payload = {
            "TableName": table_name,
            "Key": {k: self._encode_value(v) for k, v in key.items()}
        }
        
        result = self._make_request("DynamoDB_20120810.GetItem", payload)
        if result and "Item" in result:
            return {k: self._decode_value(v) for k, v in result["Item"].items()}
        return None
    
    def update_item(self, table_name: str, key: Dict[str, Any],
                    updates: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """更新数据项"""
        update_expression_parts = []
        expression_attribute_values = {}
        
        for i, (attr_name, attr_value) in enumerate(updates.items()):
            placeholder = f":val{i}"
            update_expression_parts.append(f"{attr_name} = {placeholder}")
            expression_attribute_values[placeholder] = self._encode_value(attr_value)
        
        payload = {
            "TableName": table_name,
            "Key": {k: self._encode_value(v) for k, v in key.items()},
            "UpdateExpression": "SET " + ", ".join(update_expression_parts),
            "ExpressionAttributeValues": expression_attribute_values,
            "ReturnValues": "ALL_NEW"
        }
        
        result = self._make_request("DynamoDB_20120810.UpdateItem", payload)
        if result and "Attributes" in result:
            return {k: self._decode_value(v) for k, v in result["Attributes"].items()}
        return None
    
    def delete_item(self, table_name: str, key: Dict[str, Any]) -> bool:
        """删除数据项"""
        payload = {
            "TableName": table_name,
            "Key": {k: self._encode_value(v) for k, v in key.items()}
        }
        
        result = self._make_request("DynamoDB_20120810.DeleteItem", payload)
        return result is not None
    
    def query(self, table_name: str, key_condition_expression: str,
              expression_attribute_values: Dict[str, Any],
              index_name: Optional[str] = None,
              limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """查询数据"""
        payload = {
            "TableName": table_name,
            "KeyConditionExpression": key_condition_expression,
            "ExpressionAttributeValues": {k: self._encode_value(v) for k, v in expression_attribute_values.items()}
        }
        
        if index_name:
            payload["IndexName"] = index_name
        if limit:
            payload["Limit"] = limit
        
        result = self._make_request("DynamoDB_20120810.Query", payload)
        items = []
        
        if result and "Items" in result:
            for item in result["Items"]:
                items.append({k: self._decode_value(v) for k, v in item.items()})
        
        return items
    
    def scan(self, table_name: str, filter_expression: Optional[str] = None,
             expression_attribute_values: Optional[Dict[str, Any]] = None,
             limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """扫描表"""
        payload = {"TableName": table_name}
        
        if filter_expression:
            payload["FilterExpression"] = filter_expression
        if expression_attribute_values:
            payload["ExpressionAttributeValues"] = {k: self._encode_value(v) for k, v in expression_attribute_values.items()}
        if limit:
            payload["Limit"] = limit
        
        result = self._make_request("DynamoDB_20120810.Scan", payload)
        items = []
        
        if result and "Items" in result:
            for item in result["Items"]:
                items.append({k: self._decode_value(v) for k, v in item.items()})
        
        return items


class AWSSDKWrapper:
    """AWS SDK统一封装器"""
    
    def __init__(self, config: AWSConfig):
        self.config = config
        self._s3_client: Optional[S3Client] = None
        self._ec2_client: Optional[EC2Client] = None
        self._lambda_client: Optional[LambdaClient] = None
        self._dynamodb_client: Optional[DynamoDBClient] = None
    
    @property
    def s3(self) -> S3Client:
        """获取S3客户端"""
        if self._s3_client is None:
            self._s3_client = S3Client(self.config)
        return self._s3_client
    
    @property
    def ec2(self) -> EC2Client:
        """获取EC2客户端"""
        if self._ec2_client is None:
            self._ec2_client = EC2Client(self.config)
        return self._ec2_client
    
    @property
    def lambda_(self) -> LambdaClient:
        """获取Lambda客户端"""
        if self._lambda_client is None:
            self._lambda_client = LambdaClient(self.config)
        return self._lambda_client
    
    @property
    def dynamodb(self) -> DynamoDBClient:
        """获取DynamoDB客户端"""
        if self._dynamodb_client is None:
            self._dynamodb_client = DynamoDBClient(self.config)
        return self._dynamodb_client
    
    def get_client(self, service: AWSService) -> Any:
        """根据服务类型获取客户端"""
        if service == AWSService.S3:
            return self.s3
        elif service == AWSService.EC2:
            return self.ec2
        elif service == AWSService.LAMBDA:
            return self.lambda_
        elif service == AWSService.DYNAMODB:
            return self.dynamodb
        else:
            raise ValueError(f"Unsupported service: {service}")
    
    def close(self) -> None:
        """关闭所有客户端连接"""
        self._s3_client = None
        self._ec2_client = None
        self._lambda_client = None
        self._dynamodb_client = None


def create_aws_client(access_key_id: str, secret_access_key: str,
                      region: str = "us-east-1", **kwargs) -> AWSSDKWrapper:
    """创建AWS客户端的便捷函数"""
    region_enum = AWSRegion(region)
    config = AWSConfig(
        access_key_id=access_key_id,
        secret_access_key=secret_access_key,
        region=region_enum,
        **kwargs
    )
    return AWSSDKWrapper(config)
