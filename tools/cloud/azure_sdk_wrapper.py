"""
Azure SDK Wrapper - Azure服务封装模块
提供Blob Storage、Virtual Machines、Functions等Azure服务的统一封装接口
"""

import json
import hashlib
import base64
import time
import urllib.request
import urllib.error
import urllib.parse
from typing import Dict, List, Optional, Any, Union, BinaryIO
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class AzureService(Enum):
    """Azure服务类型枚举"""
    BLOB = "blob"
    VM = "vm"
    FUNCTIONS = "functions"
    COSMOS_DB = "cosmos_db"
    SERVICE_BUS = "service_bus"
    KEY_VAULT = "key_vault"


class AzureRegion(Enum):
    """Azure区域枚举"""
    EAST_US = "eastus"
    EAST_US_2 = "eastus2"
    WEST_US = "westus"
    WEST_US_2 = "westus2"
    CENTRAL_US = "centralus"
    NORTH_EUROPE = "northeurope"
    WEST_EUROPE = "westeurope"
    EAST_ASIA = "eastasia"
    SOUTHEAST_ASIA = "southeastasia"
    JAPAN_EAST = "japaneast"
    JAPAN_WEST = "japanwest"


@dataclass
class AzureConfig:
    """Azure配置类"""
    subscription_id: str
    tenant_id: str
    client_id: str
    client_secret: str
    resource_group: str
    region: AzureRegion = AzureRegion.EAST_US
    endpoint_url: Optional[str] = None
    max_retries: int = 3
    retry_delay: float = 1.0
    timeout: int = 30
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "subscription_id": self.subscription_id,
            "tenant_id": self.tenant_id,
            "client_id": self.client_id,
            "client_secret": "***",  # 敏感信息隐藏
            "resource_group": self.resource_group,
            "region": self.region.value,
            "endpoint_url": self.endpoint_url,
            "max_retries": self.max_retries,
            "retry_delay": self.retry_delay,
            "timeout": self.timeout
        }


@dataclass
class BlobContainer:
    """Blob容器信息"""
    name: str
    last_modified: datetime
    etag: str
    public_access: str = "None"
    lease_status: str = "unlocked"


@dataclass
class BlobObject:
    """Blob对象信息"""
    name: str
    container: str
    size: int
    last_modified: datetime
    etag: str
    content_type: str = "application/octet-stream"
    blob_type: str = "BlockBlob"


@dataclass
class VirtualMachine:
    """虚拟机信息"""
    name: str
    location: str
    vm_size: str
    provisioning_state: str
    power_state: str
    os_type: str
    os_name: Optional[str] = None
    network_interfaces: List[str] = field(default_factory=list)
    tags: Dict[str, str] = field(default_factory=dict)


@dataclass
class AzureFunction:
    """Azure函数信息"""
    name: str
    location: str
    runtime: str
    runtime_version: str
    state: str
    https_url: Optional[str] = None
    tags: Dict[str, str] = field(default_factory=dict)


class AzureAuthentication:
    """Azure认证管理"""
    
    AUTHORITY_TEMPLATE = "https://login.microsoftonline.com/{}"
    RESOURCE_MANAGER = "https://management.azure.com/"
    STORAGE_RESOURCE = "https://storage.azure.com/"
    
    def __init__(self, tenant_id: str, client_id: str, client_secret: str):
        self.tenant_id = tenant_id
        self.client_id = client_id
        self.client_secret = client_secret
        self._access_token: Optional[str] = None
        self._token_expires: Optional[datetime] = None
        self._storage_token: Optional[str] = None
        self._storage_token_expires: Optional[datetime] = None
    
    def get_management_token(self) -> str:
        """获取Azure资源管理器访问令牌"""
        if self._access_token and self._token_expires and datetime.now() < self._token_expires:
            return self._access_token
        
        authority = self.AUTHORITY_TEMPLATE.format(self.tenant_id)
        token_url = f"{authority}/oauth2/token"
        
        data = urllib.parse.urlencode({
            "grant_type": "client_credentials",
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "resource": self.RESOURCE_MANAGER
        }).encode()
        
        try:
            request = urllib.request.Request(token_url, data=data, method="POST")
            with urllib.request.urlopen(request, timeout=30) as response:
                result = json.loads(response.read())
                self._access_token = result["access_token"]
                expires_in = int(result.get("expires_in", 3600))
                self._token_expires = datetime.now() + timedelta(seconds=expires_in - 300)
                return self._access_token
        except Exception as e:
            logger.error(f"Failed to get management token: {e}")
            raise
    
    def get_storage_token(self) -> str:
        """获取存储服务访问令牌"""
        if self._storage_token and self._storage_token_expires and datetime.now() < self._storage_token_expires:
            return self._storage_token
        
        authority = self.AUTHORITY_TEMPLATE.format(self.tenant_id)
        token_url = f"{authority}/oauth2/token"
        
        data = urllib.parse.urlencode({
            "grant_type": "client_credentials",
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "resource": self.STORAGE_RESOURCE
        }).encode()
        
        try:
            request = urllib.request.Request(token_url, data=data, method="POST")
            with urllib.request.urlopen(request, timeout=30) as response:
                result = json.loads(response.read())
                self._storage_token = result["access_token"]
                expires_in = int(result.get("expires_in", 3600))
                self._storage_token_expires = datetime.now() + timedelta(seconds=expires_in - 300)
                return self._storage_token
        except Exception as e:
            logger.error(f"Failed to get storage token: {e}")
            raise


class BlobClient:
    """Azure Blob Storage客户端封装"""
    
    API_VERSION = "2023-01-03"
    
    def __init__(self, config: AzureConfig, auth: AzureAuthentication, storage_account: str):
        self.config = config
        self.auth = auth
        self.storage_account = storage_account
        self._endpoint = f"https://{storage_account}.blob.core.windows.net"
    
    def _make_request(self, method: str, path: str, 
                      params: Optional[Dict[str, str]] = None,
                      data: Optional[bytes] = None,
                      headers: Optional[Dict[str, str]] = None) -> Optional[Any]:
        """发送Blob API请求"""
        url = f"{self._endpoint}{path}"
        if params:
            url += "?" + urllib.parse.urlencode(params)
        
        token = self.auth.get_storage_token()
        request_headers = {
            "Authorization": f"Bearer {token}",
            "x-ms-version": self.API_VERSION,
            "x-ms-date": datetime.utcnow().strftime("%a, %d %b %Y %H:%M:%S GMT")
        }
        
        if headers:
            request_headers.update(headers)
        
        if data and "Content-Length" not in request_headers:
            request_headers["Content-Length"] = str(len(data))
        
        try:
            request = urllib.request.Request(url, data=data, headers=request_headers, method=method)
            with urllib.request.urlopen(request, timeout=self.config.timeout) as response:
                content = response.read()
                content_type = response.headers.get("Content-Type", "")
                
                if "application/json" in content_type or "application/xml" in content_type:
                    return json.loads(content) if content else {}
                return content
        except urllib.error.HTTPError as e:
            if e.code in (201, 202, 204):
                return {}
            logger.error(f"Blob API error: {e.code} - {e.read()}")
            return None
        except Exception as e:
            logger.error(f"Blob API request failed: {e}")
            return None
    
    def list_containers(self) -> List[BlobContainer]:
        """列出所有容器"""
        result = self._make_request("GET", "/", {"comp": "list"})
        containers = []
        
        if result and isinstance(result, dict):
            for container_data in result.get("Containers", []):
                containers.append(BlobContainer(
                    name=container_data.get("Name", ""),
                    last_modified=datetime.fromisoformat(container_data["Last-Modified"].replace("Z", "+00:00")),
                    etag=container_data.get("Etag", ""),
                    public_access=container_data.get("PublicAccess", "None"),
                    lease_status=container_data.get("LeaseStatus", "unlocked")
                ))
        
        return containers
    
    def create_container(self, container_name: str, public_access: str = "None") -> bool:
        """创建容器"""
        headers = {"x-ms-blob-public-access": public_access}
        result = self._make_request("PUT", f"/{container_name}", headers=headers)
        return result is not None
    
    def delete_container(self, container_name: str) -> bool:
        """删除容器"""
        result = self._make_request("DELETE", f"/{container_name}")
        return result is not None
    
    def list_blobs(self, container_name: str, prefix: str = "") -> List[BlobObject]:
        """列出容器中的Blob"""
        params = {"comp": "list", "restype": "container"}
        if prefix:
            params["prefix"] = prefix
        
        result = self._make_request("GET", f"/{container_name}", params)
        blobs = []
        
        if result and isinstance(result, dict):
            for blob_data in result.get("Blobs", []):
                blobs.append(BlobObject(
                    name=blob_data.get("Name", ""),
                    container=container_name,
                    size=int(blob_data.get("Content-Length", 0)),
                    last_modified=datetime.fromisoformat(blob_data["Last-Modified"].replace("Z", "+00:00")),
                    etag=blob_data.get("Etag", ""),
                    content_type=blob_data.get("Content-Type", "application/octet-stream"),
                    blob_type=blob_data.get("BlobType", "BlockBlob")
                ))
        
        return blobs
    
    def get_blob(self, container_name: str, blob_name: str) -> Optional[bytes]:
        """获取Blob内容"""
        result = self._make_request("GET", f"/{container_name}/{blob_name}")
        if isinstance(result, bytes):
            return result
        return None
    
    def put_blob(self, container_name: str, blob_name: str, 
                 data: Union[bytes, str, BinaryIO],
                 content_type: str = "application/octet-stream") -> bool:
        """上传Blob"""
        if isinstance(data, str):
            data = data.encode("utf-8")
        elif hasattr(data, "read"):
            data = data.read()
        
        headers = {
            "Content-Type": content_type,
            "x-ms-blob-type": "BlockBlob"
        }
        
        result = self._make_request("PUT", f"/{container_name}/{blob_name}", data=data, headers=headers)
        return result is not None
    
    def delete_blob(self, container_name: str, blob_name: str) -> bool:
        """删除Blob"""
        result = self._make_request("DELETE", f"/{container_name}/{blob_name}")
        return result is not None
    
    def get_blob_properties(self, container_name: str, blob_name: str) -> Optional[Dict[str, Any]]:
        """获取Blob属性"""
        result = self._make_request("HEAD", f"/{container_name}/{blob_name}")
        return result if isinstance(result, dict) else None
    
    def generate_sas_url(self, container_name: str, blob_name: str,
                         permission: str = "r", expiry_hours: int = 1) -> str:
        """生成SAS URL (简化版)"""
        # 注: 完整SAS签名需要更复杂的实现
        expiry = datetime.utcnow() + timedelta(hours=expiry_hours)
        expiry_str = expiry.strftime("%Y-%m-%dT%H:%M:%SZ")
        
        sas_params = {
            "sv": self.API_VERSION,
            "ss": "b",
            "srt": "o",
            "sp": permission,
            "se": expiry_str,
            "spr": "https"
        }
        
        sas_string = urllib.parse.urlencode(sas_params)
        return f"{self._endpoint}/{container_name}/{blob_name}?{sas_string}"


class VMClient:
    """Azure虚拟机客户端封装"""
    
    API_VERSION = "2023-03-01"
    
    def __init__(self, config: AzureConfig, auth: AzureAuthentication):
        self.config = config
        self.auth = auth
        self._endpoint = "https://management.azure.com"
    
    def _make_request(self, method: str, path: str,
                      data: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """发送VM API请求"""
        url = f"{self._endpoint}{path}"
        
        if "api-version" not in url:
            url += f"?api-version={self.API_VERSION}"
        
        token = self.auth.get_management_token()
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }
        
        request_data = json.dumps(data).encode() if data else None
        
        try:
            request = urllib.request.Request(url, data=request_data, headers=headers, method=method)
            with urllib.request.urlopen(request, timeout=self.config.timeout) as response:
                content = response.read()
                return json.loads(content) if content else {}
        except urllib.error.HTTPError as e:
            if e.code in (201, 202, 204):
                content = e.read()
                return json.loads(content) if content else {}
            logger.error(f"VM API error: {e.code} - {e.read()}")
            return None
        except Exception as e:
            logger.error(f"VM API request failed: {e}")
            return None
    
    def list_vms(self) -> List[VirtualMachine]:
        """列出所有虚拟机"""
        path = f"/subscriptions/{self.config.subscription_id}/resourceGroups/{self.config.resource_group}/providers/Microsoft.Compute/virtualMachines"
        result = self._make_request("GET", path)
        vms = []
        
        if result and "value" in result:
            for vm_data in result["value"]:
                properties = vm_data.get("properties", {})
                hardware_profile = properties.get("hardwareProfile", {})
                storage_profile = properties.get("storageProfile", {})
                os_profile = storage_profile.get("osDisk", {})
                
                tags = vm_data.get("tags", {})
                network_interfaces = []
                for nic in properties.get("networkProfile", {}).get("networkInterfaces", []):
                    network_interfaces.append(nic.get("id", ""))
                
                vms.append(VirtualMachine(
                    name=vm_data.get("name", ""),
                    location=vm_data.get("location", ""),
                    vm_size=hardware_profile.get("vmSize", ""),
                    provisioning_state=properties.get("provisioningState", ""),
                    power_state="Unknown",  # 需要单独查询实例视图
                    os_type=os_profile.get("osType", ""),
                    os_name=None,
                    network_interfaces=network_interfaces,
                    tags=tags
                ))
        
        return vms
    
    def get_vm(self, vm_name: str) -> Optional[VirtualMachine]:
        """获取虚拟机详情"""
        path = f"/subscriptions/{self.config.subscription_id}/resourceGroups/{self.config.resource_group}/providers/Microsoft.Compute/virtualMachines/{vm_name}"
        result = self._make_request("GET", path)
        
        if result:
            properties = result.get("properties", {})
            hardware_profile = properties.get("hardwareProfile", {})
            storage_profile = properties.get("storageProfile", {})
            os_profile = storage_profile.get("osDisk", {})
            
            network_interfaces = []
            for nic in properties.get("networkProfile", {}).get("networkInterfaces", []):
                network_interfaces.append(nic.get("id", ""))
            
            return VirtualMachine(
                name=result.get("name", ""),
                location=result.get("location", ""),
                vm_size=hardware_profile.get("vmSize", ""),
                provisioning_state=properties.get("provisioningState", ""),
                power_state="Unknown",
                os_type=os_profile.get("osType", ""),
                os_name=None,
                network_interfaces=network_interfaces,
                tags=result.get("tags", {})
            )
        return None
    
    def start_vm(self, vm_name: str) -> bool:
        """启动虚拟机"""
        path = f"/subscriptions/{self.config.subscription_id}/resourceGroups/{self.config.resource_group}/providers/Microsoft.Compute/virtualMachines/{vm_name}/start"
        result = self._make_request("POST", path)
        return result is not None
    
    def stop_vm(self, vm_name: str, skip_shutdown: bool = False) -> bool:
        """停止虚拟机"""
        path = f"/subscriptions/{self.config.subscription_id}/resourceGroups/{self.config.resource_group}/providers/Microsoft.Compute/virtualMachines/{vm_name}/powerOff"
        if skip_shutdown:
            path += "?skipShutdown=true"
        result = self._make_request("POST", path)
        return result is not None
    
    def restart_vm(self, vm_name: str) -> bool:
        """重启虚拟机"""
        path = f"/subscriptions/{self.config.subscription_id}/resourceGroups/{self.config.resource_group}/providers/Microsoft.Compute/virtualMachines/{vm_name}/restart"
        result = self._make_request("POST", path)
        return result is not None
    
    def deallocate_vm(self, vm_name: str) -> bool:
        """释放虚拟机"""
        path = f"/subscriptions/{self.config.subscription_id}/resourceGroups/{self.config.resource_group}/providers/Microsoft.Compute/virtualMachines/{vm_name}/deallocate"
        result = self._make_request("POST", path)
        return result is not None
    
    def create_vm(self, vm_name: str, vm_size: str, image_reference: Dict[str, str],
                  os_disk: Dict[str, Any], network_interface_id: str,
                  admin_username: str, ssh_public_key: Optional[str] = None,
                  tags: Optional[Dict[str, str]] = None) -> bool:
        """创建虚拟机"""
        path = f"/subscriptions/{self.config.subscription_id}/resourceGroups/{self.config.resource_group}/providers/Microsoft.Compute/virtualMachines/{vm_name}"
        
        os_profile = {
            "adminUsername": admin_username,
            "computerName": vm_name
        }
        
        if ssh_public_key:
            os_profile["linuxConfiguration"] = {
                "disablePasswordAuthentication": True,
                "ssh": {
                    "publicKeys": [{
                        "path": f"/home/{admin_username}/.ssh/authorized_keys",
                        "keyData": ssh_public_key
                    }]
                }
            }
        
        vm_data = {
            "location": self.config.region.value,
            "properties": {
                "hardwareProfile": {"vmSize": vm_size},
                "storageProfile": {
                    "imageReference": image_reference,
                    "osDisk": os_disk
                },
                "osProfile": os_profile,
                "networkProfile": {
                    "networkInterfaces": [{
                        "id": network_interface_id,
                        "properties": {"primary": True}
                    }]
                }
            }
        }
        
        if tags:
            vm_data["tags"] = tags
        
        result = self._make_request("PUT", path, vm_data)
        return result is not None
    
    def delete_vm(self, vm_name: str) -> bool:
        """删除虚拟机"""
        path = f"/subscriptions/{self.config.subscription_id}/resourceGroups/{self.config.resource_group}/providers/Microsoft.Compute/virtualMachines/{vm_name}"
        result = self._make_request("DELETE", path)
        return result is not None


class FunctionsClient:
    """Azure Functions客户端封装"""
    
    API_VERSION = "2022-03-01"
    
    def __init__(self, config: AzureConfig, auth: AzureAuthentication):
        self.config = config
        self.auth = auth
        self._endpoint = "https://management.azure.com"
    
    def _make_request(self, method: str, path: str,
                      data: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """发送Functions API请求"""
        url = f"{self._endpoint}{path}"
        
        if "api-version" not in url:
            url += f"?api-version={self.API_VERSION}"
        
        token = self.auth.get_management_token()
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }
        
        request_data = json.dumps(data).encode() if data else None
        
        try:
            request = urllib.request.Request(url, data=request_data, headers=headers, method=method)
            with urllib.request.urlopen(request, timeout=self.config.timeout) as response:
                content = response.read()
                return json.loads(content) if content else {}
        except urllib.error.HTTPError as e:
            if e.code in (201, 202, 204):
                content = e.read()
                return json.loads(content) if content else {}
            logger.error(f"Functions API error: {e.code} - {e.read()}")
            return None
        except Exception as e:
            logger.error(f"Functions API request failed: {e}")
            return None
    
    def list_function_apps(self) -> List[AzureFunction]:
        """列出所有函数应用"""
        path = f"/subscriptions/{self.config.subscription_id}/resourceGroups/{self.config.resource_group}/providers/Microsoft.Web/sites"
        result = self._make_request("GET", path)
        functions = []
        
        if result and "value" in result:
            for site_data in result["value"]:
                properties = site_data.get("properties", {})
                site_config = properties.get("siteConfig", {})
                
                functions.append(AzureFunction(
                    name=site_data.get("name", ""),
                    location=site_data.get("location", ""),
                    runtime=site_config.get("linuxFxVersion", "").split("|")[0] if "|" in site_config.get("linuxFxVersion", "") else "",
                    runtime_version=site_config.get("linuxFxVersion", "").split("|")[1] if "|" in site_config.get("linuxFxVersion", "") else "",
                    state=properties.get("state", ""),
                    https_url=f"https://{site_data.get('name', '')}.azurewebsites.net",
                    tags=site_data.get("tags", {})
                ))
        
        return functions
    
    def get_function_app(self, app_name: str) -> Optional[AzureFunction]:
        """获取函数应用详情"""
        path = f"/subscriptions/{self.config.subscription_id}/resourceGroups/{self.config.resource_group}/providers/Microsoft.Web/sites/{app_name}"
        result = self._make_request("GET", path)
        
        if result:
            properties = result.get("properties", {})
            site_config = properties.get("siteConfig", {})
            
            return AzureFunction(
                name=result.get("name", ""),
                location=result.get("location", ""),
                runtime=site_config.get("linuxFxVersion", "").split("|")[0] if "|" in site_config.get("linuxFxVersion", "") else "",
                runtime_version=site_config.get("linuxFxVersion", "").split("|")[1] if "|" in site_config.get("linuxFxVersion", "") else "",
                state=properties.get("state", ""),
                https_url=f"https://{result.get('name', '')}.azurewebsites.net",
                tags=result.get("tags", {})
            )
        return None
    
    def create_function_app(self, app_name: str, runtime: str, runtime_version: str,
                            storage_account_name: str, plan_id: str,
                            tags: Optional[Dict[str, str]] = None) -> bool:
        """创建函数应用"""
        path = f"/subscriptions/{self.config.subscription_id}/resourceGroups/{self.config.resource_group}/providers/Microsoft.Web/sites/{app_name}"
        
        app_data = {
            "location": self.config.region.value,
            "kind": "functionapp,linux",
            "properties": {
                "serverFarmId": plan_id,
                "siteConfig": {
                    "linuxFxVersion": f"{runtime}|{runtime_version}"
                }
            }
        }
        
        if tags:
            app_data["tags"] = tags
        
        result = self._make_request("PUT", path, app_data)
        return result is not None
    
    def delete_function_app(self, app_name: str) -> bool:
        """删除函数应用"""
        path = f"/subscriptions/{self.config.subscription_id}/resourceGroups/{self.config.resource_group}/providers/Microsoft.Web/sites/{app_name}"
        result = self._make_request("DELETE", path)
        return result is not None
    
    def list_functions(self, app_name: str) -> List[Dict[str, Any]]:
        """列出函数应用中的函数"""
        path = f"/subscriptions/{self.config.subscription_id}/resourceGroups/{self.config.resource_group}/providers/Microsoft.Web/sites/{app_name}/functions"
        result = self._make_request("GET", path)
        
        if result and "value" in result:
            return result["value"]
        return []
    
    def invoke_function(self, app_name: str, function_name: str,
                        payload: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """调用函数"""
        # 注: 实际调用需要函数密钥或匿名访问
        url = f"https://{app_name}.azurewebsites.net/api/{function_name}"
        
        headers = {"Content-Type": "application/json"}
        data = json.dumps(payload or {}).encode()
        
        try:
            request = urllib.request.Request(url, data=data, headers=headers, method="POST")
            with urllib.request.urlopen(request, timeout=self.config.timeout) as response:
                content = response.read()
                return json.loads(content) if content else {}
        except Exception as e:
            logger.error(f"Failed to invoke function: {e}")
            return None


class AzureSDKWrapper:
    """Azure SDK统一封装器"""
    
    def __init__(self, config: AzureConfig):
        self.config = config
        self.auth = AzureAuthentication(
            config.tenant_id,
            config.client_id,
            config.client_secret
        )
        self._blob_client: Optional[BlobClient] = None
        self._vm_client: Optional[VMClient] = None
        self._functions_client: Optional[FunctionsClient] = None
    
    def blob(self, storage_account: str) -> BlobClient:
        """获取Blob客户端"""
        if self._blob_client is None or self._blob_client.storage_account != storage_account:
            self._blob_client = BlobClient(self.config, self.auth, storage_account)
        return self._blob_client
    
    @property
    def vm(self) -> VMClient:
        """获取VM客户端"""
        if self._vm_client is None:
            self._vm_client = VMClient(self.config, self.auth)
        return self._vm_client
    
    @property
    def functions(self) -> FunctionsClient:
        """获取Functions客户端"""
        if self._functions_client is None:
            self._functions_client = FunctionsClient(self.config, self.auth)
        return self._functions_client
    
    def get_client(self, service: AzureService, **kwargs) -> Any:
        """根据服务类型获取客户端"""
        if service == AzureService.BLOB:
            storage_account = kwargs.get("storage_account")
            if not storage_account:
                raise ValueError("storage_account is required for Blob service")
            return self.blob(storage_account)
        elif service == AzureService.VM:
            return self.vm
        elif service == AzureService.FUNCTIONS:
            return self.functions
        else:
            raise ValueError(f"Unsupported service: {service}")
    
    def close(self) -> None:
        """关闭所有客户端连接"""
        self._blob_client = None
        self._vm_client = None
        self._functions_client = None


def create_azure_client(subscription_id: str, tenant_id: str,
                        client_id: str, client_secret: str,
                        resource_group: str, region: str = "eastus",
                        **kwargs) -> AzureSDKWrapper:
    """创建Azure客户端的便捷函数"""
    region_enum = AzureRegion(region)
    config = AzureConfig(
        subscription_id=subscription_id,
        tenant_id=tenant_id,
        client_id=client_id,
        client_secret=client_secret,
        resource_group=resource_group,
        region=region_enum,
        **kwargs
    )
    return AzureSDKWrapper(config)
