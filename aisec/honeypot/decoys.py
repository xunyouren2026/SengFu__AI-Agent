"""
蜜罐诱饵 - 蜜罐系统诱饵管理
"""
import time
import hashlib
import random
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum


class DecoyType(Enum):
    """诱饵类型"""
    FAKE_FILE = "fake_file"
    FAKE_CREDENTIAL = "fake_credential"
    FAKE_SERVICE = "fake_service"
    FAKE_DATABASE = "fake_database"
    FAKE_API = "fake_api"
    FAKE_USER = "fake_user"


@dataclass
class Decoy:
    """诱饵"""
    decoy_id: str
    decoy_type: DecoyType
    name: str
    content: Any
    created_at: float = field(default_factory=time.time)
    accessed: bool = False
    access_count: int = 0
    last_access: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DecoyEvent:
    """诱饵事件"""
    decoy_id: str
    event_type: str  # access, modify, delete
    timestamp: float
    source_ip: str = ""
    source_user: str = ""
    details: Dict[str, Any] = field(default_factory=dict)


class DecoyManager:
    """诱饵管理器"""
    
    def __init__(self):
        self._decoys: Dict[str, Decoy] = {}
        self._events: List[DecoyEvent] = []
        self._alert_callbacks: List[Callable[[DecoyEvent], None]] = []
        self._load_default_decoys()
    
    def _load_default_decoys(self) -> None:
        """加载默认诱饵"""
        # 假密码文件
        self.add_decoy(Decoy(
            decoy_id="fake_password_file",
            decoy_type=DecoyType.FAKE_FILE,
            name="/etc/shadow.backup",
            content="root:$6$fake$hash:0:0:root:/root:/bin/bash\nadmin:$6$fake$hash:0:0:admin:/home/admin:/bin/bash",
            metadata={"description": "假密码文件"}
        ))
        
        # 假API密钥
        self.add_decoy(Decoy(
            decoy_id="fake_api_key",
            decoy_type=DecoyType.FAKE_CREDENTIAL,
            name="AWS_ACCESS_KEY_ID",
            content="AKIAIOSFODNN7EXAMPLE",
            metadata={"description": "假AWS密钥"}
        ))
        
        # 假数据库连接
        self.add_decoy(Decoy(
            decoy_id="fake_db_connection",
            decoy_type=DecoyType.FAKE_DATABASE,
            name="production_db",
            content={
                "host": "10.0.0.100",
                "port": 3306,
                "user": "admin",
                "password": "Sup3rS3cr3tP@ssw0rd!",
                "database": "production"
            },
            metadata={"description": "假数据库连接信息"}
        ))
    
    def add_decoy(self, decoy: Decoy) -> None:
        """添加诱饵"""
        self._decoys[decoy.decoy_id] = decoy
    
    def create_fake_file(
        self,
        name: str,
        content: str,
        file_type: str = "text"
    ) -> Decoy:
        """创建假文件诱饵"""
        decoy_id = hashlib.md5(f"{name}{time.time()}".encode()).hexdigest()[:12]
        
        decoy = Decoy(
            decoy_id=decoy_id,
            decoy_type=DecoyType.FAKE_FILE,
            name=name,
            content=content,
            metadata={"file_type": file_type}
        )
        
        self.add_decoy(decoy)
        return decoy
    
    def create_fake_credential(
        self,
        service: str,
        username: str,
        password: str
    ) -> Decoy:
        """创建假凭据诱饵"""
        decoy_id = hashlib.md5(f"{service}{username}{time.time()}".encode()).hexdigest()[:12]
        
        decoy = Decoy(
            decoy_id=decoy_id,
            decoy_type=DecoyType.FAKE_CREDENTIAL,
            name=f"{service}_credential",
            content={"username": username, "password": password},
            metadata={"service": service}
        )
        
        self.add_decoy(decoy)
        return decoy
    
    def create_fake_service(
        self,
        service_name: str,
        port: int,
        banner: str
    ) -> Decoy:
        """创建假服务诱饵"""
        decoy_id = hashlib.md5(f"{service_name}{port}{time.time()}".encode()).hexdigest()[:12]
        
        decoy = Decoy(
            decoy_id=decoy_id,
            decoy_type=DecoyType.FAKE_SERVICE,
            name=service_name,
            content={"port": port, "banner": banner},
            metadata={"listen_port": port}
        )
        
        self.add_decoy(decoy)
        return decoy
    
    def record_access(
        self,
        decoy_id: str,
        source_ip: str = "",
        source_user: str = "",
        event_type: str = "access"
    ) -> DecoyEvent:
        """记录诱饵访问"""
        decoy = self._decoys.get(decoy_id)
        if not decoy:
            raise ValueError(f"诱饵不存在: {decoy_id}")
        
        # 更新诱饵状态
        decoy.accessed = True
        decoy.access_count += 1
        decoy.last_access = time.time()
        
        # 创建事件
        event = DecoyEvent(
            decoy_id=decoy_id,
            event_type=event_type,
            timestamp=time.time(),
            source_ip=source_ip,
            source_user=source_user
        )
        
        self._events.append(event)
        
        # 触发告警
        for callback in self._alert_callbacks:
            try:
                callback(event)
            except Exception:
                pass
        
        return event
    
    def add_alert_callback(self, callback: Callable[[DecoyEvent], None]) -> None:
        """添加告警回调"""
        self._alert_callbacks.append(callback)
    
    def get_decoy(self, decoy_id: str) -> Optional[Decoy]:
        """获取诱饵"""
        return self._decoys.get(decoy_id)
    
    def get_all_decoys(self) -> List[Decoy]:
        """获取所有诱饵"""
        return list(self._decoys.values())
    
    def get_decoys_by_type(self, decoy_type: DecoyType) -> List[Decoy]:
        """按类型获取诱饵"""
        return [d for d in self._decoys.values() if d.decoy_type == decoy_type]
    
    def get_events(self, decoy_id: str = None) -> List[DecoyEvent]:
        """获取事件"""
        if decoy_id:
            return [e for e in self._events if e.decoy_id == decoy_id]
        return self._events.copy()
    
    def get_triggered_decoys(self) -> List[Decoy]:
        """获取被触发的诱饵"""
        return [d for d in self._decoys.values() if d.accessed]
    
    def generate_honeytoken(self, token_type: str = "api_key") -> str:
        """生成蜜令牌"""
        if token_type == "api_key":
            # 生成假的API密钥
            prefix = random.choice(["sk-", "AKIA", "ghp_", "xoxb-"])
            body = ''.join(random.choices('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789', k=32))
            return prefix + body
        elif token_type == "password":
            # 生成假密码
            chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!@#$%^&*"
            return ''.join(random.choices(chars, k=16))
        elif token_type == "email":
            # 生成假邮箱
            usernames = ["admin", "root", "service", "backup", "deploy"]
            domains = ["internal.local", "corp.local", "admin.local"]
            return f"{random.choice(usernames)}@{random.choice(domains)}"
        
        return hashlib.md5(str(time.time()).encode()).hexdigest()
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "total_decoys": len(self._decoys),
            "triggered_decoys": len(self.get_triggered_decoys()),
            "total_events": len(self._events),
            "by_type": {
                t.value: len(self.get_decoys_by_type(t))
                for t in DecoyType
            }
        }
    
    def remove_decoy(self, decoy_id: str) -> bool:
        """移除诱饵"""
        if decoy_id in self._decoys:
            del self._decoys[decoy_id]
            return True
        return False
