"""
内容签名 - 内容完整性签名
"""
import hashlib
import time
import json
import hmac
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum


class SignatureAlgorithm(Enum):
    """签名算法"""
    SHA256 = "sha256"
    SHA512 = "sha512"
    HMAC_SHA256 = "hmac_sha256"
    BLAKE2B = "blake2b"


@dataclass
class ContentSignature:
    """内容签名"""
    signature_id: str
    content_hash: str
    algorithm: SignatureAlgorithm
    timestamp: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    signature: str = ""
    public_key_id: str = ""


@dataclass
class VerificationResult:
    """验证结果"""
    valid: bool
    signature_id: str
    content_hash: str
    timestamp: float
    age_seconds: float
    error: str = ""


class ContentSigner:
    """内容签名器"""
    
    def __init__(
        self,
        secret_key: Optional[bytes] = None,
        algorithm: SignatureAlgorithm = SignatureAlgorithm.SHA256
    ):
        self._secret_key = secret_key or b"default_secret_key_change_in_production"
        self._algorithm = algorithm
        self._signatures: Dict[str, ContentSignature] = {}
    
    def sign_content(
        self,
        content: bytes,
        metadata: Dict[str, Any] = None
    ) -> ContentSignature:
        """签名内容"""
        # 计算内容哈希
        content_hash = self._hash_content(content)
        
        # 生成签名ID
        signature_id = hashlib.sha256(
            f"{content_hash}{time.time()}".encode()
        ).hexdigest()[:16]
        
        # 生成签名
        signature = self._generate_signature(content_hash)
        
        sig = ContentSignature(
            signature_id=signature_id,
            content_hash=content_hash,
            algorithm=self._algorithm,
            timestamp=time.time(),
            metadata=metadata or {},
            signature=signature
        )
        
        self._signatures[signature_id] = sig
        return sig
    
    def sign_json(
        self,
        data: Dict[str, Any],
        fields_to_sign: List[str] = None
    ) -> ContentSignature:
        """签名JSON数据"""
        if fields_to_sign:
            # 只签名指定字段
            to_sign = {k: v for k, v in data.items() if k in fields_to_sign}
        else:
            to_sign = data
        
        content = json.dumps(to_sign, sort_keys=True).encode()
        return self.sign_content(content, {"type": "json", "fields": fields_to_sign})
    
    def sign_model_output(
        self,
        model_id: str,
        input_hash: str,
        output: Any
    ) -> ContentSignature:
        """签名模型输出"""
        content = json.dumps({
            "model_id": model_id,
            "input_hash": input_hash,
            "output": output,
            "timestamp": time.time()
        }, sort_keys=True).encode()
        
        return self.sign_content(content, {
            "type": "model_output",
            "model_id": model_id
        })
    
    def verify_content(
        self,
        content: bytes,
        signature: ContentSignature
    ) -> VerificationResult:
        """验证内容"""
        # 计算内容哈希
        content_hash = self._hash_content(content)
        
        # 检查哈希是否匹配
        if content_hash != signature.content_hash:
            return VerificationResult(
                valid=False,
                signature_id=signature.signature_id,
                content_hash=content_hash,
                timestamp=signature.timestamp,
                age_seconds=time.time() - signature.timestamp,
                error="Content hash mismatch"
            )
        
        # 验证签名
        expected_sig = self._generate_signature(content_hash)
        if expected_sig != signature.signature:
            return VerificationResult(
                valid=False,
                signature_id=signature.signature_id,
                content_hash=content_hash,
                timestamp=signature.timestamp,
                age_seconds=time.time() - signature.timestamp,
                error="Signature mismatch"
            )
        
        return VerificationResult(
            valid=True,
            signature_id=signature.signature_id,
            content_hash=content_hash,
            timestamp=signature.timestamp,
            age_seconds=time.time() - signature.timestamp
        )
    
    def verify_json(
        self,
        data: Dict[str, Any],
        signature: ContentSignature
    ) -> VerificationResult:
        """验证JSON数据"""
        fields = signature.metadata.get("fields")
        
        if fields:
            to_verify = {k: v for k, v in data.items() if k in fields}
        else:
            to_verify = data
        
        content = json.dumps(to_verify, sort_keys=True).encode()
        return self.verify_content(content, signature)
    
    def _hash_content(self, content: bytes) -> str:
        """计算内容哈希"""
        if self._algorithm == SignatureAlgorithm.SHA256:
            return hashlib.sha256(content).hexdigest()
        elif self._algorithm == SignatureAlgorithm.SHA512:
            return hashlib.sha512(content).hexdigest()
        elif self._algorithm == SignatureAlgorithm.BLAKE2B:
            return hashlib.blake2b(content).hexdigest()
        else:
            return hashlib.sha256(content).hexdigest()
    
    def _generate_signature(self, content_hash: str) -> str:
        """生成签名"""
        if self._algorithm == SignatureAlgorithm.HMAC_SHA256:
            return hmac.new(
                self._secret_key,
                content_hash.encode(),
                hashlib.sha256
            ).hexdigest()
        else:
            # 使用HMAC作为默认签名方法
            return hmac.new(
                self._secret_key,
                content_hash.encode(),
                hashlib.sha256
            ).hexdigest()
    
    def get_signature(self, signature_id: str) -> Optional[ContentSignature]:
        """获取签名"""
        return self._signatures.get(signature_id)
    
    def get_all_signatures(self) -> List[ContentSignature]:
        """获取所有签名"""
        return list(self._signatures.values())
    
    def create_chain(
        self,
        contents: List[bytes]
    ) -> List[ContentSignature]:
        """创建签名链"""
        signatures = []
        prev_hash = ""
        
        for content in contents:
            # 包含前一个哈希
            chained_content = prev_hash.encode() + content if prev_hash else content
            
            sig = self.sign_content(chained_content, {
                "chain_position": len(signatures),
                "prev_hash": prev_hash
            })
            
            signatures.append(sig)
            prev_hash = sig.content_hash
        
        return signatures
    
    def verify_chain(
        self,
        contents: List[bytes],
        signatures: List[ContentSignature]
    ) -> Tuple[bool, List[VerificationResult]]:
        """验证签名链"""
        import hmac
        results = []
        all_valid = True
        
        for i, (content, sig) in enumerate(zip(contents, signatures)):
            # 获取前一个哈希
            prev_hash = signatures[i - 1].content_hash if i > 0 else ""
            
            # 构建链式内容
            chained_content = prev_hash.encode() + content if prev_hash else content
            
            result = self.verify_content(chained_content, sig)
            results.append(result)
            
            if not result.valid:
                all_valid = False
        
        return all_valid, results
    
    def export_signature(self, signature: ContentSignature) -> str:
        """导出签名"""
        return json.dumps({
            "signature_id": signature.signature_id,
            "content_hash": signature.content_hash,
            "algorithm": signature.algorithm.value,
            "timestamp": signature.timestamp,
            "metadata": signature.metadata,
            "signature": signature.signature
        }, indent=2)
    
    def import_signature(self, data: str) -> ContentSignature:
        """导入签名"""
        obj = json.loads(data)
        
        return ContentSignature(
            signature_id=obj["signature_id"],
            content_hash=obj["content_hash"],
            algorithm=SignatureAlgorithm(obj["algorithm"]),
            timestamp=obj["timestamp"],
            metadata=obj.get("metadata", {}),
            signature=obj.get("signature", "")
        )


from typing import Tuple
