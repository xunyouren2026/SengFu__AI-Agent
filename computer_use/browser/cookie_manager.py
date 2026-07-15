"""
Cookie Manager Module

Manages browser cookies including persistence, import/export,
and profile-based cookie management.
"""

from typing import Optional, List, Dict, Any, Union
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
import json
import logging

logger = logging.getLogger(__name__)


@dataclass
class CookieProfile:
    """
    Represents a single cookie with all its attributes.
    
    Attributes:
        name: Cookie name
        value: Cookie value
        domain: Cookie domain
        path: Cookie path
        expires: Expiration timestamp (Unix epoch) or None for session cookies
        httpOnly: Whether cookie is HTTP-only
        secure: Whether cookie requires HTTPS
        sameSite: SameSite attribute (Strict, Lax, None)
    """
    name: str
    value: str
    domain: str
    path: str = "/"
    expires: Optional[float] = None
    httpOnly: bool = False
    secure: bool = False
    sameSite: str = "Lax"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert cookie to dictionary."""
        return {
            "name": self.name,
            "value": self.value,
            "domain": self.domain,
            "path": self.path,
            "expires": self.expires,
            "httpOnly": self.httpOnly,
            "secure": self.secure,
            "sameSite": self.sameSite
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CookieProfile':
        """Create CookieProfile from dictionary."""
        return cls(
            name=data.get("name", ""),
            value=data.get("value", ""),
            domain=data.get("domain", ""),
            path=data.get("path", "/"),
            expires=data.get("expires"),
            httpOnly=data.get("httpOnly", False),
            secure=data.get("secure", False),
            sameSite=data.get("sameSite", "Lax")
        )
    
    def is_expired(self) -> bool:
        """Check if cookie has expired."""
        if self.expires is None:
            return False
        return datetime.now(timezone.utc).timestamp() > self.expires
    
    def is_session_cookie(self) -> bool:
        """Check if this is a session cookie."""
        return self.expires is None
    
    def get_expiration_date(self) -> Optional[datetime]:
        """Get expiration date as datetime object."""
        if self.expires is None:
            return None
        return datetime.fromtimestamp(self.expires, tz=timezone.utc)


class CookieManager:
    """
    Manages browser cookies for persistence and reuse.
    
    Provides functionality to save, load, import, and export cookies
    between browser sessions.
    
    Example:
        manager = CookieManager()
        await manager.save_cookies(context, "/path/to/cookies.json")
        await manager.load_cookies(context, "/path/to/cookies.json")
    """
    
    def __init__(self):
        """Initialize the cookie manager."""
        self._cookies: List[CookieProfile] = []
        logger.info("CookieManager initialized")
    
    async def save_cookies(self, context: Any, file_path: Union[str, Path]) -> int:
        """
        Save cookies from a browser context to a file.
        
        Args:
            context: Playwright browser context
            file_path: Path to save cookies
            
        Returns:
            Number of cookies saved
        """
        file_path = Path(file_path)
        
        # Get cookies from context
        cookies = await context.cookies()
        
        # Convert to CookieProfile objects
        self._cookies = [CookieProfile(
            name=c.get("name", ""),
            value=c.get("value", ""),
            domain=c.get("domain", ""),
            path=c.get("path", "/"),
            expires=c.get("expires"),
            httpOnly=c.get("httpOnly", False),
            secure=c.get("secure", False),
            sameSite=c.get("sameSite", "Lax")
        ) for c in cookies]
        
        # Save to file
        data = {
            "saved_at": datetime.now(timezone.utc).isoformat(),
            "cookie_count": len(self._cookies),
            "cookies": [c.to_dict() for c in self._cookies]
        }
        
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved {len(self._cookies)} cookies to {file_path}")
        return len(self._cookies)
    
    async def load_cookies(self, context: Any, file_path: Union[str, Path],
                          filter_expired: bool = True) -> int:
        """
        Load cookies from a file into a browser context.
        
        Args:
            context: Playwright browser context
            file_path: Path to load cookies from
            filter_expired: Whether to filter out expired cookies
            
        Returns:
            Number of cookies loaded
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            logger.warning(f"Cookie file not found: {file_path}")
            return 0
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Parse cookies
        cookie_dicts = data.get("cookies", [])
        self._cookies = [CookieProfile.from_dict(c) for c in cookie_dicts]
        
        # Filter expired if requested
        if filter_expired:
            self._cookies = [c for c in self._cookies if not c.is_expired()]
        
        # Convert to Playwright format and add to context
        cookies_to_add = []
        for cookie in self._cookies:
            cookie_dict = cookie.to_dict()
            # Playwright expects expires as Unix timestamp or -1 for session
            if cookie_dict["expires"] is None:
                cookie_dict["expires"] = -1
            cookies_to_add.append(cookie_dict)
        
        await context.add_cookies(cookies_to_add)
        
        logger.info(f"Loaded {len(cookies_to_add)} cookies from {file_path}")
        return len(cookies_to_add)
    
    def export_cookies(self, format: str = "json") -> str:
        """
        Export cookies to a string in the specified format.
        
        Args:
            format: Export format ("json", "netscape", "header")
            
        Returns:
            Exported cookie string
        """
        if format == "json":
            return json.dumps({
                "cookies": [c.to_dict() for c in self._cookies]
            }, indent=2, ensure_ascii=False)
        
        elif format == "netscape":
            # Netscape format used by curl/wget
            lines = ["# Netscape HTTP Cookie File"]
            for c in self._cookies:
                domain = c.domain if c.domain.startswith('.') else '.' + c.domain
                flag = "TRUE" if domain.startswith('.') else "FALSE"
                secure = "TRUE" if c.secure else "FALSE"
                expires = str(int(c.expires)) if c.expires else "0"
                lines.append(f"{domain}\t{flag}\t{c.path}\t{secure}\t{expires}\t{c.name}\t{c.value}")
            return "\n".join(lines)
        
        elif format == "header":
            # HTTP Cookie header format
            cookie_pairs = [f"{c.name}={c.value}" for c in self._cookies]
            return "Cookie: " + "; ".join(cookie_pairs)
        
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def import_cookies(self, cookies: Union[str, List[Dict[str, Any]]], 
                      format: str = "json") -> int:
        """
        Import cookies from a string or list.
        
        Args:
            cookies: Cookie data (JSON string or list of dicts)
            format: Import format ("json", "netscape")
            
        Returns:
            Number of cookies imported
        """
        if format == "json":
            if isinstance(cookies, str):
                data = json.loads(cookies)
                if isinstance(data, dict):
                    cookie_list = data.get("cookies", [])
                else:
                    cookie_list = data
            else:
                cookie_list = cookies
            
            self._cookies = [CookieProfile.from_dict(c) for c in cookie_list]
        
        elif format == "netscape":
            # Parse Netscape format
            self._cookies = []
            for line in cookies.split('\n'):
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                parts = line.split('\t')
                if len(parts) >= 7:
                    domain, _, path, secure, expires, name, value = parts[:7]
                    self._cookies.append(CookieProfile(
                        name=name,
                        value=value,
                        domain=domain.lstrip('.'),
                        path=path,
                        expires=float(expires) if expires != "0" else None,
                        secure=secure == "TRUE",
                        httpOnly=False
                    ))
        
        else:
            raise ValueError(f"Unsupported import format: {format}")
        
        logger.info(f"Imported {len(self._cookies)} cookies")
        return len(self._cookies)
    
    def get_cookies(self, domain: Optional[str] = None) -> List[CookieProfile]:
        """
        Get cookies, optionally filtered by domain.
        
        Args:
            domain: Optional domain to filter by
            
        Returns:
            List of CookieProfile objects
        """
        if domain:
            return [c for c in self._cookies if domain in c.domain or c.domain in domain]
        return self._cookies.copy()
    
    def get_cookie(self, name: str, domain: Optional[str] = None) -> Optional[CookieProfile]:
        """
        Get a specific cookie by name.
        
        Args:
            name: Cookie name
            domain: Optional domain to match
            
        Returns:
            CookieProfile or None
        """
        for cookie in self._cookies:
            if cookie.name == name:
                if domain is None or domain in cookie.domain or cookie.domain in domain:
                    return cookie
        return None
    
    def add_cookie(self, cookie: CookieProfile) -> None:
        """
        Add a cookie to the manager.
        
        Args:
            cookie: CookieProfile to add
        """
        # Remove existing cookie with same name and domain
        self._cookies = [
            c for c in self._cookies 
            if not (c.name == cookie.name and c.domain == cookie.domain)
        ]
        self._cookies.append(cookie)
        logger.debug(f"Added cookie: {cookie.name}")
    
    def remove_cookie(self, name: str, domain: Optional[str] = None) -> bool:
        """
        Remove a cookie by name.
        
        Args:
            name: Cookie name
            domain: Optional domain to match
            
        Returns:
            True if cookie was removed
        """
        original_count = len(self._cookies)
        self._cookies = [
            c for c in self._cookies 
            if not (c.name == name and (domain is None or c.domain == domain))
        ]
        removed = len(self._cookies) < original_count
        if removed:
            logger.debug(f"Removed cookie: {name}")
        return removed
    
    def clear_cookies(self) -> None:
        """Clear all cookies."""
        self._cookies = []
        logger.info("Cleared all cookies")
    
    def get_domains(self) -> List[str]:
        """
        Get list of unique domains in cookies.
        
        Returns:
            List of domain strings
        """
        return list(set(c.domain for c in self._cookies))
    
    def filter_expired(self) -> int:
        """
        Remove expired cookies.
        
        Returns:
            Number of cookies removed
        """
        original_count = len(self._cookies)
        self._cookies = [c for c in self._cookies if not c.is_expired()]
        removed = original_count - len(self._cookies)
        if removed > 0:
            logger.info(f"Removed {removed} expired cookies")
        return removed
    
    def merge_cookies(self, other_cookies: List[CookieProfile], 
                      overwrite: bool = True) -> int:
        """
        Merge another set of cookies into this manager.
        
        Args:
            other_cookies: List of CookieProfile to merge
            overwrite: Whether to overwrite existing cookies
            
        Returns:
            Number of cookies added/updated
        """
        added = 0
        for cookie in other_cookies:
            existing = self.get_cookie(cookie.name, cookie.domain)
            if existing is None:
                self._cookies.append(cookie)
                added += 1
            elif overwrite:
                self.remove_cookie(cookie.name, cookie.domain)
                self._cookies.append(cookie)
                added += 1
        
        logger.info(f"Merged {added} cookies")
        return added
    
    def to_playwright_format(self) -> List[Dict[str, Any]]:
        """
        Convert cookies to Playwright format.
        
        Returns:
            List of cookie dictionaries for Playwright
        """
        result = []
        for cookie in self._cookies:
            cookie_dict = cookie.to_dict()
            # Playwright uses -1 for session cookies
            if cookie_dict["expires"] is None:
                cookie_dict["expires"] = -1
            result.append(cookie_dict)
        return result
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about stored cookies.
        
        Returns:
            Dictionary with cookie statistics
        """
        expired = sum(1 for c in self._cookies if c.is_expired())
        session = sum(1 for c in self._cookies if c.is_session_cookie())
        secure = sum(1 for c in self._cookies if c.secure)
        http_only = sum(1 for c in self._cookies if c.httpOnly)
        
        return {
            "total": len(self._cookies),
            "expired": expired,
            "session": session,
            "secure": secure,
            "http_only": http_only,
            "domains": len(self.get_domains())
        }
