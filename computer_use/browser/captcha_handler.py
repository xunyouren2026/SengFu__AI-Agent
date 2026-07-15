"""
Captcha Handler Module

Provides detection and solving capabilities for various types of CAPTCHAs
including image-based, reCAPTCHA, and hCaptcha.
"""

from typing import Optional, Dict, Any, Union, Callable, List
from dataclasses import dataclass
from enum import Enum
from abc import ABC, abstractmethod
import base64
import logging

logger = logging.getLogger(__name__)


class CaptchaType(Enum):
    """Types of CAPTCHAs that can be handled."""
    IMAGE = "image"           # Standard image CAPTCHA
    RECAPTCHA_V2 = "recaptcha_v2"     # Google reCAPTCHA v2
    RECAPTCHA_V3 = "recaptcha_v3"     # Google reCAPTCHA v3
    HCAPTCHA = "hcaptcha"     # hCaptcha
    SLIDER = "slider"         # Slider CAPTCHA
    TILE = "tile"             # Tile/image selection CAPTCHA
    TEXT = "text"             # Text-based CAPTCHA
    AUDIO = "audio"           # Audio CAPTCHA
    UNKNOWN = "unknown"       # Unknown/undetected type


@dataclass
class CaptchaInfo:
    """
    Information about a detected CAPTCHA.
    
    Attributes:
        type: Type of CAPTCHA
        selector: CSS selector for the CAPTCHA element
        site_key: Site key for reCAPTCHA/hCaptcha
        is_present: Whether a CAPTCHA is present
        confidence: Detection confidence (0.0-1.0)
        metadata: Additional metadata
    """
    type: CaptchaType
    selector: Optional[str] = None
    site_key: Optional[str] = None
    is_present: bool = False
    confidence: float = 0.0
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class CaptchaSolver(ABC):
    """Abstract base class for CAPTCHA solving services."""
    
    @abstractmethod
    async def solve_image(self, image_data: Union[str, bytes]) -> str:
        """Solve an image CAPTCHA."""
        pass
    
    @abstractmethod
    async def solve_recaptcha(self, site_key: str, url: str) -> str:
        """Solve a reCAPTCHA."""
        pass
    
    @abstractmethod
    async def solve_hcaptcha(self, site_key: str, url: str) -> str:
        """Solve an hCaptcha."""
        pass


class OCRSolver(CaptchaSolver):
    """OCR-based CAPTCHA solver using standard library only."""
    
    def __init__(self):
        """Initialize OCR solver."""
        logger.info("OCRSolver initialized")
    
    async def solve_image(self, image_data: Union[str, bytes]) -> str:
        """
        Attempt to solve image CAPTCHA using OCR.
        
        Note: This is a placeholder implementation.
        Full implementation would require OCR library like pytesseract.
        
        Args:
            image_data: Image data (base64 string or bytes)
            
        Returns:
            Recognized text (placeholder)
        """
        logger.warning("OCR solving not fully implemented - requires OCR library")
        return ""
    
    async def solve_recaptcha(self, site_key: str, url: str) -> str:
        """OCR cannot solve reCAPTCHA - returns empty token with warning."""
        logger.warning(
            "OCR solver cannot solve reCAPTCHA (site_key=%s, url=%s). "
            "Use a CAPTCHA solving service (e.g., 2Captcha) instead.",
            site_key, url,
        )
        return ""
    
    async def solve_hcaptcha(self, site_key: str, url: str) -> str:
        """OCR cannot solve hCaptcha - returns empty token with warning."""
        logger.warning(
            "OCR solver cannot solve hCaptcha (site_key=%s, url=%s). "
            "Use a CAPTCHA solving service (e.g., 2Captcha) instead.",
            site_key, url,
        )
        return ""


class TwoCaptchaSolver(CaptchaSolver):
    """2Captcha service integration."""
    
    def __init__(self, api_key: str):
        """
        Initialize 2Captcha solver.
        
        Args:
            api_key: 2Captcha API key
        """
        self._api_key = api_key
        self._base_url = "http://2captcha.com"
        logger.info("TwoCaptchaSolver initialized")
    
    async def solve_image(self, image_data: Union[str, bytes]) -> str:
        """
        Submit image CAPTCHA to 2Captcha.
        
        Args:
            image_data: Image data (base64 string or bytes)
            
        Returns:
            CAPTCHA solution
        """
        # Convert to base64 if needed
        if isinstance(image_data, bytes):
            image_b64 = base64.b64encode(image_data).decode('utf-8')
        else:
            image_b64 = image_data
        
        logger.info("Submitting image CAPTCHA to 2Captcha")
        
        # Note: Full implementation would make HTTP request to 2Captcha API
        # This is a placeholder showing the structure
        return await self._submit_and_wait("in.php", {
            "method": "base64",
            "body": image_b64
        })
    
    async def solve_recaptcha(self, site_key: str, url: str) -> str:
        """
        Submit reCAPTCHA to 2Captcha.
        
        Args:
            site_key: reCAPTCHA site key
            url: Page URL
            
        Returns:
            CAPTCHA solution token
        """
        logger.info(f"Submitting reCAPTCHA to 2Captcha for {url}")
        
        return await self._submit_and_wait("in.php", {
            "method": "userrecaptcha",
            "googlekey": site_key,
            "pageurl": url
        })
    
    async def solve_hcaptcha(self, site_key: str, url: str) -> str:
        """
        Submit hCaptcha to 2Captcha.
        
        Args:
            site_key: hCaptcha site key
            url: Page URL
            
        Returns:
            CAPTCHA solution token
        """
        logger.info(f"Submitting hCaptcha to 2Captcha for {url}")
        
        return await self._submit_and_wait("in.php", {
            "method": "hcaptcha",
            "sitekey": site_key,
            "pageurl": url
        })
    
    async def _submit_and_wait(self, endpoint: str, params: Dict[str, str]) -> str:
        """
        Submit CAPTCHA and wait for solution.
        
        Args:
            endpoint: API endpoint
            params: Request parameters
            
        Returns:
            Solution string
        """
        # Placeholder for actual API implementation
        # Would use httpx or aiohttp for async HTTP requests
        logger.warning("2Captcha API integration requires HTTP library")
        return ""


class CaptchaHandler:
    """
    Handles CAPTCHA detection and solving.
    
    Provides methods to detect CAPTCHAs on pages and solve them
    using various methods (OCR, 2Captcha, etc.).
    
    Example:
        handler = CaptchaHandler()
        captcha = await handler.detect_captcha(page)
        if captcha.is_present:
            solution = await handler.solve_recaptcha(captcha.site_key, page.url)
    """
    
    # CAPTCHA detection patterns
    CAPTCHA_SELECTORS = {
        CaptchaType.RECAPTCHA_V2: [
            '.g-recaptcha',
            '[data-sitekey]',
            'iframe[src*="recaptcha"]'
        ],
        CaptchaType.HCAPTCHA: [
            '.h-captcha',
            '[data-hcaptcha-sitekey]',
            'iframe[src*="hcaptcha"]'
        ],
        CaptchaType.IMAGE: [
            'img[src*="captcha"]',
            'img[src*="verify"]',
            '.captcha-image',
            '#captcha'
        ],
        CaptchaType.SLIDER: [
            '.slider-captcha',
            '.slide-verify',
            '[class*="slider"]'
        ],
        CaptchaType.TEXT: [
            'input[name*="captcha"]',
            'input[placeholder*="captcha"]',
            '.captcha-input'
        ]
    }
    
    def __init__(self):
        """Initialize the CAPTCHA handler."""
        self._solvers: Dict[str, CaptchaSolver] = {}
        self._ocr_solver = OCRSolver()
        logger.info("CaptchaHandler initialized")
    
    async def detect_captcha(self, page: Any) -> CaptchaInfo:
        """
        Detect if a CAPTCHA is present on the page.
        
        Args:
            page: Playwright page object
            
        Returns:
            CaptchaInfo with detection results
        """
        # Check for reCAPTCHA
        for selector in self.CAPTCHA_SELECTORS[CaptchaType.RECAPTCHA_V2]:
            try:
                element = await page.query_selector(selector)
                if element:
                    site_key = await element.get_attribute('data-sitekey')
                    return CaptchaInfo(
                        type=CaptchaType.RECAPTCHA_V2,
                        selector=selector,
                        site_key=site_key,
                        is_present=True,
                        confidence=0.9
                    )
            except:
                continue
        
        # Check for hCaptcha
        for selector in self.CAPTCHA_SELECTORS[CaptchaType.HCAPTCHA]:
            try:
                element = await page.query_selector(selector)
                if element:
                    site_key = await element.get_attribute('data-hcaptcha-sitekey')
                    return CaptchaInfo(
                        type=CaptchaType.HCAPTCHA,
                        selector=selector,
                        site_key=site_key,
                        is_present=True,
                        confidence=0.9
                    )
            except:
                continue
        
        # Check for image CAPTCHA
        for selector in self.CAPTCHA_SELECTORS[CaptchaType.IMAGE]:
            try:
                element = await page.query_selector(selector)
                if element:
                    return CaptchaInfo(
                        type=CaptchaType.IMAGE,
                        selector=selector,
                        is_present=True,
                        confidence=0.7
                    )
            except:
                continue
        
        # Check for text CAPTCHA input
        for selector in self.CAPTCHA_SELECTORS[CaptchaType.TEXT]:
            try:
                element = await page.query_selector(selector)
                if element:
                    return CaptchaInfo(
                        type=CaptchaType.TEXT,
                        selector=selector,
                        is_present=True,
                        confidence=0.6
                    )
            except:
                continue
        
        return CaptchaInfo(type=CaptchaType.UNKNOWN, is_present=False)
    
    async def solve_with_ocr(self, image_selector: str, page: Any) -> str:
        """
        Attempt to solve image CAPTCHA using OCR.
        
        Args:
            image_selector: CSS selector for CAPTCHA image
            page: Playwright page object
            
        Returns:
            Recognized text
        """
        try:
            # Get image element
            element = await page.query_selector(image_selector)
            if not element:
                raise ValueError(f"CAPTCHA element not found: {image_selector}")
            
            # Take screenshot of element
            screenshot = await element.screenshot()
            
            # Solve using OCR
            solution = await self._ocr_solver.solve_image(screenshot)
            logger.info(f"OCR solution: {solution}")
            return solution
        except Exception as e:
            logger.error(f"OCR solving failed: {e}")
            return ""
    
    async def solve_with_2captcha(self, captcha_type: CaptchaType, 
                                  api_key: str, **kwargs) -> str:
        """
        Solve CAPTCHA using 2Captcha service.
        
        Args:
            captcha_type: Type of CAPTCHA
            api_key: 2Captcha API key
            **kwargs: Additional parameters (site_key, url, image_data, etc.)
            
        Returns:
            CAPTCHA solution
        """
        solver = self._get_solver("2captcha", api_key)
        
        if captcha_type == CaptchaType.IMAGE:
            image_data = kwargs.get("image_data")
            if not image_data:
                raise ValueError("image_data required for image CAPTCHA")
            return await solver.solve_image(image_data)
        
        elif captcha_type in (CaptchaType.RECAPTCHA_V2, CaptchaType.RECAPTCHA_V3):
            site_key = kwargs.get("site_key")
            url = kwargs.get("url")
            if not site_key or not url:
                raise ValueError("site_key and url required for reCAPTCHA")
            return await solver.solve_recaptcha(site_key, url)
        
        elif captcha_type == CaptchaType.HCAPTCHA:
            site_key = kwargs.get("site_key")
            url = kwargs.get("url")
            if not site_key or not url:
                raise ValueError("site_key and url required for hCaptcha")
            return await solver.solve_hcaptcha(site_key, url)
        
        else:
            raise ValueError(f"Unsupported CAPTCHA type for 2Captcha: {captcha_type}")
    
    async def solve_image_captcha(self, image_data: Union[str, bytes],
                                  method: str = "ocr") -> str:
        """
        Solve an image CAPTCHA.
        
        Args:
            image_data: Image data (base64 or bytes)
            method: Solving method ("ocr", "2captcha")
            
        Returns:
            CAPTCHA solution text
        """
        if method == "ocr":
            return await self._ocr_solver.solve_image(image_data)
        else:
            raise ValueError(f"Unknown solving method: {method}")
    
    async def solve_recaptcha(self, site_key: str, url: str,
                              api_key: Optional[str] = None) -> str:
        """
        Solve a reCAPTCHA challenge.
        
        Args:
            site_key: reCAPTCHA site key
            url: Page URL
            api_key: Optional 2Captcha API key
            
        Returns:
            Solution token
        """
        if api_key:
            return await self.solve_with_2captcha(
                CaptchaType.RECAPTCHA_V2, api_key, site_key=site_key, url=url
            )
        else:
            raise ValueError("API key required for reCAPTCHA solving")
    
    async def solve_hcaptcha(self, site_key: str, url: str,
                            api_key: Optional[str] = None) -> str:
        """
        Solve an hCaptcha challenge.
        
        Args:
            site_key: hCaptcha site key
            url: Page URL
            api_key: Optional 2Captcha API key
            
        Returns:
            Solution token
        """
        if api_key:
            return await self.solve_with_2captcha(
                CaptchaType.HCAPTCHA, api_key, site_key=site_key, url=url
            )
        else:
            raise ValueError("API key required for hCaptcha solving")
    
    def _get_solver(self, service: str, api_key: str) -> CaptchaSolver:
        """
        Get or create a solver for the specified service.
        
        Args:
            service: Service name ("2captcha")
            api_key: API key for the service
            
        Returns:
            CaptchaSolver instance
        """
        key = f"{service}:{api_key[:8]}"
        if key not in self._solvers:
            if service == "2captcha":
                self._solvers[key] = TwoCaptchaSolver(api_key)
            else:
                raise ValueError(f"Unknown service: {service}")
        return self._solvers[key]
    
    async def wait_for_captcha_solved(self, page: Any, 
                                     timeout: int = 120000) -> bool:
        """
        Wait for CAPTCHA to be solved (disappear from page).
        
        Args:
            page: Playwright page object
            timeout: Maximum wait time in milliseconds
            
        Returns:
            True if CAPTCHA was solved
        """
        import asyncio
        
        start_time = asyncio.get_event_loop().time()
        
        while (asyncio.get_event_loop().time() - start_time) * 1000 < timeout:
            info = await self.detect_captcha(page)
            if not info.is_present:
                logger.info("CAPTCHA solved successfully")
                return True
            await asyncio.sleep(2)
        
        logger.warning("Timeout waiting for CAPTCHA to be solved")
        return False
    
    def register_solver(self, name: str, solver: CaptchaSolver) -> None:
        """
        Register a custom CAPTCHA solver.
        
        Args:
            name: Solver name
            solver: CaptchaSolver instance
        """
        self._solvers[name] = solver
        logger.info(f"Registered solver: {name}")
