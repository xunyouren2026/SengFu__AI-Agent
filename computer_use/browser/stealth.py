"""
Stealth Module

Provides anti-detection capabilities for browser automation.
Includes patches for navigator properties, viewport randomization,
and detection script blocking.
"""

from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
import random
import logging

logger = logging.getLogger(__name__)


@dataclass
class StealthConfig:
    """
    Configuration for anti-detection measures.
    
    Attributes:
        n_webdriver: Whether to hide navigator.webdriver
        navigator_languages: List of languages for navigator.languages
        navigator_plugins: List of plugin names to simulate
        permissions: List of permissions to grant
        avoid_detection: Master switch for all anti-detection measures
        randomize_viewport: Whether to randomize viewport slightly
        block_detection_scripts: Whether to block known detection scripts
        mask_webgl: Whether to mask WebGL fingerprint
        mask_canvas: Whether to mask Canvas fingerprint
        mask_fonts: Whether to mask font fingerprint
    """
    n_webdriver: bool = True
    navigator_languages: List[str] = field(default_factory=lambda: ["en-US", "en"])
    navigator_plugins: List[str] = field(default_factory=lambda: [
        "Chrome PDF Plugin",
        "Chrome PDF Viewer",
        "Native Client"
    ])
    permissions: List[str] = field(default_factory=list)
    avoid_detection: bool = True
    randomize_viewport: bool = True
    block_detection_scripts: bool = True
    mask_webgl: bool = True
    mask_canvas: bool = True
    mask_fonts: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "n_webdriver": self.n_webdriver,
            "navigator_languages": self.navigator_languages,
            "navigator_plugins": self.navigator_plugins,
            "permissions": self.permissions,
            "avoid_detection": self.avoid_detection,
            "randomize_viewport": self.randomize_viewport,
            "block_detection_scripts": self.block_detection_scripts,
            "mask_webgl": self.mask_webgl,
            "mask_canvas": self.mask_canvas,
            "mask_fonts": self.mask_fonts
        }


class StealthManager:
    """
    Manages anti-detection measures for browser automation.
    
    Applies various patches and configurations to make automated
    browser sessions appear more like regular user sessions.
    
    Example:
        config = StealthConfig(avoid_detection=True)
        stealth = StealthManager(config)
        await stealth.apply_stealth(context)
    """
    
    # Common user agents for rotation
    USER_AGENTS = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.0",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15",
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ]
    
    # Detection scripts to block
    DETECTION_SCRIPTS = [
        "botdetect",
        "captcha",
        "fingerprint",
        "bot-check",
        "automation-detection",
        "headless-detection"
    ]
    
    def __init__(self, config: Optional[StealthConfig] = None):
        """
        Initialize the stealth manager.
        
        Args:
            config: Stealth configuration (uses default if None)
        """
        self._config = config or StealthConfig()
        logger.info("StealthManager initialized")
    
    async def apply_stealth(self, context: Any) -> None:
        """
        Apply all configured anti-detection measures to a context.
        
        Args:
            context: Playwright browser context
        """
        if not self._config.avoid_detection:
            logger.info("Anti-detection disabled, skipping stealth measures")
            return
        
        logger.info("Applying stealth measures...")
        
        # Apply patches in order
        await self.patch_navigator(context)
        await self.patch_user_agent(context)
        await self.patch_webdriver(context)
        await self.patch_plugins(context)
        await self.patch_permissions(context)
        
        if self._config.randomize_viewport:
            await self.randomize_viewport(context)
        
        if self._config.mask_webgl:
            await self.mask_webgl(context)
        
        if self._config.mask_canvas:
            await self.mask_canvas(context)
        
        if self._config.block_detection_scripts:
            await self.block_detection_scripts(context)
        
        logger.info("Stealth measures applied successfully")
    
    async def patch_navigator(self, context: Any) -> None:
        """
        Patch navigator object to hide automation indicators.
        
        Args:
            context: Playwright browser context
        """
        script = f"""
        () => {{
            // Override navigator properties
            Object.defineProperty(navigator, 'webdriver', {{
                get: () => undefined
            }});
            
            // Override languages
            Object.defineProperty(navigator, 'languages', {{
                get: () => {self._config.navigator_languages}
            }});
            
            // Override platform
            Object.defineProperty(navigator, 'platform', {{
                get: () => 'Win32'
            }});
            
            // Override vendor
            Object.defineProperty(navigator, 'vendor', {{
                get: () => 'Google Inc.'
            }});
            
            // Override deviceMemory
            Object.defineProperty(navigator, 'deviceMemory', {{
                get: () => 8
            }});
            
            // Override hardwareConcurrency
            Object.defineProperty(navigator, 'hardwareConcurrency', {{
                get: () => 4
            }});
            
            // Override maxTouchPoints
            Object.defineProperty(navigator, 'maxTouchPoints', {{
                get: () => 0
            }});
        }}
        """
        
        await context.add_init_script(script)
        logger.info("Navigator patched")
    
    async def patch_user_agent(self, context: Any) -> None:
        """
        Patch User-Agent string.
        
        Args:
            context: Playwright browser context
        """
        user_agent = random.choice(self.USER_AGENTS)
        await context.set_extra_http_headers({"User-Agent": user_agent})
        logger.info(f"User-Agent set: {user_agent[:50]}...")
    
    async def patch_webdriver(self, context: Any) -> None:
        """
        Patch webdriver property to hide automation.
        
        Args:
            context: Playwright browser context
        """
        script = """
        () => {
            // Remove webdriver property entirely
            delete Object.getPrototypeOf(navigator).webdriver;
            
            // Override permissions query to hide automation
            const originalQuery = window.navigator.permissions.query;
            window.navigator.permissions.query = (parameters) => (
                parameters.name === 'notifications' ?
                    Promise.resolve({ state: Notification.permission }) :
                    originalQuery(parameters)
            );
        }
        """
        
        await context.add_init_script(script)
        logger.info("Webdriver property patched")
    
    async def patch_plugins(self, context: Any) -> None:
        """
        Patch plugins to simulate real browser.
        
        Args:
            context: Playwright browser context
        """
        plugins_js = str(self._config.navigator_plugins).replace("'", '"')
        
        script = f"""
        () => {{
            // Create fake plugins
            const plugins = {plugins_js}.map((name, index) => ({{
                name: name,
                filename: name.toLowerCase().replace(/\\s+/g, '_') + '.dll',
                description: name,
                version: undefined,
                length: 0,
                item: () => null,
                namedItem: () => null
            }}));
            
            Object.defineProperty(navigator, 'plugins', {{
                get: () => plugins
            }});
            
            Object.defineProperty(navigator, 'mimeTypes', {{
                get: () => [
                    {{ type: 'application/pdf', suffixes: 'pdf', description: 'Portable Document Format' }},
                    {{ type: 'application/x-google-chrome-pdf', suffixes: 'pdf', description: 'Portable Document Format' }}
                ]
            }});
        }}
        """
        
        await context.add_init_script(script)
        logger.info("Plugins patched")
    
    async def patch_permissions(self, context: Any) -> None:
        """
        Grant permissions to avoid permission prompts.
        
        Args:
            context: Playwright browser context
        """
        if self._config.permissions:
            await context.grant_permissions(self._config.permissions)
            logger.info(f"Permissions granted: {self._config.permissions}")
    
    async def randomize_viewport(self, context: Any) -> None:
        """
        Randomize viewport size slightly to avoid fingerprinting.
        
        Args:
            context: Playwright browser context
        """
        # Base viewport sizes
        viewports = [
            {"width": 1920, "height": 1080},
            {"width": 1366, "height": 768},
            {"width": 1440, "height": 900},
            {"width": 1536, "height": 864},
            {"width": 1280, "height": 720}
        ]
        
        base_viewport = random.choice(viewports)
        
        # Add small random variation (within 10 pixels)
        width = base_viewport["width"] + random.randint(-10, 10)
        height = base_viewport["height"] + random.randint(-10, 10)
        
        await context.set_viewport_size({"width": width, "height": height})
        logger.info(f"Viewport randomized to {width}x{height}")
    
    async def mask_webgl(self, context: Any) -> None:
        """
        Mask WebGL fingerprint.
        
        Args:
            context: Playwright browser context
        """
        script = """
        () => {
            const getParameter = WebGLRenderingContext.prototype.getParameter;
            WebGLRenderingContext.prototype.getParameter = function(parameter) {
                // UNMASKED_VENDOR_WEBGL
                if (parameter === 37445) {
                    return 'Intel Inc.';
                }
                // UNMASKED_RENDERER_WEBGL
                if (parameter === 37446) {
                    return 'Intel Iris OpenGL Engine';
                }
                return getParameter(parameter);
            };
            
            // Also patch WebGL2
            if (window.WebGL2RenderingContext) {
                WebGL2RenderingContext.prototype.getParameter = WebGLRenderingContext.prototype.getParameter;
            }
        }
        """
        
        await context.add_init_script(script)
        logger.info("WebGL masked")
    
    async def mask_canvas(self, context: Any) -> None:
        """
        Mask Canvas fingerprint by adding subtle noise.
        
        Args:
            context: Playwright browser context
        """
        script = """
        () => {
            const originalToDataURL = HTMLCanvasElement.prototype.toDataURL;
            const originalGetImageData = CanvasRenderingContext2D.prototype.getImageData;
            
            // Add subtle noise to canvas data
            CanvasRenderingContext2D.prototype.getImageData = function(...args) {
                const imageData = originalGetImageData.apply(this, args);
                
                // Add imperceptible noise to random pixels
                for (let i = 0; i < imageData.data.length; i += 4) {
                    if (Math.random() < 0.01) {
                        const noise = Math.floor(Math.random() * 2) - 1;
                        imageData.data[i] = Math.max(0, Math.min(255, imageData.data[i] + noise));
                        imageData.data[i + 1] = Math.max(0, Math.min(255, imageData.data[i + 1] + noise));
                        imageData.data[i + 2] = Math.max(0, Math.min(255, imageData.data[i + 2] + noise));
                    }
                }
                
                return imageData;
            };
        }
        """
        
        await context.add_init_script(script)
        logger.info("Canvas masked")
    
    async def block_detection_scripts(self, context: Any) -> None:
        """
        Block known detection scripts.
        
        Args:
            context: Playwright browser context
        """
        # Create route handler to block detection scripts
        async def handle_route(route, request):
            url = request.url.lower()
            
            # Check if URL contains detection keywords
            for keyword in self.DETECTION_SCRIPTS:
                if keyword in url:
                    logger.debug(f"Blocked detection script: {url}")
                    await route.abort("blockedbyclient")
                    return
            
            await route.continue_()
        
        await context.route("**/*", handle_route)
        logger.info("Detection script blocking enabled")
    
    def get_stealth_scripts(self) -> List[str]:
        """
        Get all stealth scripts as a list for manual application.
        
        Returns:
            List of JavaScript scripts
        """
        scripts = []
        
        if self._config.n_webdriver:
            scripts.append("""
                Object.defineProperty(navigator, 'webdriver', { get: () => undefined });
            """)
        
        scripts.append(f"""
            Object.defineProperty(navigator, 'languages', {{ get: () => {self._config.navigator_languages} }});
        """)
        
        scripts.append("""
            Object.defineProperty(navigator, 'plugins', { 
                get: () => [
                    {name: 'Chrome PDF Plugin'},
                    {name: 'Chrome PDF Viewer'},
                    {name: 'Native Client'}
                ] 
            });
        """)
        
        return scripts
    
    async def apply_to_page(self, page: Any) -> None:
        """
        Apply stealth measures to a single page.
        
        Args:
            page: Playwright page object
        """
        if not self._config.avoid_detection:
            return
        
        # Apply scripts directly to page
        scripts = self.get_stealth_scripts()
        for script in scripts:
            await page.evaluate(script)
        
        logger.info("Stealth measures applied to page")
