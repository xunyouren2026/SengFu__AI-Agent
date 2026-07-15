"""
Playwright Manager Module

Provides comprehensive browser lifecycle management including:
- Browser launch and configuration
- Context and page management
- Standard Playwright API encapsulation with extension points
"""

from typing import Optional, Dict, Any, List, Union
from dataclasses import dataclass, field
from enum import Enum
import asyncio
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BrowserChannel(Enum):
    """Supported browser channels."""
    CHROME = "chrome"
    CHROMIUM = "chromium"
    FIREFOX = "firefox"
    WEBKIT = "webkit"
    EDGE = "msedge"


@dataclass
class BrowserConfig:
    """Configuration for browser launch."""
    headless: bool = True
    channel: Optional[str] = None
    args: List[str] = field(default_factory=list)
    timeout: int = 30000
    slow_mo: int = 0
    downloads_path: Optional[str] = None
    traces_dir: Optional[str] = None


@dataclass
class ViewportConfig:
    """Viewport configuration."""
    width: int = 1920
    height: int = 1080
    device_scale_factor: float = 1.0
    is_mobile: bool = False
    has_touch: bool = False


class BrowserContext:
    """
    Browser context wrapper providing enhanced context management.
    
    A browser context is an isolated environment within a browser instance.
    Multiple contexts can exist within a single browser, each with separate
    cookies, local storage, etc.
    """
    
    def __init__(self, playwright_context: Any, config: Optional[Dict[str, Any]] = None):
        """
        Initialize browser context wrapper.
        
        Args:
            playwright_context: The underlying Playwright context object
            config: Optional configuration dictionary
        """
        self._context = playwright_context
        self._config = config or {}
        self._pages: List['BrowserPage'] = []
        logger.info("BrowserContext initialized")
    
    @property
    def context(self) -> Any:
        """Get the underlying Playwright context."""
        return self._context
    
    async def set_viewport_size(self, width: int, height: int) -> None:
        """
        Set the viewport size for all pages in this context.
        
        Args:
            width: Viewport width in pixels
            height: Viewport height in pixels
        """
        await self._context.set_viewport_size({"width": width, "height": height})
        logger.info(f"Viewport size set to {width}x{height}")
    
    async def set_user_agent(self, user_agent: str) -> None:
        """
        Set the User-Agent string for this context.
        
        Args:
            user_agent: The User-Agent string to use
        """
        await self._context.set_extra_http_headers({"User-Agent": user_agent})
        logger.info(f"User-Agent set: {user_agent[:50]}...")
    
    async def add_cookies(self, cookies: List[Dict[str, Any]]) -> None:
        """
        Add cookies to this context.
        
        Args:
            cookies: List of cookie dictionaries with name, value, domain, etc.
        """
        await self._context.add_cookies(cookies)
        logger.info(f"Added {len(cookies)} cookies")
    
    async def get_cookies(self, urls: Optional[Union[str, List[str]]] = None) -> List[Dict[str, Any]]:
        """
        Get cookies from this context.
        
        Args:
            urls: Optional URL or list of URLs to filter cookies
            
        Returns:
            List of cookie dictionaries
        """
        if urls:
            return await self._context.cookies(urls)
        return await self._context.cookies()
    
    async def clear_cookies(self) -> None:
        """Clear all cookies in this context."""
        await self._context.clear_cookies()
        logger.info("Cookies cleared")
    
    async def grant_permissions(self, permissions: List[str], origin: Optional[str] = None) -> None:
        """
        Grant permissions to this context.
        
        Args:
            permissions: List of permissions to grant (e.g., ['geolocation', 'notifications'])
            origin: Optional origin URL to grant permissions for
        """
        await self._context.grant_permissions(permissions, origin=origin)
        logger.info(f"Granted permissions: {permissions}")
    
    async def clear_permissions(self) -> None:
        """Clear all granted permissions."""
        await self._context.clear_permissions()
        logger.info("Permissions cleared")
    
    async def new_page(self) -> 'BrowserPage':
        """
        Create a new page in this context.
        
        Returns:
            BrowserPage wrapper instance
        """
        page = await self._context.new_page()
        browser_page = BrowserPage(page, self)
        self._pages.append(browser_page)
        logger.info("New page created in context")
        return browser_page
    
    async def close(self) -> None:
        """Close this browser context."""
        for page in self._pages:
            await page.close()
        await self._context.close()
        logger.info("BrowserContext closed")


class BrowserPage:
    """
    Browser page wrapper providing enhanced page interaction methods.
    
    Wraps a Playwright page object and provides convenient methods for
    common browser automation tasks.
    """
    
    def __init__(self, playwright_page: Any, context: BrowserContext):
        """
        Initialize browser page wrapper.
        
        Args:
            playwright_page: The underlying Playwright page object
            context: The parent BrowserContext
        """
        self._page = playwright_page
        self._context = context
        logger.info("BrowserPage initialized")
    
    @property
    def page(self) -> Any:
        """Get the underlying Playwright page."""
        return self._page
    
    @property
    def url(self) -> str:
        """Get the current page URL."""
        return self._page.url
    
    @property
    def title(self) -> str:
        """Get the current page title."""
        return self._page.title()
    
    async def goto(self, url: str, wait_until: str = "load", timeout: Optional[int] = None) -> None:
        """
        Navigate to a URL.
        
        Args:
            url: The URL to navigate to
            wait_until: When to consider navigation complete ('load', 'domcontentloaded', 'networkidle')
            timeout: Navigation timeout in milliseconds
        """
        kwargs = {"wait_until": wait_until}
        if timeout:
            kwargs["timeout"] = timeout
        
        response = await self._page.goto(url, **kwargs)
        logger.info(f"Navigated to {url}, status: {response.status if response else 'N/A'}")
    
    async def screenshot(self, path: Optional[str] = None, full_page: bool = False, 
                        selector: Optional[str] = None, type_: str = "png") -> bytes:
        """
        Take a screenshot of the page.
        
        Args:
            path: Optional file path to save the screenshot
            full_page: Whether to capture the full scrollable page
            selector: Optional CSS selector to capture specific element
            type_: Screenshot format ('png' or 'jpeg')
            
        Returns:
            Screenshot bytes
        """
        kwargs = {"full_page": full_page, "type": type_}
        if path:
            kwargs["path"] = path
        if selector:
            element = await self._page.query_selector(selector)
            if element:
                result = await element.screenshot(**kwargs)
                logger.info(f"Element screenshot taken: {selector}")
                return result
        
        result = await self._page.screenshot(**kwargs)
        logger.info(f"Screenshot taken (full_page={full_page})")
        return result
    
    async def evaluate(self, script: str, arg: Any = None) -> Any:
        """
        Execute JavaScript in the page.
        
        Args:
            script: JavaScript code to execute
            arg: Optional argument to pass to the script
            
        Returns:
            Result of the JavaScript execution
        """
        if arg is not None:
            result = await self._page.evaluate(script, arg)
        else:
            result = await self._page.evaluate(script)
        logger.debug(f"JavaScript evaluated: {script[:50]}...")
        return result
    
    async def click(self, selector: str, timeout: Optional[int] = None, 
                   button: str = "left", click_count: int = 1) -> None:
        """
        Click an element on the page.
        
        Args:
            selector: CSS selector for the element to click
            timeout: Maximum time to wait for element in milliseconds
            button: Mouse button ('left', 'right', 'middle')
            click_count: Number of clicks
        """
        kwargs = {"button": button, "click_count": click_count}
        if timeout:
            kwargs["timeout"] = timeout
        
        await self._page.click(selector, **kwargs)
        logger.info(f"Clicked element: {selector}")
    
    async def fill(self, selector: str, value: str, timeout: Optional[int] = None) -> None:
        """
        Fill an input field with text.
        
        Args:
            selector: CSS selector for the input element
            value: Text to fill
            timeout: Maximum time to wait for element in milliseconds
        """
        kwargs = {}
        if timeout:
            kwargs["timeout"] = timeout
        
        await self._page.fill(selector, value, **kwargs)
        logger.info(f"Filled element {selector} with value (length: {len(value)})")
    
    async def type(self, selector: str, text: str, delay: int = 0, 
                  timeout: Optional[int] = None) -> None:
        """
        Type text into an element character by character.
        
        Args:
            selector: CSS selector for the input element
            text: Text to type
            delay: Delay between keystrokes in milliseconds
            timeout: Maximum time to wait for element in milliseconds
        """
        kwargs = {"delay": delay}
        if timeout:
            kwargs["timeout"] = timeout
        
        await self._page.type(selector, text, **kwargs)
        logger.info(f"Typed text into element {selector} (length: {len(text)})")
    
    async def select_option(self, selector: str, values: Union[str, List[str]], 
                           timeout: Optional[int] = None) -> List[str]:
        """
        Select options in a <select> element.
        
        Args:
            selector: CSS selector for the select element
            values: Option value(s) to select
            timeout: Maximum time to wait for element in milliseconds
            
        Returns:
            List of selected option values
        """
        kwargs = {}
        if timeout:
            kwargs["timeout"] = timeout
        
        result = await self._page.select_option(selector, values, **kwargs)
        logger.info(f"Selected options in element {selector}: {values}")
        return result
    
    async def wait_for_selector(self, selector: str, state: str = "visible", 
                               timeout: Optional[int] = None) -> Any:
        """
        Wait for an element matching the selector to appear.
        
        Args:
            selector: CSS selector to wait for
            state: State to wait for ('attached', 'detached', 'visible', 'hidden')
            timeout: Maximum time to wait in milliseconds
            
        Returns:
            The element handle
        """
        kwargs = {"state": state}
        if timeout:
            kwargs["timeout"] = timeout
        
        element = await self._page.wait_for_selector(selector, **kwargs)
        logger.info(f"Waited for selector: {selector} (state: {state})")
        return element
    
    async def wait_for_load_state(self, state: str = "load", timeout: Optional[int] = None) -> None:
        """
        Wait for the page to reach a specific load state.
        
        Args:
            state: Load state to wait for ('load', 'domcontentloaded', 'networkidle')
            timeout: Maximum time to wait in milliseconds
        """
        kwargs = {}
        if timeout:
            kwargs["timeout"] = timeout
        
        await self._page.wait_for_load_state(state, **kwargs)
        logger.info(f"Page reached load state: {state}")
    
    async def query_selector(self, selector: str) -> Any:
        """
        Find an element matching the selector.
        
        Args:
            selector: CSS selector
            
        Returns:
            Element handle or None
        """
        return await self._page.query_selector(selector)
    
    async def query_selector_all(self, selector: str) -> List[Any]:
        """
        Find all elements matching the selector.
        
        Args:
            selector: CSS selector
            
        Returns:
            List of element handles
        """
        return await self._page.query_selector_all(selector)
    
    async def get_text(self, selector: str) -> Optional[str]:
        """
        Get text content of an element.
        
        Args:
            selector: CSS selector
            
        Returns:
            Text content or None if element not found
        """
        element = await self._page.query_selector(selector)
        if element:
            return await element.text_content()
        return None
    
    async def get_attribute(self, selector: str, attribute: str) -> Optional[str]:
        """
        Get an attribute value of an element.
        
        Args:
            selector: CSS selector
            attribute: Attribute name
            
        Returns:
            Attribute value or None
        """
        element = await self._page.query_selector(selector)
        if element:
            return await element.get_attribute(attribute)
        return None
    
    async def scroll_to(self, selector: Optional[str] = None, x: int = 0, y: int = 0) -> None:
        """
        Scroll to an element or position.
        
        Args:
            selector: Optional CSS selector to scroll to
            x: X coordinate to scroll to (if selector not provided)
            y: Y coordinate to scroll to (if selector not provided)
        """
        if selector:
            element = await self._page.query_selector(selector)
            if element:
                await element.scroll_into_view_if_needed()
                logger.info(f"Scrolled to element: {selector}")
        else:
            await self._page.evaluate(f"window.scrollTo({x}, {y})")
            logger.info(f"Scrolled to position: ({x}, {y})")
    
    async def close(self) -> None:
        """Close this page."""
        await self._page.close()
        logger.info("BrowserPage closed")


class PlaywrightManager:
    """
    Main manager class for Playwright browser automation.
    
    Handles browser lifecycle including launch, context creation,
    and cleanup. Provides async context manager support.
    
    Example:
        async with PlaywrightManager() as manager:
            context = await manager.new_context()
            page = await context.new_page()
            await page.goto("https://example.com")
    """
    
    def __init__(self, config: Optional[BrowserConfig] = None):
        """
        Initialize the Playwright manager.
        
        Args:
            config: Optional browser configuration
        """
        self._config = config or BrowserConfig()
        self._playwright = None
        self._browser = None
        self._contexts: List[BrowserContext] = []
        logger.info("PlaywrightManager initialized")
    
    async def __aenter__(self) -> 'PlaywrightManager':
        """Async context manager entry."""
        await self.launch()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.close()
    
    async def launch(self, channel: Optional[str] = None, headless: Optional[bool] = None,
                     args: Optional[List[str]] = None) -> None:
        """
        Launch the browser.
        
        Args:
            channel: Browser channel ('chrome', 'chromium', 'firefox', 'webkit', 'msedge')
            headless: Whether to run in headless mode
            args: Additional browser arguments
        """
        from playwright.async_api import async_playwright
        
        self._playwright = await async_playwright().start()
        
        # Use provided values or fall back to config
        channel = channel or self._config.channel or "chromium"
        headless = headless if headless is not None else self._config.headless
        browser_args = args or self._config.args
        
        # Build launch options
        launch_options = {
            "headless": headless,
            "args": browser_args,
        }
        
        if self._config.downloads_path:
            launch_options["downloads_path"] = self._config.downloads_path
        if self._config.traces_dir:
            launch_options["traces_dir"] = self._config.traces_dir
        
        # Launch appropriate browser
        if channel in ["chrome", "chromium", "msedge"]:
            self._browser = await self._playwright.chromium.launch(**launch_options)
        elif channel == "firefox":
            self._browser = await self._playwright.firefox.launch(**launch_options)
        elif channel == "webkit":
            self._browser = await self._playwright.webkit.launch(**launch_options)
        else:
            raise ValueError(f"Unsupported browser channel: {channel}")
        
        logger.info(f"Browser launched: {channel} (headless={headless})")
    
    async def new_context(self, **kwargs) -> BrowserContext:
        """
        Create a new browser context.
        
        Args:
            **kwargs: Additional context options (viewport, user_agent, etc.)
            
        Returns:
            BrowserContext wrapper instance
        """
        if not self._browser:
            raise RuntimeError("Browser not launched. Call launch() first.")
        
        context = await self._browser.new_context(**kwargs)
        browser_context = BrowserContext(context, kwargs)
        self._contexts.append(browser_context)
        logger.info("New browser context created")
        return browser_context
    
    async def new_page(self, context: Optional[BrowserContext] = None) -> BrowserPage:
        """
        Create a new page.
        
        Args:
            context: Optional context to create page in (creates new context if None)
            
        Returns:
            BrowserPage wrapper instance
        """
        if context is None:
            context = await self.new_context()
        
        return await context.new_page()
    
    async def close(self) -> None:
        """Close the browser and cleanup resources."""
        # Close all contexts
        for context in self._contexts:
            await context.close()
        self._contexts.clear()
        
        # Close browser
        if self._browser:
            await self._browser.close()
            self._browser = None
        
        # Stop playwright
        if self._playwright:
            await self._playwright.stop()
            self._playwright = None
        
        logger.info("PlaywrightManager closed")
    
    @property
    def is_launched(self) -> bool:
        """Check if browser is launched."""
        return self._browser is not None
    
    @property
    def browser(self) -> Any:
        """Get the underlying browser instance."""
        return self._browser
