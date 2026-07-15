"""
Browser Automation Module

A comprehensive browser automation framework using Playwright.
Provides tools for browser control, action recording, DOM extraction,
stealth capabilities, and more.

Note: This module requires Playwright to be installed separately:
    pip install playwright
    playwright install

Example:
    from agi_unified_framework.computer_use.browser import PlaywrightManager
    
    async with PlaywrightManager() as manager:
        page = await manager.new_page()
        await page.goto("https://example.com")
        screenshot = await page.screenshot()
"""

__version__ = "1.0.0"
__author__ = "AGI Unified Framework"

# Import main classes for convenient access
try:
    from .playwright_manager import (
        PlaywrightManager,
        BrowserContext,
        BrowserPage,
    )
    from .action_recorder import (
        BrowserActionRecorder,
        RecordedBrowserAction,
        ActionReplayer,
    )
    from .dom_extractor import (
        DOMExtractor,
        InteractiveElement,
        FormInfo,
        TableInfo,
    )
    from .stealth import (
        StealthConfig,
        StealthManager,
    )
    from .screenshot_diff import (
        ScreenshotDiff,
        DiffRegion,
    )
    from .form_filler import (
        SemanticMatcher,
        SmartFormFiller,
        FormField,
    )
    from .cookie_manager import (
        CookieManager,
        CookieProfile,
    )
    from .captcha_handler import (
        CaptchaHandler,
        CaptchaType,
    )
    from .multi_tab import (
        MultiTabManager,
        TabInfo,
    )
    
    __all__ = [
        # Playwright Manager
        "PlaywrightManager",
        "BrowserContext",
        "BrowserPage",
        # Action Recorder
        "BrowserActionRecorder",
        "RecordedBrowserAction",
        "ActionReplayer",
        # DOM Extractor
        "DOMExtractor",
        "InteractiveElement",
        "FormInfo",
        "TableInfo",
        # Stealth
        "StealthConfig",
        "StealthManager",
        # Screenshot Diff
        "ScreenshotDiff",
        "DiffRegion",
        # Form Filler
        "SemanticMatcher",
        "SmartFormFiller",
        "FormField",
        # Cookie Manager
        "CookieManager",
        "CookieProfile",
        # Captcha Handler
        "CaptchaHandler",
        "CaptchaType",
        # Multi Tab
        "MultiTabManager",
        "TabInfo",
    ]
    
except ImportError as e:
    # Playwright not installed, provide informative error
    import warnings
    warnings.warn(
        f"Playwright is required for browser automation: {e}\n"
        "Install with: pip install playwright && playwright install",
        ImportWarning
    )
    __all__ = []


def check_playwright_installation():
    """Check if Playwright is properly installed."""
    try:
        import playwright
        return True
    except ImportError:
        return False


def get_module_info():
    """Get information about the browser module."""
    return {
        "version": __version__,
        "author": __author__,
        "playwright_installed": check_playwright_installation(),
        "components": [
            "PlaywrightManager - Browser lifecycle management",
            "BrowserActionRecorder - Record and replay browser actions",
            "DOMExtractor - Extract DOM elements and content",
            "StealthManager - Anti-detection capabilities",
            "ScreenshotDiff - Compare screenshots",
            "SmartFormFiller - Intelligent form filling",
            "CookieManager - Cookie persistence",
            "CaptchaHandler - CAPTCHA detection and solving",
            "MultiTabManager - Multi-tab management",
        ]
    }
