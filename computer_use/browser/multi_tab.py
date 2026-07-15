"""
Multi-Tab Manager Module

Provides comprehensive multi-tab/window management for browser automation.
Handles tab creation, switching, closing, and monitoring.
"""

from typing import Optional, List, Dict, Any, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import asyncio
import logging

logger = logging.getLogger(__name__)


class TabState(Enum):
    """States a tab can be in."""
    LOADING = "loading"
    INTERACTIVE = "interactive"
    COMPLETE = "complete"
    CLOSED = "closed"
    ERROR = "error"


@dataclass
class TabInfo:
    """
    Information about a browser tab.
    
    Attributes:
        id: Unique tab identifier
        url: Current URL
        title: Page title
        is_active: Whether this is the active tab
        state: Current tab state
        opener_id: ID of the tab that opened this one
        created_at: Timestamp when tab was created
        metadata: Additional metadata
    """
    id: str
    url: str = ""
    title: str = ""
    is_active: bool = False
    state: TabState = TabState.LOADING
    opener_id: Optional[str] = None
    created_at: float = field(default_factory=lambda: asyncio.get_event_loop().time())
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "url": self.url,
            "title": self.title,
            "is_active": self.is_active,
            "state": self.state.value,
            "opener_id": self.opener_id,
            "created_at": self.created_at,
            "metadata": self.metadata
        }


class MultiTabManager:
    """
    Manages multiple browser tabs/windows.
    
    Provides functionality to open, close, switch between, and monitor
    multiple tabs within a browser context.
    
    Example:
        manager = MultiTabManager()
        new_tab = await manager.open_new_tab(page, "https://example.com")
        await manager.switch_to_tab(page, new_tab.id)
        await manager.close_tab(page, new_tab.id)
    """
    
    def __init__(self):
        """Initialize the multi-tab manager."""
        self._tabs: Dict[str, TabInfo] = {}
        self._page_to_tab: Dict[int, str] = {}  # Maps page id to tab id
        self._tab_counter = 0
        self._event_handlers: Dict[str, List[Callable]] = {
            "tab_created": [],
            "tab_closed": [],
            "tab_switched": [],
            "tab_updated": []
        }
        logger.info("MultiTabManager initialized")
    
    async def open_new_tab(self, page: Any, url: Optional[str] = None,
                          wait_until: str = "load") -> TabInfo:
        """
        Open a new tab and optionally navigate to a URL.
        
        Args:
            page: Current Playwright page (for context)
            url: Optional URL to navigate to
            wait_until: When to consider navigation complete
            
        Returns:
            TabInfo for the new tab
        """
        # Get context from page
        context = page.context
        
        # Create new page in context
        new_page = await context.new_page()
        
        # Generate tab ID
        self._tab_counter += 1
        tab_id = f"tab_{self._tab_counter}"
        
        # Create tab info
        tab_info = TabInfo(
            id=tab_id,
            opener_id=self._page_to_tab.get(id(page)),
            state=TabState.LOADING
        )
        
        # Store mappings
        self._tabs[tab_id] = tab_info
        self._page_to_tab[id(new_page)] = tab_id
        
        # Navigate if URL provided
        if url:
            await new_page.goto(url, wait_until=wait_until)
            tab_info.url = new_page.url
            tab_info.title = await new_page.title()
            tab_info.state = TabState.COMPLETE
        else:
            tab_info.url = "about:blank"
            tab_info.state = TabState.INTERACTIVE
        
        # Set up event listeners
        await self._setup_tab_listeners(new_page, tab_id)
        
        # Fire event
        await self._fire_event("tab_created", tab_info)
        
        logger.info(f"Opened new tab: {tab_id} ({url or 'about:blank'})")
        return tab_info
    
    async def close_tab(self, page: Any, tab_id: str) -> bool:
        """
        Close a tab by ID.
        
        Args:
            page: Playwright page (for context access)
            tab_id: ID of tab to close
            
        Returns:
            True if tab was closed
        """
        tab_info = self._tabs.get(tab_id)
        if not tab_info:
            logger.warning(f"Tab not found: {tab_id}")
            return False
        
        # Find the page for this tab
        context = page.context
        pages = context.pages
        
        for p in pages:
            if self._page_to_tab.get(id(p)) == tab_id:
                await p.close()
                break
        
        # Update state
        tab_info.state = TabState.CLOSED
        del self._tabs[tab_id]
        
        # Clean up mapping
        for page_id, tid in list(self._page_to_tab.items()):
            if tid == tab_id:
                del self._page_to_tab[page_id]
                break
        
        # Fire event
        await self._fire_event("tab_closed", tab_info)
        
        logger.info(f"Closed tab: {tab_id}")
        return True
    
    async def switch_to_tab(self, page: Any, tab_id: str) -> Optional[Any]:
        """
        Switch to a different tab.
        
        Args:
            page: Current Playwright page
            tab_id: ID of tab to switch to
            
        Returns:
            The Playwright page for the switched-to tab, or None
        """
        tab_info = self._tabs.get(tab_id)
        if not tab_info:
            logger.warning(f"Tab not found: {tab_id}")
            return None
        
        # Find the page for this tab
        context = page.context
        pages = context.pages
        
        target_page = None
        for p in pages:
            if self._page_to_tab.get(id(p)) == tab_id:
                target_page = p
                break
        
        if target_page:
            # Update active states
            for tid, tinfo in self._tabs.items():
                tinfo.is_active = (tid == tab_id)
            
            # Bring to front
            await target_page.bring_to_front()
            
            # Update tab info
            tab_info.url = target_page.url
            try:
                tab_info.title = await target_page.title()
            except:
                pass
            
            # Fire event
            await self._fire_event("tab_switched", tab_info)
            
            logger.info(f"Switched to tab: {tab_id}")
            return target_page
        
        return None
    
    async def get_all_tabs(self, page: Any) -> List[TabInfo]:
        """
        Get information about all tabs.
        
        Args:
            page: Playwright page (for context access)
            
        Returns:
            List of TabInfo objects
        """
        context = page.context
        pages = context.pages
        
        # Update tab information
        for p in pages:
            page_id = id(p)
            tab_id = self._page_to_tab.get(page_id)
            if tab_id and tab_id in self._tabs:
                tab_info = self._tabs[tab_id]
                try:
                    tab_info.url = p.url
                    tab_info.title = await p.title()
                except:
                    pass
        
        return list(self._tabs.values())
    
    async def get_active_tab(self, page: Any) -> Optional[TabInfo]:
        """
        Get the currently active tab.
        
        Args:
            page: Playwright page
            
        Returns:
            TabInfo for active tab, or None
        """
        for tab_info in self._tabs.values():
            if tab_info.is_active:
                return tab_info
        return None
    
    async def wait_for_tab(self, page: Any, condition: Callable[[TabInfo], bool],
                          timeout: int = 30000) -> Optional[TabInfo]:
        """
        Wait for a tab to meet a condition.
        
        Args:
            page: Playwright page
            condition: Function that takes TabInfo and returns bool
            timeout: Maximum wait time in milliseconds
            
        Returns:
            TabInfo that met the condition, or None
        """
        start_time = asyncio.get_event_loop().time()
        
        while (asyncio.get_event_loop().time() - start_time) * 1000 < timeout:
            tabs = await self.get_all_tabs(page)
            for tab in tabs:
                if condition(tab):
                    return tab
            await asyncio.sleep(0.5)
        
        logger.warning("Timeout waiting for tab condition")
        return None
    
    async def wait_for_new_tab(self, page: Any, timeout: int = 30000) -> Optional[TabInfo]:
        """
        Wait for a new tab to be opened.
        
        Args:
            page: Playwright page
            timeout: Maximum wait time in milliseconds
            
        Returns:
            TabInfo for the new tab, or None
        """
        existing_ids = set(self._tabs.keys())
        
        start_time = asyncio.get_event_loop().time()
        while (asyncio.get_event_loop().time() - start_time) * 1000 < timeout:
            await self.get_all_tabs(page)  # Refresh tab list
            current_ids = set(self._tabs.keys())
            new_ids = current_ids - existing_ids
            
            if new_ids:
                new_tab_id = new_ids.pop()
                logger.info(f"New tab detected: {new_tab_id}")
                return self._tabs.get(new_tab_id)
            
            await asyncio.sleep(0.5)
        
        logger.warning("Timeout waiting for new tab")
        return None
    
    async def close_all_tabs_except(self, page: Any, keep_tab_id: str) -> int:
        """
        Close all tabs except the specified one.
        
        Args:
            page: Playwright page
            keep_tab_id: ID of tab to keep open
            
        Returns:
            Number of tabs closed
        """
        tabs_to_close = [tid for tid in self._tabs.keys() if tid != keep_tab_id]
        closed_count = 0
        
        for tab_id in tabs_to_close:
            if await self.close_tab(page, tab_id):
                closed_count += 1
        
        logger.info(f"Closed {closed_count} tabs, kept {keep_tab_id}")
        return closed_count
    
    async def close_all_tabs(self, page: Any) -> int:
        """
        Close all tabs.
        
        Args:
            page: Playwright page
            
        Returns:
            Number of tabs closed
        """
        tab_ids = list(self._tabs.keys())
        closed_count = 0
        
        for tab_id in tab_ids:
            if await self.close_tab(page, tab_id):
                closed_count += 1
        
        logger.info(f"Closed all {closed_count} tabs")
        return closed_count
    
    def get_tab_by_url(self, url_pattern: str) -> Optional[TabInfo]:
        """
        Find a tab by URL pattern.
        
        Args:
            url_pattern: URL substring to search for
            
        Returns:
            TabInfo or None
        """
        for tab_info in self._tabs.values():
            if url_pattern in tab_info.url:
                return tab_info
        return None
    
    def get_tab_by_title(self, title_pattern: str) -> Optional[TabInfo]:
        """
        Find a tab by title pattern.
        
        Args:
            title_pattern: Title substring to search for
            
        Returns:
            TabInfo or None
        """
        for tab_info in self._tabs.values():
            if title_pattern in tab_info.title:
                return tab_info
        return None
    
    async def refresh_tab_info(self, page: Any, tab_id: str) -> Optional[TabInfo]:
        """
        Refresh information for a specific tab.
        
        Args:
            page: Playwright page
            tab_id: Tab ID to refresh
            
        Returns:
            Updated TabInfo or None
        """
        tab_info = self._tabs.get(tab_id)
        if not tab_info:
            return None
        
        context = page.context
        pages = context.pages
        
        for p in pages:
            if self._page_to_tab.get(id(p)) == tab_id:
                try:
                    tab_info.url = p.url
                    tab_info.title = await p.title()
                    await self._fire_event("tab_updated", tab_info)
                except:
                    pass
                break
        
        return tab_info
    
    def on(self, event: str, handler: Callable) -> None:
        """
        Register an event handler.
        
        Args:
            event: Event name ("tab_created", "tab_closed", "tab_switched", "tab_updated")
            handler: Callback function
        """
        if event in self._event_handlers:
            self._event_handlers[event].append(handler)
        else:
            raise ValueError(f"Unknown event: {event}")
    
    def off(self, event: str, handler: Callable) -> None:
        """
        Unregister an event handler.
        
        Args:
            event: Event name
            handler: Callback function to remove
        """
        if event in self._event_handlers:
            if handler in self._event_handlers[event]:
                self._event_handlers[event].remove(handler)
    
    async def _fire_event(self, event: str, tab_info: TabInfo) -> None:
        """Fire an event to all registered handlers."""
        if event in self._event_handlers:
            for handler in self._event_handlers[event]:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        await handler(tab_info)
                    else:
                        handler(tab_info)
                except Exception as e:
                    logger.error(f"Event handler error: {e}")
    
    async def _setup_tab_listeners(self, page: Any, tab_id: str) -> None:
        """Set up event listeners for a tab."""
        tab_info = self._tabs.get(tab_id)
        if not tab_info:
            return
        
        # Listen for dialog events
        page.on("dialog", lambda dialog: asyncio.create_task(self._handle_dialog(dialog, tab_id)))
        
        # Listen for console messages
        page.on("console", lambda msg: self._handle_console(msg, tab_id))
    
    async def _handle_dialog(self, dialog: Any, tab_id: str) -> None:
        """Handle dialog events."""
        logger.debug(f"Dialog on tab {tab_id}: {dialog.type} - {dialog.message}")
        # Auto-dismiss for now, could be configurable
        await dialog.dismiss()
    
    def _handle_console(self, message: Any, tab_id: str) -> None:
        """Handle console messages."""
        tab_info = self._tabs.get(tab_id)
        if tab_info:
            tab_info.metadata.setdefault("console_logs", []).append({
                "type": message.type,
                "text": message.text
            })
    
    def get_tab_count(self) -> int:
        """Get the number of managed tabs."""
        return len(self._tabs)
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about managed tabs.
        
        Returns:
            Dictionary with tab statistics
        """
        states = {}
        for tab in self._tabs.values():
            state = tab.state.value
            states[state] = states.get(state, 0) + 1
        
        return {
            "total_tabs": len(self._tabs),
            "active_tab": sum(1 for t in self._tabs.values() if t.is_active),
            "states": states
        }
