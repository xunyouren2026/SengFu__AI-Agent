"""
DOM Extractor Module

Extracts and structures DOM content from web pages including
interactive elements, links, forms, and tables.
"""

from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json
import logging

logger = logging.getLogger(__name__)


@dataclass
class ElementRect:
    """Rectangle representing element position and size."""
    x: float = 0.0
    y: float = 0.0
    width: float = 0.0
    height: float = 0.0
    
    def to_dict(self) -> Dict[str, float]:
        return {
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, float]) -> 'ElementRect':
        return cls(
            x=data.get("x", 0.0),
            y=data.get("y", 0.0),
            width=data.get("width", 0.0),
            height=data.get("height", 0.0)
        )


@dataclass
class InteractiveElement:
    """
    Represents an interactive element on a web page.
    
    Attributes:
        tag: HTML tag name
        element_id: Element ID attribute
        name: Element name attribute
        element_type: Input type or element type
        text: Visible text content
        rect: Element position and dimensions
        is_visible: Whether element is visible
        is_enabled: Whether element is enabled
        selector: CSS selector for the element
        attributes: Additional element attributes
    """
    tag: str
    element_id: Optional[str] = None
    name: Optional[str] = None
    element_type: Optional[str] = None
    text: Optional[str] = None
    rect: ElementRect = field(default_factory=ElementRect)
    is_visible: bool = True
    is_enabled: bool = True
    selector: Optional[str] = None
    attributes: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "tag": self.tag,
            "id": self.element_id,
            "name": self.name,
            "type": self.element_type,
            "text": self.text,
            "rect": self.rect.to_dict(),
            "is_visible": self.is_visible,
            "is_enabled": self.is_enabled,
            "selector": self.selector,
            "attributes": self.attributes
        }


@dataclass
class FormInput:
    """Represents a form input field."""
    name: Optional[str] = None
    input_type: str = "text"
    selector: Optional[str] = None
    label: Optional[str] = None
    placeholder: Optional[str] = None
    required: bool = False
    value: Optional[str] = None
    options: List[Dict[str, str]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "type": self.input_type,
            "selector": self.selector,
            "label": self.label,
            "placeholder": self.placeholder,
            "required": self.required,
            "value": self.value,
            "options": self.options
        }


@dataclass
class FormInfo:
    """
    Represents a form on a web page.
    
    Attributes:
        action: Form action URL
        method: HTTP method (GET, POST, etc.)
        inputs: List of form input fields
        submit_button: Selector for submit button
        selector: CSS selector for the form
        name: Form name attribute
        id: Form ID attribute
    """
    action: Optional[str] = None
    method: str = "GET"
    inputs: List[FormInput] = field(default_factory=list)
    submit_button: Optional[str] = None
    selector: Optional[str] = None
    name: Optional[str] = None
    form_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "action": self.action,
            "method": self.method,
            "inputs": [inp.to_dict() for inp in self.inputs],
            "submit_button": self.submit_button,
            "selector": self.selector,
            "name": self.name,
            "id": self.form_id
        }


@dataclass
class TableCell:
    """Represents a table cell."""
    text: str = ""
    row_span: int = 1
    col_span: int = 1
    is_header: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "row_span": self.row_span,
            "col_span": self.col_span,
            "is_header": self.is_header
        }


@dataclass
class TableInfo:
    """
    Represents a table on a web page.
    
    Attributes:
        headers: List of header cell texts
        rows: List of row data (each row is a list of cell texts)
        cells: 2D list of TableCell objects
        caption: Table caption text
        selector: CSS selector for the table
        row_count: Number of rows
        col_count: Number of columns
    """
    headers: List[str] = field(default_factory=list)
    rows: List[List[str]] = field(default_factory=list)
    cells: List[List[TableCell]] = field(default_factory=list)
    caption: Optional[str] = None
    selector: Optional[str] = None
    row_count: int = 0
    col_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "headers": self.headers,
            "rows": self.rows,
            "cells": [[cell.to_dict() for cell in row] for row in self.cells],
            "caption": self.caption,
            "selector": self.selector,
            "row_count": self.row_count,
            "col_count": self.col_count
        }


@dataclass
class LinkInfo:
    """Represents a link on a web page."""
    text: str = ""
    href: str = ""
    title: Optional[str] = None
    selector: Optional[str] = None
    is_external: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "href": self.href,
            "title": self.title,
            "selector": self.selector,
            "is_external": self.is_external
        }


class DOMExtractor:
    """
    Extracts structured DOM content from web pages.
    
    Provides methods to extract interactive elements, forms, tables,
    and links from a Playwright page.
    
    Example:
        extractor = DOMExtractor()
        elements = await extractor.extract_interactive_elements(page)
        forms = await extractor.extract_forms(page)
    """
    
    # Tags considered interactive
    INTERACTIVE_TAGS = [
        'a', 'button', 'input', 'select', 'textarea',
        'details', 'summary', 'label'
    ]
    
    # Input types considered interactive
    INTERACTIVE_INPUT_TYPES = [
        'text', 'password', 'email', 'tel', 'url',
        'search', 'number', 'date', 'datetime-local',
        'checkbox', 'radio', 'file', 'submit', 'button',
        'reset', 'image'
    ]
    
    def __init__(self):
        """Initialize the DOM extractor."""
        logger.info("DOMExtractor initialized")
    
    async def extract_elements(self, page: Any, 
                               filter_tags: Optional[List[str]] = None) -> List[InteractiveElement]:
        """
        Extract elements matching specified tags.
        
        Args:
            page: Playwright page object
            filter_tags: List of tag names to extract (default: interactive tags)
            
        Returns:
            List of InteractiveElement objects
        """
        tags = filter_tags or self.INTERACTIVE_TAGS
        
        script = """
        (tags) => {
            const results = [];
            tags.forEach(tag => {
                const elements = document.querySelectorAll(tag);
                elements.forEach((el, index) => {
                    const rect = el.getBoundingClientRect();
                    const computedStyle = window.getComputedStyle(el);
                    const isVisible = computedStyle.display !== 'none' && 
                                     computedStyle.visibility !== 'hidden' &&
                                     rect.width > 0 && rect.height > 0;
                    
                    // Build selector
                    let selector = tag;
                    if (el.id) selector = '#' + el.id;
                    else if (el.name) selector = tag + '[name="' + el.name + '"]';
                    else if (el.className) {
                        const classes = el.className.split(' ').filter(c => c).join('.');
                        if (classes) selector = tag + '.' + classes;
                    }
                    
                    // Get attributes
                    const attrs = {};
                    for (const attr of el.attributes) {
                        attrs[attr.name] = attr.value;
                    }
                    
                    results.push({
                        tag: tag,
                        id: el.id || null,
                        name: el.name || null,
                        type: el.type || null,
                        text: el.textContent?.trim().substring(0, 200) || null,
                        rect: {
                            x: rect.x,
                            y: rect.y,
                            width: rect.width,
                            height: rect.height
                        },
                        is_visible: isVisible,
                        is_enabled: !el.disabled,
                        selector: selector,
                        attributes: attrs
                    });
                });
            });
            return results;
        }
        """
        
        data = await page.evaluate(script, tags)
        
        elements = []
        for item in data:
            element = InteractiveElement(
                tag=item["tag"],
                element_id=item["id"],
                name=item["name"],
                element_type=item["type"],
                text=item["text"],
                rect=ElementRect.from_dict(item["rect"]),
                is_visible=item["is_visible"],
                is_enabled=item["is_enabled"],
                selector=item["selector"],
                attributes=item["attributes"]
            )
            elements.append(element)
        
        logger.info(f"Extracted {len(elements)} elements with tags: {tags}")
        return elements
    
    async def extract_interactive_elements(self, page: Any) -> List[InteractiveElement]:
        """
        Extract all interactive elements (buttons, inputs, links, etc.).
        
        Args:
            page: Playwright page object
            
        Returns:
            List of InteractiveElement objects
        """
        elements = await self.extract_elements(page, self.INTERACTIVE_TAGS)
        
        # Filter to only visible and enabled elements
        interactive = [
            e for e in elements 
            if e.is_visible and self._is_interactive_element(e)
        ]
        
        logger.info(f"Extracted {len(interactive)} interactive elements")
        return interactive
    
    def _is_interactive_element(self, element: InteractiveElement) -> bool:
        """Check if an element is interactive."""
        if element.tag in ['a', 'button', 'select', 'textarea']:
            return True
        if element.tag == 'input':
            return element.element_type in self.INTERACTIVE_INPUT_TYPES
        if element.tag == 'label':
            return True
        return False
    
    async def extract_links(self, page: Any, 
                           include_external: bool = True) -> List[LinkInfo]:
        """
        Extract all links from the page.
        
        Args:
            page: Playwright page object
            include_external: Whether to include external links
            
        Returns:
            List of LinkInfo objects
        """
        script = """
        () => {
            const links = document.querySelectorAll('a[href]');
            const results = [];
            const currentHost = window.location.host;
            
            links.forEach((link, index) => {
                const href = link.href;
                const isExternal = !href.includes(currentHost) && href.startsWith('http');
                
                // Build selector
                let selector = 'a';
                if (link.id) selector = '#' + link.id;
                else if (link.className) {
                    const classes = link.className.split(' ').filter(c => c).join('.');
                    if (classes) selector = 'a.' + classes;
                } else {
                    selector = `a:nth-of-type(${index + 1})`;
                }
                
                results.push({
                    text: link.textContent?.trim() || '',
                    href: href,
                    title: link.title || null,
                    selector: selector,
                    is_external: isExternal
                });
            });
            return results;
        }
        """
        
        data = await page.evaluate(script)
        
        links = []
        for item in data:
            if not include_external and item["is_external"]:
                continue
            
            link = LinkInfo(
                text=item["text"],
                href=item["href"],
                title=item["title"],
                selector=item["selector"],
                is_external=item["is_external"]
            )
            links.append(link)
        
        logger.info(f"Extracted {len(links)} links")
        return links
    
    async def extract_forms(self, page: Any) -> List[FormInfo]:
        """
        Extract all forms from the page.
        
        Args:
            page: Playwright page object
            
        Returns:
            List of FormInfo objects
        """
        script = """
        () => {
            const forms = document.querySelectorAll('form');
            const results = [];
            
            forms.forEach((form, formIndex) => {
                // Build form selector
                let formSelector = 'form';
                if (form.id) formSelector = '#' + form.id;
                else if (form.name) formSelector = 'form[name="' + form.name + '"]';
                else formSelector = `form:nth-of-type(${formIndex + 1})`;
                
                // Extract inputs
                const inputs = [];
                const inputElements = form.querySelectorAll('input, select, textarea');
                
                inputElements.forEach((input, inputIndex) => {
                    const label = form.querySelector(`label[for="${input.id}"]`) ||
                                 input.closest('label');
                    const labelText = label ? label.textContent?.trim() : null;
                    
                    // Build input selector
                    let inputSelector = input.tagName.toLowerCase();
                    if (input.id) inputSelector = '#' + input.id;
                    else if (input.name) inputSelector = `${input.tagName.toLowerCase()}[name="${input.name}"]`;
                    
                    // Get options for select
                    const options = [];
                    if (input.tagName === 'SELECT') {
                        for (const opt of input.options) {
                            options.push({
                                value: opt.value,
                                text: opt.text
                            });
                        }
                    }
                    
                    inputs.push({
                        name: input.name || null,
                        type: input.type || input.tagName.toLowerCase(),
                        selector: inputSelector,
                        label: labelText,
                        placeholder: input.placeholder || null,
                        required: input.required,
                        value: input.value || null,
                        options: options
                    });
                });
                
                // Find submit button
                let submitButton = form.querySelector('input[type="submit"], button[type="submit"]');
                let submitSelector = null;
                if (submitButton) {
                    if (submitButton.id) submitSelector = '#' + submitButton.id;
                    else submitSelector = submitButton.tagName.toLowerCase() + '[type="submit"]';
                }
                
                results.push({
                    action: form.action || null,
                    method: form.method?.toUpperCase() || 'GET',
                    inputs: inputs,
                    submit_button: submitSelector,
                    selector: formSelector,
                    name: form.name || null,
                    id: form.id || null
                });
            });
            return results;
        }
        """
        
        data = await page.evaluate(script)
        
        forms = []
        for item in data:
            inputs = [
                FormInput(
                    name=inp["name"],
                    input_type=inp["type"],
                    selector=inp["selector"],
                    label=inp["label"],
                    placeholder=inp["placeholder"],
                    required=inp["required"],
                    value=inp["value"],
                    options=inp["options"]
                )
                for inp in item["inputs"]
            ]
            
            form = FormInfo(
                action=item["action"],
                method=item["method"],
                inputs=inputs,
                submit_button=item["submit_button"],
                selector=item["selector"],
                name=item["name"],
                form_id=item["id"]
            )
            forms.append(form)
        
        logger.info(f"Extracted {len(forms)} forms")
        return forms
    
    async def extract_tables(self, page: Any) -> List[TableInfo]:
        """
        Extract all tables from the page.
        
        Args:
            page: Playwright page object
            
        Returns:
            List of TableInfo objects
        """
        script = """
        () => {
            const tables = document.querySelectorAll('table');
            const results = [];
            
            tables.forEach((table, tableIndex) => {
                // Build selector
                let selector = 'table';
                if (table.id) selector = '#' + table.id;
                else if (table.className) {
                    const classes = table.className.split(' ').filter(c => c).join('.');
                    if (classes) selector = 'table.' + classes;
                } else {
                    selector = `table:nth-of-type(${tableIndex + 1})`;
                }
                
                // Get caption
                const caption = table.querySelector('caption');
                const captionText = caption ? caption.textContent?.trim() : null;
                
                // Extract headers
                const headers = [];
                const headerRow = table.querySelector('thead tr') || table.querySelector('tr');
                if (headerRow) {
                    const headerCells = headerRow.querySelectorAll('th, td');
                    headerCells.forEach(cell => {
                        headers.push(cell.textContent?.trim() || '');
                    });
                }
                
                // Extract rows
                const rows = [];
                const cells = [];
                const tbody = table.querySelector('tbody') || table;
                const dataRows = tbody.querySelectorAll('tr');
                
                dataRows.forEach(row => {
                    const rowData = [];
                    const rowCells = [];
                    const cells_in_row = row.querySelectorAll('td, th');
                    
                    cells_in_row.forEach(cell => {
                        const text = cell.textContent?.trim() || '';
                        rowData.push(text);
                        rowCells.push({
                            text: text,
                            row_span: parseInt(cell.rowSpan) || 1,
                            col_span: parseInt(cell.colSpan) || 1,
                            is_header: cell.tagName === 'TH'
                        });
                    });
                    
                    if (rowData.length > 0) {
                        rows.push(rowData);
                        cells.push(rowCells);
                    }
                });
                
                results.push({
                    headers: headers,
                    rows: rows,
                    cells: cells,
                    caption: captionText,
                    selector: selector,
                    row_count: rows.length,
                    col_count: headers.length || (rows[0]?.length || 0)
                });
            });
            return results;
        }
        """
        
        data = await page.evaluate(script)
        
        tables = []
        for item in data:
            cells = [
                [TableCell(
                    text=c["text"],
                    row_span=c["row_span"],
                    col_span=c["col_span"],
                    is_header=c["is_header"]
                ) for c in row]
                for row in item["cells"]
            ]
            
            table = TableInfo(
                headers=item["headers"],
                rows=item["rows"],
                cells=cells,
                caption=item["caption"],
                selector=item["selector"],
                row_count=item["row_count"],
                col_count=item["col_count"]
            )
            tables.append(table)
        
        logger.info(f"Extracted {len(tables)} tables")
        return tables
    
    async def extract_page_info(self, page: Any) -> Dict[str, Any]:
        """
        Extract general page information.
        
        Args:
            page: Playwright page object
            
        Returns:
            Dictionary with page information
        """
        script = """
        () => {
            return {
                title: document.title,
                url: window.location.href,
                domain: window.location.hostname,
                description: document.querySelector('meta[name="description"]')?.content || null,
                keywords: document.querySelector('meta[name="keywords"]')?.content || null,
                viewport: document.querySelector('meta[name="viewport"]')?.content || null,
                language: document.documentElement.lang || null,
                charset: document.characterSet,
                body_text: document.body?.textContent?.substring(0, 1000) || null
            };
        }
        """
        
        info = await page.evaluate(script)
        logger.info(f"Extracted page info for: {info.get('title', 'Unknown')}")
        return info
    
    async def extract_all(self, page: Any) -> Dict[str, Any]:
        """
        Extract all DOM content from the page.
        
        Args:
            page: Playwright page object
            
        Returns:
            Dictionary containing all extracted content
        """
        results = {
            "page_info": await self.extract_page_info(page),
            "interactive_elements": [
                e.to_dict() for e in await self.extract_interactive_elements(page)
            ],
            "links": [l.to_dict() for l in await self.extract_links(page)],
            "forms": [f.to_dict() for f in await self.extract_forms(page)],
            "tables": [t.to_dict() for t in await self.extract_tables(page)]
        }
        
        logger.info("Extracted all DOM content")
        return results
