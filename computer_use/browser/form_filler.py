"""
Form Filler Module

Provides intelligent form filling capabilities with semantic matching,
field type inference, and AI-assisted form completion.
"""

from typing import Optional, List, Dict, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import re
import logging

logger = logging.getLogger(__name__)


class FieldType(Enum):
    """Types of form fields."""
    TEXT = "text"
    EMAIL = "email"
    PASSWORD = "password"
    PHONE = "phone"
    URL = "url"
    NUMBER = "number"
    DATE = "date"
    CHECKBOX = "checkbox"
    RADIO = "radio"
    SELECT = "select"
    TEXTAREA = "textarea"
    FILE = "file"
    HIDDEN = "hidden"
    SEARCH = "search"
    USERNAME = "username"
    NAME = "name"
    ADDRESS = "address"
    CITY = "city"
    STATE = "state"
    ZIP = "zip"
    COUNTRY = "country"
    CREDIT_CARD = "credit_card"
    CVV = "cvv"
    EXPIRY = "expiry"
    UNKNOWN = "unknown"


@dataclass
class FormField:
    """
    Represents a form field with its metadata.
    
    Attributes:
        label: Field label text
        selector: CSS selector for the field
        field_type: Inferred field type
        value: Current or target value
        placeholder: Placeholder text
        required: Whether field is required
        options: Options for select/radio fields
        confidence: Confidence score for type inference
    """
    label: Optional[str] = None
    selector: Optional[str] = None
    field_type: FieldType = FieldType.UNKNOWN
    value: Optional[str] = None
    placeholder: Optional[str] = None
    required: bool = False
    options: List[Dict[str, str]] = field(default_factory=list)
    confidence: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "label": self.label,
            "selector": self.selector,
            "type": self.field_type.value,
            "value": self.value,
            "placeholder": self.placeholder,
            "required": self.required,
            "options": self.options,
            "confidence": self.confidence
        }


class SemanticMatcher:
    """
    Matches form labels to input fields using semantic analysis.
    
    Uses pattern matching and keyword analysis to associate labels
    with their corresponding input fields.
    """
    
    # Field type keywords for inference
    FIELD_PATTERNS = {
        FieldType.EMAIL: [
            r'email', r'e[-\s]?mail', r'mail\s*addr',
            r'correo', r'\bemail\b'
        ],
        FieldType.PASSWORD: [
            r'password', r'pass[-\s]?word', r'contrasena',
            r'pwd', r'pass\b'
        ],
        FieldType.USERNAME: [
            r'username', r'user[-\s]?name', r'usuario',
            r'login', r'account[-\s]?name', r'user\s*id'
        ],
        FieldType.NAME: [
            r'^name$', r'full[-\s]?name', r'your[-\s]?name',
            r'nombre', r'first[-\s]?name', r'last[-\s]?name'
        ],
        FieldType.PHONE: [
            r'phone', r'telephone', r'tel', r'mobile',
            r'cell', r'fax', r'celular', r'telefono'
        ],
        FieldType.ADDRESS: [
            r'address', r'street', r'direccion',
            r'addr\b', r'line\s*1', r'line\s*2'
        ],
        FieldType.CITY: [
            r'city', r'town', r'ciudad', r'municipality'
        ],
        FieldType.STATE: [
            r'state', r'province', r'region', r'estado',
            r'provincia', r'county'
        ],
        FieldType.ZIP: [
            r'zip', r'postal[-\s]?code', r'postcode',
            r'codigo[-\s]?postal', r'cp\b'
        ],
        FieldType.COUNTRY: [
            r'country', r'nation', r'pais'
        ],
        FieldType.CREDIT_CARD: [
            r'card[-\s]?number', r'credit[-\s]?card',
            r'tarjeta', r'cc[-\s]?number', r'card\s*#',
            r'cardnum'
        ],
        FieldType.CVV: [
            r'cvv', r'cvc', r'security[-\s]?code',
            r'card[-\s]?code', r'cvv2', r'csc'
        ],
        FieldType.EXPIRY: [
            r'expir', r'exp[-\s]?date', r'expiration',
            r'valid[-\s]?thru', r'expiry', r'vencimiento'
        ],
        FieldType.SEARCH: [
            r'search', r'query', r'keyword', r'buscar',
            r'busqueda'
        ],
        FieldType.URL: [
            r'url', r'website', r'web[-\s]?site', r'link',
            r'domain', r'homepage'
        ]
    }
    
    def __init__(self):
        """Initialize the semantic matcher."""
        logger.info("SemanticMatcher initialized")
    
    def match_label_to_input(self, label: str, inputs: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """
        Match a label text to the most appropriate input field.
        
        Args:
            label: The label text to match
            inputs: List of available input fields
            
        Returns:
            Best matching input field or None
        """
        if not inputs:
            return None
        
        label_lower = label.lower().strip()
        best_match = None
        best_score = 0.0
        
        for inp in inputs:
            score = self._calculate_match_score(label_lower, inp)
            if score > best_score:
                best_score = score
                best_match = inp
        
        logger.debug(f"Matched label '{label}' with score {best_score}")
        return best_match if best_score > 0.3 else None
    
    def _calculate_match_score(self, label: str, inp: Dict[str, Any]) -> float:
        """Calculate match score between label and input."""
        score = 0.0
        
        # Check input name
        name = (inp.get('name') or '').lower()
        if name and name in label:
            score += 0.5
        
        # Check input id
        input_id = (inp.get('id') or '').lower()
        if input_id and input_id in label:
            score += 0.4
        
        # Check placeholder
        placeholder = (inp.get('placeholder') or '').lower()
        if placeholder and placeholder in label:
            score += 0.3
        
        # Check aria-label
        aria_label = (inp.get('aria-label') or '').lower()
        if aria_label and aria_label in label:
            score += 0.4
        
        # Check for keyword matches
        for field_type, patterns in self.FIELD_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, label, re.IGNORECASE):
                    input_type = inp.get('type', '').lower()
                    if self._type_matches(input_type, field_type):
                        score += 0.3
                    break
        
        return min(score, 1.0)
    
    def _type_matches(self, input_type: str, field_type: FieldType) -> bool:
        """Check if input type matches field type."""
        type_mapping = {
            'email': FieldType.EMAIL,
            'password': FieldType.PASSWORD,
            'tel': FieldType.PHONE,
            'number': FieldType.NUMBER,
            'date': FieldType.DATE,
            'checkbox': FieldType.CHECKBOX,
            'radio': FieldType.RADIO,
            'url': FieldType.URL,
            'search': FieldType.SEARCH,
            'file': FieldType.FILE,
        }
        return type_mapping.get(input_type) == field_type
    
    def infer_field_type(self, label: str, input_info: Optional[Dict[str, Any]] = None) -> Tuple[FieldType, float]:
        """
        Infer the field type from label text and input info.
        
        Args:
            label: The label text
            input_info: Optional input element info
            
        Returns:
            Tuple of (inferred type, confidence)
        """
        label_lower = label.lower()
        best_type = FieldType.UNKNOWN
        best_score = 0.0
        
        for field_type, patterns in self.FIELD_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, label_lower, re.IGNORECASE):
                    score = 0.7 if re.search(pattern, label_lower) else 0.5
                    if score > best_score:
                        best_score = score
                        best_type = field_type
                    break
        
        # Boost score if HTML input type matches
        if input_info:
            html_type = input_info.get('type', '').lower()
            type_mapping = {
                'email': FieldType.EMAIL, 'password': FieldType.PASSWORD,
                'tel': FieldType.PHONE, 'number': FieldType.NUMBER,
                'date': FieldType.DATE, 'checkbox': FieldType.CHECKBOX,
                'radio': FieldType.RADIO, 'url': FieldType.URL,
            }
            if html_type in type_mapping:
                if type_mapping[html_type] == best_type:
                    best_score = min(best_score + 0.2, 1.0)
        
        return best_type, best_score
    
    def infer_field_value(self, label: str, field_type: FieldType) -> Optional[str]:
        """
        Generate a plausible field value based on label and type.
        
        Args:
            label: The label text
            field_type: The inferred field type
            
        Returns:
            Generated placeholder value or None
        """
        if field_type == FieldType.EMAIL:
            return "example@example.com"
        elif field_type == FieldType.PHONE:
            return "+1234567890"
        elif field_type == FieldType.URL:
            return "https://example.com"
        elif field_type == FieldType.SEARCH:
            return ""
        return None


class SmartFormFiller:
    """
    Intelligently fills forms on web pages.
    
    Uses semantic matching to identify fields and fill them
    with appropriate values.
    
    Example:
        filler = SmartFormFiller()
        await filler.fill_form(page, {"email": "test@example.com", "password": "secret"})
    """
    
    def __init__(self):
        """Initialize the form filler."""
        self._matcher = SemanticMatcher()
        logger.info("SmartFormFiller initialized")
    
    async def fill_form(self, page: Any, data: Dict[str, Any]) -> Dict[str, bool]:
        """
        Fill a form with provided data.
        
        Args:
            page: Playwright page object
            data: Dictionary mapping field labels to values
            
        Returns:
            Dictionary of filled fields and success status
        """
        results = {}
        
        # Extract form fields
        forms = await self._extract_forms(page)
        
        for form in forms:
            for label, value in data.items():
                # Find matching input
                matched_input = self._matcher.match_label_to_input(label, form.get('inputs', []))
                
                if matched_input and matched_input.get('selector'):
                    selector = matched_input['selector']
                    input_type = matched_input.get('type', 'text')
                    
                    try:
                        if input_type in ['checkbox', 'check']:
                            if value:
                                await page.check(selector)
                        elif input_type in ['radio']:
                            await page.click(f"{selector}[value='{value}']")
                        elif matched_input.get('tag') == 'SELECT':
                            await page.select_option(selector, value)
                        else:
                            await page.fill(selector, str(value))
                        
                        results[label] = True
                        logger.info(f"Filled field '{label}' with value")
                    except Exception as e:
                        logger.error(f"Failed to fill '{label}': {e}")
                        results[label] = False
                else:
                    results[label] = False
        
        return results
    
    async def _extract_forms(self, page: Any) -> List[Dict[str, Any]]:
        """Extract form information from page."""
        script = """
        () => {
            const forms = document.querySelectorAll('form');
            return Array.from(forms).map(form => {
                const inputs = [];
                form.querySelectorAll('input, select, textarea').forEach(el => {
                    inputs.push({
                        name: el.name || null,
                        id: el.id || null,
                        type: el.type || el.tagName.toLowerCase(),
                        tag: el.tagName.toLowerCase(),
                        placeholder: el.placeholder || null,
                        selector: el.id ? '#' + el.id : (el.name ? el.tagName.toLowerCase() + '[name="' + el.name + '"]' : null)
                    });
                });
                return { inputs };
            });
        }
        """
        try:
            return await page.evaluate(script)
        except Exception as e:
            logger.error(f"Failed to extract forms: {e}")
            return []
    
    async def auto_fill_login(self, page: Any, credentials: Dict[str, str]) -> bool:
        """
        Auto-fill a login form.
        
        Args:
            page: Playwright page object
            credentials: Dict with 'username'/'email' and 'password' keys
            
        Returns:
            True if successful
        """
        result = await self.fill_form(page, credentials)
        
        # Try to find and click submit button
        try:
            submit_selectors = [
                'button[type="submit"]',
                'input[type="submit"]',
                'button:has-text("Login")',
                'button:has-text("Sign in")',
                'button:has-text("Log in")'
            ]
            
            for selector in submit_selectors:
                try:
                    await page.click(selector, timeout=2000)
                    logger.info("Clicked submit button")
                    return True
                except:
                    continue
            
            logger.warning("Could not find submit button")
            return any(result.values())
        except Exception as e:
            logger.error(f"Failed to submit login: {e}")
            return False
    
    async def fill_with_ai(self, page: Any, page_description: str, 
                          user_data: Optional[Dict[str, str]] = None) -> Dict[str, bool]:
        """
        Fill form using AI understanding of page description.
        
        Args:
            page: Playwright page object
            page_description: Description of the form/page
            user_data: Optional predefined user data
            
        Returns:
            Dictionary of filled fields
        """
        # Extract available fields
        forms = await self._extract_forms(page)
        
        results = {}
        user_data = user_data or {}
        
        # Common field mappings
        common_mappings = {
            'email': ['email', 'e-mail', 'correo'],
            'password': ['password', 'contrasena', 'clave'],
            'username': ['username', 'usuario', 'login name'],
            'first_name': ['first name', 'nombre', 'given name'],
            'last_name': ['last name', 'apellido', 'surname', 'family name'],
            'phone': ['phone', 'telefono', 'mobile', 'tel'],
            'address': ['address', 'direccion', 'street'],
            'city': ['city', 'ciudad', 'town'],
            'state': ['state', 'provincia', 'region'],
            'zip': ['zip', 'postal', 'codigo postal', 'cp'],
            'country': ['country', 'pais', 'nation']
        }
        
        for form in forms:
            for inp in form.get('inputs', []):
                label = inp.get('name') or inp.get('id') or inp.get('placeholder') or ''
                selector = inp.get('selector')
                
                if not selector:
                    continue
                
                # Try to match label to user data
                for data_key, label_patterns in common_mappings.items():
                    if any(pattern.lower() in label.lower() for pattern in label_patterns):
                        if data_key in user_data:
                            try:
                                await page.fill(selector, user_data[data_key])
                                results[label] = True
                                break
                            except:
                                results[label] = False
        
        return results
    
    async def extract_form_schema(self, page: Any) -> List[Dict[str, Any]]:
        """
        Extract form schema with inferred types.
        
        Args:
            page: Playwright page object
            
        Returns:
            List of form field schemas
        """
        forms = await self._extract_forms(page)
        schemas = []
        
        for form in forms:
            for inp in form.get('inputs', []):
                label = inp.get('name') or inp.get('id') or inp.get('placeholder') or ''
                field_type, confidence = self._matcher.infer_field_type(label, inp)
                
                schemas.append({
                    "label": label,
                    "type": field_type.value,
                    "confidence": confidence,
                    "selector": inp.get('selector'),
                    "required": False  # Would need to check HTML attributes
                })
        
        return schemas
