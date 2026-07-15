"""
Topic-based guardrails for content classification and filtering.

This module provides comprehensive topic-based content filtering including:
- Political content detection
- Violence and harmful content classification
- Drug and illegal substance detection
- Illegal activity identification
- Keyword and regex pattern matching
- Confidence scoring and violation levels

Author: AGI Unified Framework Team
"""

from enum import Enum, auto
from dataclasses import datacaclass, field
from typing import Dict, List, Optional, Set, Tuple, Pattern, Any
import re
import html
import unicodedata


class ViolationLevel(Enum):
    """
    Violation severity levels for content classification.
    
    ALLOW: Content is safe and should be permitted
    WARN: Content requires review or contains sensitive topics
    BLOCK: Content violates safety policies and should be blocked
    """
    ALLOW = "allow"
    WARN = "warn"
    BLOCK = "block"


class TopicCategory(Enum):
    """
    Categories of topics for content classification.
    """
    # Violence-related content
    VIOLENCE = auto()
    GRAPHIC_VIOLENCE = auto()
    WEAPONS = auto()
    
    # Political content
    POLITICS = auto()
    ELECTIONS = auto()
    GOVERNANCE = auto()
    
    # Drug and substance content
    DRUGS = auto()
    ILLEGAL_SUBSTANCES = auto()
    PRESCRIPTION_DRUGS = auto()
    
    # Illegal activities
    ILLEGAL_ACTIVITY = auto()
    FRAUD = auto()
    HATE_SPEECH = auto()
    
    # Adult content
    ADULT_CONTENT = auto()
    SEXUAL = auto()
    
    # Harassment
    HARASSMENT = auto()
    BULLYING = auto()
    
    # Self-harm
    SELF_HARM = auto()
    SUICIDE = auto()
    
    # Misinformation
    MISINFORMATION = auto()
    CONSPIRACY = auto()
    
    # Safe content
    SAFE = auto()


@dataclass
class GuardrailConfig:
    """
    Configuration for topic-based guardrails.
    
    Attributes:
        enabled_categories: Set of topic categories to check
        block_threshold: Minimum violation level to block
        warn_threshold: Minimum violation level to warn
        confidence_threshold: Minimum confidence to trigger action
        keyword_weight: Weight for keyword matches (0-1)
        regex_weight: Weight for regex pattern matches (0-1)
        context_window: Characters to analyze around matches
        allow_partial_matches: Whether partial word matches count
        case_sensitive: Whether keyword matching is case-sensitive
        custom_patterns: Additional regex patterns to check
        custom_keywords: Additional keywords to check
    """
    enabled_categories: Set[TopicCategory] = field(default_factory=lambda: {
        TopicCategory.VIOLENCE,
        TopicCategory.POLITICS,
        TopicCategory.DRUGS,
        TopicCategory.ILLEGAL_ACTIVITY,
        TopicCategory.HATE_SPEECH,
        TopicCategory.SELF_HARM,
    })
    block_threshold: ViolationLevel = ViolationLevel.BLOCK
    warn_threshold: ViolationLevel = ViolationLevel.WARN
    confidence_threshold: float = 0.6
    keyword_weight: float = 0.7
    regex_weight: float = 0.8
    context_window: int = 50
    allow_partial_matches: bool = False
    case_sensitive: bool = False
    custom_patterns: Dict[str, Pattern[str]] = field(default_factory=dict)
    custom_keywords: Dict[str, List[str]] = field(default_factory=dict)
    
    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if not 0 <= self.confidence_threshold <= 1:
            raise ValueError("confidence_threshold must be between 0 and 1")
        if not 0 <= self.keyword_weight <= 1:
            raise ValueError("keyword_weight must be between 0 and 1")
        if not 0 <= self.regex_weight <= 1:
            raise ValueError("regex_weight must be between 0 and 1")
        if self.context_window < 0:
            raise ValueError("context_window must be non-negative")


@dataclass
class GuardrailResult:
    """
    Result of content classification by guardrails.
    
    Attributes:
        is_violation: Whether content violated any policy
        level: The severity level of the violation
        categories: Categories that were triggered
        confidence: Confidence score of the classification
        matches: List of (category, matched_text, match_type) tuples
        context: Surrounding context for matches
        message: Human-readable message about the result
        metadata: Additional metadata about the classification
    """
    is_violation: bool
    level: ViolationLevel
    categories: List[TopicCategory]
    confidence: float
    matches: List[Tuple[TopicCategory, str, str]] = field(default_factory=list)
    context: Dict[TopicCategory, List[str]] = field(default_factory=dict)
    message: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_match(
        self,
        category: TopicCategory,
        matched_text: str,
        match_type: str,
        config: GuardrailConfig
    ) -> None:
        """Add a match and update context."""
        self.matches.append((category, matched_text, match_type))
        
        # Extract context around the match
        if category not in self.context:
            self.context[category] = []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "is_violation": self.is_violation,
            "level": self.level.value,
            "categories": [c.name for c in self.categories],
            "confidence": self.confidence,
            "matches": [
                {"category": m[0].name, "text": m[1], "type": m[2]}
                for m in self.matches
            ],
            "context": {
                k.name: v for k, v in self.context.items()
            },
            "message": self.message,
            "metadata": self.metadata,
        }


class KeywordMatcher:
    """
    Keyword-based content matcher.
    
    Supports exact and partial matching, case sensitivity options,
    and weighted keyword scoring.
    """
    
    # Default keyword dictionaries by category
    DEFAULT_KEYWORDS: Dict[TopicCategory, List[str]] = {
        TopicCategory.VIOLENCE: [
            "kill", "murder", "assault", "attack", "beat", "torture",
            "abuse", "harm", "injure", "maim", "brutal", "violent",
            "bloodshed", "massacre", "lynch", "execute", "slay",
            "fight", "brawl", "riot", "stab", "shoot", "bomb",
        ],
        TopicCategory.GRAPHIC_VIOLENCE: [
            "gore", "disembowel", "mutilate", "decapitate", "dismember",
            "torn apart", "blood", "intestines", "skull crush",
            "guts", "entrails", "severed", "burned alive",
        ],
        TopicCategory.WEAPONS: [
            "gun", "rifle", "pistol", "shotgun", "automatic weapon",
            "bomb", "explosive", "grenade", "mine", "rocket",
            "missile", "knife", "blade", "sword", "axe", "dagger",
            "firearm", "ammunition", "firecracker",
        ],
        TopicCategory.POLITICS: [
            "politician", "government", "democracy", "republican",
            "democrat", "liberal", "conservative", "election",
            "vote", "campaign", "policy", "legislation", "senator",
            "congress", "parliament", "president", "prime minister",
        ],
        TopicCategory.ELECTIONS: [
            "ballot", "polling", "electoral", "census", "referendum",
            "primary", "convention", "nominate", "campaign finance",
        ],
        TopicCategory.GOVERNANCE: [
            "law", "regulation", "bill", "amendment", "constitution",
            "court", "judge", "ruling", "verdict", "statute",
        ],
        TopicCategory.DRUGS: [
            "cocaine", "heroin", "meth", "marijuana", "cannabis",
            "lsd", "ecstasy", "mdma", "amphetamine", "fentanyl",
            "opioid", "narcotic", "dealer", "traffick", "smuggle",
        ],
        TopicCategory.ILLEGAL_SUBSTANCES: [
            "synthetic drug", "date rape", "controlled substance",
            "illegal drug", "street drug", "hard drug",
        ],
        TopicCategory.PRESCRIPTION_DRUGS: [
            "prescription", "medication", "dosage", "pill", "tablet",
            "pharmaceutical", "rx", "refill", "doctor",
        ],
        TopicCategory.ILLEGAL_ACTIVITY: [
            "illegal", "unlawful", "crime", "criminal", "fraud",
            "theft", "robbery", "burglary", "embezzlement", "bribery",
            "corruption", "racketeering", "extortion", "blackmail",
        ],
        TopicCategory.FRAUD: [
            "scam", "phishing", "identity theft", "Ponzi",
            "pyramid scheme", "counterfeit", "forgery", "falsify",
        ],
        TopicCategory.HATE_SPEECH: [
            "hate", "racist", "sexist", "homophobic", "transphobic",
            "antisemitic", "islamophobic", "discrimination",
            "bigotry", "prejudice", "intolerance",
        ],
        TopicCategory.ADULT_CONTENT: [
            "nsfw", "explicit", "porn", "xxx", "adult content",
            "nude", "naked", "sexual act",
        ],
        TopicCategory.SEXUAL: [
            "sex", "intercourse", "orgasm", "fetish", "kinky",
        ],
        TopicCategory.HARASSMENT: [
            "harass", "stalking", "intimidate", "threaten", "bully",
        ],
        TopicCategory.BULLYING: [
            "bullying", "cyberbullying", "teasing", "mocking",
            "humiliate", "taunt", "torment",
        ],
        TopicCategory.SELF_HARM: [
            "self-harm", "cutting", "self-injury", "suicidal",
            "end my life", "want to die", "kill myself",
        ],
        TopicCategory.SUICIDE: [
            "suicide", "suicidal ideation", "overdose", "hang myself",
            "jump off", "shoot myself", "slit my wrists",
        ],
        TopicCategory.MISINFORMATION: [
            "fake news", "false claim", "misinformation", "disinformation",
            "hoax", "conspiracy theory",
        ],
        TopicCategory.CONSPIRACY: [
            "conspiracy", "cover-up", "cover up", "secret plot",
            "hidden agenda", "new world order",
        ],
    }
    
    def __init__(
        self,
        keywords: Optional[Dict[TopicCategory, List[str]]] = None,
        case_sensitive: bool = False,
        allow_partial: bool = False,
    ) -> None:
        """
        Initialize keyword matcher.
        
        Args:
            keywords: Custom keyword dictionary by category
            case_sensitive: Whether matching is case-sensitive
            allow_partial: Whether partial word matches count
        """
        self.keywords = keywords or self.DEFAULT_KEYWORDS.copy()
        self.case_sensitive = case_sensitive
        self.allow_partial = allow_partial
        
        # Build lookup structures
        self._build_lookup_structures()
    
    def _build_lookup_structures(self) -> None:
        """Build optimized lookup structures for matching."""
        self._exact_lookup: Dict[str, Tuple[TopicCategory, int]] = {}
        self._word_set: Set[str] = set()
        self._category_keywords: Dict[TopicCategory, Set[str]] = {}
        
        for category, word_list in self.keywords.items():
            self._category_keywords[category] = set()
            
            for keyword in word_list:
                # Normalize keyword
                normalized = keyword if self.case_sensitive else keyword.lower()
                
                if self.allow_partial:
                    self._word_set.add(normalized)
                else:
                    # For exact matching, store word boundary version
                    self._exact_lookup[normalized] = (category, len(keyword))
                
                self._category_keywords[category].add(normalized)
    
    def match(
        self,
        text: str,
        categories: Optional[Set[TopicCategory]] = None
    ) -> List[Tuple[TopicCategory, str, float]]:
        """
        Find keyword matches in text.
        
        Args:
            text: Text to analyze
            categories: Specific categories to check (None for all)
        
        Returns:
            List of (category, matched_keyword, confidence) tuples
        """
        matches: List[Tuple[TopicCategory, str, float]] = []
        
        # Normalize text
        check_text = text if self.case_sensitive else text.lower()
        words = set(check_text.split())
        
        # Check categories
        categories_to_check = (
            categories if categories is not None
            else set(self.keywords.keys())
        )
        
        for category in categories_to_check:
            if category not in self.keywords:
                continue
            
            for keyword in self.keywords[category]:
                if self.allow_partial:
                    # Check if keyword is contained in text
                    if keyword in check_text:
                        confidence = self._calculate_confidence(keyword, text)
                        matches.append((category, keyword, confidence))
                else:
                    # Check for word boundary match
                    pattern = r'\b' + re.escape(keyword) + r'\b'
                    if re.search(pattern, check_text):
                        confidence = self._calculate_confidence(keyword, text)
                        matches.append((category, keyword, confidence))
        
        return matches
    
    def _calculate_confidence(self, keyword: str, text: str) -> float:
        """
        Calculate confidence score for a match.
        
        Higher confidence for:
        - Longer keywords (more specific)
        - Keywords appearing multiple times
        - Keywords in longer text (more context)
        """
        # Base confidence on keyword length
        length_factor = min(len(keyword) / 10.0, 1.0)
        
        # Count occurrences
        count = text.lower().count(keyword.lower())
        count_factor = min(count / 3.0, 1.0)
        
        # Text length factor
        length_factor_text = min(len(text) / 1000.0, 1.0)
        
        # Combined confidence
        confidence = (
            0.4 * length_factor +
            0.4 * count_factor +
            0.2 * length_factor_text
        )
        
        return min(confidence, 1.0)


class RegexMatcher:
    """
    Regex-based content matcher for complex pattern detection.
    
    Supports compilation and caching of regex patterns,
    as well as pattern weighting and confidence scoring.
    """
    
    # Default regex patterns by category
    DEFAULT_PATTERNS: Dict[TopicCategory, List[str]] = {
        TopicCategory.VIOLENCE: [
            r'\b(kill|murder|assassinate)\s+(me|my|us|our)\b',
            r'\b(beat|hurt|harm|attack)\s+(me|my|us|our)\b',
            r'\b(how\s+to|instructions?\s+for)\s+(kill|murder|harm)\b',
            r'\b(violent|violent)\s+(attack|act|behavior)\b',
        ],
        TopicCategory.WEAPONS: [
            r'\b(how\s+to|make|build|create|construct)\s+(bomb|explosive|weapon)\b',
            r'\b(buy|purchase|obtain)\s+(gun|rifle|firearm)\b',
            r'\b(detonate|detonation)\s+(device|bomb)\b',
        ],
        TopicCategory.DRUGS: [
            r'\b(how\s+to|make|produce|synthesize)\s+(cocaine|heroin|meth)\b',
            r'\b(buy|sell|traffic)\s+(drugs?|cocaine|heroin)\b',
            r'\b(illegal|controlled)\s+(substance|drug)\b',
        ],
        TopicCategory.ILLEGAL_ACTIVITY: [
            r'\b(how\s+to|instructions?\s+for)\s+(hack|break\s+into|fraud)\b',
            r'\b(illegal|unlawful|criminal)\s+(activity|act|behavior)\b',
        ],
        TopicCategory.SELF_HARM: [
            r'\b(want|need|going\s+to)\s+(hurt|kill|end)\s+(myself|me)\b',
            r'\b(best\s+way\s+to|how\s+can\s+I)\s+(die|kill\s+myself)\b',
            r'\b(end\s+it\s+all|my\s+life\s+is|no\s+reason)\b',
        ],
        TopicCategory.SUICIDE: [
            r'\b(suicide|suicidal)\s+(method|plan|thought)\b',
            r'\b(overdose|pills|drugs?)\s+(to\s+die|to\s+kill)\b',
        ],
        TopicCategory.HATE_SPEECH: [
            r'\b(hate|kill|attack)\s+(all|those)\s+([a-z]+\s+)?people\b',
            r'\b([a-z]+\s+)?should\s+(die|be\s+killed|be\s+harmed)\b',
        ],
        TopicCategory.FRAUD: [
            r'\b(phishing|scam)\s+(email|link|website)\b',
            r'\b(fake|fraudulent)\s+(account|document|id)\b',
        ],
        TopicCategory.MISINFORMATION: [
            r'\b(fake|false|hoax)\s+(news|story|information)\b',
            r'\b(conspiracy|cover-up)\s+(theory|the)\b',
        ],
    }
    
    def __init__(
        self,
        patterns: Optional[Dict[TopicCategory, List[str]]] = None,
        compiled: bool = True,
    ) -> None:
        """
        Initialize regex matcher.
        
        Args:
            patterns: Custom pattern dictionary by category
            compiled: Whether to compile patterns immediately
        """
        self.patterns = patterns or self.DEFAULT_PATTERNS.copy()
        self._compiled_patterns: Dict[TopicCategory, List[Tuple[Pattern[str], str]]] = {}
        
        if compiled:
            self._compile_all()
    
    def _compile_all(self) -> None:
        """Compile all regex patterns."""
        for category, pattern_list in self.patterns.items():
            self._compiled_patterns[category] = []
            
            for pattern in pattern_list:
                try:
                    compiled = re.compile(pattern, re.IGNORECASE | re.MULTILINE)
                    self._compiled_patterns[category].append((compiled, pattern))
                except re.error as e:
                    # Log invalid pattern but continue
                    pass
    
    def add_pattern(
        self,
        category: TopicCategory,
        pattern: str,
        compile_now: bool = True
    ) -> bool:
        """
        Add a custom pattern.
        
        Args:
            category: Category to assign pattern to
            pattern: Regex pattern string
            compile_now: Whether to compile immediately
        
        Returns:
            True if pattern is valid, False otherwise
        """
        if category not in self.patterns:
            self.patterns[category] = []
        
        # Validate pattern
        try:
            re.compile(pattern)
            self.patterns[category].append(pattern)
            
            if compile_now:
                compiled = re.compile(pattern, re.IGNORECASE | re.MULTILINE)
                if category not in self._compiled_patterns:
                    self._compiled_patterns[category] = []
                self._compiled_patterns[category].append((compiled, pattern))
            
            return True
        except re.error:
            return False
    
    def match(
        self,
        text: str,
        categories: Optional[Set[TopicCategory]] = None
    ) -> List[Tuple[TopicCategory, str, float, int]]:
        """
        Find regex pattern matches in text.
        
        Args:
            text: Text to analyze
            categories: Specific categories to check (None for all)
        
        Returns:
            List of (category, matched_text, confidence, match_start) tuples
        """
        matches: List[Tuple[TopicCategory, str, float, int]] = []
        
        categories_to_check = (
            categories if categories is not None
            else set(self._compiled_patterns.keys())
        )
        
        for category in categories_to_check:
            if category not in self._compiled_patterns:
                continue
            
            for compiled, pattern_str in self._compiled_patterns[category]:
                for match in compiled.finditer(text):
                    matched_text = match.group(0)
                    start = match.start()
                    
                    # Calculate confidence
                    confidence = self._calculate_confidence(matched_text, pattern_str, text)
                    matches.append((category, matched_text, confidence, start))
        
        return matches
    
    def _calculate_confidence(
        self,
        matched_text: str,
        pattern: str,
        full_text: str
    ) -> float:
        """
        Calculate confidence for regex match.
        
        Higher confidence for:
        - Longer matches
        - More complex patterns
        - Matches in shorter text (more focused)
        """
        # Match length factor
        length_factor = min(len(matched_text) / 30.0, 1.0)
        
        # Pattern complexity (capturing groups, alternations, etc.)
        complexity = pattern.count('(') + pattern.count('|')
        complexity_factor = min(complexity / 5.0, 1.0)
        
        # Focus factor (match length vs text length)
        focus = len(matched_text) / max(len(full_text), 1)
        focus_factor = min(focus * 10, 1.0)
        
        confidence = (
            0.3 * length_factor +
            0.4 * complexity_factor +
            0.3 * focus_factor
        )
        
        return min(confidence, 1.0)


class TopicClassifier:
    """
    Main topic classifier combining keyword and regex matching.
    
    This is the primary interface for content classification.
    It combines multiple matching strategies with confidence scoring
    to determine if content violates safety policies.
    """
    
    def __init__(self, config: Optional[GuardrailConfig] = None) -> None:
        """
        Initialize topic classifier.
        
        Args:
            config: Guardrail configuration
        """
        self.config = config or GuardrailConfig()
        
        # Initialize matchers
        self.keyword_matcher = KeywordMatcher(
            case_sensitive=self.config.case_sensitive,
            allow_partial=self.config.allow_partial_matches,
        )
        
        self.regex_matcher = RegexMatcher()
        
        # Add custom keywords and patterns from config
        if self.config.custom_keywords:
            for category_name, keywords in self.config.custom_keywords.items():
                try:
                    category = TopicCategory[category_name.upper()]
                    if category not in self.keyword_matcher.keywords:
                        self.keyword_matcher.keywords[category] = []
                    self.keyword_matcher.keywords[category].extend(keywords)
                except KeyError:
                    pass
        
        if self.config.custom_patterns:
            for category_name, pattern in self.config.custom_patterns.items():
                try:
                    category = TopicCategory[category_name.upper()]
                    self.regex_matcher.add_pattern(category, pattern)
                except KeyError:
                    pass
    
    def classify(self, text: str) -> GuardrailResult:
        """
        Classify text content for policy violations.
        
        Args:
            text: Text content to classify
        
        Returns:
            GuardrailResult with classification details
        """
        # Normalize text
        normalized = self._normalize_text(text)
        
        # Collect matches
        all_matches: Dict[TopicCategory, List[Tuple[str, str, float]]] = {}
        
        # Keyword matches
        keyword_matches = self.keyword_matcher.match(
            normalized,
            self.config.enabled_categories
        )
        for category, keyword, confidence in keyword_matches:
            if category not in all_matches:
                all_matches[category] = []
            all_matches[category].append((keyword, "keyword", confidence))
        
        # Regex matches
        regex_matches = self.regex_matcher.match(
            normalized,
            self.config.enabled_categories
        )
        for category, matched_text, confidence, _ in regex_matches:
            if category not in all_matches:
                all_matches[category] = []
            all_matches[category].append((matched_text, "regex", confidence))
        
        # Calculate overall results
        categories_triggered: List[TopicCategory] = []
        total_confidence = 0.0
        weighted_scores: List[float] = []
        
        for category, matches in all_matches.items():
            # Check if any match exceeds threshold
            max_confidence = max(m[2] for m in matches)
            
            if max_confidence >= self.config.confidence_threshold:
                categories_triggered.append(category)
                
                # Calculate weighted score
                keyword_matches_count = sum(1 for m in matches if m[1] == "keyword")
                regex_matches_count = sum(1 for m in matches if m[1] == "regex")
                
                weighted = (
                    keyword_matches_count * self.config.keyword_weight +
                    regex_matches_count * self.config.regex_weight
                ) / max(keyword_matches_count + regex_matches_count, 1)
                
                weighted_scores.append(weighted * max_confidence)
        
        # Determine violation level
        if categories_triggered:
            max_confidence = max(weighted_scores) if weighted_scores else 0.0
            level = self._determine_level(categories_triggered, max_confidence)
            is_violation = level != ViolationLevel.ALLOW
            message = self._generate_message(categories_triggered, level)
        else:
            level = ViolationLevel.ALLOW
            is_violation = False
            max_confidence = 0.0
            message = "Content is safe and does not violate any policies."
        
        # Build result
        result = GuardrailResult(
            is_violation=is_violation,
            level=level,
            categories=categories_triggered,
            confidence=max_confidence,
            message=message,
        )
        
        # Add matches and context
        for category, matches in all_matches.items():
            for matched_text, match_type, confidence in matches:
                result.add_match(category, matched_text, match_type, self.config)
                
                # Extract context
                context = self._extract_context(text, matched_text)
                if context:
                    if category not in result.context:
                        result.context[category] = []
                    result.context[category].append(context)
        
        # Add metadata
        result.metadata = {
            "total_matches": sum(len(m) for m in all_matches.values()),
            "categories_checked": len(self.config.enabled_categories),
            "config_applied": {
                "confidence_threshold": self.config.confidence_threshold,
                "block_threshold": self.config.block_threshold.value,
                "warn_threshold": self.config.warn_threshold.value,
            },
        }
        
        return result
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text for matching."""
        # Decode HTML entities
        normalized = html.unescape(text)
        
        # Normalize unicode
        normalized = unicodedata.normalize("NFKC", normalized)
        
        # Remove excessive whitespace
        normalized = re.sub(r'\s+', ' ', normalized)
        
        # Lowercase if not case sensitive
        if not self.config.case_sensitive:
            normalized = normalized.lower()
        
        return normalized
    
    def _determine_level(
        self,
        categories: List[TopicCategory],
        confidence: float
    ) -> ViolationLevel:
        """Determine violation level based on categories and confidence."""
        # High-risk categories always block
        high_risk = {
            TopicCategory.SELF_HARM,
            TopicCategory.SUICIDE,
            TopicCategory.GRAPHIC_VIOLENCE,
        }
        
        medium_risk = {
            TopicCategory.VIOLENCE,
            TopicCategory.DRUGS,
            TopicCategory.ILLEGAL_SUBSTANCES,
            TopicCategory.ILLEGAL_ACTIVITY,
        }
        
        # Check categories
        if high_risk.intersection(categories):
            return ViolationLevel.BLOCK
        
        if medium_risk.intersection(categories) and confidence > 0.7:
            return ViolationLevel.BLOCK
        
        # Check thresholds
        if confidence >= 0.8:
            return ViolationLevel.BLOCK
        elif confidence >= self.config.confidence_threshold:
            return ViolationLevel.WARN
        
        return ViolationLevel.ALLOW
    
    def _generate_message(
        self,
        categories: List[TopicCategory],
        level: ViolationLevel
    ) -> str:
        """Generate human-readable message for classification result."""
        category_names = [c.name.replace("_", " ").title() for c in categories]
        
        if level == ViolationLevel.BLOCK:
            return (
                f"Content blocked: Contains potentially harmful topics "
                f"({', '.join(category_names)}). "
                f"Manual review required."
            )
        elif level == ViolationLevel.WARN:
            return (
                f"Content flagged for review: Contains sensitive topics "
                f"({', '.join(category_names)}). "
                f"Please verify the context."
            )
        else:
            return "Content appears safe."
    
    def _extract_context(self, text: str, match: str, window: Optional[int] = None) -> str:
        """Extract context around a match."""
        window = window or self.config.context_window
        
        try:
            index = text.lower().index(match.lower())
        except ValueError:
            return ""
        
        start = max(0, index - window)
        end = min(len(text), index + len(match) + window)
        
        context = text[start:end]
        
        # Add ellipsis if truncated
        if start > 0:
            context = "..." + context
        if end < len(text):
            context = context + "..."
        
        return context.strip()
    
    def update_config(self, config: GuardrailConfig) -> None:
        """
        Update guardrail configuration.
        
        Args:
            config: New guardrail configuration
        """
        self.config = config
        
        # Re-initialize matchers with new config
        self.keyword_matcher = KeywordMatcher(
            case_sensitive=config.case_sensitive,
            allow_partial=config.allow_partial_matches,
        )
    
    def get_supported_categories(self) -> List[TopicCategory]:
        """Get list of all supported topic categories."""
        return list(TopicCategory)
    
    def get_category_info(self, category: TopicCategory) -> Dict[str, Any]:
        """
        Get information about a specific category.
        
        Args:
            category: Topic category to query
        
        Returns:
            Dictionary with category information
        """
        keywords = self.keyword_matcher.keywords.get(category, [])
        patterns = self.regex_matcher.patterns.get(category, [])
        
        return {
            "category": category.name,
            "keyword_count": len(keywords),
            "pattern_count": len(patterns),
            "enabled": category in self.config.enabled_categories,
            "sample_keywords": keywords[:5],
            "sample_patterns": patterns[:3],
        }


def create_default_classifier() -> TopicClassifier:
    """
    Create a classifier with default configuration.
    
    Returns:
        TopicClassifier with default GuardrailConfig
    """
    return TopicClassifier(GuardrailConfig())


def create_strict_classifier() -> TopicClassifier:
    """
    Create a classifier with strict configuration.
    
    Returns:
        TopicClassifier with strict GuardrailConfig
    """
    config = GuardrailConfig(
        confidence_threshold=0.4,
        keyword_weight=0.8,
        regex_weight=0.9,
        enabled_categories=set(TopicCategory),
    )
    return TopicClassifier(config)


def create_permissive_classifier() -> TopicClassifier:
    """
    Create a classifier with permissive configuration.
    
    Returns:
        TopicClassifier with permissive GuardrailConfig
    """
    config = GuardrailConfig(
        confidence_threshold=0.8,
        keyword_weight=0.5,
        regex_weight=0.6,
        enabled_categories={
            TopicCategory.VIOLENCE,
            TopicCategory.SELF_HARM,
            TopicCategory.SUICIDE,
        },
    )
    return TopicClassifier(config)
