"""
OCR Language Detection Module

Provides language detection for OCR text:
- Character set analysis (Unicode ranges for CJK, Latin, Arabic, Devanagari, etc.)
- N-gram frequency matching
- Script detection

Supports 20+ languages. Pure Python standard library only.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Tuple, Dict, Optional, Set, Any


class Script(Enum):
    """Unicode script categories."""
    LATIN = "latin"
    CYRILLIC = "cyrillic"
    ARABIC = "arabic"
    DEVANAGARI = "devanagari"
    CJK = "cjk"
    HANGUL = "hangul"
    HIRAGANA = "hiragana"
    KATAKANA = "katakana"
    THAI = "thai"
    TELUGU = "telugu"
    BENGALI = "bengali"
    TAMIL = "tamil"
    GUJARATI = "gujarati"
    KANNADA = "kannada"
    MALAYALAM = "malayalam"
    PUNJABI = "punjabi"
    GEORGIAN = "georgian"
    ARMENIAN = "armenian"
    HEBREW = "hebrew"
    ETHIOPIC = "ethiopic"
    GREEK = "greek"
    UNKNOWN = "unknown"


# Unicode range definitions for each script
SCRIPT_RANGES: Dict[Script, List[Tuple[int, int]]] = {
    Script.LATIN: [
        (0x0041, 0x005A), (0x0061, 0x007A), (0x00C0, 0x00FF),
        (0x0100, 0x017F), (0x0180, 0x024F), (0x1E00, 0x1EFF),
    ],
    Script.CYRILLIC: [
        (0x0400, 0x04FF), (0x0500, 0x052F),
    ],
    Script.ARABIC: [
        (0x0600, 0x06FF), (0x0750, 0x077F), (0x08A0, 0x08FF),
        (0xFB50, 0xFDFF), (0xFE70, 0xFEFF),
    ],
    Script.DEVANAGARI: [
        (0x0900, 0x097F),
    ],
    Script.CJK: [
        (0x4E00, 0x9FFF), (0x3400, 0x4DBF), (0x20000, 0x2A6DF),
        (0xF900, 0xFAFF), (0x2F800, 0x2FA1F),
    ],
    Script.HANGUL: [
        (0xAC00, 0xD7AF), (0x1100, 0x11FF), (0x3130, 0x318F),
    ],
    Script.HIRAGANA: [
        (0x3040, 0x309F),
    ],
    Script.KATAKANA: [
        (0x30A0, 0x30FF), (0x31F0, 0x31FF),
    ],
    Script.THAI: [
        (0x0E00, 0x0E7F),
    ],
    Script.TELUGU: [
        (0x0C00, 0x0C7F),
    ],
    Script.BENGALI: [
        (0x0980, 0x09FF),
    ],
    Script.TAMIL: [
        (0x0B80, 0x0BFF),
    ],
    Script.GUJARATI: [
        (0x0A80, 0x0AFF),
    ],
    Script.KANNADA: [
        (0x0C80, 0x0CFF),
    ],
    Script.MALAYALAM: [
        (0x0D00, 0x0D7F),
    ],
    Script.PUNJABI: [
        (0x0A00, 0x0A7F),
    ],
    Script.GEORGIAN: [
        (0x10A0, 0x10FF),
    ],
    Script.ARMENIAN: [
        (0x0530, 0x058F),
    ],
    Script.HEBREW: [
        (0x0590, 0x05FF),
    ],
    Script.ETHIOPIC: [
        (0x1200, 0x137F),
    ],
    Script.GREEK: [
        (0x0370, 0x03FF), (0x1F00, 0x1FFF),
    ],
}

# Language to script mapping
LANGUAGE_SCRIPT_MAP: Dict[str, Script] = {
    "en": Script.LATIN, "fr": Script.LATIN, "de": Script.LATIN,
    "es": Script.LATIN, "it": Script.LATIN, "pt": Script.LATIN,
    "nl": Script.LATIN, "pl": Script.LATIN, "cs": Script.LATIN,
    "ro": Script.LATIN, "sv": Script.LATIN, "da": Script.LATIN,
    "fi": Script.LATIN, "no": Script.LATIN, "hu": Script.LATIN,
    "tr": Script.LATIN, "vi": Script.LATIN, "id": Script.LATIN,
    "ru": Script.CYRILLIC, "uk": Script.CYRILLIC, "bg": Script.CYRILLIC,
    "sr": Script.CYRILLIC,
    "ar": Script.ARABIC, "fa": Script.ARABIC, "ur": Script.ARABIC,
    "hi": Script.DEVANAGARI,
    "zh": Script.CJK, "ja": Script.CJK, "ko": Script.HANGUL,
    "th": Script.THAI,
    "ta": Script.TAMIL, "te": Script.TELUGU, "bn": Script.BENGALI,
    "gu": Script.GUJARATI, "kn": Script.KANNADA, "ml": Script.MALAYALAM,
    "pa": Script.PUNJABI,
    "el": Script.GREEK,
    "he": Script.HEBREW,
    "ka": Script.GEORGIAN,
    "hy": Script.ARMENIAN,
    "am": Script.ETHIOPIC,
}

# Common character n-grams for various languages (top frequency n-grams)
# Used for n-gram frequency matching
LANGUAGE_NGRAMS: Dict[str, Dict[str, float]] = {
    "en": {
        "th": 3.56, "he": 3.07, "in": 2.43, "er": 2.05, "an": 1.99,
        "re": 1.85, "on": 1.76, "at": 1.49, "en": 1.45, "nd": 1.35,
        "the": 1.81, "and": 0.73, "ing": 0.72, "her": 0.36, "hat": 0.33,
    },
    "fr": {
        "es": 3.29, "le": 2.80, "de": 2.42, "en": 2.30, "re": 2.10,
        "on": 1.96, "ou": 1.84, "nt": 1.72, "ai": 1.62, "it": 1.57,
        "les": 1.20, "des": 0.98, "ent": 0.87, "ion": 0.82, "ment": 0.72,
    },
    "de": {
        "en": 4.02, "er": 3.50, "de": 2.80, "ei": 2.56, "ch": 2.41,
        "sc": 2.21, "sc": 2.21, "ie": 2.18, "in": 2.10, "te": 1.80,
        "ich": 1.42, "die": 1.20, "sch": 1.15, "ein": 1.10, "und": 0.98,
    },
    "es": {
        "en": 3.40, "es": 2.98, "de": 2.52, "os": 2.20, "la": 2.10,
        "el": 1.95, "on": 1.80, "ar": 1.70, "co": 1.55, "er": 1.50,
        "los": 1.10, "del": 0.95, "que": 0.90, "con": 0.85, "ent": 0.78,
    },
    "ru": {
        "st": 2.80, "no": 2.60, "na": 2.50, "to": 2.30, "en": 2.20,
        "ov": 2.00, "ni": 1.90, "ko": 1.80, "ol": 1.70, "pr": 1.60,
        "ost": 1.20, "nov": 0.95, "pro": 0.88, "ego": 0.80, "rav": 0.72,
    },
    "zh": {
        "de": 3.50, "shi": 2.80, "le": 2.60, "yi": 2.40, "he": 2.20,
        "bu": 2.00, "zhong": 1.50, "guo": 1.30, "ren": 1.10, "zai": 0.95,
    },
    "ja": {
        "ni": 3.20, "wo": 2.90, "no": 2.70, "to": 2.50, "de": 2.30,
        "wa": 2.10, "ga": 1.90, "te": 1.70, "shi": 1.50, "suru": 1.20,
    },
    "ko": {
        "eu": 3.50, "da": 2.80, "eo": 2.60, "gi": 2.40, "se": 2.20,
        "do": 2.00, "ha": 1.80, "go": 1.60, "i": 1.40, "eul": 1.20,
    },
    "ar": {
        "al": 4.50, "li": 2.80, "wa": 2.60, "an": 2.40, "ah": 2.20,
        "la": 2.00, "in": 1.80, "ya": 1.60, "bi": 1.40, "ma": 1.20,
    },
    "hi": {
        "ka": 3.80, "ke": 2.90, "ki": 2.70, "ra": 2.50, "na": 2.30,
        "me": 2.10, "se": 1.90, "ha": 1.70, "pa": 1.50, "ya": 1.30,
    },
    "th": {
        "th": 3.50, "na": 2.80, "an": 2.60, "ai": 2.40, "ri": 2.20,
        "on": 2.00, "ka": 1.80, "ro": 1.60, "en": 1.40, "am": 1.20,
    },
}

# Language names for display
LANGUAGE_NAMES: Dict[str, str] = {
    "en": "English", "fr": "French", "de": "German", "es": "Spanish",
    "it": "Italian", "pt": "Portuguese", "nl": "Dutch", "pl": "Polish",
    "cs": "Czech", "ro": "Romanian", "sv": "Swedish", "da": "Danish",
    "fi": "Finnish", "no": "Norwegian", "hu": "Hungarian", "tr": "Turkish",
    "vi": "Vietnamese", "id": "Indonesian", "ru": "Russian", "uk": "Ukrainian",
    "bg": "Bulgarian", "sr": "Serbian", "ar": "Arabic", "fa": "Persian",
    "ur": "Urdu", "hi": "Hindi", "zh": "Chinese", "ja": "Japanese",
    "ko": "Korean", "th": "Thai", "ta": "Tamil", "te": "Telugu",
    "bn": "Bengali", "gu": "Gujarati", "kn": "Kannada", "ml": "Malayalam",
    "pa": "Punjabi", "el": "Greek", "he": "Hebrew", "ka": "Georgian",
    "hy": "Armenian", "am": "Amharic",
}


@dataclass
class DetectionResult:
    """Result of language detection."""
    language_code: str
    language_name: str
    confidence: float
    script: Script
    all_scores: Dict[str, float] = field(default_factory=dict)


class UnicodeRangeAnalyzer:
    """
    Analyze Unicode character ranges to determine the script(s) present in text.

    Counts characters falling into each Unicode range and determines
    the dominant script.
    """

    def __init__(self) -> None:
        self._build_range_index()

    def _build_range_index(self) -> None:
        """Build a sorted index of all Unicode ranges for fast lookup."""
        self._ranges: List[Tuple[int, int, Script]] = []
        for script, ranges in SCRIPT_RANGES.items():
            for start, end in ranges:
                self._ranges.append((start, end, script))
        self._ranges.sort(key=lambda x: x[0])

    def analyze(self, text: str) -> Dict[Script, int]:
        """
        Analyze text and count characters per script.

        Returns a dictionary mapping each detected script to its character count.
        """
        counts: Dict[Script, int] = {}
        for char in text:
            code = ord(char)
            script = self._identify_char_script(code)
            if script != Script.UNKNOWN:
                counts[script] = counts.get(script, 0) + 1
        return counts

    def _identify_char_script(self, code: int) -> Script:
        """Identify the script of a single Unicode code point."""
        # Binary search through the sorted ranges
        lo, hi = 0, len(self._ranges) - 1
        while lo <= hi:
            mid = (lo + hi) // 2
            start, end, script = self._ranges[mid]
            if code < start:
                hi = mid - 1
            elif code > end:
                lo = mid + 1
            else:
                return script
        return Script.UNKNOWN

    def get_dominant_script(self, text: str) -> Tuple[Script, float]:
        """
        Get the dominant script in the text.

        Returns (script, ratio) where ratio is the fraction of characters
        belonging to the dominant script.
        """
        counts = self.analyze(text)
        if not counts:
            return Script.UNKNOWN, 0.0

        total = sum(counts.values())
        dominant_script = max(counts, key=counts.get)
        ratio = counts[dominant_script] / total if total > 0 else 0.0
        return dominant_script, ratio

    def get_script_ratios(self, text: str) -> Dict[Script, float]:
        """Get the ratio of each script in the text."""
        counts = self.analyze(text)
        total = sum(counts.values())
        if total == 0:
            return {}
        return {script: count / total for script, count in counts.items()}

    def is_mixed_script(self, text: str, threshold: float = 0.1) -> bool:
        """Check if the text contains multiple scripts above threshold."""
        ratios = self.get_script_ratios(text)
        significant = [r for r in ratios.values() if r > threshold]
        return len(significant) > 1


class NGramProfile:
    """
    N-gram frequency profile for a language.

    Extracts character n-grams from text and computes frequency distributions.
    """

    def __init__(self, n_values: Optional[List[int]] = None) -> None:
        self.n_values = n_values or [1, 2, 3]
        self._profiles: Dict[str, Dict[str, float]] = {}

    def build_profile(self, text: str, language: str) -> Dict[str, float]:
        """
        Build an n-gram frequency profile for the given text.

        Combines n-grams of different sizes into a single frequency map.
        """
        text = self._normalize(text)
        combined: Counter = Counter()

        for n in self.n_values:
            ngrams = self._extract_ngrams(text, n)
            combined.update(ngrams)

        total = sum(combined.values())
        if total == 0:
            return {}

        profile: Dict[str, float] = {}
        for ngram, count in combined.items():
            profile[ngram] = count / total

        self._profiles[language] = profile
        return profile

    def build_profiles_from_corpus(self, corpus: Dict[str, str]) -> None:
        """Build profiles for multiple languages from a corpus dictionary."""
        for language, text in corpus.items():
            self.build_profile(text, language)

    def _extract_ngrams(self, text: str, n: int) -> List[str]:
        """Extract n-grams from text."""
        if len(text) < n:
            return []
        return [text[i:i + n] for i in range(len(text) - n + 1)]

    def _normalize(self, text: str) -> str:
        """Normalize text for n-gram extraction."""
        text = text.lower()
        # Remove non-alphabetic characters (keep spaces for word boundaries)
        text = re.sub(r'[^a-z\s\u0400-\u04ff\u0600-\u06ff\u0900-\u097f'
                       r'\u4e00-\u9fff\uac00-\ud7af\u3040-\u309f\u30a0-\u30ff'
                       r'\u0e00-\u0e7f]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def compare(self, profile1: Dict[str, float],
                profile2: Dict[str, float]) -> float:
        """
        Compare two n-gram profiles using cosine similarity.

        Returns a similarity score between 0 and 1.
        """
        all_keys = set(profile1.keys()) | set(profile2.keys())
        if not all_keys:
            return 0.0

        dot_product = sum(profile1.get(k, 0) * profile2.get(k, 0) for k in all_keys)
        norm1 = math.sqrt(sum(v * v for v in profile1.values()))
        norm2 = math.sqrt(sum(v * v for v in profile2.values()))

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)

    def get_profile(self, language: str) -> Optional[Dict[str, float]]:
        """Get the stored profile for a language."""
        return self._profiles.get(language)


class ScriptClassifier:
    """
    Classifies text into script categories using Unicode analysis.

    Handles special cases like Japanese (mixed CJK + Hiragana + Katakana)
    and Korean (Hangul).
    """

    def __init__(self) -> None:
        self.analyzer = UnicodeRangeAnalyzer()

    def classify(self, text: str) -> Tuple[Script, float]:
        """
        Classify the script of the given text.

        Returns (script, confidence).
        """
        if not text or not text.strip():
            return Script.UNKNOWN, 0.0

        counts = self.analyzer.analyze(text)
        if not counts:
            return Script.UNKNOWN, 0.0

        total = sum(counts.values())

        # Special handling for Japanese: mix of CJK, Hiragana, Katakana
        cjk_count = counts.get(Script.CJK, 0)
        hira_count = counts.get(Script.HIRAGANA, 0)
        kata_count = counts.get(Script.KATAKANA, 0)

        if cjk_count > 0 and (hira_count > 0 or kata_count > 0):
            # This is likely Japanese
            jp_count = cjk_count + hira_count + kata_count
            return Script.CJK, jp_count / total

        # Korean: Hangul characters
        if counts.get(Script.HANGUL, 0) > 0:
            return Script.HANGUL, counts[Script.HANGUL] / total

        # Get dominant script
        dominant = max(counts, key=counts.get)
        confidence = counts[dominant] / total

        return dominant, confidence

    def classify_detailed(self, text: str) -> Dict[str, Any]:
        """
        Detailed script classification with all detected scripts.

        Returns a dictionary with dominant script, confidence, and
        all script ratios.
        """
        script, confidence = self.classify(text)
        ratios = self.analyzer.get_script_ratios(text)
        is_mixed = self.analyzer.is_mixed_script(text)

        return {
            "dominant_script": script,
            "confidence": confidence,
            "script_ratios": {s.value: r for s, r in ratios.items()},
            "is_mixed_script": is_mixed,
            "char_count": len(text),
            "analyzed_count": sum(self.analyzer.analyze(text).values()),
        }


class LanguageDetector:
    """
    Language detection combining script analysis and n-gram matching.

    Supports 20+ languages using a two-stage approach:
    1. Script detection narrows down candidate languages
    2. N-gram frequency matching selects the best candidate
    """

    def __init__(self, min_text_length: int = 5,
                 use_ngram_matching: bool = True) -> None:
        self.min_text_length = min_text_length
        self.use_ngram_matching = use_ngram_matching
        self.script_classifier = ScriptClassifier()
        self.unicode_analyzer = UnicodeRangeAnalyzer()
        self.ngram_profile = NGramProfile(n_values=[2, 3])

        # Load built-in n-gram profiles
        self._load_builtin_profiles()

    def _load_builtin_profiles(self) -> None:
        """Load built-in n-gram frequency profiles."""
        for lang, ngrams in LANGUAGE_NGRAMS.items():
            self.ngram_profile._profiles[lang] = ngrams

    def detect(self, text: str) -> DetectionResult:
        """
        Detect the language of the given text.

        Returns a DetectionResult with language code, name, confidence,
        and all candidate scores.
        """
        if not text or len(text.strip()) < self.min_text_length:
            return DetectionResult(
                language_code="unknown",
                language_name="Unknown",
                confidence=0.0,
                script=Script.UNKNOWN,
            )

        # Stage 1: Script detection
        script, script_confidence = self.script_classifier.classify(text)

        # Stage 2: Get candidate languages based on script
        candidates = self._get_candidates(script)

        if not candidates:
            return DetectionResult(
                language_code="unknown",
                language_name="Unknown",
                confidence=0.0,
                script=script,
            )

        # Stage 3: N-gram matching
        if self.use_ngram_matching and len(text) >= 10:
            scores = self._ngram_score(text, candidates)
        else:
            # Fall back to equal distribution among candidates
            scores = {lang: 1.0 / len(candidates) for lang in candidates}

        if not scores:
            best_lang = candidates[0] if candidates else "unknown"
            return DetectionResult(
                language_code=best_lang,
                language_name=LANGUAGE_NAMES.get(best_lang, best_lang),
                confidence=script_confidence * 0.5,
                script=script,
                all_scores={},
            )

        # Combine script confidence with n-gram score
        best_lang = max(scores, key=scores.get)
        combined_confidence = script_confidence * scores[best_lang]

        return DetectionResult(
            language_code=best_lang,
            language_name=LANGUAGE_NAMES.get(best_lang, best_lang),
            confidence=min(1.0, combined_confidence),
            script=script,
            all_scores=scores,
        )

    def detect_multiple(self, text: str, top_k: int = 5) -> List[DetectionResult]:
        """
        Detect the top-k most likely languages.

        Returns a list of DetectionResult sorted by confidence.
        """
        result = self.detect(text)
        scores = result.all_scores

        results: List[DetectionResult] = []
        sorted_langs = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        for lang, score in sorted_langs[:top_k]:
            confidence = min(1.0, result.confidence * score / (scores.get(result.language_code, 1.0) + 1e-10))
            results.append(DetectionResult(
                language_code=lang,
                language_name=LANGUAGE_NAMES.get(lang, lang),
                confidence=confidence,
                script=LANGUAGE_SCRIPT_MAP.get(lang, Script.UNKNOWN),
            ))

        return results

    def _get_candidates(self, script: Script) -> List[str]:
        """Get candidate languages for a given script."""
        candidates: List[str] = []
        for lang, lang_script in LANGUAGE_SCRIPT_MAP.items():
            if lang_script == script:
                candidates.append(lang)
        return candidates

    def _ngram_score(self, text: str, candidates: List[str]) -> Dict[str, float]:
        """
        Score each candidate language using n-gram frequency matching.

        Builds a profile from the input text and compares it against
        stored language profiles.
        """
        # Build profile from input text
        text_profile = self.ngram_profile.build_profile(text, "__input__")

        scores: Dict[str, float] = {}
        for lang in candidates:
            ref_profile = self.ngram_profile.get_profile(lang)
            if ref_profile:
                similarity = self.ngram_profile.compare(text_profile, ref_profile)
                scores[lang] = similarity
            else:
                scores[lang] = 0.0

        # Normalize scores
        total = sum(scores.values())
        if total > 0:
            scores = {lang: score / total for lang, score in scores.items()}

        return scores

    def add_language_profile(self, language_code: str, language_name: str,
                              sample_text: str, script: Script) -> None:
        """Add a custom language profile from sample text."""
        LANGUAGE_NAMES[language_code] = language_name
        LANGUAGE_SCRIPT_MAP[language_code] = script
        self.ngram_profile.build_profile(sample_text, language_code)

    def batch_detect(self, texts: List[str]) -> List[DetectionResult]:
        """Detect language for multiple texts."""
        return [self.detect(text) for text in texts]

    def get_supported_languages(self) -> List[str]:
        """Get list of supported language codes."""
        return sorted(LANGUAGE_SCRIPT_MAP.keys())

    def get_supported_scripts(self) -> List[str]:
        """Get list of supported scripts."""
        return [s.value for s in Script if s != Script.UNKNOWN]
