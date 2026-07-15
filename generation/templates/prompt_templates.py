"""
Prompt Templates Library for AGI Unified Framework

Provides comprehensive template management, optimization, and advanced prompting techniques
including Chain-of-Thought and Few-Shot learning templates.
"""

from enum import Enum, auto
from typing import Dict, List, Tuple, Optional, Any, Callable
import re
import random
import math
from dataclasses import dataclass, field


class TemplateCategory(Enum):
    """Categories for prompt templates."""
    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"
    MUSIC = "music"
    TEXT = "text"
    CODE = "code"
    DATA = "data"


@dataclass
class TemplateDefinition:
    """Definition of a prompt template."""
    name: str
    template: str
    category: TemplateCategory
    description: str
    variables: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)


class PromptTemplateLibrary:
    """
    Central library for managing prompt templates across different categories.
    
    Supports registration, retrieval, searching, and application of templates
    with variable substitution.
    """
    
    def __init__(self):
        self._templates: Dict[str, TemplateDefinition] = {}
        self._categories: Dict[TemplateCategory, List[str]] = {
            category: [] for category in TemplateCategory
        }
        self._init_builtin_templates()
    
    def register(
        self,
        name: str,
        template: str,
        category: TemplateCategory,
        description: str = "",
        variables: Optional[List[str]] = None,
        tags: Optional[List[str]] = None
    ) -> None:
        """
        Register a new template in the library.
        
        Args:
            name: Unique template identifier
            template: The template string with {variable} placeholders
            category: Template category
            description: Human-readable description
            variables: List of required variable names
            tags: Optional tags for categorization
        """
        if name in self._templates:
            raise ValueError(f"Template '{name}' already exists")
        
        # Auto-extract variables if not provided
        if variables is None:
            variables = self._extract_variables(template)
        
        template_def = TemplateDefinition(
            name=name,
            template=template,
            category=category,
            description=description,
            variables=variables,
            tags=tags or []
        )
        
        self._templates[name] = template_def
        self._categories[category].append(name)
    
    def _extract_variables(self, template: str) -> List[str]:
        """Extract variable names from template string."""
        pattern = r'\{([a-zA-Z_][a-zA-Z0-9_]*)\}'
        return list(set(re.findall(pattern, template)))
    
    def get(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Get a template by name.
        
        Args:
            name: Template name
            
        Returns:
            Dictionary with template details or None if not found
        """
        if name not in self._templates:
            return None
        
        template_def = self._templates[name]
        return {
            "name": template_def.name,
            "template": template_def.template,
            "category": template_def.category.value,
            "description": template_def.description,
            "variables": template_def.variables,
            "tags": template_def.tags
        }
    
    def list_by_category(self, category: TemplateCategory) -> List[str]:
        """
        List all templates in a category.
        
        Args:
            category: Template category to list
            
        Returns:
            List of template names
        """
        return self._categories.get(category, []).copy()
    
    def search(self, query: str) -> List[str]:
        """
        Search templates by query string.
        
        Args:
            query: Search query (matches name, description, tags)
            
        Returns:
            List of matching template names
        """
        query_lower = query.lower()
        results = []
        
        for name, template_def in self._templates.items():
            # Search in name
            if query_lower in name.lower():
                results.append(name)
                continue
            
            # Search in description
            if query_lower in template_def.description.lower():
                results.append(name)
                continue
            
            # Search in tags
            if any(query_lower in tag.lower() for tag in template_def.tags):
                results.append(name)
                continue
            
            # Search in category
            if query_lower in template_def.category.value.lower():
                results.append(name)
                continue
        
        return list(set(results))
    
    def apply(self, name: str, variables: Dict[str, Any]) -> str:
        """
        Apply variables to a template.
        
        Args:
            name: Template name
            variables: Dictionary of variable values
            
        Returns:
            Formatted prompt string
            
        Raises:
            ValueError: If template not found or variables invalid
        """
        if name not in self._templates:
            raise ValueError(f"Template '{name}' not found")
        
        template_def = self._templates[name]
        
        if not self._validate_variables(template_def, variables):
            missing = set(template_def.variables) - set(variables.keys())
            raise ValueError(f"Missing variables: {missing}")
        
        return template_def.template.format(**variables)
    
    def _validate_variables(
        self,
        template_def: TemplateDefinition,
        variables: Dict[str, Any]
    ) -> bool:
        """
        Validate that all required variables are provided.
        
        Args:
            template_def: Template definition
            variables: Provided variables
            
        Returns:
            True if all required variables present
        """
        return all(var in variables for var in template_def.variables)
    
    def get_builtin_templates(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all built-in templates.
        
        Returns:
            Dictionary of template name to template details
        """
        return {
            name: self.get(name) 
            for name in self._templates.keys()
        }
    
    def _init_builtin_templates(self) -> None:
        """Initialize built-in templates."""
        # Image templates
        self.register(
            "realistic_photo",
            "A photorealistic image of {subject}. {details}. Shot with {camera}, {lighting} lighting, {resolution} resolution. {style_notes}",
            TemplateCategory.IMAGE,
            "Template for realistic photography",
            ["subject", "details", "camera", "lighting", "resolution", "style_notes"],
            ["photo", "realistic", "professional"]
        )
        
        self.register(
            "anime_style",
            "Anime-style illustration of {subject}. {art_style} art style, {mood} atmosphere. {character_details}. Background: {background}. {quality_tags}",
            TemplateCategory.IMAGE,
            "Template for anime-style artwork",
            ["subject", "art_style", "mood", "character_details", "background", "quality_tags"],
            ["anime", "illustration", "2D"]
        )
        
        self.register(
            "oil_painting",
            "Oil painting of {subject} in the style of {artist}. {composition} composition, {color_palette} color palette. {brushwork} brushwork technique. {era} era style.",
            TemplateCategory.IMAGE,
            "Template for oil painting style",
            ["subject", "artist", "composition", "color_palette", "brushwork", "era"],
            ["painting", "classical", "fine art"]
        )
        
        self.register(
            "watercolor",
            "Watercolor painting of {subject}. {technique} technique, {paper_texture} paper texture. {color_scheme} colors with {transparency} transparency. {mood} feeling.",
            TemplateCategory.IMAGE,
            "Template for watercolor style",
            ["subject", "technique", "paper_texture", "color_scheme", "transparency", "mood"],
            ["watercolor", "soft", "artistic"]
        )
        
        self.register(
            "sketch",
            "Pencil sketch of {subject}. {line_quality} lines, {shading} shading. Drawn with {tool} on {surface}. {detail_level} detail, {perspective} perspective.",
            TemplateCategory.IMAGE,
            "Template for sketch style",
            ["subject", "line_quality", "shading", "tool", "surface", "detail_level", "perspective"],
            ["sketch", "drawing", "monochrome"]
        )
        
        self.register(
            "3d_render",
            "3D render of {subject}. {render_engine} engine, {materials} materials. {lighting_setup} lighting, {camera_angle} camera angle. {post_processing} post-processing.",
            TemplateCategory.IMAGE,
            "Template for 3D rendering",
            ["subject", "render_engine", "materials", "lighting_setup", "camera_angle", "post_processing"],
            ["3D", "render", "CGI"]
        )
        
        self.register(
            "pixel_art",
            "Pixel art of {subject}. {resolution} resolution, {color_count} colors. {style} pixel style, {animation} animation. {dithering} dithering.",
            TemplateCategory.IMAGE,
            "Template for pixel art",
            ["subject", "resolution", "color_count", "style", "animation", "dithering"],
            ["pixel", "retro", "8-bit"]
        )
        
        self.register(
            "cyberpunk",
            "Cyberpunk scene of {subject}. Neon {neon_colors} lights, {weather} weather. {architecture} architecture, {technology} technology. {atmosphere} atmosphere.",
            TemplateCategory.IMAGE,
            "Template for cyberpunk aesthetic",
            ["subject", "neon_colors", "weather", "architecture", "technology", "atmosphere"],
            ["cyberpunk", "sci-fi", "neon"]
        )
        
        self.register(
            "fantasy_landscape",
            "Fantasy landscape of {location}. {terrain} terrain, {vegetation} vegetation. {magical_elements} magical elements, {sky} sky. {time_of_day} time.",
            TemplateCategory.IMAGE,
            "Template for fantasy landscapes",
            ["location", "terrain", "vegetation", "magical_elements", "sky", "time_of_day"],
            ["fantasy", "landscape", "magical"]
        )
        
        self.register(
            "portrait",
            "Portrait of {subject}. {pose} pose, {expression} expression. {clothing} clothing, {background} background. {lighting} lighting, {mood} mood.",
            TemplateCategory.IMAGE,
            "Template for portrait photography/art",
            ["subject", "pose", "expression", "clothing", "background", "lighting", "mood"],
            ["portrait", "person", "character"]
        )
        
        # Video templates
        self.register(
            "cinematic_video",
            "Cinematic video of {scene}. {camera_movement} camera movement, {duration} duration. Shot in {aspect_ratio}, {color_grading} color grading. {mood} mood, {pacing} pacing.",
            TemplateCategory.VIDEO,
            "Template for cinematic video",
            ["scene", "camera_movement", "duration", "aspect_ratio", "color_grading", "mood", "pacing"],
            ["cinematic", "film", "professional"]
        )
        
        self.register(
            "animation",
            "Animated video of {subject}. {animation_style} style, {frame_rate} fps. {character_design} character design, {background_art} backgrounds. {story_beat} story moment.",
            TemplateCategory.VIDEO,
            "Template for animation",
            ["subject", "animation_style", "frame_rate", "character_design", "background_art", "story_beat"],
            ["animation", "cartoon", "motion"]
        )
        
        self.register(
            "slow_motion",
            "Slow motion video of {action}. {speed} speed reduction, {duration} duration. Captured with {equipment}, {lighting} lighting. {detail} detail emphasis.",
            TemplateCategory.VIDEO,
            "Template for slow motion video",
            ["action", "speed", "duration", "equipment", "lighting", "detail"],
            ["slow-mo", "high-speed", "detail"]
        )
        
        self.register(
            "time_lapse",
            "Time-lapse video of {subject}. {interval} interval, {duration} final duration. Capturing {transformation} over {time_period}. {location} location.",
            TemplateCategory.VIDEO,
            "Template for time-lapse video",
            ["subject", "interval", "duration", "transformation", "time_period", "location"],
            ["timelapse", "speed-up", "nature"]
        )
        
        self.register(
            "documentary",
            "Documentary footage of {subject}. {interview_style} interviews, {b_roll} B-roll. {narration_tone} narration tone, {visual_style} visual style. {topic} topic focus.",
            TemplateCategory.VIDEO,
            "Template for documentary",
            ["subject", "interview_style", "b_roll", "narration_tone", "visual_style", "topic"],
            ["documentary", "educational", "informative"]
        )
        
        # Audio templates
        self.register(
            "narration",
            "Narration recording of {content}. {voice_type} voice, {pace} pace. {emotion} emotional tone, {clarity} clarity. {accent} accent, {age} age range.",
            TemplateCategory.AUDIO,
            "Template for voice narration",
            ["content", "voice_type", "pace", "emotion", "clarity", "accent", "age"],
            ["narration", "voiceover", "speech"]
        )
        
        self.register(
            "podcast",
            "Podcast audio about {topic}. {format} format, {tone} tone. {host_style} host style, {guest_count} guests. {length} length, {production} production quality.",
            TemplateCategory.AUDIO,
            "Template for podcast audio",
            ["topic", "format", "tone", "host_style", "guest_count", "length", "production"],
            ["podcast", "discussion", "talk"]
        )
        
        self.register(
            "audiobook",
            "Audiobook narration of {book_title}. {genre} genre, {narrator_style} narrator style. {character_voices} character voices, {pacing} pacing. {abridged} abridged version.",
            TemplateCategory.AUDIO,
            "Template for audiobook",
            ["book_title", "genre", "narrator_style", "character_voices", "pacing", "abridged"],
            ["audiobook", "narration", "literature"]
        )
        
        self.register(
            "news_broadcast",
            "News broadcast audio about {headline}. {urgency} urgency level, {formality} formality. {anchor_style} anchor style, {background_music} background music. {station_type} station type.",
            TemplateCategory.AUDIO,
            "Template for news broadcast",
            ["headline", "urgency", "formality", "anchor_style", "background_music", "station_type"],
            ["news", "broadcast", "journalism"]
        )
        
        self.register(
            "conversational",
            "Conversational audio about {topic}. {participants} participants, {setting} setting. {casualness} casualness level, {energy} energy level. {language_style} language style.",
            TemplateCategory.AUDIO,
            "Template for conversational audio",
            ["topic", "participants", "setting", "casualness", "energy", "language_style"],
            ["conversation", "dialogue", "casual"]
        )
        
        # Music templates
        self.register(
            "orchestral",
            "Orchestral music piece. {tempo} tempo, {key} key signature. {mood} mood, {instrumentation} instrumentation. {era} musical era, {complexity} complexity.",
            TemplateCategory.MUSIC,
            "Template for orchestral music",
            ["tempo", "key", "mood", "instrumentation", "era", "complexity"],
            ["orchestral", "classical", "symphony"]
        )
        
        self.register(
            "electronic",
            "Electronic music track. {genre} subgenre, {bpm} BPM. {synth_types} synthesizers, {rhythm} rhythm pattern. {energy} energy level, {sound_design} sound design.",
            TemplateCategory.MUSIC,
            "Template for electronic music",
            ["genre", "bpm", "synth_types", "rhythm", "energy", "sound_design"],
            ["electronic", "synth", "EDM"]
        )
        
        self.register(
            "jazz",
            "Jazz composition. {style} jazz style, {tempo} tempo. {instruments} instruments, {improvisation} improvisation level. {mood} mood, {era} era influence.",
            TemplateCategory.MUSIC,
            "Template for jazz music",
            ["style", "tempo", "instruments", "improvisation", "mood", "era"],
            ["jazz", "swing", "improvisation"]
        )
        
        self.register(
            "rock",
            "Rock song. {subgenre} subgenre, {tempo} tempo. {guitar_style} guitar style, {drum_pattern} drum pattern. {vocals} vocal style, {energy} energy level.",
            TemplateCategory.MUSIC,
            "Template for rock music",
            ["subgenre", "tempo", "guitar_style", "drum_pattern", "vocals", "energy"],
            ["rock", "guitar", "band"]
        )
        
        self.register(
            "ambient",
            "Ambient soundscape. {texture} texture, {mood} mood. {sound_sources} sound sources, {spatial_quality} spatial quality. {duration} duration, {evolution} evolution.",
            TemplateCategory.MUSIC,
            "Template for ambient music",
            ["texture", "mood", "sound_sources", "spatial_quality", "duration", "evolution"],
            ["ambient", "atmospheric", "background"]
        )
        
        self.register(
            "cinematic_score",
            "Cinematic film score. {scene_type} scene type, {emotion} emotion. {orchestration} orchestration, {dynamics} dynamics. {theme} thematic material, {resolution} resolution.",
            TemplateCategory.MUSIC,
            "Template for cinematic score",
            ["scene_type", "emotion", "orchestration", "dynamics", "theme", "resolution"],
            ["cinematic", "film", "score"]
        )
        
        self.register(
            "lo_fi",
            "Lo-fi hip hop track. {mood} mood, {bpm} BPM. {sample_type} sample style, {vinyl_effects} vinyl effects. {chord_progression} chords, {beat_style} beat style.",
            TemplateCategory.MUSIC,
            "Template for lo-fi music",
            ["mood", "bpm", "sample_type", "vinyl_effects", "chord_progression", "beat_style"],
            ["lo-fi", "chill", "study"]
        )
        
        # Text templates
        self.register(
            "blog_post",
            "Write a blog post about {topic}. Target audience: {audience}. Tone: {tone}. Length: {length}. Include: {key_points}. SEO keywords: {keywords}.",
            TemplateCategory.TEXT,
            "Template for blog posts",
            ["topic", "audience", "tone", "length", "key_points", "keywords"],
            ["blog", "article", "content"]
        )
        
        self.register(
            "technical_doc",
            "Write technical documentation for {subject}. Documentation type: {doc_type}. Technical level: {level}. Include: {sections}. Code examples: {code_language}.",
            TemplateCategory.TEXT,
            "Template for technical documentation",
            ["subject", "doc_type", "level", "sections", "code_language"],
            ["technical", "documentation", "reference"]
        )
        
        self.register(
            "creative_writing",
            "Write a {genre} piece about {theme}. Style: {style}. Length: {length}. Characters: {characters}. Setting: {setting}. Plot elements: {plot}.",
            TemplateCategory.TEXT,
            "Template for creative writing",
            ["genre", "theme", "style", "length", "characters", "setting", "plot"],
            ["creative", "fiction", "story"]
        )
        
        self.register(
            "email",
            "Write an email about {purpose}. Recipient: {recipient}. Tone: {tone}. Urgency: {urgency}. Key message: {message}. Call to action: {cta}.",
            TemplateCategory.TEXT,
            "Template for emails",
            ["purpose", "recipient", "tone", "urgency", "message", "cta"],
            ["email", "communication", "business"]
        )
        
        self.register(
            "social_media",
            "Create {platform} content about {topic}. Post type: {post_type}. Tone: {tone}. Target: {target_audience}. Hashtags: {hashtags}. CTA: {call_to_action}.",
            TemplateCategory.TEXT,
            "Template for social media",
            ["platform", "topic", "post_type", "tone", "target_audience", "hashtags", "call_to_action"],
            ["social", "media", "marketing"]
        )
        
        # Code templates
        self.register(
            "python_function",
            "Write a Python function that {purpose}. Function name: {name}. Parameters: {parameters}. Return: {return_type}. Include: {features}. Docstring: {documentation}.",
            TemplateCategory.CODE,
            "Template for Python functions",
            ["purpose", "name", "parameters", "return_type", "features", "documentation"],
            ["python", "function", "code"]
        )
        
        self.register(
            "javascript_class",
            "Write a JavaScript class for {purpose}. Class name: {class_name}. Properties: {properties}. Methods: {methods}. Inheritance: {inheritance}. ES version: {es_version}.",
            TemplateCategory.CODE,
            "Template for JavaScript classes",
            ["purpose", "class_name", "properties", "methods", "inheritance", "es_version"],
            ["javascript", "class", "OOP"]
        )
        
        self.register(
            "sql_query",
            "Write a SQL query to {purpose}. Tables: {tables}. Columns: {columns}. Conditions: {conditions}. Joins: {joins}. Ordering: {ordering}. Aggregation: {aggregation}.",
            TemplateCategory.CODE,
            "Template for SQL queries",
            ["purpose", "tables", "columns", "conditions", "joins", "ordering", "aggregation"],
            ["sql", "database", "query"]
        )
        
        self.register(
            "api_endpoint",
            "Design an API endpoint for {purpose}. HTTP method: {method}. Path: {path}. Request: {request_format}. Response: {response_format}. Authentication: {auth}. Rate limit: {rate_limit}.",
            TemplateCategory.CODE,
            "Template for API endpoints",
            ["purpose", "method", "path", "request_format", "response_format", "auth", "rate_limit"],
            ["api", "endpoint", "rest"]
        )
        
        self.register(
            "unit_test",
            "Write unit tests for {function_name}. Testing framework: {framework}. Test cases: {test_cases}. Edge cases: {edge_cases}. Mocking: {mocking}. Coverage: {coverage_goal}.",
            TemplateCategory.CODE,
            "Template for unit tests",
            ["function_name", "framework", "test_cases", "edge_cases", "mocking", "coverage_goal"],
            ["test", "unit", "quality"]
        )


class TemplateOptimizer:
    """
    Optimizes prompts for better generation results.
    
    Provides techniques for keyword expansion, reordering, quality enhancement,
    redundancy removal, and negative prompt generation.
    """
    
    # Quality boosters by model type
    QUALITY_BOOSTERS = {
        "sdxl": ["masterpiece", "best quality", "highly detailed", "8k uhd"],
        "sd15": ["masterpiece", "best quality", "detailed"],
        "dalle": ["high quality", "detailed", "professional"],
        "midjourney": ["highly detailed", "professional", "8k"],
        "default": ["high quality", "detailed"]
    }
    
    # Keyword expansions
    KEYWORD_EXPANSIONS = {
        "beautiful": ["stunning", "gorgeous", "elegant", "exquisite"],
        "detailed": ["intricate details", "highly detailed", "fine details"],
        "realistic": ["photorealistic", "lifelike", "true to life"],
        "dark": ["shadowy", "dim", "moody", "atmospheric"],
        "bright": ["vibrant", "luminous", "radiant", "brilliant"],
        "old": ["ancient", "aged", "vintage", "antique"],
        "fast": ["swift", "rapid", "speedy", "quick"],
        "big": ["large", "massive", "enormous", "gigantic"],
        "small": ["tiny", "miniature", "petite", "compact"]
    }
    
    # Importance keywords (should come first)
    HIGH_IMPORTANCE = [
        "subject", "main", "character", "person", "object",
        "style", "medium", "art style", "photography"
    ]
    
    def __init__(self):
        self._expansion_cache: Dict[str, List[str]] = {}
    
    def optimize(self, prompt: str, target_model: str = "default") -> str:
        """
        Optimize a prompt for better generation results.
        
        Args:
            prompt: Original prompt
            target_model: Target model for optimization
            
        Returns:
            Optimized prompt
        """
        optimized = prompt
        
        # Apply optimization steps
        optimized = self._expand_keywords(optimized)
        optimized = self._reorder_for_importance(optimized)
        optimized = self._add_quality_boosters(optimized, target_model)
        optimized = self._remove_redundancy(optimized)
        
        return optimized.strip()
    
    def _expand_keywords(self, prompt: str) -> str:
        """
        Expand keywords with synonyms for richer prompts.
        
        Args:
            prompt: Input prompt
            
        Returns:
            Prompt with expanded keywords
        """
        words = prompt.lower().split()
        expanded = []
        
        for word in words:
            clean_word = re.sub(r'[^\w]', '', word)
            expanded.append(word)
            
            # Add expansions occasionally
            if clean_word in self.KEYWORD_EXPANSIONS and random.random() < 0.3:
                expansion = random.choice(self.KEYWORD_EXPANSIONS[clean_word])
                if expansion not in prompt.lower():
                    expanded.append(expansion)
        
        return " ".join(expanded)
    
    def _reorder_for_importance(self, prompt: str) -> str:
        """
        Reorder prompt components by importance.
        
        Args:
            prompt: Input prompt
            
        Returns:
            Reordered prompt
        """
        # Split into components
        components = [c.strip() for c in prompt.split(",")]
        
        # Score each component
        scored = []
        for comp in components:
            score = 0
            comp_lower = comp.lower()
            
            # Check for high importance keywords
            for keyword in self.HIGH_IMPORTANCE:
                if keyword in comp_lower:
                    score += 10
            
            # Prefer shorter components for subject
            score -= len(comp) * 0.1
            
            scored.append((score, comp))
        
        # Sort by score (descending)
        scored.sort(key=lambda x: x[0], reverse=True)
        
        return ", ".join([comp for _, comp in scored])
    
    def _add_quality_boosters(self, prompt: str, model: str) -> str:
        """
        Add quality enhancement keywords.
        
        Args:
            prompt: Input prompt
            model: Target model
            
        Returns:
            Prompt with quality boosters
        """
        boosters = self.QUALITY_BOOSTERS.get(model, self.QUALITY_BOOSTERS["default"])
        
        # Check if boosters already present
        prompt_lower = prompt.lower()
        missing_boosters = [b for b in boosters if b.lower() not in prompt_lower]
        
        if missing_boosters:
            return ", ".join(missing_boosters[:2]) + ", " + prompt
        
        return prompt
    
    def _remove_redundancy(self, prompt: str) -> str:
        """
        Remove redundant words and phrases.
        
        Args:
            prompt: Input prompt
            
        Returns:
            Deduplicated prompt
        """
        words = prompt.split()
        seen = set()
        result = []
        
        for word in words:
            clean = re.sub(r'[^\w]', '', word.lower())
            if clean not in seen or clean in ["and", "or", "the", "a", "an"]:
                result.append(word)
                if clean:
                    seen.add(clean)
        
        return " ".join(result)
    
    def estimate_complexity(self, prompt: str) -> float:
        """
        Estimate the complexity of a prompt.
        
        Args:
            prompt: Input prompt
            
        Returns:
            Complexity score (0.0 - 1.0)
        """
        complexity = 0.0
        
        # Length factor
        word_count = len(prompt.split())
        complexity += min(word_count / 50, 0.3)
        
        # Unique words factor
        unique_words = len(set(prompt.lower().split()))
        complexity += min(unique_words / 30, 0.2)
        
        # Special characters/punctuation
        special_chars = len(re.findall(r'[^\w\s]', prompt))
        complexity += min(special_chars / 10, 0.1)
        
        # Technical terms
        technical_terms = [
            "render", "lighting", "composition", "perspective",
            "algorithm", "function", "implementation", "architecture"
        ]
        tech_count = sum(1 for term in technical_terms if term in prompt.lower())
        complexity += min(tech_count / 5, 0.2)
        
        # Variables/templates
        var_count = len(re.findall(r'\{[^}]+\}', prompt))
        complexity += min(var_count / 5, 0.2)
        
        return min(complexity, 1.0)
    
    def suggest_negative_prompt(self, prompt: str) -> str:
        """
        Suggest a negative prompt based on the positive prompt.
        
        Args:
            prompt: Positive prompt
            
        Returns:
            Suggested negative prompt
        """
        # Common negative prompt elements
        common_negatives = [
            "low quality", "blurry", "distorted", "deformed",
            "ugly", "duplicate", "watermark", "signature",
            "text", "error", "cropped", "worst quality"
        ]
        
        # Context-specific negatives
        context_negatives = []
        prompt_lower = prompt.lower()
        
        if "photo" in prompt_lower or "realistic" in prompt_lower:
            context_negatives.extend([
                "painting", "drawing", "illustration", "cartoon",
                "anime", "3d render", "artificial"
            ])
        
        if "portrait" in prompt_lower or "person" in prompt_lower:
            context_negatives.extend([
                "extra limbs", "bad anatomy", "disfigured",
                "mutated", "poorly drawn face"
            ])
        
        if "landscape" in prompt_lower or "scenery" in prompt_lower:
            context_negatives.extend([
                "people", "person", "human", "building",
                "urban", "city"
            ])
        
        all_negatives = common_negatives + context_negatives
        return ", ".join(all_negatives)


class ChainOfThoughtTemplate:
    """
    Template for Chain-of-Thought prompting.
    
    Breaks down complex problems into sequential reasoning steps.
    """
    
    def __init__(self):
        self._steps: List[Dict[str, str]] = []
    
    def add_step(self, description: str, expected_output: str = "") -> None:
        """
        Add a reasoning step.
        
        Args:
            description: Step description
            expected_output: Expected output from this step
        """
        self._steps.append({
            "description": description,
            "expected_output": expected_output
        })
    
    def generate(self, prompt: str) -> str:
        """
        Generate a Chain-of-Thought prompt.
        
        Args:
            prompt: Base problem/prompt
            
        Returns:
            Formatted CoT prompt
        """
        lines = [
            "Let's solve this step by step:",
            "",
            f"Problem: {prompt}",
            "",
            "Reasoning process:"
        ]
        
        for i, step in enumerate(self._steps, 1):
            lines.append(self._format_step(step, i))
        
        lines.extend([
            "",
            "Final Answer:"
        ])
        
        return "\n".join(lines)
    
    def _format_step(self, step: Dict[str, str], index: int) -> str:
        """
        Format a single step.
        
        Args:
            step: Step definition
            index: Step number
            
        Returns:
            Formatted step string
        """
        lines = [f"{index}. {step['description']}"]
        
        if step["expected_output"]:
            lines.append(f"   Expected: {step['expected_output']}")
        
        return "\n".join(lines)
    
    def get_reasoning_trace(self, output: str) -> List[str]:
        """
        Extract reasoning trace from model output.
        
        Args:
            output: Model output text
            
        Returns:
            List of reasoning steps extracted
        """
        trace = []
        
        # Look for numbered steps
        step_pattern = r'(?:^|\n)\s*(\d+)[:.\)]\s*(.+?)(?=\n\s*\d+[:.\)]|$)'
        matches = re.findall(step_pattern, output, re.DOTALL)
        
        for _, content in matches:
            trace.append(content.strip())
        
        # If no numbered steps, look for bullet points
        if not trace:
            bullet_pattern = r'(?:^|\n)\s*[-•]\s*(.+?)(?=\n\s*[-•]|$)'
            matches = re.findall(bullet_pattern, output, re.DOTALL)
            trace = [m.strip() for m in matches]
        
        return trace
    
    def clear(self) -> None:
        """Clear all steps."""
        self._steps.clear()


class FewShotTemplate:
    """
    Template for Few-Shot learning.
    
    Provides examples to guide model behavior on similar tasks.
    """
    
    def __init__(self):
        self._examples: List[Tuple[str, str]] = []
        self._instruction: str = ""
    
    def set_instruction(self, instruction: str) -> None:
        """
        Set the task instruction.
        
        Args:
            instruction: Task description
        """
        self._instruction = instruction
    
    def add_example(self, input_text: str, output_text: str) -> None:
        """
        Add an example to the template.
        
        Args:
            input_text: Example input
            output_text: Example output
        """
        self._examples.append((input_text, output_text))
    
    def generate(self, query: str, k: int = 3) -> str:
        """
        Generate a few-shot prompt.
        
        Args:
            query: Input query
            k: Number of examples to include
            
        Returns:
            Formatted few-shot prompt
        """
        lines = []
        
        if self._instruction:
            lines.append(self._instruction)
            lines.append("")
        
        # Select relevant examples
        selected = self._select_relevant_examples(query, k)
        
        # Add examples
        for i, (inp, out) in enumerate(selected, 1):
            lines.append(f"Example {i}:")
            lines.append(f"Input: {inp}")
            lines.append(f"Output: {out}")
            lines.append("")
        
        # Add query
        lines.append("Now, please process the following:")
        lines.append(f"Input: {query}")
        lines.append("Output:")
        
        return "\n".join(lines)
    
    def _select_relevant_examples(
        self,
        query: str,
        k: int
    ) -> List[Tuple[str, str]]:
        """
        Select most relevant examples for the query.
        
        Args:
            query: Input query
            k: Number of examples to select
            
        Returns:
            List of selected examples
        """
        if len(self._examples) <= k:
            return self._examples
        
        # Score examples by similarity
        scored = []
        for example in self._examples:
            similarity = self._similarity(query, example[0])
            scored.append((similarity, example))
        
        # Sort by similarity (descending)
        scored.sort(key=lambda x: x[0], reverse=True)
        
        return [ex for _, ex in scored[:k]]
    
    def _similarity(self, a: str, b: str) -> float:
        """
        Calculate simple similarity between two strings.
        
        Args:
            a: First string
            b: Second string
            
        Returns:
            Similarity score (0.0 - 1.0)
        """
        # Tokenize
        tokens_a = set(a.lower().split())
        tokens_b = set(b.lower().split())
        
        if not tokens_a or not tokens_b:
            return 0.0
        
        # Jaccard similarity
        intersection = len(tokens_a & tokens_b)
        union = len(tokens_a | tokens_b)
        
        return intersection / union if union > 0 else 0.0
    
    def clear_examples(self) -> None:
        """Clear all examples."""
        self._examples.clear()
    
    def get_example_count(self) -> int:
        """
        Get number of stored examples.
        
        Returns:
            Example count
        """
        return len(self._examples)


# Utility functions
def create_default_library() -> PromptTemplateLibrary:
    """Create a template library with all built-in templates."""
    return PromptTemplateLibrary()


def quick_optimize(prompt: str, model: str = "default") -> str:
    """Quickly optimize a prompt."""
    optimizer = TemplateOptimizer()
    return optimizer.optimize(prompt, model)


def estimate_tokens(text: str, avg_token_length: int = 4) -> int:
    """
    Roughly estimate token count.
    
    Args:
        text: Input text
        avg_token_length: Average characters per token
        
    Returns:
        Estimated token count
    """
    return len(text) // avg_token_length
