"""
Music Generation Module
A pure Python music generation framework using only standard library.
"""

import math
import random
import wave
import struct
from typing import List, Tuple, Dict, Optional, Any
from dataclasses import dataclass
from enum import Enum


@dataclass
class GenerationResult:
    """Result container for music generation."""
    audio_data: List[float]
    sample_rate: int
    duration: float
    bpm: int
    metadata: Dict[str, Any]


class MusicTheory:
    """Music theory tools for note, scale, and chord operations."""
    
    # Note frequencies for octave 4 (A4 = 440Hz)
    _base_notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    
    _scales: Dict[str, List[int]] = {
        'major': [0, 2, 4, 5, 7, 9, 11],
        'minor': [0, 2, 3, 5, 7, 8, 10],
        'harmonic_minor': [0, 2, 3, 5, 7, 8, 11],
        'melodic_minor': [0, 2, 3, 5, 7, 9, 11],
        'dorian': [0, 2, 3, 5, 7, 9, 10],
        'phrygian': [0, 1, 3, 5, 7, 8, 10],
        'lydian': [0, 2, 4, 6, 7, 9, 11],
        'mixolydian': [0, 2, 4, 5, 7, 9, 10],
        'locrian': [0, 1, 3, 5, 6, 8, 10],
        'pentatonic_major': [0, 2, 4, 7, 9],
        'pentatonic_minor': [0, 3, 5, 7, 10],
        'blues': [0, 3, 5, 6, 7, 10],
        'chromatic': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
    }
    
    _chords: Dict[str, List[int]] = {
        'major': [0, 4, 7],
        'minor': [0, 3, 7],
        'diminished': [0, 3, 6],
        'augmented': [0, 4, 8],
        'major7': [0, 4, 7, 11],
        'minor7': [0, 3, 7, 10],
        'dominant7': [0, 4, 7, 10],
        'minor7b5': [0, 3, 6, 10],
        'sus2': [0, 2, 7],
        'sus4': [0, 5, 7],
        'add9': [0, 4, 7, 14],
        '6': [0, 4, 7, 9],
        '9': [0, 4, 7, 10, 14],
        'maj9': [0, 4, 7, 11, 14],
    }
    
    _progressions: Dict[str, List[List[str]]] = {
        'pop': [['I', 'major'], ['V', 'major'], ['vi', 'minor'], ['IV', 'major']],
        'jazz': [['ii', 'minor7'], ['V', 'dominant7'], ['I', 'major7']],
        'blues': [['I', 'dominant7'], ['IV', 'dominant7'], ['I', 'dominant7'], ['V', 'dominant7']],
        'classical': [['I', 'major'], ['IV', 'major'], ['V', 'major'], ['I', 'major']],
        'rock': [['I', 'major'], ['IV', 'major'], ['V', 'major']],
        'minor_pop': [['vi', 'minor'], ['IV', 'major'], ['I', 'major'], ['V', 'major']],
        'jazz_turnaround': [['iii', 'minor7'], ['vi', 'minor7'], ['ii', 'minor7'], ['V', 'dominant7']],
    }
    
    _roman_numerals: Dict[str, int] = {
        'I': 0, 'ii': 1, 'II': 1, 'iii': 2, 'III': 2,
        'IV': 3, 'iv': 3, 'V': 4, 'v': 4, 'vi': 5, 'VI': 5, 'vii': 6, 'VII': 6
    }
    
    def __init__(self):
        self._note_cache: Dict[str, int] = {}
        self._freq_cache: Dict[str, float] = {}
    
    def get_scale_notes(self, root: str, scale_type: str = 'major') -> List[str]:
        """Get notes for a scale given root note and scale type."""
        if scale_type not in self._scales:
            scale_type = 'major'
        
        root_idx = self._base_notes.index(root) if root in self._base_notes else 0
        intervals = self._scales[scale_type]
        
        notes = []
        for interval in intervals:
            note_idx = (root_idx + interval) % 12
            notes.append(self._base_notes[note_idx])
        
        return notes
    
    def get_chord_notes(self, root: str, chord_type: str = 'major') -> List[str]:
        """Get notes for a chord given root note and chord type."""
        if chord_type not in self._chords:
            chord_type = 'major'
        
        root_idx = self._base_notes.index(root) if root in self._base_notes else 0
        intervals = self._chords[chord_type]
        
        notes = []
        for interval in intervals:
            note_idx = (root_idx + interval) % 12
            notes.append(self._base_notes[note_idx])
        
        return notes
    
    def _note_to_freq(self, note: str) -> float:
        """Convert note name to frequency (e.g., 'A4' -> 440.0)."""
        if note in self._freq_cache:
            return self._freq_cache[note]
        
        # Parse note and octave
        if len(note) >= 2:
            if note[1] == '#':
                note_name = note[:2]
                octave = int(note[2:]) if len(note) > 2 else 4
            else:
                note_name = note[0]
                octave = int(note[1:]) if len(note) > 1 else 4
        else:
            note_name = note
            octave = 4
        
        # Calculate frequency
        if note_name in self._base_notes:
            note_idx = self._base_notes.index(note_name)
            # A4 is index 9 in octave 4
            semitones_from_a4 = (octave - 4) * 12 + (note_idx - 9)
            freq = 440.0 * (2 ** (semitones_from_a4 / 12.0))
            self._freq_cache[note] = freq
            return freq
        
        return 440.0
    
    def _freq_to_note(self, freq: float) -> str:
        """Convert frequency to nearest note name."""
        if freq <= 0:
            return 'A4'
        
        # Calculate semitones from A4
        semitones = 12 * math.log2(freq / 440.0)
        semitones_rounded = round(semitones)
        
        octave = 4 + (semitones_rounded + 9) // 12
        note_idx = (semitones_rounded + 9) % 12
        
        return f"{self._base_notes[note_idx]}{octave}"
    
    def generate_progression(self, key: str, progression_type: str = 'pop') -> List[List[str]]:
        """Generate a chord progression in the given key."""
        if progression_type not in self._progressions:
            progression_type = 'pop'
        
        progression = self._progressions[progression_type]
        scale_notes = self.get_scale_notes(key, 'major')
        
        chords = []
        for roman, chord_type in progression:
            degree = self._roman_numerals.get(roman, 0)
            root = scale_notes[degree % len(scale_notes)]
            chords.append([root, chord_type])
        
        return chords
    
    def _circle_of_fifths(self) -> List[str]:
        """Generate the circle of fifths."""
        # Start at C, go up by perfect fifth (7 semitones)
        circle = []
        current = 0  # C
        for _ in range(12):
            circle.append(self._base_notes[current])
            current = (current + 7) % 12
        return circle


class MelodyGenerator:
    """Melody generation using music theory and patterns."""
    
    def __init__(self, scale: Optional[List[str]] = None):
        self._scale = scale or ['C', 'D', 'E', 'F', 'G', 'A', 'B']
        self._theory = MusicTheory()
        self._motif_memory: List[List[Tuple[str, float, float]]] = []
    
    def generate_melody(self, chords: List[List[str]], bars: int, density: float = 0.5) -> List[Tuple[str, float, float]]:
        """
        Generate a melody over given chords.
        Returns list of (note, start_time, duration) tuples.
        """
        melody = []
        beats_per_bar = 4
        current_beat = 0.0
        
        rhythm_pattern = self._generate_rhythm(bars, density)
        
        for chord_info in chords:
            chord_root, chord_type = chord_info
            chord_notes = self._theory.get_chord_notes(chord_root, chord_type)
            available_notes = self._select_notes_for_chord(chord_info, self._scale)
            
            # Generate notes for this chord (assuming 1 bar per chord)
            bar_start = current_beat
            bar_rhythm = [r for r in rhythm_pattern if bar_start <= r[0] < bar_start + beats_per_bar]
            
            for start, duration in bar_rhythm:
                if random.random() < density:
                    # Prefer chord tones
                    if random.random() < 0.7:
                        note = random.choice(chord_notes)
                    else:
                        note = random.choice(available_notes)
                    
                    # Add octave
                    octave = random.choice([4, 5])
                    note_with_octave = f"{note}{octave}"
                    
                    melody.append((note_with_octave, start, duration))
            
            current_beat += beats_per_bar
        
        # Add musical embellishments
        melody = self._add_passing_tones(melody)
        melody = self._add_ornaments(melody)
        
        return sorted(melody, key=lambda x: x[1])
    
    def _generate_rhythm(self, bars: int, density: float) -> List[Tuple[float, float]]:
        """Generate rhythmic pattern for given number of bars."""
        rhythms = []
        beats_per_bar = 4
        
        # Common rhythmic subdivisions
        subdivisions = [0.25, 0.5, 1.0]
        if density > 0.7:
            subdivisions = [0.25, 0.5]
        elif density < 0.3:
            subdivisions = [1.0, 2.0]
        
        for bar in range(bars):
            bar_start = bar * beats_per_bar
            current_beat = bar_start
            
            while current_beat < bar_start + beats_per_bar:
                duration = random.choice(subdivisions)
                if current_beat + duration > bar_start + beats_per_bar:
                    duration = bar_start + beats_per_bar - current_beat
                
                if duration > 0:
                    rhythms.append((current_beat, duration))
                
                # Rest probability based on density
                if random.random() < density:
                    current_beat += duration
                else:
                    current_beat += duration * 0.5
        
        return rhythms
    
    def _select_notes_for_chord(self, chord: List[str], scale: List[str]) -> List[str]:
        """Select appropriate notes for a given chord from the scale."""
        chord_root, chord_type = chord
        chord_notes = self._theory.get_chord_notes(chord_root, chord_type)
        
        # Combine chord tones with scale tones
        available = list(set(chord_notes + scale))
        return available
    
    def _add_passing_tones(self, melody: List[Tuple[str, float, float]]) -> List[Tuple[str, float, float]]:
        """Add passing tones between melody notes."""
        if len(melody) < 2:
            return melody
        
        enhanced = []
        for i in range(len(melody) - 1):
            note1, start1, dur1 = melody[i]
            note2, start2, dur2 = melody[i + 1]
            
            enhanced.append((note1, start1, dur1))
            
            # Check if there's a gap for a passing tone
            gap = start2 - (start1 + dur1)
            if gap > 0.25 and random.random() < 0.3:
                # Add passing tone
                passing_octave = note1[-1] if note1[-1].isdigit() else '4'
                passing_note = random.choice(self._scale) + passing_octave
                enhanced.append((passing_note, start1 + dur1, gap))
        
        enhanced.append(melody[-1])
        return enhanced
    
    def _add_ornaments(self, melody: List[Tuple[str, float, float]]) -> List[Tuple[str, float, float]]:
        """Add ornamental notes (grace notes, turns)."""
        if not melody:
            return melody
        
        ornamented = []
        for note, start, duration in melody:
            # Occasionally add a grace note before
            if random.random() < 0.1 and duration > 0.25:
                grace_octave = note[-1] if note[-1].isdigit() else '4'
                grace_note = random.choice(self._scale) + grace_octave
                ornamented.append((grace_note, start - 0.05, 0.05))
            
            ornamented.append((note, start, duration))
        
        return ornamented


class RhythmGenerator:
    """Generate rhythmic patterns for various genres."""
    
    def __init__(self, time_signature: Tuple[int, int] = (4, 4)):
        self._time_signature = time_signature
        self._drum_sounds = ['kick', 'snare', 'hihat', 'tom', 'crash', 'ride']
    
    def generate_drum_pattern(self, genre: str, bars: int) -> List[List[Tuple[str, float]]]:
        """Generate drum pattern for specified genre."""
        genre_patterns = {
            'rock': self._rock_pattern,
            'jazz': self._jazz_pattern,
            'electronic': self._electronic_pattern,
            'latin': self._latin_pattern,
            'pop': self._rock_pattern,
            'funk': self._electronic_pattern,
        }
        
        pattern_func = genre_patterns.get(genre, self._rock_pattern)
        base_pattern = pattern_func()
        
        # Repeat pattern for requested bars
        full_pattern = []
        beats_per_bar = self._time_signature[0]
        
        for bar in range(bars):
            bar_offset = bar * beats_per_bar
            for hit in base_pattern:
                sound, beat = hit
                full_pattern.append([sound, beat + bar_offset])
        
        # Apply humanization
        full_pattern = self._humanize_timing(full_pattern)
        full_pattern = self._add_velocity_variation(full_pattern)
        
        return full_pattern
    
    def _rock_pattern(self) -> List[List]:
        """Generate rock drum pattern."""
        pattern = []
        beats_per_bar = self._time_signature[0]
        
        for beat in range(beats_per_bar):
            # Kick on 1 and 3
            if beat % 2 == 0:
                pattern.append(['kick', float(beat)])
            # Snare on 2 and 4
            else:
                pattern.append(['snare', float(beat)])
            
            # Hi-hat on every 8th note
            pattern.append(['hihat', float(beat)])
            pattern.append(['hihat', float(beat) + 0.5])
        
        return pattern
    
    def _jazz_pattern(self) -> List[List]:
        """Generate jazz drum pattern (ride cymbal pattern)."""
        pattern = []
        beats_per_bar = self._time_signature[0]
        
        for beat in range(beats_per_bar):
            # Ride cymbal on 1, 2&, 3, 4&
            pattern.append(['ride', float(beat)])
            pattern.append(['ride', float(beat) + 0.5])
            
            # Hi-hat on 2 and 4 (with foot)
            if beat % 2 == 1:
                pattern.append(['hihat', float(beat)])
            
            # Kick and snare sparse
            if beat == 0:
                pattern.append(['kick', float(beat)])
            elif beat == 2:
                pattern.append(['snare', float(beat)])
        
        return pattern
    
    def _electronic_pattern(self) -> List[List]:
        """Generate electronic/EDM drum pattern."""
        pattern = []
        beats_per_bar = self._time_signature[0]
        
        for beat in range(beats_per_bar):
            # Four-on-the-floor kick
            pattern.append(['kick', float(beat)])
            
            # Off-beat hi-hats
            pattern.append(['hihat', float(beat) + 0.5])
            
            # Snare on 2 and 4
            if beat % 2 == 1:
                pattern.append(['snare', float(beat)])
        
        return pattern
    
    def _latin_pattern(self) -> List[List]:
        """Generate Latin drum pattern."""
        pattern = []
        beats_per_bar = self._time_signature[0]
        
        for beat in range(beats_per_bar):
            # Bossa nova style pattern
            pattern.append(['kick', float(beat)])
            
            if beat % 2 == 0:
                pattern.append(['hihat', float(beat) + 0.25])
                pattern.append(['hihat', float(beat) + 0.75])
            
            if beat == 1 or beat == 3:
                pattern.append(['snare', float(beat)])
        
        return pattern
    
    def _humanize_timing(self, pattern: List[List], amount: float = 0.1) -> List[List]:
        """Add subtle timing variations for human feel."""
        humanized = []
        for hit in pattern:
            sound, beat = hit
            # Add small random offset
            offset = random.uniform(-amount, amount) * 0.1
            humanized.append([sound, max(0, beat + offset)])
        
        return sorted(humanized, key=lambda x: x[1])
    
    def _add_velocity_variation(self, pattern: List[List]) -> List[List]:
        """Add velocity (volume) variations to drum hits."""
        varied = []
        for hit in pattern:
            sound, beat = hit
            # Random velocity between 0.6 and 1.0
            velocity = random.uniform(0.6, 1.0)
            varied.append([sound, beat, velocity])
        
        return varied


class InstrumentSynthesizer:
    """Synthesize instrument sounds using waveform generation."""
    
    def __init__(self, sample_rate: int = 44100):
        self._sample_rate = sample_rate
        self._two_pi = 2.0 * math.pi
    
    def synthesize_note(self, note: str, duration: float, instrument: str = 'piano') -> List[float]:
        """Synthesize a single note for given instrument."""
        theory = MusicTheory()
        freq = theory._note_to_freq(note)
        
        synthesizers = {
            'piano': self._synth_piano,
            'guitar': self._synth_guitar,
            'bass': self._synth_bass,
            'pad': self._synth_pad,
            'pluck': self._synth_pluck,
        }
        
        synth_func = synthesizers.get(instrument, self._synth_piano)
        return synth_func(freq, duration)
    
    def _synth_piano(self, freq: float, duration: float) -> List[float]:
        """Synthesize piano-like sound using multiple harmonics."""
        num_samples = int(self._sample_rate * duration)
        samples = []
        
        # ADSR envelope
        envelope = self._adsr_envelope(duration, 0.01, 0.1, 0.7, 0.3)
        
        for i in range(num_samples):
            t = i / self._sample_rate
            
            # Fundamental + harmonics
            sample = 0.0
            harmonics = [(1.0, 1.0), (0.5, 0.5), (0.25, 0.25), (0.125, 0.125)]
            
            for harmonic, amplitude in harmonics:
                sample += amplitude * math.sin(self._two_pi * freq * harmonic * t)
                # Add slight detuning for richness
                sample += amplitude * 0.1 * math.sin(self._two_pi * freq * harmonic * 1.003 * t)
            
            # Apply envelope
            env_idx = min(i, len(envelope) - 1)
            sample *= envelope[env_idx]
            
            samples.append(sample * 0.3)
        
        return samples
    
    def _synth_guitar(self, freq: float, duration: float) -> List[float]:
        """Synthesize guitar-like sound."""
        num_samples = int(self._sample_rate * duration)
        samples = []
        
        envelope = self._adsr_envelope(duration, 0.05, 0.2, 0.6, 0.4)
        
        for i in range(num_samples):
            t = i / self._sample_rate
            
            # Guitar has brighter harmonics
            sample = 0.0
            for n in range(1, 8):
                amplitude = 1.0 / n
                sample += amplitude * math.sin(self._two_pi * freq * n * t)
            
            # Add some "pluck" noise at start
            if i < 1000:
                sample += random.uniform(-0.1, 0.1) * (1 - i / 1000)
            
            env_idx = min(i, len(envelope) - 1)
            sample *= envelope[env_idx]
            
            samples.append(sample * 0.25)
        
        return samples
    
    def _synth_bass(self, freq: float, duration: float) -> List[float]:
        """Synthesize bass sound."""
        num_samples = int(self._sample_rate * duration)
        samples = []
        
        envelope = self._adsr_envelope(duration, 0.02, 0.1, 0.8, 0.2)
        
        for i in range(num_samples):
            t = i / self._sample_rate
            
            # Bass: strong fundamental, few harmonics
            sample = math.sin(self._two_pi * freq * t)
            sample += 0.3 * math.sin(self._two_pi * freq * 2 * t)
            sample += 0.1 * math.sin(self._two_pi * freq * 3 * t)
            
            # Sub-octave
            sample += 0.5 * math.sin(self._two_pi * freq * 0.5 * t)
            
            env_idx = min(i, len(envelope) - 1)
            sample *= envelope[env_idx]
            
            samples.append(sample * 0.4)
        
        return samples
    
    def _synth_pad(self, freq: float, duration: float) -> List[float]:
        """Synthesize pad/synth string sound."""
        num_samples = int(self._sample_rate * duration)
        samples = []
        
        envelope = self._adsr_envelope(duration, 0.5, 0.5, 0.8, 1.0)
        
        for i in range(num_samples):
            t = i / self._sample_rate
            
            # Rich harmonic content with slow attack
            sample = 0.0
            for n in range(1, 6):
                amplitude = 1.0 / (n * n)
                # Detuned oscillators for chorus effect
                sample += amplitude * math.sin(self._two_pi * freq * n * t)
                sample += amplitude * 0.5 * math.sin(self._two_pi * freq * n * 1.01 * t)
            
            env_idx = min(i, len(envelope) - 1)
            sample *= envelope[env_idx]
            
            samples.append(sample * 0.2)
        
        return samples
    
    def _synth_pluck(self, freq: float, duration: float) -> List[float]:
        """Synthesize plucked string sound."""
        num_samples = int(self._sample_rate * duration)
        samples = []
        
        # Fast decay for pluck
        envelope = self._adsr_envelope(duration, 0.005, 0.1, 0.3, 0.1)
        
        for i in range(num_samples):
            t = i / self._sample_rate
            
            # Sawtooth-like with filtering
            sample = 0.0
            for n in range(1, 10):
                amplitude = 1.0 / n
                if n > 4:
                    amplitude *= 0.5  # Reduce high harmonics
                sample += amplitude * math.sin(self._two_pi * freq * n * t)
            
            env_idx = min(i, len(envelope) - 1)
            sample *= envelope[env_idx]
            
            samples.append(sample * 0.3)
        
        return samples
    
    def _adsr_envelope(self, duration: float, attack: float, decay: float, 
                       sustain: float, release: float) -> List[float]:
        """Generate ADSR envelope."""
        num_samples = int(self._sample_rate * duration)
        envelope = []
        
        attack_samples = int(self._sample_rate * attack)
        decay_samples = int(self._sample_rate * decay)
        release_samples = int(self._sample_rate * release)
        sustain_samples = num_samples - attack_samples - decay_samples - release_samples
        
        # Attack phase
        for i in range(attack_samples):
            envelope.append(i / attack_samples if attack_samples > 0 else 1.0)
        
        # Decay phase
        for i in range(decay_samples):
            progress = i / decay_samples if decay_samples > 0 else 1.0
            envelope.append(1.0 - progress * (1.0 - sustain))
        
        # Sustain phase
        for i in range(max(0, sustain_samples)):
            envelope.append(sustain)
        
        # Release phase
        for i in range(release_samples):
            progress = i / release_samples if release_samples > 0 else 1.0
            envelope.append(sustain * (1.0 - progress))
        
        # Pad or truncate to exact length
        while len(envelope) < num_samples:
            envelope.append(0.0)
        envelope = envelope[:num_samples]
        
        return envelope
    
    def _apply_filter(self, audio: List[float], filter_type: str, cutoff: float) -> List[float]:
        """Apply simple low-pass or high-pass filter."""
        if not audio:
            return audio
        
        # Simple one-pole filter
        output = []
        prev_output = 0.0
        
        # Normalize cutoff to 0-1 range
        rc = 1.0 / (2 * math.pi * cutoff)
        dt = 1.0 / self._sample_rate
        alpha = dt / (rc + dt)
        
        if filter_type == 'lowpass':
            for sample in audio:
                prev_output = prev_output + alpha * (sample - prev_output)
                output.append(prev_output)
        elif filter_type == 'highpass':
            for sample in audio:
                prev_output = alpha * (prev_output + sample - (output[-1] if output else sample))
                output.append(prev_output)
        else:
            return audio
        
        return output


class MusicGenerator:
    """Main music generation class that orchestrates all components."""
    
    def __init__(self):
        self._theory = MusicTheory()
        self._melody_gen = MelodyGenerator()
        self._rhythm_gen = RhythmGenerator()
        self._synthesizer = InstrumentSynthesizer()
        self._sample_rate = 44100
    
    def generate(self, prompt: str, duration: float = 30.0, bpm: int = 120) -> GenerationResult:
        """
        Generate music based on text prompt.
        
        Args:
            prompt: Text description of desired music
            duration: Length in seconds
            bpm: Beats per minute
        
        Returns:
            GenerationResult containing audio data and metadata
        """
        # Parse prompt
        params = self._parse_music_prompt(prompt)
        
        # Compose structure
        composition = self._compose_structure(params.get('genre', 'pop'), duration)
        
        # Render to audio
        audio = self._render_to_audio(composition)
        
        # Mix and master
        audio = self._mix_tracks([audio])
        audio = self._mastering(audio)
        
        metadata = {
            'genre': params.get('genre', 'pop'),
            'key': params.get('key', 'C'),
            'bpm': bpm,
            'duration': duration,
            'composition': composition
        }
        
        return GenerationResult(
            audio_data=audio,
            sample_rate=self._sample_rate,
            duration=duration,
            bpm=bpm,
            metadata=metadata
        )
    
    def _parse_music_prompt(self, prompt: str) -> dict:
        """Parse music description prompt to extract parameters."""
        prompt_lower = prompt.lower()
        
        params = {
            'genre': 'pop',
            'mood': 'neutral',
            'key': 'C',
            'tempo': 'medium'
        }
        
        # Detect genre
        genres = ['rock', 'jazz', 'electronic', 'classical', 'pop', 'blues', 'latin', 'funk']
        for genre in genres:
            if genre in prompt_lower:
                params['genre'] = genre
                break
        
        # Detect mood
        moods = ['happy', 'sad', 'energetic', 'calm', 'melancholic', 'upbeat']
        for mood in moods:
            if mood in prompt_lower:
                params['mood'] = mood
                break
        
        # Detect key
        keys = ['C', 'D', 'E', 'F', 'G', 'A', 'B']
        for key in keys:
            if f'key of {key}' in prompt_lower or f'in {key}' in prompt_lower:
                params['key'] = key
                break
        
        # Detect tempo
        if any(word in prompt_lower for word in ['fast', 'upbeat', 'energetic', 'quick']):
            params['tempo'] = 'fast'
        elif any(word in prompt_lower for word in ['slow', 'calm', 'relaxing', 'ballad']):
            params['tempo'] = 'slow'
        
        return params
    
    def _compose_structure(self, genre: str, duration: float) -> List[dict]:
        """Create composition structure with sections."""
        bpm = 120
        beats_per_second = bpm / 60.0
        total_beats = int(duration * beats_per_second)
        bars = total_beats // 4
        
        # Standard song structure
        sections = []
        
        # Determine progression type
        progression_type = 'pop'
        if genre in ['jazz', 'blues']:
            progression_type = genre
        
        # Create sections
        key = 'C'
        progression = self._theory.generate_progression(key, progression_type)
        
        # Intro (2 bars)
        intro_bars = min(2, bars // 8)
        sections.append({
            'type': 'intro',
            'bars': intro_bars,
            'chords': progression[:2] * intro_bars,
            'instruments': ['pad', 'bass']
        })
        
        # Verse (8 bars)
        verse_bars = min(8, bars // 4)
        sections.append({
            'type': 'verse',
            'bars': verse_bars,
            'chords': progression * (verse_bars // len(progression) + 1),
            'instruments': ['piano', 'bass', 'drums']
        })
        
        # Chorus (8 bars)
        chorus_bars = min(8, bars // 4)
        sections.append({
            'type': 'chorus',
            'bars': chorus_bars,
            'chords': progression * (chorus_bars // len(progression) + 1),
            'instruments': ['piano', 'guitar', 'bass', 'drums']
        })
        
        # Outro (2 bars)
        outro_bars = min(2, bars // 8)
        sections.append({
            'type': 'outro',
            'bars': outro_bars,
            'chords': progression[:2] * outro_bars,
            'instruments': ['pad']
        })
        
        return sections
    
    def _render_to_audio(self, composition: List[dict]) -> List[float]:
        """Render composition to audio samples."""
        max_duration = 30.0  # seconds
        max_samples = int(self._sample_rate * max_duration)
        
        # Initialize mix buffer
        mix = [0.0] * max_samples
        
        bpm = 120
        seconds_per_beat = 60.0 / bpm
        
        for section in composition:
            section_start_beat = 0
            
            for bar_idx in range(section['bars']):
                bar_start_beat = section_start_beat + bar_idx * 4
                
                # Get chord for this bar
                chord_idx = bar_idx % len(section['chords'])
                chord = section['chords'][chord_idx]
                
                # Render each instrument
                for instrument in section['instruments']:
                    if instrument == 'drums':
                        # Generate drum pattern
                        drum_pattern = self._rhythm_gen.generate_drum_pattern('pop', 1)
                        for hit in drum_pattern:
                            if len(hit) >= 2:
                                sound, beat = hit[0], hit[1]
                                hit_time = (bar_start_beat + beat) * seconds_per_beat
                                hit_sample = int(hit_time * self._sample_rate)
                                
                                if hit_sample < max_samples:
                                    # Simple drum synthesis
                                    velocity = hit[2] if len(hit) > 2 else 0.8
                                    drum_sound = self._synthesize_drum(sound, velocity)
                                    
                                    for i, sample in enumerate(drum_sound):
                                        if hit_sample + i < max_samples:
                                            mix[hit_sample + i] += sample
                    
                    else:
                        # Melodic instrument
                        note_duration = 4 * seconds_per_beat  # 1 bar
                        note_time = bar_start_beat * seconds_per_beat
                        note_sample = int(note_time * self._sample_rate)
                        
                        # Play chord root
                        chord_root = chord[0]
                        octave = 3 if instrument == 'bass' else 4
                        note = f"{chord_root}{octave}"
                        
                        samples = self._synthesizer.synthesize_note(
                            note, note_duration, instrument
                        )
                        
                        for i, sample in enumerate(samples):
                            if note_sample + i < max_samples:
                                mix[note_sample + i] += sample * 0.5
        
        return mix
    
    def _synthesize_drum(self, sound: str, velocity: float) -> List[float]:
        """Synthesize a single drum hit."""
        duration = 0.5
        num_samples = int(self._sample_rate * duration)
        samples = []
        
        if sound == 'kick':
            # Sine sweep for kick
            for i in range(num_samples):
                t = i / self._sample_rate
                freq = 100 * math.exp(-t * 20)
                amp = velocity * math.exp(-t * 10)
                samples.append(amp * math.sin(2 * math.pi * freq * t))
        
        elif sound == 'snare':
            # Noise burst for snare
            for i in range(num_samples):
                t = i / self._sample_rate
                amp = velocity * math.exp(-t * 15)
                noise = random.uniform(-1, 1)
                # Add some tonal component
                tone = 0.3 * math.sin(2 * math.pi * 200 * t)
                samples.append(amp * (noise * 0.7 + tone * 0.3))
        
        elif sound == 'hihat':
            # High frequency noise
            for i in range(min(num_samples // 4, num_samples)):
                t = i / self._sample_rate
                amp = velocity * math.exp(-t * 50)
                noise = random.uniform(-1, 1)
                samples.append(amp * noise * 0.3)
        
        else:
            samples = [0.0] * num_samples
        
        return samples
    
    def _mix_tracks(self, tracks: List[List[float]]) -> List[float]:
        """Mix multiple audio tracks together."""
        if not tracks:
            return []
        
        # Find max length
        max_len = max(len(t) for t in tracks)
        mix = [0.0] * max_len
        
        for track in tracks:
            for i, sample in enumerate(track):
                mix[i] += sample
        
        # Normalize to prevent clipping
        max_amp = max(abs(s) for s in mix) if mix else 1.0
        if max_amp > 1.0:
            mix = [s / max_amp for s in mix]
        
        return mix
    
    def _mastering(self, audio: List[float]) -> List[float]:
        """Apply final mastering effects."""
        if not audio:
            return audio
        
        # Soft compression
        compressed = []
        threshold = 0.7
        ratio = 4.0
        
        for sample in audio:
            abs_sample = abs(sample)
            if abs_sample > threshold:
                # Apply compression above threshold
                excess = abs_sample - threshold
                compressed_excess = excess / ratio
                new_abs = threshold + compressed_excess
                sign = 1 if sample > 0 else -1
                compressed.append(sign * new_abs)
            else:
                compressed.append(sample)
        
        # Normalize
        max_amp = max(abs(s) for s in compressed) if compressed else 1.0
        if max_amp > 0:
            compressed = [s / max_amp * 0.95 for s in compressed]
        
        return compressed


def save_wav(audio_data: List[float], filename: str, sample_rate: int = 44100):
    """Utility function to save audio data as WAV file."""
    # Convert float to 16-bit PCM
    pcm_data = [int(s * 32767) for s in audio_data]
    
    with wave.open(filename, 'w') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        
        for sample in pcm_data:
            # Clamp to 16-bit range
            sample = max(-32768, min(32767, sample))
            wav_file.writeframes(struct.pack('h', sample))


# Example usage
if __name__ == '__main__':
    # Create generator
    generator = MusicGenerator()
    
    # Generate music
    result = generator.generate(
        prompt="upbeat pop song in C major",
        duration=10.0,
        bpm=120
    )
    
    print(f"Generated {result.duration}s of music at {result.bpm} BPM")
    print(f"Audio samples: {len(result.audio_data)}")
    print(f"Metadata: {result.metadata}")
