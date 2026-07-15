"""
Human-Like Mouse/Keyboard Simulation Module

Simulates natural human input behavior:
- Bezier curve mouse trajectories
- Perlin noise for natural movement
- Random micro-tremors
- Variable speed profiles
- Click timing randomization
- Typing rhythm simulation

Pure Python standard library only.
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Tuple, Dict, Optional, Any, Callable


@dataclass
class Point:
    """2D point with x, y coordinates."""
    x: float
    y: float

    def distance_to(self, other: Point) -> float:
        return math.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)

    def lerp(self, other: Point, t: float) -> Point:
        return Point(
            self.x + (other.x - self.x) * t,
            self.y + (other.y - self.y) * t,
        )

    def __add__(self, other: Point) -> Point:
        return Point(self.x + other.x, self.y + other.y)

    def __sub__(self, other: Point) -> Point:
        return Point(self.x - other.x, self.y - other.y)

    def __mul__(self, scalar: float) -> Point:
        return Point(self.x * scalar, self.y * scalar)

    def __repr__(self) -> str:
        return f"Point({self.x:.1f}, {self.y:.1f})"


@dataclass
class MouseAction:
    """Recorded mouse action."""
    action_type: str  # "move", "click", "scroll", "drag"
    x: float
    y: float
    timestamp: float
    button: str = "left"
    scroll_delta: int = 0
    duration: float = 0.0


@dataclass
class KeyEvent:
    """Recorded keyboard event."""
    key: str
    event_type: str  # "press", "release"
    timestamp: float
    duration: float = 0.0


class PerlinNoise:
    """
    1D and 2D Perlin noise generator for natural movement.

    Uses a permutation table and gradient vectors to generate
    smooth, continuous noise.
    """

    def __init__(self, seed: Optional[int] = None) -> None:
        self.rng = random.Random(seed)
        self.perm: List[int] = list(range(256))
        self.rng.shuffle(self.perm)
        self.perm = self.perm + self.perm  # Double for overflow

        # 2D gradient vectors
        self._gradients_2d: List[Tuple[float, float]] = [
            (1, 0), (-1, 0), (0, 1), (0, -1),
            (1, 1), (-1, 1), (1, -1), (-1, -1),
        ]

    def _fade(self, t: float) -> float:
        """Quintic fade function: 6t^5 - 15t^4 + 10t^3."""
        return t * t * t * (t * (t * 6 - 15) + 10)

    def _lerp(self, a: float, b: float, t: float) -> float:
        return a + t * (b - a)

    def noise1d(self, x: float) -> float:
        """Generate 1D Perlin noise."""
        xi = int(math.floor(x)) & 255
        xf = x - math.floor(x)
        u = self._fade(xf)

        g0 = self.perm[xi] / 255.0 * 2.0 - 1.0
        g1 = self.perm[xi + 1] / 255.0 * 2.0 - 1.0

        n0 = g0 * xf
        n1 = g1 * (xf - 1.0)

        return self._lerp(n0, n1, u)

    def noise2d(self, x: float, y: float) -> float:
        """Generate 2D Perlin noise."""
        xi = int(math.floor(x)) & 255
        yi = int(math.floor(y)) & 255
        xf = x - math.floor(x)
        yf = y - math.floor(y)
        u = self._fade(xf)
        v = self._fade(yf)

        # Hash corners
        aa = self.perm[self.perm[xi] + yi]
        ab = self.perm[self.perm[xi] + yi + 1]
        ba = self.perm[self.perm[xi + 1] + yi]
        bb = self.perm[self.perm[xi + 1] + yi + 1]

        # Gradient dot products
        def grad_dot(hash_val: int, dx: float, dy: float) -> float:
            g = self._gradients_2d[hash_val % len(self._gradients_2d)]
            return g[0] * dx + g[1] * dy

        n00 = grad_dot(aa, xf, yf)
        n10 = grad_dot(ba, xf - 1, yf)
        n01 = grad_dot(ab, xf, yf - 1)
        n11 = grad_dot(bb, xf - 1, yf - 1)

        x1 = self._lerp(n00, n10, u)
        x2 = self._lerp(n01, n11, u)
        return self._lerp(x1, x2, v)

    def octave_noise1d(self, x: float, octaves: int = 4,
                       persistence: float = 0.5) -> float:
        """Generate fractal 1D noise with multiple octaves."""
        total = 0.0
        amplitude = 1.0
        frequency = 1.0
        max_value = 0.0

        for _ in range(octaves):
            total += self.noise1d(x * frequency) * amplitude
            max_value += amplitude
            amplitude *= persistence
            frequency *= 2.0

        return total / max_value if max_value > 0 else 0.0

    def octave_noise2d(self, x: float, y: float, octaves: int = 4,
                       persistence: float = 0.5) -> float:
        """Generate fractal 2D noise with multiple octaves."""
        total = 0.0
        amplitude = 1.0
        frequency = 1.0
        max_value = 0.0

        for _ in range(octaves):
            total += self.noise2d(x * frequency, y * frequency) * amplitude
            max_value += amplitude
            amplitude *= persistence
            frequency *= 2.0

        return total / max_value if max_value > 0 else 0.0


class BezierTrajectory:
    """
    Generate Bezier curve mouse trajectories.

    Creates natural-looking mouse movement paths using cubic Bezier curves
    with randomized control points.
    """

    def __init__(self, seed: Optional[int] = None) -> None:
        self.rng = random.Random(seed)

    def cubic_bezier(self, p0: Point, p1: Point, p2: Point,
                     p3: Point, t: float) -> Point:
        """Evaluate a cubic Bezier curve at parameter t."""
        u = 1.0 - t
        return Point(
            u ** 3 * p0.x + 3 * u ** 2 * t * p1.x + 3 * u * t ** 2 * p2.x + t ** 3 * p3.x,
            u ** 3 * p0.y + 3 * u ** 2 * t * p1.y + 3 * u * t ** 2 * p2.y + t ** 3 * p3.y,
        )

    def generate_control_points(self, start: Point, end: Point,
                                 curvature: float = 0.3) -> Tuple[Point, Point]:
        """
        Generate random control points for a cubic Bezier curve.

        The control points are offset perpendicular to the straight line
        between start and end, creating a natural curve.
        """
        dx = end.x - start.x
        dy = end.y - start.y
        dist = math.sqrt(dx * dx + dy * dy)

        if dist < 1.0:
            return (start, end)

        # Perpendicular direction
        px = -dy / dist
        py = dx / dist

        # Random offsets for control points
        offset1 = self.rng.gauss(0, dist * curvature)
        offset2 = self.rng.gauss(0, dist * curvature)

        # Control points at 1/3 and 2/3 along the line
        cp1 = Point(
            start.x + dx / 3.0 + px * offset1,
            start.y + dy / 3.0 + py * offset1,
        )
        cp2 = Point(
            start.x + 2 * dx / 3.0 + px * offset2,
            start.y + 2 * dy / 3.0 + py * offset2,
        )

        return (cp1, cp2)

    def generate_trajectory(self, start: Point, end: Point,
                            num_points: int = 50,
                            curvature: float = 0.3) -> List[Point]:
        """
        Generate a Bezier curve trajectory from start to end.

        Returns a list of points along the curve.
        """
        cp1, cp2 = self.generate_control_points(start, end, curvature)
        points: List[Point] = []

        for i in range(num_points + 1):
            t = i / num_points
            point = self.cubic_bezier(start, cp1, cp2, end, t)
            points.append(point)

        return points

    def generate_multi_segment(self, waypoints: List[Point],
                               points_per_segment: int = 30,
                               curvature: float = 0.2) -> List[Point]:
        """
        Generate a multi-segment Bezier trajectory through waypoints.

        Smoothly connects multiple waypoints with Bezier curves.
        """
        if len(waypoints) < 2:
            return waypoints

        all_points: List[Point] = []
        for i in range(len(waypoints) - 1):
            segment = self.generate_trajectory(
                waypoints[i], waypoints[i + 1],
                points_per_segment, curvature,
            )
            if i > 0:
                segment = segment[1:]  # Avoid duplicate points
            all_points.extend(segment)

        return all_points


class SpeedProfile:
    """
    Variable speed profiles for mouse movement.

    Simulates natural acceleration and deceleration patterns.
    """

    def __init__(self, profile_type: str = "natural") -> None:
        self.profile_type = profile_type

    def ease_in_out(self, t: float) -> float:
        """Smooth ease-in-out (cubic)."""
        if t < 0.5:
            return 4 * t * t * t
        else:
            return 1 - (-2 * t + 2) ** 3 / 2

    def ease_out_quad(self, t: float) -> float:
        """Quadratic ease-out."""
        return 1 - (1 - t) ** 2

    def ease_in_quad(self, t: float) -> float:
        """Quadratic ease-in."""
        return t * t

    def natural_profile(self, t: float) -> float:
        """
        Natural human movement speed profile.

        Features a quick acceleration phase, sustained speed,
        and gradual deceleration near the target.
        """
        if t < 0.1:
            # Quick start
            return self.ease_in_quad(t / 0.1) * 0.3
        elif t < 0.7:
            # Sustained speed with slight variation
            base = 0.3 + (t - 0.1) / 0.6 * 0.5
            return base + 0.05 * math.sin(t * 20)
        else:
            # Deceleration
            return 0.8 + self.ease_out_quad((t - 0.7) / 0.3) * 0.2

    def get_profile(self, t: float) -> float:
        """Get the speed multiplier at parameter t."""
        if self.profile_type == "natural":
            return self.natural_profile(t)
        elif self.profile_type == "linear":
            return t
        elif self.profile_type == "ease_in_out":
            return self.ease_in_out(t)
        elif self.profile_type == "ease_out":
            return self.ease_out_quad(t)
        return t

    def apply_profile(self, trajectory: List[Point]) -> List[Tuple[Point, float]]:
        """
        Apply the speed profile to a trajectory.

        Returns a list of (point, time_offset) tuples where time_offset
        represents when each point should be reached.
        """
        if not trajectory:
            return []

        total_dist = 0.0
        distances: List[float] = [0.0]
        for i in range(1, len(trajectory)):
            d = trajectory[i].distance_to(trajectory[i - 1])
            total_dist += d
            distances.append(total_dist)

        if total_dist == 0:
            return [(p, 0.0) for p in trajectory]

        # Base duration: proportional to distance
        base_duration = 0.2 + total_dist * 0.002  # seconds

        result: List[Tuple[Point, float]] = []
        for i, point in enumerate(trajectory):
            t = distances[i] / total_dist
            speed = self.get_profile(t)
            time_offset = speed * base_duration
            result.append((point, time_offset))

        return result


class HumanLikeMouse:
    """
    Human-like mouse movement simulator.

    Combines Bezier trajectories, Perlin noise, micro-tremors,
    and variable speed profiles to produce natural mouse movement.
    """

    def __init__(self, seed: Optional[int] = None,
                 noise_amplitude: float = 2.0,
                 tremor_amplitude: float = 0.5,
                 tremor_frequency: float = 15.0) -> None:
        self.rng = random.Random(seed)
        self.perlin = PerlinNoise(seed)
        self.bezier = BezierTrajectory(seed)
        self.speed_profile = SpeedProfile("natural")
        self.noise_amplitude = noise_amplitude
        self.tremor_amplitude = tremor_amplitude
        self.tremor_frequency = tremor_frequency
        self._time_offset = self.rng.random() * 1000

    def move(self, start: Point, end: Point,
             duration: Optional[float] = None) -> List[MouseAction]:
        """
        Generate a human-like mouse movement from start to end.

        Returns a list of MouseAction objects representing the movement.
        """
        # Generate Bezier trajectory
        dist = start.distance_to(end)
        num_points = max(20, int(dist / 2))
        trajectory = self.bezier.generate_trajectory(start, end, num_points)

        # Apply speed profile
        profiled = self.speed_profile.apply_profile(trajectory)

        # Add noise and tremors
        actions: List[MouseAction] = []
        base_time = time.time()

        for i, (point, time_offset) in enumerate(profiled):
            # Add Perlin noise for organic movement
            noise_x = self.perlin.noise1d(i * 0.1 + self._time_offset) * self.noise_amplitude
            noise_y = self.perlin.noise1d(i * 0.1 + self._time_offset + 100) * self.noise_amplitude

            # Add micro-tremors (high frequency, low amplitude)
            tremor_x = math.sin(i * self.tremor_frequency * 0.1) * self.tremor_amplitude
            tremor_y = math.cos(i * self.tremor_frequency * 0.13) * self.tremor_amplitude

            final_x = point.x + noise_x + tremor_x
            final_y = point.y + noise_y + tremor_y

            actions.append(MouseAction(
                action_type="move",
                x=final_x, y=final_y,
                timestamp=base_time + time_offset,
                duration=0.0,
            ))

        return actions

    def click(self, position: Point, button: str = "left",
              double_click: bool = False) -> List[MouseAction]:
        """
        Generate a human-like click with natural timing.

        Includes a small pre-click pause and slight position jitter.
        """
        rng = random.Random()
        base_time = time.time()
        actions: List[MouseAction] = []

        # Pre-click pause (50-150ms)
        pause = rng.uniform(0.05, 0.15)
        actions.append(MouseAction(
            action_type="move",
            x=position.x + rng.gauss(0, 0.5),
            y=position.y + rng.gauss(0, 0.5),
            timestamp=base_time,
            duration=pause,
        ))

        # Click down
        actions.append(MouseAction(
            action_type="click",
            x=position.x, y=position.y,
            timestamp=base_time + pause,
            button=button,
            duration=rng.uniform(0.05, 0.12),
        ))

        if double_click:
            # Inter-click delay (30-80ms)
            delay = rng.uniform(0.03, 0.08)
            actions.append(MouseAction(
                action_type="click",
                x=position.x + rng.gauss(0, 0.3),
                y=position.y + rng.gauss(0, 0.3),
                timestamp=base_time + pause + delay,
                button=button,
                duration=rng.uniform(0.05, 0.12),
            ))

        return actions

    def drag(self, start: Point, end: Point,
             button: str = "left") -> List[MouseAction]:
        """Generate a human-like drag operation."""
        move_actions = self.move(start, end)
        actions: List[MouseAction] = []

        # Press at start
        actions.append(MouseAction(
            action_type="click",
            x=start.x, y=start.y,
            timestamp=move_actions[0].timestamp - 0.05,
            button=button,
        ))

        # Move with button held
        actions.extend(move_actions)

        # Release at end
        if move_actions:
            actions.append(MouseAction(
                action_type="click",
                x=end.x, y=end.y,
                timestamp=move_actions[-1].timestamp + 0.05,
                button=button,
            ))

        return actions

    def scroll(self, position: Point, delta: int = 3,
               direction: str = "down") -> List[MouseAction]:
        """Generate human-like scroll actions."""
        rng = random.Random()
        base_time = time.time()
        actions: List[MouseAction] = []
        scroll_delta = delta if direction == "down" else -delta

        for i in range(abs(delta)):
            actions.append(MouseAction(
                action_type="scroll",
                x=position.x + rng.gauss(0, 1.0),
                y=position.y + rng.gauss(0, 1.0),
                timestamp=base_time + i * rng.uniform(0.02, 0.06),
                scroll_delta=scroll_delta,
            ))

        return actions


class TypingRhythm:
    """
    Simulates natural human typing rhythm.

    Models variable inter-key timing, common typing patterns,
    and realistic error rates.
    """

    def __init__(self, wpm: float = 60.0, error_rate: float = 0.02,
                 seed: Optional[int] = None) -> None:
        self.rng = random.Random(seed)
        self.wpm = wpm  # Words per minute
        self.error_rate = error_rate
        self.base_interval = 60.0 / (wpm * 5.0)  # Average char interval in seconds

        # Common bigram frequencies (simplified)
        self._bigram_delays: Dict[str, float] = {
            "th": 0.8, "he": 0.85, "in": 0.9, "er": 0.85, "an": 0.9,
            "en": 0.9, "to": 0.8, "it": 0.85, "is": 0.8, "ha": 0.9,
            "on": 0.85, "nd": 0.9, "or": 0.85, "re": 0.85, "at": 0.8,
            "ed": 0.9, "es": 0.85, "st": 0.9, "nt": 0.9, "ti": 0.85,
        }

        # Same finger delays (slower when same finger types consecutive keys)
        self._same_hand_delay = 1.3
        self._same_finger_delay = 1.8

        # Key to finger mapping (simplified QWERTY)
        self._key_fingers: Dict[str, int] = {
            "q": 0, "a": 0, "z": 0,
            "w": 1, "s": 1, "x": 1,
            "e": 2, "d": 2, "c": 2,
            "r": 2, "f": 2, "v": 2,
            "t": 3, "g": 3, "b": 3,
            "y": 4, "h": 4, "n": 4,
            "u": 5, "j": 5, "m": 5,
            "i": 6, "k": 6,
            "o": 7, "l": 7,
            "p": 7,
        }

    def type_text(self, text: str) -> List[KeyEvent]:
        """
        Generate typing events for the given text.

        Returns a list of KeyEvent objects with natural timing.
        """
        events: List[KeyEvent] = []
        current_time = time.time()

        for i, char in enumerate(text):
            # Calculate inter-key delay
            delay = self._calculate_delay(text, i)

            # Check for typing error
            if self.rng.random() < self.error_rate and char.isalpha():
                # Type wrong character first, then backspace and correct
                wrong_char = self._get_wrong_char(char)
                events.append(KeyEvent(
                    key=wrong_char, event_type="press",
                    timestamp=current_time,
                    duration=self.base_interval * 0.5,
                ))
                current_time += delay * 0.5

                # Pause to notice error (longer pause)
                error_pause = self.rng.uniform(0.2, 0.5)
                current_time += error_pause

                # Backspace
                events.append(KeyEvent(
                    key="backspace", event_type="press",
                    timestamp=current_time,
                    duration=self.base_interval * 0.3,
                ))
                current_time += delay * 0.3

                # Type correct character
                events.append(KeyEvent(
                    key=char, event_type="press",
                    timestamp=current_time,
                    duration=self.base_interval * self.rng.uniform(0.8, 1.2),
                ))
                current_time += delay
            else:
                events.append(KeyEvent(
                    key=char, event_type="press",
                    timestamp=current_time,
                    duration=self.base_interval * self.rng.uniform(0.7, 1.3),
                ))
                current_time += delay

        return events

    def _calculate_delay(self, text: str, index: int) -> float:
        """Calculate the inter-key delay based on context."""
        if index == 0:
            return self.base_interval * self.rng.uniform(1.5, 3.0)  # Initial hesitation

        prev_char = text[index - 1].lower()
        curr_char = text[index].lower()

        delay = self.base_interval

        # Bigram-based adjustment
        bigram = prev_char + curr_char
        if bigram in self._bigram_delays:
            delay *= self._bigram_delays[bigram]

        # Same finger / same hand adjustment
        prev_finger = self._key_fingers.get(prev_char, -1)
        curr_finger = self._key_fingers.get(curr_char, -1)
        if prev_finger >= 0 and curr_finger >= 0:
            if prev_finger == curr_finger:
                delay *= self._same_finger_delay
            elif (prev_finger < 4 and curr_finger < 4) or \
                 (prev_finger >= 4 and curr_finger >= 4):
                delay *= self._same_hand_delay

        # Space after punctuation (longer pause)
        if prev_char in ".!?;:":
            delay *= self.rng.uniform(2.0, 4.0)
        elif prev_char == ",":
            delay *= self.rng.uniform(1.5, 2.5)
        elif prev_char == " ":
            delay *= self.rng.uniform(0.9, 1.1)

        # Add random variation
        delay *= self.rng.gauss(1.0, 0.15)

        return max(0.02, delay)

    def _get_wrong_char(self, correct_char: str) -> str:
        """Get a plausible wrong character near the correct one on the keyboard."""
        nearby_keys: Dict[str, str] = {
            "a": "sq", "b": "vn", "c": "xv", "d": "sf", "e": "wr",
            "f": "dg", "g": "fh", "h": "gj", "i": "uo", "j": "hk",
            "k": "jl", "l": "k;", "m": "n,", "n": "bm", "o": "ip",
            "p": "o[", "q": "wa", "r": "et", "s": "ad", "t": "ry",
            "u": "yi", "v": "cb", "w": "qe", "x": "zc", "y": "tu",
            "z": "xa",
        }
        neighbors = nearby_keys.get(correct_char.lower(), "abcdefghijklmnopqrstuvwxyz")
        return self.rng.choice(neighbors)


class GestureSimulator:
    """
    Simulates common mouse gestures.

    Includes swipe, pinch, zoom, and custom gesture patterns.
    """

    def __init__(self, seed: Optional[int] = None) -> None:
        self.rng = random.Random(seed)
        self.mouse = HumanLikeMouse(seed)

    def swipe(self, start: Point, end: Point,
              speed: str = "medium") -> List[MouseAction]:
        """
        Simulate a swipe gesture.

        Speed can be "slow", "medium", or "fast".
        """
        speed_map = {"slow": 1.5, "medium": 1.0, "fast": 0.5}
        duration_factor = speed_map.get(speed, 1.0)

        actions = self.mouse.move(start, end)
        # Adjust timing
        if actions:
            total_duration = actions[-1].timestamp - actions[0].timestamp
            start_time = actions[0].timestamp
            for i, action in enumerate(actions):
                progress = i / max(1, len(actions) - 1)
                action.timestamp = start_time + progress * total_duration * duration_factor

        return actions

    def circle(self, center: Point, radius: float = 50.0,
               num_points: int = 36) -> List[MouseAction]:
        """Simulate a circular gesture around a center point."""
        actions: List[MouseAction] = []
        base_time = time.time()

        for i in range(num_points + 1):
            angle = 2 * math.pi * i / num_points
            x = center.x + radius * math.cos(angle) + self.rng.gauss(0, 0.5)
            y = center.y + radius * math.sin(angle) + self.rng.gauss(0, 0.5)

            actions.append(MouseAction(
                action_type="move",
                x=x, y=y,
                timestamp=base_time + i * 0.02,
            ))

        return actions

    def zigzag(self, start: Point, end: Point,
               amplitude: float = 20.0,
               segments: int = 8) -> List[MouseAction]:
        """Simulate a zigzag gesture between two points."""
        dx = end.x - start.x
        dy = end.y - start.y
        dist = math.sqrt(dx * dx + dy * dy)

        if dist < 1:
            return []

        # Perpendicular direction
        px = -dy / dist
        py = dx / dist

        actions: List[MouseAction] = []
        base_time = time.time()

        for i in range(segments + 1):
            t = i / segments
            base_x = start.x + dx * t
            base_y = start.y + dy * t

            # Zigzag offset
            offset = amplitude * (1 if i % 2 == 0 else -1)
            if i == 0 or i == segments:
                offset = 0  # Start and end on the line

            x = base_x + px * offset + self.rng.gauss(0, 0.3)
            y = base_y + py * offset + self.rng.gauss(0, 0.3)

            actions.append(MouseAction(
                action_type="move",
                x=x, y=y,
                timestamp=base_time + t * 0.5,
            ))

        return actions

    def random_movement(self, bounds: Tuple[float, float, float, float],
                        duration: float = 2.0,
                        num_points: int = 100) -> List[MouseAction]:
        """
        Generate random mouse movement within bounds.

        Uses Perlin noise for smooth, natural-looking random movement.
        """
        perlin = PerlinNoise(seed=self.rng.randint(0, 100000))
        actions: List[MouseAction] = []
        base_time = time.time()
        x_min, y_min, x_max, y_max = bounds
        cx = (x_min + x_max) / 2
        cy = (y_min + y_max) / 2
        range_x = (x_max - x_min) / 2
        range_y = (y_max - y_min) / 2

        for i in range(num_points):
            t = i / num_points
            nx = perlin.noise1d(i * 0.05) * range_x * 0.8
            ny = perlin.noise1d(i * 0.05 + 500) * range_y * 0.8

            x = max(x_min, min(x_max, cx + nx))
            y = max(y_min, min(y_max, cy + ny))

            actions.append(MouseAction(
                action_type="move",
                x=x, y=y,
                timestamp=base_time + t * duration,
            ))

        return actions
