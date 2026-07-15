import cv2
import numpy as np
from typing import List


class CameraController:
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.center = (width // 2, height // 2)

    def apply_motion(self, frames: List[np.ndarray], motion_type: str,
                     strength: float = 0.5, reverse: bool = False) -> List[np.ndarray]:
        num_frames = len(frames)
        if num_frames < 2:
            return frames

        func_map = {
            'pan': self._pan,
            'tilt': self._tilt,
            'zoom': self._zoom,
            'rotate': self._rotate,
            'dolly': self._dolly,
            'track': self._track
        }
        if motion_type not in func_map:
            raise ValueError(f"Unknown motion type: {motion_type}")

        processed = []
        for i, frame in enumerate(frames):
            t = i / (num_frames - 1) if num_frames > 1 else 0.5
            t = 1 - t if reverse else t
            processed.append(func_map[motion_type](frame, t, strength))
        return processed

    def _pan(self, frame: np.ndarray, t: float, strength: float) -> np.ndarray:
        dx = (t - 0.5) * self.width * strength
        M = np.float32([[1, 0, dx], [0, 1, 0]])
        return cv2.warpAffine(frame, M, (self.width, self.height), borderMode=cv2.BORDER_REFLECT)

    def _tilt(self, frame: np.ndarray, t: float, strength: float) -> np.ndarray:
        dy = (t - 0.5) * self.height * strength
        M = np.float32([[1, 0, 0], [0, 1, dy]])
        return cv2.warpAffine(frame, M, (self.width, self.height), borderMode=cv2.BORDER_REFLECT)

    def _zoom(self, frame: np.ndarray, t: float, strength: float) -> np.ndarray:
        scale = 1 + (t - 0.5) * strength * 0.5
        M = cv2.getRotationMatrix2D(self.center, 0, scale)
        return cv2.warpAffine(frame, M, (self.width, self.height), borderMode=cv2.BORDER_REFLECT)

    def _rotate(self, frame: np.ndarray, t: float, strength: float) -> np.ndarray:
        angle = (t - 0.5) * strength * 360
        M = cv2.getRotationMatrix2D(self.center, angle, 1)
        return cv2.warpAffine(frame, M, (self.width, self.height), borderMode=cv2.BORDER_REFLECT)

    def _dolly(self, frame: np.ndarray, t: float, strength: float) -> np.ndarray:
        scale = 1 + (t - 0.5) * strength
        M = cv2.getRotationMatrix2D(self.center, 0, scale)
        return cv2.warpAffine(frame, M, (self.width, self.height), borderMode=cv2.BORDER_REFLECT)

    def _track(self, frame: np.ndarray, t: float, strength: float) -> np.ndarray:
        dx = np.sin(t * np.pi) * self.width * strength * 0.3
        dy = np.cos(t * np.pi) * self.height * strength * 0.3
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        return cv2.warpAffine(frame, M, (self.width, self.height), borderMode=cv2.BORDER_REFLECT)


def apply_camera(frames: List[np.ndarray], motion_type: str,
                 strength: float = 0.5, reverse: bool = False) -> List[np.ndarray]:
    if not frames:
        return frames
    h, w = frames[0].shape[:2]
    controller = CameraController(w, h)
    return controller.apply_motion(frames, motion_type, strength, reverse)
