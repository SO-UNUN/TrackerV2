"""
modules/video_loader.py
Video loading and frame extraction for SHM tracking.
Supports high frame rate (15-60 fps) for capturing oscillation.

Author: University of Phayao | Physics Department
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Iterator, Tuple, Optional


class VideoLoader:
    """โหลดวิดีโอ + extract frames ทุกเฟรม (default) หรือตาม sampling rate"""

    def __init__(self, video_path: str):
        self.video_path = Path(video_path)
        if not self.video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        self.cap = cv2.VideoCapture(str(self.video_path))
        if not self.cap.isOpened():
            raise IOError(f"Cannot open video: {video_path}")

        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.duration_sec = self.total_frames / self.fps if self.fps > 0 else 0

    def get_metadata(self) -> dict:
        return {
            "fps": round(self.fps, 2),
            "total_frames": self.total_frames,
            "width": self.width,
            "height": self.height,
            "duration_sec": round(self.duration_sec, 2),
            "duration_min": round(self.duration_sec / 60, 2),
            "filename": self.video_path.name,
        }

    def get_first_frame(self) -> Optional[np.ndarray]:
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame = self.cap.read()
        if not ret:
            return None
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def get_frame_at(self, time_sec: float) -> Optional[np.ndarray]:
        if time_sec < 0 or time_sec > self.duration_sec:
            return None
        frame_idx = int(time_sec * self.fps)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self.cap.read()
        if not ret:
            return None
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def iter_frames(
        self, sampling_fps: Optional[float] = None
    ) -> Iterator[Tuple[float, np.ndarray]]:
        """Yield (timestamp_sec, frame_rgb) 
        
        Args:
            sampling_fps: None = ทุกเฟรม (default สำหรับ SHM ต้องการความถี่สูง)
                          ค่าตัวเลข = sample ที่ rate นั้น
        """
        if sampling_fps is None or sampling_fps >= self.fps:
            frame_skip = 1  # ทุกเฟรม
        elif sampling_fps <= 0:
            raise ValueError("sampling_fps must be > 0")
        else:
            frame_skip = max(1, int(round(self.fps / sampling_fps)))

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        frame_idx = 0
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            if frame_idx % frame_skip == 0:
                timestamp = frame_idx / self.fps
                yield timestamp, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_idx += 1

    def estimate_total_samples(self, sampling_fps: Optional[float] = None) -> int:
        if sampling_fps is None or sampling_fps >= self.fps:
            return self.total_frames
        if sampling_fps <= 0 or self.fps <= 0:
            return 0
        return int(self.duration_sec * sampling_fps)

    def close(self):
        if self.cap.isOpened():
            self.cap.release()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def get_supported_formats() -> list:
    return ["mp4", "mov", "avi", "mkv", "webm"]
