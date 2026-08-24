# =============================================================================
# Base Classes and Common Utilities
# =============================================================================

import os
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import decord

DEFAULT_VIDEO_EXTENSIONS = ('.mp4', '.avi', '.mov', '.mkv', '.webm')

class BaseVideoDataset(Dataset):
    """
    Base class for video datasets with common video processing logic.
    
    Provides:
    - Video reader caching via `_get_video_reader`
    - Frame sampling with FPS-based temporal alignment
    - Batch tensor processing (resize, normalize)
    """
    
    def __init__(
        self,
        video_size: Tuple[int, int] = (256, 256),
        num_frames: int = 16,
        target_fps: float = 8,
        cache_size: int = 16,
    ):
        self.video_size = video_size
        self.num_frames = num_frames
        self.target_fps = target_fps
        self._vr_cache_size = cache_size
        self._vr_cache = {}
        
        self.normalize = transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        )
        decord.bridge.set_bridge("torch")
    
    def _get_video_reader(self, video_path: str):
        """Create a new video reader every time (no cache)."""
        try:
            return decord.VideoReader(
                str(video_path),
                width=self.video_size[1],
                height=self.video_size[0]
            )
        except:
            return decord.VideoReader(str(video_path))
    

    def _batch_resize(self, video_tensor: torch.Tensor) -> torch.Tensor:
        """Batch resize video tensor to target size."""
        if video_tensor.shape[2] == self.video_size[0] and video_tensor.shape[3] == self.video_size[1]:
            return video_tensor
            
        B, C, H, W = video_tensor.shape
        video_tensor = video_tensor.view(B * C, 1, H, W)
        video_tensor = torch.nn.functional.interpolate(
            video_tensor, 
            size=self.video_size, 
            mode='bilinear', 
            antialias=True
        )
        return video_tensor.view(B, C, self.video_size[0], self.video_size[1])
    
    @staticmethod
    def sample_weighted_video_frames(
        total_frames: int,
        original_fps: float,
        num_frames: int,
        target_fps: float,
        strafes: np.ndarray,
        duration_alpha_coef: float = 1.0,
        strafe_clip_power: float = 0.0,
        eps: float = 1e-12,
    ) -> np.ndarray:
        """
        Sample video frames with importance weighting based on strafe scores.
        """
        if total_frames == 0:
            raise ValueError("Video has no frames")
        if num_frames <= 0:
            return np.empty((0,), dtype=np.int32)
        if original_fps <= 0 or target_fps <= 0:
            raise ValueError("FPS must be positive")

        strafes = np.asarray(strafes)
        if strafes.shape[0] != total_frames:
            raise ValueError(f"strafes length ({strafes.shape[0]}) must equal total_frames ({total_frames})")

        duration_0 = num_frames / original_fps
        duration_1 = num_frames / target_fps
        duration_alpha = (1 - duration_alpha_coef) * duration_0 + duration_alpha_coef * duration_1
        frames_needed = duration_alpha * original_fps

        if frames_needed > total_frames:
            return np.linspace(0, total_frames - 1, num_frames).astype(np.int32)

        window_len = int(max(1, np.ceil(frames_needed)))
        S = total_frames - window_len + 1

        base = np.abs(np.sin(np.deg2rad(strafes.astype(np.float64))))

        if strafe_clip_power <= 0:
            frame_w = np.ones_like(base)
        else:
            frame_w = np.power(base + eps, strafe_clip_power)

        if S <= 1:
            start_frame = 0
        else:
            prefix = np.zeros(total_frames + 1, dtype=np.float64)
            prefix[1:] = np.cumsum(frame_w)
            window_sums = prefix[window_len:] - prefix[:-window_len]

            total = window_sums.sum()
            if not np.isfinite(total) or total <= 0:
                start_frame = np.random.randint(0, S)
            else:
                p_start = window_sums / total
                start_frame = int(np.random.choice(np.arange(S), p=p_start))

        n = num_frames
        if n <= 1:
            return np.array([min(start_frame, total_frames - 1)], dtype=np.int32)

        sampling_interval = (frames_needed - 1) / (n - 1)

        indices = []
        current_frame = float(start_frame)
        for _ in range(n):
            frame_idx = int(np.round(current_frame))
            frame_idx = min(frame_idx, total_frames - 1)
            indices.append(frame_idx)
            current_frame += sampling_interval
            if current_frame >= total_frames:
                remaining = n - len(indices)
                indices.extend([total_frames - 1] * remaining)
                break

        return np.array(indices[:n], dtype=np.int32)

    @staticmethod
    def sample_video_frames(
        total_frames: int,
        original_fps: float,
        num_frames: int,
        target_fps: float,
        duration_alpha_coef: float = 1.0,
    ) -> np.ndarray:
        """
        Sample video frames uniformly while preserving temporal relationships.
        """
        duration_0 = num_frames / original_fps
        duration_1 = num_frames / target_fps
        duration_alpha = (1 - duration_alpha_coef) * duration_0 + duration_alpha_coef * duration_1
        frames_needed = duration_alpha * original_fps

        if total_frames == 0:
            raise ValueError("Video has no frames")

        if frames_needed > total_frames:
            return np.linspace(0, total_frames - 1, num_frames).astype(np.int32)

        max_start_frame = max(0, total_frames - int(np.ceil(frames_needed)))
        start_frame = np.random.randint(0, max_start_frame + 1) if max_start_frame > 0 else 0

        n = num_frames
        if n <= 1:
            return np.array([min(start_frame, total_frames - 1)], dtype=np.int32)
        sampling_interval = (frames_needed - 1) / (n - 1)

        indices = []
        current_frame = float(start_frame)
        for _ in range(n):
            frame_idx = int(np.round(current_frame))
            frame_idx = min(frame_idx, total_frames - 1)
            indices.append(frame_idx)
            current_frame += sampling_interval
            if current_frame >= total_frames:
                remaining = n - len(indices)
                indices.extend([total_frames - 1] * remaining)
                break

        return np.array(indices[:n], dtype=np.int32)
