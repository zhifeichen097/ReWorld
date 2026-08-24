# =============================================================================
# Pair Dataset Classes (Text, Text-Image, Text-Video)
# =============================================================================

import json
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image


class TextDataset(Dataset):
    """
    Simple text-only dataset for prompt loading.
    
    Supports optional extended prompts (e.g., LLM-expanded versions).
    """
    
    def __init__(self, prompt_path: str, extended_prompt_path: Optional[str] = None):
        """
        Args:
            prompt_path: Path to text file containing prompts (one per line).
            extended_prompt_path: Optional path to extended prompts file.
        """
        with open(prompt_path, encoding="utf-8") as f:
            self.prompt_list = [line.rstrip() for line in f]

        if extended_prompt_path is not None:
            with open(extended_prompt_path, encoding="utf-8") as f:
                self.extended_prompt_list = [line.rstrip() for line in f]
            assert len(self.extended_prompt_list) == len(self.prompt_list)
        else:
            self.extended_prompt_list = None

    def __len__(self):
        return len(self.prompt_list)

    def __getitem__(self, idx):
        batch = {
            "prompts": self.prompt_list[idx],
            "idx": idx,
        }
        if self.extended_prompt_list is not None:
            batch["extended_prompts"] = self.extended_prompt_list[idx]
        return batch


class TextImagePairDataset(Dataset):
    """
    Dataset for loading text-image pairs with crop metadata.
    
    Expects directory structure:
        data_dir/
            target_crop_info_{aspect_ratio}.json
            {aspect_ratio}/
                *.jpg/png
    """
    
    def __init__(
        self,
        data_dir: str,
        transform=None,
        eval_first_n: int = -1,
        pad_to_multiple_of: Optional[int] = None
    ):
        """
        Args:
            data_dir: Root directory containing metadata and images.
            transform: Optional transform to apply to images.
            eval_first_n: Limit dataset to first N samples (-1 = all).
            pad_to_multiple_of: Pad dataset size to multiple of this value.
        """
        self.transform = transform
        data_dir = Path(data_dir)

        # Find the metadata JSON file
        metadata_files = list(data_dir.glob('target_crop_info_*.json'))
        if not metadata_files:
            raise FileNotFoundError(f"No metadata file found in {data_dir}")
        if len(metadata_files) > 1:
            raise ValueError(f"Multiple metadata files found in {data_dir}")

        metadata_path = metadata_files[0]
        # Extract aspect ratio from metadata filename
        aspect_ratio = metadata_path.stem.split('_')[-1]

        # Use aspect ratio subfolder for images
        self.image_dir = data_dir / aspect_ratio
        if not self.image_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {self.image_dir}")

        # Load metadata
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)

        eval_first_n = eval_first_n if eval_first_n != -1 else len(self.metadata)
        self.metadata = self.metadata[:eval_first_n]

        # Verify all images exist
        for item in self.metadata:
            image_path = self.image_dir / item['file_name']
            if not image_path.exists():
                raise FileNotFoundError(f"Image not found: {image_path}")

        self.pre_pad_len = len(self.metadata)
        if pad_to_multiple_of is not None and len(self.metadata) % pad_to_multiple_of != 0:
            # Duplicate the last entry
            self.metadata += [self.metadata[-1]] * (
                pad_to_multiple_of - len(self.metadata) % pad_to_multiple_of
            )

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        """
        Returns:
            dict: A dictionary containing:
                - image: PIL Image
                - prompts: str (caption)
                - target_bbox: list of int [x1, y1, x2, y2]
                - target_ratio: str
                - type: str
                - origin_size: tuple of int (width, height)
                - idx: int
        """
        item = self.metadata[idx]

        # Load image
        image_path = self.image_dir / item['file_name']
        image = Image.open(image_path).convert('RGB')

        # Apply transform if specified
        if self.transform:
            image = self.transform(image)

        return {
            'image': image,
            'prompts': item['caption'],
            'target_bbox': item['target_crop']['target_bbox'],
            'target_ratio': item['target_crop']['target_ratio'],
            'type': item['type'],
            'origin_size': (item['origin_width'], item['origin_height']),
            'idx': idx
        }


class TextVideoPairDataset(Dataset):
    """
    Simple dataset for loading video-text pairs.
    
    Loads videos and their corresponding captions from a JSON file.
    """
    
    def __init__(
        self,
        data_dir: str,
        caption_json: str = "caption.json",
        video_size: Tuple[int, int] = (256, 256),
        num_frames: int = 16,
        target_fps: float = 8,
        video_extensions: Tuple[str, ...] = ('.mp4', '.avi', '.mov', '.mkv', '.webm'),
    ):
        """
        Args:
            data_dir: Directory containing videos and caption JSON.
            caption_json: Name of caption JSON file.
            video_size: Target (height, width) for video frames.
            num_frames: Number of frames to sample per video.
            target_fps: Target frames per second for sampling.
            video_extensions: Tuple of supported video file extensions.
        """
        from dataset.base import sample_video_frames
        
        self.data_dir = Path(data_dir)
        self.video_size = video_size
        self.num_frames = num_frames
        self.target_fps = target_fps
        self.sample_video_frames = sample_video_frames
        
        # Load caption JSON
        caption_path = self.data_dir / caption_json
        if not caption_path.exists():
            raise FileNotFoundError(f"Caption JSON file not found: {caption_path}")
        
        with open(caption_path, 'r', encoding='utf-8') as f:
            self.captions = json.load(f)
        
        # Find all video files in the directory
        self.video_files = []
        for ext in video_extensions:
            self.video_files.extend(list(self.data_dir.glob(f'*{ext}')))
            self.video_files.extend(list(self.data_dir.glob(f'*{ext.upper()}')))
        
        if len(self.video_files) == 0:
            raise ValueError(f"No video files found in {data_dir}")
        
        # Filter videos that have captions
        self.video_paths = []
        self.prompts = []
        for video_path in self.video_files:
            base_name = video_path.stem
            if base_name in self.captions:
                self.video_paths.append(video_path)
                self.prompts.append(self.captions[base_name])
        
        if len(self.video_paths) == 0:
            raise ValueError(f"No videos found with matching captions in {caption_json}")
        
        # Setup transforms for normalization
        from torchvision import transforms
        self.normalize = transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        )
        
        # Setup decord
        import decord
        decord.bridge.set_bridge('torch')
        
        # Cache video readers
        self._vr_cache = {}
        self._vr_cache_size = 16
    
    def _get_video_reader(self, video_path):
        """Cache video reader to avoid repeated file opening."""
        import decord
        
        if video_path not in self._vr_cache:
            if len(self._vr_cache) >= self._vr_cache_size:
                self._vr_cache.clear()
            try:
                self._vr_cache[video_path] = decord.VideoReader(
                    str(video_path), 
                    width=self.video_size[1], 
                    height=self.video_size[0]
                )
            except:
                self._vr_cache[video_path] = decord.VideoReader(str(video_path))
        return self._vr_cache[video_path]
    
    def __len__(self):
        return len(self.video_paths)
    
    def __getitem__(self, idx):
        """
        Returns:
            dict: A dictionary containing:
                - frames: Tensor of shape (num_frames, 3, height, width)
                - prompts: str
                - idx: int
        """
        video_path = self.video_paths[idx]
        prompt = self.prompts[idx]
        
        vr = self._get_video_reader(video_path)
        
        total_frames = len(vr)
        original_fps = vr.get_avg_fps()
        
        # Calculate frame sampling based on fps
        target_duration = self.num_frames / self.target_fps
        frames_needed = target_duration * original_fps
        
        if total_frames == 0:
            raise ValueError(f"Video {video_path} has no frames")
        
        if frames_needed > total_frames:
            indices = np.linspace(0, total_frames - 1, self.num_frames).astype(np.int32)
        else:
            sampling_interval = original_fps / self.target_fps
            indices = []
            current_frame = 0.0
            for _ in range(self.num_frames):
                frame_idx = int(np.round(current_frame))
                frame_idx = min(frame_idx, total_frames - 1)
                indices.append(frame_idx)
                current_frame += sampling_interval
                if current_frame >= total_frames:
                    remaining = self.num_frames - len(indices)
                    indices.extend([total_frames - 1] * remaining)
                    break
            
            indices = np.array(indices[:self.num_frames], dtype=np.int32)
        
        # Get video frames: shape (num_frames, height, width, 3)
        video_frames = vr.get_batch(indices).numpy()
        
        # Convert to tensor and permute to (num_frames, 3, height, width)
        video_tensor = torch.from_numpy(video_frames).permute(0, 3, 1, 2).contiguous()
        
        # Convert to float and normalize pixel values from [0, 255] to [0, 1]
        video_tensor = video_tensor.float() / 255.0
        
        # Batch resize
        if video_tensor.shape[2] != self.video_size[0] or video_tensor.shape[3] != self.video_size[1]:
            B, C, H, W = video_tensor.shape
            video_tensor = video_tensor.view(B * C, 1, H, W)
            video_tensor = torch.nn.functional.interpolate(
                video_tensor, 
                size=self.video_size, 
                mode='bilinear', 
                antialias=True
            )
            video_tensor = video_tensor.view(B, C, self.video_size[0], self.video_size[1])
        
        # Apply normalization: (x - 0.5) / 0.5 -> range [-1, 1]
        video_tensor = self.normalize(video_tensor)
        
        return {
            'frames': video_tensor,
            'prompts': prompt,
            'idx': idx
        }
