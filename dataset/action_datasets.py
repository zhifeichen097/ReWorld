# Video datasets with paired action/camera trajectories.

import os, sys
import io
import json
from pathlib import Path
from typing import Any, Optional, Tuple
import glob
import pickle
import numpy as np
import pandas as pd
import torch

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from dataset.base import (
    BaseVideoDataset,
    DEFAULT_VIDEO_EXTENSIONS,
)
from utils.camera import (
    generate_constant_action_sequence,
    generate_action_sequence_from_keys,
    generate_multikey_action_sequence,
    parse_key_sequence,
    c2w_to_relative_c2w,
)
from utils.interp_pose import sample_and_interpolate_poses_se3, downsample_poses_se3
from utils.default_coord import default_coord
from utils.match_coord import match_coord_to_standard, standard_to_match_coord
from utils.visualize import visualize_trajectory

def _as_float32_ndarray(x: Any, *, name: str) -> np.ndarray:
    arr = np.asarray(x)
    if arr.dtype == object:
        raise TypeError(f"{name} must be a numeric np.ndarray, got dtype=object")
    if arr.ndim < 1:
        raise ValueError(f"{name} must have at least 1 dimension (time axis=0), got shape={arr.shape}")
    if not np.issubdtype(arr.dtype, np.number):
        raise TypeError(f"{name} must be numeric, got dtype={arr.dtype}")
    return arr.astype(np.float32, copy=False)


def _as_index_array(x: Any, *, name: str) -> np.ndarray:
    arr = np.asarray(x)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a 1D index array, got shape={arr.shape}")
    if arr.dtype.kind not in {"i", "u"}:
        raise TypeError(f"{name} must be integer indices, got dtype={arr.dtype}")
    return arr


def upsample_sequence(
    sequence: np.ndarray,
    df: int = 4,
) -> np.ndarray:
    """
    Upsample a sequence along axis=0 by repeating steps.

    Keeps the first step intact, and repeats each subsequent step `df` times.
    This is the inverse-style companion of `downsample_sequence(..., sampling_mode="interval")`.
    """
    sequence = np.asarray(sequence)
    if sequence.dtype == object:
        raise TypeError("sequence must be a numeric np.ndarray, got dtype=object")
    if df < 1:
        raise ValueError(f"df must be >= 1, got {df}")
    if sequence.shape[0] < 2:
        return sequence

    first_step = sequence[0:1]
    rest_steps = sequence[1:]
    upsampled_rest = np.repeat(rest_steps, df, axis=0)
    return np.concatenate([first_step, upsampled_rest], axis=0)


def downsample_sequence(
    sequence: np.ndarray, 
    df: int = 4, 
    sampling_mode: str = 'interval'
) -> np.ndarray:
    """
    Downsample an action/pose sequence.

    Operates only on axis=0 (time/step). All other dimensions are preserved.

    - interval: keep the first step, then take every df-th step.
    - interpolation: keep the first step, then average each df-sized window.
    """
    sequence = np.asarray(sequence)
    if sequence.dtype == object:
        raise TypeError("sequence must be a numeric np.ndarray, got dtype=object")
    if df < 1:
        raise ValueError(f"df must be >= 1, got {df}")
    if sampling_mode == 'interval':
        return np.concatenate([sequence[0:1], sequence[1::df]], axis=0)
    elif sampling_mode == 'interpolation':
        T = sequence.shape[0]
        if T <= 1:
            return sequence
        chunks = []
        for start in range(1, T, df):
            end = min(start + df, T)
            window = sequence[start:end]
            chunks.append(window.mean(axis=0, keepdims=True))
        return np.concatenate([sequence[0:1]] + chunks, axis=0)
    else:
        raise ValueError(f"Invalid sampling mode: {sampling_mode}")


class BaseActionDataset(BaseVideoDataset):
    def __init__(
        self,
        data_dir: str,
        video_size: Tuple[int, int] = (480, 832),
        num_frames: int = 16,
        target_fps: float = 24,
        cache_size: int = 1,
        source_action_mode: str = 'plucker_embedding',
        target_action_mode: str = 'delta_relative_pose',
        action_downsampling_factor: int = 4,
        prope_fixed_divisor: Optional[float] = None,
    ):
        super().__init__(
            video_size=video_size,
            num_frames=num_frames,
            target_fps=target_fps,
            cache_size=cache_size,
        )
        self.source_action_mode = default_coord[source_action_mode]  # source action/camera convention/parameterization
        self.target_action_mode = default_coord[target_action_mode] # e.g. relative pose / delta pose / Plücker, etc.
        self.action_downsampling_factor = action_downsampling_factor
        # Normalization of the prope_context translation: None = per-clip max-abs
        # (legacy behavior, reproduces older checkpoints); a number = fixed divisor
        # (1.0 = raw metric units; all clips share one unit, preserving absolute
        # cross-clip scale).
        self.prope_fixed_divisor = None if prope_fixed_divisor is None else float(prope_fixed_divisor)
        self.initialize_files(data_dir)
    
    def initialize_files(self, data_dir):
        self.video_paths = []
        self.action_paths = []
        self.prompts = []

    def __len__(self):
        return len(self.video_paths)

    def load_action_from_path(
        self,
        action_path: str,
    ) -> np.ndarray:
        raise NotImplementedError("Subclasses must implement this method")

    def convert_source_action_to_target_action(
        self,
        source_action: np.ndarray
    ) -> np.ndarray:
        return standard_to_match_coord(match_coord_to_standard(source_action, self.source_action_mode), self.target_action_mode)
    
    def temporal_augmentation(
        self,
        indices: np.ndarray
    ) -> np.ndarray:
        return indices

    # def __getitem__(self, idx):
    #     if len(self.video_paths) == 0:
    #         raise IndexError("Empty dataset")

    #     last_exc: Optional[BaseException] = None
    #     for attempt in range(8):
    #         cur_idx = int(idx) if attempt == 0 else int(np.random.randint(0, len(self.video_paths)))
    #         video_path = self.video_paths[cur_idx]
    #         action_path = self.action_paths[cur_idx]
    #         prompt = self.prompts[cur_idx]
    #         try:
    #             action = _as_float32_ndarray(self.load_action_from_path(action_path), name="action")

    #             vr = self._get_video_reader(video_path)
    #             total_frames = len(vr)
    #             original_fps = vr.get_avg_fps()

    #             raw_indices = _as_index_array(
    #                 self.sample_video_frames(
    #                     total_frames=total_frames,
    #                     original_fps=original_fps,
    #                     num_frames=self.num_frames,
    #                     target_fps=self.target_fps,
    #                 ),
    #                 name="indices",
    #             )

    #             video_frames = vr.get_batch(raw_indices).numpy()
    #             action = action[raw_indices]

    #             seq_len = len(raw_indices)
    #             base_idx_map = np.arange(seq_len, dtype=np.int32)
    #             aug_map = _as_index_array(
    #                 self.temporal_augmentation(base_idx_map), 
    #                 name="augmented_indices"
    #             )

    #             video_frames = video_frames[aug_map].copy()
    #             action = action[aug_map].copy()

    #             idx = cur_idx
    #             break
    #         except (ValueError, IndexError, FileNotFoundError, OSError, RuntimeError) as e:
    #             last_exc = e
    #             continue
    #     else:
    #         raise RuntimeError("Failed to sample a valid video/action pair after retries") from last_exc

    #     action_downsample = downsample_sequence(
    #         action, self.action_downsampling_factor, sampling_mode='interpolation'
    #     )
    #     target_action_downsample = self.convert_source_action_to_target_action(
    #         action_downsample
    #     )

    #     target_action_downsample_tensor = torch.from_numpy(target_action_downsample).float()
    #     video_tensor = torch.from_numpy(video_frames).permute(0, 3, 1, 2).contiguous()
    #     video_tensor = video_tensor.float() / 255.0
    #     video_tensor = self._batch_resize(video_tensor)
    #     video_tensor = self.normalize(video_tensor)

    #     return {
    #         'frames': video_tensor.permute(1, 0, 2, 3),
    #         'prompts': prompt,
    #         'idx': idx,
    #         'actions': target_action_downsample_tensor,
    #     }
    def __getitem__(self, idx):
        if len(self.video_paths) == 0:
            raise IndexError("Empty dataset")

        last_exc: Optional[BaseException] = None
        for attempt in range(8):
            cur_idx = int(idx) if attempt == 0 else int(np.random.randint(0, len(self.video_paths)))
            video_path = self.video_paths[cur_idx]
            action_path = self.action_paths[cur_idx]
            prompt = self.prompts[cur_idx]
            
            try:
                # 1. guard against non-video entries
                if not str(video_path).lower().endswith(('.mp4', '.mov', '.avi', '.mkv')):
                    raise ValueError(f"Invalid video format: {video_path}")

                # 2. load the raw video and actions
                action_raw = self.load_action_from_path(action_path)
                vr = self._get_video_reader(video_path)
                
                total_video_frames = len(vr)
                original_fps = vr.get_avg_fps()

                # 3. build the sampled video frame indices (e.g. 81 frame indices)
                raw_indices = _as_index_array(
                    self.sample_video_frames(
                        total_frames=total_video_frames,
                        original_fps=original_fps,
                        num_frames=self.num_frames,
                        target_fps=self.target_fps,
                    ),
                    name="indices",
                )

                # =========================================================
                # Exact on-demand interpolation on the SE(3) Lie group.
                # Pipeline: raw absolute c2w  ->  SE3 upsample on absolute  ->  sample clip
                #            ->  c2w_to_relative_c2w on clip (anchor at clip's first frame)
                # =========================================================
                # Step A: convert the source-specific raw poses into 4x4 SE(3) **absolute c2w** (OpenCV coords).
                # skip_relative=True skips the "video-frame-0 anchor" inside match_coord; anchoring is
                # deferred until after clip sampling — otherwise the clip start would not be identity,
                # making training and inference inconsistent.
                standard_poses = match_coord_to_standard(
                    action_raw, self.source_action_mode, skip_relative=True
                )

                # Step B: SE(3) Lie-group upsampling on absolute c2w, picking the clip's N poses.
                # SE3 interpolation is left-equivariant: upsampling on absolute vs. relative poses
                # gives the same result (up to a global left-multiplication), so the order does not
                # change the numbers — only the clarity of the logic.
                sampled_standard_poses = sample_and_interpolate_poses_se3(
                    poses=standard_poses,
                    original_video_len=total_video_frames,
                    target_indices=raw_indices
                )
                
                # Step C: fetch the video frames for the sampled indices
                video_frames = vr.get_batch(raw_indices).numpy()

                # 4. temporal augmentation (reverse, palindrome, ...)
                seq_len = len(raw_indices)
                base_idx_map = np.arange(seq_len, dtype=np.int32)
                aug_map = _as_index_array(self.temporal_augmentation(base_idx_map), name="augmented_indices")

                video_frames = video_frames[aug_map].copy()
                sampled_standard_poses = sampled_standard_poses[aug_map].copy()

                idx = cur_idx
                break
            except Exception as e:
                last_exc = e
                continue
        else:
            raise RuntimeError(f"Failed to sample after 8 retries. Last error: {last_exc}")

        # 5. action-space downsampling and target-format conversion
        # Note: sampled_standard_poses are already standard 4x4 matrices.
        # Use the SE(3)-aware downsampler: rotations via Lie-group interpolation (slerp),
        # translations averaged separately. The generic `downsample_sequence(... 'interpolation')`
        # would arithmetically average 4x4 matrices — with palindrome augmentation the mirror
        # point mixes opposite-facing poses, the averaged rotation's det collapses toward 0,
        # and np.linalg.inv inside `c2w_to_delta_rt` throws Singular matrix.
        standard_action_downsample = downsample_poses_se3(
            sampled_standard_poses,
            df=self.action_downsampling_factor,
            sampling_mode='interpolation',
        )

        # ---- Step D: apply c2w_to_relative_c2w on the clip -> the clip's first frame becomes identity ----
        # standard_action_downsample is still absolute c2w here (step A used skip_relative=True).
        # Convert to relative over the clip length so clip[0] = (I, 0) and the other frames are
        # relative to it. This matches SimulatedActionDataset (eval accumulates from c2w0=I),
        # keeping training and inference consistent.
        standard_action_downsample = c2w_to_relative_c2w(standard_action_downsample)

        # Only convert standard coordinates to the requested target format (replaces the
        # base class's nested call). delta_euler is a frame-to-frame difference, invariant
        # to the global anchor, so the result is identical.
        target_action_downsample = standard_to_match_coord(
            standard_action_downsample,
            self.target_action_mode
        )

        # ---- Translation scale normalization (single-sided: max-radius only) ----
        # Problem: Minecraft block-coordinate t is on the order of [-5, 5], 3-5x larger than
        # R entries (in [-1,1]); feeding raw t lets the t signal dominate R's gradient in
        # pose_mlp and yaw fails to learn.
        # Fix: apply max-radius scaling only, WITHOUT mean-centering, so the re-anchoring
        # guarantee `t[0] = 0` ("the clip starts at the origin") still holds and t shrinks
        # to ~[-1, 1], on par with R.
        t = standard_action_downsample[:, :3, 3]                            # (N, 3)
        if self.prope_fixed_divisor is None:
            scale_t = float(np.maximum(np.abs(t).max(), 1e-6))              # per-clip max-abs (legacy behavior)
        else:
            scale_t = self.prope_fixed_divisor                              # fixed divisor (1.0 = raw metric units)
        standard_action_downsample[:, :3, 3] = t / scale_t

        target_action_downsample_tensor = torch.from_numpy(target_action_downsample).float()
        standard_action_downsample_tensor = torch.from_numpy(standard_action_downsample).float()
        video_tensor = torch.from_numpy(video_frames).permute(0, 3, 1, 2).contiguous().float() / 255.0
        video_tensor = self._batch_resize(video_tensor)
        video_tensor = self.normalize(video_tensor)

        return {
            'frames': video_tensor.permute(1, 0, 2, 3),
            'prompts': prompt,
            'idx': idx,
            'actions': target_action_downsample_tensor,
            'prope_context': standard_action_downsample_tensor,
        }



class SekaiGameDataset(BaseActionDataset):
    def __init__(
        self,
        data_dir: str,
        translation_scale: float = 1.0,
        palindrome_prob: float = 0.0,
        reverse_prob: float = 0.1,
        **base_kwargs: Any,
    ):
        _allowed_base_keys = {
            "data_dir", "video_size", "num_frames", "target_fps",
            "cache_size", "source_action_mode", "target_action_mode",
            "action_downsampling_factor", "prope_fixed_divisor",
        }
        unknown = set(base_kwargs) - _allowed_base_keys
        if unknown:
            raise TypeError(f"Unknown BaseActionDataset kwargs: {sorted(unknown)}")

        # DIVISOR on the c2w translation (mirrors _VipePosePairDataset). Sekai poses are
        # NOT in UE-metric units (game=engine units, real=MegaSaM normalized), so divide
        # the absolute translation to bring per-frame delta_euler |Δt| onto the UE regime
        # (~0.3667). Default 1.0 = no-op (legacy behaviour). Applied in load_action_from_path.
        self.translation_scale = float(translation_scale)
        super().__init__(data_dir=data_dir, **base_kwargs)
        self.palindrome_prob = palindrome_prob
        self.reverse_prob = reverse_prob
    
    def initialize_files(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.video_paths = []
        self.action_paths = []
        self.prompts = []

        cache_path = self.data_dir / "sekai_manifest.pkl"

        if cache_path.exists():
            print(f"[SekaiGameDataset] found manifest cache, loading: {cache_path}")
            with open(cache_path, 'rb') as f:
                cache_data = pickle.load(f)
            
            self.video_paths = cache_data['video_paths']
            self.action_paths = cache_data['action_paths']
            self.prompts = cache_data['prompts']
            
            print(f"[SekaiGameDataset] cache loaded, {len(self.video_paths)} samples")
        else:
            print("[SekaiGameDataset] no cache found, parsing CSV files...")
            csv_files = list(self.data_dir.glob("*.csv"))
            if not csv_files:
                raise FileNotFoundError(f"No CSV file found in {data_dir}")
            
            self.metadata_path = str(csv_files[0])
            with open(self.metadata_path, 'r', encoding='utf-8') as f:
                metadata = pd.read_csv(f)

            required_cols = ["videoFile", "cameraFile", "caption"]
            for col in required_cols:
                if col not in metadata.columns:
                    raise ValueError(f"CSV is missing required column: '{col}'")

            def _resolve_path(p):
                p = Path(p)
                return p if p.is_absolute() else (self.data_dir / p)

            from tqdm import tqdm
            for _, row in tqdm(metadata.iterrows(), total=len(metadata), desc="Processing Sekai Metadata"):
                v_path = _resolve_path(row["videoFile"])
                a_path = _resolve_path(row["cameraFile"])
                
                if v_path.exists() and a_path.exists():
                    self.video_paths.append(str(v_path))
                    self.action_paths.append(str(a_path))
                    self.prompts.append(str(row["caption"]))

            if len(self.video_paths) == 0:
                raise ValueError("Parsing finished but no valid video/camera pairs were found; check the CSV paths.")

            print(f"[SekaiGameDataset] saving manifest cache to: {cache_path} ...")
            cache_data = {
                'video_paths': self.video_paths,
                'action_paths': self.action_paths,
                'prompts': self.prompts
            }
            with open(cache_path, 'wb') as f:
                pickle.dump(cache_data, f)
            
            print("[SekaiGameDataset] cache written; the next startup will load instantly.")

    def load_action_from_path(self, action_path: str) -> np.ndarray:
        npz_obj = np.load(action_path, allow_pickle=True)
        if isinstance(npz_obj, np.lib.npyio.NpzFile):
            if "extrinsic" in npz_obj.files:
                extrinsic_np = npz_obj["extrinsic"]
            elif "arr_0" in npz_obj.files:
                extrinsic_np = npz_obj["arr_0"]
            else:
                raise ValueError(
                    f"Unsupported npz structure in {action_path}, "
                    f"available keys: {npz_obj.files}"
                )
        else:
            extrinsic_np = npz_obj
        extrinsic_np = _as_float32_ndarray(extrinsic_np, name="extrinsic")
        if self.translation_scale != 1.0:
            extrinsic_np = extrinsic_np.copy()
            extrinsic_np[:, :3, 3] /= self.translation_scale   # c2w translation -> UE metric regime
        return extrinsic_np

    def temporal_augmentation(
        self,
        indices: np.ndarray
    ) -> np.ndarray:
        if self.reverse_prob > 0 and np.random.rand() < self.reverse_prob:
            indices = indices[::-1]

        if self.palindrome_prob > 0 and np.random.rand() < self.palindrome_prob:
            T = indices.shape[0]
            mid = (T + 1) // 2
            use_first_half = np.random.rand() < 0.5
            if use_first_half:
                half = indices[:mid]
            else:
                half = indices[T - mid:]
            reversed_half = half[::-1].copy()
            indices = np.concatenate([half, reversed_half], axis=0)[:T].copy()

        return indices


class MinecraftDataset(BaseActionDataset):
    DEFAULT_PROMPT = (
        "Cinematic Minecraft world aesthetic, blocky voxel-based environment, "
        "low-poly 3D pixel art style, distinct Minecraft terrain generation with "
        "cubic geometry, saturated color palette typical of the Minecraft game engine, "
        "soft sunlight casting blocky shadows, expansive biome landscapes, "
        "authentic Minecraft visual distribution"
    )
    
    def __init__(
        self,
        data_dir: str,
        palindrome_prob: float = 0.0,
        reverse_prob: float = 0.0,
        **base_kwargs: Any,
    ):
        _allowed_base_keys = {
            "data_dir", "video_size", "num_frames", "target_fps",
            "cache_size", "source_action_mode", "target_action_mode",
            "action_downsampling_factor", "prope_fixed_divisor",
        }
        unknown = set(base_kwargs) - _allowed_base_keys
        if unknown:
            raise TypeError(f"Unknown BaseActionDataset kwargs: {sorted(unknown)}")

        super().__init__(data_dir=data_dir, **base_kwargs)
        self.palindrome_prob = palindrome_prob
        self.reverse_prob = reverse_prob

    def initialize_files(self, data_dir):
        self.data_dir = Path(data_dir)

        self.video_paths = []
        for ext in DEFAULT_VIDEO_EXTENSIONS:
            self.video_paths.extend(
                glob.glob(os.path.join(str(self.data_dir), "**", f"*{ext}"), recursive=True)
            )
        self.action_paths = [
            str(Path(video_path).with_suffix(".npz"))
            for video_path in self.video_paths
        ]
        self.prompts = [self.DEFAULT_PROMPT] * len(self.video_paths)

    def load_action_from_path(self, action_path: str) -> np.ndarray:
        poses = np.load(action_path, allow_pickle=True)["poses"]
        poses[:,-1] *= -1
        return _as_float32_ndarray(poses, name="poses")

    def temporal_augmentation(self, indices: np.ndarray) -> np.ndarray:
        if self.reverse_prob > 0 and np.random.rand() < self.reverse_prob:
            indices = indices[::-1].copy()
        if self.palindrome_prob > 0 and np.random.rand() < self.palindrome_prob:
            T = indices.shape[0]
            mid = (T + 1) // 2
            use_first_half = np.random.rand() < 0.5
            half = indices[:mid] if use_first_half else indices[T - mid:]
            indices = np.concatenate([half, half[::-1]], axis=0)[:T].copy()
        return indices


class UEControlDataset(BaseActionDataset):
    """UE (Unreal Engine) fly-through world-model data WITH per-clip captions.

    Layout:
      <data_dir>/index/captions.jsonl   # one json/line: {"clip","video"(abs .mp4 path),"prompt"(EN caption)}
      <...>/seg{0..3}/clip.mp4          # 15s / 450 frames / 30fps
      <...>/seg{0..3}/camera.json       # frames[i]['c2w'] = absolute 4x4 c2w (right_handed_y_up)

    Unlike MinecraftDataset (one hardcoded DEFAULT_PROMPT for every clip), this emits the
    REAL per-clip caption as 'prompts' — the only data-side change needed for caption
    conditioning (the trainer already encodes batch['prompts'] via the umT5 text encoder).
    Poses come from camera.json c2w (no euler synthesis); source_action_mode='ue' applies the
    GL->OpenCV axis flip so the derived delta_euler matches the synthetic eval convention.
    """

    def __init__(
        self,
        data_dir: str,
        index_file: str = "index/captions.jsonl",
        max_clips: Optional[int] = None,
        translation_scale: float = 100.0,
        palindrome_prob: float = 0.0,
        reverse_prob: float = 0.0,
        **base_kwargs: Any,
    ):
        _allowed_base_keys = {
            "data_dir", "video_size", "num_frames", "target_fps",
            "cache_size", "source_action_mode", "target_action_mode",
            "action_downsampling_factor", "prope_fixed_divisor",
        }
        unknown = set(base_kwargs) - _allowed_base_keys
        if unknown:
            raise TypeError(f"Unknown BaseActionDataset kwargs for UEControlDataset: {sorted(unknown)}")

        self.index_file = index_file
        self.max_clips = max_clips
        # UE world poses are in CENTIMETRES (a fly-through forward step is ~10 units/frame
        # @30fps ~= 3 m/s). Raw delta_euler translation would then be ~22/chunk, ~100x the
        # Minecraft/eval scale (~0.2), which would push the action MLP out of distribution at
        # eval. Divide absolute translation by 100 (cm->m) so per-chunk delta_euler |t| ~= 0.2,
        # matching MinecraftDataset and the synthetic NIAH eval. prope_context: with the
        # default per-clip max norm downstream this pre-scale cancels out for PRoPE; with
        # prope_fixed_divisor set (metric recipe) it DOES reach PRoPE — that is the point.
        self.translation_scale = float(translation_scale)
        # default to UE coord convention unless the YAML overrides it
        base_kwargs.setdefault("source_action_mode", "ue")
        base_kwargs.setdefault("target_action_mode", "delta_euler")
        super().__init__(data_dir=data_dir, **base_kwargs)
        self.palindrome_prob = palindrome_prob
        self.reverse_prob = reverse_prob

    def initialize_files(self, data_dir: str):
        self.data_dir = Path(data_dir)
        index_path = self.data_dir / self.index_file
        if not index_path.exists():
            raise FileNotFoundError(f"UE caption index not found: {index_path}")

        self.video_paths = []
        self.action_paths = []
        self.prompts = []

        with open(index_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                video = rec["video"]                       # already an absolute path
                prompt = rec.get("prompt", "")
                if not prompt:
                    continue                               # skip un-captioned clips (rejected/missing)
                cam = str(Path(video).parent / "camera.json")
                self.video_paths.append(video)
                self.action_paths.append(cam)
                self.prompts.append(prompt)
                if self.max_clips is not None and len(self.video_paths) >= self.max_clips:
                    break

        if len(self.video_paths) == 0:
            raise ValueError(f"UEControlDataset: no captioned clips found under {index_path}")
        print(f"[UEControlDataset] loaded {len(self.video_paths)} captioned UE clips from {index_path}")

    def load_action_from_path(self, action_path: str) -> np.ndarray:
        with open(action_path, "r") as f:
            cam = json.load(f)
        # absolute 4x4 c2w per frame (right_handed_y_up); -> (T,4,4)
        poses = np.asarray([fr["c2w"] for fr in cam["frames"]], dtype=np.float32)
        if poses.ndim != 3 or poses.shape[1:] != (4, 4):
            raise ValueError(f"camera.json {action_path} c2w stack has bad shape {poses.shape}")
        poses[:, :3, 3] /= self.translation_scale   # cm -> m, bring delta_euler |t| to the MC/eval scale
        return _as_float32_ndarray(poses, name="ue_c2w")

    def temporal_augmentation(self, indices: np.ndarray) -> np.ndarray:
        if self.reverse_prob > 0 and np.random.rand() < self.reverse_prob:
            indices = indices[::-1].copy()
        if self.palindrome_prob > 0 and np.random.rand() < self.palindrome_prob:
            T = indices.shape[0]
            mid = (T + 1) // 2
            use_first_half = np.random.rand() < 0.5
            half = indices[:mid] if use_first_half else indices[T - mid:]
            indices = np.concatenate([half, half[::-1]], axis=0)[:T].copy()
        return indices


class RealEstateDataset(BaseActionDataset):
    def __init__(
        self,
        data_dir: str,
        index_path: str = "",
        palindrome_prob: float = 0.0,
        reverse_prob: float = 0.1,
        **base_kwargs: Any,
    ):
        _allowed_base_keys = {
            "data_dir", "video_size", "num_frames", "target_fps",
            "cache_size", "source_action_mode", "target_action_mode",
            "action_downsampling_factor", "prope_fixed_divisor",
        }
        unknown = set(base_kwargs) - _allowed_base_keys
        if unknown:
            raise TypeError(f"Unknown BaseActionDataset kwargs for RealEstate: {sorted(unknown)}")

        self.index_path = index_path
        self.palindrome_prob = palindrome_prob
        self.reverse_prob = reverse_prob
        
        super().__init__(data_dir=data_dir, **base_kwargs)

    def initialize_files(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.video_paths, self.action_paths, self.prompts = [], [], []
        cache_path = self.data_dir / "realestate_manifest.pkl"
        
        if cache_path.exists():
            with open(cache_path, 'rb') as f:
                samples = pickle.load(f)
        else:
            samples = []
            self._local_recursive_load(self.index_path, samples)
            with open(cache_path, 'wb') as f:
                pickle.dump(samples, f)

        for entry in samples:
            self.video_paths.append(str(self.data_dir / entry['oss_key']))
            self.action_paths.append(str(self.data_dir / entry['meta']))
            self.prompts.append(entry.get('prompt_en', "A cinematic shot of real estate interior."))

    def _local_recursive_load(self, rel_path: str, samples: list):
        abs_path = self.data_dir / rel_path.split(':')[0]
        if not abs_path.exists(): return
        with open(abs_path, 'r', encoding='utf-8') as f:
            content = json.load(f)
        if isinstance(content, dict) and 'list' in content:
            for item in content['list']: self._local_recursive_load(item.split(':')[0], samples)
        elif isinstance(content, list) and len(content) > 0:
            if isinstance(content[0], str): 
                for item in content: self._local_recursive_load(item.split(':')[0], samples)
            elif isinstance(content[0], dict): 
                for entry in content:
                    if (self.data_dir / entry['oss_key']).exists() and (self.data_dir / entry['meta']).exists():
                        samples.append(entry)

    def load_action_from_path(self, action_path: str) -> np.ndarray:
        with open(action_path, 'rb') as f:
            meta_data = pickle.load(f)
        return _as_float32_ndarray(meta_data['traj'], name="realestate_traj")


class _VipePosePairDataset(BaseActionDataset):
    """Pair videos with VIPE-style pose .npz files (key 'data' = (T,4,4) c2w).

    Layout:
      <data_dir>/<videos_subdir>/<stem>.<ext>   # mp4/mov/avi/mkv/webm
      <data_dir>/<poses_subdir>/<stem>.npz      # np.load(...)["data"] -> (T,4,4)

    Manifest of pairs is cached at <data_dir>/<cache_name>.
    Subclasses are expected to set DEFAULT_VIDEOS_SUBDIR / DEFAULT_POSES_SUBDIR /
    DEFAULT_CACHE_NAME / DEFAULT_PROMPT and pass any per-dataset extras through.
    """

    DEFAULT_VIDEOS_SUBDIR: str = ""
    DEFAULT_POSES_SUBDIR: str = ""
    DEFAULT_CACHE_NAME: str = "pose_pair_manifest.pkl"
    DEFAULT_PROMPT: str = "A high quality real-world video."
    DEFAULT_SOURCE_ACTION_MODE: str = "vipe"
    DEFAULT_TARGET_ACTION_MODE: str = "delta_euler"

    def __init__(
        self,
        data_dir: str,
        videos_subdir: Optional[str] = None,
        poses_subdir: Optional[str] = None,
        cache_name: Optional[str] = None,
        default_prompt: Optional[str] = None,
        prompt_pkl_path: Optional[str] = None,
        palindrome_prob: float = 0.0,
        reverse_prob: float = 0.0,
        translation_scale: float = 1.0,
        repeat: int = 1,
        **base_kwargs: Any,
    ):
        _allowed_base_keys = {
            "data_dir", "video_size", "num_frames", "target_fps",
            "cache_size", "source_action_mode", "target_action_mode",
            "action_downsampling_factor", "prope_fixed_divisor",
        }
        unknown = set(base_kwargs) - _allowed_base_keys
        if unknown:
            raise TypeError(
                f"Unknown BaseActionDataset kwargs for {type(self).__name__}: {sorted(unknown)}"
            )

        base_kwargs.setdefault("source_action_mode", self.DEFAULT_SOURCE_ACTION_MODE)
        base_kwargs.setdefault("target_action_mode", self.DEFAULT_TARGET_ACTION_MODE)

        self.videos_subdir = videos_subdir if videos_subdir is not None else self.DEFAULT_VIDEOS_SUBDIR
        self.poses_subdir = poses_subdir if poses_subdir is not None else self.DEFAULT_POSES_SUBDIR
        self.cache_name = cache_name or self.DEFAULT_CACHE_NAME
        self.default_prompt = default_prompt if default_prompt is not None else self.DEFAULT_PROMPT
        self.prompt_pkl_path = prompt_pkl_path
        self.palindrome_prob = palindrome_prob
        self.reverse_prob = reverse_prob
        # VIPE poses are near-metric but ~4.5x SMALLER than UE's metric regime (after UE's /100).
        # Divisor convention (same as UEControlDataset): set translation_scale=1/4.5≈0.222 to
        # bring vipe |delta_t| up to match UE. 1.0 = no-op (legacy behaviour). Ratio
        # measured empirically (DL3DV ~4.6x, RealEstate ~4.4x, unified 4.5x).
        self.translation_scale = float(translation_scale)
        # Oversample factor: tile this source's sample list `repeat` times to raise
        # its share in the joint ConcatDataset mix (default 1 = no-op). Since
        # __len__ = len(video_paths) and __getitem__ draws a fresh RANDOM window per
        # access, the extra copies are NOT identical frames (each re-samples the clip),
        # so this raises the source's sampling frequency without exact-repeat overfitting.
        self.repeat = max(1, int(repeat))

        super().__init__(data_dir=data_dir, **base_kwargs)

        if self.repeat > 1:
            self.video_paths = list(self.video_paths) * self.repeat
            self.action_paths = list(self.action_paths) * self.repeat
            self.prompts = list(self.prompts) * self.repeat

    def _build_prompt_lookup(self) -> dict:
        """Optional stem→prompt map. Reads `prompt_pkl_path` if set; entries are
        list[dict] each with `oss_key` (stem under videos dir) and `prompt_en`/`prompt`."""
        if not self.prompt_pkl_path:
            return {}
        path = Path(self.prompt_pkl_path)
        if not path.is_absolute():
            path = self.data_dir / path
        if not path.exists():
            return {}
        with open(path, "rb") as f:
            entries = pickle.load(f)
        out: dict = {}
        for entry in entries:
            oss_key = entry.get("oss_key", "")
            stem = Path(oss_key).stem
            prompt = entry.get("prompt_en") or entry.get("prompt") or ""
            if stem and prompt:
                out[stem] = prompt
        return out

    def initialize_files(self, data_dir: str):
        self.data_dir = Path(data_dir)
        videos_dir = self.data_dir / self.videos_subdir
        poses_dir = self.data_dir / self.poses_subdir
        cache_path = self.data_dir / self.cache_name

        self.video_paths = []
        self.action_paths = []
        self.prompts = []

        prompt_lookup = self._build_prompt_lookup()

        if cache_path.exists():
            print(f"[{type(self).__name__}] found manifest cache, loading: {cache_path}")
            with open(cache_path, "rb") as f:
                cache = pickle.load(f)
            video_paths = cache["video_paths"]
            action_paths = cache["action_paths"]
        else:
            if not videos_dir.exists():
                raise FileNotFoundError(f"videos_dir does not exist: {videos_dir}")
            if not poses_dir.exists():
                raise FileNotFoundError(f"poses_dir does not exist: {poses_dir}")

            print(f"[{type(self).__name__}] scanning {videos_dir} and {poses_dir} for pairs...")
            video_paths = []
            action_paths = []
            for ext in DEFAULT_VIDEO_EXTENSIONS:
                for v in videos_dir.glob(f"*{ext}"):
                    a = poses_dir / f"{v.stem}.npz"
                    if a.exists():
                        video_paths.append(str(v))
                        action_paths.append(str(a))
            print(
                f"[{type(self).__name__}] paired {len(video_paths)} samples, writing cache: {cache_path}"
            )
            with open(cache_path, "wb") as f:
                pickle.dump({"video_paths": video_paths, "action_paths": action_paths}, f)

        for v_path, a_path in zip(video_paths, action_paths):
            stem = Path(v_path).stem
            self.video_paths.append(v_path)
            self.action_paths.append(a_path)
            self.prompts.append(prompt_lookup.get(stem, self.default_prompt))

        if len(self.video_paths) == 0:
            raise ValueError(
                f"[{type(self).__name__}] no video/npz pairs found; check "
                f"data_dir={self.data_dir} videos_subdir={self.videos_subdir} "
                f"poses_subdir={self.poses_subdir}"
            )

    def load_action_from_path(self, action_path: str) -> np.ndarray:
        with np.load(action_path, allow_pickle=True) as obj:
            if "data" not in obj.files:
                raise ValueError(
                    f"npz {action_path} is missing the 'data' key (expected (T,4,4) c2w matrices); "
                    f"keys found: {obj.files}"
                )
            poses = obj["data"]
        poses = _as_float32_ndarray(poses, name="vipe_data")
        if self.translation_scale != 1.0:
            poses = poses.copy()
            poses[:, :3, 3] /= self.translation_scale   # vipe near-metric → ÷0.222 = ×4.5 to UE metric regime
        return poses

    def temporal_augmentation(self, indices: np.ndarray) -> np.ndarray:
        if self.reverse_prob > 0 and np.random.rand() < self.reverse_prob:
            indices = indices[::-1].copy()
        if self.palindrome_prob > 0 and np.random.rand() < self.palindrome_prob:
            T = indices.shape[0]
            mid = (T + 1) // 2
            use_first_half = np.random.rand() < 0.5
            half = indices[:mid] if use_first_half else indices[T - mid:]
            indices = np.concatenate([half, half[::-1]], axis=0)[:T].copy()
        return indices


class DL3DVDataset(_VipePosePairDataset):
    """DL3DV dataset (point data_dir at any layout-compatible root).

    Default layout:
        videos/<stem>.mp4
        poses/_root/pose/<stem>.npz   # {'data': (T,4,4) c2w (vipe), 'inds': (T,)}
    """
    DEFAULT_VIDEOS_SUBDIR = "videos"
    DEFAULT_POSES_SUBDIR = "poses/_root/pose"
    DEFAULT_CACHE_NAME = "dl3dv_pose_pair_manifest.pkl"
    DEFAULT_PROMPT = (
        "A cinematic real-world scene captured with a moving camera, "
        "smooth dolly motion through varied indoor and outdoor environments."
    )


class RealEstate10KDataset(_VipePosePairDataset):
    """RealEstate10K dataset (point data_dir at any layout-compatible root).

    Default layout:
        datasets/RealEstate/video_inpaint/<stem>.mp4
        datasets/RealEstate/meta/_root/pose/<stem>.npz   # vipe-format
    """
    DEFAULT_VIDEOS_SUBDIR = "datasets/RealEstate/video_inpaint"
    DEFAULT_POSES_SUBDIR = "datasets/RealEstate/meta/_root/pose"
    DEFAULT_CACHE_NAME = "realestate10k_pose_pair_manifest.pkl"
    DEFAULT_PROMPT = (
        "A real estate interior tour video, slow steadicam camera "
        "moving through residential rooms with natural lighting."
    )


class OmniWorldGameDataset(_VipePosePairDataset):
    """OmniWorld-Game clips in a per-clip-folder layout.

    Per-CLIP-FOLDER layout (NOT the flat videos/poses split of the other vipe sources):
        <data_dir>/<clip_id>/video.mp4       # native fps, contiguous run window
        <data_dir>/<clip_id>/poses.npz       # {'data': (T,4,4) c2w, OpenCV, REAL METERS, data[0]=I}
        <data_dir>/<clip_id>/caption.json    # {'caption': <en>, ...}
        <data_dir>/<clip_id>/meta.json
    Poses already carry real meters (per-scene MetricScale applied upstream); use
    translation_scale to bring |delta_t| onto UE's metric regime (1.0 = no rescale).
    initialize_files is overridden to scan the per-clip folders and read each clip's
    caption from its caption.json (no separate prompt pkl / no flat videos+poses dirs);
    everything else (npz 'data' load + translation_scale, temporal aug) is inherited.
    """
    DEFAULT_CACHE_NAME = "omniworld_game_pose_pair_manifest.pkl"
    DEFAULT_PROMPT = (
        "A third-person video-game scene rendered by a game engine, the camera "
        "following the character through the environment."
    )

    def initialize_files(self, data_dir: str):
        self.data_dir = Path(data_dir)
        cache_path = self.data_dir / self.cache_name
        self.video_paths = []
        self.action_paths = []
        self.prompts = []

        if cache_path.exists():
            print(f"[{type(self).__name__}] found manifest cache, loading: {cache_path}")
            with open(cache_path, "rb") as f:
                cache = pickle.load(f)
            self.video_paths = cache["video_paths"]
            self.action_paths = cache["action_paths"]
            self.prompts = cache["prompts"]
        else:
            print(f"[{type(self).__name__}] scanning per-clip folders under {self.data_dir} ...")
            # video.mp4 and clip.mp4 are both accepted (same per-clip-dir contract)
            vids = sorted(self.data_dir.glob("*/video.mp4")) + sorted(self.data_dir.glob("*/clip.mp4"))
            for vid in vids:
                clip_dir = vid.parent
                npz = clip_dir / "poses.npz"
                if not npz.exists():
                    continue
                prompt = self.default_prompt
                cap_json = clip_dir / "caption.json"
                if cap_json.exists():
                    try:
                        c = json.load(open(cap_json)).get("caption")
                        if c:
                            prompt = c
                    except Exception:
                        pass
                self.video_paths.append(str(vid))
                self.action_paths.append(str(npz))
                self.prompts.append(prompt)
            print(f"[{type(self).__name__}] paired {len(self.video_paths)} samples, writing cache: {cache_path}")
            with open(cache_path, "wb") as f:
                pickle.dump({"video_paths": self.video_paths,
                             "action_paths": self.action_paths,
                             "prompts": self.prompts}, f)

        if len(self.video_paths) == 0:
            raise ValueError(
                f"[{type(self).__name__}] no <clip>/video.mp4 + poses.npz found; "
                f"check data_dir={self.data_dir}"
            )


class SimulatedActionDataset(BaseActionDataset):
    def __init__(self, data_dir: str, delta_r: float = 0.26, delta_t: float = 0.1, **base_kwargs: Any):
        self.delta_r = float(delta_r)
        self.delta_t = float(delta_t)
        super().__init__(data_dir=data_dir, **base_kwargs)
        
    def __len__(self):
        return len(self.prompts)

    def initialize_files(self, data_dir):
        c2w0 = np.eye(4, dtype=np.float32)
        num_latent = (self.num_frames - 1) // 4 + 1

        lines = []
        with open(data_dir, "r") as f:
            for line in f:
                lines.append(line.strip())

        has_custom_keys = any("@" in ln for ln in lines)

        if has_custom_keys:
            self.prompts = []
            self.actions = []
            self.action_strs = []
            for ln in lines:
                if "@" in ln:
                    prompt, raw_keys = ln.split("@", 1)
                else:
                    prompt, raw_keys = ln, ""
                self.prompts.append(prompt.strip())
                mouse_seq, kb_seq = parse_key_sequence(raw_keys, num_latent)
                poses = generate_action_sequence_from_keys(
                    c2w0, kb_seq, mouse_seq,
                    num_frames=num_latent,
                    delta_t=self.delta_t, delta_r=self.delta_r,
                )
                self.actions.append(poses)
                # FULL key (was raw_keys[:32]) so the on-screen green-key overlay
                # (inference_action.py draws from action_str) shows actions for ALL
                # chunks, not just the first 32 (~21s). The model's conditioning was
                # never truncated (built from full raw_keys above); this only fixes the
                # visualization. Filenames are separately capped to [:40] downstream and
                # the eval resolves the true key from prompts_k*.txt by index.
                self.action_strs.append(raw_keys if raw_keys else "static")
        else:
            traj_specs = [
                ("move_forward",  "u", "w"),
                ("move_backward", "u", "s"),
                ("move_left",     "u", "a"),
                ("move_right",    "u", "d"),
                ("look_up",       "i", "q"),
                ("look_down",     "k", "q"),
                ("look_left",     "j", "q"),
                ("look_right",    "l", "q"),
                #("static",        "u", "q"),
            ]
            self.actions = [
                generate_constant_action_sequence(c2w0, mouse, key, num_frames=num_latent, delta_r=self.delta_r, delta_t=self.delta_t)
                for _, mouse, key in traj_specs
            ]
            self.prompts = lines
            self.action_strs = [name for name, _, _ in traj_specs]

    def __getitem__(self, idx):
        action = self.actions[idx]
        target_action = self.convert_source_action_to_target_action(action)
        target_action_tensor = torch.from_numpy(target_action).float()
        standard_poses = match_coord_to_standard(action, self.source_action_mode)
        # standard_poses here are already relative-to-c2w0 (=identity); the first frame is
        # I, so no re-anchoring is needed. Translation normalization IS required
        # (single-sided, matching the training path), otherwise the prope_context t scale
        # differs between training and inference.
        standard_poses = standard_poses.astype(np.float32, copy=True)
        t = standard_poses[:, :3, 3]
        if self.prope_fixed_divisor is None:
            scale_t = float(np.maximum(np.abs(t).max(), 1e-6))
        else:
            scale_t = self.prope_fixed_divisor
        standard_poses[:, :3, 3] = t / scale_t
        return {
            'prompts': self.prompts[idx],
            'idx': idx,
            'actions': target_action_tensor,
            'action_str': self.action_strs[idx],
            'prope_context': torch.from_numpy(standard_poses).float(),
        }


class MultiKeySimulatedActionDataset(SimulatedActionDataset):
    """Eval dataset for SIMULTANEOUS multi-key actions (W+A, W+L, W+A+L, ...).

    Prompt format: ``<text>@<combo1>-<combo2>-...`` — each combo is the keys pressed
    together that block (subset of w/s/a/d + i/j/k/l), one combo per block (block_size
    latent frames). e.g. ``...@wa-wa-wa-wl-wl-wl`` = 3 blocks W+A then 3 blocks W+L.
    Builds COMPOSITE poses (summed deltas) via ``generate_multikey_action_sequence`` —
    everything downstream (target_action / prope_context in __getitem__) is pose-based and
    inherited unchanged. Tests whether the pose manifold generalizes to composite/diagonal
    poses never seen as single keys (proxy for continuous camera control).
    """

    def initialize_files(self, data_dir):
        c2w0 = np.eye(4, dtype=np.float32)
        num_latent = (self.num_frames - 1) // 4 + 1
        lines = [ln.strip() for ln in open(data_dir, "r") if ln.strip()]
        self.prompts = []
        self.actions = []
        self.action_strs = []
        for ln in lines:
            if "@" in ln:
                prompt, raw = ln.split("@", 1)
            else:
                prompt, raw = ln, ""
            combos = [c for c in raw.strip().split("-") if c] if raw.strip() else []
            poses = generate_multikey_action_sequence(
                c2w0, combos, num_frames=num_latent, block_size=4,
                delta_t=self.delta_t, delta_r=self.delta_r,
            )
            self.prompts.append(prompt.strip())
            self.actions.append(poses)
            self.action_strs.append(raw.strip() if raw.strip() else "static")

