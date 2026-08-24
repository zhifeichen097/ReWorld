from typing import Optional, Dict, Any, List
import numpy as np
import torch

from utils.camera import (
    get_new_camera_from_keyboard,
    c2w_to_relative_c2w,
)
from utils.default_coord import default_coord
from utils.match_coord import standard_to_match_coord


class ControlSignalManager:
    """
    Unified manager for streaming causal video generation control signals.

    Maintains a global action history (absolute c2w poses) and produces
    per-chunk ``combined_action_ids`` and ``prope_context`` for each
    denoising block.

    Two modes of operation:

    * **Precomputed** -- call :meth:`set_sequence` with the full
      ``combined_action_ids`` (and optionally ``prope_context``) from an
      eval dataset.  :meth:`next` simply advances a cursor and slices.
    * **Interactive** -- :meth:`next` blocks on keyboard input via
      ``get_new_camera_from_keyboard``, converts the resulting c2w poses
      to the target action format, and builds ``prope_context`` from the
      accumulated history.
    """

    def __init__(
        self,
        action_mode: str = "delta_euler",
        chunk_size: int = 4,
        device: str = "cuda",
        dtype: torch.dtype = torch.float32,
        initial_c2w: Optional[np.ndarray] = None,
        delta_t: float = 0.1,
        delta_r: float = 0.26,
        plucker_H: Optional[int] = None,
        plucker_W: Optional[int] = None,
        disable_action_ids: bool = False,
        prope_fixed_divisor: Optional[float] = None,
    ):
        self.action_mode = action_mode
        self.chunk_size = chunk_size
        self.device = device
        self.dtype = dtype
        self.delta_t = delta_t
        self.delta_r = delta_r
        self.plucker_H = plucker_H
        self.plucker_W = plucker_W
        # PRoPE translation divisor for the interactive path (same semantics as the
        # training loader's prope_fixed_divisor): None = no division (legacy behavior,
        # metric pass-through); a number = fixed divisor. Use the same value as training.
        self.prope_fixed_divisor = None if prope_fixed_divisor is None else float(prope_fixed_divisor)
        # Drop the action embedding at inference: next() returns None for
        # combined_action_ids in the dict, consistent with the training-time
        # `action_drop_prob` path (the model block's `if combined_action_ids is not None:`
        # guard simply skips the action_embedding).
        self.disable_action_ids = bool(disable_action_ids)

        self._initial_c2w = (
            initial_c2w.astype(np.float32).copy()
            if initial_c2w is not None
            else np.eye(4, dtype=np.float32)
        )
        self._format_dict = default_coord[action_mode]

        self._precomputed_action_ids: Optional[torch.Tensor] = None
        self._precomputed_prope: Optional[torch.Tensor] = None

        self.reset()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset all state for a new generation session."""
        self._c2w_history: List[np.ndarray] = [self._initial_c2w.copy()]
        self._generated_frame_idx: int = 0
        self._initial_frame_count: int = 0
        self._precomputed_action_ids = None
        self._precomputed_prope = None
        self._mouse_action_history: List[str] = []
        self._keyboard_action_history: List[str] = []

    def set_sequence(
        self,
        combined_action_ids: torch.Tensor,
        prope_context: Optional[torch.Tensor] = None,
    ) -> None:
        """Store a full precomputed control sequence (eval / offline mode).

        Args:
            combined_action_ids: ``[B, T, ...]`` -- the complete action
                tensor covering all *generated* frames.
            prope_context: ``[B, T_total, ...]`` -- optional full prope
                tensor covering *all* frames (initial + generated).
        """
        self._precomputed_action_ids = combined_action_ids
        self._precomputed_prope = prope_context
        self._generated_frame_idx = 0

    def cache_initial_frames(self, num_initial_frames: int) -> None:
        """Record that *num_initial_frames* conditioning frames were cached
        into the KV cache before generation begins.  This shifts the
        ``prope_context`` window accordingly.
        """
        self._initial_frame_count = num_initial_frames

    def next(self, num_frames: Optional[int] = None) -> Dict[str, Any]:
        """Get control signals for the next chunk of generated frames.

        Args:
            num_frames: number of frames in this chunk.  Defaults to
                ``self.chunk_size``.

        Returns:
            dict with keys:
                * ``combined_action_ids`` -- ``Tensor [B, num_frames, ...]``
                  (current chunk only).
                * ``prope_context`` -- ``Tensor [B, total_seen_frames, ...]``
                  (full history up to current; *None* if unavailable).
                * ``mouse_actions`` -- ``list[str]`` (interactive only,
                  else *None*).
                * ``keyboard_actions`` -- ``list[str]`` (interactive only,
                  else *None*).
        """
        num_frames = num_frames or self.chunk_size
        if self.is_precomputed:
            return self._next_precomputed(num_frames)
        return self._next_interactive(num_frames)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_precomputed(self) -> bool:
        return self._precomputed_action_ids is not None

    @property
    def total_generated_frames(self) -> int:
        return self._generated_frame_idx

    @property
    def c2w_history(self) -> np.ndarray:
        """Full ``(N, 4, 4)`` absolute c2w trajectory (for visualization)."""
        return np.stack(self._c2w_history, axis=0)

    @property
    def full_prope_context(self) -> Optional[torch.Tensor]:
        """Full precomputed ``prope_context`` tensor (or *None*)."""
        return self._precomputed_prope

    @property
    def mouse_action_history(self) -> List[str]:
        """Per-latent-frame mouse action keys from interactive mode."""
        return self._mouse_action_history

    @property
    def keyboard_action_history(self) -> List[str]:
        """Per-latent-frame keyboard action keys from interactive mode."""
        return self._keyboard_action_history

    # ------------------------------------------------------------------
    # Precomputed path
    # ------------------------------------------------------------------

    def _next_precomputed(self, num_frames: int) -> Dict[str, Any]:
        start = self._generated_frame_idx
        end = start + num_frames

        action_ids = self._precomputed_action_ids[:, start:end]

        prope = None
        if self._precomputed_prope is not None:
            total_seen = self._initial_frame_count + end
            prope = self._precomputed_prope[:, :total_seen]

        self._generated_frame_idx = end

        return {
            "combined_action_ids": None if self.disable_action_ids else action_ids,
            "prope_context": prope,
            "mouse_actions": None,
            "keyboard_actions": None,
        }

    # ------------------------------------------------------------------
    # Interactive path
    # ------------------------------------------------------------------

    def _next_interactive(self, num_frames: int) -> Dict[str, Any]:
        last_c2w = self._c2w_history[-1]

        new_c2w, mouse_actions, keyboard_actions = get_new_camera_from_keyboard(
            last_c2w,
            delta_t=self.delta_t,
            delta_r=self.delta_r,
            chunk_size=num_frames,
        )
        # new_c2w: (num_frames, 4, 4) absolute c2w
        for i in range(new_c2w.shape[0]):
            self._c2w_history.append(new_c2w[i].copy())

        self._mouse_action_history.extend(mouse_actions)
        self._keyboard_action_history.extend(keyboard_actions)

        action_ids = self._c2w_to_action_ids(new_c2w)

        block_idx = self._generated_frame_idx // num_frames
        print(f"[ControlMgr] Block {block_idx}: mouse={mouse_actions[0]}, kb={keyboard_actions[0]}")
        print(f"  action_ids shape={action_ids.shape}, "
              f"values per frame: {[action_ids[i].tolist() for i in range(min(2, len(action_ids)))]}")
        print(f"  c2w_history len={len(self._c2w_history)}, "
              f"last c2w t={self._c2w_history[-1][:3, 3].tolist()}")

        action_ids_tensor = (
            torch.from_numpy(action_ids)
            .to(device=self.device, dtype=self.dtype)
            .unsqueeze(0)  # add batch dim → [1, num_frames, ...]
        )

        prope = self._build_prope_context()

        self._generated_frame_idx += num_frames

        return {
            "combined_action_ids": action_ids_tensor,
            "prope_context": prope,
            "mouse_actions": mouse_actions,
            "keyboard_actions": keyboard_actions,
        }

    # ------------------------------------------------------------------
    # Conversion helpers
    # ------------------------------------------------------------------

    def _c2w_to_action_ids(self, chunk_c2w: np.ndarray) -> np.ndarray:
        """Convert an absolute c2w chunk to the configured action format.

        For ``delta`` modes the previous frame is prepended so that the
        inter-frame delta is well-defined for the first frame of the chunk.
        For other modes each frame is converted independently (relative to
        the very first frame of the whole history).
        """
        is_delta = self._format_dict["mode"] == "delta"

        if is_delta:
            prev = self._c2w_history[-(len(chunk_c2w) + 1)]
            seq = np.concatenate([prev[None], chunk_c2w], axis=0)
            rel = c2w_to_relative_c2w(seq)
            full = standard_to_match_coord(rel, self._format_dict).astype(np.float32)
            # c2w_to_delta_rt prepends an identity frame; strip it so output
            # length matches the chunk (num_frames actions for num_frames frames).
            return full[1:]

        # Non-delta: compute relative to the very first frame
        full_abs = np.stack(
            self._c2w_history[-len(chunk_c2w):], axis=0
        )
        first_frame = self._c2w_history[0][None]
        seq = np.concatenate([first_frame, full_abs], axis=0)
        rel = c2w_to_relative_c2w(seq)
        # Drop the reference (identity) first frame
        rel_chunk = rel[1:]
        return standard_to_match_coord(rel_chunk, self._format_dict).astype(np.float32)

    def _build_prope_context(self) -> torch.Tensor:
        """Build ``prope_context`` from the full c2w history.

        Returns the relative c2w sequence as ``[1, T, 4, 4]``.  The
        actual positional encoding is handled downstream by ``prope_apply``
        which auto-windows to match the local attention size.
        """
        all_c2w = np.stack(self._c2w_history, axis=0)  # (T, 4, 4)
        rel = c2w_to_relative_c2w(all_c2w)  # (T, 4, 4)
        if self.prope_fixed_divisor is not None:
            rel = rel.astype(np.float32, copy=True)
            rel[:, :3, 3] /= self.prope_fixed_divisor
        return (
            torch.from_numpy(rel)
            .to(device=self.device, dtype=self.dtype)
            .unsqueeze(0)  # [1, T, 4, 4]
        )
