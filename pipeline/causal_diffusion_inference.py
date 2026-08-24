from tqdm import tqdm
from typing import List, Optional
import torch

from wan.utils.fm_solvers import FlowDPMSolverMultistepScheduler, get_sampling_sigmas, retrieve_timesteps
from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
from utils.wan_wrapper import WanDiffusionWrapper, WanTextEncoder, WanVAEWrapper, WanActionDiffusionWrapper
from utils.camera import *
import copy
import numpy as np
from einops import rearrange

from utils.visualize import process_video, export_to_video, visualize_trajectory
from utils.wan_default_config import wan_default_config
from demo_utils.constant import GET_ZERO_VAE_CACHE
import os
import math
MAX_DISCRETE_ACTIONS = 6


def _get_model_config_value(model, key, default=None):
    config = getattr(model, "config", None)
    if config is None:
        return default
    if isinstance(config, dict):
        return config.get(key, default)
    return getattr(config, key, default)


def _normalize_cache_local_attn_size(value):
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().tolist()
    is_sequence = isinstance(value, (list, tuple)) or value.__class__.__name__ == "ListConfig"
    if is_sequence:
        finite_sizes = [abs(int(size)) for size in value if int(size) != -1]
        return max(finite_sizes) if finite_sizes else -1
    return abs(int(value)) if int(value) != -1 else -1


def _resolve_cache_local_attn_size(model):
    inference_local_attn_size = _get_model_config_value(model, "inference_local_attn_size", None)
    normalized = _normalize_cache_local_attn_size(inference_local_attn_size)
    if normalized is not None:
        return normalized
    fallback = _normalize_cache_local_attn_size(getattr(model, "local_attn_size", -1))
    return fallback if fallback is not None else -1


class CausalDiffusionInferencePipeline(torch.nn.Module):
    def __init__(
            self,
            args,
            device,
            generator=None,
            text_encoder=None,
            vae=None
    ):
        super().__init__()
        # Step 1: Initialize all models
        self.generator = WanDiffusionWrapper(
            **getattr(args, "model_kwargs", {}), is_causal=True) if generator is None else generator
        self.text_encoder = WanTextEncoder() if text_encoder is None else text_encoder
        self.vae = WanVAEWrapper() if vae is None else vae

        # Step 2: Initialize scheduler
        self.num_train_timesteps = args.num_train_timestep
        self.sampling_steps = 30
        self.sample_solver = 'unipc'
        self.shift = args.timestep_shift

        
        self.frame_seq_length = math.prod(args.image_or_video_shape[-2:]) // 4
        # self.model_name = args.model_kwargs.get("model_name", "Wan2.2-TI2V-5B")
        # model_key = os.path.basename(self.model_name)
        self.model_name = os.path.basename(args.model_kwargs.get("model_name", "Wan2.2-TI2V-5B"))
        self.num_transformer_blocks = wan_default_config[self.model_name]["num_transformer_blocks"]

        self.kv_cache_pos = None
        self.kv_cache_neg = None
        self.crossattn_cache_pos = None
        self.crossattn_cache_neg = None
        self.args = args
        self.num_frame_per_block = getattr(args, "num_frame_per_block", 1)
        self.independent_first_frame = args.independent_first_frame
        # Get max_discrete_actions from model config if available
        self.max_discrete_actions = getattr(
            self.generator.model.config, 'max_discrete_actions', MAX_DISCRETE_ACTIONS
        ) if hasattr(self.generator.model, 'config') else MAX_DISCRETE_ACTIONS
        self.local_attn_size = self.generator.model.local_attn_size
        self.cache_local_attn_size = _resolve_cache_local_attn_size(self.generator.model)

        print(f"KV inference with {self.num_frame_per_block} frames per block")

        if self.num_frame_per_block > 1:
            self.generator.model.num_frame_per_block = self.num_frame_per_block

    def inference(
        self,
        noise: torch.Tensor,
        text_prompts: List[str],
        combined_action_ids: Optional[torch.Tensor] = None,
        initial_latent: Optional[torch.Tensor] = None,
        return_latents: bool = False,
        start_frame_index: Optional[int] = 0,
        num_inference_steps: Optional[int] = 30,
        prope_context: Optional[torch.Tensor] = None,
        control_manager=None,
    ) -> torch.Tensor:
        """
        Perform inference on the given noise and text prompts.
        Inputs:
            noise (torch.Tensor): The input noise tensor of shape
                (batch_size, num_output_frames, num_channels, height, width).
            text_prompts (List[str]): The list of text prompts.
            initial_latent (torch.Tensor): The initial latent tensor of shape
                (batch_size, num_input_frames, num_channels, height, width).
                If num_input_frames is 1, perform image to video.
                If num_input_frames is greater than 1, perform video extension.
            return_latents (bool): Whether to return the latents.
            start_frame_index (int): In long video generation, where does the current window start?
        Outputs:
            video (torch.Tensor): The generated video tensor of shape
                (batch_size, num_frames, num_channels, height, width). It is normalized to be in the range [0, 1].
        """
        batch_size, num_frames, num_channels, height, width = noise.shape
        self.frame_seq_length = height * width // 4
        if not self.independent_first_frame or (self.independent_first_frame and initial_latent is not None):
            # If the first frame is independent and the first frame is provided, then the number of frames in the
            # noise should still be a multiple of num_frame_per_block
            assert num_frames % self.num_frame_per_block == 0
            num_blocks = num_frames // self.num_frame_per_block
        elif self.independent_first_frame and initial_latent is None:
            # Using a [1, 4, 4, 4, 4, 4] model to generate a video without image conditioning
            assert (num_frames - 1) % self.num_frame_per_block == 0
            num_blocks = (num_frames - 1) // self.num_frame_per_block
        num_input_frames = initial_latent.shape[1] if initial_latent is not None else 0
        num_output_frames = num_frames + num_input_frames  # add the initial latent frames
        conditional_dict = self.text_encoder(
            text_prompts=text_prompts
        )
        unconditional_dict = self.text_encoder(
            text_prompts=[self.args.negative_prompt] * len(text_prompts)
        )

        output = torch.zeros(
            [batch_size, num_output_frames, num_channels, height, width],
            device=noise.device,
            dtype=noise.dtype
        )
        # Step 1: Initialize KV cache to all zeros
        if self.kv_cache_pos is None:
            self._initialize_kv_cache(
                batch_size=batch_size,
                dtype=noise.dtype,
                device=noise.device,
                num_frames=num_output_frames,
            )
            self._initialize_crossattn_cache(
                batch_size=batch_size,
                dtype=noise.dtype,
                device=noise.device
            )
        else:
            # reset cross attn cache
            for block_index in range(self.num_transformer_blocks):
                self.crossattn_cache_pos[block_index]["is_init"] = False
                self.crossattn_cache_neg[block_index]["is_init"] = False
            # reset kv cache
            for block_index in range(len(self.kv_cache_pos)):
                self.kv_cache_pos[block_index]["global_end_index"] = torch.tensor(
                    [0], dtype=torch.long, device=noise.device)
                self.kv_cache_pos[block_index]["local_end_index"] = torch.tensor(
                    [0], dtype=torch.long, device=noise.device)
                self.kv_cache_neg[block_index]["global_end_index"] = torch.tensor(
                    [0], dtype=torch.long, device=noise.device)
                self.kv_cache_neg[block_index]["local_end_index"] = torch.tensor(
                    [0], dtype=torch.long, device=noise.device)

        # Resolve prope_context: prefer control_manager if given
        prope_ctx_for_caching = (
            control_manager.full_prope_context if control_manager is not None
            else prope_context
        )

        # Step 2: Cache context feature
        current_start_frame = start_frame_index
        cache_start_frame = 0
        if initial_latent is not None:
            timestep = torch.ones([batch_size, 1], device=noise.device, dtype=torch.int64) * 0
            if self.independent_first_frame:
                # Assume num_input_frames is 1 + self.num_frame_per_block * num_input_blocks
                assert (num_input_frames - 1) % self.num_frame_per_block == 0
                num_input_blocks = (num_input_frames - 1) // self.num_frame_per_block
                output[:, :1] = initial_latent[:, :1]
                self.generator(
                    noisy_image_or_video=initial_latent[:, :1],
                    conditional_dict=conditional_dict,
                    timestep=timestep * 0,
                    kv_cache=self.kv_cache_pos,
                    crossattn_cache=self.crossattn_cache_pos,
                    current_start=current_start_frame * self.frame_seq_length,
                    cache_start=cache_start_frame * self.frame_seq_length,
                    prope_context=prope_ctx_for_caching,
                )
                self.generator(
                    noisy_image_or_video=initial_latent[:, :1],
                    conditional_dict=unconditional_dict,
                    timestep=timestep * 0,
                    kv_cache=self.kv_cache_neg,
                    crossattn_cache=self.crossattn_cache_neg,
                    current_start=current_start_frame * self.frame_seq_length,
                    cache_start=cache_start_frame * self.frame_seq_length,
                    prope_context=prope_ctx_for_caching,
                )
                current_start_frame += 1
                cache_start_frame += 1
            else:
                # Assume num_input_frames is self.num_frame_per_block * num_input_blocks
                assert num_input_frames % self.num_frame_per_block == 0
                num_input_blocks = num_input_frames // self.num_frame_per_block

            for block_index in range(num_input_blocks):
                current_ref_latents = \
                    initial_latent[:, cache_start_frame:cache_start_frame + self.num_frame_per_block]
                output[:, cache_start_frame:cache_start_frame + self.num_frame_per_block] = current_ref_latents
                self.generator(
                    noisy_image_or_video=current_ref_latents,
                    conditional_dict=conditional_dict,
                    timestep=timestep * 0,
                    kv_cache=self.kv_cache_pos,
                    crossattn_cache=self.crossattn_cache_pos,
                    current_start=current_start_frame * self.frame_seq_length,
                    cache_start=cache_start_frame * self.frame_seq_length,
                    prope_context=prope_ctx_for_caching,
                )
                self.generator(
                    noisy_image_or_video=current_ref_latents,
                    conditional_dict=unconditional_dict,
                    timestep=timestep * 0,
                    kv_cache=self.kv_cache_neg,
                    crossattn_cache=self.crossattn_cache_neg,
                    current_start=current_start_frame * self.frame_seq_length,
                    cache_start=cache_start_frame * self.frame_seq_length,
                    prope_context=prope_ctx_for_caching,
                )
                current_start_frame += self.num_frame_per_block
                cache_start_frame += self.num_frame_per_block

        # Tell the manager how many initial frames were cached
        if control_manager is not None:
            control_manager.cache_initial_frames(num_input_frames)

        # Step 3: Temporal denoising loop
        all_num_frames = [self.num_frame_per_block] * num_blocks
        if self.independent_first_frame and initial_latent is None:
            all_num_frames = [1] + all_num_frames
        _is_rank0 = (not torch.distributed.is_initialized()) or torch.distributed.get_rank() == 0
        _total_chunks = len(all_num_frames)
        for _chunk_idx, current_num_frames in enumerate(all_num_frames):
            # --- obtain control signals for this chunk ---
            if control_manager is not None:
                ctrl = control_manager.next(current_num_frames)
                combined_action_ids_input = ctrl["combined_action_ids"]
                prope_ctx = ctrl["prope_context"]
            else:
                if combined_action_ids is not None:
                    combined_action_ids_input = combined_action_ids[:, cache_start_frame - num_input_frames:cache_start_frame + current_num_frames - num_input_frames]
                else:
                    combined_action_ids_input = None
                prope_ctx = prope_context

            noisy_input = noise[
                :, cache_start_frame - num_input_frames:cache_start_frame + current_num_frames - num_input_frames]
            latents = noisy_input

            # Step 3.1: Spatial denoising loop
            sample_scheduler = self._initialize_sample_scheduler(noise, num_inference_steps=num_inference_steps)
            for _, t in enumerate(tqdm(sample_scheduler.timesteps)):
                latent_model_input = latents
                timestep = t * torch.ones(
                    [batch_size, current_num_frames], device=noise.device, dtype=torch.float32
                )

                flow_pred_cond, _ = self.generator(
                    noisy_image_or_video=latent_model_input,
                    conditional_dict=conditional_dict,
                    timestep=timestep,
                    kv_cache=self.kv_cache_pos,
                    crossattn_cache=self.crossattn_cache_pos,
                    current_start=current_start_frame * self.frame_seq_length,
                    cache_start=cache_start_frame * self.frame_seq_length,
                    combined_action_ids=combined_action_ids_input,
                    prope_context=prope_ctx,
                )
                flow_pred_uncond, _ = self.generator(
                    noisy_image_or_video=latent_model_input,
                    conditional_dict=unconditional_dict,
                    timestep=timestep,
                    kv_cache=self.kv_cache_neg,
                    crossattn_cache=self.crossattn_cache_neg,
                    current_start=current_start_frame * self.frame_seq_length,
                    cache_start=cache_start_frame * self.frame_seq_length,
                    combined_action_ids=combined_action_ids_input,
                    prope_context=prope_ctx,
                )

                flow_pred = flow_pred_uncond + self.args.guidance_scale * (
                    flow_pred_cond - flow_pred_uncond)

                temp_x0 = sample_scheduler.step(
                    flow_pred,
                    t,
                    latents,
                    return_dict=False)[0]
                latents = temp_x0

            # Step 3.2: record the model's output
            output[:, cache_start_frame:cache_start_frame + current_num_frames] = latents

            # Step 3.3: rerun with timestep zero to update KV cache using clean context.
            # Enable the inspect-capture flag — the trainer monkey-patch syncs q/k to CPU
            # only inside this window, skipping Step 3.1's wasted ~1800 calls/chunk of
            # syncs (~30x speedup of the monkey-patched part).
            import wan_5b.modules.action_model as _am_for_inspect
            _am_for_inspect._INSPECT_CAPTURE_NOW = True
            try:
                self.generator(
                    noisy_image_or_video=latents,
                    conditional_dict=conditional_dict,
                    timestep=timestep * 0,
                    kv_cache=self.kv_cache_pos,
                    crossattn_cache=self.crossattn_cache_pos,
                    current_start=current_start_frame * self.frame_seq_length,
                    cache_start=cache_start_frame * self.frame_seq_length,
                    prope_context=prope_ctx,
                )
                self.generator(
                    noisy_image_or_video=latents,
                    conditional_dict=unconditional_dict,
                    timestep=timestep * 0,
                    kv_cache=self.kv_cache_neg,
                    crossattn_cache=self.crossattn_cache_neg,
                    current_start=current_start_frame * self.frame_seq_length,
                    cache_start=cache_start_frame * self.frame_seq_length,
                    prope_context=prope_ctx,
                )
            finally:
                _am_for_inspect._INSPECT_CAPTURE_NOW = False

            # Step 3.4: update the start and end frame indices
            current_start_frame += current_num_frames
            cache_start_frame += current_num_frames
            if _is_rank0:
                _tok_end = current_start_frame * self.frame_seq_length
                _le_pos = int(self.kv_cache_pos[0]["local_end_index"].item())
                _le_neg = int(self.kv_cache_neg[0]["local_end_index"].item())
                print(f"[infer] chunk {_chunk_idx+1}/{_total_chunks} frames={current_num_frames} "
                      f"tok_end={_tok_end} (cur_start_frame={current_start_frame}) "
                      f"kv_local_end pos={_le_pos} neg={_le_neg}", flush=True)

        # Step 4: Decode the output
        video = self.vae.decode_to_pixel(output)
        video = (video * 0.5 + 0.5).clamp(0, 1)

        if return_latents:
            return video, output
        else:
            return video

    def _initialize_kv_cache(self, batch_size, dtype, device, num_frames=None):
        """
        Initialize a Per-GPU KV cache for the Wan model.

        When `cache_local_attn_size == -1` (unlimited cache，e.g. global / memory-only
        per-head configs + `inference_local_attn_size: -1`), the rolling-eviction logic
        (the `if cache_local_attn_size != -1 and ...` branch in action_model.py inference)
        never triggers, so the cache must be **allocated once, large enough for the whole
        generation**. The fallback `3 * num_frame_per_block` only fits 3 chunks; longer
        videos run past the end and the `[:, A:B]` slice becomes zero-length and errors.

        Prefer the caller-provided `num_frames` (= the full num_output_frames), then fall back to
        `args.image_or_video_shape[1]`。
        """
        kv_cache_pos = []
        kv_cache_neg = []
        num_heads = wan_default_config[self.model_name]["num_heads"]
        head_dim = wan_default_config[self.model_name]["head_dim"]
        if self.cache_local_attn_size != -1:
            # Use the local attention size to compute the KV cache size
            kv_cache_size = self.cache_local_attn_size * self.frame_seq_length
        else:
            # Unlimited cache: must hold the full generation length
            if num_frames is None:
                num_frames = int(self.args.image_or_video_shape[1])
            kv_cache_size = int(num_frames) * self.frame_seq_length

        for _ in range(self.num_transformer_blocks):
            kv_cache_pos.append({
                "k": torch.zeros([batch_size, kv_cache_size, num_heads, head_dim], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, kv_cache_size, num_heads, head_dim], dtype=dtype, device=device),
                "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
                "local_end_index": torch.tensor([0], dtype=torch.long, device=device)
            })
            kv_cache_neg.append({
                "k": torch.zeros([batch_size, kv_cache_size, num_heads, head_dim], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, kv_cache_size, num_heads, head_dim], dtype=dtype, device=device),
                "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
                "local_end_index": torch.tensor([0], dtype=torch.long, device=device)
            })

        self.kv_cache_pos = kv_cache_pos  # always store the clean cache
        self.kv_cache_neg = kv_cache_neg  # always store the clean cache

    def _initialize_crossattn_cache(self, batch_size, dtype, device):
        """
        Initialize a Per-GPU cross-attention cache for the Wan model.
        """
        crossattn_cache_pos = []
        crossattn_cache_neg = []
        num_heads = wan_default_config[self.model_name]["num_heads"]
        head_dim = wan_default_config[self.model_name]["head_dim"]
        for _ in range(self.num_transformer_blocks):
            crossattn_cache_pos.append({
                "k": torch.zeros([batch_size, 512, num_heads, head_dim], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, 512, num_heads, head_dim], dtype=dtype, device=device),
                "is_init": False
            })
            crossattn_cache_neg.append({
                "k": torch.zeros([batch_size, 512, num_heads, head_dim], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, 512, num_heads, head_dim], dtype=dtype, device=device),
                "is_init": False
            })

        self.crossattn_cache_pos = crossattn_cache_pos  # always store the clean cache
        self.crossattn_cache_neg = crossattn_cache_neg  # always store the clean cache

    def clear_cache(self):
        """
        Explicitly release large KV / cross-attention caches to free GPU memory.
        Safe to call between independent inference calls; caches will be
        re-created on demand by _initialize_kv_cache/_initialize_crossattn_cache.
        """
        self.kv_cache_pos = None
        self.kv_cache_neg = None
        self.crossattn_cache_pos = None
        self.crossattn_cache_neg = None

    def _initialize_sample_scheduler(self, noise, num_inference_steps=30):
        if self.sample_solver == 'unipc':
            sample_scheduler = FlowUniPCMultistepScheduler(
                num_train_timesteps=self.num_train_timesteps,
                shift=1,
                use_dynamic_shifting=False)
            sample_scheduler.set_timesteps(
                num_inference_steps, device=noise.device, shift=self.shift)
            self.timesteps = sample_scheduler.timesteps
        elif self.sample_solver == 'dpm++':
            sample_scheduler = FlowDPMSolverMultistepScheduler(
                num_train_timesteps=self.num_train_timesteps,
                shift=1,
                use_dynamic_shifting=False)
            sampling_sigmas = get_sampling_sigmas(num_inference_steps, self.shift)
            self.timesteps, _ = retrieve_timesteps(
                sample_scheduler,
                device=noise.device,
                sigmas=sampling_sigmas)
        else:
            raise NotImplementedError("Unsupported solver.")
        return sample_scheduler

class InteractiveCausalDiffusionInferencePipeline(CausalDiffusionInferencePipeline):
    def __init__(
            self,
            args,
            device="cuda",
            generator=None,
            text_encoder=None,
            vae_decoder=None,
    ):
        torch.nn.Module.__init__(self)
        # Step 1: Initialize all models
        self.generator = WanActionDiffusionWrapper(
            **getattr(args, "model_kwargs", {}), is_causal=True) if generator is None else generator
        self.text_encoder = WanTextEncoder() if text_encoder is None else text_encoder
        self.vae_decoder = vae_decoder

        # Step 2: Initialize scheduler
        self.num_train_timesteps = args.num_train_timestep
        self.sampling_steps = 30
        self.sample_solver = 'unipc'
        self.shift = args.timestep_shift

        self.num_transformer_blocks = 30
        if hasattr(args, "image_or_video_shape") and args.image_or_video_shape is not None:
            latent_h, latent_w = args.image_or_video_shape[-2], args.image_or_video_shape[-1]
            self.frame_seq_length = int(latent_h) * int(latent_w) // 4
        else:
            # Will be overwritten in `inference()` based on actual noise shape.
            self.frame_seq_length = None

        self.kv_cache_pos = None
        self.kv_cache_neg = None
        self.crossattn_cache_pos = None
        self.crossattn_cache_neg = None
        self.args = args
        self.num_frame_per_block = getattr(args, "num_frame_per_block", 1)
        self.independent_first_frame = False
        self.local_attn_size = self.generator.model.local_attn_size
        self.cache_local_attn_size = _resolve_cache_local_attn_size(self.generator.model)

        print(f"KV inference with {self.num_frame_per_block} frames per block")

        if self.num_frame_per_block > 1:
            self.generator.model.num_frame_per_block = self.num_frame_per_block

    def inference(
        self,
        noise: torch.Tensor,
        text_prompts: List[str],
        initial_latent: Optional[torch.Tensor] = None,
        return_latents: bool = False,
        start_frame_index: Optional[int] = 0,
        output_folder: Optional[str] = None,
        action_mode: str = 'plucker_embedding',
    ) -> torch.Tensor:
        """
        Perform inference on the given noise and text prompts.
        Inputs:
            noise (torch.Tensor): The input noise tensor of shape
                (batch_size, num_output_frames, num_channels, height, width).
            text_prompts (List[str]): The list of text prompts.
            initial_latent (torch.Tensor): The initial latent tensor of shape
                (batch_size, num_input_frames, num_channels, height, width).
                If num_input_frames is 1, perform image to video.
                If num_input_frames is greater than 1, perform video extension.
            return_latents (bool): Whether to return the latents.
            start_frame_index (int): In long video generation, where does the current window start?
        Outputs:
            video (torch.Tensor): The generated video tensor of shape
                (batch_size, num_frames, num_channels, height, width). It is normalized to be in the range [0, 1].
        """
        if output_folder is not None:
            os.makedirs(output_folder, exist_ok=True)
        else:
            output_folder = 'temp_videos'
            os.makedirs(output_folder, exist_ok=True)

        batch_size, num_frames, num_channels, height, width = noise.shape
        if not self.independent_first_frame or (self.independent_first_frame and initial_latent is not None):
            # If the first frame is independent and the first frame is provided, then the number of frames in the
            # noise should still be a multiple of num_frame_per_block
            assert num_frames % self.num_frame_per_block == 0
            num_blocks = num_frames // self.num_frame_per_block
        elif self.independent_first_frame and initial_latent is None:
            # Using a [1, 4, 4, 4, 4, 4] model to generate a video without image conditioning
            assert (num_frames - 1) % self.num_frame_per_block == 0
            num_blocks = (num_frames - 1) // self.num_frame_per_block
        num_input_frames = initial_latent.shape[1] if initial_latent is not None else 0
        num_output_frames = num_frames + num_input_frames  # add the initial latent frames
        conditional_dict = self.text_encoder(
            text_prompts=text_prompts
        )
        unconditional_dict = self.text_encoder(
            text_prompts=[self.args.negative_prompt] * len(text_prompts)
        )

        output = torch.zeros(
            [batch_size, num_output_frames, num_channels, height, width],
            device=noise.device,
            dtype=noise.dtype
        )

        history = np.eye(4, dtype=np.float32)
        history = history[None]
        
        videos =    []
        videos_with_icon = []
        # Wan2.2-TI2V-5B VAE spatial stride 16× (not 8×); see configs/wan_ti2v_5B.py
        vae_cache = copy.deepcopy(GET_ZERO_VAE_CACHE(height * 16, width * 16))
        for j in range(len(vae_cache)):
            vae_cache[j] = None

        # Step 1: Initialize KV cache to all zeros
        if self.kv_cache_pos is None:
            self._initialize_kv_cache(
                batch_size=batch_size,
                dtype=noise.dtype,
                device=noise.device,
                num_frames=num_output_frames,
            )
            self._initialize_crossattn_cache(
                batch_size=batch_size,
                dtype=noise.dtype,
                device=noise.device
            )
        else:
            # reset cross attn cache
            for block_index in range(self.num_transformer_blocks):
                self.crossattn_cache_pos[block_index]["is_init"] = False
                self.crossattn_cache_neg[block_index]["is_init"] = False
            # reset kv cache
            for block_index in range(len(self.kv_cache_pos)):
                self.kv_cache_pos[block_index]["global_end_index"] = torch.tensor(
                    [0], dtype=torch.long, device=noise.device)
                self.kv_cache_pos[block_index]["local_end_index"] = torch.tensor(
                    [0], dtype=torch.long, device=noise.device)
                self.kv_cache_neg[block_index]["global_end_index"] = torch.tensor(
                    [0], dtype=torch.long, device=noise.device)
                self.kv_cache_neg[block_index]["local_end_index"] = torch.tensor(
                    [0], dtype=torch.long, device=noise.device)

        # Step 2: Cache context feature
        current_start_frame = start_frame_index
        cache_start_frame = 0
        if initial_latent is not None:
            timestep = torch.ones([batch_size, 1], device=noise.device, dtype=torch.int64) * 0
            if self.independent_first_frame:
                # Assume num_input_frames is 1 + self.num_frame_per_block * num_input_blocks
                assert (num_input_frames - 1) % self.num_frame_per_block == 0
                num_input_blocks = (num_input_frames - 1) // self.num_frame_per_block
                output[:, :1] = initial_latent[:, :1]
                self.generator(
                    noisy_image_or_video=initial_latent[:, :1],
                    conditional_dict=conditional_dict,
                    timestep=timestep * 0,
                    kv_cache=self.kv_cache_pos,
                    crossattn_cache=self.crossattn_cache_pos,
                    current_start=current_start_frame * self.frame_seq_length,
                    cache_start=cache_start_frame * self.frame_seq_length
                )
                self.generator(
                    noisy_image_or_video=initial_latent[:, :1],
                    conditional_dict=unconditional_dict,
                    timestep=timestep * 0,
                    kv_cache=self.kv_cache_neg,
                    crossattn_cache=self.crossattn_cache_neg,
                    current_start=current_start_frame * self.frame_seq_length,
                    cache_start=cache_start_frame * self.frame_seq_length
                )
                current_start_frame += 1
                cache_start_frame += 1
            else:
                # Assume num_input_frames is self.num_frame_per_block * num_input_blocks
                assert num_input_frames % self.num_frame_per_block == 0
                num_input_blocks = num_input_frames // self.num_frame_per_block

            for block_index in range(num_input_blocks):
                current_ref_latents = \
                    initial_latent[:, cache_start_frame:cache_start_frame + self.num_frame_per_block]
                output[:, cache_start_frame:cache_start_frame + self.num_frame_per_block] = current_ref_latents
                self.generator(
                    noisy_image_or_video=current_ref_latents,
                    conditional_dict=conditional_dict,
                    timestep=timestep * 0,
                    kv_cache=self.kv_cache_pos,
                    crossattn_cache=self.crossattn_cache_pos,
                    current_start=current_start_frame * self.frame_seq_length,
                    cache_start=cache_start_frame * self.frame_seq_length
                )
                self.generator(
                    noisy_image_or_video=current_ref_latents,
                    conditional_dict=unconditional_dict,
                    timestep=timestep * 0,
                    kv_cache=self.kv_cache_neg,
                    crossattn_cache=self.crossattn_cache_neg,
                    current_start=current_start_frame * self.frame_seq_length,
                    cache_start=cache_start_frame * self.frame_seq_length
                )
                current_start_frame += self.num_frame_per_block
                cache_start_frame += self.num_frame_per_block

        # Step 3: Temporal denoising loop
        all_num_frames = [self.num_frame_per_block] * num_blocks
        if self.independent_first_frame and initial_latent is None:
            all_num_frames = [1] + all_num_frames
        for current_num_frames in all_num_frames:
            noisy_input = noise[
                :, cache_start_frame - num_input_frames:cache_start_frame + current_num_frames - num_input_frames]
            latents = noisy_input

            # current_actions = get_current_action(
            #     max_actions=self.max_discrete_actions,
            #     return_mode='plucker_embedding'
            # ).repeat(batch_size, current_num_frames, 1).to(noisy_input.device, dtype=noisy_input.dtype)

            new_c2w_camera, mouse_actions, keyboard_actions = get_new_camera_from_keyboard(history[-1], chunk_size=self.num_frame_per_block)
            history = np.concatenate([history, new_c2w_camera], axis=0)

            if action_mode == 'plucker_embedding':
                current_actions = torch.from_numpy(
                    c2w_to_plucker_embedding(new_c2w_camera, H=height//2, W=width//2)
                ).to(noisy_input.device, dtype=noisy_input.dtype).unsqueeze(0)
            elif action_mode == 'c2w':
                current_actions = torch.from_numpy(
                    new_c2w_camera
                ).to(noisy_input.device, dtype=noisy_input.dtype).unsqueeze(0)
            else:
                raise ValueError(f"Invalid action mode: {action_mode}")

            # Step 3.1: Spatial denoising loop
            sample_scheduler = self._initialize_sample_scheduler(noise)
            for _, t in enumerate(tqdm(sample_scheduler.timesteps)):
                latent_model_input = latents
                timestep = t * torch.ones(
                    [batch_size, current_num_frames], device=noise.device, dtype=torch.float32
                )

                flow_pred_cond, _ = self.generator(
                    noisy_image_or_video=latent_model_input,
                    conditional_dict=conditional_dict,
                    timestep=timestep,
                    kv_cache=self.kv_cache_pos,
                    crossattn_cache=self.crossattn_cache_pos,
                    current_start=current_start_frame * self.frame_seq_length,
                    cache_start=cache_start_frame * self.frame_seq_length,
                    combined_action_ids=current_actions
                )
                flow_pred_uncond, _ = self.generator(
                    noisy_image_or_video=latent_model_input,
                    conditional_dict=unconditional_dict,
                    timestep=timestep,
                    kv_cache=self.kv_cache_neg,
                    crossattn_cache=self.crossattn_cache_neg,
                    current_start=current_start_frame * self.frame_seq_length,
                    cache_start=cache_start_frame * self.frame_seq_length,
                    combined_action_ids=current_actions
                )

                flow_pred = flow_pred_uncond + self.args.guidance_scale * (
                    flow_pred_cond - flow_pred_uncond)

                temp_x0 = sample_scheduler.step(
                    flow_pred,
                    t,
                    latents,
                    return_dict=False)[0]
                latents = temp_x0

            # Step 3.2: record the model's output
            output[:, cache_start_frame:cache_start_frame + current_num_frames] = latents

            # Step 3.3: rerun with timestep zero to update KV cache using clean context
            # TODO: consider whether combined_action_ids_input should be passed here
            self.generator(
                noisy_image_or_video=latents,
                conditional_dict=conditional_dict,
                timestep=timestep * 0,
                kv_cache=self.kv_cache_pos,
                crossattn_cache=self.crossattn_cache_pos,
                current_start=current_start_frame * self.frame_seq_length,
                cache_start=cache_start_frame * self.frame_seq_length
            )
            self.generator(
                noisy_image_or_video=latents,
                conditional_dict=unconditional_dict,
                timestep=timestep * 0,
                kv_cache=self.kv_cache_neg,
                crossattn_cache=self.crossattn_cache_neg,
                current_start=current_start_frame * self.frame_seq_length,
                cache_start=cache_start_frame * self.frame_seq_length
            )

            # Step 3.4: update the start and end frame indices
            # latents = latents.transpose(1,2)
            # [batch_size, num_channels, num_frames, height, width]
            video, vae_cache = self.vae_decoder(latents.half(), *vae_cache)
            video = rearrange(video, "B T C H W -> B T H W C")
            video = ((video.float() + 1) * 127.5).clip(0, 255).cpu().numpy().astype(np.uint8)[0]
            video = np.ascontiguousarray(video)
            videos += [video]
            # mouse_icon = 'images/mouse.png'

            # convert actions to key_data and mouse_data
            if cache_start_frame == 0:
                # repeat the current_actions for the first frame
                mouse_actions_per_frame = mouse_actions * 3
                keyboard_actions_per_frame = keyboard_actions * 3
            else:
                mouse_actions_per_frame = mouse_actions * 4
                keyboard_actions_per_frame = keyboard_actions * 4
            

            out_video = process_video(
                video.copy(),
                mouse_actions_per_frame,
                keyboard_actions_per_frame
            )
            videos_with_icon += [out_video]

            current_start_frame += current_num_frames
            cache_start_frame += current_num_frames

            # save current video and video with icon
            export_to_video(out_video / 255.0, output_folder+f'/current_with_icon.mp4', fps=16)
            export_to_video(video / 255.0, output_folder+f'/current.mp4', fps=16)

            if input("Continue? (Press `n` to break)").strip() == "n":
                break

        # Step 4: Decode the output
        all_frames_with_icon = np.concatenate(videos_with_icon, axis=0)
        all_frames = np.concatenate(videos, axis=0)
        export_to_video(all_frames_with_icon / 255.0, output_folder+f'/all_with_icon.mp4', fps=16)
        export_to_video(all_frames / 255.0, output_folder+f'/all.mp4', fps=16)
        # save history camera pose
        visualize_trajectory(np.asarray(history), output_folder+f'/history.html')


        if return_latents:
            return video, output
        else:
            return video
