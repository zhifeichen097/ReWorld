import types
from typing import List, Optional
import os
import torch
from torch import nn

from utils.scheduler import SchedulerInterface, FlowMatchScheduler

from wan_5b.modules.tokenizers import HuggingfaceTokenizer
from wan_5b.modules.model import WanModel
from wan_5b.modules.vae2_2 import _video_vae
from wan_5b.modules.t5 import umt5_xxl
from wan_5b.modules.causal_model import CausalWanModel
from wan_5b.modules.action_model import CausalWanActionModel

from transformers import AutoConfig
import gc

try:
    from safetensors import safe_open
    SAFETENSORS_AVAILABLE = True
except ImportError:
    SAFETENSORS_AVAILABLE = False

def custom_init(module):
    # Collect all Linear layers.
    linears = [m for m in module.modules() if isinstance(m, torch.nn.Linear)]
    for i, l in enumerate(linears):
        if i == len(linears) - 1:
            torch.nn.init.zeros_(l.weight)
            if l.bias is not None: torch.nn.init.zeros_(l.bias)
        else:
            torch.nn.init.xavier_uniform_(l.weight)
            if l.bias is not None: torch.nn.init.zeros_(l.bias)

    # If LayerNorm is present, make sure its initial state preserves the zero output.
    for m in module.modules():
        if isinstance(m, torch.nn.LayerNorm):
            torch.nn.init.ones_(m.weight)
            torch.nn.init.zeros_(m.bias)

# NOTE: This is not working for Multi-layer MLP.
def init_all_zeros(module):
    """
    Initialize the weights and biases of every Linear layer in the module to 0.
    """
    if isinstance(module, torch.nn.Linear):
        torch.nn.init.zeros_(module.weight)
        if module.bias is not None:
            torch.nn.init.zeros_(module.bias)

    # Optionally keep LayerNorm from introducing an offset at init.
    elif isinstance(module, torch.nn.LayerNorm):
        torch.nn.init.ones_(module.weight)  # LN weight is conventionally initialized to 1
        torch.nn.init.zeros_(module.bias)

class WanTextEncoder(torch.nn.Module):
    def __init__(self, model_dir: Optional[str] = None) -> None:
        super().__init__()
        # Directory of the Wan2.2-TI2V-5B base model (official Wan release).
        # Set via the `model_dir` argument or the WAN_MODEL_DIR env var.
        model_dir = model_dir or os.environ.get("WAN_MODEL_DIR", "./checkpoints/Wan2.2-TI2V-5B")

        self.text_encoder = umt5_xxl(
            encoder_only=True,
            return_tokenizer=False,
            dtype=torch.float32,
            device=torch.device('cpu')
        ).eval().requires_grad_(False)
        self.text_encoder.load_state_dict(
            torch.load(os.path.join(model_dir, "models_t5_umt5-xxl-enc-bf16.pth"),
                       map_location='cpu', weights_only=False)
        )

        # Move text encoder to GPU if available
        if torch.cuda.is_available():
            self.text_encoder = self.text_encoder.cuda()

        self.tokenizer = HuggingfaceTokenizer(
            name=os.path.join(model_dir, "google/umt5-xxl/"), seq_len=512, clean='whitespace')

    @property
    def device(self):
        # Assume we are always on GPU
        return torch.cuda.current_device()

    def forward(self, text_prompts: List[str]) -> dict:
        ids, mask = self.tokenizer(
            text_prompts, return_mask=True, add_special_tokens=True)
        ids = ids.to(self.device)
        mask = mask.to(self.device)
        seq_lens = mask.gt(0).sum(dim=1).long()
        context = self.text_encoder(ids, mask)
        # ids = ids.to(torch.device('cpu'))
        # mask = mask.to(torch.device('cpu'))
        for u, v in zip(context, seq_lens):
            u[v:] = 0.0  # set padding to 0.0

        return {
            "prompt_embeds": context
        }


class WanVAEWrapper(torch.nn.Module):
    def __init__(self, model_dir: Optional[str] = None):
        super().__init__()
        # Directory of the Wan2.2-TI2V-5B base model (official Wan release).
        # Set via the `model_dir` argument or the WAN_MODEL_DIR env var.
        model_dir = model_dir or os.environ.get("WAN_MODEL_DIR", "./checkpoints/Wan2.2-TI2V-5B")
        mean = [
                -0.2289,
                -0.0052,
                -0.1323,
                -0.2339,
                -0.2799,
                0.0174,
                0.1838,
                0.1557,
                -0.1382,
                0.0542,
                0.2813,
                0.0891,
                0.1570,
                -0.0098,
                0.0375,
                -0.1825,
                -0.2246,
                -0.1207,
                -0.0698,
                0.5109,
                0.2665,
                -0.2108,
                -0.2158,
                0.2502,
                -0.2055,
                -0.0322,
                0.1109,
                0.1567,
                -0.0729,
                0.0899,
                -0.2799,
                -0.1230,
                -0.0313,
                -0.1649,
                0.0117,
                0.0723,
                -0.2839,
                -0.2083,
                -0.0520,
                0.3748,
                0.0152,
                0.1957,
                0.1433,
                -0.2944,
                0.3573,
                -0.0548,
                -0.1681,
                -0.0667,
            ]
        std = [
                0.4765,
                1.0364,
                0.4514,
                1.1677,
                0.5313,
                0.4990,
                0.4818,
                0.5013,
                0.8158,
                1.0344,
                0.5894,
                1.0901,
                0.6885,
                0.6165,
                0.8454,
                0.4978,
                0.5759,
                0.3523,
                0.7135,
                0.6804,
                0.5833,
                1.4146,
                0.8986,
                0.5659,
                0.7069,
                0.5338,
                0.4889,
                0.4917,
                0.4069,
                0.4999,
                0.6866,
                0.4093,
                0.5709,
                0.6065,
                0.6415,
                0.4944,
                0.5726,
                1.2042,
                0.5458,
                1.6887,
                0.3971,
                1.0600,
                0.3943,
                0.5537,
                0.5444,
                0.4089,
                0.7468,
                0.7744,
            ]
        self.mean = torch.tensor(mean, dtype=torch.float32)
        self.std = torch.tensor(std, dtype=torch.float32)

        # init model
        self.model = _video_vae(
            pretrained_path=os.path.join(model_dir, "Wan2.2_VAE.pth"),
        ).eval().requires_grad_(False)

    def encode_to_latent(self, pixel: torch.Tensor) -> torch.Tensor:
        # pixel: [batch_size, num_channels, num_frames, height, width]
        device, dtype = pixel.device, pixel.dtype

        scale = [self.mean.to(device=device, dtype=dtype),
                 1.0 / self.std.to(device=device, dtype=dtype)]

        output = [
            self.model.encode(u.unsqueeze(0), scale).float().squeeze(0)
            for u in pixel
        ]
        output = torch.stack(output, dim=0)
        # from [batch_size, num_channels, num_frames, height, width]
        # to [batch_size, num_frames, num_channels, height, width]
        output = output.permute(0, 2, 1, 3, 4)
        return output

    def decode_to_pixel(self, latent: torch.Tensor, use_cache: bool = False) -> torch.Tensor:
        # from [batch_size, num_frames, num_channels, height, width]
        # to [batch_size, num_channels, num_frames, height, width]
        zs = latent.permute(0, 2, 1, 3, 4)
        if use_cache:
            assert latent.shape[0] == 1, "Batch size must be 1 when using cache"

        device, dtype = latent.device, latent.dtype
        scale = [self.mean.to(device=device, dtype=dtype),
                 1.0 / self.std.to(device=device, dtype=dtype)]

        if use_cache:
            decode_function = self.model.cached_decode
        else:
            decode_function = self.model.decode

        output = []
        for u in zs:
            output.append(decode_function(u.unsqueeze(0), scale).float().clamp_(-1, 1).squeeze(0))
        output = torch.stack(output, dim=0)
        # from [batch_size, num_channels, num_frames, height, width]
        # to [batch_size, num_frames, num_channels, height, width]
        output = output.permute(0, 2, 1, 3, 4)
        return output

    def decode_to_pixel_chunk(self, latent: torch.Tensor, use_cache: bool = False, chunk_size: int = 1) -> torch.Tensor:
        """
        Decode latent frames to pixel space.
        
        Args:
            latent: Latent tensor with shape [batch_size, num_frames, num_channels, height, width]
            use_cache: Whether to use cached decoding (for streaming)
            chunk_size: Number of latent frames to decode at once (default 240 to avoid OOM)
        
        Returns:
            Decoded video tensor with shape [batch_size, num_frames, num_channels, height, width]
        """
        # latent shape: [batch_size, num_frames, num_channels, height, width]
        # zs shape after permute: [batch_size, num_channels, num_frames, height, width]
        zs = latent.permute(0, 2, 1, 3, 4)
        if use_cache:
            assert latent.shape[0] == 1, "Batch size must be 1 when using cache"

        device, dtype = latent.device, latent.dtype
        scale = [self.mean.to(device=device, dtype=dtype),
                 1.0 / self.std.to(device=device, dtype=dtype)]

        if use_cache:
            decode_function = self.model.cached_decode
        else:
            decode_function = self.model.decode

        output = []
        for u in zs:
            num_frames = u.shape[1]
            if num_frames <= chunk_size:
                # Few frames: decode in one shot.
                if use_cache:
                    # Start this segment from a clean cache.
                    self.model.clear_cache()
                decoded = decode_function(u.unsqueeze(0), scale).float().clamp_(-1, 1).squeeze(0)
                decoded = decoded.cpu()
                if use_cache:
                    # Clear after the segment so it does not affect the next video.
                    self.model.clear_cache()
            else:
                # Many frames: decode in temporal chunks.
                decoded_chunks = []
                if use_cache:
                    # Clear the cache once before the segment; chunks share the internal cache.
                    self.model.clear_cache()
                for start_idx in range(0, num_frames, chunk_size):
                    end_idx = min(start_idx + chunk_size, num_frames)
                    chunk = u[:, start_idx:end_idx, :, :]  # [C, chunk_frames, H, W]
                    decoded_chunk = decode_function(chunk.unsqueeze(0), scale).float().clamp_(-1, 1).squeeze(0)
                    decoded_chunks.append(decoded_chunk.cpu())

                    del decoded_chunk
                    torch.cuda.empty_cache()
                decoded = torch.cat(decoded_chunks, dim=1)
                if use_cache:
                    # Clear the cache after the whole segment.
                    self.model.clear_cache()
            output.append(decoded)

        output = torch.stack(output, dim=0)
        output = output.permute(0, 2, 1, 3, 4)
        return output


class WanDiffusionWrapper(torch.nn.Module):
    def __init__(
            self,
            model_name="Wan2.2-TI2V-5B",
            timestep_shift=8.0,
            is_causal=False,
            local_attn_size=-1,
            sink_size=0
    ):
        super().__init__()

        if is_causal:
            self.model = CausalWanModel.from_pretrained(
                f"wan_models/{model_name}/", local_attn_size=local_attn_size, sink_size=sink_size)
        else:
            self.model = WanModel.from_pretrained(f"wan_models/{model_name}/")
        self.model.eval()

        # For non-causal diffusion, all frames share the same timestep
        self.uniform_timestep = not is_causal

        self.scheduler = FlowMatchScheduler(
            shift=timestep_shift, sigma_min=0.0, extra_one_step=True
        )
        self.scheduler.set_timesteps(1000, training=True)

        self.seq_len = 28160  # [1, 32, 48, 44, 80]
        # self.seq_len = 27280  # [1, 31, 48, 44, 80]
    
        self.post_init()

    def enable_gradient_checkpointing(self) -> None:
        self.model.enable_gradient_checkpointing()

    def _convert_flow_pred_to_x0(self, flow_pred: torch.Tensor, xt: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        """
        Convert flow matching's prediction to x0 prediction.
        flow_pred: the prediction with shape [B, C, H, W]
        xt: the input noisy data with shape [B, C, H, W]
        timestep: the timestep with shape [B]

        pred = noise - x0
        x_t = (1-sigma_t) * x0 + sigma_t * noise
        we have x0 = x_t - sigma_t * pred
        """
        # use higher precision for calculations
        original_dtype = flow_pred.dtype
        flow_pred, xt, sigmas, timesteps = map(
            lambda x: x.double().to(flow_pred.device), [flow_pred, xt,
                                                        self.scheduler.sigmas,
                                                        self.scheduler.timesteps]
        )

        timestep_id = torch.argmin(
            (timesteps.unsqueeze(0) - timestep.unsqueeze(1)).abs(), dim=1)
        sigma_t = sigmas[timestep_id].reshape(-1, 1, 1, 1)
        x0_pred = xt - sigma_t * flow_pred
        return x0_pred.to(original_dtype)

    @staticmethod
    def _convert_x0_to_flow_pred(scheduler, x0_pred: torch.Tensor, xt: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        """
        Convert x0 prediction to flow matching's prediction.
        x0_pred: the x0 prediction with shape [B, C, H, W]
        xt: the input noisy data with shape [B, C, H, W]
        timestep: the timestep with shape [B]

        pred = (x_t - x_0) / sigma_t
        """
        # use higher precision for calculations
        original_dtype = x0_pred.dtype
        x0_pred, xt, sigmas, timesteps = map(
            lambda x: x.double().to(x0_pred.device), [x0_pred, xt,
                                                      scheduler.sigmas,
                                                      scheduler.timesteps]
        )
        timestep_id = torch.argmin(
            (timesteps.unsqueeze(0) - timestep.unsqueeze(1)).abs(), dim=1)
        sigma_t = sigmas[timestep_id].reshape(-1, 1, 1, 1)
        flow_pred = (xt - x0_pred) / sigma_t
        return flow_pred.to(original_dtype)

    def forward(
        self,
        noisy_image_or_video: torch.Tensor, 
        conditional_dict: dict,
        timestep: torch.Tensor, 
        kv_cache: Optional[List[dict]] = None,
        crossattn_cache: Optional[List[dict]] = None,
        current_start: Optional[int] = None,
        classify_mode: Optional[bool] = False,
        concat_time_embeddings: Optional[bool] = False,
        clean_x: Optional[torch.Tensor] = None,
        aug_t: Optional[torch.Tensor] = None,
        cache_start: Optional[int] = None
    ) -> torch.Tensor:
        prompt_embeds = conditional_dict["prompt_embeds"]

        # [B, F] -> [B]
        if self.uniform_timestep:
            input_timestep = timestep[:, 0]
        else:
            input_timestep = timestep

        logits = None
        # X0 prediction
        if kv_cache is not None:
            flow_pred = self.model(
                noisy_image_or_video.permute(0, 2, 1, 3, 4),
                t=input_timestep, context=prompt_embeds,
                seq_len=self.seq_len,
                kv_cache=kv_cache,
                crossattn_cache=crossattn_cache,
                current_start=current_start,
                cache_start=cache_start
            ).permute(0, 2, 1, 3, 4)
        else:
            if clean_x is not None:
                # teacher forcing
                flow_pred = self.model(
                    noisy_image_or_video.permute(0, 2, 1, 3, 4),
                    t=input_timestep, context=prompt_embeds,
                    seq_len=self.seq_len,
                    clean_x=clean_x.permute(0, 2, 1, 3, 4),
                    aug_t=aug_t,
                ).permute(0, 2, 1, 3, 4)
            else:
                if classify_mode:
                    flow_pred, logits = self.model(
                        noisy_image_or_video.permute(0, 2, 1, 3, 4),
                        t=input_timestep, context=prompt_embeds,
                        seq_len=self.seq_len,
                        classify_mode=True,
                        register_tokens=self._register_tokens,
                        cls_pred_branch=self._cls_pred_branch,
                        gan_ca_blocks=self._gan_ca_blocks,
                        concat_time_embeddings=concat_time_embeddings
                    )
                    flow_pred = flow_pred.permute(0, 2, 1, 3, 4)
                else:
                    flow_pred = self.model(
                        noisy_image_or_video.permute(0, 2, 1, 3, 4),
                        t=input_timestep, context=prompt_embeds,
                        seq_len=self.seq_len
                    ).permute(0, 2, 1, 3, 4)

        # print("flow_pred.shape", flow_pred.shape)
        # print("noisy_image_or_video.shape", noisy_image_or_video.shape)
        # print("timestep.shape", timestep.shape)
        pred_x0 = self._convert_flow_pred_to_x0(
            flow_pred=flow_pred.flatten(0, 1),
            xt=noisy_image_or_video.flatten(0, 1),
            timestep=timestep.flatten(0, 1)
        ).unflatten(0, flow_pred.shape[:2])

        if logits is not None:
            return flow_pred, pred_x0, logits

        return flow_pred, pred_x0

    def get_scheduler(self) -> SchedulerInterface:
        """
        Update the current scheduler with the interface's static method
        """
        scheduler = self.scheduler
        scheduler.convert_x0_to_noise = types.MethodType(
            SchedulerInterface.convert_x0_to_noise, scheduler)
        scheduler.convert_noise_to_x0 = types.MethodType(
            SchedulerInterface.convert_noise_to_x0, scheduler)
        scheduler.convert_velocity_to_x0 = types.MethodType(
            SchedulerInterface.convert_velocity_to_x0, scheduler)
        self.scheduler = scheduler
        return scheduler

    def post_init(self):
        """
        A few custom initialization steps that should be called after the object is created.
        Currently, the only one we have is to bind a few methods to scheduler.
        We can gradually add more methods here if needed.
        """
        self.get_scheduler()

class WanActionDiffusionWrapper(WanDiffusionWrapper):
    def __init__(self,
        model_name="Wan2.2-TI2V-5B",
        timestep_shift=8.0,
        is_causal=False,
        local_attn_size=-1,
        sink_size=0,
        control_mode='layer-wise',
        max_discrete_actions=6,
        logging_history=False,
        action_embedding_type="add",
        use_prope=False,
        use_prope_gate=True,
        prope_all_heads=False,
        inference_local_attn_size=None,
        chunk_drop=None,
        random_head_routing=False,
        # (routing_resample_every is a removed legacy knob; the actual mechanism
        #  is a fixed pool of routing_pool_size(=12) routings cycled EVERY step —
        #  see causal_model._maybe_randomize_routing. Configs still passing it
        #  are absorbed harmlessly by **kwargs.)
        rope_spatial_interp=False,
        rope_spatial_ref_grid=None,
        rope_spatial_cur_grid=None,
        action_head="v1",             # [plucker-v2, opt-in] "v2" = LayerNorm-free plucker entry head
        action_gate_zero_init=False,  # [plucker-v2, opt-in] zero-init scalar gate on injection
        action_inject_layers=None,    # [plucker-v2, opt-in] block idx list; None = all blocks (legacy)
        prope_log_t=False,            # [metric-prope, opt-in] log-bounded translation input to pose_mlp
        prope_se3_t_bound=False,      # [metric-prope, opt-in] SE3 multiplicative path with t/(1+||t||)
        plucker_dm_split=False,       # [metric-prope, opt-in] split plucker head into d/m, skip the joint LN(6)
        **kwargs
    ):
        nn.Module.__init__(self)


        if is_causal:
            if control_mode == 'layer-wise':
                base_model = CausalWanModel.from_pretrained(
                    model_name,
                    local_attn_size=local_attn_size,
                    sink_size=sink_size,
                    max_discrete_actions=max_discrete_actions,
                )
                config = base_model.config
                config = dict(config)
                config["model_type"] = "t2v_action"
                config["action_embedding_type"] = action_embedding_type
                config["use_prope"] = use_prope
                config["use_prope_gate"] = use_prope_gate
                config["prope_all_heads"] = prope_all_heads
                # [plucker-v2, opt-in] all three default to legacy behavior.
                config["action_head"] = action_head
                config["action_gate_zero_init"] = action_gate_zero_init
                config["action_inject_layers"] = action_inject_layers
                # [metric-prope, opt-in] input-transform switches (zero state-dict impact)
                config["prope_log_t"] = prope_log_t
                config["prope_se3_t_bound"] = prope_se3_t_bound
                config["plucker_dm_split"] = plucker_dm_split
                # [RoPE-interp] spatial Position-Interpolation (480p->720p warmstart). Must go
                # through from_config: it rescales the H/W RoPE freq table built in __init__.
                config["rope_spatial_interp"] = rope_spatial_interp
                config["rope_spatial_ref_grid"] = rope_spatial_ref_grid
                config["rope_spatial_cur_grid"] = rope_spatial_cur_grid
                if inference_local_attn_size is not None:
                    config["inference_local_attn_size"] = inference_local_attn_size
                self.model = CausalWanActionModel.from_config(config)
                # Partial weight load: only keys present in the checkpoint are overwritten;
                # newly added action-related parameters keep their initialization.
                self.model.load_state_dict(base_model.state_dict(), strict=False)

                del base_model
                gc.collect()
                torch.cuda.empty_cache()
            else:
                raise NotImplementedError("Only layer-wise control is supported for WanActionDiffusionWrapper5B")

        else:
            raise NotImplementedError("Non-causal diffusion is not supported for WanActionDiffusionWrapper5B")

        # chunk_drop training augmentation: when enabled, training forward
        # randomly masks K/V of past chunks (see CausalWanModel._forward_train).
        # Inference is untouched — cached self.block_mask (no drop) is used.
        # Accept OmegaConf DictConfig too; normalise to a plain dict for safety.
        if chunk_drop is not None and not isinstance(chunk_drop, dict):
            try:
                from omegaconf import OmegaConf
                chunk_drop = OmegaConf.to_container(chunk_drop, resolve=True)
            except Exception:
                # fall back: best-effort dict() conversion
                chunk_drop = dict(chunk_drop)
        if isinstance(chunk_drop, dict) and chunk_drop.get("enabled", False):
            self.model.chunk_drop_config = chunk_drop
        else:
            self.model.chunk_drop_config = None

        # [head-routing ablation] when True, each TRAIN step swaps which heads are
        # global(-1)/local, preserving local_attn_size's global:local budget:
        # a FIXED pool of routing_pool_size(=12) permutations is cycled every step
        # (causal_model._maybe_randomize_routing; bounded flex kernel memory).
        # Inference/eval keep the fixed pattern. Default False → no behaviour change.
        self.model.random_head_routing = bool(random_head_routing)

        self.model.eval()

        # For non-causal diffusion, all frames share the same timestep
        self.uniform_timestep = not is_causal

        self.scheduler = FlowMatchScheduler(
            shift=timestep_shift, sigma_min=0.0, extra_one_step=True
        )
        self.scheduler.set_timesteps(1000, training=True)

        self.seq_len = 880 * 24  # [1, 21, 16, 60, 104]
        self.post_init()

    def forward(
        self,
        noisy_image_or_video: torch.Tensor, 
        conditional_dict: dict,
        timestep: torch.Tensor, 
        kv_cache: Optional[List[dict]] = None,
        crossattn_cache: Optional[List[dict]] = None,
        current_start: Optional[int] = None,
        classify_mode: Optional[bool] = False,
        concat_time_embeddings: Optional[bool] = False,
        clean_x: Optional[torch.Tensor] = None,
        aug_t: Optional[torch.Tensor] = None,
        cache_start: Optional[int] = None,
        combined_action_ids: Optional[torch.Tensor] = None,
        prope_context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        prompt_embeds = conditional_dict["prompt_embeds"]

        # [B, F] -> [B]
        if self.uniform_timestep:
            input_timestep = timestep[:, 0]
        else:
            input_timestep = timestep

        logits = None
        # X0 prediction
        if kv_cache is not None:
            flow_pred = self.model(
                noisy_image_or_video.permute(0, 2, 1, 3, 4),
                t=input_timestep, context=prompt_embeds,
                seq_len=self.seq_len,
                kv_cache=kv_cache,
                crossattn_cache=crossattn_cache,
                current_start=current_start,
                cache_start=cache_start,
                combined_action_ids=combined_action_ids,
                prope_context=prope_context,
            ).permute(0, 2, 1, 3, 4)
        else:
            if clean_x is not None:
                # teacher forcing
                flow_pred = self.model(
                    noisy_image_or_video.permute(0, 2, 1, 3, 4),
                    t=input_timestep, context=prompt_embeds,
                    seq_len=self.seq_len,
                    clean_x=clean_x.permute(0, 2, 1, 3, 4),
                    aug_t=aug_t,
                    combined_action_ids=combined_action_ids,
                    prope_context=prope_context,
                ).permute(0, 2, 1, 3, 4)
            else:
                if classify_mode:
                    flow_pred, logits = self.model(
                        noisy_image_or_video.permute(0, 2, 1, 3, 4),
                        t=input_timestep, context=prompt_embeds,
                        seq_len=self.seq_len,
                        classify_mode=True,
                        register_tokens=self._register_tokens,
                        cls_pred_branch=self._cls_pred_branch,
                        gan_ca_blocks=self._gan_ca_blocks,
                        concat_time_embeddings=concat_time_embeddings,
                        combined_action_ids=combined_action_ids,
                        prope_context=prope_context,
                    )
                    flow_pred = flow_pred.permute(0, 2, 1, 3, 4)
                else:
                    flow_pred = self.model(
                        noisy_image_or_video.permute(0, 2, 1, 3, 4),
                        t=input_timestep, context=prompt_embeds,
                        seq_len=self.seq_len,
                        combined_action_ids=combined_action_ids,
                        prope_context=prope_context,
                    ).permute(0, 2, 1, 3, 4)

        # print("flow_pred.shape", flow_pred.shape)
        # print("noisy_image_or_video.shape", noisy_image_or_video.shape)
        # print("timestep.shape", timestep.shape)
        pred_x0 = self._convert_flow_pred_to_x0(
            flow_pred=flow_pred.flatten(0, 1),
            xt=noisy_image_or_video.flatten(0, 1),
            timestep=timestep.flatten(0, 1)
        ).unflatten(0, flow_pred.shape[:2])

        if logits is not None:
            return flow_pred, pred_x0, logits

        return flow_pred, pred_x0