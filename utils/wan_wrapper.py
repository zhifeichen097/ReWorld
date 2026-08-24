import types
from typing import List, Optional
import os
import torch
from torch import nn

from utils.scheduler import SchedulerInterface, FlowMatchScheduler
from wan.modules.tokenizers import HuggingfaceTokenizer
from wan.modules.model import WanModel, RegisterTokens, GanAttentionBlock
from wan.modules.vae import _video_vae
from wan.modules.t5 import umt5_xxl
from wan.modules.causal_model import CausalWanModel
from wan.modules.action_model import CausalWanActionModel, WanActionModel

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
        # Directory of the Wan2.1-T2V-1.3B base model (official Wan release).
        # Set via the `model_dir` argument or the WAN21_MODEL_DIR env var.
        model_dir = model_dir or os.environ.get("WAN21_MODEL_DIR", "./checkpoints/Wan2.1-T2V-1.3B")

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

        for u, v in zip(context, seq_lens):
            u[v:] = 0.0  # set padding to 0.0

        return {
            "prompt_embeds": context
        }


class WanVAEWrapper(torch.nn.Module):
    def __init__(self, model_dir: Optional[str] = None):
        super().__init__()
        # Directory of the Wan2.1-T2V-1.3B base model (official Wan release).
        # Set via the `model_dir` argument or the WAN21_MODEL_DIR env var.
        model_dir = model_dir or os.environ.get("WAN21_MODEL_DIR", "./checkpoints/Wan2.1-T2V-1.3B")
        mean = [
            -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
            0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921
        ]
        std = [
            2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
            3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160
        ]
        self.mean = torch.tensor(mean, dtype=torch.float32)
        self.std = torch.tensor(std, dtype=torch.float32)

        # init model
        self.model = _video_vae(
            pretrained_path=os.path.join(model_dir, "Wan2.1_VAE.pth"),
            z_dim=16,
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


class WanDiffusionWrapper(torch.nn.Module):
    def __init__(
            self,
            model_name="Wan2.1-T2V-1.3B",
            timestep_shift=8.0,
            is_causal=False,
            local_attn_size=-1,
            sink_size=0,
            logging_history=False,
    ):
        super().__init__()

        if is_causal:
            # Load in t2v mode first (without action_embedding) to avoid missing-key errors.
            self.model = CausalWanModel.from_pretrained(
                model_name,
                local_attn_size=local_attn_size,
                sink_size=sink_size,
                logging_history=logging_history
            )

        else:
            self.model = WanModel.from_pretrained(model_name)
        self.model.eval()

        # For non-causal diffusion, all frames share the same timestep
        self.uniform_timestep = not is_causal

        self.scheduler = FlowMatchScheduler(
            shift=timestep_shift, sigma_min=0.0, extra_one_step=True
        )
        self.scheduler.set_timesteps(1000, training=True)

        self.seq_len = 32760  # [1, 21, 16, 60, 104]
        self.post_init()

    def enable_gradient_checkpointing(self) -> None:
        self.model.enable_gradient_checkpointing()

    def adding_cls_branch(self, atten_dim=1536, num_class=4, time_embed_dim=0) -> None:
        # NOTE: This is hard coded for WAN2.1-T2V-1.3B for now!!!!!!!!!!!!!!!!!!!!
        self._cls_pred_branch = nn.Sequential(
            # Input: [B, 384, 21, 60, 104]
            nn.LayerNorm(atten_dim * 3 + time_embed_dim),
            nn.Linear(atten_dim * 3 + time_embed_dim, 1536),
            nn.SiLU(),
            nn.Linear(atten_dim, num_class)
        )
        self._cls_pred_branch.requires_grad_(True)
        num_registers = 3
        self._register_tokens = RegisterTokens(num_registers=num_registers, dim=atten_dim)
        self._register_tokens.requires_grad_(True)

        gan_ca_blocks = []
        for _ in range(num_registers):
            block = GanAttentionBlock()
            gan_ca_blocks.append(block)
        self._gan_ca_blocks = nn.ModuleList(gan_ca_blocks)
        self._gan_ca_blocks.requires_grad_(True)
        # self.has_cls_branch = True

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
        cache_start: Optional[int] = None,
        combined_action_ids: Optional[torch.Tensor] = None,
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
                combined_action_ids=combined_action_ids
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
                    )
                    flow_pred = flow_pred.permute(0, 2, 1, 3, 4)
                else:
                    flow_pred = self.model(
                        noisy_image_or_video.permute(0, 2, 1, 3, 4),
                        t=input_timestep, context=prompt_embeds,
                        seq_len=self.seq_len,
                        combined_action_ids=combined_action_ids,
                    ).permute(0, 2, 1, 3, 4)

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
        model_name="Wan2.1-T2V-1.3B", 
        timestep_shift=8.0, 
        is_causal=False, 
        local_attn_size=-1, 
        sink_size=0,
        control_mode='layer-wise',
        max_discrete_actions=6,
        logging_history=False,
        action_embedding_type="add",
        **kwargs
    ):
        nn.Module.__init__(self)
        

        if is_causal:
            if control_mode == 'layer-wise':
                # First load the original pretrained CausalWanModel weights
                # (without the layer-wise action embedding).
                base_model = CausalWanModel.from_pretrained(
                    model_name,
                    local_attn_size=local_attn_size,
                    sink_size=sink_size,
                    max_discrete_actions=max_discrete_actions,
                    logging_history=logging_history,
                )
                # Build CausalWanActionModel from the same config (its init_weights
                # performs the custom initialization).
                config = base_model.config
                config = dict(config)  # FrozenDict is immutable; convert to dict before adding fields
                config["model_type"] = "t2v_action"
                config["action_embedding_type"] = action_embedding_type
                self.model = CausalWanActionModel.from_config(config)
                # Partial weight load: only keys present in the checkpoint are overwritten;
                # newly added action-related parameters keep their initialization.
                self.model.load_state_dict(base_model.state_dict(), strict=False)

                del base_model
                gc.collect()
                torch.cuda.empty_cache()
            else:
                # Load in t2v mode first (without action_embedding) to avoid missing-key errors.
                self.model = CausalWanModel.from_pretrained(
                    model_name, 
                    local_attn_size=local_attn_size, 
                    sink_size=sink_size,
                    max_discrete_actions=max_discrete_actions,
                    logging_history=logging_history,
                )
                self.model.model_type = "t2v_action"
                from wan.modules.model import MLPProj
                self.model.action_embedding = MLPProj(max_discrete_actions, self.model.dim)
                self.model.action_embedding.apply(custom_init)
            
        else:
            if control_mode == 'layer-wise':
                # First load the original pretrained WanModel weights
                # (without the layer-wise action embedding).
                base_model = WanModel.from_pretrained(
                    model_name,
                    max_discrete_actions=max_discrete_actions,
                )
                # Build WanActionModel from the same config (its init_weights
                # performs the custom initialization).
                config = base_model.config
                config = dict(config)  # FrozenDict is immutable; convert to dict before adding fields
                config["model_type"] = "t2v_action"
                config["action_embedding_type"] = action_embedding_type
                self.model = WanActionModel.from_config(config)
                # Partial weight load: only keys present in the checkpoint are overwritten;
                # newly added action-related parameters keep their initialization.
                self.model.load_state_dict(base_model.state_dict(), strict=False)

                del base_model
                gc.collect()
                torch.cuda.empty_cache()
            else:
                self.model = WanModel.from_pretrained(
                    model_name,
                    max_discrete_actions=max_discrete_actions,
                )
                self.model.model_type = "t2v_action"
                from wan.modules.model import MLPProj
                self.model.action_embedding = MLPProj(max_discrete_actions, self.model.dim)
                self.model.action_embedding.apply(custom_init)

        self.model.eval()

        # For non-causal diffusion, all frames share the same timestep
        self.uniform_timestep = not is_causal

        self.scheduler = FlowMatchScheduler(
            shift=timestep_shift, sigma_min=0.0, extra_one_step=True
        )
        self.scheduler.set_timesteps(1000, training=True)

        self.seq_len = 32760  # [1, 21, 16, 60, 104]
        self.post_init()