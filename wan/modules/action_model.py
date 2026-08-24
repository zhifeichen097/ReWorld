from wan.modules.causal_model import CausalWanModel, CausalWanSelfAttention

from wan.modules.model import (
    WanLayerNorm,
    WAN_CROSSATTENTION_CLASSES,
    MLPProj,
    sinusoidal_embedding_1d,
    WanModel,
    WanSelfAttention,
)
import torch.nn as nn
import torch
from einops import repeat
from diffusers.configuration_utils import register_to_config

class WanActionAttentionBlock(nn.Module):

    def __init__(self,
                 cross_attn_type,
                 dim,
                 ffn_dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=False,
                 eps=1e-6,
                 max_discrete_actions=6,
                 action_embedding_type="add"):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps
        self.action_embedding_type = action_embedding_type

        # layers
        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = WanSelfAttention(dim, num_heads, window_size, qk_norm,
                                          eps)
        self.norm3 = WanLayerNorm(
            dim, eps,
            elementwise_affine=True) if cross_attn_norm else nn.Identity()
        self.cross_attn = WAN_CROSSATTENTION_CLASSES[cross_attn_type](dim,
                                                                      num_heads,
                                                                      (-1, -1),
                                                                      qk_norm,
                                                                      eps)
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim), nn.GELU(approximate='tanh'),
            nn.Linear(ffn_dim, dim))

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

        # action embedding: "add" -> dim, "adain" -> 2*dim (scale + shift)
        if action_embedding_type not in ("add", "adain"):
            raise ValueError(
                f"action_embedding_type must be 'add' or 'adain', got '{action_embedding_type}'"
            )
        action_out_dim = dim if action_embedding_type == "add" else 2 * dim
        self.action_embedding = MLPProj(max_discrete_actions, action_out_dim)

    def forward(
        self,
        x,
        e,
        seq_lens,
        grid_sizes,
        freqs,
        context,
        context_lens,
        combined_action_ids=None,
    ):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
            e(Tensor): Shape [B, 6, C]
            seq_lens(Tensor): Shape [B], length of each sequence in batch
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        # assert e.dtype == torch.float32
        # with amp.autocast(dtype=torch.float32):
        e = (self.modulation + e).chunk(6, dim=1)
        # assert e[0].dtype == torch.float32

        # self-attention
        y = self.self_attn(
            self.norm1(x) * (1 + e[1]) + e[0], seq_lens, grid_sizes,
            freqs)
        # with amp.autocast(dtype=torch.float32):
        x = x + y * e[2]

        # action embedding
        if combined_action_ids is not None:
            if combined_action_ids.ndim == 3: # e.g., delta relative pose
                action_embedding = self.action_embedding(combined_action_ids)  # [B, F, dim] or [B, F, 2*dim]
                spatial_size = grid_sizes[0, 1] * grid_sizes[0, 2]  # [B] H*W per sample; assumes all samples share the same H*W
                action_embedding = torch.repeat_interleave(action_embedding, spatial_size.item(), dim=1)  # [B, F*H*W, ...]

            elif combined_action_ids.ndim == 5: # e.g., plucker embedding
                action_embedding = self.action_embedding(combined_action_ids) # [B, F, H, W, dim] or [B, F, H, W, 2*dim]
                # Flatten to [B, F*H*W, ...] with same order as x (F, H, W flattened)
                B, F, H, W, _ = action_embedding.shape
                action_embedding = action_embedding.view(B, F * H * W, -1)

            if self.action_embedding_type == "adain":
                action_scale, action_shift = action_embedding.chunk(2, dim=-1)
                x = x * (1 + action_scale) + action_shift
            else:
                x = x + action_embedding

        # cross-attention & ffn function
        def cross_attn_ffn(x, context, context_lens, e):
            x = x + self.cross_attn(self.norm3(x), context, context_lens)
            y = self.ffn(self.norm2(x) * (1 + e[4]) + e[3])
            # with amp.autocast(dtype=torch.float32):
            x = x + y * e[5]
            return x

        x = cross_attn_ffn(x, context, context_lens, e)
        return x

class WanActionModel(WanModel):
    @register_to_config
    def __init__(
        self,
        model_type: str = "t2v",
        patch_size=(1, 2, 2),
        text_len: int = 512,
        in_dim: int = 16,
        dim: int = 2048,
        ffn_dim: int = 8192,
        freq_dim: int = 256,
        text_dim: int = 4096,
        out_dim: int = 16,
        num_heads: int = 16,
        num_layers: int = 32,
        window_size=(-1, -1),
        qk_norm: bool = True,
        cross_attn_norm: bool = True,
        eps: float = 1e-6,
        max_discrete_actions: int = 6,
        action_embedding_type: str = "add",
    ):
        # call the parent constructor, passing the config fields through
        super().__init__(
            model_type=model_type,
            patch_size=patch_size,
            text_len=text_len,
            in_dim=in_dim,
            dim=dim,
            ffn_dim=ffn_dim,
            freq_dim=freq_dim,
            text_dim=text_dim,
            out_dim=out_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            window_size=window_size,
            qk_norm=qk_norm,
            cross_attn_norm=cross_attn_norm,
            eps=eps,
            max_discrete_actions=max_discrete_actions,
        )

        # then override the blocks + custom initialization (keeping the original logic)
        cross_attn_type = "t2v_cross_attn" if "t2v" in self.model_type else "i2v_cross_attn"
        self.blocks = nn.ModuleList(
            [
                WanActionAttentionBlock(
                    cross_attn_type=cross_attn_type,
                    dim=self.dim,
                    ffn_dim=self.ffn_dim,
                    num_heads=self.num_heads,
                    window_size=self.window_size,
                    qk_norm=self.qk_norm,
                    cross_attn_norm=self.cross_attn_norm,
                    eps=self.eps,
                    max_discrete_actions=self.max_discrete_actions,
                    action_embedding_type=action_embedding_type,
                )
                for _ in range(self.num_layers)
            ]
        )

        try:
            from utils.wan_wrapper import custom_init
        except ImportError:
            custom_init = None

        if custom_init is not None:
            for block in self.blocks:
                if hasattr(block, "action_embedding") and block.action_embedding is not None:
                    custom_init(block.action_embedding)

    def _forward(
        self,
        x,
        t,
        context,
        seq_len,
        classify_mode=False,
        concat_time_embeddings=False,
        register_tokens=None,
        cls_pred_branch=None,
        gan_ca_blocks=None,
        clip_fea=None,
        y=None,
        combined_action_ids=None,
    ):
        r"""
        Forward pass through the diffusion model

        Args:
            x (List[Tensor]):
                List of input video tensors, each with shape [C_in, F, H, W]
            t (Tensor):
                Diffusion timesteps tensor of shape [B]
            context (List[Tensor]):
                List of text embeddings each with shape [L, C]
            seq_len (`int`):
                Maximum sequence length for positional encoding
            clip_fea (Tensor, *optional*):
                CLIP image features for image-to-video mode
            y (List[Tensor], *optional*):
                Conditional video inputs for image-to-video mode, same shape as x

        Returns:
            List[Tensor]:
                List of denoised video tensors with original input shapes [C_out, F, H / 8, W / 8]
        """
        if self.model_type == 'i2v':
            assert clip_fea is not None and y is not None
        # params
        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        if y is not None:
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

        # embeddings
        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
        grid_sizes = torch.stack(
            [torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
        x = [u.flatten(2).transpose(1, 2) for u in x]
        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        # assert seq_lens.max() <= seq_len
        x = torch.cat([
            torch.cat([u, u.new_zeros(1, seq_lens[0] - u.size(1), u.size(2))],
                      dim=1) for u in x
        ])

        # time embeddings
        # with amp.autocast(dtype=torch.float32):
        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, t).type_as(x))
        e0 = self.time_projection(e).unflatten(1, (6, self.dim))
        # assert e.dtype == torch.float32 and e0.dtype == torch.float32

        # context
        context_lens = None
        context = self.text_embedding(
            torch.stack([
                torch.cat(
                    [u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                for u in context
            ]))

        if clip_fea is not None:
            context_clip = self.img_emb(clip_fea)  # bs x 257 x dim
            context = torch.concat([context_clip, context], dim=1)
        
        # arguments
        kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=self.freqs,
            context=context,
            context_lens=context_lens,
            combined_action_ids=combined_action_ids,
        )

        def create_custom_forward(module):
            def custom_forward(*inputs, **kwargs):
                return module(*inputs, **kwargs)
            return custom_forward

        # TODO: Tune the number of blocks for feature extraction
        final_x = None
        if classify_mode:
            assert register_tokens is not None
            assert gan_ca_blocks is not None
            assert cls_pred_branch is not None

            final_x = []
            registers = repeat(register_tokens(), "n d -> b n d", b=x.shape[0])
            # x = torch.cat([registers, x], dim=1)

        gan_idx = 0
        for ii, block in enumerate(self.blocks):
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                x = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    x, **kwargs,
                    use_reentrant=False,
                )
            else:
                x = block(x, **kwargs)

            if classify_mode and ii in [13, 21, 29]:
                gan_token = registers[:, gan_idx: gan_idx + 1]
                final_x.append(gan_ca_blocks[gan_idx](x, gan_token))
                gan_idx += 1

        if classify_mode:
            final_x = torch.cat(final_x, dim=1)
            if concat_time_embeddings:
                final_x = cls_pred_branch(torch.cat([final_x, 10 * e[:, None, :]], dim=1).view(final_x.shape[0], -1))
            else:
                final_x = cls_pred_branch(final_x.view(final_x.shape[0], -1))

        # head
        x = self.head(x, e)

        # unpatchify
        x = self.unpatchify(x, grid_sizes)

        if classify_mode:
            return torch.stack(x), final_x

        return torch.stack(x)

class CausalWanActionAttentionBlock(nn.Module):

    def __init__(self,
                 cross_attn_type,
                 dim,
                 ffn_dim,
                 num_heads,
                 local_attn_size=-1,
                 sink_size=0,
                 qk_norm=True,
                 cross_attn_norm=False,
                 eps=1e-6,
                 logging_history=False,
                 max_discrete_actions=6,
                 action_embedding_type="add"
        ):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.local_attn_size = local_attn_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps
        self.action_embedding_type = action_embedding_type

        # layers
        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = CausalWanSelfAttention(dim, num_heads, local_attn_size, sink_size, qk_norm, eps, logging_history)
        self.norm3 = WanLayerNorm(
            dim, eps,
            elementwise_affine=True) if cross_attn_norm else nn.Identity()
        self.cross_attn = WAN_CROSSATTENTION_CLASSES[cross_attn_type](dim,
                                                                      num_heads,
                                                                      (-1, -1),
                                                                      qk_norm,
                                                                      eps)
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim), nn.GELU(approximate='tanh'),
            nn.Linear(ffn_dim, dim))

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

        # action embedding: "add" -> dim, "adain" -> 2*dim (scale + shift)
        if action_embedding_type not in ("add", "adain"):
            raise ValueError(
                f"action_embedding_type must be 'add' or 'adain', got '{action_embedding_type}'"
            )
        action_out_dim = 2 * dim if action_embedding_type == "adain" else dim
        self.action_embedding = MLPProj(max_discrete_actions, action_out_dim)



    def forward(
        self,
        x,
        e,
        seq_lens,
        grid_sizes,
        freqs,
        context,
        context_lens,
        block_mask,
        kv_cache=None,
        crossattn_cache=None,
        current_start=0,
        cache_start=None,
        combined_action_ids=None,
        teacher_forcing=False,
    ):
        num_frames, frame_seqlen = e.shape[1], x.shape[1] // e.shape[1]
        e = (self.modulation.unsqueeze(1) + e).chunk(6, dim=2)

        # =====================================================================
        # Optimization 1: extract the branch feature
        # =====================================================================
        # this is the feature fed to self-attention after LayerNorm and the base time/space modulation
        normed_x = (self.norm1(x).unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * (1 + e[1]) + e[0]).flatten(1, 2)

        # =====================================================================
        # Optimization 2: inject the action without torch.cat (avoids OOM) and without ever polluting the trunk x
        # =====================================================================
        if combined_action_ids is not None:
            if combined_action_ids.ndim == 3: # e.g., delta relative pose
                action_embedding = self.action_embedding(combined_action_ids)  
                spatial_size = grid_sizes[0, 1] * grid_sizes[0, 2]  
                action_embedding = torch.repeat_interleave(action_embedding, spatial_size.item(), dim=1) 
            elif combined_action_ids.ndim == 5: # e.g., plucker embedding
                action_embedding = self.action_embedding(combined_action_ids) 
                B, F, H, W, _ = action_embedding.shape
                action_embedding = action_embedding.view(B, F * H * W, -1)
            else:
                raise ValueError(f"Unsupported action_ids dimension: {combined_action_ids.ndim}")

            if teacher_forcing:
                half = normed_x.shape[1] // 2
                
                if self.action_embedding_type == "adain":
                    action_scale, action_shift = action_embedding.chunk(2, dim=-1)
                    # [memory optimization] build a same-shape zero tensor instead of slicing+concatenation
                    padded_scale = torch.zeros_like(normed_x)
                    padded_shift = torch.zeros_like(normed_x)
                    padded_scale[:, half:, :] = action_scale
                    padded_shift[:, half:, :] = action_shift
                    # the first (clean) half effectively gets * (1 + 0) + 0 — kept exactly as-is
                    normed_x = normed_x * (1 + padded_scale) + padded_shift
                else:
                    # [add-mode memory optimization] zero-pad the first half and do a single global addition
                    padded_action = torch.zeros_like(normed_x)
                    padded_action[:, half:, :] = action_embedding
                    normed_x = normed_x + padded_action
            else:
                # DF mode: applied over the full sequence; no memory pressure here
                if self.action_embedding_type == "adain":
                    action_scale, action_shift = action_embedding.chunk(2, dim=-1)
                    normed_x = normed_x * (1 + action_scale) + action_shift
                else:
                    normed_x = normed_x + action_embedding

        # =====================================================================
        # Optimization 3: self-attention consumes the modified normed_x
        # =====================================================================
        y = self.self_attn(
            normed_x,  # <--- attention operates on the action-injected branch feature
            seq_lens, grid_sizes,
            freqs, block_mask, kv_cache, current_start, cache_start)

        # =====================================================================
        # Optimization 4: add the attention output back onto the clean trunk x
        # =====================================================================
        # the trunk x stays a clean highway, never truncated by the action, fully inheriting the pretrained base's physical priors
        x = x + (y.unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * e[2]).flatten(1, 2)

        # =====================================================================
        # Cross-attention & FFN function (original logic kept untouched)
        # =====================================================================
        def cross_attn_ffn(x, context, context_lens, e, crossattn_cache=None):
            x = x + self.cross_attn(self.norm3(x), context,
                                    context_lens, crossattn_cache=crossattn_cache)
            y = self.ffn(
                (self.norm2(x).unflatten(dim=1, sizes=(num_frames,
                 frame_seqlen)) * (1 + e[4]) + e[3]).flatten(1, 2)
            )
            x = x + (y.unflatten(dim=1, sizes=(num_frames,
                     frame_seqlen)) * e[5]).flatten(1, 2)
            return x

        x = cross_attn_ffn(x, context, context_lens, e, crossattn_cache)
        return x

    # def forward(
    #     self,
    #     x,
    #     e,
    #     seq_lens,
    #     grid_sizes,
    #     freqs,
    #     context,
    #     context_lens,
    #     block_mask,
    #     kv_cache=None,
    #     crossattn_cache=None,
    #     current_start=0,
    #     cache_start=None,
    #     combined_action_ids=None,
    #     teacher_forcing=False,
    # ):
    #     r"""
    #     Args:
    #         x(Tensor): Shape [B, L, C]
    #         e(Tensor): Shape [B, F, 6, C]
    #         seq_lens(Tensor): Shape [B], length of each sequence in batch
    #         grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
    #         freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
    #     """
    #     num_frames, frame_seqlen = e.shape[1], x.shape[1] // e.shape[1]
    #     # assert e.dtype == torch.float32
    #     # with amp.autocast(dtype=torch.float32):
    #     e = (self.modulation.unsqueeze(1) + e).chunk(6, dim=2)
    #     # assert e[0].dtype == torch.float32

    #     # self-attention
    #     y = self.self_attn(
    #         (self.norm1(x).unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * (1 + e[1]) + e[0]).flatten(1, 2),
    #         seq_lens, grid_sizes,
    #         freqs, block_mask, kv_cache, current_start, cache_start)

    #     # with amp.autocast(dtype=torch.float32):
    #     x = x + (y.unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * e[2]).flatten(1, 2)

    #     # action embedding
    #     if combined_action_ids is not None:
    #         if combined_action_ids.ndim == 3: # e.g., delta relative pose
    #             action_embedding = self.action_embedding(combined_action_ids)  # [B, F, dim] or [B, F, 2*dim]
    #             spatial_size = grid_sizes[0, 1] * grid_sizes[0, 2]  # [B] H*W per sample; assumes all samples share the same H*W
    #             action_embedding = torch.repeat_interleave(action_embedding, spatial_size.item(), dim=1)  # [B, F*H*W, ...]

    #         elif combined_action_ids.ndim == 5: # e.g., plucker embedding
    #             action_embedding = self.action_embedding(combined_action_ids) # [B, F, H, W, dim] or [B, F, H, W, 2*dim]
    #             # Flatten to [B, F*H*W, ...] with same order as x (F, H, W flattened)
    #             B, F, H, W, _ = action_embedding.shape
    #             action_embedding = action_embedding.view(B, F * H * W, -1)

    #         # under teacher forcing, the first half of x is clean GT; the action embedding is applied only to the (noisy) second half
    #         if teacher_forcing:
    #             half = x.shape[1] // 2
    #             x_apply = x[:, half:, :]
    #             if self.action_embedding_type == "adain":
    #                 action_scale, action_shift = action_embedding.chunk(2, dim=-1)
    #                 x_apply = x_apply * (1 + action_scale) + action_shift
    #             else:
    #                 x_apply = x_apply + action_embedding
    #             x = torch.cat([x[:, :half, :], x_apply], dim=1)
    #         else:
    #             if self.action_embedding_type == "adain":
    #                 action_scale, action_shift = action_embedding.chunk(2, dim=-1)
    #                 x = x * (1 + action_scale) + action_shift
    #             else:
    #                 x = x + action_embedding

    #     # cross-attention & ffn function
    #     def cross_attn_ffn(x, context, context_lens, e, crossattn_cache=None):
    #         x = x + self.cross_attn(self.norm3(x), context,
    #                                 context_lens, crossattn_cache=crossattn_cache)
    #         y = self.ffn(
    #             (self.norm2(x).unflatten(dim=1, sizes=(num_frames,
    #              frame_seqlen)) * (1 + e[4]) + e[3]).flatten(1, 2)
    #         )
    #         # with amp.autocast(dtype=torch.float32):
    #         x = x + (y.unflatten(dim=1, sizes=(num_frames,
    #                  frame_seqlen)) * e[5]).flatten(1, 2)
    #         return x

    #     x = cross_attn_ffn(x, context, context_lens, e, crossattn_cache)
    #     return x

class CausalWanActionModel(CausalWanModel):
    """
    Layer-wise action variant built on CausalWanModel:
    - inherits CausalWanModel and reuses most of its logic
    - replaces blocks with CausalWanActionAttentionBlock
    - passes combined_action_ids to every block in forward
    """

    @register_to_config
    def __init__(
        self,
        model_type: str = "t2v",
        patch_size=(1, 2, 2),
        text_len: int = 512,
        in_dim: int = 16,
        dim: int = 2048,
        ffn_dim: int = 8192,
        freq_dim: int = 256,
        text_dim: int = 4096,
        out_dim: int = 16,
        num_heads: int = 16,
        num_layers: int = 32,
        local_attn_size: int = -1,
        sink_size: int = 0,
        qk_norm: bool = True,
        cross_attn_norm: bool = True,
        max_discrete_actions: int = 6,
        eps: float = 1e-6,
        logging_history: bool = False,
        action_embedding_type: str = "add",
    ):
        super().__init__(
            model_type=model_type,
            patch_size=patch_size,
            text_len=text_len,
            in_dim=in_dim,
            dim=dim,
            ffn_dim=ffn_dim,
            freq_dim=freq_dim,
            text_dim=text_dim,
            out_dim=out_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            local_attn_size=local_attn_size,
            sink_size=sink_size,
            qk_norm=qk_norm,
            cross_attn_norm=cross_attn_norm,
            max_discrete_actions=max_discrete_actions,
            eps=eps,
            logging_history=logging_history,
        )

        # replace the original blocks with action-aware blocks
        cross_attn_type = "t2v_cross_attn" if "t2v" in self.model_type else "i2v_cross_attn"
        self.blocks = nn.ModuleList(
            [
                CausalWanActionAttentionBlock(
                    cross_attn_type=cross_attn_type,
                    dim=self.dim,
                    ffn_dim=self.ffn_dim,
                    num_heads=self.num_heads,
                    local_attn_size=self.local_attn_size,
                    sink_size=self.config.sink_size if hasattr(self, "config") else 0,
                    qk_norm=self.qk_norm,
                    cross_attn_norm=self.cross_attn_norm,
                    eps=self.eps,
                    logging_history=self.logging_history,
                    max_discrete_actions=self.max_discrete_actions,
                    action_embedding_type=action_embedding_type,
                )
                for _ in range(self.num_layers)
            ]
        )

        # run a custom initialization on each block's action_embedding
        try:
            from utils.wan_wrapper import custom_init
        except ImportError:
            custom_init = None

        if custom_init is not None:
            for block in self.blocks:
                if hasattr(block, "action_embedding") and block.action_embedding is not None:
                    custom_init(block.action_embedding)

    # -------- The two functions below modify CausalWanModel to pass combined_action_ids into each block --------

    def _forward_inference(
        self,
        x,
        t,
        context,
        seq_len,
        clip_fea=None,
        y=None,
        kv_cache: dict = None,
        crossattn_cache: dict = None,
        current_start: int = 0,
        cache_start: int = 0,
        combined_action_ids=None,
    ):
        if self.model_type == "i2v":
            assert clip_fea is not None and y is not None

        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        if y is not None:
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

        # embeddings
        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
        grid_sizes = torch.stack(
            [torch.tensor(u.shape[2:], dtype=torch.long) for u in x]
        )
        x = [u.flatten(2).transpose(1, 2) for u in x]
        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        # assert seq_lens.max() <= seq_len
        x = torch.cat(x)

        # time embeddings
        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, t.flatten()).type_as(x)
        )
        e0 = self.time_projection(e).unflatten(1, (6, self.dim)).unflatten(
            dim=0, sizes=t.shape
        )

        # text context
        context_lens = None
        context = self.text_embedding(
            torch.stack(
                [
                    torch.cat(
                        [u, u.new_zeros(self.text_len - u.size(0), u.size(1))]
                    )
                    for u in context
                ]
            )
        )

        if clip_fea is not None:
            context_clip = self.img_emb(clip_fea)  # bs x 257 x dim
            context = torch.concat([context_clip, context], dim=1)

        # The action embedding is no longer added at the model level;
        # combined_action_ids is passed to each block via kwargs and handled inside the block.
        kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=self.freqs,
            context=context,
            context_lens=context_lens,
            block_mask=self.block_mask,
            combined_action_ids=combined_action_ids,
        )

        def create_custom_forward(module):
            def custom_forward(*inputs, **inner_kwargs):
                return module(*inputs, **inner_kwargs)

            return custom_forward

        for block_index, block in enumerate(self.blocks):
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                kwargs.update(
                    {
                        "kv_cache": kv_cache[block_index],
                        "current_start": current_start,
                        "cache_start": cache_start,
                    }
                )
                x = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    x,
                    **kwargs,
                    use_reentrant=False,
                )
            else:
                kwargs.update(
                    {
                        "kv_cache": kv_cache[block_index],
                        "crossattn_cache": crossattn_cache[block_index],
                        "current_start": current_start,
                        "cache_start": cache_start,
                    }
                )
                x = block(x, **kwargs)

        x = self.head(x, e.unflatten(dim=0, sizes=t.shape).unsqueeze(2))
        x = self.unpatchify(x, grid_sizes)
        return torch.stack(x)

    def _forward_train(
        self,
        x,
        t,
        context,
        seq_len,
        clean_x=None,
        aug_t=None,
        clip_fea=None,
        y=None,
        combined_action_ids=None,
    ):
        if self.model_type == "i2v":
            assert clip_fea is not None and y is not None

        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        # build the block-wise causal mask (reusing the base class's static method)
        if self.block_mask is None:
            if clean_x is not None:
                if self.independent_first_frame:
                    raise NotImplementedError()
                else:
                    self.block_mask = self._prepare_teacher_forcing_mask(
                        device,
                        num_frames=x.shape[2],
                        frame_seqlen=x.shape[-2]
                        * x.shape[-1]
                        // (self.patch_size[1] * self.patch_size[2]),
                        num_frame_per_block=self.num_frame_per_block,
                        local_attn_size=self.local_attn_size,
                    )
            else:
                if self.independent_first_frame:
                    self.block_mask = self._prepare_blockwise_causal_attn_mask_i2v(
                        device,
                        num_frames=x.shape[2],
                        frame_seqlen=x.shape[-2]
                        * x.shape[-1]
                        // (self.patch_size[1] * self.patch_size[2]),
                        num_frame_per_block=self.num_frame_per_block,
                        local_attn_size=self.local_attn_size,
                    )
                else:
                    self.block_mask = self._prepare_blockwise_causal_attn_mask(
                        device,
                        num_frames=x.shape[2],
                        frame_seqlen=x.shape[-2]
                        * x.shape[-1]
                        // (self.patch_size[1] * self.patch_size[2]),
                        num_frame_per_block=self.num_frame_per_block,
                        local_attn_size=self.local_attn_size,
                    )

        if y is not None:
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

        # embeddings
        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
        grid_sizes = torch.stack(
            [torch.tensor(u.shape[2:], dtype=torch.long) for u in x]
        )
        x = [u.flatten(2).transpose(1, 2) for u in x]

        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        # assert seq_lens.max() <= seq_len
        x = torch.cat(
            [
                torch.cat(
                    [u, u.new_zeros(1, seq_lens[0] - u.size(1), u.size(2))], dim=1
                )
                for u in x
            ]
        )

        # time embeddings
        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, t.flatten()).type_as(x)
        )
        e0 = self.time_projection(e).unflatten(1, (6, self.dim)).unflatten(
            dim=0, sizes=t.shape
        )

        # text context
        context_lens = None
        context = self.text_embedding(
            torch.stack(
                [
                    torch.cat(
                        [u, u.new_zeros(self.text_len - u.size(0), u.size(1))]
                    )
                    for u in context
                ]
            )
        )

        if clip_fea is not None:
            context_clip = self.img_emb(clip_fea)  # bs x 257 x dim
            context = torch.concat([context_clip, context], dim=1)

        if clean_x is not None:
            clean_x = [self.patch_embedding(u.unsqueeze(0)) for u in clean_x]
            clean_x = [u.flatten(2).transpose(1, 2) for u in clean_x]

            seq_lens_clean = torch.tensor(
                [u.size(1) for u in clean_x], dtype=torch.long
            )
            # assert seq_lens_clean.max() <= seq_len
            clean_x = torch.cat(
                [
                    torch.cat(
                        [u, u.new_zeros(1, seq_lens_clean[0] - u.size(1), u.size(2))],
                        dim=1,
                    )
                    for u in clean_x
                ]
            )

            x = torch.cat([clean_x, x], dim=1)
            if aug_t is None:
                aug_t = torch.zeros_like(t)
            e_clean = self.time_embedding(
                sinusoidal_embedding_1d(self.freq_dim, aug_t.flatten()).type_as(x)
            )
            e0_clean = self.time_projection(e_clean).unflatten(
                1, (6, self.dim)
            ).unflatten(dim=0, sizes=t.shape)
            e0 = torch.cat([e0_clean, e0], dim=1)

        # likewise, no model-level action embedding here; combined_action_ids is handed to each block
        kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=self.freqs,
            context=context,
            context_lens=context_lens,
            block_mask=self.block_mask,
            combined_action_ids=combined_action_ids,
            teacher_forcing=(clean_x is not None),
        )

        def create_custom_forward(module):
            def custom_forward(*inputs, **inner_kwargs):
                return module(*inputs, **inner_kwargs)

            return custom_forward

        for block in self.blocks:
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                x = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    x,
                    **kwargs,
                    use_reentrant=False,
                )
            else:
                x = block(x, **kwargs)

        if clean_x is not None:
            x = x[:, x.shape[1] // 2 :]

        x = self.head(x, e.unflatten(dim=0, sizes=t.shape).unsqueeze(2))
        x = self.unpatchify(x, grid_sizes)
        return torch.stack(x)