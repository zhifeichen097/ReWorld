# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import torch
import torch.distributed as dist
import math
from ..modules.attention import flash_attention
from .util import all_to_all, all_to_all_with_grad, get_world_size
from torch.nn.attention.flex_attention import flex_attention
from utils.debug_option import DEBUG

# NOTE: mode="max-autotune-no-cudagraphs" triggers a 29-config triton autotune whose
# first compile per shape takes ~800s — longer than the NCCL watchdog (600s default),
# guaranteeing a crash. The default compile mode (~10s) is slightly slower but stable.
# For maximum performance, switch back to max-autotune AND raise the NCCL timeout to
# 2h+ (see distributed/util.py).
flex_attention = torch.compile(flex_attention, dynamic=False)

def distributed_attention(
        q,
        k,
        v,
        seq_lens,
        window_size=(-1, -1),
):
    """
    Performs distributed attention based on DeepSpeed Ulysses attention mechanism.
    please refer to https://arxiv.org/pdf/2309.14509

    Args:
        q:           [B, Lq // p, Nq, C1].
        k:           [B, Lk // p, Nk, C1].
        v:           [B, Lk // p, Nk, C2]. Nq must be divisible by Nk.
        seq_lens:    [B], length of each sequence in batch
        window_size: (left right). If not (-1, -1), apply sliding window local attention.
    """
    if not dist.is_initialized():
        raise ValueError("distributed group should be initialized.")
    b = q.shape[0]

    # gather q/k/v sequence
    q = all_to_all(q, scatter_dim=2, gather_dim=1)
    k = all_to_all(k, scatter_dim=2, gather_dim=1)
    v = all_to_all(v, scatter_dim=2, gather_dim=1)

    # apply attention
    x = flash_attention(
        q,
        k,
        v,
        k_lens=seq_lens,
        window_size=window_size,
    )

    # scatter q/k/v sequence
    x = all_to_all(x, scatter_dim=1, gather_dim=2)
    return x





def distributed_flex_attention(
    roped_q: torch.Tensor,
    roped_k: torch.Tensor,
    v: torch.Tensor,
    block_mask,
    pad_multiple: int = 128,
    is_tf: bool = False,
):
    """
    roped_q, roped_k, v: [B, L, N, D]
    Returns x: [B, L, N, D]; internally pads to a multiple of pad_multiple before calling flex_attention.
    """
    if DEBUG:
        print("distributed_flex_attention is called", is_tf)
    # gather q/k/v sequence
    roped_q = all_to_all_with_grad(roped_q, scatter_dim=2, gather_dim=1)
    roped_k = all_to_all_with_grad(roped_k, scatter_dim=2, gather_dim=1)
    v = all_to_all_with_grad(v, scatter_dim=2, gather_dim=1)

    B, L_global, N_local, D = roped_q.shape
    if is_tf:
        """
        [r0_clean, r0_noisy, r1_clean, r1_noisy] -> [r0_clean, r1_clean, r0_noisy, r1_noisy]
        """
        world_size = get_world_size()  # in-SP-group size (sp_size), consistent with all_to_all

        assert L_global % (2 * world_size) == 0, \
            f"L_global ({L_global}) must be divisible by 2 * world_size ({2 * world_size})"
        sp_chunk_len = L_global // (2 * world_size)  # per-rank clean/noisy sub-chunk length
        # [B, L_global, N, D] -> [B, world_size, 2, chunk_len, N, D]
        roped_q = roped_q.view(B, world_size, 2, sp_chunk_len, N_local, D)
        roped_k = roped_k.view(B, world_size, 2, sp_chunk_len, N_local, D)
        v       = v.view(B, world_size, 2, sp_chunk_len, N_local, D)
        # concatenate all clean chunks first, then all noisy chunks
        q_clean = roped_q[:, :, 0].reshape(B, world_size * sp_chunk_len, N_local, D)
        q_noisy = roped_q[:, :, 1].reshape(B, world_size * sp_chunk_len, N_local, D)
        k_clean = roped_k[:, :, 0].reshape(B, world_size * sp_chunk_len, N_local, D)
        k_noisy = roped_k[:, :, 1].reshape(B, world_size * sp_chunk_len, N_local, D)
        v_clean = v[:, :, 0].reshape(B, world_size * sp_chunk_len, N_local, D)
        v_noisy = v[:, :, 1].reshape(B, world_size * sp_chunk_len, N_local, D)
        roped_q = torch.cat([q_clean, q_noisy], dim=1)  # [B, 2*S, N_local, D]
        roped_k = torch.cat([k_clean, k_noisy], dim=1)
        v       = torch.cat([v_clean, v_noisy], dim=1)
        # the order changed, but the total length is still L_global
        L_global = roped_q.size(1)


    # pad q/k/v to a multiple of 128, aligned with create_block_mask
    padded_length = math.ceil(L_global / pad_multiple) * pad_multiple - L_global
    if padded_length > 0:
        pad_q = torch.zeros(B, padded_length, N_local, D, device=roped_q.device, dtype=roped_q.dtype)
        pad_k = torch.zeros(B, padded_length, N_local, D, device=roped_k.device, dtype=roped_k.dtype)
        pad_v = torch.zeros(B, padded_length, N_local, D, device=v.device, dtype=v.dtype)
        roped_q = torch.cat([roped_q, pad_q], dim=1)
        roped_k = torch.cat([roped_k, pad_k], dim=1)
        v       = torch.cat([v,       pad_v], dim=1)

    # flex_attention expects [B, N, L, D]
    # grouped-mask aware (BLOCKER-1): plain BlockMask → identical single call;
    # PerHeadGroupedBlockMask (mixed per-head windows + chunk_drop) → one call per
    # head group on that group's LOCAL-head slice (mask was built from the
    # rank-sliced window list, BLOCKER-2), stitched back in head order.
    from ..modules.causal_model import run_flex_attention_with_grouped_mask
    x = run_flex_attention_with_grouped_mask(
        flex_attention,
        roped_q.transpose(2, 1),
        roped_k.transpose(2, 1),
        v.transpose(2, 1),
        block_mask,
    )

    # strip the padding
    if padded_length > 0:
        x = x[:, :, :-padded_length]
    x = x.transpose(2, 1)

    if is_tf:
        x = x.view(B, 2, world_size, sp_chunk_len, N_local, D)
        x = x.permute(0, 2, 1, 3, 4, 5).reshape(B, L_global, N_local, D)

    x = all_to_all_with_grad(x, scatter_dim=1, gather_dim=2)

    return x
