"""
Inference script for causal diffusion model with camera action control.

Supports two modes:
  1. dataset  -- Uses eval_datasets from config (e.g. SimulatedActionDataset)
                 with precomputed camera trajectories.
  2. interactive -- Blocking keyboard input per chunk for real-time camera control.

Both modes use CausalDiffusionInferencePipeline + ControlSignalManager.

Example usage (dataset mode, single GPU):
    python inference_action.py \
        --config_path configs/plucker720p_dmd_infer.yaml \
        --mode dataset \
        --output_folder outputs/eval_action

Example usage (dataset mode, multi-GPU):
    torchrun --nproc_per_node=4 inference_action.py \
        --config_path configs/plucker720p_dmd_infer.yaml \
        --mode dataset \
        --output_folder outputs/eval_action

Example usage (interactive mode, single GPU only):
    python inference_action.py \
        --config_path configs/plucker720p_dmd_infer.yaml \
        --mode interactive \
        --prompt "A drone flying over a mountain" \
        --num_latent_frames 20 \
        --output_folder outputs/interactive
"""

import argparse
import os

import numpy as np
import torch
import torch.distributed as dist
from einops import rearrange
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from torchvision.io import write_video
from tqdm import tqdm

from pipeline import (
    CausalDiffusionInferenceCompressedPipeline,
    CausalDiffusionInferencePipeline,
)
from utils.control_manager import ControlSignalManager
from utils.visualize import draw_eval_action_overlay, draw_interactive_action_overlay
from utils.camera import parse_key_sequence
from utils.misc import set_seed
from demo_utils.memory import get_cuda_free_memory_gb, DynamicSwapInstaller


def parse_args():
    parser = argparse.ArgumentParser(description="Action-conditioned causal diffusion inference")
    parser.add_argument("--config_path", type=str, default="configs/plucker720p_dmd_infer.yaml")
    parser.add_argument("--checkpoint_path", type=str, default=None,
                        help="Override config's generator_ckpt")
    parser.add_argument("--output_folder", type=str, default="outputs/inference_action")
    parser.add_argument("--mode", type=str, choices=["dataset", "interactive"], default="dataset")
    parser.add_argument("--num_latent_frames", type=int, default=None,
                        help="Override image_or_video_shape[1] (number of latent frames to generate)")
    parser.add_argument("--num_inference_steps", type=int, default=30)
    parser.add_argument("--prompt", type=str, default=None,
                        help="Text prompt (interactive mode)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--inference_local_attn_size", type=int, default=None,
                        help="Override model_kwargs.inference_local_attn_size (KV cache window)")
    parser.add_argument("--dummy_prope", action="store_true",
                        help="Replace prope_context with the first frame repeated T times "
                             "(ablation: kills any camera-pose signal in PRoPE).")
    parser.add_argument("--kv_compression", action="store_true",
                        help="Force-enable inference-time KV compression "
                             "(sink+pose_cons).  Equivalent to setting "
                             "kv_compression.enabled=true in the yaml.  "
                             "Other knobs (epsilon, budget, etc.) must still "
                             "come from the yaml; use --kv_epsilon to override.")
    parser.add_argument("--kv_epsilon", type=float, default=None,
                        help="Override kv_compression.epsilon")
    parser.add_argument("--kv_budget_chunks", type=int, default=None,
                        help="Override kv_compression.budget_chunks")
    parser.add_argument("--kv_n_sink", type=int, default=None,
                        help="Override kv_compression.n_sink")
    parser.add_argument("--kv_policy", type=str, default=None,
                        help="Override kv_compression.policy: merge|window|naive|select")
    parser.add_argument("--kv_recent_w", type=int, default=None,
                        help="Override kv_compression.recent_w (recent full-res window, chunks)")
    parser.add_argument("--kv_pool_ratio", type=int, default=None,
                        help="Override kv_compression.pool_ratio (naive)")
    parser.add_argument("--kv_h2o_keep_frac", type=float, default=None,
                        help="Override kv_compression.h2o_keep_frac (select)")
    parser.add_argument("--kv_landmark_k", type=int, default=None,
                        help="Override kv_compression.landmark_k (v15b bank capacity)")
    parser.add_argument("--kv_retrieve_k", type=int, default=None,
                        help="Override kv_compression.retrieve_k (v15a/b landmarks attended)")
    parser.add_argument("--kv_landmark_eps", type=float, default=None,
                        help="Override kv_compression.landmark_eps (pose dedup threshold)")
    parser.add_argument("--kv_log_path", type=str, default=None,
                        help="Where to dump compression event log (JSON)")
    parser.add_argument("--no_action_overlay", action="store_true",
                        help="Save raw generated frames WITHOUT the key/action overlay")
    parser.add_argument("--pose_noise_sigma", type=float, default=0.0,
                        help="Inject Gaussian N(0, sigma) noise into the translation "
                             "component (c2w[..., :3, 3]) of prope_context before model "
                             "forward.  One noise vector per (batch, frame) shared across "
                             "all chunks within a sample (reproducible via torch.manual_seed("
                             "args.seed)).  Rotation untouched.  Default 0.0 = no noise.")
    parser.add_argument("--lora_checkpoint_path", type=str, default=None,
                        help="Plug-in DMD few-step LoRA (generator_lora). peft-wraps the "
                             "CausalWanActionAttentionBlock Linears and loads the adapter; "
                             "pair with --num_inference_steps 4. rank from config.adapter "
                             "(default 128 if absent).")
    return parser.parse_args()


def load_config(config_path: str):
    config = OmegaConf.load(config_path)
    default_config = OmegaConf.load("configs/default_config.yaml")
    config = OmegaConf.merge(default_config, config)
    return config


def build_models(config, device):
    """Instantiate generator, text_encoder, vae based on model variant."""
    model_name = config.model_kwargs.model_name
    is_5b = "5B" in model_name or "5b" in model_name

    if is_5b:
        from utils.wan_5b_wrapper import (
            WanActionDiffusionWrapper,
            WanTextEncoder,
            WanVAEWrapper,
        )
    else:
        from utils.wan_wrapper import (
            WanActionDiffusionWrapper,
            WanTextEncoder,
            WanVAEWrapper,
        )

    model_kwargs = OmegaConf.to_container(config.model_kwargs, resolve=True)
    generator = WanActionDiffusionWrapper(**model_kwargs, is_causal=True)
    text_encoder = WanTextEncoder()
    vae = WanVAEWrapper()

    return generator, text_encoder, vae


def load_checkpoint(generator, ckpt_path: str):
    """Load generator weights from a checkpoint file."""
    print(f"Loading checkpoint from {ckpt_path}")
    state_dict = torch.load(ckpt_path, map_location="cpu")

    if isinstance(state_dict, dict):
        if "generator_ema" in state_dict:
            generator.load_state_dict(state_dict["generator_ema"], strict=False)
        elif "generator" in state_dict:
            generator.load_state_dict(state_dict["generator"], strict=False)
        else:
            generator.load_state_dict(state_dict, strict=False)
    else:
        generator.load_state_dict(state_dict, strict=False)


def _find_action_lora_targets(transformer):
    """All Linear submodules inside CausalWanActionAttentionBlock (q/k/v/o self+cross
    + ffn.0/2). Excludes the action/control pathway → control preserved."""
    target_modules = set()
    for name, module in transformer.named_modules():
        if module.__class__.__name__ != "CausalWanActionAttentionBlock":
            continue
        for full_name, submodule in module.named_modules(prefix=name):
            if isinstance(submodule, torch.nn.Linear):
                target_modules.add(full_name)
    if not target_modules:
        raise ValueError("No CausalWanActionAttentionBlock Linear layers found for LoRA")
    return sorted(target_modules)


def load_lora_checkpoint(generator, config, lora_ckpt_path: str):
    """Plug the decoupled DMD few-step LoRA (generator_lora) onto the stage1 backbone.
    Mirrors the 0603 loader incl. the peft<->transformers manual-fallback. config.adapter
    optional → defaults rank128/alpha128/dropout0 (matches the published LoRA)."""
    import peft
    adapter_cfg = getattr(config, "adapter", None)
    if adapter_cfg is not None:
        rank = int(adapter_cfg.get("rank", 128))
        alpha = adapter_cfg.get("alpha", None) or rank
        dropout = float(adapter_cfg.get("dropout", 0.0))
    else:
        rank, alpha, dropout = 128, 128, 0.0
    print(f"Loading LoRA checkpoint from {lora_ckpt_path} (rank={rank}, alpha={alpha})")
    target_modules = _find_action_lora_targets(generator.model)
    lora_config = peft.LoraConfig(
        r=rank, lora_alpha=alpha, lora_dropout=dropout, target_modules=target_modules,
    )
    generator.model = peft.get_peft_model(generator.model, lora_config)

    raw_state = torch.load(lora_ckpt_path, map_location="cpu")
    if "generator_lora" not in raw_state:
        raise ValueError(
            f"LoRA checkpoint {lora_ckpt_path} missing generator_lora. "
            f"Found keys: {list(raw_state.keys())}")
    gen_lora = raw_state["generator_lora"]
    try:
        peft.set_peft_model_state_dict(generator.model, gen_lora)
    except ImportError as e:
        # peft<->transformers mismatch (EmbeddingParallel import) → version-independent
        # manual load: insert active adapter name ("default") into lora_A/lora_B keys.
        print(f"[lora] set_peft_model_state_dict failed ({e}); manual fallback load")
        remapped = {
            k.replace(".lora_A.weight", ".lora_A.default.weight")
             .replace(".lora_B.weight", ".lora_B.default.weight"): v
            for k, v in gen_lora.items()
        }
        res = generator.model.load_state_dict(remapped, strict=False)
        if res.unexpected_keys:
            raise RuntimeError(
                f"manual LoRA load: {len(res.unexpected_keys)} unexpected keys "
                f"(adapter-name remap wrong?), e.g. {res.unexpected_keys[:3]}")
        print(f"[lora] manual load OK: {len(remapped)} tensors "
              f"(missing={len(res.missing_keys)} = base+untrained control adapters, expected)")
    print(f"Loaded generator_lora into {len(target_modules)} LoRA target modules")


def run_dataset_mode(args, config, pipeline, device, local_rank, world_size):
    """Eval mode: iterate over eval_datasets, using precomputed trajectories."""
    from dataset import initial_cls_from_config

    shape = list(config.image_or_video_shape)
    latent_channels, latent_h, latent_w = shape[2], shape[3], shape[4]
    num_pixel_frames = (shape[1] - 1) * 4 + 1
    # Wan2.2-TI2V-5B VAE spatial stride 16× (not 8×); see configs/wan_ti2v_5B.py
    pixel_h, pixel_w = latent_h * 16, latent_w * 16
    num_frame_per_block = config.num_frame_per_block

    dataset = initial_cls_from_config(
        config.eval_datasets,
        num_frames=num_pixel_frames,
        video_size=(pixel_h, pixel_w),
    )
    num_prompts = len(dataset)

    if dist.is_initialized():
        sampler = DistributedSampler(dataset, shuffle=False, drop_last=True)
    else:
        sampler = SequentialSampler(dataset)
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        sampler=sampler,
        num_workers=0,
    )
    print(f"[Rank {local_rank}] Eval dataset size: {num_prompts}, "
          f"samples per rank: {len(dataloader)}")

    for i, batch in tqdm(enumerate(dataloader), total=len(dataloader),
                         disable=(local_rank != 0)):
        idx = batch["idx"].item()
        prompt = batch["prompts"][0]
        actions = batch["actions"].to(device=device, dtype=torch.bfloat16)
        prope_ctx = (
            batch["prope_context"].to(device=device, dtype=torch.bfloat16)
            if "prope_context" in batch
            else None
        )
        if args.dummy_prope and prope_ctx is not None:
            # Ablation: replace the whole camera trajectory with "the first frame repeated T times".
            # Shape [B, T, 4, 4] -> take [:, :1] and expand back to T.
            first = prope_ctx[:, :1]
            prope_ctx = first.expand_as(prope_ctx).contiguous()
            if local_rank == 0 and i == 0:
                print(f"[DummyPRoPE] prope_context replaced by repeating first frame "
                      f"(shape={tuple(prope_ctx.shape)})")

        # Pose-noise injection (PRoPE robustness probe).  One noise tensor per
        # (rank, sample) — reproducible across σ sweeps because we re-seed with
        # (args.seed, idx) per sample; σ only scales the same draw.  Noise is
        # added to the translation component only; rotation is untouched.
        if args.pose_noise_sigma > 0.0 and prope_ctx is not None:
            gen = torch.Generator(device="cpu").manual_seed(int(args.seed) * 1_000_003 + int(idx))
            t_noise = torch.randn(prope_ctx[..., :3, 3].shape, generator=gen,
                                  dtype=torch.float32) * float(args.pose_noise_sigma)
            t_noise = t_noise.to(device=prope_ctx.device, dtype=prope_ctx.dtype)
            prope_ctx = prope_ctx.clone()
            prope_ctx[..., :3, 3] = prope_ctx[..., :3, 3] + t_noise
            if local_rank == 0 and i == 0:
                print(f"[PoseNoise] Added N(0, sigma={args.pose_noise_sigma}) to "
                      f"prope_context[..., :3, 3] (shape={tuple(prope_ctx.shape)}, "
                      f"||delta||_mean={t_noise.abs().mean().item():.4f})")
        action_str = batch.get("action_str", [f"action_{i}"])[0]

        num_latent_frames = actions.shape[1]
        assert num_latent_frames % num_frame_per_block == 0, (
            f"num_latent_frames ({num_latent_frames}) must be divisible by "
            f"num_frame_per_block ({num_frame_per_block})"
        )

        ctrl_mgr = ControlSignalManager(
            chunk_size=num_frame_per_block,
            device=str(device),
            dtype=torch.bfloat16,
        )
        ctrl_mgr.set_sequence(actions, prope_context=prope_ctx)

        noise = torch.randn(
            [args.num_samples, num_latent_frames, latent_channels, latent_h, latent_w],
            device=device,
            dtype=torch.bfloat16,
        )
        prompts = [prompt] * args.num_samples

        video, latents = pipeline.inference(
            noise=noise,
            text_prompts=prompts,
            control_manager=ctrl_mgr,
            return_latents=True,
            num_inference_steps=args.num_inference_steps,
        )

        video = rearrange(video, "b t c h w -> b t h w c")
        video = (video * 255.0).clamp(0, 255).cpu().to(torch.uint8)

        if idx < num_prompts:
            for s in range(args.num_samples):
                vid_np = video[s].numpy()

                if getattr(args, "no_action_overlay", False):
                    vid_with_keys = vid_np                      # raw frames, NO key/action overlay
                elif action_str in ("move_forward", "move_backward", "move_left",
                                  "move_right", "look_up", "look_down",
                                  "look_left", "look_right", "static"):
                    vid_with_keys = draw_eval_action_overlay(vid_np, action_str)
                else:
                    num_latent = actions.shape[1]
                    mouse_seq, kb_seq = parse_key_sequence(action_str, num_latent)
                    vid_with_keys = draw_interactive_action_overlay(
                        vid_np, mouse_seq, kb_seq)

                # INDEX-based filename (no key string): long keys (24-96 chars) made
                # unreadable filenames. downstream tools can read the TRUE key from
                # prompts_k*.txt by this index, so the index prefix is all that's needed.
                output_path = os.path.join(
                    args.output_folder,
                    f"{idx:03d}_s{s}.mp4",
                )
                write_video(output_path, torch.from_numpy(vid_with_keys), fps=24)
                if local_rank == 0:
                    print(f"Saved: {output_path}")

        pipeline.clear_cache()
        pipeline.vae.model.clear_cache()


def run_interactive_mode(args, config, pipeline, device):
    """Interactive mode: blocking keyboard input for camera control per chunk."""
    from utils.visualize import export_to_video, process_video, visualize_trajectory
    import copy
    from demo_utils.constant import GET_ZERO_VAE_CACHE

    shape = list(config.image_or_video_shape)
    latent_channels, latent_h, latent_w = shape[2], shape[3], shape[4]
    num_frame_per_block = config.num_frame_per_block
    num_output_frames = shape[1]

    assert num_output_frames % num_frame_per_block == 0, (
        f"num_output_frames ({num_output_frames}) must be divisible by "
        f"num_frame_per_block ({num_frame_per_block})"
    )

    prompt = args.prompt or "A scenic landscape with moving camera"
    print(f"Prompt: {prompt}")
    print(f"Generating {num_output_frames} latent frames ({(num_output_frames - 1) * 4 + 1} pixel frames)")
    print(f"Blocks: {num_output_frames // num_frame_per_block}, each {num_frame_per_block} frames")

    target_action_mode = "delta_euler"
    eval_ds_cfg = getattr(config, "eval_datasets", None)
    if eval_ds_cfg is not None:
        eval_params = OmegaConf.to_container(eval_ds_cfg, resolve=True)
        if isinstance(eval_params, list) and len(eval_params) > 0:
            target_action_mode = eval_params[0].get("params", {}).get(
                "target_action_mode", "delta_euler"
            )

    delta_t = 0.15
    delta_r = 0.26
    prope_fixed_divisor = None
    if eval_ds_cfg is not None:
        eval_params = OmegaConf.to_container(eval_ds_cfg, resolve=True)
        if isinstance(eval_params, list) and len(eval_params) > 0:
            p = eval_params[0].get("params", {})
            delta_t = p.get("delta_t", delta_t)
            delta_r = p.get("delta_r", delta_r)
            prope_fixed_divisor = p.get("prope_fixed_divisor", prope_fixed_divisor)

    ctrl_mgr = ControlSignalManager(
        action_mode=target_action_mode,
        chunk_size=num_frame_per_block,
        device=str(device),
        dtype=torch.bfloat16,
        delta_t=delta_t,
        delta_r=delta_r,
        prope_fixed_divisor=prope_fixed_divisor,
    )

    noise = torch.randn(
        [args.num_samples, num_output_frames, latent_channels, latent_h, latent_w],
        device=device,
        dtype=torch.bfloat16,
    )
    prompts = [prompt] * args.num_samples

    video, latents = pipeline.inference(
        noise=noise,
        text_prompts=prompts,
        control_manager=ctrl_mgr,
        return_latents=True,
        num_inference_steps=args.num_inference_steps,
    )

    video = rearrange(video, "b t c h w -> b t h w c")
    video = (video * 255.0).clamp(0, 255).cpu().to(torch.uint8)

    for s in range(args.num_samples):
        vid_np = video[s].numpy()
        vid_with_keys = draw_interactive_action_overlay(
            vid_np,
            ctrl_mgr.mouse_action_history,
            ctrl_mgr.keyboard_action_history,
        )
        output_path = os.path.join(args.output_folder, f"interactive_s{s}.mp4")
        write_video(output_path, torch.from_numpy(vid_with_keys), fps=24)
        print(f"Saved: {output_path}")

    traj_path = os.path.join(args.output_folder, "trajectory.html")
    visualize_trajectory(ctrl_mgr.c2w_history, traj_path)
    print(f"Trajectory saved: {traj_path}")

    pipeline.clear_cache()
    pipeline.vae.model.clear_cache()


def main():
    args = parse_args()
    torch.set_grad_enabled(False)

    # --- Distributed init ---
    if "LOCAL_RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        world_size = dist.get_world_size()
        set_seed(args.seed + local_rank)
    else:
        local_rank = 0
        world_size = 1
        device = torch.device("cuda")
        set_seed(args.seed)

    config = load_config(args.config_path)

    # --- CLI overrides ---
    if args.num_latent_frames is not None:
        shape = list(config.image_or_video_shape)
        shape[1] = args.num_latent_frames
        config.image_or_video_shape = shape
        if local_rank == 0:
            print(f"Overriding image_or_video_shape[1] -> {args.num_latent_frames}")
    if args.inference_local_attn_size is not None:
        config.model_kwargs.inference_local_attn_size = args.inference_local_attn_size
        if local_rank == 0:
            print(f"Overriding model_kwargs.inference_local_attn_size -> {args.inference_local_attn_size}")

    # --- KV compression overrides (CLI > yaml) ---
    if args.kv_compression or args.kv_epsilon is not None or args.kv_budget_chunks is not None \
            or args.kv_n_sink is not None or args.kv_log_path is not None \
            or args.kv_policy is not None or args.kv_recent_w is not None \
            or args.kv_pool_ratio is not None or args.kv_h2o_keep_frac is not None \
            or args.pose_noise_sigma > 0.0:
        if "kv_compression" not in config:
            config.kv_compression = OmegaConf.create({})
        if args.kv_compression:
            config.kv_compression.enabled = True
        if args.kv_policy is not None:
            config.kv_compression.policy = args.kv_policy
            config.kv_compression.enabled = True
        if args.kv_recent_w is not None:
            config.kv_compression.recent_w = args.kv_recent_w
        if args.kv_pool_ratio is not None:
            config.kv_compression.pool_ratio = args.kv_pool_ratio
        if args.kv_h2o_keep_frac is not None:
            config.kv_compression.h2o_keep_frac = args.kv_h2o_keep_frac
        if args.kv_landmark_k is not None:
            config.kv_compression.landmark_k = args.kv_landmark_k
        if args.kv_retrieve_k is not None:
            config.kv_compression.retrieve_k = args.kv_retrieve_k
        if args.kv_landmark_eps is not None:
            config.kv_compression.landmark_eps = args.kv_landmark_eps
        if args.kv_epsilon is not None:
            config.kv_compression.epsilon = args.kv_epsilon
        if args.kv_budget_chunks is not None:
            config.kv_compression.budget_chunks = args.kv_budget_chunks
        if args.kv_n_sink is not None:
            config.kv_compression.n_sink = args.kv_n_sink
        if args.kv_log_path is not None:
            config.kv_compression.log_path = args.kv_log_path
        if args.pose_noise_sigma > 0.0:
            config.kv_compression.pose_noise_sigma = args.pose_noise_sigma
        if local_rank == 0:
            print(f"[KVCompression] CLI overrides applied: "
                  f"{OmegaConf.to_container(config.kv_compression, resolve=True)}")

    print(f"[Rank {local_rank}] Free VRAM {get_cuda_free_memory_gb(device)} GB")
    low_memory = get_cuda_free_memory_gb(device) < 40

    generator, text_encoder, vae = build_models(config, device)

    ckpt_path = args.checkpoint_path or getattr(config, "generator_ckpt", None)
    if ckpt_path:
        load_checkpoint(generator, ckpt_path)

    # Plug-in DMD few-step LoRA (decoupled, 0-cost) — applied AFTER base weights,
    # BEFORE the pipeline wraps the generator. Pair with --num_inference_steps 4.
    lora_ckpt_path = args.lora_checkpoint_path or getattr(config, "lora_ckpt", None)
    if lora_ckpt_path:
        load_lora_checkpoint(generator, config, lora_ckpt_path)

    # Pick pipeline class based on whether KV compression is enabled.  When
    # ``kv_compression.enabled: true`` (or --kv_compression), use the
    # compressed subclass; otherwise the base path with zero overhead.
    kvc_enabled = bool(
        getattr(getattr(config, "kv_compression", None), "enabled", False)
        if hasattr(config, "kv_compression") else False
    )
    pipeline_cls = (
        CausalDiffusionInferenceCompressedPipeline if kvc_enabled
        else CausalDiffusionInferencePipeline
    )
    if local_rank == 0:
        print(f"[Pipeline] Using {pipeline_cls.__name__} (kvc_enabled={kvc_enabled})")
    pipeline = pipeline_cls(
        args=config,
        device=device,
        generator=generator,
        text_encoder=text_encoder,
        vae=vae,
    )
    pipeline = pipeline.to(dtype=torch.bfloat16)

    if low_memory:
        DynamicSwapInstaller.install_model(pipeline.text_encoder, device=device)
    else:
        pipeline.text_encoder.to(device=device)
    pipeline.generator.to(device=device)
    pipeline.vae.to(device=device)

    # Only rank 0 creates output dir to avoid race conditions
    if local_rank == 0:
        os.makedirs(args.output_folder, exist_ok=True)
    if dist.is_initialized():
        dist.barrier()

    if args.mode == "dataset":
        run_dataset_mode(args, config, pipeline, device, local_rank, world_size)
    else:
        assert not dist.is_initialized(), (
            "Interactive mode does not support distributed inference. "
            "Use single GPU: python inference_action.py --mode interactive ..."
        )
        run_interactive_mode(args, config, pipeline, device)

    if dist.is_initialized():
        dist.barrier()
    if local_rank == 0:
        print("Done!")


if __name__ == "__main__":
    main()
