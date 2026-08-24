# ReWorld Inference Guide

This page is the full reference for the inference entry points: trajectory input formats, memory (KV-cache) policies, every CLI flag, and the relevant environment variables. For a quick start, see the [README](../README.md#inference).

> **Note** — pretrained ReWorld checkpoints are not released yet. Commands below become runnable once `checkpoints/reworld_generator_ema.pt` and `checkpoints/reworld_dmd_lora.pt` are published.

## Entry points

| Script | Conditioning | Modes | Notes |
|---|---|---|---|
| `inference_i2v_v2.py` | start image + text + actions | `dataset` | Recommended i2v entry. Adds the v2 landmark bank and chunked 720p VAE decode. Single-process runs. |
| `inference_action_v2.py` | text + actions | `dataset`, `interactive` | Recommended t2w entry. Adds the v2 landmark bank and the 720p decode memory fix. `torchrun` OK in dataset mode. |
| `inference_i2v.py` / `inference_action.py` | same | same | Plain variants; identical CLI. Use only if you explicitly want the v1 landmark cache. |

The `_v2` wrappers accept **exactly the same arguments** as their plain counterparts — they swap `LandmarkCache` for `LandmarkCacheV2` (always-full storage, protect-2 + most-redundant eviction, top-k nearest-pose retrieval, pinned-host tiered KV) before the pipeline is imported, then run the plain script unchanged.

## Configuration

`--config_path` (default `configs/plucker720p_dmd_infer.yaml`) is merged on top of `configs/default_config.yaml`. The fields you are most likely to touch:

| Config field | Meaning |
|---|---|
| `model_kwargs.model_name` | Path to the Wan2.2-TI2V-5B base directory (or set `WAN_MODEL_DIR`) |
| `generator_ckpt` / `lora_ckpt` | ReWorld generator EMA / 4-step DMD LoRA paths |
| `image_or_video_shape` | `[batch, latent_T, C, latent_H, latent_W]` — `[1, 96, 48, 44, 80]` is 704×1280 (VAE stride 16), 96 latent frames = 381 pixel frames ≈ 16 s at 24 fps |
| `num_frame_per_block` | Latent frames per autoregressive chunk (4) |
| `eval_datasets[0].params.data_dir` | Rollout list for dataset mode (see below) |
| `eval_datasets[0].params.delta_t` / `delta_r` | Per-chunk translation / rotation step of the keyboard motion model |

Running **without** the LoRA: remove `lora_ckpt` from the config (and do not pass `--lora_checkpoint_path`), then sample with `--num_inference_steps 24`.

## Dataset mode: scripted trajectories

`--mode dataset` iterates over a plain-text rollout list (one rollout per line):

```
<text prompt>@<key sequence>
```

Example (from `assets/mc_eval_random_keys_96latent_more7.txt`):

```
Cinematic Minecraft world aesthetic, ...@dlwssasjdaswdsjlajjasllw
```

Each **character** of the key sequence controls one 4-latent-frame chunk:

| Key | Action |
|---|---|
| `w` / `s` | move forward / backward |
| `a` / `d` | move left / right |
| `i` / `k` | look up / down |
| `j` / `l` | look left / right |
| `q`, `u`, other | stay |

Sequences shorter than the rollout are padded with "stay"; longer ones are truncated. The keys are converted into a metric camera trajectory (`delta_t` per translation step, `delta_r` per rotation step) that drives both the action conditioning and the pose (PRoPE) context.

For image-conditioned rollouts, add `--init_image path/to/image.png` (`inference_i2v*` only): the image is resized to the target resolution, VAE-encoded, and used as the first latent, so every rollout in the list starts from that frame.

Outputs land in `--output_folder` as `{idx:03d}_s{sample}.mp4` (24 fps). The i2v scripts additionally save raw latents to `<output_folder>/latents/` and decode in chunks to keep 720p VRAM bounded.

Multi-GPU (dataset mode only; rollouts are sharded across ranks):

```bash
torchrun --nproc_per_node=4 inference_action_v2.py \
    --config_path configs/plucker720p_dmd_infer.yaml \
    --mode dataset \
    --num_inference_steps 4 \
    --output_folder outputs/eval_action
```

## Interactive mode: live keyboard control

`--mode interactive` (single GPU only, `inference_action*`) blocks before each chunk and asks for two keys in the terminal:

```
Please input the mouse action (e.g. `U`):
Please input the keyboard action (e.g. `W`):
```

| Input | Keys |
|---|---|
| Mouse (look) | `i` up · `k` down · `j` left · `l` right · `u` none |
| Keyboard (move) | `w` forward · `s` back · `a` left · `d` right · `space` up · `ctrl` down · `q` none |

Each pair of inputs is applied for one 4-latent-frame chunk with a constant-velocity motion model. Outputs: `interactive_s{sample}.mp4` with a key-press HUD overlay, plus `trajectory.html` (3D plot of the camera path).

```bash
python inference_action_v2.py \
    --config_path configs/plucker720p_dmd_infer.yaml \
    --mode interactive \
    --prompt "A cinematic Minecraft village at sunset" \
    --num_inference_steps 4 \
    --output_folder outputs/interactive
```

## Memory policies (`--kv_policy`)

By default (no `--kv_policy`, no `kv_compression` block in the yaml) the pipeline keeps the **full KV cache** — memory grows linearly with rollout length. Passing `--kv_policy` selects a bounded-memory pipeline:

| Policy | What it keeps | Extra knobs |
|---|---|---|
| `v15b` **(recommended)** | sink + recent window + top-k landmarks retrieved by camera-pose distance from a larger bank | `--kv_landmark_k` (bank capacity, 30), `--kv_landmark_eps` (pose dedup threshold) |
| `v15a` | sink + recent window + a small landmark bank that is attended in full (no retrieval) | `--kv_retrieve_k` (bank cap, 6) |
| `window` | sink + sliding window of the most recent chunks | — |
| `naive` | sink + recent window + temporally pooled older chunks | `--kv_pool_ratio` |
| `select` | sink + recent window + per-token importance selection | `--kv_h2o_keep_frac` |

**Budget accounting** — you set the total budget and the split auto-fills:

```
budget_chunks = n_sink + recent_w (cap) + landmark slots (remainder)
```

`recent_w` is capped so at least one landmark slot survives; extra budget buys more landmarks, not more recent frames. Examples (printed at startup as `[KVCompression] ... = sink S + recent R + landmark L`):

| `--kv_budget_chunks` | `--kv_n_sink` | `--kv_recent_w` | Split |
|---|---|---|---|
| 12 | 1 | 5 | sink 1 + recent 5 + **6 landmarks** |
| 24 | 1 | 7 | sink 1 + recent 7 + **16 landmarks** |

For `v15b`, the landmarks attended per step are retrieved from a bank of `--kv_landmark_k` (default 30) distinct poses; chunks closer than `--kv_landmark_eps` in pose space are deduplicated on admission.

`--kv_compression` (boolean) enables whatever `kv_compression.*` block the yaml carries without overriding anything; `--kv_policy` implies enablement on its own. `--kv_log_path out.json` dumps a per-chunk admission/eviction/retrieval event log.

## Full flag reference

Flags are shared between `inference_action*.py` and `inference_i2v*.py` unless marked. For the `--kv_*` knobs the CLI default is "unset"; the values shown are the effective defaults that apply when the flag (and the yaml) leave them untouched. Pass the `--kv_landmark_*` / `--kv_retrieve_k` knobs together with `--kv_policy` (alone they select nothing).

| Flag | Default | Description |
|---|---|---|
| `--config_path` | `configs/plucker720p_dmd_infer.yaml` | YAML config, merged over `configs/default_config.yaml` |
| `--checkpoint_path` | `None` | Override `generator_ckpt` (EMA weights preferred if present) |
| `--lora_checkpoint_path` | `None` | Override `lora_ckpt`; 4-step DMD LoRA, pair with `--num_inference_steps 4` |
| `--output_folder` | `outputs/inference_action` | Output directory |
| `--mode` | `dataset` | `dataset` or `interactive` |
| `--num_latent_frames` | config (96) | Override rollout length (must be divisible by `num_frame_per_block`) |
| `--num_inference_steps` | `30` | Denoising steps per chunk: 4 with the LoRA, ~24 without |
| `--prompt` | `None` | Text prompt (interactive mode) |
| `--init_image` | `None` | *(i2v only)* start image for image-conditioned rollout |
| `--seed` | `0` | Random seed (offset by rank in distributed runs) |
| `--num_samples` | `1` | Samples per rollout |
| `--no_overlay` | off | *(i2v only)* save clean frames without the key HUD |
| `--no_action_overlay` | off | *(action only)* save clean frames without the key HUD |
| `--inference_local_attn_size` | config (`-1`) | Override the attention window (chunks); `-1` = unrestricted |
| `--kv_policy` | `None` | Bounded-memory policy, see [Memory policies](#memory-policies---kv_policy) |
| `--kv_budget_chunks` | `20` | Total KV budget in chunks |
| `--kv_n_sink` | `4` | Sink chunks kept forever (the first chunks of the rollout) |
| `--kv_recent_w` | `7` | Cap on the recent full-resolution window (chunks) |
| `--kv_landmark_k` | `30` | `v15b`: landmark-bank capacity |
| `--kv_retrieve_k` | `6` | `v15a`: bank cap; top-k variants: landmarks attended |
| `--kv_landmark_eps` | `0.5` | Pose-distance threshold for landmark deduplication |
| `--kv_pool_ratio` | `4` | `naive`: temporal pooling ratio for aged-out chunks |
| `--kv_h2o_keep_frac` | `0.25` | `select`: fraction of tokens kept per aged-out chunk |
| `--kv_epsilon` | `0.5` | Pose-consolidation threshold (merge-style policies) |
| `--kv_compression` | off | Enable the yaml's `kv_compression` block as-is |
| `--kv_log_path` | `None` | JSON event log of compression decisions |
| `--dummy_prope` | off | Ablation: replace the pose context with the first frame repeated |
| `--pose_noise_sigma` | `0.0` | Robustness probe: Gaussian noise on the pose translation |

## Environment variables

| Variable | Effect |
|---|---|
| `WAN_MODEL_DIR` | Wan2.2-TI2V-5B base directory (T5, tokenizer, VAE); overrides the `./checkpoints/Wan2.2-TI2V-5B` default |
| `BANKV2_HOT_GPU` | Hot-cache size (chunks) the v2 bank keeps on GPU; `0` keeps the whole bank in pinned CPU memory — recommended when 720p VRAM is tight |
| `BANKV2_GPU_SPARE` | GPU slots the v2 bank leaves spare when auto-sizing its hot cache |
| `FORCE_LOW_MEM=1` | *(i2v only)* force the low-VRAM path (text-encoder swapping) even on large GPUs |

## Troubleshooting

- **OOM during VAE decode at 720p** — use the `_v2` entries (chunked decode + DiT offload are built in); the plain `inference_action.py` lacks the decode fix at 720p.
- **OOM during generation on long rollouts** — that is what the bounded policies are for: add `--kv_policy v15b --kv_budget_chunks 12 --kv_n_sink 1 --kv_recent_w 5`.
- **Blurry or drifting results at 4 steps** — verify the LoRA actually loaded: the startup log must show `Loaded generator_lora into N LoRA target modules`; zero-coverage LoRA loads abort with an explicit error on the i2v path.
- **Interactive mode under `torchrun`** — not supported; it asserts single-process. Run with plain `python`.
