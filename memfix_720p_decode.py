"""Decode-time memory fix for long-sequence 720p inference.

Problem: at the end of a 720p x 96/192-latent rollout the process holds ~76GB
(weights + KV + text encoder); a one-shot VAE decode needs a few extra GB and
OOMs. The fix follows the validated recipe from the i2v decode path: move the
DiT to CPU before decoding to free ~10GB + empty_cache, and use the wrapper's
existing decode_to_pixel_chunk for chunked decoding. In dataset mode one process
produces several videos back to back, so the DiT is moved back to the GPU at the
start of the next inference() call.

Usage: call install() after importing the pipeline and before running main.
Does not modify any existing file.
"""
import torch


def install():
    from pipeline.causal_diffusion_inference_compressed import (
        CausalDiffusionInferenceCompressedPipeline as _P,
    )

    _orig_inf = _P.inference

    def inference(self, *args, **kwargs):
        # The DiT was moved to CPU while decoding the previous video; move it back before the next rollout.
        try:
            if next(self.generator.parameters()).device.type == "cpu":
                self.generator.to("cuda")
                torch.cuda.empty_cache()
        except Exception:
            pass
        # The text encoder (~11GB) is used once at the start of each video and idles otherwise — offload it right after use.
        te = getattr(self, "text_encoder", None)
        if te is not None and not getattr(te, "_memfix_te", False):
            _te_fwd = te.forward
            def _te_offload_fwd(*ta, **tk):
                te.to("cuda")
                out = _te_fwd(*ta, **tk)
                te.to("cpu")
                torch.cuda.empty_cache()
                return out
            te.forward = _te_offload_fwd
            te.to("cpu")
            torch.cuda.empty_cache()
            te._memfix_te = True
        vae = self.vae
        if not getattr(vae, "_memfix_installed", False):
            def _decode_offload_chunk(latent, *a, **k):
                try:
                    self.generator.to("cpu")
                    torch.cuda.empty_cache()
                except Exception:
                    pass
                return vae.decode_to_pixel_chunk(latent, use_cache=False, chunk_size=24)
            vae.decode_to_pixel = _decode_offload_chunk
            vae._memfix_installed = True
        return _orig_inf(self, *args, **kwargs)

    _P.inference = inference
    import os
    if int(os.environ.get("RANK", "0")) == 0:
        print("[memfix] 720p decode: DiT->CPU before VAE + chunked decode (chunk_size=24)", flush=True)
