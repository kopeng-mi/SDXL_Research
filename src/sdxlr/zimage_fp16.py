"""Make Z-Image run in fp16 on Turing GPUs (T4) without black images.

The T4 has no fast bf16, and plain fp16 overflows inside the transformer
blocks (NaN latents -> black image; Tongyi-MAI/Z-Image#14). fp32 works but is
~5x slower. This applies the fix used by ComfyUI (Lumina 2) and musubi-tuner
8c36802, via hooks so diffusers itself stays untouched:

  * feed_forward.w3 output / 32 and attention output / 4: both feed an RMSNorm
    (attention_norm2 / ffn_norm2), which is scale-invariant, so the result is
    unchanged but values stay inside fp16 range
  * clamp inf/nan before w2 and before those norms
"""

from __future__ import annotations

import torch

FP16_MAX = 65504.0


def _clamp(x):
    if isinstance(x, torch.Tensor) and x.dtype == torch.float16:
        return torch.nan_to_num(x, nan=0.0, posinf=FP16_MAX, neginf=-FP16_MAX)
    return x


def _scale(div: float):
    def hook(_mod, _inp, out):
        if isinstance(out, torch.Tensor) and out.dtype == torch.float16:
            return out / div
        return out
    return hook


def _clamp_input(_mod, args):
    return (_clamp(args[0]), *args[1:])


def patch(transformer) -> int:
    """Register the hooks on every ZImageTransformerBlock; returns blocks patched."""
    from diffusers.models.transformers.transformer_z_image import ZImageTransformerBlock

    n = 0
    for block in transformer.modules():
        if not isinstance(block, ZImageTransformerBlock):
            continue
        block.feed_forward.w3.register_forward_hook(_scale(32))
        block.feed_forward.w2.register_forward_pre_hook(_clamp_input)
        block.attention.register_forward_hook(_scale(4))
        block.attention_norm2.register_forward_pre_hook(_clamp_input)
        block.ffn_norm2.register_forward_pre_hook(_clamp_input)
        n += 1
    return n
