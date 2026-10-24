"""Z-Image ControlNet lite for two jobs: a better inpaint blend and tile detail for upscaling. GPU code, untested.

Both use diffusers' ZImageControlNetInpaintPipeline, which feeds the ControlNet a 33-channel context:
[16 control latents | 1 mask channel (1 - mask) | 16 latents of the masked source image]. VideoX-Fun (the model's authors)
fill the parts they do not use with ZEROS, not with an encoded black image:

  plain inpaint  control latents = zeros; mask = the painted mask; masked image = encoded as usual
  tile           control latents = the tile to refine; mask = all white (mask channel 0); masked-image latents = zeros

diffusers builds the context inline inside `__call__` with no hook, so the subclass swaps `vae.encode` for the duration
of one call: the Nth encode call returns latents that become exactly zero after the pipeline's
`(latents - shift_factor) * scaling_factor`.

The lite files (2601, 8 steps, bf16, ~1.9 GB) have control_layers 0-2 and control_noise_refiner 0-1, which diffusers 0.40
does not detect on its own, so `zcn_lite/config.json` (control_layers_places [0, 10, 20], control_in_dim 33) is passed
explicitly. The Tile lite header has the identical layout, so the same config serves both files.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

CONFIG_DIR = Path(__file__).parent / "zcn_lite"
_cls = None


def load_controlnet(path: str):
    """ZImageControlNetModel from a lite .safetensors, in fp16 and with the T4 overflow fix."""
    import torch
    from diffusers import ZImageControlNetModel

    from ..zimage_fp16 import patch
    cn = ZImageControlNetModel.from_single_file(str(path), config=str(CONFIG_DIR), torch_dtype=torch.float16)
    patch(cn)
    return cn


def _zero_latents(vae, x):
    """What vae.encode(x) must return so the pipeline's (z - shift) * scale is exactly zero."""
    import torch
    f = 2 ** (len(vae.config.block_out_channels) - 1)
    shape = (x.shape[0], vae.config.latent_channels, x.shape[2] // f, x.shape[3] // f)
    z = torch.full(shape, float(vae.config.shift_factor or 0.0), dtype=x.dtype, device=x.device)
    return SimpleNamespace(latent_dist=SimpleNamespace(mode=lambda: z, sample=lambda generator=None: z), latents=z)


def pipeline_class():
    """ZImageControlNetInpaintPipeline whose __call__ zeroes the chosen vae.encode call(s) (`zero_calls`, 1-based)."""
    global _cls
    if _cls is None:
        from diffusers import ZImageControlNetInpaintPipeline

        class ZImageControlLiteInpaint(ZImageControlNetInpaintPipeline):
            zero_calls: tuple = (1,)  # 1 = the control image (plain inpaint), 2 = the masked source image (tile)

            def __call__(self, *args, **kwargs):
                vae, real, n = self.vae, self.vae.encode, [0]

                def encode(x, *a, **k):
                    n[0] += 1
                    return _zero_latents(vae, x) if n[0] in self.zero_calls else real(x, *a, **k)

                vae.encode = encode  # instance attribute shadows the method for this call only
                try:
                    return super().__call__(*args, **kwargs)
                finally:
                    vae.__dict__.pop("encode", None)

        _cls = ZImageControlLiteInpaint
    return _cls


def make_pipeline(base_pipe, controlnet, zero_calls: tuple):
    """A control pipeline sharing the loaded transformer, text encoder and VAE of a ZImagePipeline."""
    cls = pipeline_class()
    pipe = cls(**base_pipe.components, controlnet=controlnet)
    pipe.zero_calls = zero_calls
    pipe.set_progress_bar_config(disable=True)
    return pipe
