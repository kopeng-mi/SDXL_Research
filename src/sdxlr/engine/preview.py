"""Live step previews with the tiny autoencoders (TAEF1 for Z-Image, TAESDXL for SDXL). GPU code, untested.

Every 2nd denoising step the engine hands the latents here; they are decoded by a ~5 MB tiny VAE and
written as a JPEG (q70, longest side 512) for the page to show under the progress bar. Cheap on
purpose: while the previous preview is still being written the next one is skipped, a missing tiny VAE
turns previews off (never a download in the middle of a generation: prefetch fetches it), and any
error turns previews off for this model.

Latent layout (diffusers 0.40 ZImagePipeline): [B, 16, H/8, W/8], unpacked; the pipeline decodes with
`latents / vae.config.scaling_factor + vae.config.shift_factor` (0.3611 and 0.1159 for the Flux VAE), the
same formula the tiny VAE takes. SDXL latents are scaled by 0.13025 only.
"""

from __future__ import annotations

import os
import sys
import threading
from pathlib import Path

from ..paths import PREVIEW_VAES

MAX_SIDE = 512


class Previewer:
    def __init__(self) -> None:
        self.vae = None
        self.kind: str | None = None
        self.failed = False
        self.busy = False

    def _load(self, kind: str):
        import torch
        from diffusers import AutoencoderTiny
        try:  # local only: a missing tiny VAE must not start a download mid-generation
            vae = AutoencoderTiny.from_pretrained(PREVIEW_VAES[kind], torch_dtype=torch.float16, local_files_only=True)
            return vae.to("cuda").eval()
        except Exception as e:
            print(f"engine worker: no live previews ({type(e).__name__}); run the prefetch for {PREVIEW_VAES[kind]}",
                  file=sys.stderr, flush=True)
            return None

    def make(self, kind: str, pipe, latents, path: Path, done) -> None:
        """Decode `latents` (first image of the batch) and save it to `path` in a thread; done(path) when written."""
        if self.busy or self.failed:
            return
        try:
            import torch
            if self.vae is None or self.kind != kind:
                self.vae, self.kind = self._load(kind), kind
                if self.vae is None:
                    self.failed = True
                    return
            cfg = pipe.vae.config
            with torch.no_grad():
                x = latents[:1].to("cuda", torch.float16) / cfg.scaling_factor
                if kind == "zimage":
                    x = x + (cfg.shift_factor or 0.0)
                img = self.vae.decode(x).sample  # [-1, 1]
                arr = img[0].float().mul(0.5).add(0.5).clamp(0, 1).mul(255).byte().permute(1, 2, 0).cpu().numpy()
        except Exception as e:
            self.failed = True
            print(f"engine worker: previews off ({type(e).__name__}: {e})", file=sys.stderr, flush=True)
            return
        self.busy = True
        threading.Thread(target=self._save, args=(arr, path, done), daemon=True).start()

    def _save(self, arr, path: Path, done) -> None:
        try:
            from PIL import Image
            im = Image.fromarray(arr)
            im.thumbnail((MAX_SIDE, MAX_SIDE))
            tmp = path.with_name(path.name + ".part")
            path.parent.mkdir(parents=True, exist_ok=True)
            im.save(tmp, "JPEG", quality=70)
            os.replace(tmp, path)  # the page never sees a half-written file
            done(path)
        except Exception as e:
            print(f"engine worker: preview not saved ({e})", file=sys.stderr, flush=True)
        finally:
            self.busy = False
