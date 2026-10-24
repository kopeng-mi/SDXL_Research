"""Small pure decisions the torch engine makes (no torch here, so they are unit-tested locally).

  * z_embed_key      - what a Z-Image prompt embedding depends on
  * reuse_plan       - which loaded Z-Image parts survive a model change
  * needs_tiling     - VAE tiling only for big decodes
  * tile_starts      - the upscale tile plan along one side
"""

from __future__ import annotations

from .. import loraformat

TILING_MP = 1.5  # decodes over this many megapixels use the tiled VAE


def z_embed_key(prompt: str, negative: str, cfg: float, te_id: tuple, loras: list[tuple[str, float]]) -> tuple:
    """Cache key of a Z-Image prompt embedding: the prompt, whether guidance is on and its negative (diffusers 0.40
    Z-Image pipelines turn guidance on for any cfg > 0, and then need negative embeds), the text encoder and the
    LoRAs that touch the text encoder (the usual DiT-only LoRA does not change the embedding)."""
    te_loras = tuple(sorted((p, round(float(w), 4)) for p, w in loras
                            if p and w and loraformat.inspect(p).has_te))
    cfg_on = cfg > 0
    return (prompt, cfg_on, negative if cfg_on else "", tuple(te_id), te_loras)


def reuse_plan(old_key: str | None, transformer: str, text_encoder: str, te_quant: str) -> tuple[bool, bool]:
    """(keep_transformer, keep_text_encoder) given the loaded model key (`zimage|transformer|encoder|quant`)."""
    if not old_key or not old_key.startswith("zimage|"):
        return False, False
    parts = old_key.split("|")
    if len(parts) != 4:
        return False, False
    _, tr, te, q = parts
    return tr == transformer, (te, q) == (text_encoder, te_quant)


def needs_tiling(width: int, height: int) -> bool:
    return width * height > TILING_MP * 1e6


def tile_starts(length: int, max_tile: int, overlap: int, multiple: int) -> tuple[list[int], int]:
    """Upscale tiles along one side: (start offsets, tile size). The fewest tiles of at most max_tile px that
    overlap by at least `overlap`, all the same size (a multiple of `multiple`) and spread evenly. Fewer, evenly
    sized tiles were the T4 win: 1632 px took 2 tiles of 880 instead of a 1024 + a 608-offset one."""
    if length <= max_tile:
        return [0], length
    n = -(-(length - overlap) // (max_tile - overlap))  # ceil
    size = -(-(length + (n - 1) * overlap) // n)
    size = min(max_tile, -(-size // multiple) * multiple, length)
    last = length - size
    starts = sorted({round(i * last / (n - 1)) // multiple * multiple for i in range(n - 1)} | {last})
    return starts, size

