"""Small pure decisions the torch engine makes (no torch here, so they are unit-tested locally).

  * z_embed_key      - what a Z-Image prompt embedding depends on
  * reuse_plan       - which loaded Z-Image parts survive a model change
  * needs_tiling     - VAE tiling only for big decodes
"""

from __future__ import annotations

from .. import loraformat

TILING_MP = 1.5  # decodes over this many megapixels use the tiled VAE


def z_embed_key(prompt: str, negative: str, cfg: float, te_id: tuple, loras: list[tuple[str, float]]) -> tuple:
    """Cache key of a Z-Image prompt embedding: the prompt, the negative (only when guidance uses it), the text encoder
    and the LoRAs that touch the text encoder (the usual DiT-only LoRA does not change the embedding)."""
    te_loras = tuple(sorted((p, round(float(w), 4)) for p, w in loras
                            if p and w and loraformat.inspect(p).has_te))
    return (prompt, negative if cfg > 1 else "", tuple(te_id), te_loras)


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
