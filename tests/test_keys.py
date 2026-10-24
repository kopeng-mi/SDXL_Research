"""Tests for engine/keys.py: embedding cache key, Z-Image part reuse, VAE tiling threshold.

Run: python tests/test_keys.py   (stdlib only)
"""

from __future__ import annotations

import json
import struct
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sdxlr.engine import keys  # noqa: E402
from sdxlr.paths import te_prequantized  # noqa: E402

TMP = Path(tempfile.mkdtemp(prefix="sdxlr_keys_test_"))


def lora(name: str, tensors: dict) -> str:
    head = {k: {"dtype": "F16", "shape": s, "data_offsets": [0, 0]} for k, s in tensors.items()}
    raw = json.dumps(head).encode()
    p = TMP / name
    p.write_bytes(struct.pack("<Q", len(raw)) + raw)
    return str(p)


def test_embed_key() -> None:
    dit_only = lora("dit.safetensors", {"transformer.layers.0.attention.to_q.lora_A.weight": [8, 3840]})
    with_te = lora("te.safetensors", {"transformer.layers.0.attention.to_q.lora_A.weight": [8, 3840],
                                      "text_encoder.layers.0.q.lora_A.weight": [8, 2560]})
    te = ["Tongyi-MAI/Z-Image-Turbo:text_encoder", "nf4"]
    base = keys.z_embed_key("a cat", "", 0.0, te, [])
    assert keys.z_embed_key("a cat", "ugly", 0.0, te, []) == base, "negative ignored without guidance"
    assert keys.z_embed_key("a cat", "ugly", 3.0, te, []) != base, "negative counts with guidance"
    assert keys.z_embed_key("a cat", "", 0.0, te, [(dit_only, 0.8)]) == base, "a DiT-only LoRA does not change the embedding"
    assert keys.z_embed_key("a cat", "", 0.0, te, [(with_te, 0.8)]) != base
    assert keys.z_embed_key("a cat", "", 0.0, te, [(with_te, 0.8)]) != keys.z_embed_key("a cat", "", 0.0, te, [(with_te, 0.5)])
    assert keys.z_embed_key("a cat", "", 0.0, te, [(with_te, 0)]) == base, "weight 0 is off"
    assert keys.z_embed_key("a cat", "", 0.0, ["x", "int8"], []) != base, "another text encoder"
    assert keys.z_embed_key("a dog", "", 0.0, te, []) != base


def test_reuse_plan() -> None:
    old = "zimage|unsloth/z-image-turbo-Q8_0.gguf|Tongyi-MAI/Z-Image-Turbo:text_encoder|nf4"
    assert keys.reuse_plan(old, "unsloth/z-image-turbo-Q8_0.gguf", "Tongyi-MAI/Z-Image-Turbo:text_encoder", "nf4") == (True, True)
    assert keys.reuse_plan(old, "unsloth/z-image-turbo-Q6_K.gguf", "Tongyi-MAI/Z-Image-Turbo:text_encoder", "nf4") == (False, True)
    assert keys.reuse_plan(old, "unsloth/z-image-turbo-Q8_0.gguf", "other/te", "nf4") == (True, False)
    assert keys.reuse_plan(old, "unsloth/z-image-turbo-Q8_0.gguf", "Tongyi-MAI/Z-Image-Turbo:text_encoder", "int8") == (True, False)
    assert keys.reuse_plan(None, "a", "b", "c") == (False, False)
    assert keys.reuse_plan("/m/sdxl.safetensors", "a", "b", "c") == (False, False), "an SDXL model has no Z parts"


def test_tiling_threshold() -> None:
    assert not keys.needs_tiling(1024, 1024) and not keys.needs_tiling(1216, 1216)  # 1.48 MP
    assert keys.needs_tiling(1280, 1280) and keys.needs_tiling(2048, 1536)


def test_prequantized_encoder() -> None:
    from sdxlr.paths import Z_ENCODERS
    assert te_prequantized(Z_ENCODERS["unsloth"]) and not te_prequantized(Z_ENCODERS["official"])
    assert not te_prequantized(Z_ENCODERS["abliterated"])


if __name__ == "__main__":
    failed = 0
    for name, fn in [(k, v) for k, v in list(globals().items()) if k.startswith("test_")]:
        try:
            fn()
            print(f"{name}: ok")
        except Exception as e:
            failed += 1
            print(f"{name}: FAILED {type(e).__name__}: {e}")
    print("keys tests passed" if not failed else f"{failed} keys tests FAILED")
    sys.exit(1 if failed else 0)
