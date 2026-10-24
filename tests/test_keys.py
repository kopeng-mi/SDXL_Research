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
    assert keys.z_embed_key("a cat", "", 0.5, te, []) != base, "any cfg > 0 turns guidance on (needs negative embeds)"
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
    assert te_prequantized("unsloth/Z-Image-Turbo-unsloth-bnb-4bit:text_encoder") and not te_prequantized(Z_ENCODERS["official"])
    assert not te_prequantized(Z_ENCODERS["abliterated"])


def test_controlnet_lite_config() -> None:
    """The shipped config matches what the lite headers contain (read from Hugging Face on 2026-10: Union and Tile
    lite have control_layers 0-2, control_noise_refiner 0-1, control_all_x_embedder.2-1.weight [3840, 132])."""
    cfg = json.loads((Path(__file__).resolve().parents[1] / "src/sdxlr/engine/zcn_lite/config.json").read_text())
    assert len(cfg["control_layers_places"]) == 3 and cfg["control_layers_places"][0] == 0
    assert len(cfg["control_refiner_layers_places"]) == 2 and cfg["n_refiner_layers"] == 2
    assert cfg["add_control_noise_refiner"] == "control_noise_refiner"
    assert cfg["control_in_dim"] * 2 * 2 * 1 == 132, "patch 2x2, f_patch 1"
    assert cfg["all_patch_size"] == [2] and cfg["all_f_patch_size"] == [1] and cfg["dim"] == 3840
    from sdxlr.paths import CONTROLNETS
    assert set(CONTROLNETS) == {"union", "tile"} and all("lite" in f and f.endswith(".safetensors") for _, f in CONTROLNETS.values())


def test_tile_plan() -> None:
    for length in (512, 1280, 1281, 1632, 2048, 2912, 4000, 5792):
        for m in (8, 16):
            starts, size = keys.tile_starts(length, 1280, 128, m)
            assert size <= 1280 and size % m == 0 or size == length, (length, m, size)
            assert starts[0] == 0 and starts[-1] + size == length, (length, starts, size)
            for a, b in zip(starts, starts[1:]):
                assert a + size - b >= 128 - m, ("overlap too small", length, starts, size)
                assert b > a
            n_old = len(sorted({*range(0, max(length - 1024, 0) + 1, 896), max(length - 1024, 0)}))
            assert len(starts) <= n_old, ("more tiles than the old 1024 plan", length, starts)
    assert keys.tile_starts(1632, 1280, 128, 16) == ([0, 752], 880)


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
