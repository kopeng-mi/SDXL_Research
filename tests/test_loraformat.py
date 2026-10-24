"""Tests for loraformat.py: header-only LoRA/checkpoint inspection with fixture headers.

Run: python tests/test_loraformat.py   (stdlib only)
"""

from __future__ import annotations

import json
import struct
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sdxlr import loraformat as lf  # noqa: E402

TMP = Path(tempfile.mkdtemp(prefix="sdxlr_lf_test_"))


def make(name: str, shapes: dict[str, list[int]], meta: dict | None = None) -> Path:
    """A safetensors file with this header and no tensor data (the header is all we read)."""
    head: dict = {k: {"dtype": "F16", "shape": s, "data_offsets": [0, 0]} for k, s in shapes.items()}
    if meta:
        head["__metadata__"] = meta
    raw = json.dumps(head).encode()
    p = TMP / name
    p.write_bytes(struct.pack("<Q", len(raw)) + raw)
    return p


def test_sdxl_kohya() -> None:
    p = make("sdxl_kohya.safetensors", {
        "lora_unet_down_blocks_1_attentions_0_transformer_blocks_0_attn2_to_k.lora_down.weight": [16, 2048],
        "lora_unet_down_blocks_1_attentions_0_transformer_blocks_0_attn2_to_k.lora_up.weight": [640, 16],
        "lora_te1_text_model_encoder_layers_0_self_attn_q_proj.lora_down.weight": [16, 768],
        "lora_te2_text_model_encoder_layers_0_self_attn_q_proj.lora_down.weight": [16, 1280],
    }, {"ss_output_name": "myperson", "ss_tag_frequency": json.dumps({"10_x": {"ohwx_person": 30, "woman": 28, "smile": 3}})})
    i = lf.inspect(p)
    assert i.ok and i.family == "sdxl" and i.rank == 16 and i.has_te, i
    assert i.output_name == "myperson" and i.tags[:2] == ("ohwx_person", "woman"), i.tags


def test_sdxl_diffusers_unet_only() -> None:
    p = make("sdxl_diffusers.safetensors", {
        "unet.down_blocks.1.attentions.0.transformer_blocks.0.attn2.to_k.lora_A.weight": [32, 2048],
        "unet.down_blocks.1.attentions.0.transformer_blocks.0.attn2.to_k.lora_B.weight": [640, 32],
    })
    i = lf.inspect(p)
    assert i.ok and i.family == "sdxl" and i.rank == 32 and not i.has_te, i


def test_zimage_musubi_and_diffusers() -> None:
    a = lf.inspect(make("z_musubi.safetensors", {
        "lora_unet_layers_0_attention_to_q.lora_down.weight": [8, 3840],
        "lora_unet_layers_0_attention_to_q.lora_up.weight": [3840, 8]}))
    assert a.ok and a.family == "zimage" and a.rank == 8, a
    b = lf.inspect(make("z_diffusers.safetensors", {
        "transformer.layers.3.feed_forward.w1.lora_A.weight": [16, 3840],
        "transformer.noise_refiner.0.attention.to_q.lora_A.weight": [16, 3840]}))
    assert b.ok and b.family == "zimage", b
    c = lf.inspect(make("z_comfy.safetensors", {"diffusion_model.context_refiner.1.attention.to_k.lora_A.weight": [4, 3840]}))
    assert c.ok and c.family == "zimage", c


def test_flux_and_sd15_are_refused() -> None:
    f = lf.inspect(make("flux.safetensors", {"lora_unet_double_blocks_0_img_attn_qkv.lora_down.weight": [16, 3072]}))
    assert not f.ok and "Flux" in f.reason, f
    s = lf.inspect(make("sd15.safetensors", {
        "lora_unet_down_blocks_0_attentions_0_transformer_blocks_0_attn2_to_k.lora_down.weight": [8, 768],
        "lora_te_text_model_encoder_layers_0_self_attn_q_proj.lora_down.weight": [8, 768]}))
    assert not s.ok and "SD 1.5" in s.reason, s
    # 768-wide cross attention without a text encoder: still SD 1.5
    s2 = lf.inspect(make("sd15_unet.safetensors", {
        "lora_unet_down_blocks_0_attentions_0_transformer_blocks_0_attn2_to_k.lora_down.weight": [8, 768]}))
    assert not s2.ok, s2


def test_corrupt_and_missing() -> None:
    bad = TMP / "bad.safetensors"
    bad.write_bytes(b"not a safetensors file at all")
    assert not lf.inspect(bad).ok and "not a readable" in lf.inspect(bad).reason
    bad2 = TMP / "bad2.safetensors"
    bad2.write_bytes(struct.pack("<Q", 10**12) + b"{}")
    assert not lf.inspect(bad2).ok
    assert not lf.inspect(TMP / "nope.safetensors").ok
    assert not lf.inspect(make("other.safetensors", {"foo.weight": [1, 2]})).ok


def test_compatible_reasons() -> None:
    z = lf.inspect(TMP / "z_musubi.safetensors")
    assert lf.compatible(z, "zimage") == (True, "")
    ok, why = lf.compatible(z, "sdxl")
    assert not ok and "Z-Image LoRA" in why and "SDXL" in why
    ok, why = lf.compatible(lf.inspect(TMP / "flux.safetensors"), "sdxl")
    assert not ok and "Flux" in why


def test_cache_by_mtime() -> None:
    p = make("cache.safetensors", {"lora_unet_layers_0_attention_to_q.lora_down.weight": [8, 3840]})
    a = lf.inspect(p)
    assert lf.inspect(p) is a
    import os
    import time
    time.sleep(0.02)
    make("cache.safetensors", {"lora_unet_double_blocks_0_x.lora_down.weight": [8, 8]})
    os.utime(p, (time.time() + 5, time.time() + 5))
    assert not lf.inspect(p).ok, "a changed file is read again"


def test_sidecar_roundtrip() -> None:
    d = TMP / "run"
    d.mkdir()
    (d / "a.safetensors").write_bytes(b"x")
    (d / "a-000005.safetensors").write_bytes(b"x")
    done = lf.write_sidecars(d, trigger="ohwx_person", class_word="woman", dataset="my_photos", base="sdxl.safetensors",
                             steps=330, empty="")
    assert len(done) == 2
    assert lf.sidecar(d / "a.safetensors") == {"trigger": "ohwx_person", "class_word": "woman", "dataset": "my_photos",
                                               "base": "sdxl.safetensors", "steps": 330}
    assert lf.sidecar(d / "missing.safetensors") == {}


def test_checkpoint_family() -> None:
    assert lf.checkpoint_family(make("c_sdxl.safetensors", {"conditioner.embedders.1.model.x": [1], "model.diffusion_model.label_emb.0.0.weight": [1, 2]})) == "sdxl"
    assert lf.checkpoint_family(make("c_sd15.safetensors", {"cond_stage_model.transformer.text_model.encoder.layers.0.w": [1]})) == "sd15"
    assert lf.checkpoint_family(make("c_flux.safetensors", {"double_blocks.0.img_attn.qkv.weight": [1]})) == "flux"
    assert lf.checkpoint_family(TMP / "x.gguf") == "unknown"
    try:
        lf.check_sdxl_checkpoint(TMP / "c_sd15.safetensors")
        raise AssertionError("expected a refusal")
    except ValueError as e:
        assert "SD 1.5" in str(e) and "SDXL slot" in str(e)
    lf.check_sdxl_checkpoint(TMP / "c_sdxl.safetensors")
    lf.check_sdxl_checkpoint(TMP / "never_seen.safetensors")  # unreadable = not refused


if __name__ == "__main__":
    failed = 0
    for name, fn in [(k, v) for k, v in list(globals().items()) if k.startswith("test_")]:
        try:
            fn()
            print(f"{name}: ok")
        except Exception as e:
            failed += 1
            print(f"{name}: FAILED {type(e).__name__}: {e}")
    print("loraformat tests passed" if not failed else f"{failed} loraformat tests FAILED")
    sys.exit(1 if failed else 0)
