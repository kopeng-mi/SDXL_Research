"""What a LoRA or checkpoint file is, from its safetensors header only (no weights are read).

Stdlib only: the web process, the guard and the jobs all import it. `inspect(path)`
says which model family a LoRA belongs to (SDXL or Z-Image), its rank, whether it
touches the text encoders and what the training metadata says (top tags, output
name); Flux and SD 1.5 files are refused with a plain reason. Results are cached
by (path, mtime). `write_sidecar` / `sidecar` keep what training knows (trigger
word, class word, dataset, base, steps) next to each LoRA as `<lora>.json`.
`checkpoint_family` tells SDXL from SD 1.5 and Flux checkpoints.
"""

from __future__ import annotations

import json
import re
import struct
from dataclasses import dataclass, field
from pathlib import Path

MAX_HEADER = 64 * 2**20
FAMILY_NAMES = {"sdxl": "SDXL", "zimage": "Z-Image"}

_Z_KEYS = re.compile(r"layers[._]\d+[._](attention|feed_forward|adaLN)|noise_refiner|context_refiner")
_UNET_KEYS = re.compile(r"input_blocks|output_blocks|middle_block|down_blocks|up_blocks|mid_block")
_CROSS_K = re.compile(r"attn2[._]to_k")

_cache: dict[tuple[str, float], "LoraInfo"] = {}
_ckpt_cache: dict[tuple[str, float], str] = {}


@dataclass
class LoraInfo:
    path: str
    ok: bool
    family: str | None = None  # "sdxl" | "zimage"
    reason: str = ""  # why it is refused (plain words)
    rank: int | None = None
    has_te: bool = False
    size_mb: float = 0.0
    tags: tuple = ()
    output_name: str = ""
    trigger: str = ""  # from the sidecar when training wrote one
    extra: dict = field(default_factory=dict)

    @property
    def name(self) -> str:
        return Path(self.path).name


def read_header(path: str | Path) -> tuple[dict, dict]:
    """(tensors, metadata) of a .safetensors file; ValueError when it is not one."""
    try:
        with open(path, "rb") as f:
            raw = f.read(8)
            if len(raw) < 8:
                raise ValueError("too short")
            (n,) = struct.unpack("<Q", raw)
            if n <= 1 or n > MAX_HEADER:
                raise ValueError("bad header size")
            head = json.loads(f.read(n))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError, struct.error) as e:
        raise ValueError(str(e)) from None
    if not isinstance(head, dict):
        raise ValueError("bad header")
    meta = head.pop("__metadata__", None) or {}
    return head, meta if isinstance(meta, dict) else {}


def _shape(t) -> list:
    return t.get("shape", []) if isinstance(t, dict) else []


def _top_tags(meta: dict, n: int = 8) -> tuple:
    try:
        freq = json.loads(meta.get("ss_tag_frequency") or "{}")
        total: dict[str, int] = {}
        for d in freq.values():
            for tag, c in d.items():
                total[tag] = total.get(tag, 0) + int(c)
        return tuple(t for t, _ in sorted(total.items(), key=lambda kv: -kv[1])[:n])
    except (ValueError, AttributeError, TypeError):
        return ()


def inspect(path: str | Path) -> LoraInfo:
    """Family, rank, text-encoder use and metadata of a LoRA file (cached by path and mtime)."""
    p = Path(path)
    try:
        key = (str(p), p.stat().st_mtime)
    except OSError:
        return LoraInfo(str(p), False, reason="The file is missing.")
    if key in _cache:
        info = _cache[key]
    else:
        info = _inspect(p)
        _cache[key] = info
    info.trigger = sidecar(p).get("trigger", "")
    return info


def _inspect(p: Path) -> LoraInfo:
    size_mb = p.stat().st_size / 2**20
    try:
        tensors, meta = read_header(p)
    except ValueError:
        return LoraInfo(str(p), False, size_mb=size_mb, reason="This file is not a readable LoRA (corrupt, or not a .safetensors file).")
    keys = list(tensors)
    joined = "\n".join(keys)
    rank = next((_shape(tensors[k])[0] for k in keys
                 if re.search(r"lora_down|lora_A", k) and len(_shape(tensors[k])) == 2), None)
    has_te = any(re.search(r"lora_te\d?_|text_encoder|^te\d?[._]", k) for k in keys)
    common = dict(path=str(p), size_mb=size_mb, rank=rank, has_te=has_te, tags=_top_tags(meta),
                  output_name=meta.get("ss_output_name", "") or meta.get("modelspec.title", ""))
    if "double_blocks" in joined or "single_blocks" in joined:
        return LoraInfo(ok=False, reason="This is a Flux LoRA. This app makes SDXL and Z-Image pictures.", **common)
    if _Z_KEYS.search(joined):
        return LoraInfo(ok=True, family="zimage", **common)
    if _UNET_KEYS.search(joined):
        if re.search(r"lora_te2_|text_encoder_2", joined):
            return LoraInfo(ok=True, family="sdxl", **common)
        cross = next((_shape(tensors[k])[1] for k in keys if _CROSS_K.search(k) and re.search(r"lora_down|lora_A", k)
                      and len(_shape(tensors[k])) == 2), None)
        if cross == 768 or re.search(r"lora_te_", joined):
            return LoraInfo(ok=False, reason="This is an SD 1.5 LoRA. This app makes SDXL and Z-Image pictures.", **common)
        return LoraInfo(ok=True, family="sdxl", **common)
    return LoraInfo(ok=False, reason="This LoRA is for a model this app does not know (not SDXL or Z-Image).", **common)


def compatible(info: LoraInfo, family: str) -> tuple[bool, str]:
    """(usable, reason) for a model family key: "sdxl" or "zimage"."""
    if not info.ok:
        return False, info.reason
    if info.family != family:
        return False, f"It is a {FAMILY_NAMES[info.family]} LoRA; the picked model is {FAMILY_NAMES[family]}."
    return True, ""


# ---- sidecar written by training ----
def sidecar_path(lora: str | Path) -> Path:
    return Path(lora).with_suffix(".json")


def sidecar(lora: str | Path) -> dict:
    try:
        d = json.loads(sidecar_path(lora).read_text(encoding="utf-8"))
        return d if isinstance(d, dict) else {}
    except (OSError, ValueError):
        return {}


def write_sidecars(out_dir: str | Path, **info) -> list[Path]:
    """One `<lora>.json` for every .safetensors in a training output folder (trigger, class word, dataset, base, steps)."""
    done = []
    for f in sorted(Path(out_dir).glob("*.safetensors")):
        sp = sidecar_path(f)
        sp.write_text(json.dumps({k: v for k, v in info.items() if v not in (None, "")}, indent=1), encoding="utf-8")
        done.append(sp)
    return done


# ---- checkpoints ----
def checkpoint_family(path: str | Path) -> str:
    """"sdxl" | "sd15" | "flux" | "unknown" for a checkpoint, from the header keys (only .safetensors can be told)."""
    p = Path(path)
    if p.suffix.lower() != ".safetensors":
        return "unknown"
    try:
        key = (str(p), p.stat().st_mtime)
        if key in _ckpt_cache:
            return _ckpt_cache[key]
        tensors, _ = read_header(p)
    except (OSError, ValueError):
        return "unknown"
    joined = "\n".join(tensors)
    if "double_blocks" in joined or "single_blocks" in joined:
        fam = "flux"
    elif "conditioner.embedders" in joined or "text_encoder_2" in joined or "add_embedding" in joined or "label_emb" in joined:
        fam = "sdxl"
    elif "cond_stage_model.transformer" in joined or "text_model.encoder" in joined:
        fam = "sd15"
    else:
        fam = "unknown"
    _ckpt_cache[key] = fam
    return fam


def check_sdxl_checkpoint(path: str | Path) -> None:
    """ValueError with a plain sentence when the file is a Flux or SD 1.5 checkpoint."""
    fam = checkpoint_family(path)
    if fam in ("sd15", "flux"):
        what = "an SD 1.5" if fam == "sd15" else "a Flux"
        raise ValueError(f"{Path(path).name} is {what} checkpoint. The SDXL slot needs an SDXL checkpoint.")
