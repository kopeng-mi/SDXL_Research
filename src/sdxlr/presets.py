"""Generation presets as JSON, plus sync of presets/LoRAs to a private HF repo.

A preset holds everything needed to repeat a look:
  {"name", "model": {"kind", "path" | "z_model", "z_te", "z_quant"}, "mode",
   "prompt", "negative", "loras": [[file_name, weight], ...], "aspect",
   "width", "height", "steps", "cfg", "sampler", "clip_skip", "seed", "count"}
LoRAs are stored by file name, so a preset works after the LoRA is re-downloaded.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from .fetch import LORAS, WORK

PRESETS = WORK / "presets"


def _safe(name: str) -> str:
    return re.sub(r"[^\w\-. ]", "_", name).strip() or "preset"


def list_presets() -> list[str]:
    return sorted(p.stem for p in PRESETS.glob("*.json"))


def save(name: str, data: dict) -> Path:
    PRESETS.mkdir(parents=True, exist_ok=True)
    data = {**data, "name": name}
    p = PRESETS / f"{_safe(name)}.json"
    p.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return p


def load(name: str) -> dict:
    return json.loads((PRESETS / f"{_safe(name)}.json").read_text(encoding="utf-8"))


def delete(name: str) -> None:
    (PRESETS / f"{_safe(name)}.json").unlink(missing_ok=True)


# ---- HF sync (private repo; token from HF_TOKEN / Colab secrets) ----
def _api():
    from huggingface_hub import HfApi
    return HfApi()


def push(repo: str, what: tuple[str, ...] = ("presets", "loras")) -> str:
    api = _api()
    api.create_repo(repo, private=True, exist_ok=True)
    done = []
    if "presets" in what and PRESETS.exists():
        api.upload_folder(repo_id=repo, folder_path=str(PRESETS), path_in_repo="presets", allow_patterns=["*.json"])
        done.append(f"{len(list_presets())} presets")
    if "loras" in what:
        files = [*LORAS.glob("*.safetensors"), *(WORK / "runs").glob("*/*.safetensors")]
        for f in files:
            api.upload_file(repo_id=repo, path_or_fileobj=str(f), path_in_repo=f"loras/{f.name}")
        done.append(f"{len(files)} LoRAs")
    return f"pushed {', '.join(done) or 'nothing'} to {repo}"


def pull(repo: str, what: tuple[str, ...] = ("presets", "loras")) -> str:
    from huggingface_hub import snapshot_download
    pats = [f"{w}/*" for w in what]
    tmp = Path(snapshot_download(repo, allow_patterns=pats, local_dir=WORK / ".hfsync"))
    n = {}
    for w, dest in (("presets", PRESETS), ("loras", LORAS)):
        src = tmp / w
        if w in what and src.exists():
            dest.mkdir(parents=True, exist_ok=True)
            files = [f for f in src.iterdir() if f.is_file()]
            for f in files:
                f.replace(dest / f.name)
            n[w] = len(files)
    return f"pulled {', '.join(f'{v} {k}' for k, v in n.items()) or 'nothing'} from {repo}"
