"""App settings set from the UI (HF repos etc.), saved in /content/work/settings.json.

The HF token is never stored here: it comes from Colab Secrets (HF_TOKEN),
which the Setup cell exports before the app starts.
"""

from __future__ import annotations

import json

from .fetch import WORK

FILE = WORK / "settings.json"
DEFAULTS = {
    "sync_repo": "",      # private HF repo for presets + LoRAs, e.g. "me/sdxlr-sync"
    "datasets_repo": "",  # private HF repo with dataset zips
    "models_repo": "",    # private HF repo with base models
}


def get() -> dict:
    data = json.loads(FILE.read_text(encoding="utf-8")) if FILE.exists() else {}
    return {**DEFAULTS, **data}


def set(**kw) -> dict:
    data = {**get(), **{k: v.strip() if isinstance(v, str) else v for k, v in kw.items()}}
    FILE.parent.mkdir(parents=True, exist_ok=True)
    FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return data
