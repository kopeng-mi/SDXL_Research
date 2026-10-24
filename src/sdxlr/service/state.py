"""Per-session UI state as a bindable dataclass.

NiceGUI polls plain-dict bindings on the event loop ("active links", every
0.1 s); binding to a bindable_dataclass's fields pushes updates instead. The
pages keep their terse dict-style access (S["prompt"], S.update(...)).

A saved session JSON is migrated on load: unknown keys are dropped, missing
fields take their defaults.
"""

from __future__ import annotations

import dataclasses
import json
import time
from dataclasses import field

from nicegui import binding

from .. import session
from ..activity import log


@binding.bindable_dataclass
class State:
    # generate
    kind: str = "Z-Image"
    sdxl_model: str | None = None
    vae: str = "fp16fix"
    z_model: str = "turbo Q8"
    z_te: str = "official"
    z_quant: str = "nf4"
    mode: str = "Speed"
    prompt: str = ""
    negative: str = ("(worst quality:1.3), (low quality:1.2), lowres, bad anatomy, bad hands, watermark, blur, "
                     "cartoon, anime, drawing")
    style: str = "Photo"
    aspect: str = "Portrait 896x1152"
    count: int = 1
    seed: int = -1
    last_seed: int | None = None
    use_adv: bool = False
    steps: int = 20
    cfg: float = 5.0
    sampler: str = "dpmpp_2m_karras"
    clip_skip: int = 2
    loras: list = field(default_factory=list)  # [[name, weight], ...]
    image: str | None = None
    info: str = ""
    # tool on the shown image: None = new image, "area" = change marked boxes, "whole" = change the whole
    # image, "upscale" = enlarge and add detail; src = the image the tool works on
    tool: str | None = None
    src: str | None = None
    boxes: list = field(default_factory=list)
    strength: float = 0.75
    whole_strength: float = 0.5
    up_strength: float = 0.3
    up_scale: float = 2.0
    # editing: painted mask (id from the upload route), brush, feather, extend % per side (top, right, bottom, left)
    mask_id: str | None = None
    ptool: str = "brush"  # brush | eraser | box | move
    brush: int = 48
    feather: int = 12
    extend: list = field(default_factory=lambda: [0, 0, 0, 0])
    keep_size: bool = False
    edit_count: int = 1
    compare_on: bool = True  # show the before/after slider on an edit result
    # dataset
    ds_name: str = "my_photos"
    ds_link: str = ""
    ds_pick: str | None = None
    cap_model: str = "qwen"
    cap_trigger: str = "ohwx_person"
    cap_class: str = "woman"
    cap_prompt: str = ""
    cap_limit: int = 0
    cap_over: bool = False
    # train
    t_kind: str = "SDXL"
    t_ds: str | None = None
    t_name: str = ""
    t_base: str | None = None
    t_epochs: int = 15
    t_repeats: int = 6
    t_batch: int = 3
    t_res: int = 512
    t_dim: int = 24
    t_lr: float = 2e-4
    t_te_lr: float = 2e-5
    t_type: str = "LoRA"
    t_hf: str = ""
    t_resume: bool = False
    # compare
    c_kind: str = "SDXL"
    c_base: str | None = None
    c_z: str = "turbo Q8"
    c_loras: list = field(default_factory=list)
    c_combos: str = ""
    c_str: str = "0.6 0.8 1.0"
    c_prompt: str = "ohwx_person, woman, portrait photo, natural light"
    c_neg: str = ("(worst quality:1.3), (low quality:1.2), lowres, bad anatomy, bad hands, watermark, blur, "
                  "cartoon, anime, drawing")
    c_steps: int = 20
    c_cfg: float = 5.0
    c_seed: int = 42
    c_name: str = "grid1"
    # library
    l_link: str = ""
    l_kind: str = "LoRA"

    # ---- dict-style access the pages use ----
    def __getitem__(self, key: str):
        return getattr(self, key)

    def __setitem__(self, key: str, value) -> None:
        setattr(self, key, value)

    def __contains__(self, key: str) -> bool:
        return hasattr(self, key)

    def get(self, key: str, default=None):
        return getattr(self, key, default)

    def update(self, **kw) -> None:
        for k, v in kw.items():
            setattr(self, k, v)

    def as_dict(self) -> dict:
        return dataclasses.asdict(self)


FIELDS = {f.name for f in dataclasses.fields(State)}

STATES: dict[str, State] = {}
_saved: dict[str, str] = {}


def state_for(code: str) -> State:
    if code not in STATES:
        saved = session.load(code) or {}
        STATES[code] = State(**{k: v for k, v in saved.items() if k in FIELDS})  # migrate: unknown keys drop
    return STATES[code]


def autosave_loop() -> None:
    while True:
        time.sleep(5)
        for code, s in list(STATES.items()):
            try:
                d = s.as_dict()  # may race with the page changing a list: the next pass gets it
                blob = json.dumps(d, default=str, sort_keys=True)
                if _saved.get(code) != blob:
                    session.save(code, d)
                    _saved[code] = blob
            except Exception as e:
                log.info(f"autosave of session {code} failed: {type(e).__name__}: {e}")
