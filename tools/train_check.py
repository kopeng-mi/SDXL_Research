"""T4 training session (plan session C) through the app's own job runner. Run in the Colab venv, app NOT running:

  /content/venv/bin/python tools/train_check.py --photos /content/t4_ds [--skip sdxl,zturbo] [--trigger ohwx_person]
      [--class-word woman]

Steps (each a real app job: the same child processes, logs and pidfiles as the UI):
  import   photos -> dataset "me" (dedupe check) and a copy "me_wd"
  caption  WD tags on "me_wd", Qwen identity captions on "me" (the one trained on)
  sdxl     SDXL LoRA on "me" (11 photos x 6 repeats x 15 epochs / batch 3)
  zbase    Z-Image LoRA on Base, 512 px (x 4 repeats x 10 epochs)
  zturbo   Z-Image LoRA on Turbo + ostris's training adapter (same settings): the A/B
  compare  grids: each LoRA's last epochs x strengths, same seed
  inpaint  the face of one photo repainted with each Z-Image LoRA (trigger in the prompt)

Writes WORK/train_check/report.md (+ json). Photos, captions and LoRAs stay on the VM for review; delete them in
Library (or this VM) when done.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from PIL import Image, ImageDraw  # noqa: E402

from sdxlr.activity import Task  # noqa: E402
from sdxlr.paths import DATASETS, RUNS, SDXL_BASE, WORK  # noqa: E402
from sdxlr.service import files, guard, masks  # noqa: E402
from sdxlr.service.gen import engc, generate_blocking  # noqa: E402
from sdxlr.service.jobs import jm  # noqa: E402
from sdxlr.service.state import State  # noqa: E402

OUT = WORK / "train_check"
REPORT: dict = {"started": time.strftime("%Y-%m-%d %H:%M:%S"), "steps": []}


def wait(job, label: str) -> dict:
    t0 = time.time()
    last = 0.0
    while job.status in ("queued", "running"):
        time.sleep(5)
        if time.time() - last > 120:
            last = time.time()
            print(f"  {label}: {job.status} {job.done}/{job.total}", flush=True)
    rec = {"label": label, "ok": job.status == "done", "seconds": round(time.time() - t0),
           "status": job.status, "tail": list(job.log)[-6:]}
    print(json.dumps({k: v for k, v in rec.items() if k != "tail"}), flush=True)
    if not rec["ok"]:
        print("  " + "\n  ".join(rec["tail"]), flush=True)
    REPORT["steps"].append(rec)
    write()
    return rec


def submit(kind: str, label: str, args: list, gpu: bool = True) -> dict:
    return wait(jm.submit(kind, label, args, gpu=gpu), label)


def write() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    REPORT["ended"] = time.strftime("%Y-%m-%d %H:%M:%S")
    (OUT / "report.json").write_text(json.dumps(REPORT, indent=1), encoding="utf-8")
    lines = [f"# Training check {REPORT['started']}", "", "| step | ok | time | note |", "|---|---|---|---|"]
    for r in REPORT["steps"]:
        note = r.get("note") or (r["tail"][-1] if r.get("tail") else "")
        lines.append(f"| {r['label']} | {'yes' if r.get('ok') else 'NO'} | {r.get('seconds', '')} s | "
                     f"{str(note)[:150].replace('|', '/')} |")
    (OUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def loras_of(run: str, diffusers: bool) -> list[Path]:
    fs = sorted((RUNS / run).glob("*.safetensors"))
    fs = [f for f in fs if f.stem.endswith("_diffusers") == diffusers and "state" not in f.parts]
    return fs[-2:]  # the last two epochs


def face_mask(src: Path) -> bytes:
    """An oval over the upper middle of a portrait: where the face usually is."""
    import io
    w, h = Image.open(src).size
    f = min(1.0, 2048 / max(w, h))
    dw, dh = int(w * f), int(h * f)
    m = Image.new("L", (dw, dh), 0)
    ImageDraw.Draw(m).ellipse([0.33 * dw, 0.12 * dh, 0.67 * dw, 0.45 * dh], fill=255)
    b = io.BytesIO()
    m.save(b, "PNG")
    return b.getvalue()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--photos", default="/content/t4_ds")
    ap.add_argument("--skip", default="")
    ap.add_argument("--trigger", default="ohwx_person")
    ap.add_argument("--class-word", default="woman")
    a = ap.parse_args()
    skip = set(a.skip.split(","))
    guard.start(engc, jm)
    files.start_refresh()
    me, wd = DATASETS / "me", DATASETS / "me_wd"

    try:
        if "import" not in skip:
            submit("import", "import photos", ["-m", "sdxlr.data", "--dest", me, a.photos], gpu=False)
            shutil.copytree(me, wd, dirs_exist_ok=True)
            REPORT["steps"][-1]["note"] = f"{len(list(me.iterdir()))} files in me"
        if "caption" not in skip:
            submit("caption", "captions WD (me_wd)", ["-m", "sdxlr.caption", "--images", wd, "--model", "wd",
                                                     "--trigger", a.trigger, "--class-word", a.class_word])
            submit("caption", "captions Qwen (me)", ["-m", "sdxlr.caption", "--images", me, "--model", "qwen",
                                                    "--trigger", a.trigger, "--class-word", a.class_word])
            caps = {p.name: p.read_text(encoding="utf-8")[:200] for p in sorted(me.glob("*.txt"))[:3]}
            REPORT["captions_sample"] = caps
        tw = ["--trigger", a.trigger, "--class-word", a.class_word]
        if "sdxl" not in skip:
            submit("train", "train SDXL", ["-m", "sdxlr.train", "--name", "me_sdxl", "--images", me,
                                           "--model", SDXL_BASE, "--set", "epochs=15", "repeats=6", "batch_size=3",
                                           *tw])
        if "zbase" not in skip:
            submit("train", "train Z-Image (Base)", ["-m", "sdxlr.train_zimage", "--name", "me_zbase", "--images", me,
                                                     "--set", "epochs=10", "repeats=4", *tw])
        if "zturbo" not in skip:
            submit("train", "train Z-Image (Turbo + ostris adapter)",
                   ["-m", "sdxlr.train_zimage", "--name", "me_zturbo", "--images", me,
                    "--set", "epochs=10", "repeats=4", "dit=turbo", "adapter=ostris", *tw])
        files._refresh(force=True)
        prompt = f"{a.trigger}, {a.class_word}, portrait photo, natural light, looking at the camera"
        if "compare" not in skip:
            from sdxlr.fetch import MODELS, fetch
            if (RUNS / "me_sdxl").exists():
                combos = [f"{f.name}:1.0" for f in loras_of("me_sdxl", False)]
                if combos:
                    submit("compare", "compare SDXL", ["-m", "sdxlr.compare", "--name", "me_sdxl", "--model",
                                                       str(fetch(SDXL_BASE, MODELS)), "--strengths", "0.6", "0.8",
                                                       "1.0", "--prompt", prompt, "--seed", "42", "--combos", *combos])
            for run in ("me_zbase", "me_zturbo"):
                combos = [f"{f.name}:1.0" for f in loras_of(run, True)]
                if combos:
                    submit("compare", f"compare {run}", ["-m", "sdxlr.compare", "--name", run, "--zimage", "--model", "turbo Q8",
                                                         "--strengths", "0.8", "1.0", "1.3", "--prompt", prompt,
                                                         "--steps", "9", "--cfg", "0", "--seed", "42",
                                                         "--combos", *combos])
        if "inpaint" not in skip:
            files._refresh(force=True)
            photo = sorted(p for p in me.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))[0]
            mid = masks.store("9998", face_mask(photo))
            for run in ("me_zbase", "me_zturbo"):
                ls = loras_of(run, True)
                if not ls:
                    continue
                S = State()
                S.update(kind="Z-Image", z_model="turbo Q8", mode="Speed", prompt=prompt, seed=42, tool="area",
                         src=str(photo), image=str(photo), mask_id=mid, strength=0.85, feather=12, edit_count=1,
                         loras=[[ls[-1].name, 1.0]])
                t0, rec = time.time(), {"label": f"face inpaint with {ls[-1].name}"}
                try:
                    paths, info, _ = generate_blocking(S, Task("generate", rec["label"]))
                    rec.update(ok=True, note=f"{paths[0]} | {info}")
                except Exception as e:
                    rec.update(ok=False, note=f"{type(e).__name__}: {e}", tb=traceback.format_exc()[-800:])
                rec["seconds"] = round(time.time() - t0)
                print(json.dumps({k: v for k, v in rec.items() if k != "tb"}), flush=True)
                REPORT["steps"].append(rec)
                write()
    finally:
        engc.kill()
        write()
    print(f"done: {OUT / 'report.md'}", flush=True)


if __name__ == "__main__":
    main()
