"""Caption a folder of images for LoRA training. Runs inside the main venv.

  python -m sdxlr.caption --images DIR [--model qwen3vl-4b] [--style id] [--target sdxl] [--trigger ohwx]

Models come from paths.CAPTIONERS (or any HF id): Qwen3-VL / Qwen3.5 (plain and
abliterated), JoyCaption, and the WD tagger (booru tags, no prompt). fp16 or nf4
is picked from the model size (plan.caption_quant) unless --quant says otherwise;
Qwen captions run in left-padded batches. --mirror copies each caption to a Drive
folder as soon as it is written, so a disconnect loses at most one batch.

One model is loaded once per process; the process exits when done, so all
VRAM/RAM goes back to the system before training starts.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import torch
from PIL import Image, ImageOps

from sdxlr import drive, plan
from sdxlr.paths import CAPTIONER_ALIASES, CAPTIONERS

# Identity LoRA style: describe what should stay promptable, leave out what the
# LoRA should learn from the pixels (face, body, skin).
IDENTITY_PROMPT = (
    "Write a training caption for this photo, {words} words. "
    "Begin with 'a {cls}' and call the main subject only 'a {cls}'. "
    "Describe the clothing and accessories, pose, expression, camera angle, framing, "
    "background and lighting. "
    "Do not describe the subject's face, facial features, eye color, body shape, figure, "
    "skin, age or ethnicity. "
    "Only describe what is visible; do not mention a photographer or invent a story. "
    "Plain text only, no lists."
)
# Style and concept LoRAs: describe the content, never name the look being trained.
DETAILED_PROMPT = (
    "Describe this image for a training caption, {words} words: the subjects, what they wear and do, "
    "the setting, objects, composition, camera angle and lighting. "
    "Do not name an art style, artist, medium or overall aesthetic. "
    "Only describe what is visible. Plain text only, no lists."
)
STYLES = {"id": IDENTITY_PROMPT, "detailed": DETAILED_PROMPT}
# words per target model: SDXL keeps one 75-token CLIP chunk, Z-Image likes long prompts
WORDS = {"sdxl": (30, 50), "zimage": (80, 150)}

EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
MAX_SIDE = 768  # 576 vision tokens at most; captions don't need more detail
DTYPE = torch.float16  # T4 (sm75) has no fast bf16


def _load_image(path: Path) -> Image.Image:
    img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    img.thumbnail((MAX_SIDE, MAX_SIDE), Image.LANCZOS)
    return img


def _nf4(skip: list[str]):
    from transformers import BitsAndBytesConfig
    return BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=DTYPE,
                              bnb_4bit_use_double_quant=True, llm_int8_skip_modules=skip)


class Qwen:
    """Qwen3-VL and Qwen3.5 (thinking off), batched."""

    def __init__(self, model_id: str, quant: str):
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.proc = AutoProcessor.from_pretrained(model_id)
        self.proc.tokenizer.padding_side = "left"  # batched generate needs left padding
        q = _nf4(["visual", "lm_head"]) if quant == "nf4" else None
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id, dtype=DTYPE, quantization_config=q, device_map="auto", attn_implementation="sdpa").eval()

    @torch.inference_mode()
    def __call__(self, imgs: list[Image.Image], prompt: str, max_new_tokens: int) -> list[str]:
        convs = [[{"role": "user", "content": [{"type": "image", "image": im}, {"type": "text", "text": prompt}]}]
                 for im in imgs]
        x = self.proc.apply_chat_template(
            convs, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt",
            padding=True, enable_thinking=False,
        ).to(self.model.device)
        ids = self.model.generate(**x, max_new_tokens=max_new_tokens, do_sample=False)
        return self.proc.batch_decode(ids[:, x["input_ids"].shape[1]:], skip_special_tokens=True)


class Joy:
    def __init__(self, model_id: str, quant: str):
        from transformers import AutoProcessor, LlavaForConditionalGeneration

        self.proc = AutoProcessor.from_pretrained(model_id)
        # nf4 keeps the 8B LLM near 5 GB; the vision tower stays fp16 for quality
        self.model = LlavaForConditionalGeneration.from_pretrained(
            model_id, dtype=DTYPE, device_map="auto",
            quantization_config=_nf4(["vision_tower", "multi_modal_projector"]) if quant == "nf4" else None,
        ).eval()

    @torch.inference_mode()
    def __call__(self, imgs: list[Image.Image], prompt: str, max_new_tokens: int) -> list[str]:
        out = []
        for img in imgs:
            convo = [{"role": "system", "content": "You are a helpful image captioner."},
                     {"role": "user", "content": prompt}]
            text = self.proc.apply_chat_template(convo, tokenize=False, add_generation_prompt=True)
            x = self.proc(text=[text], images=[img], return_tensors="pt").to(self.model.device)
            x["pixel_values"] = x["pixel_values"].to(DTYPE)
            ids = self.model.generate(**x, max_new_tokens=max_new_tokens, do_sample=False)
            out.append(self.proc.batch_decode(ids[:, x["input_ids"].shape[1]:], skip_special_tokens=True)[0])
        return out


class WD:
    """WD tagger v3 through timm (same recipe as SmilingWolf's reference script)."""

    def __init__(self, model_id: str, threshold: float = 0.35):
        import csv

        import timm
        from huggingface_hub import hf_hub_download
        from timm.data import create_transform, resolve_data_config

        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = DTYPE if self.dev == "cuda" else torch.float32
        self.model = timm.create_model("hf-hub:" + model_id, pretrained=True).eval().to(self.dev, self.dtype)
        self.tf = create_transform(**resolve_data_config(self.model.pretrained_cfg, model=self.model))
        with open(hf_hub_download(model_id, "selected_tags.csv"), encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        self.names = [r["name"] for r in rows]
        self.keep = [i for i, r in enumerate(rows) if r["category"] in ("0", "4")]  # general + character, no ratings
        self.threshold = threshold

    @torch.inference_mode()
    def __call__(self, imgs: list[Image.Image], prompt: str, max_new_tokens: int) -> list[str]:
        xs = []
        for img in imgs:
            px = max(img.size)  # pad to a white square, as the tagger was trained
            sq = Image.new("RGB", (px, px), (255, 255, 255))
            sq.paste(img, ((px - img.width) // 2, (px - img.height) // 2))
            xs.append(self.tf(sq)[[2, 1, 0]])  # model expects BGR
        probs = torch.sigmoid(self.model(torch.stack(xs).to(self.dev, self.dtype)).float()).cpu()
        out = []
        for pr in probs:
            tags = sorted(((pr[i].item(), self.names[i]) for i in self.keep if pr[i] > self.threshold), reverse=True)
            out.append(", ".join(n.replace("_", " ").replace("(", "\\(").replace(")", "\\)") for _, n in tags))
        return out


def _clean(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    return " ".join(text.replace("\n", " ").split()).strip().strip('"')


def resolve(model: str) -> tuple[str, float, str]:
    """(repo, params_b, loader) for a catalog key, alias or HF id (unknown ids: size guessed from the name)."""
    key = CAPTIONER_ALIASES.get(model, model)
    if key in CAPTIONERS:
        return CAPTIONERS[key]
    m = re.search(r"(\d+(?:\.\d+)?)B", model, re.I)
    loader = "joy" if "joycaption" in model.lower() else "wd" if "tagger" in model.lower() else "qwen"
    return model, float(m[1]) if m else 4.0, loader


def build_prompt(style: str, target: str, cls: str, max_words: int = 0, custom: str = "") -> str:
    lo, hi = WORDS[target]
    if max_words:
        lo, hi = max(1, max_words * 3 // 5), max_words
    text = custom or STYLES[style]
    return text.replace("{words}", f"{lo} to {hi}").replace("{cls}", cls)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", type=Path, required=True)
    ap.add_argument("--out", type=Path, help="caption dir (default: next to images)")
    ap.add_argument("--model", default="qwen3vl-4b", help=f"{', '.join(CAPTIONERS)} or a HF model id")
    ap.add_argument("--style", default="id", choices=[*STYLES, "tags"], help="tags = WD tagger")
    ap.add_argument("--target", default="sdxl", choices=list(WORDS), help="sets the caption length")
    ap.add_argument("--max-words", type=int, default=0, help="0 = from --target")
    ap.add_argument("--trigger", default="ohwx_person", help="prepended to every caption; '' for none")
    ap.add_argument("--class-word", default="woman", help="what the subject is called in captions")
    ap.add_argument("--prompt", default="", help="custom prompt; {cls} and {words} are filled in")
    ap.add_argument("--quant", default="auto", choices=["auto", "fp16", "nf4"])
    ap.add_argument("--batch", type=int, default=0, help="0 = auto (4 fp16, 2 nf4)")
    ap.add_argument("--limit", type=int, default=0, help="caption only the first N (preview)")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--threshold", type=float, default=0.35, help="wd: minimum tag confidence")
    ap.add_argument("--mirror", type=Path, help="copy each caption here too (Drive)")
    a = ap.parse_args()

    out = a.out or a.images
    out.mkdir(parents=True, exist_ok=True)
    files = sorted(p for p in a.images.rglob("*") if p.suffix.lower() in EXTS)
    if a.limit:
        files = files[: a.limit]
    todo = [p for p in files if a.overwrite or not (out / f"{p.stem}.txt").exists()]
    print(f"{len(files)} images, {len(todo)} to caption", flush=True)
    if not todo:
        return

    model_id, params_b, loader = resolve("wd" if a.style == "tags" else a.model)
    quant = plan.caption_quant(params_b) if a.quant == "auto" else a.quant
    batch = a.batch or (8 if loader == "wd" else 1 if loader == "joy" else plan.caption_batch(quant))
    prompt = build_prompt(a.style, a.target, a.class_word, a.max_words, a.prompt)
    max_new = int((a.max_words or WORDS[a.target][1]) * 1.3 * 1.5) + 16
    t0 = time.time()
    print(f"loading {model_id} ({quant}, about {plan.caption_vram(params_b, quant):.1f} GB; "
          f"the first time this downloads it)", flush=True)
    cap = WD(model_id, a.threshold) if loader == "wd" else (Joy if loader == "joy" else Qwen)(model_id, quant)
    print(f"loaded in {time.time() - t0:.0f}s, batch {batch}", flush=True)

    done, failed = 0, {}

    def write(p: Path, text: str, i: int, secs: float) -> None:
        nonlocal done
        text = _clean(text)
        final = f"{a.trigger}, {text}" if a.trigger else text
        txt = out / f"{p.stem}.txt"
        txt.write_text(final, encoding="utf-8")
        if a.mirror:
            drive.mirror(txt, a.mirror)
        done += 1
        print(f"[{i}/{len(todo)}] {p.name} ({secs:.1f}s): {final}", flush=True)

    for s in range(0, len(todo), batch):
        chunk = todo[s:s + batch]
        t = time.time()
        try:
            texts = cap([_load_image(p) for p in chunk], prompt, max_new)
            for j, (p, text) in enumerate(zip(chunk, texts)):
                write(p, text, s + j + 1, (time.time() - t) / len(chunk))
            continue
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            if len(chunk) == 1:
                failed[chunk[0].name] = "out of memory"
                print(f"[{s + 1}/{len(todo)}] {chunk[0].name}: OOM, skipped", flush=True)
                continue
            print("out of memory: this batch goes one image at a time", flush=True)
        except Exception as e:  # a bad file: retry one by one so only it fails
            print(f"batch failed ({type(e).__name__}: {e}); retrying one by one", flush=True)
        for j, p in enumerate(chunk):
            try:
                t = time.time()
                write(p, cap([_load_image(p)], prompt, max_new)[0], s + j + 1, time.time() - t)
            except Exception as e:  # one bad file must not stop the batch
                torch.cuda.empty_cache()
                failed[p.name] = f"{type(e).__name__}: {e}"
                print(f"[{s + j + 1}/{len(todo)}] {p.name}: {failed[p.name]}", flush=True)

    secs = time.time() - t0
    summary = {"model": model_id, "quant": quant, "done": done, "failed": failed, "seconds": round(secs)}
    (out / "_captions.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
