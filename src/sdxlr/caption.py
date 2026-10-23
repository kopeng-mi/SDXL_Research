"""Caption a folder of images for LoRA training. Runs inside the main venv.

  python -m sdxlr.caption --images DIR --out DIR [--model qwen|joy] [--trigger ohwx_person]

One model is loaded once per process; the process exits when done, so all
VRAM/RAM goes back to the system before training starts.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from PIL import Image, ImageOps

MODELS = {
    "qwen": "Qwen/Qwen3-VL-4B-Instruct",
    "qwen-2b": "Qwen/Qwen3-VL-2B-Instruct",
    "joy": "fancyfeast/llama-joycaption-beta-one-hf-llava",
}

# Identity LoRA style: describe what should stay promptable, leave out what the
# LoRA should learn from the pixels (face, body, skin).
IDENTITY_PROMPT = (
    "Write a training caption for this photo in one or two sentences, 30 to 50 words. "
    "Describe the clothing, pose, camera angle, framing, background and lighting. "
    "Do not describe the person's face, facial features, body shape, skin, age or ethnicity. "
    "Refer to the person only as 'a woman' or 'a man'. "
    "Do not start with 'The image' or 'This photo'. Plain text only."
)

EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
MAX_SIDE = 1024  # caps vision tokens; captions don't need more detail


def _dtype() -> torch.dtype:
    if not torch.cuda.is_available():
        return torch.float32
    # T4 (sm75) has no fast bf16
    return torch.bfloat16 if torch.cuda.is_bf16_supported(including_emulation=False) else torch.float16


def _load_image(path: Path) -> Image.Image:
    img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    img.thumbnail((MAX_SIDE, MAX_SIDE), Image.LANCZOS)
    return img


class Qwen:
    def __init__(self, model_id: str, dtype: torch.dtype):
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        self.proc = AutoProcessor.from_pretrained(model_id)
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_id, dtype=dtype, device_map="auto" if torch.cuda.is_available() else None,
            attn_implementation="sdpa",
        ).eval()

    @torch.inference_mode()
    def __call__(self, img: Image.Image, prompt: str, max_new_tokens: int) -> str:
        msgs = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": prompt}]}]
        x = self.proc.apply_chat_template(
            msgs, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt",
        ).to(self.model.device)
        ids = self.model.generate(**x, max_new_tokens=max_new_tokens, do_sample=False)
        return self.proc.batch_decode(ids[:, x["input_ids"].shape[1]:], skip_special_tokens=True)[0]


class Joy:
    def __init__(self, model_id: str, dtype: torch.dtype):
        from transformers import AutoProcessor, BitsAndBytesConfig, LlavaForConditionalGeneration

        self.dtype = dtype
        self.proc = AutoProcessor.from_pretrained(model_id)
        # nf4 keeps the 8B LLM near 8 GB; the vision tower stays fp16 for quality
        quant = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=dtype,
            llm_int8_skip_modules=["vision_tower", "multi_modal_projector"],
        )
        self.model = LlavaForConditionalGeneration.from_pretrained(
            model_id, dtype=dtype, quantization_config=quant, device_map="auto",
        ).eval()

    @torch.inference_mode()
    def __call__(self, img: Image.Image, prompt: str, max_new_tokens: int) -> str:
        convo = [
            {"role": "system", "content": "You are a helpful image captioner."},
            {"role": "user", "content": prompt},
        ]
        text = self.proc.apply_chat_template(convo, tokenize=False, add_generation_prompt=True)
        x = self.proc(text=[text], images=[img], return_tensors="pt").to(self.model.device)
        x["pixel_values"] = x["pixel_values"].to(self.dtype)
        ids = self.model.generate(**x, max_new_tokens=max_new_tokens, do_sample=False)
        return self.proc.batch_decode(ids[:, x["input_ids"].shape[1]:], skip_special_tokens=True)[0]


def _clean(text: str) -> str:
    return " ".join(text.replace("\n", " ").split()).strip().strip('"')


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", type=Path, required=True)
    ap.add_argument("--out", type=Path, help="caption dir (default: next to images)")
    ap.add_argument("--model", default="qwen", help=f"{', '.join(MODELS)} or a HF model id")
    ap.add_argument("--trigger", default="ohwx_person", help="prepended to every caption; '' for none")
    ap.add_argument("--prompt", default=IDENTITY_PROMPT)
    ap.add_argument("--max-new-tokens", type=int, default=120)
    ap.add_argument("--limit", type=int, default=0, help="caption only the first N (preview)")
    ap.add_argument("--overwrite", action="store_true")
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

    model_id = MODELS.get(a.model, a.model)
    dtype = _dtype()
    t0 = time.time()
    cap = (Joy if "joycaption" in model_id else Qwen)(model_id, dtype)
    print(f"loaded {model_id} ({dtype}) in {time.time() - t0:.0f}s", flush=True)

    done, failed = 0, {}
    for i, p in enumerate(todo, 1):
        try:
            t = time.time()
            text = _clean(cap(_load_image(p), a.prompt, a.max_new_tokens))
            final = f"{a.trigger}, {text}" if a.trigger else text
            (out / f"{p.stem}.txt").write_text(final, encoding="utf-8")
            done += 1
            print(f"[{i}/{len(todo)}] {p.name} ({time.time() - t:.1f}s): {final}", flush=True)
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            failed[p.name] = "out of memory"
            print(f"[{i}/{len(todo)}] {p.name}: OOM, skipped", flush=True)
        except Exception as e:  # one bad file must not stop the batch
            failed[p.name] = f"{type(e).__name__}: {e}"
            print(f"[{i}/{len(todo)}] {p.name}: {failed[p.name]}", flush=True)

    summary = {"model": model_id, "done": done, "failed": failed, "seconds": round(time.time() - t0)}
    (out / "_captions.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
