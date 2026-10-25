"""Write a detailed prompt with the text encoder that is already loaded (no extra model).

Z-Image's text encoder is a full Qwen3-4B with tied embeddings, so a generation
head costs nothing: a Qwen3ForCausalLM shell whose body is the loaded encoder and
whose lm_head shares its embedding weight (bnb quantizes Linear layers only, so
the embedding is still plain fp16). An abliterated encoder gives an uncensored
writer for free.

mode "official": the Z-Image Space's prompt-enhancer template (pe_template.py).
mode "engineer": no system prompt, as the Z-Image-Engineer card says.
"""

from __future__ import annotations

import re

import torch

GEN = {"do_sample": True, "temperature": 0.7, "top_p": 0.8, "top_k": 20}  # Qwen3 non-thinking defaults


def _lm(te):
    from transformers import Qwen3ForCausalLM
    from accelerate import init_empty_weights
    with init_empty_weights():
        lm = Qwen3ForCausalLM(te.config)
    lm.model = te
    lm.lm_head.weight = te.embed_tokens.weight
    return lm.eval()


def messages(text: str, mode: str) -> list[dict]:
    if mode == "engineer":
        return [{"role": "user", "content": text}]
    from .pe_template import prompt_template
    return [{"role": "user", "content": prompt_template.replace("{prompt}", text)}]


def clean(out: str) -> str:
    out = re.sub(r"<think>.*?</think>", "", out, flags=re.S)
    return out.strip().strip('"').strip("“”").strip()


@torch.inference_mode()
def write(pipe, text: str, mode: str = "official", max_new: int = 400) -> str:
    tok, te = pipe.tokenizer, pipe.text_encoder
    lm = _lm(te)
    chat = tok.apply_chat_template(messages(text, mode), tokenize=False, add_generation_prompt=True,
                                   enable_thinking=False)
    x = tok(chat, return_tensors="pt").to(te.device)
    ids = lm.generate(**x, max_new_tokens=max_new, pad_token_id=tok.eos_token_id, **GEN)
    return clean(tok.decode(ids[0, x["input_ids"].shape[1]:], skip_special_tokens=True))
