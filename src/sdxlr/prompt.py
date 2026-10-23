"""A1111-style prompt weights and long prompts (>77 tokens) for SDXL.

  (word)      x1.1        ((word))  x1.21
  [word]      x1/1.1      (word:1.3) x1.3      \\( \\) literal parens
  BREAK       start a new 75-token chunk

Each prompt is split into 75-token chunks, every chunk is encoded by both
SDXL text encoders, token embeddings are scaled by their weight (keeping the
original mean, like A1111), and chunks are concatenated.

clip_skip uses A1111 numbering: 1 = last layer, 2 = penultimate. SDXL models
(including Pony) are meant for 2, which is also diffusers' default.
"""

from __future__ import annotations

import re

import torch

_TOKEN = re.compile(r"\\\(|\\\)|\\\[|\\]|\\\\|\\|\(|\[|:\s*([+-]?[.\d]+)\s*\)|\)|]|[^\\()\[\]:]+|:")
_BREAK = re.compile(r"\s*\bBREAK\b\s*")


def parse(text: str) -> list[tuple[str, float]]:
    """'a (b:1.3) [c]' -> [('a ', 1.0), ('b', 1.3), (' ', 1.0), ('c', 0.909)]"""
    out: list[list] = []
    round_stack: list[int] = []
    square_stack: list[int] = []

    def scale(start: int, mult: float) -> None:
        for p in out[start:]:
            p[1] *= mult

    for m in _TOKEN.finditer(text):
        tok, weight = m.group(0), m.group(1)
        if tok.startswith("\\"):
            out.append([tok[-1], 1.0])
        elif tok == "(":
            round_stack.append(len(out))
        elif tok == "[":
            square_stack.append(len(out))
        elif weight is not None and round_stack:
            scale(round_stack.pop(), float(weight))
        elif tok == ")" and round_stack:
            scale(round_stack.pop(), 1.1)
        elif tok == "]" and square_stack:
            scale(square_stack.pop(), 1 / 1.1)
        else:
            out.append([tok, 1.0])
    for pos in round_stack:
        scale(pos, 1.1)
    for pos in square_stack:
        scale(pos, 1 / 1.1)

    merged: list[list] = []
    for t, w in out:
        if merged and merged[-1][1] == w:
            merged[-1][0] += t
        else:
            merged.append([t, w])
    return [(t, w) for t, w in merged if t]


def _tokens(tokenizer, text: str) -> list[list[tuple[int, float]]]:
    """Weighted token ids, split into chunks of at most 75 (BREAK forces a split)."""
    chunks: list[list[tuple[int, float]]] = []
    for part in _BREAK.split(text):
        cur: list[tuple[int, float]] = []
        for t, w in parse(part):
            for tid in tokenizer(t, add_special_tokens=False).input_ids:
                if len(cur) == 75:
                    chunks.append(cur)
                    cur = []
                cur.append((tid, w))
        chunks.append(cur)
    return chunks or [[]]


def _encode(tokenizer, encoder, chunks, clip_skip: int, device) -> tuple[torch.Tensor, torch.Tensor]:
    bos, eos = tokenizer.bos_token_id, tokenizer.eos_token_id
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos
    embeds, pooled = [], None
    for chunk in chunks:
        ids = [bos] + [t for t, _ in chunk] + [eos]
        ws = [1.0] + [w for _, w in chunk] + [1.0]
        ids += [pad] * (77 - len(ids))
        ws += [1.0] * (77 - len(ws))
        ids_t = torch.tensor([ids], device=device)
        out = encoder(ids_t, output_hidden_states=True)
        z = out.hidden_states[-clip_skip]
        if pooled is None and hasattr(out, "text_embeds"):
            pooled = out.text_embeds
        w = torch.tensor(ws, device=device, dtype=z.dtype)[None, :, None]
        mean = z.float().mean()
        z = z * w
        z = z * (mean / z.float().mean()).to(z.dtype)
        embeds.append(z)
    return torch.cat(embeds, dim=1), pooled


@torch.inference_mode()
def encode_sdxl(pipe, prompt: str, negative: str, clip_skip: int = 2):
    """Returns kwargs for an SDXL pipeline call (prompt_embeds etc.)."""
    clip_skip = max(1, clip_skip)  # 1 -> hidden_states[-1] (last), 2 -> [-2] (penultimate)
    dev = pipe.text_encoder.device
    tok1, tok2 = pipe.tokenizer, pipe.tokenizer_2
    c1p, c1n = _tokens(tok1, prompt), _tokens(tok1, negative)
    c2p, c2n = _tokens(tok2, prompt), _tokens(tok2, negative)
    # positive and negative must have the same number of chunks
    n = max(len(c1p), len(c1n), len(c2p), len(c2n))
    for c in (c1p, c1n, c2p, c2n):
        c += [[]] * (n - len(c))

    p1, _ = _encode(tok1, pipe.text_encoder, c1p, clip_skip, dev)
    n1, _ = _encode(tok1, pipe.text_encoder, c1n, clip_skip, dev)
    p2, pool_p = _encode(tok2, pipe.text_encoder_2, c2p, clip_skip, dev)
    n2, pool_n = _encode(tok2, pipe.text_encoder_2, c2n, clip_skip, dev)
    if not negative.strip():
        n1, n2, pool_n = torch.zeros_like(n1), torch.zeros_like(n2), torch.zeros_like(pool_n)
    dt = pipe.unet.dtype
    return {
        "prompt_embeds": torch.cat([p1, p2], dim=-1).to(dt),
        "negative_prompt_embeds": torch.cat([n1, n2], dim=-1).to(dt),
        "pooled_prompt_embeds": pool_p.to(dt),
        "negative_pooled_prompt_embeds": pool_n.to(dt),
    }
