"""Resource bars for the header: what sits in VRAM, RAM and disk, by name.

  bars(eng, jm, preview)  -> HTML with three stacked bars
  estimate(kind, ...)     -> (label, GB) VRAM a model will take once loaded

VRAM is split into the loaded model's parts (measured from the weights),
LoRA adapters, the CUDA cache, and GPU jobs (child processes, via nvidia-smi).
A preview block (hatched) shows what a model picked in Generate would take;
loading it replaces the current model, so the bar shows it in that place.
"""

from __future__ import annotations

import html
import os
import shutil
import time
from pathlib import Path

import psutil
import torch

GB = 2**30
COLORS = ["#6366f1", "#22c55e", "#f59e0b", "#ec4899", "#06b6d4", "#a855f7", "#84cc16", "#f97316"]
HF_CACHE = Path.home() / ".cache" / "huggingface"
# VRAM once loaded (measured on the T4; file size is a fair guess for others)
TE_GB = {"nf4": 2.5, "int8": 4.4}
Z_EXTRA_GB = 0.4  # VAE + norms outside the GGUF file
SDXL_GB = 6.9  # fp16 UNet + both CLIPs + VAE

_parts_cache: dict = {}


def _bytes(module) -> int:
    seen, n = set(), 0
    for t in [*module.parameters(), *module.buffers()]:
        if t.data_ptr() not in seen and t.device.type == "cuda":
            seen.add(t.data_ptr())
            n += t.numel() * t.element_size()
    return n


def model_parts(eng) -> list[tuple[str, float]]:
    """[(part, GB)] for the loaded pipeline; LoRA weights counted separately."""
    if eng.pipe is None:
        return []
    key = (eng.model_path, tuple(eng.adapters.values()))
    if key in _parts_cache:
        return _parts_cache[key]
    names = {"unet": "UNet", "transformer": "DiT", "text_encoder": "text encoder", "text_encoder_2": "text encoder 2",
             "vae": "VAE"}
    out, lora = [], 0
    for attr, label in names.items():
        m = getattr(eng.pipe, attr, None)
        if m is None:
            continue
        lb = sum(p.numel() * p.element_size() for n, p in m.named_parameters() if "lora_" in n and p.device.type == "cuda")
        lora += lb
        out.append((label, (_bytes(m) - lb) / GB))
    if lora:
        out.append((f"LoRAs ({len(eng.adapters)})", lora / GB))
    _parts_cache.clear()
    _parts_cache[key] = out
    return out


_nvml: list = []


def _gpu_procs() -> dict[int, float]:
    """pid -> GB for processes using the GPU, via NVML (in-process, no nvidia-smi subprocess)."""
    try:
        import pynvml
        if not _nvml:
            pynvml.nvmlInit()
            _nvml.append(pynvml.nvmlDeviceGetHandleByIndex(0))
        return {p.pid: (p.usedGpuMemory or 0) / GB for p in pynvml.nvmlDeviceGetComputeRunningProcesses(_nvml[0])}
    except Exception:
        return {}


def _tree_pids(pid: int) -> set[int]:
    try:
        p = psutil.Process(pid)
        return {pid, *(c.pid for c in p.children(recursive=True))}
    except psutil.Error:
        return set()


def _tree_rss(pid: int) -> float:
    n = 0
    for q in _tree_pids(pid):
        try:
            n += psutil.Process(q).memory_info().rss
        except psutil.Error:
            pass
    return n / GB


def _dir_gb(p: Path) -> float:
    if not p.exists():
        return 0.0
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file() and not f.is_symlink()) / GB


def estimate(kind: str, sdxl_path: str | None, z_path: str | None, te_quant: str) -> tuple[str, float] | None:
    if kind == "Z-Image":
        size = Path(z_path).stat().st_size / GB if z_path and Path(z_path).exists() else 7.2  # Q8 default
        return "Z-Image (to load)", size + TE_GB.get(te_quant, 2.5) + Z_EXTRA_GB
    if sdxl_path:
        return "SDXL (to load)", SDXL_GB
    return None


def _bar(title: str, total: float, blocks: list[tuple[str, float]], preview: tuple[str, float] | None = None) -> str:
    segs, used = [], 0.0
    for i, (label, gb) in enumerate(blocks):
        if gb < 0.05:
            continue
        used += gb
        w = 100 * gb / total
        segs.append(f'<div class="rb-seg" style="width:{w:.2f}%;background:{COLORS[i % len(COLORS)]}" '
                    f'title="{html.escape(label)}: {gb:.1f} GB">{html.escape(label) if w > 9 else ""}</div>')
    if preview:
        label, gb = preview
        w = min(100 * gb / total, 100 - 100 * used / total) if gb else 0
        over = gb > total - used
        segs.append(f'<div class="rb-seg rb-ghost{" rb-over" if over else ""}" style="width:{max(w, 0):.2f}%" '
                    f'title="{html.escape(label)}: ~{gb:.1f} GB">{html.escape(label) if w > 12 else ""}</div>')
    shown = used
    return (f'<div class="rb-row"><div class="rb-name">{title}</div><div class="rb-bar">{"".join(segs)}</div>'
            f'<div class="rb-num">{shown:.1f} / {total:.1f} GB</div></div>')


def usage(eng, jm) -> dict[str, tuple[float, list[tuple[str, float]]]]:
    """{"VRAM"|"RAM"|"Disk": (total GB, [(block, GB), ...])}."""
    out = {}
    if torch.cuda.is_available():
        free, total = (x / GB for x in torch.cuda.mem_get_info())
        parts = model_parts(eng)
        own = sum(g for _, g in parts)
        cache = max(torch.cuda.memory_reserved() / GB - own, 0)
        name = "SDXL" if eng.kind == "sdxl" else "Z-Image"
        blocks = [(f"{name} {p}", g) for p, g in parts]
        if cache > 0.05:
            blocks.append(("working memory", cache))
        gpu = _gpu_procs()
        for j in jm.jobs.values():
            if j.status == "running" and j.gpu and j.proc:
                blocks.append((f"job: {j.kind}", sum(gpu.get(p, 0) for p in _tree_pids(j.proc.pid))))
        known = sum(g for _, g in blocks)
        if total - free - known > 0.3:
            blocks.append(("other (CUDA context)", total - free - known))
        out["VRAM"] = (total, blocks)
    m = psutil.virtual_memory()
    running = [j for j in jm.jobs.values() if j.status == "running" and j.proc]
    blocks = [("app", _tree_rss(os.getpid()) - sum(_tree_rss(j.proc.pid) for j in running))]
    blocks += [(f"job: {j.kind}", _tree_rss(j.proc.pid)) for j in running]
    blocks.append(("system + other", max((m.total - m.available) / GB - sum(g for _, g in blocks), 0)))
    out["RAM"] = (m.total / GB, blocks)
    out["Disk"] = _disk()
    return out


def bars(eng, jm, preview: tuple[str, float] | None = None) -> str:
    rows = []
    for name, (total, blocks) in usage(eng, jm).items():
        if name == "VRAM" and preview and eng.pipe is not None:  # loading replaces the current model
            blocks = [b for b in blocks if not b[0].startswith(("SDXL", "Z-Image", "working"))]
        rows.append(_bar(name, total, blocks, preview if name == "VRAM" else None))
    return '<div class="rb">' + "".join(rows) + "</div>"


_disk_cache: list = [0.0, None]


def _disk() -> tuple[float, list[tuple[str, float]]]:
    """Folder sizes are walked at most every 15 s."""
    if time.time() - _disk_cache[0] > 15:
        d = shutil.disk_usage("/content" if Path("/content").exists() else ".")
        work = Path("/content/work")
        blocks = [("models", _dir_gb(work / "models")), ("download cache", _dir_gb(HF_CACHE)),
                  ("LoRAs + runs", _dir_gb(work / "loras") + _dir_gb(work / "runs")),
                  ("datasets + outputs", _dir_gb(work / "datasets") + _dir_gb(work / "outputs"))]
        blocks.append(("system + venv", max((d.total - d.free) / GB - sum(g for _, g in blocks), 0)))
        _disk_cache[:] = [time.time(), (d.total / GB, blocks)]
    return _disk_cache[1]


def clear_download_cache(keep: list[str]) -> float:
    """Delete HF hub cache repos (re-downloadable) except those whose id appears in keep; returns GB freed."""
    hub = HF_CACHE / "hub"
    freed = 0.0
    for d in hub.glob("models--*") if hub.exists() else []:
        repo = d.name[len("models--"):].replace("--", "/")
        if any(repo in k for k in keep):
            continue
        freed += _dir_gb(d)
        shutil.rmtree(d, ignore_errors=True)
    return freed


CSS = """
.rb {display:flex; flex-direction:column; gap:4px; font-size:12px;}
.rb-row {display:grid; grid-template-columns:44px 1fr 110px; align-items:center; gap:8px;}
.rb-name {font-weight:600; opacity:.8;}
.rb-num {text-align:right; opacity:.8; font-variant-numeric:tabular-nums;}
.rb-bar {display:flex; height:16px; border-radius:6px; overflow:hidden; background:var(--neutral-200);}
.dark .rb-bar {background:var(--neutral-700);}
.rb-seg {height:100%; color:#fff; font-size:10px; line-height:16px; padding:0 4px; white-space:nowrap;
         overflow:hidden; text-overflow:ellipsis; border-right:1px solid rgba(255,255,255,.35);}
.rb-ghost {background:repeating-linear-gradient(45deg,#94a3b8 0 6px,#cbd5e1 6px 12px); color:#1e293b;}
.rb-over {background:repeating-linear-gradient(45deg,#ef4444 0 6px,#fca5a5 6px 12px);}
"""
