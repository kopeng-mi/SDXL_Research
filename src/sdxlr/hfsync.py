"""Keep the newest LoRA + training state of a run in a private HF repo, for resume.

The repo holds one folder per run:
  NAME/ckpt/     only the latest LoRA + state; each push replaces it in a
                 single commit (older checkpoint removed), so the repo stays small
  NAME/dataset/  the training images + captions, so a new session needs nothing local

  push_dataset(repo, name, images_dir)  # once at start (skipped if unchanged)
  pull_dataset(repo, name, images_dir)  # when the local dataset is missing
  Pusher(repo, out_dir, name).start()   # background thread while training
  pull_latest(repo, name, dst)          # -> (state_dir, epochs_done) or None

State dir names (kohya + musubi): NAME-000003-state, or NAME_r3-000002-state
for a resumed musubi run (musubi restarts its epoch counter on resume).
"""

from __future__ import annotations

import re
import shutil
import tempfile
import threading
import time
from pathlib import Path

PUSH_EVERY = 120  # seconds between checks for a new checkpoint


def done(state: str, name: str) -> int:
    """Epochs finished according to a state dir name, -1 if it isn't this run's."""
    m = re.fullmatch(rf"{re.escape(name)}(?:_r(\d+))?-(\d+)-state", state)
    return int(m[1] or 0) + int(m[2]) if m else -1


def _api():
    from huggingface_hub import HfApi
    return HfApi()


def _files(repo: str) -> list[str]:
    try:
        return _api().list_repo_files(repo)
    except Exception as e:  # repo not created yet / no access
        print(f"hf: {repo} not readable ({type(e).__name__})", flush=True)
        return []


def _manifest(images: Path) -> str:
    return "\n".join(f"{p.name} {p.stat().st_size}" for p in sorted(images.iterdir()) if p.is_file())


def push_dataset(repo: str, name: str, images: Path) -> None:
    from huggingface_hub import hf_hub_download
    _api().create_repo(repo, private=True, exist_ok=True)
    man = _manifest(images)
    if f"{name}/dataset/_manifest.txt" in _files(repo):
        if Path(hf_hub_download(repo, f"{name}/dataset/_manifest.txt")).read_text() == man:
            print("hf: dataset already uploaded", flush=True)
            return
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / "_manifest.txt").write_text(man)
        _api().upload_folder(repo_id=repo, folder_path=str(images), path_in_repo=f"{name}/dataset",
                             delete_patterns="*", ignore_patterns=["_*"], commit_message=f"{name}: dataset")
        _api().upload_file(path_or_fileobj=str(Path(tmp) / "_manifest.txt"), path_in_repo=f"{name}/dataset/_manifest.txt",
                           repo_id=repo, commit_message=f"{name}: dataset manifest")
    print(f"hf: dataset uploaded to {repo}/{name}/dataset", flush=True)


def pull_dataset(repo: str, name: str, images: Path) -> bool:
    from huggingface_hub import snapshot_download
    if not any(f.startswith(f"{name}/dataset/") for f in _files(repo)):
        return False
    with tempfile.TemporaryDirectory() as tmp:
        snapshot_download(repo, allow_patterns=[f"{name}/dataset/*"], local_dir=tmp)
        images.mkdir(parents=True, exist_ok=True)
        for f in (Path(tmp) / name / "dataset").iterdir():
            if not f.name.startswith("_"):
                shutil.move(str(f), images / f.name)
    print(f"hf: dataset restored from {repo}/{name}/dataset", flush=True)
    return True


def pull_latest(repo: str, name: str, dst: Path) -> tuple[Path, int] | None:
    from huggingface_hub import snapshot_download
    pre = f"{name}/ckpt/"
    states = {f[len(pre):].split("/")[0] for f in _files(repo) if f.startswith(pre) and "-state/" in f}
    best = max(states, key=lambda s: done(s, name), default=None)
    if best is None or done(best, name) < 0:
        return None
    snapshot_download(repo, allow_patterns=[f"{pre}{best}/*"], local_dir=dst)
    return dst / name / "ckpt" / best, done(best, name)


class Pusher(threading.Thread):
    """Uploads the newest state dir (+ the LoRA saved with it) whenever a new one appears."""

    def __init__(self, repo: str, out: Path, name: str):
        super().__init__(daemon=True)
        self.repo, self.out, self.name = repo, out, name
        self.stop_ev = threading.Event()
        self.pushed: str | None = None
        _api().create_repo(repo, private=True, exist_ok=True)

    def newest(self) -> Path | None:
        states = [p for p in self.out.glob("*-state") if p.is_dir() and done(p.name, self.name) >= 0]
        return max(states, key=lambda p: done(p.name, self.name), default=None)

    def push(self, final: bool = False) -> None:
        st = self.newest()
        if st is None or (st.name == self.pushed and not final):
            return
        time.sleep(5)  # let the trainer finish writing the state dir
        ep = done(st.name, self.name)
        lora_stem = st.name[: -len("-state")]
        with tempfile.TemporaryDirectory() as tmp:
            t = Path(tmp)
            shutil.copytree(st, t / st.name)
            real = f"{self.name}-{ep:06d}"  # resumed musubi files get renamed to this at the end
            for f in {*self.out.glob(f"{lora_stem}*.safetensors"), *self.out.glob(f"{real}*.safetensors")}:
                shutil.copy2(f, t / f.name.replace(lora_stem, real))  # LoRA of that epoch (+ diffusers copy)
            if final:  # finished run: also keep the final LoRA(s)
                for f in self.out.glob(f"{self.name}.safetensors"):
                    shutil.copy2(f, t / f.name)
                for f in self.out.glob(f"{self.name}*_diffusers.safetensors"):
                    shutil.copy2(f, t / f.name)
            (t / "progress.txt").write_text(f"epochs_done={ep}\nstate={st.name}\n")
            _api().upload_folder(repo_id=self.repo, folder_path=tmp, path_in_repo=f"{self.name}/ckpt", delete_patterns="*",
                                 commit_message=f"{self.name}: epoch {ep}")
        self.pushed = st.name
        print(f"hf: pushed {self.name} epoch {ep} to {self.repo} (older checkpoint removed)", flush=True)

    def run(self) -> None:
        while not self.stop_ev.wait(PUSH_EVERY):
            try:
                self.push()
            except Exception as e:  # network hiccup: try again next round
                print(f"hf: push failed ({type(e).__name__}: {e}), will retry", flush=True)

    def finish(self) -> None:
        self.stop_ev.set()
        try:
            self.push(final=True)
        except Exception as e:
            print(f"hf: final push failed ({type(e).__name__}: {e})", flush=True)
