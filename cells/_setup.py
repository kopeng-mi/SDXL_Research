# %% [md]
# # __TITLE__
# __INTRO__
#
# 1. **Runtime > Change runtime type > T4 GPU**
# 2. **Secrets** (key icon, left), allow notebook access: `HF_TOKEN` (gated or private repos),
#    `CIVITAI_TOKEN` (Civitai downloads), `TS_AUTHKEY` (Tailscale, tick Reusable + Ephemeral; only the app notebooks).
# 3. Run **Setup** (re-runs skip finished steps), then the cells below in order.
# %%
#@title Setup: repo, venv, secrets, Drive
import os, subprocess, sys
from pathlib import Path

REPO_URL = "https://github.com/kopeng-mi/SDXL_Research.git"  #@param {type:"string"}
BRANCH = "step5-notebooks"  #@param {type:"string"}
USE_DRIVE = __DRIVE__  #@param {type:"boolean"}
DRIVE_ROOT = "/content/drive/MyDrive/SDXLR"  #@param {type:"string"}
GROUPS = __GROUPS__  # venv install groups for this notebook (None = no shared venv)
REPO = Path("/content/SDXL_Research")

if not REPO.exists():
    subprocess.run(["git", "clone", "-q", "--recurse-submodules", "-b", BRANCH, REPO_URL, str(REPO)], check=True)
else:
    # follow BRANCH even if an earlier run cloned another one
    subprocess.run(["git", "fetch", "-q", "origin", BRANCH], cwd=REPO, check=True)
    subprocess.run(["git", "checkout", "-q", "-B", BRANCH, f"origin/{BRANCH}"], cwd=REPO, check=True)
    subprocess.run(["git", "submodule", "update", "-q", "--init", "--recursive"], cwd=REPO, check=True)

# The kernel only orchestrates (stdlib); GPU work runs in venv subprocesses,
# so this kernel never holds VRAM.
sys.path.insert(0, str(REPO / "src"))
from sdxlr import nb
if GROUPS is not None:
    r = subprocess.run([sys.executable, "-m", "sdxlr.bootstrap", "--groups", GROUPS], stdout=subprocess.PIPE,
                       stderr=subprocess.STDOUT, text=True, env={**os.environ, "PYTHONPATH": str(REPO / "src")})
    print(r.stdout[-3000:])
    assert r.returncode == 0, "setup failed (see above)"
found = nb.load_secrets()
print("secrets:", ", ".join(f"{k} {'found' if v else 'missing'}" for k, v in found.items()))
ROOT = nb.drive_mount(DRIVE_ROOT) if USE_DRIVE else None
print("drive:", ROOT or "off")
