# %% [md]
# # SDXL_Research
# Caption, train, compare and generate SDXL and Z-Image LoRAs on the Colab free T4, and edit photos.
#
# 1. **Runtime > Change runtime type > T4 GPU**
# 2. **Secrets** (key icon, left): add `HF_TOKEN` (needed for private repos and sync);
#    `TS_AUTHKEY` (Tailscale auth key, tick Reusable + Ephemeral) to open the app at http://sdxlr:7860;
#    optional `CIVITAI_TOKEN` for Civitai downloads. Allow notebook access.
# 3. Run **Setup** (installs everything; re-runs skip finished steps), then **Launch**.
# %%
#@title Setup: repo + venv (idempotent)
import os, subprocess, sys
from pathlib import Path

REPO_URL = "https://github.com/kopeng-mi/SDXL_Research.git"  #@param {type:"string"}
BRANCH = "step4-overhaul"  #@param {type:"string"}
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
r = subprocess.run([sys.executable, "-m", "sdxlr.bootstrap", "--groups", "caption,train-sdxl,train-z,app"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                   env={**os.environ, "PYTHONPATH": str(REPO / "src")})
print(r.stdout[-3000:])
assert r.returncode == 0, "setup failed (see above)"
found = nb.load_secrets()
print("secrets:", ", ".join(f"{k} {'found' if v else 'missing'}" for k, v in found.items()))
