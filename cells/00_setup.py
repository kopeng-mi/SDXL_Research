# %% [md]
# # SDXL_Research
# Caption, train, compare and generate SDXL LoRAs on the Colab free T4.
#
# Run **Setup** once per session. Re-runs skip finished steps.
# %%
#@title Setup: repo + shared venv (idempotent)
import os, subprocess, sys
from pathlib import Path

REPO_URL = "https://github.com/kopeng-mi/SDXL_Research.git"  #@param {type:"string"}
BRANCH = "main"  #@param {type:"string"}
REPO = Path("/content/SDXL_Research")

if not REPO.exists():
    subprocess.run(["git", "clone", "-q", "--recurse-submodules", "-b", BRANCH, REPO_URL, str(REPO)], check=True)
else:
    subprocess.run(["git", "pull", "-q", "--ff-only"], cwd=REPO, check=True)
    subprocess.run(["git", "submodule", "update", "-q", "--init", "--recursive"], cwd=REPO, check=True)

# venv is built by the system Python; the GPU work later runs in venv subprocesses,
# so this kernel never imports torch and never holds VRAM.
env = {**os.environ, "PYTHONPATH": str(REPO / "src")}
subprocess.run([sys.executable, "-m", "sdxlr.bootstrap"], env=env, check=True)
