# %% [md]
# # SDXL_Research
# Caption, train, compare and generate SDXL LoRAs on the Colab free T4.
#
# 1. **Runtime > Change runtime type > T4 GPU**
# 2. Add your Hugging Face token in **Secrets** (key icon) as `HF_TOKEN` (needed for private repos)
# 3. Run **Setup** once per session; re-runs skip finished steps.
# %%
#@title Setup: repo + venv (idempotent)
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

# The kernel only orchestrates (stdlib); GPU work runs in venv subprocesses,
# so this kernel never holds VRAM.
sys.path.insert(0, str(REPO / "src"))
from sdxlr import nb
r = subprocess.run([sys.executable, "-m", "sdxlr.bootstrap"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                   env={**os.environ, "PYTHONPATH": str(REPO / "src")})
print(r.stdout[-3000:])
assert r.returncode == 0, "setup failed (see above)"
nb.hf_token()
