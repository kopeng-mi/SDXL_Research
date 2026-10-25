# %% [md]
# ## 5. Pick
# Copies the epoch you like to `SDXLR/loras/sdxl/<RUN>_e<N>.safetensors` (with its trigger-word sidecar).
# Fooocus and the Z-Image app list that folder.
# %%
#@title Pick an epoch
PICK_EPOCH = 0  #@param {type:"integer"}
import shutil, sys
from sdxlr import compare_pick
dst = compare_pick.pick(run_dir, PICK_EPOCH, (ROOT or paths.WORK) / "loras" / "sdxl", RUN)
print(f"saved {dst}")
