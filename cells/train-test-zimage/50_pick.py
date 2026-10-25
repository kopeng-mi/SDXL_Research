# %% [md]
# ## 5. Pick
# Copies the epoch you like to `SDXLR/loras/zimage/<RUN>_e<N>.safetensors` (with its trigger-word sidecar).
# The Z-Image app lists that folder.
# %%
#@title Pick an epoch
PICK_EPOCH = 0  #@param {type:"integer"}
from sdxlr import loraformat
dst = loraformat.pick(run_dir, PICK_EPOCH, (ROOT or paths.WORK) / "loras" / "zimage", RUN)
print(f"saved {dst}")
