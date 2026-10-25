# %% [md]
# ## 3. Launch
# Open **http://sdxl:7865** on a device signed in to your Tailscale (needs TS_AUTHKEY in Secrets).
# Without a key it falls back to a public gradio.live link (anyone with the link can use it).
# The cell keeps running while Fooocus is open; stop it to close Fooocus.
# %%
#@title Launch Fooocus
USE_TAILSCALE = True  #@param {type:"boolean"}
OUTPUTS_TO_DRIVE = False  #@param {type:"boolean"}
EXTRA_ARGS = ""  #@param {type:"string"}
import os
from sdxlr import tailnet
out = ROOT / "outputs" / "fooocus" if ROOT and OUTPUTS_TO_DRIVE else fooocus.OUTPUTS
fooocus.write_config(ckpt.name, CKPT_DIRS, LORA_DIRS, out)
share = True
if USE_TAILSCALE and os.environ.get("TS_AUTHKEY"):
    print("open", tailnet.up(fooocus.PORT, hostname="sdxl"))
    share = False
elif USE_TAILSCALE:
    print("TS_AUTHKEY is missing: using a public gradio.live link instead")
os.environ["MPLBACKEND"] = "Agg"
nb.serve(fooocus.launch_args(share=share, extra=EXTRA_ARGS), cwd=fooocus.DIR, py=str(fooocus.PY))
