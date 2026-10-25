# %% [md]
# ## 6. Download LoRAs
# Packs the epochs you choose into one zip and downloads it: the `_diffusers` files (what the Z-Image app and
# ComfyUI-style loaders use) plus each one's trigger-word sidecar. Tick RAW to add musubi's original files too.
# %%
#@title Download LoRAs as a zip
EPOCHS_TO_ZIP = "last:3"  #@param {type:"string"}
#@markdown last:K, all, or a list like 2,4
RAW = False  #@param {type:"boolean"}
import zipfile
from google.colab import files
from sdxlr import loraformat
if ROOT and not any(run_dir.glob("*_diffusers.safetensors")):  # a new VM: zip what Drive has
    drive.pull(ROOT / "loras" / "zimage" / RUN, run_dir)
picked = loraformat.pick_epochs(run_dir, EPOCHS_TO_ZIP)
assert picked, f"no LoRAs in {run_dir} for {EPOCHS_TO_ZIP}"
z = paths.WORK / f"{RUN}_loras.zip"
with zipfile.ZipFile(z, "w", zipfile.ZIP_STORED) as f:  # safetensors don't compress
    for e, p in picked:
        name = f"{RUN}_e{e}"
        f.write(p, f"{name}.safetensors")
        if loraformat.sidecar_path(p).exists():
            f.write(loraformat.sidecar_path(p), f"{name}.json")
        raw = p.with_name(p.name.replace("_diffusers", ""))
        if RAW and raw.exists() and raw != p:
            f.write(raw, f"raw/{name}.safetensors")
print(f"{z.name}: epochs {[e for e, _ in picked]}, {z.stat().st_size / 2**20:.0f} MB")
files.download(str(z))
