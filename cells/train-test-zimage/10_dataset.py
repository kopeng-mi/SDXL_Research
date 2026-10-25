# %% [md]
# ## 1. Dataset
# Builds the dataset on the VM from any mix of: the Drive dataset `SDXLR/datasets/<DS_NAME>`, a **.zip you upload**
# (tick UPLOAD_ZIP; images with their .txt captions, folders inside are fine), and SRC links.
# Duplicates are skipped and phone photos are turned upright. The result is copied back to Drive.
# Photos without a caption can be captioned here (Qwen3-VL 4B, identity style, 80 to 150 words).
# %%
#@title Dataset
DS_NAME = "my_dataset"  #@param {type:"string"}
UPLOAD_ZIP = True  #@param {type:"boolean"}
SRC = ""  #@param {type:"string"}
#@markdown SRC: comma list of .zip links, HF files `org/repo/file.zip`, or Drive paths (optional).
CAPTION_MISSING = True  #@param {type:"boolean"}
TRIGGER = "ohwx"  #@param {type:"string"}
CLASS_WORD = "woman"  #@param {type:"string"}
from pathlib import Path
from sdxlr import drive, plan, paths
IMGS = (".png", ".jpg", ".jpeg", ".webp")
LOCAL = paths.DATASETS / DS_NAME
LOCAL.mkdir(parents=True, exist_ok=True)
DRIVE_DS = ROOT / "datasets" / DS_NAME if ROOT else None
if DRIVE_DS and DRIVE_DS.exists():
    print(f"from Drive: {drive.pull(DRIVE_DS, LOCAL)} files copied")
srcs = [s.strip() for s in SRC.split(",") if s.strip()]
if UPLOAD_ZIP:
    from google.colab import files
    up = paths.WORK / "uploads"
    up.mkdir(parents=True, exist_ok=True)
    for name, data in files.upload().items():  # a file picker opens below the cell
        (up / name).write_bytes(data)
        srcs.append(str(up / name))
if srcs:
    nb.run(["-m", "sdxlr.data", "--dest", LOCAL, *srcs, "--res", 512, "--kind", "zimage"])
    for s in srcs:  # uploads are copied into the dataset; free the disk
        if s.startswith(str(paths.WORK / "uploads")):
            Path(s).unlink(missing_ok=True)
imgs = sorted(p for p in LOCAL.iterdir() if p.suffix.lower() in IMGS)
assert imgs, "no images: tick UPLOAD_ZIP, fill SRC, or check DS_NAME"
missing = [p for p in imgs if not p.with_suffix(".txt").exists()]
print(f"{len(imgs)} images, {len(imgs) - len(missing)} captioned")
if missing and CAPTION_MISSING:
    args = ["-m", "sdxlr.caption", "--images", LOCAL, "--style", "id", "--target", "zimage",
            "--trigger", TRIGGER, "--class-word", CLASS_WORD]
    nb.run(args)
if DRIVE_DS:
    print(f"to Drive: {drive.pull(LOCAL, DRIVE_DS)} files copied")
