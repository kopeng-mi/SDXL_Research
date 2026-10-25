# %% [md]
# ## 1. Dataset
# Copies `SDXLR/datasets/<DS_NAME>` from Drive to the VM (training reads many small files; Drive is slow for that).
# Photos without a caption can be captioned here (Qwen3-VL 4B, identity style, 80 to 150 words); the captioning notebook has more options.
# %%
#@title Dataset
DS_NAME = "my_dataset"  #@param {type:"string"}
CAPTION_MISSING = True  #@param {type:"boolean"}
TRIGGER = "ohwx"  #@param {type:"string"}
CLASS_WORD = "woman"  #@param {type:"string"}
from sdxlr import drive, plan, paths
IMGS = (".png", ".jpg", ".jpeg", ".webp")
LOCAL = paths.DATASETS / DS_NAME
LOCAL.mkdir(parents=True, exist_ok=True)
DRIVE_DS = ROOT / "datasets" / DS_NAME if ROOT else None
if DRIVE_DS:
    assert DRIVE_DS.exists(), f"{DRIVE_DS} not found: run the captioning notebook first, or check DS_NAME"
    print(f"from Drive: {drive.pull(DRIVE_DS, LOCAL)} files copied")
imgs = sorted(p for p in LOCAL.iterdir() if p.suffix.lower() in IMGS)
missing = [p for p in imgs if not p.with_suffix(".txt").exists()]
print(f"{len(imgs)} images, {len(imgs) - len(missing)} captioned")
if missing and CAPTION_MISSING:
    args = ["-m", "sdxlr.caption", "--images", LOCAL, "--style", "id", "--target", "zimage",
            "--trigger", TRIGGER, "--class-word", CLASS_WORD]
    nb.run(args + (["--mirror", DRIVE_DS] if DRIVE_DS else []))
