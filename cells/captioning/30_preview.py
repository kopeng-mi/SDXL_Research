# %% [md]
# ## 3. Preview
# Captions the first few photos into a scratch folder, so you can change the settings and run it again.
# %%
#@title Preview
N = 4  #@param {type:"integer"}
import shutil
PREVIEW = paths.WORK / "caption_preview"
shutil.rmtree(PREVIEW, ignore_errors=True)
nb.run(cap_args + ["--images", LOCAL, "--out", PREVIEW, "--limit", N, "--overwrite"])
nb.show_images(sorted(p for p in LOCAL.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp"))[:N],
               captions=PREVIEW)
