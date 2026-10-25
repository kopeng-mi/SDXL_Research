# %% [md]
# ## 5. Review
# Flags missing captions, captions over the word budget, and captions not starting with the trigger.
# FIND/REPLACE edits every caption (and its Drive copy). Run the last cell before closing the tab.
# %%
#@title Review
FIND = ""  #@param {type:"string"}
REPLACE = ""  #@param {type:"string"}
SHOW = 8  #@param {type:"integer"}
MAKE_ZIP = False  #@param {type:"boolean"}
from sdxlr import review
if FIND:
    print(f"replaced in {review.replace(LOCAL, FIND, REPLACE, DRIVE_DS if ROOT else None)} captions")
print(review.report(LOCAL, TARGET, TRIGGER))
nb.show_images(sorted(p for p in LOCAL.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp"))[:SHOW])
if MAKE_ZIP:
    z = review.zip_dataset(LOCAL, paths.WORK / f"{DS_NAME}.zip")
    from google.colab import files
    files.download(str(z))
