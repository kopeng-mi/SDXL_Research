# %% [md]
# ## Finish
# Saves Drive, then **Runtime > Disconnect and delete runtime** (pictures on the VM are gone with it).
# %%
#@title Finish
if ROOT:
    from sdxlr import drive
    drive.flush_and_unmount()
