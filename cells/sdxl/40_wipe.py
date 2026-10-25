# %% [md]
# ## 4. Finish
# Deletes the pictures on the VM (the Drive copy stays when OUTPUTS_TO_DRIVE was on), signs out of Tailscale,
# and saves Drive. Then **Runtime > Disconnect and delete runtime**.
# %%
#@title Wipe outputs and finish
print(f"deleted {fooocus.wipe_outputs()} files")
tailnet.down()
if ROOT:
    from sdxlr import drive
    drive.flush_and_unmount()
