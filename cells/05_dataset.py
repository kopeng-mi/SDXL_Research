# %% [md]
# ## Dataset
# Puts images (and any existing `.txt` captions) into one folder.
# `test` downloads 5 public dog photos to try the pipeline end to end.
# %%
#@title Get dataset
SOURCE = "test"  #@param ["test", "hf", "drive"]
DATASET_DIR = "/content/dataset"  #@param {type:"string"}
#@markdown **hf**: private/public repo + zip filename
HF_REPO = ""  #@param {type:"string"}
HF_FILE = "dataset.zip"  #@param {type:"string"}
#@markdown **drive**: folder or .zip in your Drive
DRIVE_PATH = "/content/drive/MyDrive/datasets/my_images"  #@param {type:"string"}

args = ["-m", "sdxlr.data", "--dest", DATASET_DIR]
if SOURCE == "test":
    args += ["--source", "test"]
elif SOURCE == "hf":
    args += ["--source", "hf", "--repo", HF_REPO, "--file", HF_FILE]
else:
    from google.colab import drive
    drive.mount("/content/drive")
    args += ["--source", "dir", "--path", DRIVE_PATH]
nb.run(args)
