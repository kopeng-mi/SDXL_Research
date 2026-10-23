# %% [md]
# ## Caption
# Writes one `.txt` per image. Captions describe clothing, pose, background and
# lighting, and skip face and body, so the LoRA learns those from the pixels.
# Run with **Preview 3** first, check the captions, then run the full set.
# %%
#@title Caption a dataset (Qwen3-VL / JoyCaption)
import subprocess
IMAGES_DIR = "/content/dataset"  #@param {type:"string"}
CAPTIONS_DIR = ""  #@param {type:"string"}
MODEL = "qwen"  #@param ["qwen", "qwen-2b", "joy"]
TRIGGER = "ohwx_person"  #@param {type:"string"}
PROMPT = ""  #@param {type:"string"}
PREVIEW = 3  #@param {type:"integer"}
OVERWRITE = False  #@param {type:"boolean"}

cmd = ["/content/venv/bin/python", "-m", "sdxlr.caption", "--images", IMAGES_DIR,
       "--model", MODEL, "--trigger", TRIGGER, "--limit", str(PREVIEW)]
if CAPTIONS_DIR: cmd += ["--out", CAPTIONS_DIR]
if PROMPT: cmd += ["--prompt", PROMPT]
if OVERWRITE: cmd += ["--overwrite"]
subprocess.run(cmd, cwd="/content/SDXL_Research", check=True)
