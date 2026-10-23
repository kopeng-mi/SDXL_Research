# %% [md]
# ## Caption
# Writes one `.txt` per image. Captions describe clothing, pose, background and
# lighting, and skip face and body, so the LoRA learns those from the pixels.
# Check the first few with `PREVIEW = 3`, then set `PREVIEW = 0` for all.
# %%
#@title Caption (Qwen3-VL / JoyCaption)
MODEL = "qwen"  #@param ["qwen", "qwen-2b", "joy"]
TRIGGER = "ohwx_person"  #@param {type:"string"}
CLASS_WORD = "woman"  #@param {type:"string"}
PROMPT = ""  #@param {type:"string"}
#@markdown blank PROMPT = identity preset; `{cls}` in a custom prompt = CLASS_WORD
PREVIEW = 3  #@param {type:"integer"}
OVERWRITE = False  #@param {type:"boolean"}

args = ["-m", "sdxlr.caption", "--images", DATASET_DIR, "--model", MODEL,
        "--trigger", TRIGGER, "--class-word", CLASS_WORD, "--limit", PREVIEW]
if PROMPT: args += ["--prompt", PROMPT]
if OVERWRITE: args += ["--overwrite"]
nb.run(args)
