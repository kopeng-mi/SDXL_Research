# %% [md]
# ## 4. Test
# A grid: rows = the last epochs (x prompts), columns = LoRA strengths, same seed everywhere.
# Time is about 20 s per image (SDXL, 20 steps) plus the model load.
# %%
#@title Test
EPOCHS_TO_TEST = "last:3"  #@param {type:"string"}
#@markdown last:K, all, or a list like 6,10,15
STRENGTHS = "0.6 0.8 1.0"  #@param {type:"string"}
PROMPT_1 = "photo of {t} {c}, portrait, natural light, sharp focus"  #@param {type:"string"}
PROMPT_2 = "photo of {t} {c} walking in a city street, full body"  #@param {type:"string"}
NEGATIVE = "blurry, lowres, deformed"  #@param {type:"string"}
STEPS = 20  #@param {type:"integer"}
SEED = 42  #@param {type:"integer"}
prompts = [x.replace("{t}", TRIGGER).replace("{c}", CLASS_WORD) for x in (PROMPT_1, PROMPT_2) if x.strip()]
run_dir = paths.WORK / "runs" / RUN
if ROOT and not any(run_dir.glob("*.safetensors")):  # a new VM: test what Drive has
    drive.pull(ROOT / "loras" / "sdxl" / RUN, run_dir)
args = ["-m", "sdxlr.compare", "--name", RUN, "--model", paths.SDXL_BASES.get(BASE_SRC, BASE_SRC),
        "--lora-dir", run_dir, "--epochs", EPOCHS_TO_TEST, "--strengths", *STRENGTHS.split(),
        "--prompts", *prompts, "--negative", NEGATIVE, "--steps", STEPS, "--seed", SEED]
if ROOT:
    args += ["--drive-out", ROOT / "compare" / RUN]
nb.run(args)
from IPython.display import Image as _Img, display
display(_Img(filename=str(paths.WORK / "compare" / RUN / "grid.png")))
