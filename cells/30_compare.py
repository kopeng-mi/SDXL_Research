# %% [md]
# ## Compare
# Rows = LoRAs (or LoRA mixes), columns = strengths, same seed everywhere.
# Re-running continues where it stopped; change any setting to start a new grid.
# %%
#@title LoRA comparison grid
GRID_NAME = "grid1"  #@param {type:"string"}
MODEL_TYPE = "SDXL"  #@param ["SDXL", "Z-Image"]
Z_MODEL = "turbo Q8"  #@param ["turbo Q8", "turbo Q6", "turbo Q5", "turbo Q4"] {allow-input: true}
Z_TEXT_ENCODER = "official"  #@param ["official", "abliterated"] {allow-input: true}
#@markdown Z-Image: use STEPS 9, CFG 0; prompt weights like (word:1.2) are SDXL-only
LORA_GLOB = "*.safetensors"  #@param {type:"string"}
COMBOS = ""  #@param {type:"string"}
#@markdown COMBOS (optional) replaces LORA_GLOB: `face_a.safetensors:0.44+face_b.safetensors:0.44, other.safetensors:0.8`. Strengths then multiply the mix.
STRENGTHS = "0.78 0.82 0.88"  #@param {type:"string"}
PROMPT = "(ohwx_person:1.2), woman, highly detailed face, from thigh up, white background"  #@param {type:"string"}
NEGATIVE = "(worst quality:1.3), (low quality:1.2), lowres, bad anatomy, bad hands, watermark, blur, cartoon, anime, drawing"  #@param {type:"string"}
STEPS = 25  #@param {type:"integer"}
CFG = 5  #@param {type:"number"}
SAMPLER = "dpmpp_2m_karras"  #@param ["dpmpp_2m_karras", "dpmpp_2m_sde_karras", "dpmpp_sde_karras", "euler_a", "euler"]
CLIP_SKIP = 2  #@param {type:"integer"}
WIDTH = 896  #@param {type:"integer"}
HEIGHT = 1152  #@param {type:"integer"}
SEED = 42  #@param {type:"integer"}

model_args = (["--model", Z_MODEL, "--zimage", "--te", Z_TEXT_ENCODER] if MODEL_TYPE == "Z-Image"
              else ["--model", BASE_MODEL])
args = ["-m", "sdxlr.compare", "--name", GRID_NAME, *model_args, "--strengths", *STRENGTHS.split(),
        "--prompt", PROMPT, "--negative", NEGATIVE, "--steps", STEPS, "--cfg", CFG, "--sampler", SAMPLER,
        "--clip-skip", CLIP_SKIP, "--width", WIDTH, "--height", HEIGHT, "--seed", SEED]
combos = [c.strip() for c in COMBOS.split(",") if c.strip()]
args += ["--combos", *combos] if combos else ["--loras", LORA_GLOB]
nb.run(args)
from IPython.display import Image as _Img, display
display(_Img(filename=f"/content/work/compare/{GRID_NAME}/grid.png"))
