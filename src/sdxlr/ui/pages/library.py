"""Library: add models and LoRAs by link or upload; see, delete and sync what is on the VM."""

from __future__ import annotations

from pathlib import Path

from nicegui import ui

from ... import presets, resources, settings
from ...engine import FP16_VAE, Z_REPO
from ...fetch import LORAS, MODELS, ZIMAGE, add_file, fetch_loras, list_loras, list_models, list_zimage
from ..core import UPLOADS, eng, err, jm, size_text
from ..theme import JobLine, button, confirm, labeled, main_button, seg, title

KINDS = {"LoRA": "loras", "SDXL checkpoint": "model", "Z-Image model": "zimage"}


def page(S):
    with ui.row().classes("w-full gap-8 no-wrap items-start two-col"):
        with ui.column().classes("gap-4").style("width:460px; min-width:360px"):
            title("Library")
            with ui.column().classes("panel w-full gap-3"):
                ui.label("Add a model or LoRA").classes("t-head")
                with ui.column().classes("gap-0"):
                    ui.label("What is it").classes("lbl")
                    seg(list(KINDS), S, "l_kind")
                labeled("Links, one per line", ui.textarea(
                    placeholder="org/repo/file.safetensors\nhttps://huggingface.co/org/repo/blob/main/file.safetensors\n"
                                "https://civitai.com/api/download/models/12345").bind_value(S, "l_link").props("autogrow"),
                    "Hugging Face, Civitai, any https link, or a Drive path. Private files need HF_TOKEN or CIVITAI_TOKEN in Secrets")

                def dl():
                    links = [s.strip() for s in S["l_link"].replace("\n", ",").split(",") if s.strip()]
                    if not links:
                        ui.notify("Paste a link first", type="warning")
                        return
                    line.track(jm.submit("download", f"Download {', '.join(Path(x).name for x in links)[:60]}",
                                         ["-m", "sdxlr.fetch", KINDS[S["l_kind"]], *links], gpu=False))
                    S["l_link"] = ""

                main_button("Download", dl)
                line = JobLine(("download",))

                async def on_up(e):
                    d = UPLOADS / "library"
                    d.mkdir(parents=True, exist_ok=True)
                    p = d / Path(e.file.name).name
                    await e.file.save(p)
                    k = KINDS[S["l_kind"]]
                    try:
                        if k == "loras":
                            names = [x.name for x in fetch_loras(str(p))] if p.suffix == ".zip" else [add_file(str(p), LORAS).name]
                        else:
                            names = [add_file(str(p), MODELS if k == "model" else ZIMAGE).name]
                        ui.notify(f"Added {', '.join(names)}", type="positive")
                    except Exception as ex:
                        ui.notify(err(ex), type="negative")
                    draw()

                ui.upload(label="Or drop files here: .safetensors, .gguf, or a zip of LoRAs", multiple=True,
                          auto_upload=True, on_upload=on_up) \
                    .props("flat bordered accept=.safetensors,.gguf,.ckpt,.zip hide-upload-btn").classes("drop")

            with ui.expansion("Sync presets and LoRAs with a private HF repo").classes("w-full panel"):
                st = settings.get()
                sync = labeled("Sync repo", ui.input(value=st["sync_repo"], placeholder="yourname/sdxlr-sync"))

                def do_sync(direction):
                    repo = (sync.value or "").strip()
                    if not repo:
                        ui.notify("Type the sync repo first", type="warning")
                        return
                    settings.set(sync_repo=repo)
                    try:
                        ui.notify(presets.push(repo) if direction == "push" else presets.pull(repo), type="positive")
                        draw()
                    except Exception as e:
                        ui.notify(err(e), type="negative")

                with ui.row().classes("gap-2"):
                    button("Upload to the repo", lambda: do_sync("push"), icon="upload")
                    button("Download from the repo", lambda: do_sync("pull"), icon="download")

        with ui.column().classes("gap-2").style("flex:1 1 0; min-width:0; padding-top:44px"):
            with ui.row().classes("w-full items-center"):
                ui.label("On this VM").classes("t-head")
                ui.space()
                button("Free disk space", lambda: free(), icon="cleaning_services")
            ui.label("Free disk space deletes cached downloads that are not in use; they download again when needed.") \
                .classes("hint")
            table = ui.column().classes("w-full gap-0")

    def free():
        gb = resources.clear_download_cache([eng.model_path or "", Z_REPO, FP16_VAE])
        ui.notify(f"Freed {gb:.1f} GB" if gb >= 0.05 else "Nothing to free: the cache only holds files in use",
                  type="positive")

    def files():
        return ([("SDXL checkpoint", p) for p in list_models()] + [("Z-Image model", p) for p in list_zimage()]
                + [("LoRA", p) for p in list_loras()])

    def delete(p: Path):
        p.unlink(missing_ok=True)
        ui.notify(f"Deleted {p.name}")
        draw()

    seen = {"sig": None}

    def draw():
        fs = files()
        seen["sig"] = [str(p) for _, p in fs]
        table.clear()
        with table:
            if not fs:
                ui.label("Nothing here yet. Paste a link or drop a file on the left.").classes("muted")
            for kind, p in fs:
                with ui.row().classes("w-full items-center no-wrap gap-3 py-2").style("border-bottom:1px solid var(--line)"):
                    ui.label(p.name).classes("grow").style("overflow-wrap:anywhere")
                    ui.label(kind).classes("t-small muted").style("width:120px")
                    ui.label(size_text(p.stat().st_size)).classes("t-small muted tnum text-right").style("width:80px")
                    ui.button(icon="delete_outline", on_click=lambda p=p: confirm(
                        f"Delete {p.name} from this VM?", "Delete", lambda: delete(p))) \
                        .props("flat dense round size=sm").tooltip("Delete")

    def tick():
        if seen["sig"] != [str(p) for _, p in files()]:
            draw()

    draw()
    ui.timer(3.0, tick)
