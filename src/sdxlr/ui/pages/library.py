"""Library: add models and LoRAs by link or upload; see, delete and sync what is on the VM."""

from __future__ import annotations

import threading
from pathlib import Path

from nicegui import run, ui

from ... import presets, resources, settings
from ...activity import log
from ...engine import FP16_VAE, Z_REPO
from ...fetch import LORAS, MODELS, ZIMAGE, add_file, fetch_loras, list_loras
from ...service.files import files_on_vm, poke
from ..core import SHRED, UPLOADS, engc, err, jm, shred, shred_sizes, size_text
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
                    poke()
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
                        poke()
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
            ui.label("This VM is temporary: when the Colab session ends, everything on it is gone. Trained LoRAs and "
                     "datasets survive only in your HF sync repo (set in the notebook) or as downloads.") \
                .classes("hint").style("color:#E8A33D")
            with ui.row().classes("w-full items-center gap-2"):
                ui.space()
                button("Download all LoRAs as a zip", lambda: download_loras(), icon="folder_zip")
                button("Delete everything…", lambda: shred_dialog(), icon="delete_forever", kind="bad")
            table = ui.column().classes("w-full gap-0")

    def download_loras():
        import zipfile
        ls = list_loras()
        if not ls:
            ui.notify("There are no LoRAs yet", type="warning")
            return
        out = UPLOADS / "loras.zip"
        out.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(out, "w", zipfile.ZIP_STORED) as z:  # safetensors do not compress
            for p in ls:
                z.write(p, p.name)
        ui.download.file(out)

    def shred_dialog():
        sizes = shred_sizes()
        picks = {}
        with ui.dialog() as d, ui.card().classes("panel").style("width:460px"):
            ui.label("Delete from this VM").classes("t-head")
            ui.label("Pick what to delete. This cannot be undone; files in your HF repos are not touched.") \
                .classes("t-small")
            for k, (label, _) in SHRED.items():
                picks[k] = ui.checkbox(f"{label}, {sizes[k]:.1f} GB")
            ui.label("Deleting unloads the image model first.").classes("hint")

            def go():
                ks = [k for k, c in picks.items() if c.value]
                if not ks:
                    ui.notify("Tick at least one", type="warning")
                    return
                try:
                    gb = shred(ks)
                    d.close()
                    ui.notify(f"Deleted, {gb:.1f} GB freed", type="positive")
                    draw()
                except ValueError as e:
                    ui.notify(str(e), type="warning")

            with ui.row().classes("w-full justify-end gap-2"):
                button("Keep everything", d.close)
                button("Delete the ticked items", go, kind="bad")
        d.open()

    async def free():
        gb = await run.io_bound(resources.clear_download_cache, [engc.model_path or "", Z_REPO, FP16_VAE])
        poke()
        ui.notify(f"Freed {gb:.1f} GB" if gb >= 0.05 else "Nothing to free: the cache only holds files in use",
                  type="positive")

    def files():
        return files_on_vm()

    def busy() -> str | None:
        if jm.gpu_busy() or engc.busy:
            return "Wait until nothing is running: a running job or image may be using it"
        return None

    def delete(e: dict):
        if e["kind"] != "LoRA" and (why := busy()):
            ui.notify(why, type="warning")
            return

        def work():  # a few GB: off the event loop; the list redraws when the file index changes
            try:
                in_use = e["cache_repo"] or str(e["path"])
                if engc.loaded and in_use in (engc.model_path or ""):
                    engc.kill()  # the loaded model reads from it; the next Generate loads again
                if e["cache_repo"]:
                    resources.delete_cache_repos([e["cache_repo"]])
                else:
                    e["path"].unlink(missing_ok=True)
            except Exception as ex:
                log.info(f"library delete of {e['name']} failed: {ex}")
            poke()

        threading.Thread(target=work, daemon=True).start()
        ui.notify(f"Deleting {e['name']}" + ("; it downloads again when needed" if e["cache_repo"] else ""))

    seen = {"sig": None}

    def draw():
        fs = files()
        seen["sig"] = [(e["name"], e["bytes"]) for e in fs]
        table.clear()
        with table:
            if not fs:
                ui.label("Nothing here yet. Paste a link or drop a file on the left.").classes("muted")
            else:
                total = sum(e["bytes"] for e in fs)
                ui.label(f"{len(fs)} items, {size_text(total)}. Models the app fetched by itself download again when "
                         "needed if you delete them.").classes("hint pb-1")
            for e in fs:
                with ui.row().classes("w-full items-center no-wrap gap-3 py-2").style("border-bottom:1px solid var(--line)"):
                    ui.label(e["name"]).classes("grow").style("overflow-wrap:anywhere")
                    ui.label(e["kind"]).classes("t-small muted").style("width:190px")
                    ui.label(size_text(e["bytes"])).classes("t-small muted tnum text-right").style("width:80px")
                    if e["cache_repo"]:
                        ui.element("div").style("width:32px")  # a cache folder, not one file to download
                    else:
                        ui.button(icon="download", on_click=lambda e=e: ui.download.file(e["path"]))                             .props("flat dense round size=sm").tooltip("Download to this computer")
                    ui.button(icon="delete_outline", on_click=lambda e=e: confirm(
                        f"Delete {e['name']} from this VM?", "Delete", lambda: delete(e)))                         .props("flat dense round size=sm").tooltip("Delete")

    def tick():
        if seen["sig"] != [(e["name"], e["bytes"]) for e in files()]:
            draw()

    draw()
    ui.timer(3.0, tick)
