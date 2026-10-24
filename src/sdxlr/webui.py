"""SDXL_Research UI (NiceGUI). Runs inside the main venv.

  python -m sdxlr.webui [--port 7860] [--access tailscale|none] [--sync]

Open http://sdxlr:7860 on a device signed in to your Tailscale (needs
TS_AUTHKEY). The code lives in sdxlr/ui: core (state and work), theme (look and
shared widgets), pages/ (one file per page) and frame (sidebar and startup).
"""

from .ui.frame import main

if __name__ == "__main__":
    main()
