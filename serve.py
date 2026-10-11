"""Run the interface and the REST API together in one container.

They are deliberately not separate containers. Both load the same read-only model set, measured at
2.35 GB resident, so splitting them would double the memory for no isolation benefit. The API runs
in a background thread of this process and the Streamlit server in a child process, because Streamlit
insists on owning its own process and does not embed.

Ports are read from the environment so a host that assigns them can do so: PORT for the interface,
which is the convention most platforms use, and API_PORT for the API.

If either dies the other is stopped, so an orchestrator sees a failed container and restarts it,
rather than a half-working one that answers health checks on the surviving half.
"""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
# Hugging Face Spaces exposes exactly one port and defaults it to 7860; other hosts set PORT. The
# default here is 7860 so a Space works with no configuration, and PORT overrides it everywhere else.
UI_PORT = os.environ.get("PORT", "7860")
API_PORT = os.environ.get("API_PORT", "8000")
# A Space publishes only the one port, so the API is reachable inside the container but not from the
# internet. Serving both publicly needs either a reverse proxy that can carry Streamlit's websocket,
# which is not worth writing, or a second Space running api.py. DEPLOY.md sets that out.
# Accepts the former BRAINSAFE_API_ONLY too; see _flag in model_fetch.py for why.
from model_fetch import _flag  # noqa: E402
API_ONLY = _flag("API_ONLY")


def run_api():
    sys.argv = ["api.py", API_PORT]
    sys.path.insert(0, str(ROOT))
    import api
    api.main()


def main():
    if API_ONLY:
        # a Space dedicated to the API: it must own the single published port
        print(f"API-only mode, serving on {UI_PORT}", flush=True)
        sys.argv = ["api.py", UI_PORT]
        sys.path.insert(0, str(ROOT))
        import api
        return api.main()

    print(f"starting API on {API_PORT} and interface on {UI_PORT}", flush=True)
    t = threading.Thread(target=run_api, daemon=True, name="sai_cns-api")
    t.start()

    ui = subprocess.Popen([sys.executable, "-m", "streamlit", "run", str(ROOT / "app.py"),
                           f"--server.port={UI_PORT}", "--server.address=0.0.0.0",
                           "--server.headless=true", "--browser.gatherUsageStats=false"])

    def stop(*_):
        ui.terminate()
        sys.exit(0)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)

    while True:
        code = ui.poll()
        if code is not None:
            print(f"interface exited with {code}; stopping container", flush=True)
            sys.exit(code or 1)
        if not t.is_alive():
            print("API thread died; stopping container", flush=True)
            ui.terminate()
            sys.exit(1)
        time.sleep(5)


if __name__ == "__main__":
    main()
