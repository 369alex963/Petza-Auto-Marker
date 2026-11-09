"""python -m petza  ->  start the app on 127.0.0.1 and open the browser."""
from __future__ import annotations

import os
import socket
import threading
import webbrowser


def free_port(preferred: int = 8765) -> int:
    for port in [preferred, *range(preferred + 1, preferred + 50)]:
        with socket.socket() as s:
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    return 0


def main() -> None:
    import uvicorn

    port = free_port(int(os.environ.get("PETZA_PORT", 8765)))
    url = f"http://127.0.0.1:{port}"
    print(f"\n  Wound-Recognizer is running at {url}\n  Keep this window open. Close it to stop the app.\n")
    if not os.environ.get("PETZA_NO_BROWSER"):
        threading.Timer(1.5, lambda: webbrowser.open(url)).start()
    uvicorn.run("petza.api:app", host="127.0.0.1", port=port, log_level="warning")


if __name__ == "__main__":
    main()
