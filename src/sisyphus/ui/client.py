"""Thread-based SSE consumer and thin REST helpers used by the Streamlit app."""

from __future__ import annotations

import json
import threading
from collections import deque
from typing import Any

import httpx


class StreamClient:
    """Consumes ``/matches/{id}/stream`` in a daemon thread and keeps the latest state."""

    def __init__(self, base_url: str, match_id: str, params: dict[str, Any]) -> None:
        self.url = f"{base_url}/matches/{match_id}/stream"
        self.params = params
        self.lock = threading.Lock()
        self.hello: dict | None = None
        self.tick: dict | None = None
        self.positions: dict | None = None
        self.seek: dict | None = None
        self.insights: list[dict] = []
        self.triggers: list[dict] = []
        self.overlays: list[dict] = []
        self.history: deque[dict] = deque(maxlen=6000)
        self.errors: list[str] = []
        self.status = "connecting"
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True, name="sse-client")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    @property
    def running(self) -> bool:
        return self._thread.is_alive()

    def _run(self) -> None:
        try:
            with httpx.stream("GET", self.url, params=self.params, timeout=httpx.Timeout(10, read=None)) as r:
                if r.status_code != 200:
                    self.status = f"error {r.status_code}"
                    return
                self.status = "live"
                event = None
                for line in r.iter_lines():
                    if self._stop.is_set():
                        break
                    if line.startswith("event:"):
                        event = line[6:].strip()
                    elif line.startswith("data:") and event:
                        self._handle(event, json.loads(line[5:]))
        except Exception as exc:  # pragma: no cover - UI feedback only
            self.errors.append(str(exc))
            self.status = "disconnected"
        else:
            if self.status == "live":
                self.status = "finished"

    def _handle(self, event: str, data: dict) -> None:
        with self.lock:
            if event == "hello":
                self.hello = data
            elif event == "seek":
                self.seek = data
            elif event == "tick":
                self.tick = data
                self.seek = None
                t = data["teams"]
                self.history.append(
                    {
                        "minute": (data["period"] - 1) * 45 + data["t"] / 60,
                        "clock": data["clock"],
                        "home_energy": t[0]["energy"],
                        "away_energy": t[1]["energy"],
                        "momentum": data["momentum"],
                        "home_sprints": t[0]["sprints_5min"],
                        "away_sprints": t[1]["sprints_5min"],
                    }
                )
            elif event == "positions":
                self.positions = data
            elif event == "trigger":
                self.triggers.append(data)
            elif event == "insight":
                self.insights.append(data)
            elif event == "overlay":
                self.overlays.append(data)
            elif event == "error":
                self.errors.append(data.get("message", "unknown error"))
            elif event == "end":
                self.status = "finished"


def api_get(base_url: str, path: str, **params: Any) -> Any:
    r = httpx.get(f"{base_url}{path}", params=params or None, timeout=120)
    r.raise_for_status()
    return r.json()


def api_post(base_url: str, path: str, body: dict | None = None, **params: Any) -> Any:
    r = httpx.post(f"{base_url}{path}", json=body, params=params or None, timeout=120)
    r.raise_for_status()
    return r.json()
