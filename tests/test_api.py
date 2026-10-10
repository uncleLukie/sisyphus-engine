from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from sisyphus.service import app as app_module


@pytest.fixture(scope="module")
def client(short_match_dir):
    app_module.DATA_DIR = short_match_dir.parent
    with TestClient(app_module.app) as c:
        yield c


def test_health_and_listing(client, short_match):
    h = client.get("/health").json()
    assert h["status"] == "ok" and h["agent_mode"] == "template"
    ms = client.get("/matches").json()
    assert [m["match_id"] for m in ms] == [short_match.meta.match_id]
    assert client.get("/matches/nope").status_code == 404
    assert client.get("/scenarios").json()[0]["scenario_id"] == "press_collapse"


def test_stream_delivers_ticks_triggers_and_insights(client, short_match):
    mid = short_match.meta.match_id
    counts: dict[str, int] = {}
    first_insight = None
    with client.stream(
        "GET",
        f"/matches/{mid}/stream",
        params={"speed": 120, "from_minute": 2, "personas": "casual:es,analyst:en", "live_agents": "true"},
    ) as r:
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/event-stream")
        event = None
        for line in r.iter_lines():
            if line.startswith("event:"):
                event = line[6:].strip()
            elif line.startswith("data:"):
                counts[event] = counts.get(event, 0) + 1
                if event == "insight" and first_insight is None:
                    first_insight = json.loads(line[5:])
    assert counts.get("hello") == 1 and counts.get("end") == 1
    assert counts.get("tick", 0) > 60
    assert counts.get("positions", 0) > 100
    assert counts.get("trigger", 0) >= 1
    assert counts.get("insight") == counts.get("trigger")
    assert counts.get("overlay") == 2 * counts.get("trigger")
    assert {p["persona_id"] for p in first_insight["personalised"]} == {"casual-es", "analyst-en"}


def test_on_demand_explain_and_eval(client, short_match):
    mid = short_match.meta.match_id
    trig = client.get(f"/matches/{mid}/triggers").json()  # analysed lazily on first request
    assert trig
    body = {"personas": [{"persona_id": "casual-fr", "kind": "casual", "language": "fr"}]}
    res = client.post(f"/matches/{mid}/explain/{trig[-1]['trigger_id']}", json=body).json()
    assert res["insight"]["personalised"][0]["persona_id"] == "casual-fr"
    assert res["overlays"][0]["persona_id"] == "casual-fr"
    ev = client.get(f"/matches/{mid}/eval").json()
    assert set(ev["teams"]) == {"HAR", "KMR"}
