"""FastAPI application exposing the pipeline over REST and Server-Sent Events."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import BackgroundTasks, FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from sisyphus import __version__
from sisyphus.agents import InsightBundle, InsightPipeline, Persona
from sisyphus.agents.clients import describe_client
from sisyphus.analytics import Trigger
from sisyphus.io import generate_match, list_matches
from sisyphus.pipeline import (
    analyze_match,
    evaluate_match,
    narrate_match,
    read_insights,
    read_timeline,
    read_triggers,
    save_analysis,
)
from sisyphus.render import OverlayPayload, render_overlays
from sisyphus.service.stream import MatchStream
from sisyphus.sim.scenario import SCENARIOS

log = logging.getLogger("sisyphus.service")

DATA_DIR = Path(os.environ.get("SISYPHUS_DATA_DIR", "data/generated"))
JOBS: dict[str, dict[str, Any]] = {}


class State:
    pipeline: InsightPipeline


state = State()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    state.pipeline = InsightPipeline(personas=Persona.defaults())
    log.info("insight pipeline mode=%s data=%s", state.pipeline.mode, DATA_DIR.resolve())
    yield


app = FastAPI(
    title="Inside the Game - Sisyphus Engine",
    version=__version__,
    description="Synthetic football tracking -> interpretable analytics -> multi-agent narrative.",
    lifespan=lifespan,
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# ---- helpers ----------------------------------------------------------------
def _match_dir(match_id: str) -> Path:
    if "/" in match_id or "\\" in match_id or ".." in match_id:
        raise HTTPException(400, "bad match id")
    d = DATA_DIR / match_id
    if not (d / "meta.json").exists():
        raise HTTPException(404, f"match {match_id!r} not found")
    return d


def _parse_personas(spec: str | None, favourite_team: str | None = None) -> list[Persona]:
    """``casual:en,analyst:en,casual:es`` -> personas."""
    if not spec:
        personas = Persona.defaults()
    else:
        personas = []
        for part in spec.split(","):
            kind, _, lang = part.strip().partition(":")
            if kind not in ("casual", "analyst"):
                raise HTTPException(400, f"unknown persona kind {kind!r}")
            lang = lang or "en"
            personas.append(Persona(persona_id=f"{kind}-{lang}", kind=kind, language=lang))  # type: ignore[arg-type]
    if favourite_team:
        personas = [
            p.model_copy(update={"favourite_team": favourite_team}) if p.kind == "casual" else p for p in personas
        ]
    return personas


def _sse(event: str, data: Any) -> str:
    payload = data.model_dump_json() if isinstance(data, BaseModel) else json.dumps(data)
    return f"event: {event}\ndata: {payload}\n\n"


# ---- meta -------------------------------------------------------------------
@app.get("/health")
def health() -> dict:
    info = describe_client()
    return {
        "status": "ok",
        "version": __version__,
        "agent_mode": state.pipeline.mode,
        "provider": info.provider if info else "template",
        "model": info.model if info else None,
        "data_dir": str(DATA_DIR),
    }


@app.get("/scenarios")
def scenarios() -> list[dict]:
    return [
        {"scenario_id": s.scenario_id, "title": s.title, "description": s.description, "default_seed": s.default_seed}
        for s in SCENARIOS.values()
    ]


@app.get("/personas")
def personas() -> list[Persona]:
    return Persona.defaults()


# ---- matches ----------------------------------------------------------------
@app.get("/matches")
def matches() -> list[dict]:
    out = []
    for m in list_matches(DATA_DIR):
        d = DATA_DIR / m.match_id
        out.append(
            {
                "match_id": m.match_id,
                "scenario_id": m.scenario_id,
                "scenario_title": m.scenario_title,
                "seed": m.seed,
                "home": m.home.short_name,
                "away": m.away.short_name,
                "final_score": m.final_score,
                "analyzed": (d / "triggers.jsonl").exists(),
                "narrated": (d / "insights.jsonl").exists(),
            }
        )
    return out


@app.get("/matches/{match_id}")
def match_meta(match_id: str) -> dict:
    return json.loads((_match_dir(match_id) / "meta.json").read_text(encoding="utf-8"))


def _ensure_analyzed(d: Path) -> list[Trigger]:
    trig = read_triggers(d)
    if not trig and not (d / "triggers.jsonl").exists():
        result = analyze_match(d)
        save_analysis(d, result)
        trig = result.triggers
    return trig


@app.get("/matches/{match_id}/triggers")
def match_triggers(match_id: str, kind: str | None = None) -> list[Trigger]:
    trig = _ensure_analyzed(_match_dir(match_id))
    return [t for t in trig if kind is None or t.kind == kind]


@app.get("/matches/{match_id}/insights")
def match_insights(match_id: str) -> list[InsightBundle]:
    return read_insights(_match_dir(match_id))


@app.get("/matches/{match_id}/overlays")
def match_overlays(match_id: str, persona: str | None = None) -> list[OverlayPayload]:
    d = _match_dir(match_id)
    p = d / "overlays.jsonl"
    if not p.exists():
        return []
    with p.open(encoding="utf-8") as fh:
        rows = [OverlayPayload.model_validate_json(line) for line in fh if line.strip()]
    return [r for r in rows if persona is None or r.persona_id == persona]


@app.get("/matches/{match_id}/timeline")
def match_timeline(match_id: str, every: int = Query(1, ge=1)) -> list[dict]:
    d = _match_dir(match_id)
    _ensure_analyzed(d)
    return read_timeline(d)[::every]


@app.get("/matches/{match_id}/eval")
def match_eval(match_id: str) -> dict:
    return evaluate_match(_match_dir(match_id))


# ---- on-demand explain (personalisation per viewer) --------------------------
class ExplainRequest(BaseModel):
    personas: list[Persona] = Field(default_factory=Persona.defaults)


@app.post("/matches/{match_id}/explain/{trigger_id}")
async def explain_trigger(match_id: str, trigger_id: str, req: ExplainRequest) -> dict:
    d = _match_dir(match_id)
    trig = next((t for t in _ensure_analyzed(d) if t.trigger_id == trigger_id), None)
    if trig is None:
        raise HTTPException(404, "trigger not found")
    pipe = InsightPipeline(personas=req.personas, client=state.pipeline.client, auto_client=False)
    bundle = await pipe.explain(trig)
    return {"insight": bundle, "overlays": render_overlays(bundle, trig.snapshot)}


# ---- jobs: simulate / analyze / narrate --------------------------------------
class SimulateRequest(BaseModel):
    scenario_id: str = "press_collapse"
    seed: int | None = None
    narrate: bool = False
    languages: list[str] = Field(default_factory=lambda: ["en"])


def _run_simulate(job_id: str, req: SimulateRequest) -> None:
    job = JOBS[job_id]
    try:
        job["status"] = "simulating"
        meta, path = generate_match(req.scenario_id, DATA_DIR, seed=req.seed, progress=lambda p: job.update(progress=p))
        job.update(match_id=meta.match_id, status="analyzing", progress=0.0)
        save_analysis(path, analyze_match(path, progress=lambda p: job.update(progress=p)))
        if req.narrate:
            job.update(status="narrating")
            personas = Persona.defaults() + [
                Persona(persona_id=f"casual-{lang}", kind="casual", language=lang)
                for lang in req.languages
                if lang != "en"
            ]
            asyncio.run(narrate_match(path, personas=personas, pipeline=InsightPipeline(personas=personas)))
        job.update(status="done", progress=1.0)
    except Exception as exc:  # pragma: no cover
        log.exception("job %s failed", job_id)
        job.update(status="failed", error=str(exc))


@app.post("/simulate", status_code=202)
def simulate(req: SimulateRequest, background: BackgroundTasks) -> dict:
    if req.scenario_id not in SCENARIOS:
        raise HTTPException(400, f"unknown scenario; choose from {sorted(SCENARIOS)}")
    job_id = uuid.uuid4().hex[:8]
    JOBS[job_id] = {"job_id": job_id, "status": "queued", "progress": 0.0, "request": req.model_dump()}
    background.add_task(_run_simulate, job_id, req)
    return JOBS[job_id]


@app.post("/matches/{match_id}/narrate", status_code=202)
def narrate(
    match_id: str,
    background: BackgroundTasks,
    personas: str | None = None,
    kinds: str | None = None,
    limit: int | None = None,
) -> dict:
    d = _match_dir(match_id)
    job_id = uuid.uuid4().hex[:8]
    JOBS[job_id] = {"job_id": job_id, "status": "queued", "progress": 0.0, "match_id": match_id}
    plist = _parse_personas(personas)

    def _run() -> None:
        try:
            JOBS[job_id]["status"] = "narrating"
            asyncio.run(
                narrate_match(
                    d,
                    personas=plist,
                    kinds=set(kinds.split(",")) if kinds else None,
                    max_triggers=limit,
                    pipeline=InsightPipeline(personas=plist),
                )
            )
            JOBS[job_id].update(status="done", progress=1.0)
        except Exception as exc:  # pragma: no cover
            log.exception("narrate job failed")
            JOBS[job_id].update(status="failed", error=str(exc))

    background.add_task(_run)
    return JOBS[job_id]


@app.get("/jobs/{job_id}")
def job(job_id: str) -> dict:
    if job_id not in JOBS:
        raise HTTPException(404, "job not found")
    return JOBS[job_id]


# ---- live stream --------------------------------------------------------------
@app.get("/matches/{match_id}/stream")
async def stream(
    request: Request,
    match_id: str,
    speed: float = Query(10.0, ge=0.1, le=120.0, description="Replay speed multiplier"),
    from_minute: float = Query(0.0, ge=0.0, le=95.0),
    personas: str | None = Query(None, description="e.g. casual:en,analyst:en,casual:es"),
    favourite_team: str | None = None,
    live_agents: bool = Query(False, description="Call the agent workflow live instead of replaying stored insights"),
    kinds: str | None = Query(None, description="comma-separated trigger kinds to narrate"),
) -> StreamingResponse:
    d = _match_dir(match_id)
    plist = _parse_personas(personas, favourite_team)
    pipeline = InsightPipeline(personas=plist, client=state.pipeline.client, auto_client=False)
    ms = MatchStream(
        d,
        pipeline,
        plist,
        speed=speed,
        from_minute=from_minute,
        live_agents=live_agents,
        kinds=set(kinds.split(",")) if kinds else None,
    )

    async def gen() -> AsyncIterator[str]:
        async for msg in ms.run():
            if await request.is_disconnected():
                ms.stop.set()
                break
            yield _sse(msg.event, msg.data)

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
