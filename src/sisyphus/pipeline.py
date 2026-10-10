"""Offline orchestration of the five stages for one recorded match.

Ingest (replay) -> Interpret (AnalyticsEngine) -> Explain/Personalize (InsightPipeline)
-> Render (overlays). Outputs are written next to the match data so the API and
UI can serve them without recomputation, and so the whole chain is reproducible.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from sisyphus.agents import InsightBundle, InsightPipeline, Persona
from sisyphus.analytics import AnalyticsEngine, Trigger
from sisyphus.io import load_match, replay
from sisyphus.render import render_overlays, render_tick_overlay

TRIGGERS_FILE = "triggers.jsonl"
TIMELINE_FILE = "timeline.jsonl"
INSIGHTS_FILE = "insights.jsonl"
OVERLAYS_FILE = "overlays.jsonl"


@dataclass
class AnalysisResult:
    triggers: list[Trigger]
    timeline: list[dict] = field(default_factory=list)


def analyze_match(match_dir: Path, progress: Callable[[float], None] | None = None) -> AnalysisResult:
    """Replay a match through the analytics engine; collect triggers and a per-second timeline."""
    match_dir = Path(match_dir)
    data = load_match(match_dir, with_truth=False)
    engine = AnalyticsEngine(data.meta)
    total = max(1, len(data.frames))
    triggers: list[Trigger] = []
    timeline: list[dict] = []
    last_key: tuple[int, int] | None = None
    for i, tick in enumerate(replay(match_dir)):
        triggers.extend(engine.ingest(tick))
        key = (tick.frame.period, int(tick.frame.t))
        if key != last_key:
            timeline.append(render_tick_overlay(engine.snapshot()))
            last_key = key
        if progress and i % 1000 == 0:
            progress(i / total)
    return AnalysisResult(triggers, timeline)


def write_jsonl(path: Path, rows) -> None:
    with Path(path).open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(r.model_dump_json() if hasattr(r, "model_dump_json") else json.dumps(r))
            fh.write("\n")


def read_triggers(match_dir: Path) -> list[Trigger]:
    p = Path(match_dir) / TRIGGERS_FILE
    if not p.exists():
        return []
    with p.open(encoding="utf-8") as fh:
        return [Trigger.model_validate_json(line) for line in fh if line.strip()]


def read_insights(match_dir: Path) -> list[InsightBundle]:
    p = Path(match_dir) / INSIGHTS_FILE
    if not p.exists():
        return []
    with p.open(encoding="utf-8") as fh:
        return [InsightBundle.model_validate_json(line) for line in fh if line.strip()]


def read_timeline(match_dir: Path) -> list[dict]:
    p = Path(match_dir) / TIMELINE_FILE
    if not p.exists():
        return []
    with p.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def save_analysis(match_dir: Path, result: AnalysisResult) -> None:
    write_jsonl(Path(match_dir) / TRIGGERS_FILE, result.triggers)
    write_jsonl(Path(match_dir) / TIMELINE_FILE, result.timeline)


async def narrate_match(
    match_dir: Path,
    personas: list[Persona] | None = None,
    kinds: set[str] | None = None,
    max_triggers: int | None = None,
    concurrency: int = 3,
    pipeline: InsightPipeline | None = None,
) -> list[InsightBundle]:
    """Run the agent workflow over stored triggers and persist insights + overlays."""
    match_dir = Path(match_dir)
    triggers = read_triggers(match_dir)
    if not triggers:
        result = analyze_match(match_dir)
        save_analysis(match_dir, result)
        triggers = result.triggers
    if kinds:
        triggers = [t for t in triggers if t.kind in kinds]
    if max_triggers:
        triggers = triggers[:max_triggers]
    pipe = pipeline or InsightPipeline(personas=personas)
    bundles = await pipe.explain_many(triggers, concurrency=concurrency)
    by_id = {t.trigger_id: t for t in triggers}
    overlays = [o for b in bundles for o in render_overlays(b, by_id[b.trigger_id].snapshot)]
    write_jsonl(match_dir / INSIGHTS_FILE, bundles)
    write_jsonl(match_dir / OVERLAYS_FILE, overlays)
    return bundles


def narrate_match_sync(match_dir: Path, **kwargs) -> list[InsightBundle]:
    return asyncio.run(narrate_match(match_dir, **kwargs))


def evaluate_match(match_dir: Path) -> dict:
    """Score the observable analytics against the simulator's hidden truth.

    The analytics never see stamina; this checks that what they infer from
    positions alone tracks what the simulator actually did to the players.
    """
    match_dir = Path(match_dir)
    data = load_match(match_dir, with_truth=True)
    timeline = read_timeline(match_dir)
    if not timeline:
        result = analyze_match(match_dir)
        save_analysis(match_dir, result)
        timeline = result.timeline
    triggers = read_triggers(match_dir)
    out: dict = {"match_id": data.meta.match_id, "teams": {}}
    if data.truth is None:
        out["note"] = "truth.parquet missing; cannot evaluate"
        return out

    tr = data.truth.merge(data.tracking[["frame", "player_id", "team_id"]], on=["frame", "player_id"])
    fr = data.frames[["frame", "period", "t"]]
    tr = tr.merge(fr, on="frame")
    tr["sec"] = tr["t"].astype(int)
    truth_sec = tr.groupby(["team_id", "period", "sec"])["e_long"].mean().reset_index()

    rows = [
        {"period": row["period"], "sec": int(row["t"]), "team_id": team["team_id"], "energy": team["energy"]}
        for row in timeline
        for team in row["teams"]
        if team.get("baseline_ready", True)
    ]
    if not rows:
        out["note"] = "baseline never completed (match too short); no energy evaluation"
        for team_id in (data.meta.home.team_id, data.meta.away.team_id):
            out["teams"][team_id] = {"energy_vs_truth_corr": None}
        out["trigger_counts"] = {}
        return out
    tl = pd.DataFrame(rows)
    merged = tl.merge(truth_sec, on=["team_id", "period", "sec"])
    for team_id, g in merged.groupby("team_id"):
        if len(g) < 30 or g["energy"].std() < 1e-6:
            corr = float("nan")
        else:
            corr = float(np.corrcoef(g["energy"], g["e_long"])[0, 1])
        fatigue = [t for t in triggers if t.kind == "fatigue_breakpoint" and t.team_id == team_id]
        truth_team = truth_sec[truth_sec["team_id"] == team_id]
        first_below = truth_team[truth_team["e_long"] < 0.55]
        truth_minute = (
            float((first_below.iloc[0]["period"] - 1) * 45 + first_below.iloc[0]["sec"] / 60)
            if len(first_below)
            else None
        )
        out["teams"][team_id] = {
            "energy_vs_truth_corr": round(corr, 3) if corr == corr else None,
            "final_energy_index": round(float(g["energy"].iloc[-1]), 1),
            "final_true_e_long": round(float(g["e_long"].iloc[-1]), 3),
            "first_fatigue_breakpoint_minute": round(fatigue[0].minute, 1) if fatigue else None,
            "truth_e_long_below_0.55_minute": round(truth_minute, 1) if truth_minute else None,
            "fatigue_triggers": len(fatigue),
        }
    kinds: dict[str, int] = {}
    for t in triggers:
        kinds[t.kind] = kinds.get(t.kind, 0) + 1
    out["trigger_counts"] = kinds
    insights = read_insights(match_dir)
    if insights:
        out["insights"] = {
            "count": len(insights),
            "grounded_pct": round(100 * sum(b.provenance.grounding_ok for b in insights) / len(insights), 1),
            "mode_counts": {m: sum(b.provenance.mode == m for b in insights) for m in ("llm", "mixed", "template")},
            "fallbacks": sum(len(b.provenance.fallbacks) for b in insights),
            "retries": sum(b.provenance.retries for b in insights),
            "mean_latency_ms": round(float(np.mean([b.provenance.latency_ms for b in insights])), 1),
        }
    return out
