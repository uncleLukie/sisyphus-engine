"""Paced live replay: Ingest -> Interpret in a worker thread, Explain/Personalize async.

The analytics engine is CPU-bound and synchronous, so it runs in a thread and
pushes typed messages onto an asyncio queue. Triggers are handed to the agent
workflow as fire-and-forget tasks so commentary arrives a realistic beat after
the moment, never blocking the tick stream. If ``insights.jsonl`` exists for the
match the stored bundles are replayed instead of calling the model (cheap demos).
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import threading
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from sisyphus.agents import InsightBundle, InsightPipeline, Persona
from sisyphus.analytics import AnalyticsEngine, Trigger
from sisyphus.io import load_match, replay
from sisyphus.pipeline import read_insights
from sisyphus.render import render_overlays, render_tick_overlay

log = logging.getLogger(__name__)

POSITION_HZ = 2  # player positions pushed this often (per match second)
MAX_SPEED = 120.0


@dataclass
class StreamMessage:
    event: str
    data: Any


class MatchStream:
    def __init__(
        self,
        match_dir: Path,
        pipeline: InsightPipeline,
        personas: list[Persona],
        speed: float = 10.0,
        from_minute: float = 0.0,
        live_agents: bool = True,
        kinds: set[str] | None = None,
    ) -> None:
        self.match_dir = Path(match_dir)
        self.pipeline = pipeline
        self.personas = personas
        self.speed = max(0.1, min(MAX_SPEED, speed))
        self.from_minute = from_minute
        self.kinds = kinds
        self.queue: asyncio.Queue[StreamMessage | None] = asyncio.Queue(maxsize=2000)
        self.loop = asyncio.get_event_loop()
        self.stop = threading.Event()
        self.stored = {b.trigger_id: b for b in read_insights(self.match_dir)} if not live_agents else {}
        self.use_stored = bool(self.stored)
        self.tasks: set[asyncio.Task] = set()
        self.data = load_match(self.match_dir, with_truth=False)

    # ---- producer (thread) ----------------------------------------------------
    def _put(self, msg: StreamMessage | None) -> None:
        self.loop.call_soon_threadsafe(self._put_nowait, msg)

    def _put_nowait(self, msg: StreamMessage | None) -> None:
        with contextlib.suppress(asyncio.QueueFull):
            self.queue.put_nowait(msg)

    def _producer(self) -> None:
        engine = AnalyticsEngine(self.data.meta)
        pid_team = {p.player_id: p.team_id for t in (self.data.meta.home, self.data.meta.away) for p in t.players}
        last_sec: tuple[int, int] | None = None
        wall0 = time.perf_counter()
        match0: float | None = None
        sub_every = max(1, int(round(10 / POSITION_HZ)))
        try:
            for tick in replay(self.match_dir):
                if self.stop.is_set():
                    break
                f = tick.frame
                minute = (f.period - 1) * 45 + f.t / 60
                triggers = engine.ingest(tick)
                if minute < self.from_minute:
                    # Fast-forward: analytics still see every frame so windows and
                    # baselines are exact when the client joins mid-match.
                    if f.frame % 3000 == 0:
                        self._put(StreamMessage("seek", {"minute": round(minute, 1), "target": self.from_minute}))
                    continue
                if match0 is None:
                    match0 = minute * 60
                    wall0 = time.perf_counter()
                for tr in triggers:
                    if self.kinds and tr.kind not in self.kinds:
                        continue
                    self._put(StreamMessage("trigger", tr))
                sec_key = (f.period, int(f.t))
                if sec_key != last_sec:
                    last_sec = sec_key
                    snap = engine.snapshot()
                    self._put(StreamMessage("tick", render_tick_overlay(snap)))
                if f.frame % sub_every == 0:
                    self._put(
                        StreamMessage(
                            "positions",
                            {
                                "period": f.period,
                                "t": round(float(f.t), 1),
                                "ball": [round(float(f.ball_xy[0]), 2), round(float(f.ball_xy[1]), 2)],
                                "owner": f.owner_id,
                                "players": [
                                    [pid, pid_team.get(pid, ""), float(x), float(y), float(s)]
                                    for pid, (x, y), s in zip(
                                        f.player_ids,
                                        np.round(f.xy, 2).tolist(),
                                        np.round(f.speed * 3.6, 1).tolist(),
                                        strict=True,
                                    )
                                ],
                            },
                        )
                    )
                # pacing against the wall clock
                target_wall = wall0 + (minute * 60 - match0) / self.speed
                lag = target_wall - time.perf_counter()
                if lag > 0.002:
                    time.sleep(lag)
        except Exception as exc:  # pragma: no cover - surfaced to the client
            log.exception("stream producer failed")
            self._put(StreamMessage("error", {"message": str(exc)}))
        finally:
            self._put(None)

    # ---- consumer (event loop) -------------------------------------------------
    async def _explain(self, trigger: Trigger) -> None:
        try:
            if self.use_stored and trigger.trigger_id in self.stored:
                bundle: InsightBundle = self.stored[trigger.trigger_id]
                await asyncio.sleep(min(2.0, 20.0 / self.speed))  # a believable commentary beat
            else:
                bundle = await self.pipeline.explain(trigger)
            self._put_nowait(StreamMessage("insight", bundle))
            for ov in render_overlays(bundle, trigger.snapshot):
                self._put_nowait(StreamMessage("overlay", ov))
        except Exception as exc:  # pragma: no cover
            log.exception("explain failed for %s", trigger.trigger_id)
            self._put_nowait(StreamMessage("error", {"trigger_id": trigger.trigger_id, "message": str(exc)}))

    async def run(self) -> AsyncIterator[StreamMessage]:
        yield StreamMessage(
            "hello",
            {
                "match_id": self.data.meta.match_id,
                "home": self.data.meta.home.model_dump(),
                "away": self.data.meta.away.model_dump(),
                "speed": self.speed,
                "agent_mode": "stored" if self.use_stored else self.pipeline.mode,
                "personas": [p.model_dump() for p in self.personas],
            },
        )
        thread = threading.Thread(target=self._producer, name="match-stream", daemon=True)
        thread.start()
        try:
            while True:
                msg = await self.queue.get()
                if msg is None:
                    break
                if msg.event == "trigger":
                    task = asyncio.create_task(self._explain(msg.data))
                    self.tasks.add(task)
                    task.add_done_callback(self.tasks.discard)
                yield msg
            if self.tasks:
                await asyncio.wait(self.tasks, timeout=60)
                while not self.queue.empty():
                    msg = self.queue.get_nowait()
                    if msg is not None:
                        yield msg
            yield StreamMessage("end", {"match_id": self.data.meta.match_id})
        finally:
            self.stop.set()
            for t in self.tasks:
                t.cancel()
