from __future__ import annotations

import asyncio

import pytest

from sisyphus.agents import InsightBundle, InsightPipeline, Persona
from sisyphus.agents.contracts import Explanation, PersonalisedBatch, PersonalisedOutput
from sisyphus.agents.workflow import (
    AnalystExecutor,
    AssembleExecutor,
    ModelCaller,
    NarratorExecutor,
    PersonaliserExecutor,
    grounded,
)
from sisyphus.analytics import AnalyticsEngine
from sisyphus.io import replay
from sisyphus.render import render_overlays


@pytest.fixture(scope="module")
def triggers(short_match_dir, short_match):
    eng = AnalyticsEngine(short_match.meta, window_s=60, baseline_s=90)
    out = []
    for tick in replay(short_match_dir):
        out.extend(eng.ingest(tick))
    return out


def test_template_pipeline_end_to_end(triggers):
    personas = [
        Persona(persona_id="casual-en", kind="casual", language="en", favourite_team="Kingsmere Rovers"),
        Persona(persona_id="analyst-en", kind="analyst", language="en"),
    ]
    pipe = InsightPipeline(personas=personas, auto_client=False)
    assert pipe.mode == "template"
    bundles = asyncio.run(pipe.explain_many(triggers, concurrency=4))
    assert len(bundles) == len(triggers)
    for b, t in zip(bundles, triggers, strict=True):
        assert isinstance(b, InsightBundle)
        assert b.trigger_id == t.trigger_id
        assert b.provenance.mode == "template"
        assert b.provenance.grounding_ok
        assert {p.persona_id for p in b.personalised} == {"casual-en", "analyst-en"}
        assert b.commentary.broadcast_line
        assert b.explanation.evidence
        overlays = render_overlays(b, t.snapshot)
        assert len(overlays) == 2
        assert all(o.duration_s > 0 and 1 <= o.priority <= 10 for o in overlays)


class ScriptedCaller(ModelCaller):
    """Stands in for the model: replays scripted outputs/exceptions per agent name."""

    def __init__(self, script: dict[str, list]):
        super().__init__(client=None)
        self.script = {k: list(v) for k, v in script.items()}
        self.calls: list[str] = []

    @property
    def enabled(self) -> bool:
        return True

    def agent(self, name, instructions, schema):
        return name

    async def structured(self, agent, prompt, schema):
        self.calls.append(agent)
        step = self.script[agent].pop(0)
        if isinstance(step, Exception):
            raise step
        return step


def _pipeline_with(caller: ScriptedCaller, personas: list[Persona]) -> InsightPipeline:
    pipe = InsightPipeline(personas=personas, auto_client=False)
    pipe.caller = caller
    pipe._analyst = AnalystExecutor(caller)
    pipe._narrator = NarratorExecutor(caller)
    pipe._personaliser = PersonaliserExecutor(caller)
    pipe._assemble = AssembleExecutor(caller)
    return pipe


def test_model_path_retries_and_falls_back(triggers):
    trig = next(t for t in triggers if t.kind not in ("speed_burst", "milestone", "card"))
    personas = Persona.defaults()
    good = Explanation(
        headline="Grounded",
        why_it_matters="Uses real numbers.",
        evidence=[f"{k} {v}" for k, v in list(trig.evidence.items())[:3] if isinstance(v, int | float)]
        or ["severity 1"],
        tactical_label="control",
        confidence=0.7,
    )
    invented = good.model_copy(update={"evidence": ["99999 sprints", "xG 7.77"]})
    one_persona = PersonalisedBatch(
        outputs=[PersonalisedOutput(persona_id="casual-en", language="en", headline="Hola", body="Texto.")]
    )
    caller = ScriptedCaller(
        {
            "TacticalAnalyst": [invented, good],  # first answer ungrounded -> retry with feedback
            "BroadcastNarrator": [TimeoutError(), TimeoutError()],  # model down -> template fallback
            "AudiencePersonaliser": [one_persona],  # missing analyst persona -> filled from template
        }
    )
    pipe = _pipeline_with(caller, personas)
    bundle = asyncio.run(pipe.explain(trig))

    assert caller.calls == [
        "TacticalAnalyst",
        "TacticalAnalyst",
        "BroadcastNarrator",
        "BroadcastNarrator",
        "AudiencePersonaliser",
    ]
    assert bundle.explanation.headline == "Grounded"
    p = bundle.provenance
    assert p.mode == "mixed"
    assert p.retries >= 2
    assert set(p.fallbacks) == {"broadcast_narrator", "audience_personaliser"}
    assert p.grounding_ok is True
    assert p.agents == ["tactical_analyst", "broadcast_narrator", "audience_personaliser"]
    assert [o.persona_id for o in bundle.personalised] == ["casual-en", "analyst-en"]
    assert bundle.personalised[0].headline == "Hola"


def test_model_path_all_failures_still_completes(triggers):
    trig = next(t for t in triggers if t.kind not in ("speed_burst", "milestone", "card"))
    caller = ScriptedCaller(
        {
            "TacticalAnalyst": [RuntimeError("503"), ValueError("bad json")],
            "BroadcastNarrator": [RuntimeError("503"), RuntimeError("503")],
            "AudiencePersonaliser": [RuntimeError("503")],
        }
    )
    bundle = asyncio.run(_pipeline_with(caller, Persona.defaults()).explain(trig))
    assert bundle.provenance.mode == "mixed"
    assert set(bundle.provenance.fallbacks) == {"tactical_analyst", "broadcast_narrator", "audience_personaliser"}
    assert bundle.provenance.grounding_ok is False
    assert len(bundle.personalised) == 2 and bundle.commentary.broadcast_line


def test_grounding_rejects_invented_numbers():
    evidence = {"xg": 0.31, "shot_distance_m": 14.2, "score": "1-0"}
    ok = Explanation(
        headline="Clinical finish",
        why_it_matters="A good chance taken.",
        evidence=["xG 0.31 from 14.2 m", "makes it 1-0"],
        tactical_label="transition",
        confidence=0.8,
    )
    bad = ok.model_copy(update={"evidence": ["xG 0.75 from 30 m", "sprint count 99"]})
    assert grounded(ok, evidence)[0]
    assert not grounded(bad, evidence)[0]
    assert not grounded(ok.model_copy(update={"evidence": ["no numbers here"]}), evidence)[0]
