"""Explain + Personalize as a Microsoft Agent Framework workflow.

Graph (one run per trigger)::

    intake ──► analyst ──► narrator ──► personaliser ──► assemble
       └──────────────────────┘  (low-severity triggers skip the analyst)

* **intake** stores the trigger and persona list in workflow state and routes.
* **analyst** (Tactical Analyst agent) explains *why* the moment matters, returning
  a structured :class:`Explanation`. Its output is checked for grounding: the
  numbers it cites must come from the trigger evidence. One retry with feedback,
  then a template fallback.
* **narrator** (Broadcast Narrator agent) turns the explanation into a speakable
  line and a studio note.
* **personaliser** (Audience agent) renders the insight once per persona, in the
  persona's language and depth, honouring favourite team/player.
* **assemble** builds the final :class:`InsightBundle` with provenance.

Each node is a :class:`agent_framework.Executor`; handoffs are typed messages
carrying only ids, with the payloads living in shared workflow state. The same
graph runs with or without a model: when no client is configured every agent
node answers from templates, so the pipeline is testable offline.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Never

from agent_framework import Agent, Executor, WorkflowBuilder, WorkflowContext, handler
from pydantic import BaseModel, ValidationError

from sisyphus.agents import templates
from sisyphus.agents.clients import describe_client, make_chat_client
from sisyphus.agents.contracts import (
    Commentary,
    Explanation,
    InsightBundle,
    Persona,
    PersonalisedBatch,
    PersonalisedOutput,
    Provenance,
)
from sisyphus.analytics.contracts import Trigger

LOW_SEVERITY_KINDS = {"speed_burst", "milestone", "card"}
LLM_TIMEOUT_S = 40.0


# ---------------------------------------------------------------------- messages
@dataclass
class AnalystTask:
    trigger_id: str


@dataclass
class NarratorTask:
    trigger_id: str
    explanation: Explanation


@dataclass
class PersonaliseTask:
    trigger_id: str
    explanation: Explanation
    commentary: Commentary


@dataclass
class AssembleTask:
    trigger_id: str
    explanation: Explanation
    commentary: Commentary
    personalised: list[PersonalisedOutput]


@dataclass
class RunTrace:
    """Mutable provenance shared by the executors of one run."""

    agents: list[str] = field(default_factory=list)
    fallbacks: list[str] = field(default_factory=list)
    retries: int = 0
    grounding_ok: bool = True
    started: float = field(default_factory=time.perf_counter)


# ---------------------------------------------------------------------- grounding
_NUM = re.compile(r"-?\d+(?:\.\d+)?")


def evidence_numbers(evidence: dict[str, Any]) -> set[str]:
    """All numeric tokens an agent may legitimately quote, in a few equivalent forms."""
    out: set[str] = set()
    for v in evidence.values():
        for tok in _NUM.findall(str(v)):
            out.add(tok)
            try:
                x = float(tok)
            except ValueError:
                continue
            out.add(f"{x:.0f}")
            out.add(f"{x:.1f}")
            if 0 < abs(x) <= 1.5:
                out.add(f"{100 * x:.0f}")  # ratios quoted as percentages
                out.add(f"{100 * (1 - x):.0f}")  # "down 39%"
            if x >= 1000:
                out.add(f"{x / 1000:.1f}")  # metres quoted as km
    return out


def grounded(explanation: Explanation, evidence: dict[str, Any]) -> tuple[bool, str]:
    allowed = evidence_numbers(evidence)
    cited = [tok for b in explanation.evidence for tok in _NUM.findall(b)]
    if not cited:
        return False, "evidence bullets contain no numbers"
    unknown = [c for c in cited if c not in allowed and f"{float(c):.0f}" not in allowed]
    if len(unknown) > len(cited) / 2:
        return False, f"numbers not found in the evidence: {sorted(set(unknown))[:6]}"
    return True, ""


# ---------------------------------------------------------------------- LLM helper
class ModelCaller:
    """Thin wrapper that runs an :class:`Agent` and parses structured JSON output."""

    def __init__(self, client: Any | None):
        self.client = client
        self.info = describe_client()

    @property
    def enabled(self) -> bool:
        return self.client is not None

    def agent(self, name: str, instructions: str, schema: type[BaseModel]) -> Agent:
        from agent_framework.openai import OpenAIChatOptions

        return Agent(
            client=self.client,
            name=name,
            instructions=instructions,
            default_options=OpenAIChatOptions[Any](response_format=schema, temperature=0.4),
        )

    async def structured(self, agent: Agent, prompt: str, schema: type[BaseModel]) -> BaseModel:
        result = await asyncio.wait_for(agent.run(prompt), timeout=LLM_TIMEOUT_S)
        text = result.text if hasattr(result, "text") else str(result)
        return schema.model_validate_json(_strip_fences(text))


def _strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        text = text.rsplit("```", 1)[0]
    return text.strip()


def _context_block(t: Trigger) -> str:
    s = t.snapshot
    return json.dumps(
        {
            "clock": t.clock,
            "score": f"{s.home.short_name} {s.home.score}-{s.away.score} {s.away.short_name}",
            "home": {
                "name": s.home.name,
                "possession_pct": s.home.possession_pct,
                "energy_index": s.home.energy.index,
                "energy_trend": s.home.energy.trend,
                "xg": s.home.shooting.xg,
                "pass_accuracy": s.home.passing.accuracy,
                "sprints_last_5min": s.home.physical.sprint_count,
            },
            "away": {
                "name": s.away.name,
                "possession_pct": s.away.possession_pct,
                "energy_index": s.away.energy.index,
                "energy_trend": s.away.energy.trend,
                "xg": s.away.shooting.xg,
                "pass_accuracy": s.away.passing.accuracy,
                "sprints_last_5min": s.away.physical.sprint_count,
            },
            "momentum_home_minus_away": s.momentum,
        },
        ensure_ascii=False,
    )


ANALYST_INSTRUCTIONS = """You are the Tactical Analyst in a live football broadcast intelligence system.
You receive one match moment (a trigger) with the exact evidence that fired it, plus match context.
Explain WHY the moment matters tactically - not just what happened. Frame it in terms a coach would use:
control vs chaos, pressing structure, fatigue, rhythm, transitions.
Rules:
- Only use numbers that appear in the evidence or context. Never invent statistics, names or events.
- Each evidence bullet must quote at least one number from the evidence.
- Energy index is 0-100 where 100 means the team is matching its own opening intensity.
- Be concrete and brief. No exclamation marks. No hedging filler.
Return JSON matching the Explanation schema."""

NARRATOR_INSTRUCTIONS = """You are the Broadcast Narrator. Turn a tactical explanation into language for live television.
- broadcast_line: at most 22 words, present tense, natural for a commentator to say in one breath.
- studio_note: at most 60 words for a studio pundit, may cite one statistic from the explanation.
Keep names and numbers exactly as given. Return JSON matching the Commentary schema."""

PERSONALISER_INSTRUCTIONS = """You are the Audience Personaliser. Render ONE insight for EACH persona provided.
- casual: plain, warm, at most 40 words, no jargon, one simple number at most; headline at most 10 words.
- analyst: precise, at most 90 words, cite the specific numbers from the evidence; headline at most 10 words;
  include up to 4 stat_callouts (label/value/unit/trend) taken from the evidence.
- Write entirely in the persona's language code (e.g. 'es' -> Spanish, 'pt-BR' -> Brazilian Portuguese). Keep
  player and club names unchanged. Set the output language field to that code.
- If the persona has a favourite_team or favourite_player that is involved, lead with their angle.
- Never invent facts; only use the explanation, commentary and evidence given.
Return JSON matching the PersonalisedBatch schema with exactly one output per persona, same persona_id."""


# ---------------------------------------------------------------------- executors
class IntakeExecutor(Executor):
    def __init__(self, personas: list[Persona]):
        super().__init__(id="intake")
        self.personas = personas

    @handler
    async def run(self, trigger: Trigger, ctx: WorkflowContext[AnalystTask | NarratorTask]) -> None:
        ctx.set_state("trigger", trigger)
        ctx.set_state("personas", self.personas)
        ctx.set_state("trace", RunTrace())
        if trigger.kind in LOW_SEVERITY_KINDS:
            # Ticker-grade items do not need tactical analysis: hand straight to the narrator.
            await ctx.send_message(NarratorTask(trigger.trigger_id, templates.explanation(trigger)))
        else:
            await ctx.send_message(AnalystTask(trigger.trigger_id))


class AnalystExecutor(Executor):
    def __init__(self, caller: ModelCaller):
        super().__init__(id="tactical_analyst")
        self.caller = caller
        self._agent = caller.agent("TacticalAnalyst", ANALYST_INSTRUCTIONS, Explanation) if caller.enabled else None

    @handler
    async def run(self, task: AnalystTask, ctx: WorkflowContext[NarratorTask]) -> None:
        trigger: Trigger = ctx.get_state("trigger")
        trace: RunTrace = ctx.get_state("trace")
        trace.agents.append(self.id)
        explanation: Explanation | None = None
        if self._agent is not None:
            prompt = (
                f"TRIGGER kind={trigger.kind} clock={trigger.clock} severity={trigger.severity:.2f}\n"
                f"HEADLINE: {trigger.headline}\n"
                f"EVIDENCE: {json.dumps(trigger.evidence, ensure_ascii=False)}\n"
                f"MATCH CONTEXT: {_context_block(trigger)}"
            )
            feedback = ""
            for _attempt in range(2):
                try:
                    candidate = await self.caller.structured(self._agent, prompt + feedback, Explanation)
                    ok, reason = grounded(candidate, trigger.evidence)
                    if ok:
                        explanation = candidate
                        break
                    feedback = (
                        f"\n\nYour previous answer failed validation: {reason}. Quote only numbers from EVIDENCE."
                    )
                except (ValidationError, ValueError, TimeoutError, OSError, RuntimeError):
                    pass
                trace.retries += 1
        if explanation is None:
            explanation = templates.explanation(trigger)
            if self._agent is not None:
                # The model never produced a grounded explanation; the template is grounded by construction,
                # but we flag the run so reviewers can see the model output was rejected.
                trace.grounding_ok = False
                trace.fallbacks.append(self.id)
        ctx.set_state("trace", trace)  # workflow state is copy-on-read; write the trace back
        await ctx.send_message(NarratorTask(task.trigger_id, explanation))


class NarratorExecutor(Executor):
    def __init__(self, caller: ModelCaller):
        super().__init__(id="broadcast_narrator")
        self.caller = caller
        self._agent = caller.agent("BroadcastNarrator", NARRATOR_INSTRUCTIONS, Commentary) if caller.enabled else None

    @handler
    async def run(self, task: NarratorTask, ctx: WorkflowContext[PersonaliseTask]) -> None:
        trigger: Trigger = ctx.get_state("trigger")
        trace: RunTrace = ctx.get_state("trace")
        trace.agents.append(self.id)
        commentary: Commentary | None = None
        if self._agent is not None:
            prompt = (
                f"MOMENT ({trigger.kind} at {trigger.clock}): {trigger.headline}\n"
                f"EXPLANATION: {task.explanation.model_dump_json()}\n"
                f"KEY EVIDENCE: {json.dumps(trigger.evidence, ensure_ascii=False)}"
            )
            try:
                commentary = await self.caller.structured(self._agent, prompt, Commentary)
            except (ValidationError, ValueError, TimeoutError, OSError, RuntimeError):
                trace.retries += 1
                try:
                    commentary = await self.caller.structured(self._agent, prompt, Commentary)
                except (ValidationError, ValueError, TimeoutError, OSError, RuntimeError):
                    commentary = None
        if commentary is None:
            commentary = templates.commentary(trigger, task.explanation)
            if self._agent is not None:
                trace.fallbacks.append(self.id)
        ctx.set_state("trace", trace)
        await ctx.send_message(PersonaliseTask(task.trigger_id, task.explanation, commentary))


class PersonaliserExecutor(Executor):
    def __init__(self, caller: ModelCaller):
        super().__init__(id="audience_personaliser")
        self.caller = caller
        self._agent = (
            caller.agent("AudiencePersonaliser", PERSONALISER_INSTRUCTIONS, PersonalisedBatch)
            if caller.enabled
            else None
        )

    @handler
    async def run(self, task: PersonaliseTask, ctx: WorkflowContext[AssembleTask]) -> None:
        trigger: Trigger = ctx.get_state("trigger")
        personas: list[Persona] = ctx.get_state("personas")
        trace: RunTrace = ctx.get_state("trace")
        trace.agents.append(self.id)
        outputs: dict[str, PersonalisedOutput] = {}
        if self._agent is not None:
            prompt = (
                f"PERSONAS: {json.dumps([p.model_dump() for p in personas], ensure_ascii=False)}\n"
                f"MOMENT ({trigger.kind} at {trigger.clock}): {trigger.headline}\n"
                f"EXPLANATION: {task.explanation.model_dump_json()}\n"
                f"COMMENTARY: {task.commentary.model_dump_json()}\n"
                f"EVIDENCE: {json.dumps(trigger.evidence, ensure_ascii=False)}"
            )
            try:
                batch = await self.caller.structured(self._agent, prompt, PersonalisedBatch)
                outputs = {o.persona_id: o for o in batch.outputs}
            except (ValidationError, ValueError, TimeoutError, OSError, RuntimeError):
                trace.retries += 1
        missing = [p for p in personas if p.persona_id not in outputs]
        if missing and self._agent is not None:
            trace.fallbacks.append(self.id)
        for p in missing:
            outputs[p.persona_id] = templates.personalise(trigger, task.explanation, task.commentary, p)
        ordered = [outputs[p.persona_id] for p in personas]
        ctx.set_state("trace", trace)
        await ctx.send_message(AssembleTask(task.trigger_id, task.explanation, task.commentary, ordered))


class AssembleExecutor(Executor):
    def __init__(self, caller: ModelCaller):
        super().__init__(id="assemble")
        self.caller = caller

    @handler
    async def run(self, task: AssembleTask, ctx: WorkflowContext[Never, InsightBundle]) -> None:
        trigger: Trigger = ctx.get_state("trigger")
        trace: RunTrace = ctx.get_state("trace")
        if not self.caller.enabled:
            mode = "template"
        elif trace.fallbacks:
            mode = "mixed"
        else:
            mode = "llm"
        bundle = InsightBundle(
            trigger_id=trigger.trigger_id,
            match_id=trigger.match_id,
            kind=trigger.kind,
            period=trigger.period,
            t=trigger.t,
            clock=trigger.clock,
            minute=trigger.minute,
            severity=trigger.severity,
            team_id=trigger.team_id,
            player_id=trigger.player_id,
            explanation=task.explanation,
            commentary=task.commentary,
            personalised=task.personalised,
            provenance=Provenance(
                mode=mode,
                model=self.caller.info.model,
                agents=list(trace.agents),
                retries=trace.retries,
                fallbacks=list(trace.fallbacks),
                grounding_ok=trace.grounding_ok,
                latency_ms=round((time.perf_counter() - trace.started) * 1000, 1),
            ),
        )
        await ctx.yield_output(bundle)


# ---------------------------------------------------------------------- pipeline
class InsightPipeline:
    """Runs the Explain/Personalize workflow for triggers."""

    def __init__(self, personas: list[Persona] | None = None, client: Any | None = None, auto_client: bool = True):
        self.personas = personas or Persona.defaults()
        if client is None and auto_client:
            client = make_chat_client()
        self.client = client
        self.caller = ModelCaller(client)
        # Agents are created once and reused; the workflow graph is rebuilt per run
        # so executor state never leaks between triggers.
        self._analyst = AnalystExecutor(self.caller)
        self._narrator = NarratorExecutor(self.caller)
        self._personaliser = PersonaliserExecutor(self.caller)
        self._assemble = AssembleExecutor(self.caller)

    @property
    def mode(self) -> str:
        return self.caller.info.provider

    def build(self):
        intake = IntakeExecutor(self.personas)
        return (
            WorkflowBuilder(start_executor=intake)
            .add_edge(intake, self._analyst, condition=lambda m: isinstance(m, AnalystTask))
            .add_edge(intake, self._narrator, condition=lambda m: isinstance(m, NarratorTask))
            .add_edge(self._analyst, self._narrator)
            .add_edge(self._narrator, self._personaliser)
            .add_edge(self._personaliser, self._assemble)
            .build()
        )

    async def explain(self, trigger: Trigger) -> InsightBundle:
        events = await self.build().run(trigger)
        outputs = events.get_outputs()
        if not outputs:
            raise RuntimeError("workflow produced no output")
        return outputs[0]

    async def explain_many(self, triggers: list[Trigger], concurrency: int = 3) -> list[InsightBundle]:
        sem = asyncio.Semaphore(concurrency)

        async def one(t: Trigger) -> InsightBundle:
            async with sem:
                return await self.explain(t)

        return list(await asyncio.gather(*(one(t) for t in triggers)))
