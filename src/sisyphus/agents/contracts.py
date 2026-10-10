"""Structured contracts exchanged between the Explain / Personalize agents.

Every agent returns one of these Pydantic models (enforced through structured
outputs when an LLM is used), so the handoffs between agents are typed and the
final :class:`InsightBundle` is always machine-readable.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

TacticalLabel = Literal[
    "control",
    "chaos",
    "pressure_shift",
    "rhythm_change",
    "transition",
    "fatigue",
    "individual",
    "set_piece",
    "neutral",
]

PersonaKind = Literal["casual", "analyst"]


class Persona(BaseModel):
    """Who the output is for. Drives tone, depth and language."""

    persona_id: str
    kind: PersonaKind
    language: str = Field(default="en", description="BCP-47 language code, e.g. 'en', 'es', 'pt-BR'")
    favourite_team: str | None = None
    favourite_player: str | None = None

    @staticmethod
    def defaults() -> list[Persona]:
        return [
            Persona(persona_id="casual-en", kind="casual", language="en"),
            Persona(persona_id="analyst-en", kind="analyst", language="en"),
        ]


class Explanation(BaseModel):
    """Output of the Tactical Analyst agent: *why* this moment matters."""

    headline: str = Field(description="<= 12 words, no exclamation marks")
    why_it_matters: str = Field(description="1-2 sentences of tactical reasoning grounded in the evidence")
    evidence: list[str] = Field(description="2-4 short facts, each quoting a number from the evidence")
    tactical_label: TacticalLabel
    confidence: float = Field(ge=0, le=1)


class Commentary(BaseModel):
    """Output of the Narrator agent: broadcast-ready language."""

    broadcast_line: str = Field(description="<= 22 words, present tense, speakable in one breath")
    studio_note: str = Field(description="<= 60 words for a studio pundit, may reference one statistic")


class StatCallout(BaseModel):
    label: str
    value: str
    unit: str | None = None
    trend: Literal["up", "down", "flat"] | None = None


class PersonalisedOutput(BaseModel):
    """The same insight, rendered for one persona."""

    persona_id: str
    language: str
    headline: str = Field(description="<= 10 words in the persona's language")
    body: str = Field(description="Casual: <= 40 plain words. Analyst: <= 90 words with specific numbers.")
    stat_callouts: list[StatCallout] = Field(default_factory=list, max_length=4)


class PersonalisedBatch(BaseModel):
    outputs: list[PersonalisedOutput]


class Provenance(BaseModel):
    mode: Literal["llm", "template", "mixed"]
    model: str | None = None
    agents: list[str] = Field(default_factory=list)
    retries: int = 0
    fallbacks: list[str] = Field(default_factory=list, description="Agents whose output was replaced by templates")
    grounding_ok: bool = True
    latency_ms: float = 0.0


class InsightBundle(BaseModel):
    """Everything the Render stage needs for one trigger."""

    trigger_id: str
    match_id: str
    kind: str
    period: int
    t: float
    clock: str
    minute: float
    severity: float
    team_id: str | None
    player_id: str | None
    explanation: Explanation
    commentary: Commentary
    personalised: list[PersonalisedOutput]
    provenance: Provenance
