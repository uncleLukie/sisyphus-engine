# Project Title
Sisyphus Engine — Inside the Game

# Tagline
Sisyphus Engine watches a synthetic football match the way a fitness coach does: it measures when a high-pressing team's legs go, explains *why* the game is turning, and tells every viewer in their own language — a fan gets a gauge and a sentence, an analyst gets the evidence.

# Keywords
Sports Analytics, Player Tracking, Fatigue Detection, Multi-Agent Systems, Microsoft Agent Framework, Microsoft Foundry, Azure Container Apps, Real-Time Analytics, Synthetic Data, Explainable AI, Multi-Language Storytelling, SPADL

# Description

## Overview
High-pressing teams win the ball high and early — until they can't. The drop-off is physical before it is tactical: sprint counts fall, pressers arrive a step late, the block stretches, and a counter-attack that was harmless at 20' is a goal at 75'. Broadcast statistics (distance covered, possession) show this long after it has decided the match.

Sisyphus Engine is a complete **Ingest → Interpret → Explain → Render → Personalize** pipeline over fully synthetic tracking and event data. A purpose-built agent-based simulator produces 10 Hz positions for 22 players plus a typed event stream; a transparent analytics layer turns that into rolling physical metrics, a *Team Energy Index*, pass quality, shot speed and momentum; a Microsoft Agent Framework workflow of four specialised agents turns detected moments into grounded, personalised commentary in any language; and a FastAPI service streams the whole thing — positions, metrics, triggers, commentary and timed overlay payloads — to a Streamlit second screen.

Everything is synthetic: fictional clubs (Harbourside Athletic, Kingsmere Rovers), fictional players, generated data. No third-party footage, logos, names or statistics.

## Problem
1. **Fatigue is invisible on TV.** Nobody tells you a press has died until a goal makes it obvious.
2. **Insight is one-size-fits-all.** The same graphic goes to a first-time viewer and a coach.
3. **AI commentary hallucinates.** Language models are happy to invent a statistic. In sport that is disqualifying.

## Key Features
- **Live fatigue detection from positions alone.** 5-minute rolling windows versus each team's own 15-minute opening baseline; sprint count, high-intensity distance, mean speed and press rate are *phase-normalised* (in vs out of possession) so a team sitting in a low block is not mistaken for a tired one. The simulator's hidden stamina is never read by the analytics; it is only used afterwards to score them (`sisyphus eval`: observable index vs hidden truth r ≈ 0.7–0.85).
- **Team Energy Index and breakpoints.** A 0–100 index with trend; thresholds at 80/65/50 fire `fatigue_breakpoint` triggers with the evidence attached (sprints now vs opening, HI distance ratio, press ratio, compactness, line height, most-fatigued player).
- **Judged features covered.** Player identification with speed/distance thresholds (sprint ≥ 7 m/s, HI ≥ 5.5 m/s, top speeds, per-player distance milestones), pass quality (distance, accuracy, progressive, a transparent logistic difficulty score), ball and shot speed (km/h), narrative generation, explainability (control vs chaos labels, pressure shifts, rhythm change, momentum swings), synthetic data innovation (two-compartment stamina + willingness model drives emergent collapse), multi-language storytelling.
- **Multi-agent Explain/Personalize with provenance.** Four agents with distinct roles, typed handoffs, shared workflow state, a grounding validator that rejects numbers not present in the evidence, bounded retries and a deterministic template fallback. Every output carries a `Provenance` record: mode (llm / mixed / template), model, agents involved, retries, fallbacks, grounding result, latency.
- **Two audiences, N languages.** Personas are `casual` or `analyst` with a BCP-47 language and optional favourite team/player. The same trigger produces a 40-word plain-language card for a fan in Spanish and a 90-word evidence-led note for an analyst in English, from one run.
- **Render as timed, machine-readable overlays.** Each insight becomes `OverlayPayload`s (lower third / stat card / gauge / ticker / recap) with `show_at`, `duration_s`, `priority`, persona and language — what a graphics system actually consumes.
- **Live or stored.** The SSE stream replays a match at 0.1–120× with analytics computed on the fly; agents can run live per trigger or replay pre-narrated insights for zero-cost demos. Join mid-match and the engine fast-forwards so baselines are exact.

## Pipeline Architecture
| Stage | Module | What it does |
|---|---|---|
| **Ingest** | `sisyphus.sim`, `sisyphus.io` | Agent-based 10 fps simulator (105 × 68 m, tactical styles, scripted substitutions, two-compartment stamina `e_short`/`e_long`, sigmoid "willingness" that throttles sprinting as energy falls). Writes `tracking.parquet` (long format), `events.jsonl` (typed schema), `frames.parquet`, hidden `truth.parquet`, and a SPADL-style `spadl.csv` export. `replay()` re-emits the recorded match tick by tick so every downstream stage is source-agnostic. |
| **Interpret** | `sisyphus.analytics` | O(1) rolling-window sums per second; per-team physical/energy/passing/shooting snapshots; per-player physical state; convex-hull compactness, block length, defensive line; press rate × closing intensity; xT-like threat and a 3-minute momentum EMA; `TriggerDetector` emits 11 trigger kinds with evidence. |
| **Explain / Personalize** | `sisyphus.agents` | Microsoft Agent Framework `WorkflowBuilder` graph: `Intake` → `TacticalAnalyst` → `BroadcastNarrator` → `AudiencePersonaliser` → `Assemble`. Low-severity moments bypass the analyst (conditional edge). Structured outputs (Pydantic), grounding check, retry with feedback, template fallback, shared `RunTrace` state. Model via `FoundryChatClient` (Entra ID) or Azure OpenAI; no credentials → template mode. |
| **Render** | `sisyphus.render` | `OverlayPayload` per persona per insight; compact per-second tick payload (score, energy, trend, possession, xG, sprints, press rate, momentum, ball). |
| **Serve** | `sisyphus.service`, `sisyphus.ui` | FastAPI REST + SSE (`/matches/{id}/stream`), on-demand per-viewer explanation (`POST /matches/{id}/explain/{trigger}`), background simulate/narrate jobs, evaluation endpoint. Streamlit dashboard: pitch view, energy gauges, momentum, energy timeline with trigger markers, commentary feed, fan/analyst/language/favourite-team selectors, match review tabs. |

```
sim (tick) ──► replay/SSE producer ──► AnalyticsEngine ──► Trigger(evidence, snapshot)
                                              │                      │
                                              ▼                      ▼
                                      tick overlay JSON     InsightPipeline (MAF workflow)
                                                                     │
                                                       InsightBundle + Provenance
                                                                     │
                                                      OverlayPayload[] per persona/language
```

## Multi-agent design (Best Multi-Agent System)
- **Distinct roles:** Intake (routing + evidence framing), Tactical Analyst (why it matters, tactical label, confidence), Broadcast Narrator (speakable line + studio note), Audience Personaliser (per-persona, per-language), Assemble (provenance).
- **Shared state:** `WorkflowContext` state holds the trigger, evidence, personas and a mutable `RunTrace`; each executor reads and appends.
- **Handoffs:** typed dataclass messages (`AnalystTask` → `NarratorTask` → `PersonaliseTask` → `AssembleTask`) over `WorkflowBuilder` edges with `isinstance` conditions.
- **Failure recovery:** analyst output that fails grounding is retried once with the validator's feedback, then replaced by the template; a timed-out or failed personaliser falls back per persona; the run always completes and says what happened in `Provenance`.

## Cloud native (Best Azure Cloud Native Integration)
- One image, two roles (`sisyphus serve`, `sisyphus ui`), `Dockerfile` with a pre-generated demo match baked in.
- `infra/main.bicep` + `azure.yaml` (`azd up`): Container Apps environment, ACR, user-assigned managed identity (AcrPull + *Azure AI User*), Microsoft Foundry resource + project + `gpt-4.1-mini` deployment with `disableLocalAuth`, Azure Files mounted for persistent generated matches, Log Analytics + Application Insights, liveness probes, scale 1–2.
- `docker-compose.yml` for local parity; GitHub Actions CI (ruff, pytest, Docker build).

## Tech Stack
- **AI / orchestration:** Microsoft Foundry (model hosting, Entra ID auth), Microsoft Agent Framework (Python `agent-framework-core`, `agent-framework-foundry`): `Agent`, `Executor`, `WorkflowBuilder`, structured outputs.
- **Data:** Python 3.12, numpy, pandas, pyarrow (parquet), scipy (convex hulls), pydantic schemas; SPADL-style export for compatibility with the open football-analytics ecosystem.
- **Service / UI:** FastAPI + uvicorn (SSE), Streamlit + Plotly.
- **Cloud:** Azure Container Apps, Azure Container Registry, Azure Files, Log Analytics, Application Insights; Bicep + Azure Developer CLI.
- **Quality:** uv, ruff, pytest (simulator realism, analytics, agent workflow, API/SSE), GitHub Actions.

## Target Categories
- **Grand Prize:** complete five-stage pipeline with a testable live demo.
- **Best Multi-Agent System:** distinct roles, shared state, typed handoffs, grounding + fallback failure recovery, full provenance.
- **Best Azure Cloud Native Integration:** Container Apps + Foundry provisioned by Bicep/azd, managed identity end to end.
- Also exercises **Best Use of Microsoft Foundry** (model deployment + Agent Framework) and the enterprise angle (synthetic-only data, no secrets in code, deterministic offline mode).

---

## Running it

```bash
uv sync                                   # Python 3.12, installs everything
uv run sisyphus simulate --seed 7 --analyze    # ~1 min: data/generated/press_collapse-s7
uv run sisyphus narrate data/generated/press_collapse-s7 --language es   # template mode without credentials
uv run sisyphus eval data/generated/press_collapse-s7                    # observable analytics vs hidden truth
uv run sisyphus serve                     # API on :8000  (docs at /docs)
uv run sisyphus ui                        # dashboard on :8501
```

Or `docker compose up --build` (API :8000, UI :8501, demo match included).

### Turning on the model
Copy `.env.example` to `.env`. Recommended: a Microsoft Foundry project with a `gpt-4.1-mini` deployment, `az login`, and

```
FOUNDRY_PROJECT_ENDPOINT=https://<resource>.services.ai.azure.com/api/projects/<project>
FOUNDRY_MODEL=gpt-4.1-mini
```

`GET /health` reports which provider is active. Without credentials the agents run in **template mode** — the same workflow graph, deterministic text — which is also the failure-recovery path when a model call fails or produces ungrounded numbers.

### Cost and credits (as researched, October 2026)
- We found **no hackathon-provided Azure or Foundry credits** in the rules; budget for pay-as-you-go.
- Per match the full narrated run is roughly 40–60 triggers × 3 agent calls on `gpt-4.1-mini` ≈ **well under US$0.10**; live demos re-use stored insights by default (`live_agents=false`), so a judge clicking around costs nothing.
- **GitHub Models' free tier was retired (July 2026)** — it is not an option. Azure free account ($200/30 days) and Azure for Students ($100) exist but new/trial subscriptions are often quota-limited for model deployments; a pay-as-you-go subscription with a small `gpt-4.1-mini` GlobalStandard deployment is the dependable route.

### API sketch
| Endpoint | Purpose |
|---|---|
| `GET /health`, `/scenarios`, `/personas`, `/matches`, `/matches/{id}` | discovery |
| `GET /matches/{id}/triggers|insights|overlays|timeline|eval` | stored artefacts (analysis runs lazily if missing) |
| `GET /matches/{id}/stream?speed=20&from_minute=55&personas=casual:es,analyst:en&favourite_team=...&live_agents=true` | SSE: `hello`, `seek`, `tick`, `positions`, `trigger`, `insight`, `overlay`, `end` |
| `POST /matches/{id}/explain/{trigger_id}` | per-viewer explanation with any personas |
| `POST /simulate`, `POST /matches/{id}/narrate`, `GET /jobs/{id}` | background generation |

### Repository layout
```
src/sisyphus/
  sim/        simulator: pitch, roster, tactics, stamina, models, engine, scenario
  io/         writers (parquet/jsonl/spadl), readers, replay
  analytics/  contracts, engine (rolling windows), detectors (triggers), passes
  agents/     contracts, clients (Foundry/Azure OpenAI/template), templates, workflow (MAF)
  render/     overlay payloads
  service/    FastAPI app + SSE stream
  ui/         Streamlit dashboard
  pipeline.py offline orchestration + evaluation
  cli.py      sisyphus simulate | analyze | narrate | eval | serve | ui
infra/        Bicep for Azure Container Apps + Foundry;  azure.yaml for azd
tests/        simulator realism, analytics, agents, API/SSE
docs/hackathon-rules/   the brief and rules this was built against
```

### Known limitations
- High-intensity distance per player (~2.5 km) is above elite norms (~1 km); sprint counts and total distance are in range. Calibration item, does not affect the fatigue signal.
- The simulator has no offside rule; attackers are kept out of the six-yard box positionally instead.
- Template mode is English-only; non-English personas need a model.
- Bicep is written to current API versions but has not been deployed from this machine (no `az`/`azd` here); expect to adjust `modelLocation`/`modelVersion` to your subscription's availability.
