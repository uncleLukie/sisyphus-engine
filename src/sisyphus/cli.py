"""``sisyphus`` command line: simulate, analyze, narrate, eval, serve, ui."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

DEFAULT_DATA = Path(os.environ.get("SISYPHUS_DATA_DIR", "data/generated"))


def _progress_bar(label: str):
    start = time.perf_counter()
    last = {"pct": -1}

    def cb(p: float) -> None:
        pct = int(p * 100)
        if pct != last["pct"]:
            last["pct"] = pct
            sys.stderr.write(f"\r{label}: {pct:3d}%  ({time.perf_counter() - start:5.1f}s)")
            sys.stderr.flush()

    return cb


def cmd_simulate(args: argparse.Namespace) -> int:
    from sisyphus.io import generate_match
    from sisyphus.sim.scenario import SCENARIOS

    if args.list:
        for s in SCENARIOS.values():
            print(f"{s.scenario_id:16s} {s.title} - {s.description[:90]}...")
        return 0
    cb = _progress_bar("simulating")
    meta, path = generate_match(args.scenario, args.out, seed=args.seed, progress=cb)
    sys.stderr.write("\n")
    print(f"match {meta.match_id} written to {path}")
    print(f"final score {meta.home.name} {meta.final_score[0]} - {meta.final_score[1]} {meta.away.name}")
    if args.analyze:
        args.match = path
        return cmd_analyze(args)
    return 0


def cmd_analyze(args: argparse.Namespace) -> int:
    from sisyphus.pipeline import analyze_match, save_analysis

    cb = _progress_bar("analyzing")
    result = analyze_match(args.match, progress=cb)
    sys.stderr.write("\n")
    save_analysis(args.match, result)
    print(f"{len(result.triggers)} triggers, {len(result.timeline)} timeline rows -> {args.match}")
    for t in result.triggers:
        if t.kind in ("speed_burst", "milestone") and not args.verbose:
            continue
        print(f"  {t.clock:>6} {t.kind:19s} sev={t.severity:.2f}  {t.headline}")
    return 0


def _personas(args: argparse.Namespace):
    from sisyphus.agents import Persona

    personas = Persona.defaults()
    for lang in args.language or []:
        if lang != "en":
            personas.append(Persona(persona_id=f"casual-{lang}", kind="casual", language=lang))
    if args.favourite_team:
        personas[0] = personas[0].model_copy(update={"favourite_team": args.favourite_team})
    return personas


def cmd_narrate(args: argparse.Namespace) -> int:
    from sisyphus.pipeline import narrate_match_sync

    kinds = set(args.kinds.split(",")) if args.kinds else None
    t0 = time.perf_counter()
    bundles = narrate_match_sync(
        args.match,
        personas=_personas(args),
        kinds=kinds,
        max_triggers=args.limit,
        concurrency=args.concurrency,
    )
    print(f"{len(bundles)} insights in {time.perf_counter() - t0:.1f}s -> {args.match}")
    for b in bundles:
        if b.kind in ("speed_burst", "milestone") and not args.verbose:
            continue
        print(f"\n[{b.clock}] {b.kind}  ({b.provenance.mode}, grounded={b.provenance.grounding_ok})")
        print(f"  {b.commentary.broadcast_line}")
        print(f"  why: {b.explanation.why_it_matters}")
        for p in b.personalised:
            print(f"  - {p.persona_id}: {p.headline}")
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    from sisyphus.pipeline import evaluate_match

    print(json.dumps(evaluate_match(args.match), indent=2))
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    os.environ.setdefault("SISYPHUS_DATA_DIR", str(DEFAULT_DATA))
    uvicorn.run("sisyphus.service.app:app", host=args.host, port=args.port, reload=args.reload)
    return 0


def cmd_ui(args: argparse.Namespace) -> int:
    app = Path(__file__).parent / "ui" / "app.py"
    env = dict(os.environ)
    env.setdefault("SISYPHUS_API_URL", args.api)
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app),
        "--server.port",
        str(args.port),
        "--server.headless",
        "true",
        "--browser.gatherUsageStats",
        "false",
    ]
    return subprocess.call(cmd, env=env)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="sisyphus", description="Inside the Game: synthetic match insight engine")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("simulate", help="generate a synthetic match")
    s.add_argument("--scenario", default="press_collapse")
    s.add_argument("--seed", type=int, default=None)
    s.add_argument("--out", type=Path, default=DEFAULT_DATA)
    s.add_argument("--analyze", action="store_true", help="run analysis immediately after")
    s.add_argument("--list", action="store_true", help="list scenarios")
    s.add_argument("-v", "--verbose", action="store_true")
    s.set_defaults(func=cmd_simulate)

    a = sub.add_parser("analyze", help="run the Interpret stage over a match directory")
    a.add_argument("match", type=Path)
    a.add_argument("-v", "--verbose", action="store_true")
    a.set_defaults(func=cmd_analyze)

    n = sub.add_parser("narrate", help="run the agent workflow over a match's triggers")
    n.add_argument("match", type=Path)
    n.add_argument("--kinds", default=None, help="comma-separated trigger kinds to narrate")
    n.add_argument("--limit", type=int, default=None)
    n.add_argument("--language", action="append", help="extra casual-fan language (repeatable)")
    n.add_argument("--favourite-team", default=None)
    n.add_argument("--concurrency", type=int, default=3)
    n.add_argument("-v", "--verbose", action="store_true")
    n.set_defaults(func=cmd_narrate)

    e = sub.add_parser("eval", help="score observable analytics against hidden simulator truth")
    e.add_argument("match", type=Path)
    e.set_defaults(func=cmd_eval)

    sv = sub.add_parser("serve", help="start the FastAPI service")
    sv.add_argument("--host", default="0.0.0.0")
    sv.add_argument("--port", type=int, default=8000)
    sv.add_argument("--reload", action="store_true")
    sv.set_defaults(func=cmd_serve)

    u = sub.add_parser("ui", help="start the Streamlit dashboard")
    u.add_argument("--port", type=int, default=8501)
    u.add_argument("--api", default="http://localhost:8000")
    u.set_defaults(func=cmd_ui)
    return p


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
