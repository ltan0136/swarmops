"""Command line for the replay 2x2 eval.

Record live traffic as replayable episodes:
    python -m swarmops.evals record evals/inputs/sample.jsonl --out evals/data/my_week.jsonl

Compare two prompt versions on two recorded weeks of traffic:
    python -m swarmops.evals replay \\
        --old-traffic evals/data/old_traffic.jsonl --new-traffic evals/data/new_traffic.jsonl \\
        --new-prompts evals/prompts/v2_cautious.json --gate
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from swarmops import __version__
from swarmops.config import Settings, get_settings
from swarmops.evals.episodes import Expected, load_episodes, read_jsonl, save_episodes
from swarmops.evals.report import FINDING_TEXT, build_report, to_markdown
from swarmops.evals.runner import record, run_grid
from swarmops.swarm.prompts import DEFAULT_PROMPTS, PromptSet

BLOCKING_FINDINGS = {"prompt", "interaction"}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]


def _manifest(
    settings: Settings, args: argparse.Namespace, prompts: dict[str, PromptSet], counts: dict
) -> dict[str, Any]:
    """Everything needed to reproduce the run, so a later reader can tell what was compared."""
    return {
        "swarmops_version": __version__,
        "llm_provider": settings.llm_provider,
        "llm_model": settings.llm_model if settings.llm_provider != "fake" else "scripted",
        "llm_temperature": settings.llm_temperature,
        "recursion_limit": settings.recursion_limit,
        "on_miss": args.on_miss,
        "alpha": args.alpha,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "prompts": {
            key: {"version": p.version, "fingerprint": p.fingerprint} for key, p in prompts.items()
        },
        "traffic": {
            key: {"path": str(path), "sha256": _sha256(path), "episodes": counts[key]}
            for key, path in (("old", args.old_traffic), ("new", args.new_traffic))
        },
    }


async def _replay(args: argparse.Namespace) -> int:
    settings = get_settings()
    traffic = {"old": load_episodes(args.old_traffic), "new": load_episodes(args.new_traffic)}
    prompts = {
        "old": PromptSet.from_json(args.old_prompts) if args.old_prompts else DEFAULT_PROMPTS,
        "new": PromptSet.from_json(args.new_prompts),
    }
    if prompts["old"].fingerprint == prompts["new"].fingerprint:
        print("note: old and new prompts are identical, so this run is an A/A check.")
    runs = 2 * sum(len(episodes) for episodes in traffic.values())
    print(f"Running {runs} swarm runs (4 cells) with LLM_PROVIDER={settings.llm_provider} ...")

    results = await run_grid(
        traffic, prompts, settings, on_miss=args.on_miss, concurrency=args.concurrency
    )
    inputs = {e.id: e.input for episodes in traffic.values() for e in episodes}
    counts = {key: len(episodes) for key, episodes in traffic.items()}
    report = build_report(results, _manifest(settings, args, prompts, counts), inputs, args.alpha)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "replay_2x2.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.out_dir / "replay_2x2.md").write_text(to_markdown(report), encoding="utf-8")

    cells = report["cells"]
    print(f"\n{'':14}{'old traffic':>14}{'new traffic':>14}")
    for row, (left, right) in (("old prompt", ("A", "B")), ("new prompt", ("C", "D"))):
        print(f"{row:14}{cells[left]['rate']:>14.0%}{cells[right]['rate']:>14.0%}")
    print()
    for code in report["findings"]:
        print(f"[{code}] {FINDING_TEXT[code]}")
    for warning in report["warnings"]:
        print(f"warning: {warning}")
    print(f"\nreport: {args.out_dir / 'replay_2x2.md'}")

    if args.gate and BLOCKING_FINDINGS & set(report["findings"]):
        print("gate: FAILED (the new prompt is significantly worse on held-fixed traffic)")
        return 1
    return 0


async def _record(args: argparse.Namespace) -> int:
    settings = get_settings()
    prompts = PromptSet.from_json(args.prompts) if args.prompts else DEFAULT_PROMPTS
    requests = []
    for number, row in enumerate(read_jsonl(args.inputs), 1):
        if not isinstance(row.get("input"), str):
            raise ValueError(f"{args.inputs}: row {number} needs a string field 'input'")
        expected = row.get("expected") or {}
        requests.append(
            (
                str(row.get("id", f"{args.inputs.stem}-{number:03d}")),
                row["input"],
                Expected(
                    tuple(expected.get("contains", ())), tuple(expected.get("not_contains", ()))
                ),
                str(row.get("note", "")),
            )
        )
    episodes = await record(requests, settings, prompts=prompts)
    save_episodes(args.out, episodes)
    unlabelled = sum(not e.expected.defined for e in episodes)
    print(f"Recorded {len(episodes)} episodes to {args.out}.")
    if unlabelled:
        print(f"{unlabelled} episode(s) have no 'expected': add it before using them in a replay.")
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m swarmops.evals",
        description="Record traffic and run the replay 2x2 eval (prompt x traffic).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    rec = sub.add_parser("record", help="Run inputs through the live swarm and save episodes.")
    rec.add_argument("inputs", type=Path, help='JSONL rows: {"input": ..., "expected": {...}}')
    rec.add_argument("--out", type=Path, required=True, help="Episode file to write (JSONL).")
    rec.add_argument("--prompts", type=Path, help="Prompt variant JSON (default: built-in v1).")

    rep = sub.add_parser("replay", help="Run {old,new} prompts x {old,new} recorded traffic.")
    rep.add_argument("--old-traffic", type=Path, required=True)
    rep.add_argument("--new-traffic", type=Path, required=True)
    rep.add_argument("--old-prompts", type=Path, help="Prompt variant JSON (default: built-in v1).")
    rep.add_argument("--new-prompts", type=Path, required=True)
    rep.add_argument(
        "--on-miss",
        choices=("live", "error"),
        default="live",
        help="A tool call the recording never saw: call the live tool, or return an error.",
    )
    rep.add_argument("--alpha", type=float, default=0.05, help="Significance level per comparison.")
    rep.add_argument("--concurrency", type=int, default=4, help="Episodes run at the same time.")
    rep.add_argument("--out-dir", type=Path, default=Path("evals/reports"))
    rep.add_argument(
        "--gate",
        action="store_true",
        help="Exit 1 when the new prompt is significantly worse on held-fixed traffic.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    command = _replay if args.command == "replay" else _record
    try:
        return asyncio.run(command(args))
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
