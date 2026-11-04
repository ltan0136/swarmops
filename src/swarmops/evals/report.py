"""Turn the four cells into effects, a diagnosis and a report.

Four comparisons, each changing exactly one thing:

    C vs A   prompt changed, old traffic held fixed     (paired: same episodes)
    D vs B   prompt changed, new traffic held fixed     (paired: same episodes)
    B vs A   traffic changed, old prompt held fixed     (independent: different episodes)
    D vs C   traffic changed, new prompt held fixed     (independent: different episodes)
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any, Literal

from swarmops.evals.runner import EpisodeResult
from swarmops.evals.stats import mcnemar_exact, newcombe_interval, wilson_interval

Verdict = Literal["worse", "better", "none", "n/a"]
MISS_WARNING_RATE = 0.10
SMALL_SAMPLE = 30


@dataclass(frozen=True)
class CellSummary:
    cell: str
    scored: int
    unscored: int
    successes: int
    rate: float
    low: float
    high: float
    mean_steps: float
    mean_handoffs: float
    episodes_with_misses: int

    @property
    def miss_rate(self) -> float:
        total = self.scored + self.unscored
        return self.episodes_with_misses / total if total else 0.0


@dataclass(frozen=True)
class Effect:
    name: str  # e.g. "C-A"
    changed: str  # what differs between the two cells
    delta: float  # success-rate difference, later cell minus earlier cell
    low: float | None  # confidence interval (independent comparisons)
    high: float | None
    p_value: float | None  # exact McNemar (paired comparisons)
    detail: str
    verdict: Verdict


def summarise(cell: str, results: Sequence[EpisodeResult]) -> CellSummary:
    scored = [r for r in results if r.scored]
    successes = sum(r.success for r in scored)
    low, high = wilson_interval(successes, len(scored))
    count = len(results) or 1
    return CellSummary(
        cell=cell,
        scored=len(scored),
        unscored=len(results) - len(scored),
        successes=successes,
        rate=successes / len(scored) if scored else 0.0,
        low=low,
        high=high,
        mean_steps=sum(r.steps for r in results) / count,
        mean_handoffs=sum(r.handoffs for r in results) / count,
        episodes_with_misses=sum(r.replay_misses > 0 for r in results),
    )


def paired_effect(
    name: str,
    changed: str,
    before: Sequence[EpisodeResult],
    after: Sequence[EpisodeResult],
    alpha: float,
) -> Effect:
    """Same episodes, two variants: only the episodes whose outcome flipped are informative."""
    after_by_id = {r.episode_id: r for r in after}
    pairs = [(b, after_by_id[b.episode_id]) for b in before if b.scored]
    if not pairs:
        return Effect(name, changed, 0.0, None, None, None, "no scored episodes", "n/a")
    lost = sum(b.success and not a.success for b, a in pairs)  # right before, wrong after
    gained = sum(a.success and not b.success for b, a in pairs)
    p_value = mcnemar_exact(lost, gained)
    verdict: Verdict = "none"
    if p_value < alpha:
        verdict = "worse" if lost > gained else "better"
    return Effect(
        name=name,
        changed=changed,
        delta=(gained - lost) / len(pairs),
        low=None,
        high=None,
        p_value=p_value,
        detail=f"McNemar p={p_value:.3g} ({lost} lost, {gained} gained, n={len(pairs)})",
        verdict=verdict,
    )


def independent_effect(name: str, changed: str, before: CellSummary, after: CellSummary) -> Effect:
    """Different episodes: compare the two rates with a Newcombe interval."""
    if not before.scored or not after.scored:
        return Effect(name, changed, 0.0, None, None, None, "no scored episodes", "n/a")
    low, high = newcombe_interval(after.successes, after.scored, before.successes, before.scored)
    verdict: Verdict = "worse" if high < 0 else "better" if low > 0 else "none"
    return Effect(
        name=name,
        changed=changed,
        delta=after.rate - before.rate,
        low=low,
        high=high,
        p_value=None,
        detail=f"95% CI [{low:+.0%}, {high:+.0%}]",
        verdict=verdict,
    )


def compute_effects(
    results: dict[str, list[EpisodeResult]], summaries: dict[str, CellSummary], alpha: float
) -> dict[str, Effect]:
    return {
        "C-A": paired_effect("C-A", "prompt (old traffic)", results["A"], results["C"], alpha),
        "D-B": paired_effect("D-B", "prompt (new traffic)", results["B"], results["D"], alpha),
        "B-A": independent_effect("B-A", "traffic (old prompt)", summaries["A"], summaries["B"]),
        "D-C": independent_effect("D-C", "traffic (new prompt)", summaries["C"], summaries["D"]),
    }


def diagnose(effects: dict[str, Effect]) -> list[str]:
    """Read the pattern of the four effects. Returns finding codes, most important first.

    prompt            new prompt is worse with the traffic held fixed (C-A)
    interaction       new prompt is fine on old traffic but worse on new traffic (D-B only)
    environment       old prompt got worse when only the traffic changed (B-A)
    not_reproduced    nothing got significantly worse inside the replay
    """
    findings = []
    if effects["C-A"].verdict == "worse":
        findings.append("prompt")
    elif effects["D-B"].verdict == "worse":
        findings.append("interaction")
    if effects["B-A"].verdict == "worse":
        findings.append("environment")
    if not findings:
        findings.append("not_reproduced")
    return findings


FINDING_TEXT = {
    "prompt": "The new prompt is worse even with the traffic held fixed: the prompt itself "
    "is a cause. The offline eval set did not catch it.",
    "interaction": "The new prompt is fine on old traffic but worse on new traffic: it is fragile "
    "on the kind of input that appeared this period.",
    "environment": "With the prompt held fixed, the new traffic scores worse: the environment "
    "changed (user inputs, tool outputs or retrieved content).",
    "not_reproduced": "Nothing got significantly worse inside the replay. Either the difference "
    "is noise, or the cause is outside the recording (live model behind an alias, live tool "
    "behaviour, latency).",
}


def warnings(summaries: dict[str, CellSummary], effects: dict[str, Effect]) -> list[str]:
    notes = []
    for summary in summaries.values():
        if summary.miss_rate > MISS_WARNING_RATE:
            notes.append(
                f"Cell {summary.cell}: {summary.miss_rate:.0%} of episodes left the recording "
                "(replay misses). Conclusions involving this cell are only partly frozen."
            )
        if summary.unscored:
            notes.append(
                f"Cell {summary.cell}: {summary.unscored} episodes have no `expected` and were "
                "not scored. Ask a domain expert to label them."
            )
    smallest = min(s.scored for s in summaries.values())
    if smallest < SMALL_SAMPLE and any(e.verdict == "none" for e in effects.values()):
        notes.append(
            f"Smallest cell has {smallest} scored episodes. 'none' means 'not detected at this "
            "sample size', not 'no difference'."
        )
    return notes


def review_queue(
    results: dict[str, list[EpisodeResult]], inputs: dict[str, str], limit: int = 20
) -> list[dict[str, str]]:
    """Episodes a domain expert should look at: outcomes the prompt change flipped, then
    failures on new traffic under the new prompt."""
    queue: list[dict[str, str]] = []
    for before, after in (("A", "C"), ("B", "D")):
        after_by_id = {r.episode_id: r for r in results[after]}
        for b in results[before]:
            a = after_by_id[b.episode_id]
            if b.scored and b.success != a.success:
                change = "lost" if b.success else "gained"
                queue.append(_queue_item(a, inputs, f"{before}->{after} {change}"))
    seen = {item["id"] for item in queue}
    for r in results["D"]:
        if r.scored and not r.success and r.episode_id not in seen:
            queue.append(_queue_item(r, inputs, "D failed"))
    return queue[:limit]


def _queue_item(result: EpisodeResult, inputs: dict[str, str], why: str) -> dict[str, str]:
    return {
        "id": result.episode_id,
        "why": why,
        "input": inputs.get(result.episode_id, ""),
        "answer": result.answer or f"(no answer: {result.error})",
    }


def build_report(
    results: dict[str, list[EpisodeResult]],
    manifest: dict[str, Any],
    inputs: dict[str, str],
    alpha: float,
) -> dict[str, Any]:
    summaries = {cell: summarise(cell, cell_results) for cell, cell_results in results.items()}
    effects = compute_effects(results, summaries, alpha)
    notes = warnings(summaries, effects)
    if manifest.get("llm_provider") == "fake":
        notes.insert(
            0,
            "LLM_PROVIDER=fake: the scripted model ignores system prompts, so the prompt axis is "
            "flat by construction (an A/A check of the harness). Use a real model to measure "
            "prompt effects.",
        )
    return {
        "manifest": manifest,
        "cells": {cell: {**asdict(s), "miss_rate": s.miss_rate} for cell, s in summaries.items()},
        "effects": {name: asdict(effect) for name, effect in effects.items()},
        "findings": diagnose(effects),
        "warnings": notes,
        "review_queue": review_queue(results, inputs),
        "episodes": {
            cell: [asdict(r) for r in cell_results] for cell, cell_results in results.items()
        },
    }


def to_markdown(report: dict[str, Any]) -> str:
    m, cells, effects = report["manifest"], report["cells"], report["effects"]

    def cell_text(cell: str) -> str:
        c = cells[cell]
        return f"{cell}: {c['rate']:.0%} [{c['low']:.0%}, {c['high']:.0%}] n={c['scored']}"

    old_p, new_p = m["prompts"]["old"], m["prompts"]["new"]
    lines = [
        "# Replay 2x2 eval",
        "",
        "## Grid (success rate, 95% Wilson CI)",
        "",
        f"| | old traffic ({m['traffic']['old']['episodes']}) "
        f"| new traffic ({m['traffic']['new']['episodes']}) |",
        "|---|---|---|",
        f"| old prompt {old_p['version']} | {cell_text('A')} | {cell_text('B')} |",
        f"| new prompt {new_p['version']} | {cell_text('C')} | {cell_text('D')} |",
        "",
        "## Effects (one thing changed at a time)",
        "",
        "| comparison | what changed | change in success rate | evidence | verdict |",
        "|---|---|---|---|---|",
    ]
    for name, e in effects.items():
        lines.append(
            f"| {name} | {e['changed']} | {e['delta'] * 100:+.1f} pt | {e['detail']} "
            f"| **{e['verdict']}** |"
        )
    lines += ["", "## Diagnosis", ""]
    lines += [f"- **{code}**: {FINDING_TEXT[code]}" for code in report["findings"]]
    if report["warnings"]:
        lines += ["", "## Warnings", ""]
        lines += [f"- {w}" for w in report["warnings"]]
    lines += [
        "",
        "## Trajectories",
        "",
        "| cell | mean steps | mean handoffs | episodes with replay misses |",
        "|---|---|---|---|",
    ]
    for cell, c in cells.items():
        lines.append(
            f"| {cell} | {c['mean_steps']:.1f} | {c['mean_handoffs']:.1f} "
            f"| {c['episodes_with_misses']} ({c['miss_rate']:.0%}) |"
        )
    if report["review_queue"]:
        lines += ["", "## Review queue (for domain experts)", ""]
        lines += ["| id | why | input | answer |", "|---|---|---|---|"]
        for item in report["review_queue"]:
            answer = item["answer"].replace("\n", " ").replace("|", "\\|")[:120]
            text = item["input"].replace("|", "\\|")
            lines.append(f"| {item['id']} | {item['why']} | {text} | {answer} |")
    lines += ["", "## Manifest", "", "| key | value |", "|---|---|"]
    lines.append(f"| prompts.old | {old_p['version']} ({old_p['fingerprint']}) |")
    lines.append(f"| prompts.new | {new_p['version']} ({new_p['fingerprint']}) |")
    for key in ("old", "new"):
        t = m["traffic"][key]
        lines.append(f"| traffic.{key} | {t['path']} (sha256 {t['sha256']}) |")
    for key in (
        "swarmops_version",
        "llm_provider",
        "llm_model",
        "llm_temperature",
        "recursion_limit",
        "on_miss",
        "alpha",
        "created_at",
    ):
        lines.append(f"| {key} | {m[key]} |")
    return "\n".join(lines) + "\n"
