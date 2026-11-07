from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from swarmops.config import Settings
from swarmops.evals import __main__ as cli
from swarmops.evals.episodes import Episode, Expected, ToolReturn, load_episodes
from swarmops.evals.replay import ReplayToolbox
from swarmops.evals.report import Effect, build_report, diagnose
from swarmops.evals.runner import record, run_grid, run_replayed
from swarmops.evals.stats import mcnemar_exact, newcombe_interval, wilson_interval
from swarmops.llm.fake import ScriptedSwarmModel
from swarmops.messages import message_text
from swarmops.roles import REVIEWER, AgentName
from swarmops.swarm.prompts import DEFAULT_PROMPTS, PromptSet
from swarmops.swarm.tools import WORK_TOOLS

EVALS = Path(__file__).resolve().parents[1] / "evals"
OLD_TRAFFIC = EVALS / "data" / "old_traffic.jsonl"
NEW_TRAFFIC = EVALS / "data" / "new_traffic.jsonl"
CAUTIOUS = EVALS / "prompts" / "v2_cautious.json"


def _traffic() -> dict[str, list[Episode]]:
    return {"old": load_episodes(OLD_TRAFFIC), "new": load_episodes(NEW_TRAFFIC)}


# --- statistics ------------------------------------------------------------------------------


def test_wilson_interval_near_the_edges() -> None:
    low, high = wilson_interval(16, 16)
    assert high == 1.0 and low == pytest.approx(0.806, abs=1e-3)
    assert wilson_interval(0, 10)[0] == 0.0


def test_mcnemar_only_counts_flipped_episodes() -> None:
    assert mcnemar_exact(0, 0) == 1.0
    assert mcnemar_exact(3, 3) == 1.0
    assert mcnemar_exact(8, 0) == pytest.approx(2 / 2**8)


def test_newcombe_interval_excludes_zero_for_a_clear_drop() -> None:
    low, high = newcombe_interval(10, 16, 16, 16)
    assert low < high < 0


# --- prompts ---------------------------------------------------------------------------------


def test_prompt_variant_keeps_default_roles(tmp_path: Path) -> None:
    variant = PromptSet.from_json(CAUTIOUS)
    assert variant.planner == DEFAULT_PROMPTS.planner
    assert variant.reviewer != DEFAULT_PROMPTS.reviewer
    assert variant.fingerprint != DEFAULT_PROMPTS.fingerprint
    # The fingerprint depends on the text only, not on the version label.
    assert PromptSet(version="renamed").fingerprint == DEFAULT_PROMPTS.fingerprint

    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"reviwer": "typo"}))
    with pytest.raises(ValueError, match="unknown keys"):
        PromptSet.from_json(bad)


# --- replay ----------------------------------------------------------------------------------


def test_replayed_tool_answers_from_the_recording() -> None:
    # A planted wrong value proves the answer comes from the recording, not the live tool.
    toolbox = ReplayToolbox([ToolReturn("calculator", {"expression": "2 + 2"}, "5")], WORK_TOOLS)
    calculator = {t.name: t for t in toolbox.tools()}["calculator"]
    assert calculator.invoke({"expression": "2 + 2"}) == "5"
    assert calculator.invoke({"expression": "2 + 2"}) == "5"  # repeated call reuses
    assert (toolbox.stats.hits, toolbox.stats.reused, toolbox.stats.misses) == (1, 1, [])


def test_default_arguments_match_the_recording() -> None:
    toolbox = ReplayToolbox([ToolReturn("get_current_time", {}, "frozen")], WORK_TOOLS)
    clock = {t.name: t for t in toolbox.tools()}["get_current_time"]
    assert clock.invoke({"timezone": "UTC"}) == "frozen"


@pytest.mark.parametrize(("on_miss", "answer_part"), [("live", "60"), ("error", "replay miss")])
async def test_calls_outside_the_recording_are_counted(
    settings: Settings, on_miss: str, answer_part: str
) -> None:
    episode = Episode("miss", "What is (12 + 3) * 4?", expected=Expected(contains=("60",)))
    result = await run_replayed(episode, settings, DEFAULT_PROMPTS, on_miss=on_miss)
    assert result.replay_misses == 1
    assert answer_part in result.answer


async def test_record_then_replay_round_trip(settings: Settings) -> None:
    [episode] = await record([("r1", "What is 6 * 7?", Expected(("42",)), "")], settings)
    assert episode.tool_returns == (ToolReturn("calculator", {"expression": "6 * 7"}, "42"),)

    result = await run_replayed(episode, settings, DEFAULT_PROMPTS, on_miss="error")
    assert result.success and result.replay_hits == 1 and result.replay_misses == 0


def test_episode_files_are_validated(tmp_path: Path) -> None:
    path = tmp_path / "dup.jsonl"
    row = json.dumps({"id": "x", "input": "hi"})
    path.write_text(f"{row}\n{row}\n")
    with pytest.raises(ValueError, match="duplicate episode ids"):
        load_episodes(path)
    path.write_text("{not json}\n")
    with pytest.raises(ValueError, match=r"dup\.jsonl:1"):
        load_episodes(path)


# --- the 2x2 grid ----------------------------------------------------------------------------


async def test_fixtures_point_at_the_environment(settings: Settings) -> None:
    prompts = {"old": DEFAULT_PROMPTS, "new": PromptSet.from_json(CAUTIOUS)}
    results = await run_grid(_traffic(), prompts, settings, on_miss="error")
    report = build_report(results, {"llm_provider": "fake"}, {}, alpha=0.05)

    assert report["findings"] == ["environment"]
    assert all(cell["episodes_with_misses"] == 0 for cell in report["cells"].values())
    assert report["cells"]["A"]["rate"] == 1.0 and report["cells"]["B"]["rate"] == 10 / 16
    assert {item["id"] for item in report["review_queue"]} == {f"new-{i}" for i in range(11, 17)}


class CautiousScriptedModel(ScriptedSwarmModel):
    """Test double whose reviewer obeys an "ask a human" line in its system prompt,
    so the prompt axis has a real effect without calling an LLM."""

    def _generate(self, messages: list[BaseMessage], *args: Any, **kwargs: Any) -> ChatResult:
        result = super()._generate(messages, *args, **kwargs)
        system = " ".join(message_text(m) for m in messages if isinstance(m, SystemMessage))
        reply = result.generations[0].message
        if self.role == REVIEWER and "asking a human" in system and not reply.tool_calls:
            refusal = AIMessage(content="I could not complete the task. Please ask a human.")
            return ChatResult(generations=[ChatGeneration(message=refusal)])
        return result


def _cautious_model(role: AgentName, settings: Settings) -> ScriptedSwarmModel:
    return CautiousScriptedModel(role=role)


async def test_a_prompt_regression_is_attributed_to_the_prompt(settings: Settings) -> None:
    prompts = {"old": DEFAULT_PROMPTS, "new": PromptSet.from_json(CAUTIOUS)}
    results = await run_grid(_traffic(), prompts, settings, model_factory=_cautious_model)
    report = build_report(results, {"llm_provider": "test"}, {}, alpha=0.05)

    assert report["findings"] == ["prompt", "environment"]
    assert report["effects"]["C-A"]["verdict"] == "worse"
    assert report["cells"]["C"]["rate"] == 0.0


def _effect(name: str, verdict: str) -> Effect:
    return Effect(name, "", 0.0, None, None, None, "", verdict)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("verdicts", "findings"),
    [
        ({"C-A": "none", "D-B": "worse", "B-A": "none"}, ["interaction"]),
        ({"C-A": "none", "D-B": "none", "B-A": "none"}, ["not_reproduced"]),
        ({"C-A": "worse", "D-B": "worse", "B-A": "worse"}, ["prompt", "environment"]),
    ],
)
def test_diagnosis_patterns(verdicts: dict[str, str], findings: list[str]) -> None:
    effects = {name: _effect(name, v) for name, v in {**verdicts, "D-C": "none"}.items()}
    assert diagnose(effects) == findings


def test_cli_writes_reports_and_gate_passes_for_environment_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    offline = Settings(_env_file=None, llm_provider="fake", database_url=None)
    monkeypatch.setattr(cli, "get_settings", lambda: offline)
    args = ["replay", "--old-traffic", str(OLD_TRAFFIC), "--new-traffic", str(NEW_TRAFFIC)]
    args += ["--new-prompts", str(CAUTIOUS), "--out-dir", str(tmp_path), "--gate"]

    assert cli.main(args) == 0
    report = json.loads((tmp_path / "replay_2x2.json").read_text(encoding="utf-8"))
    assert report["manifest"]["prompts"]["new"]["version"] == "v2-cautious"
    assert "flat by construction" in report["warnings"][0]
    assert (tmp_path / "replay_2x2.md").read_text(encoding="utf-8").startswith("# Replay 2x2")
