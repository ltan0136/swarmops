# ADR 0004: A replay-based 2x2 eval to separate prompt regressions from environment changes

- Status: proposed
- Date: 2026-11-04

## Context

A prompt change ships, its offline score goes up, and a week later live quality drops. Two
causes are now confounded: the change itself, and everything outside the config manifest that
moved in the same week (what users ask, what tools return, the clock). Comparing this week's live
numbers with last week's offline score mixes both.

`get_current_time` is the smallest example in this repo: its output changes every second, so a
live re-run can never reproduce last week's environment. Answers that depend on it cannot even
be scored against a fixed expectation unless the tool output is frozen.

## Decision

1. **Record.** An episode is one user request, every work-tool return it saw, and an
   expert-written `expected`. Handoffs are not recorded: they are the agents' decisions, not
   part of the environment. Episodes are JSON Lines under `evals/data/`.
2. **Replay.** Stand-in tools keep the live tool's name, description and argument schema and
   answer from the recording. Arguments are canonicalised through the schema, so `{}` and
   `{"timezone": "UTC"}` match when `UTC` is the default.
3. **2x2 grid.** {old, new prompt set} x {old, new traffic}, each episode in a fresh
   single-turn thread with no checkpointer:

   |            | old traffic | new traffic |
   |------------|-------------|-------------|
   | old prompt | A           | B           |
   | new prompt | C           | D           |

4. **Four comparisons, one variable each.** C-A and D-B change only the prompt and run on the
   same episodes, so they use the exact McNemar test on the episodes whose outcome flipped.
   B-A and D-C change only the traffic, on different episodes, so they use Newcombe's interval.
5. **Diagnosis.** `prompt` (C-A worse), `interaction` (only D-B worse), `environment`
   (B-A worse), otherwise `not_reproduced`.
6. **Seams, not forks.** `build_swarm` gains keyword arguments `prompts`, `work_tools` and
   `model_factory` whose defaults are today's production values. Prompts become a `PromptSet`
   with a content fingerprint.
7. **Gate.** `--gate` exits 1 on `prompt` or `interaction`. An environment-only change does not
   block a prompt release but is reported.
8. **Report.** Markdown and JSON with a manifest (prompt fingerprints, traffic file hashes,
   model, temperature, recursion limit, miss policy) and a review queue for domain experts.

## Alternatives considered

- **Offline score vs live metrics.** What failed in the first place: confounded.
- **Re-run both prompts live on this week's traffic.** Holds the prompt axis fixed but not the
  tool outputs, and cannot reach last week's environment at all.
- **Online A/B test.** The gold standard for the prompt axis, but it needs live traffic, takes
  days and exposes users to the worse variant. Replay is the cheap first step; A/B remains the
  confirmation.
- **Hand-written tool mocks.** Test the harness, not the environment the agents actually saw.

## How this can fail

- **Off-policy gap.** The new prompt can make tool calls the recording never saw. These are
  counted as misses, and a cell with more than 10 % missed episodes gets a warning.
  `--on-miss live` lets those calls hit the live tool (no longer frozen); `--on-miss error`
  biases the comparison against the new prompt. Neither is neutral.
- **Only tool outputs are frozen.** A model behind a provider alias, retrieved documents (none in
  v0.1), latency and multi-turn state are not.
- **The scripted model ignores prompts.** With `LLM_PROVIDER=fake` the prompt axis is flat by
  construction, which makes the run an A/A check of the harness. Measuring a real prompt effect
  needs a real model and costs 2 x (old + new episodes) swarm runs.
- **Small samples.** 16 episodes per side only detect large effects; `none` means "not
  detected", not "no difference". The four comparisons use alpha = 0.05 each, uncorrected.
- **Non-determinism.** Each episode runs once per cell. Real models need repeated runs (pass^k).
- **Crude scoring.** `expected` is substring matching, a stand-in for rubric or judge scoring.

## Consequences

- First piece of the v0.4 eval suite: `python -m swarmops.evals` and `make eval-replay`.
- The committed fixtures are synthetic: 16 old episodes, and 16 new ones with input drift
  (Chinese numerals, word problems) and a simulated time-service outage. They exercise the
  `environment` finding; tests inject a prompt-sensitive model to exercise `prompt`.
- Next: export real episodes from the Postgres checkpointer, repeated runs per cell, and
  rubric/judge scoring with expert-calibrated judges.
- Review in 2027-01: did replay diagnoses agree with what A/B tests or incident reviews found?
