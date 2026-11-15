# Feedback-Driven Skill Learning — Final Development Report

## Research question

Can successful but inefficient ALFWorld trajectories be
used to extract reusable conditional skills without
updating LLM parameters?

## Workflow

Successful trajectory analysis -> exploratory counterfactual
replay -> LLM candidate extraction -> offline trigger audit ->
online ReAct A/B/C evaluation -> frozen promotion gate ->
feedback-driven revision or abstention.

## Candidate

- ID: `c001`
- Name: `stalled_search_countertop_redirect`
- Trigger: 3 unproductive inspected locations, target not found,
  and an unvisited admissible countertop.
- Hint: supply optional search advice to the LLM.
- Execute: directly substitute one search action.
- Fixed memory: 32 lessons, BM25 Top-3, State Tracking enabled.
- LLM: `qwen3.8-chat`; max 50 decisions.
- No model parameter updates.

## Online development results

12 previously unused ALFWorld Train tasks,
4 clean / 4 cool / 4 heat.

| Metric | Baseline | Hint | Execute |
|---|---:|---:|---:|
| Successful tasks | 12/12 | 12/12 | 12/12 |
| Total environment actions | 167 | 158 | 178 |
| Mean environment actions | 13.92 | 13.17 | 14.83 |
| Total agent-level LLM calls | 167 | 158 | 175 |
| Actual skill triggers | 0 | 5 | 3 |

Hint saves 0.75 environment actions per task on average.
Execute uses 0.92 more environment actions per task on average.

The predeclared development gate required no success
regressions and at least 1.0 mean environment-action saving
when successes tied. Both interventions failed the gate.

Hint effects among actually triggered tasks:
- Dev #02: -18 environment actions.
- Dev #07: +7.
- Dev #08: +2.
- Dev #09: 0.
- Dev #11: 0.

The net 9-action saving is concentrated in Dev #02.
Independent model generations can differ even when
skills do not trigger. Therefore no-trigger changes
are not attributed to the skill.

## Exploratory counterfactual evidence

10 changed-decision historical replays:
original ReAct 9/10 successes, intervention 8/10.
The intervention reduced total environment actions but
caused one success-to-failure regression. These historical
examples contributed to skill development and are not
independent generalization evidence.

## Promotion and feedback revision

- c001: `REJECTED_ON_DEV`.
- Active skill: none.
- Feedback-driven proposal: `ABSTAIN`.
- No c002 candidate was created.
- No additional threshold tuning or model calls were
  performed after abstention.

The LLM's explanation of search-history counts was not
independently verified by a dedicated counting audit.
Promotion decisions are based on saved episode outcomes.

## Evaluation boundaries

The 12 development tasks are ALFWorld Train games,
not ALFWorld Valid Unseen.
Thirty additional frozen Train games spanning six task
types remain completely unevaluated.
The skill was not promoted, so the final holdout was not run.

The counts of agent-level LLM calls exclude service-side
retries, token billing, candidate extraction and
counterfactual-analysis costs.

## Conclusion

The engineering workflow for feedback-driven skill
proposal, runtime use, environment evaluation, rejection,
and abstention has been demonstrated.

No reusable search skill passed the predefined
development promotion gate. These results do not prove
stable improvement from autonomous skill learning.

## Reproducibility

- Protocol SHA256: `8d384c8d302daa489f72ea9367859f51cb904aab9b82da9a0c75ae2ae9e13844`
- Promotion decision SHA256: `ec302b22bdf4964967873b450dca7a233f865ad29ce611be04e08da5d25081a9`
- Revision record SHA256: `a9ef43d4402d2e4f0992cf5fa999ffda9a2e560df805a45e8c2514fc13d4d662`
- Candidate registry: `outputs/skill_learning_v1/skills/registry/`
- Dev episode data: `outputs/skill_learning_v1/online_abc_v1/runs/`
- Evaluator: `scripts/evaluation/evaluate_online_skill.py`

The outputs directory is locally retained and excluded
from Git by default; the repository alone does not include
the complete experimental evidence.
