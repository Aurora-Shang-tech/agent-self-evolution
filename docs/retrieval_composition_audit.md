# Retrieval Composition Audit

The integrated Hybrid + State versus Utility + State experiment reported 11/11
tasks with changed Top-3 rankings, with no success-rate gain and a +2.73 mean
decision-step difference. Ranking changes do **not** necessarily mean different
lessons were injected.

This read-only audit separates:

- `identical`: same ordered Top-K IDs.
- `reordered_only`: identical Top-K ID set, different order.
- `membership_changed`: at least one memory ID enters or leaves Top-K.

It reads the frozen `experiment_plan.json` and the saved per-arm
`episode_XXX_meta.json` files, validates matching task/game/selected IDs,
and reports paired outcome changes by group. Missing arms are marked incomplete,
not counted as failures or successes.

Run without LLM or ALFWorld:

```bash
uv run python -m unittest tests.test_pipeline_retrieval_audit -v

uv run python -m scripts.analysis.audit_pipeline_retrieval \
  --directory outputs/memory_pipeline/paired_state_v1/qwen3.8-chat/v2_seed_20261112
```

Optionally save a report:

```bash
uv run python -m scripts.analysis.audit_pipeline_retrieval \
  --directory outputs/memory_pipeline/paired_state_v1/qwen3.8-chat/v2_seed_20261112 \
  --output outputs/analysis/retrieval_composition_v1.json
```

Interpretation:

- If most changes are `reordered_only`, inspect the impact of **prompt lesson
  ordering** before attributing effects to selecting a different lesson.
- If many changes are `membership_changed`, inspect the specific incoming and
  outgoing memory IDs and associated regressions.
- The audit cannot establish a causal per-lesson credit assignment: jointly
  retrieved lessons can interact.
- Because this evaluation selected tasks with a changed ranking, its result is
  **conditional** on that subset and is not a full-distribution success estimate.
- The tool does not infer token costs, compare model seeds, or rerun any task.
- Do not report a new numerical result until the local frozen artifacts are read.

Related open-source designs to study: [MemRL](https://github.com/MemTensor/MemRL)
(two-phase utility-aware retrieval), [Recuris](https://github.com/Gen-Verse/Recuris)
(paired held-out validation gates), and
[ExpeL](https://github.com/LeapLabTHU/ExpeL) (experience/insight retrieval).
