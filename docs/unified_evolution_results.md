# Unified Agent Self-Evolution — Online Pilot

## Architecture

The unified system connects active capability loading,
ReAct execution, environment feedback routing,
persistent review events and separate Memory/Skill review paths.

Active configuration:
- Memory: v000, 32 lessons
- Active skill: none
- Retrieval: BM25 Top-3
- State Tracking: enabled
- Model: qwen3.8-chat
- Maximum decisions: 50

## Experiment

Six frozen ALFWorld Train development tasks,
one from each supported task type.

| Metric | Result |
|---|---:|
| Successful tasks | 5/6 (83.33%) |
| Archived successes | 3 |
| Memory Review jobs | 1 |
| Skill Review jobs | 2 |
| Memory promotions | 0 |
| Skill promotions | 0 |

### Memory Review: fresh_04

Task: look at bowl under the desklamp.

The original Agent failed after 50 decisions.
FailureAnalyzer selected step index 1 and proposed three
alternative actions. Two branches were tested using real
ALFWorld counterfactual replay:

- examine desk 1: failed, 48 continuation decisions.
- take bowl 1 from desk 1: failed, 48 continuation decisions.

The inventory alternative was not tested to limit cost.
Status: STOPPED_WITH_UNTESTED_CANDIDATE.
No verified lesson or memory update was produced.

### Skill Review

- fresh_01: heat cup and place in cabinet; 27 actions.
  First observed the cup on decision 22.
- fresh_05: place spraybottle on countertop; 23 actions.
  First observed the spraybottle on decision 20.

Both tasks exhibited substantial object-search overhead.
Neither supplied sufficient evidence for a verified
transferable skill.

Both ended as NO_CANDIDATE_IN_PILOT.

## Outcome and limitations

The pilot demonstrates working online execution,
feedback routing, persisted review evidence, cached
recovery and conservative no-update decisions.

Review processing used separately invoked commands;
this pilot does not demonstrate a fully autonomous
end-to-end learning and promotion process.

No new Memory or Skill was accepted.
The active policy remained unchanged.
No performance improvement from continual evolution
can be inferred from this pilot.

The six games are Train development examples, not
independent Valid Unseen benchmarks. The frozen
30-game Skill final holdout remains unevaluated.

## Evidence

- Protocol SHA256: `4191109b412f763943faa243d02eea7f757e39fe04c73334a0afebbe9e19103a`
- Review plan SHA256: `7f9cf8818e394277a889ed1d66d23cee11d0c4a14a42a283528b1e01b59761c0`
- Summary SHA256: `02f816c2b6b7e84a9c5b2c5362b44cef96e06872c40a3b7631b6789f8e148482`
- Local experiment directory: `outputs/unified_evolution_v1/online_pilot_v1`
- Episode and review evidence under `episodes/` and `reviews/`

The `outputs/` directory is excluded from Git by
default. Complete experiment reproduction requires
the corresponding local evidence and ALFWorld data.
