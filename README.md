# Agent Self-Evolution

**Training-Free LLM Agent Self-Evolution via Counterfactual Feedback and Memory Management**

基于 ALFWorld 的大语言模型智能体自进化研究项目。在不更新模型参数的情况下，研究如何利用环境反馈、反事实验证和长期记忆管理改善 Agent 决策。

## System Overview

```text
ALFWorld Task
    ↓
ReAct Agent + Memory Retrieval + State Tracking
    ↓
Trajectory & Environment Feedback
    ↓
Failure Analysis
    ↓
Counterfactual Replay & Verification
    ↓
Candidate Lesson Extraction
    ↓
Memory Update: ADD / EDIT / REMOVE / KEEP
    ↓
Candidate Memory Version
    ↓
Paired Validation & Version Gating
    ↓
Accept / Reject / Rollback
    ↓
Next Learning Round
```

## Main Features

- ReAct execution and replayable ALFWorld trajectories
- State tracking for clean, cool and heat operations
- Counterfactual verification using real environment feedback
- Evidence-based lesson extraction from successful counterfactual branches
- Long-term memory updates: ADD, EDIT, REMOVE and KEEP
- BM25 retrieval with task-type-aware priority for new lessons
- Immutable memory versions, validation gates, recovery and rollback
- Fixed, Append-only and Dynamic controlled experiments

## Experimental Setup

| Setting | Value |
|---|---|
| Environment | ALFWorld / TextWorld |
| LLM | qwen3.8-chat |
| Python | 3.11 |
| Initial Memory | 32 stratified lessons / 130 original lessons |
| Learning Tasks | 3 rounds × 20 tasks |
| Validation Tasks | 12 |
| Final Holdout | 30 |
| Max Steps | 50 |
| Retrieval Top-K | 3 |
| State Tracking | Enabled |
| Parameter Updates | None |

## Continuous Learning Results

| Method | Success | Success Rate | Avg. Steps | Final Version |
|---|---:|---:|---:|---|
| Fixed-32 | 57/60 | 95.00% | 15.15 | v000 |
| Append-32 | 57/60 | 95.00% | 14.77 | v000 |
| Dynamic-32 | 57/60 | 95.00% | 14.73 | v000 |
| Fixed-130 | 58/60 | 96.67% | 15.02 | v000 |

Append-32 generated a candidate memory version, but its validation improvement did not meet the predefined acceptance threshold. No method accepted a new memory version.

## Independent Holdout

| Metric | Fixed-32 | Fixed-130 |
|---|---:|---:|
| Success | 30/30 | 29/30 |
| Success Rate | 100.00% | 96.67% |
| Average Steps | 14.73 | 17.13 |

Since Fixed-32, Append-32 and Dynamic-32 ended with identical memories, a single Fixed-32 holdout evaluation represents their shared final policy. The 130-lesson baseline was evaluated separately on the same tasks.

These results suggest that increasing memory size does not necessarily improve performance. However, this small evaluation does not establish a general superiority of 32 lessons.

## Installation

Requires Python 3.11, uv, ALFWorld data, an initial experience store, and frozen experiment manifests.

```bash
uv sync
```

Configure an OpenAI-compatible API:

```bash
export OPENAI_API_KEY="your-api-key"
export OPENAI_BASE_URL="your-api-endpoint"
export OPENAI_MODEL="qwen3.8-chat"
```

## Usage

Run tests:

```bash
uv run python -m unittest discover -s tests -v
```

Initialize a frozen experiment:

```bash
uv run python -m scripts.run_continual_evolution \
  --stage plan \
  --protocol-root outputs/continual_32 \
  --run-name dynamic_32 \
  --mode dynamic
```

Execute the experiment:

```bash
uv run python -m scripts.run_continual_evolution \
  --stage run \
  --protocol-root outputs/continual_32 \
  --run-name dynamic_32 \
  --mode dynamic
```

Modes: `fixed`, `append`, `dynamic`. Use `--reference-memory` with fixed mode for the 130-lesson reference.

Frozen manifests and experimental trajectories are stored under `outputs/`, which is excluded from Git by default. Cloning the repository alone does not reproduce the saved experiments without these data.

## Research Conclusion

The project implements a complete training-free agent feedback and memory evolution workflow, including counterfactual verification and controlled version management.

Current experiments demonstrate engineering feasibility and local candidate-lesson utility, but do not establish that continual memory evolution outperforms fixed memory. No candidate memory version passed the frozen acceptance gate.

Historical experiments and detailed results are documented in `docs/experiment_results.md`.
