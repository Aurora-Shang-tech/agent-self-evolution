from dataclasses import asdict
from pathlib import Path

from scripts.memory.run_evolution import memory_hash
from src.core.trajectory import Trajectory
from src.evolution.online_memory_update_bridge import (
    OnlineMemoryUpdateBridge,
)
from src.evolution.version_store import read_json
from src.memory.updater import LessonCandidate


REQUIRED_REFS = {
    "baseline:critical",
    "counterfactual:replacement",
}


def load_verified_online_candidate(
    job,
    outcome_path,
    memory_handler,
):
    if (
        job.get("route") != "MEMORY_REVIEW"
        or job.get("source_scope")
        != "ONLINE_TRAIN_DEVELOPMENT"
        or not job.get("game_id", "").startswith("train:")
    ):
        raise ValueError("Unauthorized online memory source")

    outcome_path = Path(outcome_path)

    # The existing handler validates the saved analysis,
    # counterfactual branch, candidate hashes and outcome.
    memory_handler.validate_outcome(job, outcome_path)

    outcome = read_json(outcome_path)

    if outcome.get("schema") != "online_memory_outcome_v2":
        raise RuntimeError("Invalid online memory outcome schema")

    if outcome.get("status") != "CANDIDATE_PENDING_GATE":
        raise ValueError("Review has no candidate pending gate")

    candidate_path = (
        outcome_path.parent / "automated_candidate.json"
    )

    record = read_json(candidate_path)

    if record.get("schema") != "automated_lesson_candidate_v1":
        raise RuntimeError("Invalid saved candidate schema")

    raw = record.get("candidate")

    if not isinstance(raw, dict):
        raise RuntimeError("Saved candidate is missing")

    expected_fields = set(LessonCandidate.__dataclass_fields__)

    if set(raw) != expected_fields:
        raise RuntimeError("Candidate fields mismatch")

    candidate = LessonCandidate(
        **{
            **raw,
            "evidence_refs": tuple(raw["evidence_refs"]),
        }
    )

    if candidate.decision != "CANDIDATE":
        raise ValueError("ABSTAIN cannot enter update")

    if not REQUIRED_REFS <= set(candidate.evidence_refs):
        raise ValueError("Candidate lacks required evidence")

    trace = Trajectory.load_json(
        job["trajectory_path"]
    )

    if (
        trace.success
        or not 0 <= candidate.critical_step < len(trace.steps)
        or candidate.source_task != trace.task
    ):
        raise ValueError("Candidate baseline mismatch")

    critical = trace.steps[candidate.critical_step]

    if (
        candidate.original_action != critical.action
        or candidate.replacement_action
        not in critical.admissible_actions
        or candidate.replacement_action == critical.action
    ):
        raise ValueError("Candidate actions mismatch")

    evidence = candidate.evidence

    if (
        evidence["baseline:critical"]["action"]
        != candidate.original_action
        or evidence["counterfactual:replacement"]["action"]
        != candidate.replacement_action
    ):
        raise ValueError("Candidate evidence/action mismatch")

    return candidate


def stage_verified_online_update(
    job,
    outcome_path,
    memory_handler,
    llm,
    parent_memory,
    expected_parent_hash,
    *,
    top_k=8,
):
    if memory_hash(parent_memory) != expected_parent_hash:
        raise ValueError("Active parent memory hash mismatch")

    candidate = load_verified_online_candidate(
        job,
        outcome_path,
        memory_handler,
    )

    output_path = (
        Path(outcome_path).parent
        / "memory_update_proposal.json"
    )

    return OnlineMemoryUpdateBridge(
        llm,
        top_k=top_k,
    ).process(
        job,
        candidate,
        parent_memory,
        output_path,
    )
