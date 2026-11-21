import hashlib
import json
import re
from dataclasses import asdict

from scripts.memory.run_evolution import apply_proposal, memory_hash
from src.evolution.controller import gate
from src.memory.updater import LessonCandidate, UpdateProposal


REQUIRED_EVIDENCE = {
    "baseline:critical",
    "counterfactual:replacement",
}


def fingerprint(data):
    payload = json.dumps(
        data,
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def stage_memory_candidate(job, candidate, proposal, parent_memory):
    if (
        job.get("route") != "MEMORY_REVIEW"
        or job.get("source_scope") != "ONLINE_TRAIN_DEVELOPMENT"
        or not job.get("game_id", "").startswith("train:")
    ):
        raise ValueError("Memory candidate must come from Train failure")

    if not isinstance(candidate, LessonCandidate):
        raise TypeError("Invalid LessonCandidate")

    if not isinstance(proposal, UpdateProposal):
        raise TypeError("Invalid UpdateProposal")

    if candidate.decision != "CANDIDATE":
        raise ValueError("Cannot stage ABSTAIN")

    if not REQUIRED_EVIDENCE <= set(candidate.evidence_refs):
        raise ValueError("Missing counterfactual evidence references")

    if not set(candidate.evidence_refs) <= set(candidate.evidence):
        raise ValueError("Candidate references unknown evidence")

    if proposal.operation not in {"ADD", "EDIT", "REMOVE", "KEEP"}:
        raise ValueError("Invalid update operation")

    if proposal.memory_snapshot_sha256 != memory_hash(parent_memory):
        raise ValueError("Stale parent memory")

    if proposal.operation != "KEEP":
        if not REQUIRED_EVIDENCE <= set(proposal.evidence_refs):
            raise ValueError("Update lacks required evidence")

    if not set(proposal.evidence_refs) <= set(candidate.evidence_refs):
        raise ValueError("Update references unknown evidence")

    if proposal.operation == "ADD":
        if (
            proposal.target_id is not None
            or proposal.new_lesson != candidate.lesson
        ):
            raise ValueError("Invalid ADD proposal")

    event_id = job["job_id"]

    if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", event_id):
        raise ValueError("Invalid event ID")

    updated = list(parent_memory)
    before = memory_hash(parent_memory)

    apply_proposal(
        updated,
        candidate,
        proposal,
        event_id,
    )

    after = memory_hash(updated)
    changed = after != before

    return {
        "schema": "online_memory_candidate_stage_v1",
        "status": (
            "PENDING_VALIDATION" if changed else "NO_CHANGE"
        ),
        "job_id": event_id,
        "learning_game_id": job["game_id"],
        "trajectory_sha256": job["trajectory_sha256"],
        "event_sha256": job["event_sha256"],
        "operation": proposal.operation,
        "candidate_sha256": fingerprint(asdict(candidate)),
        "proposal_sha256": fingerprint(asdict(proposal)),
        "parent_memory_sha256": before,
        "candidate_memory_sha256": after,
        "candidate_memory": [
            asdict(item) for item in updated
        ],
        "active_memory_changed": False,
    }


def assess_memory_candidate(
    staged,
    baseline_results,
    candidate_results,
    *,
    validation_game_ids,
    validation_manifest_sha256,
):
    if staged["status"] != "PENDING_VALIDATION":
        raise ValueError("No changed candidate to validate")

    if not re.fullmatch(
        r"[0-9a-f]{64}",
        validation_manifest_sha256,
    ):
        raise ValueError("Invalid validation manifest hash")

    ids = list(validation_game_ids)

    if len(ids) < 2 or len(ids) != len(set(ids)):
        raise ValueError("Invalid frozen validation games")

    if staged["learning_game_id"] in ids:
        raise ValueError("Learning and validation overlap")

    if not all(
        isinstance(game_id, str)
        and game_id.startswith("train:")
        for game_id in ids
    ):
        raise ValueError("Validation must use frozen Train games")

    def validate_results(records):
        if [record.get("game_id") for record in records] != ids:
            raise ValueError("Paired evaluation IDs mismatch")

        for record in records:
            if (
                type(record.get("success")) is not bool
                or type(record.get("decision_steps")) is not int
                or record["decision_steps"] <= 0
            ):
                raise ValueError("Invalid evaluation result")

    validate_results(baseline_results)
    validate_results(candidate_results)

    result = gate(
        baseline_results,
        candidate_results,
    )

    return {
        "schema": "online_memory_gate_result_v1",
        "status": (
            "ELIGIBLE_FOR_PROMOTION"
            if result["accept"]
            else "REJECTED"
        ),
        "job_id": staged["job_id"],
        "parent_memory_sha256": staged["parent_memory_sha256"],
        "candidate_memory_sha256": staged["candidate_memory_sha256"],
        "validation_manifest_sha256": validation_manifest_sha256,
        "baseline_results_sha256": fingerprint(baseline_results),
        "candidate_results_sha256": fingerprint(candidate_results),
        "validation_games": ids,
        "result": result,
        "active_memory_changed": False,
    }
