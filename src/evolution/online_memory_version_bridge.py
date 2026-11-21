import hashlib
import re
from pathlib import Path

from scripts.memory.run_evolution import memory_hash
from src.evolution.controller import EvolutionController
from src.evolution.review_consumer import write_immutable
from src.evolution.version_store import (
    active_version,
    load_memory_version,
    read_json,
)
from src.memory.updater import MemoryItem


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stage_online_memory_version(
    job,
    staged,
    *,
    parent_run_dir,
    staging_root,
):
    if (
        job.get("route") != "MEMORY_REVIEW"
        or job.get("source_scope") != "ONLINE_TRAIN_DEVELOPMENT"
        or not job.get("game_id", "").startswith("train:")
    ):
        raise ValueError("Unauthorized memory version source")

    job_id = job["job_id"]

    if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", job_id):
        raise ValueError("Invalid job ID")

    if (
        staged.get("schema")
        != "online_memory_candidate_stage_v1"
        or staged.get("status") != "PENDING_VALIDATION"
        or staged.get("active_memory_changed") is not False
        or staged.get("job_id") != job_id
        or staged.get("learning_game_id") != job["game_id"]
        or staged.get("trajectory_sha256")
        != job["trajectory_sha256"]
        or staged.get("event_sha256") != job["event_sha256"]
    ):
        raise ValueError("Invalid staged candidate provenance")

    parent_root = Path(parent_run_dir)
    workspace = Path(staging_root) / job_id

    if workspace.resolve().is_relative_to(parent_root.resolve()):
        raise ValueError(
            "Candidate workspace cannot be inside active memory run"
        )

    parent_version = active_version(parent_root)

    _, parent_info = load_memory_version(
        parent_root,
        parent_version,
    )

    parent_snapshot = read_json(
        parent_root / "versions" / f"{parent_version}.json"
    )

    parent_memory = [
        MemoryItem(**item)
        for item in parent_snapshot["memory"]
    ]

    if (
        memory_hash(parent_memory) != parent_info.memory_hash
        or staged["parent_memory_sha256"]
        != parent_info.memory_hash
    ):
        raise ValueError("Active parent memory hash mismatch")

    raw = staged.get("candidate_memory")

    if not isinstance(raw, list) or not raw:
        raise ValueError("Invalid candidate memory")

    expected_fields = {"id", "lesson", "source_task"}

    if any(
        not isinstance(item, dict)
        or set(item) != expected_fields
        or any(
            not isinstance(value, str) or not value.strip()
            for value in item.values()
        )
        for item in raw
    ):
        raise ValueError("Invalid candidate memory item")

    candidate_memory = [
        MemoryItem(**item) for item in raw
    ]

    ids = [item.id for item in candidate_memory]

    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate candidate memory IDs")

    candidate_hash = memory_hash(candidate_memory)

    if (
        candidate_hash != staged["candidate_memory_sha256"]
        or candidate_hash == parent_info.memory_hash
    ):
        raise ValueError("Candidate memory hash mismatch")

    candidate_version = (
        f"v{int(parent_version[1:]) + 1:03d}"
    )

    # Reuse the existing immutable snapshot writer.
    # Do not initialize or run the continual controller.
    writer = EvolutionController.__new__(EvolutionController)
    writer.home = workspace

    writer.save_version(
        candidate_version,
        parent_version,
        parent_info.round_id + 1,
        candidate_memory,
    )

    snapshot_path = (
        workspace / "versions"
        / f"{candidate_version}.json"
    )

    _, candidate_info = load_memory_version(
        workspace,
        candidate_version,
    )

    if (
        candidate_info.parent != parent_version
        or candidate_info.memory_hash != candidate_hash
    ):
        raise RuntimeError("Candidate snapshot verification failed")

    receipt = {
        "schema": "online_memory_version_stage_v1",
        "status": "PENDING_VALIDATION",
        "job_id": job_id,
        "learning_game_id": job["game_id"],
        "parent_version": parent_version,
        "parent_memory_sha256": parent_info.memory_hash,
        "candidate_version": candidate_version,
        "candidate_memory_sha256": candidate_hash,
        "candidate_snapshot_sha256": file_hash(snapshot_path),
        "staged_candidate_sha256": hashlib.sha256(
            __import__("json").dumps(
                staged,
                ensure_ascii=False,
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest(),
        "active_memory_changed": False,
    }

    write_immutable(
        workspace / "stage_receipt.json",
        receipt,
    )

    return receipt
