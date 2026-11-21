import hashlib
import json
import re
from dataclasses import asdict
from pathlib import Path

from scripts.memory.run_evolution import memory_hash
from src.data.manifest import load_manifest
from src.evolution.controller import EvolutionController
from src.evolution.review_consumer import write_immutable
from src.evolution.version_store import (
    active_version,
    load_memory_version,
    read_json,
)
from src.memory.updater import MemoryItem


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run_retrieval_precheck(
    *,
    job_id,
    parent_run_dir,
    staging_root,
    validation_manifest_path,
    expected_manifest_sha256,
    excluded_game_ids,
    top_k=3,
):
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", job_id):
        raise ValueError("Invalid job ID")

    if type(top_k) is not int or top_k < 1:
        raise ValueError("Invalid top_k")

    parent_root = Path(parent_run_dir)
    workspace = Path(staging_root) / job_id
    manifest_path = Path(validation_manifest_path)

    if workspace.resolve().is_relative_to(parent_root.resolve()):
        raise ValueError("Candidate workspace overlaps active run")

    if digest(manifest_path) != expected_manifest_sha256:
        raise RuntimeError("Validation manifest changed")

    receipt = read_json(workspace / "stage_receipt.json")

    if (
        receipt.get("schema") != "online_memory_version_stage_v1"
        or receipt.get("status") != "PENDING_VALIDATION"
        or receipt.get("job_id") != job_id
        or receipt.get("active_memory_changed") is not False
    ):
        raise RuntimeError("Invalid candidate version receipt")

    parent_version = active_version(parent_root)

    if receipt["parent_version"] != parent_version:
        raise RuntimeError("Active parent version changed")

    candidate_version = receipt["candidate_version"]

    if not re.fullmatch(r"v\d{3,}", candidate_version):
        raise RuntimeError("Invalid candidate version")

    snapshot_path = (
        workspace / "versions" / f"{candidate_version}.json"
    )

    if digest(snapshot_path) != receipt["candidate_snapshot_sha256"]:
        raise RuntimeError("Candidate snapshot changed")

    _, parent_info = load_memory_version(
        parent_root, parent_version
    )

    _, candidate_info = load_memory_version(
        workspace, candidate_version
    )

    if (
        parent_info.memory_hash != receipt["parent_memory_sha256"]
        or candidate_info.memory_hash
        != receipt["candidate_memory_sha256"]
        or candidate_info.parent != parent_version
    ):
        raise RuntimeError("Candidate memory provenance mismatch")

    parent_snapshot = read_json(
        parent_root / "versions" / f"{parent_version}.json"
    )
    candidate_snapshot = read_json(snapshot_path)

    parent_memory = [
        MemoryItem(**item)
        for item in parent_snapshot["memory"]
    ]
    candidate_memory = [
        MemoryItem(**item)
        for item in candidate_snapshot["memory"]
    ]

    if (
        memory_hash(parent_memory) != parent_info.memory_hash
        or memory_hash(candidate_memory)
        != candidate_info.memory_hash
    ):
        raise RuntimeError("Memory content hash mismatch")

    games = load_manifest(manifest_path)

    if len(games) < 2:
        raise ValueError("Insufficient validation games")

    excluded = set(excluded_game_ids)
    excluded.add(receipt["learning_game_id"])

    ids = [game.id for game in games]

    if (
        len(ids) != len(set(ids))
        or any(game.split != "train" for game in games)
        or any(game.id in excluded for game in games)
    ):
        raise ValueError("Invalid or overlapping validation games")

    game_hashes = {}

    for game in games:
        if not game.game_file.is_file():
            raise FileNotFoundError(game.game_file)
        game_hashes[game.id] = digest(game.game_file)

    # Existing implementation compares actual Top-K retrieved lessons
    # using task text read from ALFWorld reset().
    controller = EvolutionController.__new__(EvolutionController)
    controller.validation_games = games
    controller.top_k = top_k

    changed = controller.retrieval_changes(
        parent_memory,
        candidate_memory,
    )

    if not 0 <= changed <= len(games):
        raise RuntimeError("Invalid retrieval comparison count")

    if digest(manifest_path) != expected_manifest_sha256:
        raise RuntimeError("Validation manifest changed during check")

    for game in games:
        if digest(game.game_file) != game_hashes[game.id]:
            raise RuntimeError("Validation game changed during check")

    result = {
        "schema": "online_memory_retrieval_precheck_v1",
        "job_id": job_id,
        "parent_version": parent_version,
        "candidate_version": candidate_version,
        "parent_memory_sha256": parent_info.memory_hash,
        "candidate_memory_sha256": candidate_info.memory_hash,
        "candidate_snapshot_sha256": digest(snapshot_path),
        "validation_manifest_sha256": expected_manifest_sha256,
        "validation_game_ids": ids,
        "validation_game_hashes": game_hashes,
        "excluded_game_ids_sha256": hashlib.sha256(
            json.dumps(sorted(excluded)).encode("utf-8")
        ).hexdigest(),
        "top_k": top_k,
        "total_games": len(games),
        "changed_retrieval_games": changed,
        "status": (
            "READY_FOR_PAIRED_EVAL"
            if changed > 0
            else "SKIP_UNCHANGED_TOP3"
        ),
        "active_memory_changed": False,
    }

    write_immutable(
        workspace / "retrieval_precheck.json",
        result,
    )

    return result
