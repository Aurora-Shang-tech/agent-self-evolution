import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from src.agents.react_agent import ReActAgent
from src.evolution.version_store import (
    load_active_memory,
    MemoryVersionInfo,
)
from src.memory.store import ExperienceStore


def read_json(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return data


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass(frozen=True)
class ActiveCapabilities:
    memory_store: ExperienceStore
    memory_info: MemoryVersionInfo
    skill_id: str | None
    skill_mode: str | None
    skill_proposal: dict | None


def load_active_capabilities(
    memory_run_dir: str | Path,
    skill_registry_path: str | Path,
) -> ActiveCapabilities:
    store, memory_info = load_active_memory(memory_run_dir)

    registry = read_json(Path(skill_registry_path))

    if registry.get("schema") != "skill_registry_v1":
        raise ValueError("Unsupported skill registry schema")

    candidates = registry.get("candidates")

    if not isinstance(candidates, list):
        raise ValueError("Invalid candidate list")

    records = {}

    for entry in candidates:
        if not isinstance(entry, dict):
            raise ValueError("Invalid skill registry entry")

        candidate_id = entry["id"]

        if candidate_id in records:
            raise ValueError("Duplicate skill candidate ID")

        path = Path(entry["record_path"])

        if file_hash(path) != entry["record_sha256"]:
            raise ValueError(
                f"Skill record hash mismatch: {candidate_id}"
            )

        record = read_json(path)

        if record.get("candidate_id") != candidate_id:
            raise ValueError("Candidate ID mismatch")

        if record.get("status") != entry["status"]:
            raise ValueError("Candidate status mismatch")

        skill_path = Path(record["skill_path"])
        decision_path = Path(record["decision_path"])

        if file_hash(skill_path) != record["skill_sha256"]:
            raise ValueError("Skill proposal hash mismatch")

        if file_hash(decision_path) != record["decision_sha256"]:
            raise ValueError("Skill decision hash mismatch")

        records[candidate_id] = record

    active_id = registry.get("active_skill_id")

    if active_id is None:
        if any(r["active"] for r in records.values()):
            raise ValueError(
                "Registry declares no active skill but "
                "a candidate is marked active"
            )

        return ActiveCapabilities(
            memory_store=store,
            memory_info=memory_info,
            skill_id=None,
            skill_mode=None,
            skill_proposal=None,
        )

    if active_id not in records:
        raise ValueError("Active skill ID not found")

    record = records[active_id]

    if (
        record["status"] != "ACCEPTED_ON_HOLDOUT"
        or record["active"] is not True
        or record.get("holdout_evaluated") is not True
    ):
        raise ValueError(
            "Cannot activate an unverified or rejected skill"
        )

    holdout_path = record.get("holdout_decision_path")
    holdout_hash = record.get("holdout_decision_sha256")

    if not holdout_path or not holdout_hash:
        raise ValueError(
            "Missing independent holdout acceptance evidence"
        )

    holdout_path = Path(holdout_path)

    if file_hash(holdout_path) != holdout_hash:
        raise ValueError("Holdout evidence hash mismatch")

    holdout_decision = read_json(holdout_path)

    if (
        holdout_decision.get("candidate_id") != active_id
        or holdout_decision.get("status") != "ACCEPTED_ON_HOLDOUT"
        or holdout_decision.get("accept") is not True
    ):
        raise ValueError("Invalid holdout acceptance decision")

    mode = record.get("intervention_mode")

    if mode not in ("hint", "execute"):
        raise ValueError("Invalid skill intervention mode")

    proposal = read_json(
        Path(record["skill_path"])
    )["proposal"]

    if (
        proposal.get("decision") != "CANDIDATE"
        or proposal.get("name") != record["name"]
    ):
        raise ValueError("Invalid active skill proposal")

    return ActiveCapabilities(
        memory_store=store,
        memory_info=memory_info,
        skill_id=active_id,
        skill_mode=mode,
        skill_proposal=proposal,
    )


def build_active_agent(
    llm,
    capabilities: ActiveCapabilities,
    *,
    max_steps: int = 50,
    top_k: int = 3,
) -> ReActAgent:
    kwargs = {}

    if capabilities.skill_id is not None:
        kwargs = {
            "search_skill_mode": capabilities.skill_mode,
            "search_skill_proposal": capabilities.skill_proposal,
        }

    return ReActAgent(
        llm,
        max_steps=max_steps,
        experience_store=capabilities.memory_store,
        experience_top_k=top_k,
        use_state_tracking=True,
        **kwargs,
    )
