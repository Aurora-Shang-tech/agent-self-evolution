
"""Read-only access to versioned continual-evolution memories."""

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

from src.memory.experience import Experience
from src.memory.store import ExperienceStore


@dataclass(frozen=True)
class MemoryVersionInfo:
    version: str
    parent: str | None
    round_id: int
    memory_size: int
    memory_hash: str


def read_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)

    data = json.loads(path.read_text(encoding="utf-8"))

    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object: {path}")

    return data


def active_version(run_dir: str | Path) -> str:
    root = Path(run_dir)
    state = read_json(root / "state.json")
    version = state.get("active_version")

    if not isinstance(version, str) or not re.fullmatch(
        r"v\d{3,}", version
    ):
        raise ValueError("Invalid active memory version")

    return version


def load_memory_version(
    run_dir: str | Path,
    version: str,
) -> tuple[ExperienceStore, MemoryVersionInfo]:
    if not re.fullmatch(r"v\d{3,}", version):
        raise ValueError("Invalid version identifier")

    root = Path(run_dir)
    path = root / "versions" / f"{version}.json"
    snapshot = read_json(path)

    if snapshot.get("version") != version:
        raise ValueError("Memory version mismatch")

    records = snapshot.get("memory")

    if not isinstance(records, list) or not records:
        raise ValueError("Memory snapshot is empty or invalid")

    if snapshot.get("memory_size") != len(records):
        raise ValueError("Memory size mismatch")

    ids = set()
    store = ExperienceStore()

    for index, item in enumerate(records):
        if not isinstance(item, dict):
            raise ValueError(f"Invalid memory item: {index}")

        memory_id = item.get("id")
        lesson = item.get("lesson")
        source_task = item.get("source_task")

        if not all(
            isinstance(value, str) and value.strip()
            for value in (memory_id, lesson, source_task)
        ):
            raise ValueError(f"Incomplete memory item: {index}")

        if memory_id in ids:
            raise ValueError(f"Duplicate memory ID: {memory_id}")

        ids.add(memory_id)

        store.add(
            Experience(
                failure_type="memory_evolution",
                lesson=lesson,
                source_task=source_task,
            )
        )

    actual_hash = hashlib.sha256(
        json.dumps(
            records,
            ensure_ascii=False,
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()

    if actual_hash != snapshot.get("memory_hash"):
        raise ValueError("Memory snapshot hash mismatch")

    info = MemoryVersionInfo(
        version=version,
        parent=snapshot.get("parent"),
        round_id=snapshot["round"],
        memory_size=len(records),
        memory_hash=actual_hash,
    )

    return store, info


def load_active_memory(
    run_dir: str | Path,
) -> tuple[ExperienceStore, MemoryVersionInfo]:
    version = active_version(run_dir)
    return load_memory_version(run_dir, version)


def read_round_history(
    run_dir: str | Path,
) -> list[dict]:
    root = Path(run_dir)
    state = read_json(root / "state.json")

    completed = state.get("completed_rounds")

    if type(completed) is not int or completed < 0:
        raise ValueError("Invalid completed round count")

    history = []
    previous = "v000"

    for round_id in range(1, completed + 1):
        path = (
            root
            / "rounds"
            / f"round_{round_id:02d}"
            / "decision.json"
        )

        record = read_json(path)

        if (
            record.get("round") != round_id
            or record.get("parent") != previous
        ):
            raise ValueError(
                f"Round history mismatch: {round_id}"
            )

        after = record["active_after"]
        result = record["result"]

        history.append({
            "round": round_id,
            "parent": previous,
            "candidate": record.get("candidate"),
            "active_after": after,
            "accepted": bool(result.get("accept", False)),
            "reason": result["reason"],
            "learning_failures": record["learning_failures"],
        })

        previous = after

    return history

