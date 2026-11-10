
"""ID-stable Memory Evolution and Utility Retrieval integration."""

import hashlib
import json
import os
from dataclasses import asdict
from pathlib import Path

from src.memory.experience import Experience
from src.memory.utility import UtilityStat, UtilityTracker
from src.memory.utility_store import HybridUtilityStore


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def read_catalog(path: Path) -> list[dict]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    records = raw.get("memory") if isinstance(raw, dict) else raw

    if not isinstance(records, list) or not records:
        raise ValueError("Memory snapshot must contain a nonempty memory list")

    catalog = []
    seen = set()

    for index, item in enumerate(records):
        if not isinstance(item, dict):
            raise ValueError(f"Invalid memory item {index}")

        memory_id = item.get("id")
        lesson = item.get("lesson")
        source_task = item.get("source_task")

        if not all(
            isinstance(value, str) and value.strip()
            for value in (memory_id, lesson, source_task)
        ):
            raise ValueError(
                f"Missing stable ID, lesson, or source_task: {index}"
            )

        if memory_id in seen:
            raise ValueError(f"Duplicate memory ID: {memory_id}")

        seen.add(memory_id)

        catalog.append({
            "id": memory_id,
            "lesson": lesson,
            "source_task": source_task,
        })

    return catalog


def empty_stat() -> dict:
    return asdict(UtilityStat())


class MemoryPipeline:
    def __init__(
        self,
        source_path: str | Path,
        state_path: str | Path,
        *,
        mode: str = "utility",
        import_v1: str | Path | None = None,
        sync: bool = False,
    ):
        if mode not in {"hybrid", "utility"}:
            raise ValueError("mode must be hybrid or utility")

        self.source_path = Path(source_path)
        self.state_path = Path(state_path)
        self.mode = mode

        self.source_hash = sha256(self.source_path)
        catalog = read_catalog(self.source_path)

        self.sync_counts = {
            "added": 0,
            "edited": 0,
            "removed": 0,
        }

        if self.state_path.is_file():
            if import_v1 is not None:
                raise ValueError(
                    "v1 import is allowed only when creating the state"
                )

            state = json.loads(
                self.state_path.read_text(encoding="utf-8")
            )

            if state.get("schema") != "unified_memory_v1":
                raise ValueError("Unrecognized pipeline state")

            if state["source_hash"] != self.source_hash:
                if not sync:
                    raise RuntimeError(
                        "Evolution snapshot changed. "
                        "Run --stage sync explicitly."
                    )

                state = self._reconcile(state, catalog)
                write_json(self.state_path, state)

            elif state["catalog"] != catalog:
                raise RuntimeError(
                    "Catalog changed without source hash change"
                )

        else:
            state = {
                "schema": "unified_memory_v1",
                "source_hash": self.source_hash,
                "catalog": catalog,
                "alpha": 0.3,
                "weight": 0.35,
                "pool_size": 8,
                "stats_by_id": {
                    item["id"]: empty_stat()
                    for item in catalog
                },
                "history": {},
                "archive": [],
                "imported_v1": None,
            }

            if import_v1 is not None:
                path = Path(import_v1)

                old = json.loads(
                    path.read_text(encoding="utf-8")
                )

                if old.get("memory_hash") != self.source_hash:
                    raise RuntimeError(
                        "v1 Q values belong to a different memory snapshot"
                    )

                old_tracker = UtilityTracker.load(
                    old["tracker"]
                )

                if len(old_tracker.stats) != len(catalog):
                    raise RuntimeError(
                        "v1 Q values and memory count differ"
                    )

                state["alpha"] = old_tracker.alpha

                state["stats_by_id"] = {
                    item["id"]: asdict(stat)
                    for item, stat in zip(
                        catalog,
                        old_tracker.stats,
                    )
                }

                state["imported_v1"] = {
                    "path": str(path),
                    "sha256": sha256(path),
                }

            write_json(self.state_path, state)

        self.state = state
        self.catalog = state["catalog"]

        self.id_to_index = {
            item["id"]: index
            for index, item in enumerate(self.catalog)
        }

        self.tracker = UtilityTracker(
            len(self.catalog),
            alpha=state["alpha"],
        )

        self.tracker.stats = [
            UtilityStat(
                **state["stats_by_id"][item["id"]]
            )
            for item in self.catalog
        ]

        experiences = [
            Experience(
                failure_type="memory_evolution",
                lesson=item["lesson"],
                source_task=item["source_task"],
            )
            for item in self.catalog
        ]

        self.hybrid = HybridUtilityStore(
            experiences,
            tracker=None,
            weight=state["weight"],
            pool_size=state["pool_size"],
        )

        self.utility = HybridUtilityStore(
            experiences,
            tracker=self.tracker,
            weight=state["weight"],
            pool_size=state["pool_size"],
        )

        self.last_task: str | None = None
        self.last_ids: list[str] = []

    def _reconcile(
        self,
        state: dict,
        catalog: list[dict],
    ) -> dict:
        old = {
            item["id"]: item
            for item in state["catalog"]
        }

        new = {
            item["id"]: item
            for item in catalog
        }

        stats = {}

        for item in catalog:
            memory_id = item["id"]

            if memory_id not in old:
                self.sync_counts["added"] += 1
                stats[memory_id] = empty_stat()

            elif old[memory_id] != item:
                self.sync_counts["edited"] += 1
                stats[memory_id] = empty_stat()

            else:
                stats[memory_id] = (
                    state["stats_by_id"][memory_id]
                )

        for memory_id in old.keys() - new.keys():
            self.sync_counts["removed"] += 1

            state["archive"].append({
                "memory": old[memory_id],
                "stats": state["stats_by_id"][memory_id],
            })

        state["catalog"] = catalog
        state["stats_by_id"] = stats
        state["source_hash"] = self.source_hash

        return state

    def rank_ids(
        self,
        task: str,
        *,
        mode: str | None = None,
        top_k: int = 3,
    ) -> list[str]:
        which = self.mode if mode is None else mode

        if which not in {"hybrid", "utility"}:
            raise ValueError("Unknown retrieval mode")

        store = (
            self.hybrid
            if which == "hybrid"
            else self.utility
        )

        indices = store.rank_indices(
            task,
            top_k,
        )

        return [
            self.catalog[index]["id"]
            for index in indices
        ]

    def retrieve(
        self,
        task: str,
        top_k: int = 3,
    ) -> list[Experience]:
        store = (
            self.hybrid
            if self.mode == "hybrid"
            else self.utility
        )

        selected = store.retrieve(
            task,
            top_k,
        )

        self.last_task = task

        self.last_ids = [
            self.catalog[index]["id"]
            for index in store.last_indices
        ]

        return selected

    def record_feedback(
        self,
        *,
        event_id: str,
        task: str,
        success: bool,
        decision_steps: int,
        selected_ids: list[str] | None = None,
    ) -> bool:
        if not event_id.strip() or decision_steps < 1:
            raise ValueError("Invalid feedback event")

        ids = (
            list(selected_ids)
            if selected_ids is not None
            else list(self.last_ids)
        )

        if selected_ids is None and task != self.last_task:
            raise RuntimeError(
                "No retrieval was recorded for this task"
            )

        if not ids or len(ids) != len(set(ids)):
            raise ValueError(
                "Invalid selected memory IDs"
            )

        if any(
            memory_id not in self.id_to_index
            for memory_id in ids
        ):
            raise RuntimeError(
                "Selected memory was removed; "
                "cannot assign feedback"
            )

        record = {
            "task": task,
            "selected_ids": ids,
            "success": bool(success),
            "decision_steps": decision_steps,
            "source_hash": self.source_hash,
        }

        previous = self.state["history"].get(event_id)

        if previous is not None:
            if previous != record:
                raise RuntimeError(
                    f"Feedback event conflict: {event_id}"
                )
            return False

        if sha256(self.source_path) != self.source_hash:
            raise RuntimeError(
                "Evolution snapshot changed during this episode"
            )

        indices = [
            self.id_to_index[memory_id]
            for memory_id in ids
        ]

        self.tracker.observe(
            indices,
            bool(success),
        )

        self.state["history"][event_id] = record

        self.state["stats_by_id"] = {
            item["id"]: asdict(
                self.tracker.stats[index]
            )
            for index, item in enumerate(self.catalog)
        }

        write_json(
            self.state_path,
            self.state,
        )

        return True

