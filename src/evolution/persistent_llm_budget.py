import hashlib
import json
import os
import re
from pathlib import Path

from src.evolution.review_consumer import write_immutable


def read_json(path):
    return json.loads(
        Path(path).read_text(encoding="utf-8")
    )


def sha256(path):
    return hashlib.sha256(
        Path(path).read_bytes()
    ).hexdigest()


class PersistentLLMBudget:
    def __init__(
        self,
        root,
        *,
        job,
        model,
        memory_version,
        memory_sha256,
        active_skill_id,
        max_calls,
        client_factory,
    ):
        if type(max_calls) is not int or not 1 <= max_calls <= 9999:
            raise ValueError("Invalid persistent LLM budget")

        job_id = job["job_id"]

        if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", job_id):
            raise ValueError("Invalid job ID")

        if (
            job["route"] != "MEMORY_REVIEW"
            or job["source_scope"] != "ONLINE_TRAIN_DEVELOPMENT"
            or not job["game_id"].startswith("train:")
        ):
            raise ValueError("Unauthorized LLM budget source")

        self.root = Path(root) / job_id
        self.root.mkdir(parents=True, exist_ok=True)

        self.max_calls = max_calls
        self.client_factory = client_factory
        self.client = None

        self.metadata = {
            "schema": "persistent_llm_budget_v1",
            "job_id": job_id,
            "game_id": job["game_id"],
            "route": job["route"],
            "trajectory_sha256": job["trajectory_sha256"],
            "event_sha256": job["event_sha256"],
            "model": model,
            "memory_version": memory_version,
            "memory_sha256": memory_sha256,
            "active_skill_id": active_skill_id,
            "max_calls": max_calls,
        }

        self.metadata_path = self.root / "budget.json"

        write_immutable(
            self.metadata_path,
            self.metadata,
        )

        self.metadata_hash = sha256(self.metadata_path)
        self._used_calls()

    def _used_calls(self):
        paths = sorted(self.root.glob("call_*.json"))

        if len(paths) > self.max_calls:
            raise RuntimeError("LLM budget overrun")

        for index, path in enumerate(paths, start=1):
            expected_name = f"call_{index:04d}.json"

            if path.name != expected_name:
                raise RuntimeError(
                    "LLM budget ledger has missing or invalid slots"
                )

            expected = {
                "schema": "llm_call_reservation_v1",
                "index": index,
                "metadata_sha256": self.metadata_hash,
            }

            if read_json(path) != expected:
                raise RuntimeError(
                    "Corrupted LLM call reservation"
                )

        return len(paths)

    @property
    def used_calls(self):
        return self._used_calls()

    @property
    def remaining_calls(self):
        return self.max_calls - self.used_calls

    def chat(self, messages):
        used = self._used_calls()

        if used >= self.max_calls:
            raise RuntimeError(
                "Persistent LLM call budget exhausted"
            )

        if self.client is None:
            self.client = self.client_factory()

        index = used + 1
        path = self.root / f"call_{index:04d}.json"

        reservation = {
            "schema": "llm_call_reservation_v1",
            "index": index,
            "metadata_sha256": self.metadata_hash,
        }

        try:
            with path.open("x", encoding="utf-8") as f:
                json.dump(reservation, f)
                f.write("\n")
                f.flush()
                os.fsync(f.fileno())
        except FileExistsError as exc:
            raise RuntimeError(
                "Concurrent LLM reservation detected"
            ) from exc

        response = self.client.chat(messages)

        if not isinstance(response, str):
            raise RuntimeError(
                "LLM returned an invalid response"
            )

        write_immutable(
            self.root / f"done_{index:04d}.json",
            {
                "schema": "llm_call_completed_v1",
                "index": index,
                "reservation_sha256": sha256(path),
                "response_sha256": hashlib.sha256(
                    response.encode("utf-8")
                ).hexdigest(),
            },
        )

        return response
