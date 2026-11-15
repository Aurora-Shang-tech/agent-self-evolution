import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from src.evolution.capabilities import (
    build_active_agent,
    load_active_capabilities,
)


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ActiveCapabilitiesTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)

        root = Path(temp.name)

        self.memory_dir = root / "memory"
        self.registry_dir = root / "skills"
        self.registry_path = self.registry_dir / "index.json"

        memory = [{
            "id": "base_0000",
            "lesson": "Check the target object type.",
            "source_task": "clean a butterknife",
        }]

        memory_hash = hashlib.sha256(
            json.dumps(
                memory, ensure_ascii=False, sort_keys=True
            ).encode("utf-8")
        ).hexdigest()

        write_json(
            self.memory_dir / "versions/v000.json",
            {
                "version": "v000",
                "parent": None,
                "round": 0,
                "memory_hash": memory_hash,
                "memory_size": 1,
                "memory": memory,
            },
        )

        write_json(
            self.memory_dir / "state.json",
            {
                "active_version": "v000",
                "completed_rounds": 0,
            },
        )

        self.skill_path = root / "candidate.json"
        self.decision_path = root / "decision.json"
        self.record_path = self.registry_dir / "c001.json"

        write_json(
            self.skill_path,
            {
                "proposal": {
                    "decision": "CANDIDATE",
                    "name": "stalled_search_countertop_redirect",
                }
            },
        )

        write_json(
            self.decision_path,
            {"skill_status": "REJECTED_ON_DEV"},
        )

        write_json(
            self.record_path,
            {
                "candidate_id": "c001",
                "name": "stalled_search_countertop_redirect",
                "status": "REJECTED_ON_DEV",
                "active": False,
                "skill_path": str(self.skill_path),
                "skill_sha256": sha(self.skill_path),
                "decision_path": str(self.decision_path),
                "decision_sha256": sha(self.decision_path),
            },
        )

        self.save_registry(None)

    def save_registry(self, active_id):
        write_json(
            self.registry_path,
            {
                "schema": "skill_registry_v1",
                "active_skill_id": active_id,
                "candidates": [{
                    "id": "c001",
                    "status": "REJECTED_ON_DEV",
                    "record_path": str(self.record_path),
                    "record_sha256": sha(self.record_path),
                }],
            },
        )

    def load(self):
        return load_active_capabilities(
            self.memory_dir,
            self.registry_path,
        )

    def test_rejected_skill_stays_inactive(self):
        capabilities = self.load()

        self.assertEqual(capabilities.memory_info.version, "v000")
        self.assertEqual(capabilities.memory_info.memory_size, 1)
        self.assertEqual(len(capabilities.memory_store), 1)
        self.assertIsNone(capabilities.skill_id)

        agent = build_active_agent(
            object(),
            capabilities,
        )

        self.assertIsNone(agent.search_skill_mode)
        self.assertIsNone(agent.search_skill_proposal)

    def test_rejected_skill_cannot_be_activated(self):
        self.save_registry("c001")

        with self.assertRaisesRegex(
            ValueError, "Cannot activate"
        ):
            self.load()

    def test_tampered_skill_record_is_rejected(self):
        record = json.loads(
            self.record_path.read_text(encoding="utf-8")
        )
        record["active"] = True
        write_json(self.record_path, record)

        with self.assertRaisesRegex(
            ValueError, "hash mismatch"
        ):
            self.load()

    def test_corrupted_memory_is_rejected(self):
        path = self.memory_dir / "versions/v000.json"
        record = json.loads(path.read_text(encoding="utf-8"))
        record["memory"][0]["lesson"] = "Tampered lesson."
        write_json(path, record)

        with self.assertRaisesRegex(
            ValueError, "hash mismatch"
        ):
            self.load()


if __name__ == "__main__":
    unittest.main()
