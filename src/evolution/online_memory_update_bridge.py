from dataclasses import asdict
from pathlib import Path

from scripts.memory.run_evolution import memory_hash
from src.evolution.online_memory_gate import (
    fingerprint,
    stage_memory_candidate,
)
from src.evolution.review_consumer import write_immutable
from src.evolution.version_store import read_json
from src.memory.updater import (
    ExperienceUpdater,
    LessonCandidate,
    UpdateProposal,
)


class OnlineMemoryUpdateBridge:
    def __init__(self, llm, *, top_k=8):
        if type(top_k) is not int or top_k < 1:
            raise ValueError("Invalid update retrieval top_k")

        self.llm = llm
        self.top_k = top_k

    @staticmethod
    def validate(record, job, candidate, parent_memory):
        if (
            record.get("schema")
            != "online_memory_update_proposal_v1"
            or record.get("job_id") != job["job_id"]
            or record.get("trajectory_sha256")
            != job["trajectory_sha256"]
            or record.get("event_sha256")
            != job["event_sha256"]
            or record.get("candidate_sha256")
            != fingerprint(asdict(candidate))
            or record.get("parent_memory_sha256")
            != memory_hash(parent_memory)
        ):
            raise RuntimeError("Cached update provenance mismatch")

        proposal = UpdateProposal(**record["proposal"])

        expected_stage = stage_memory_candidate(
            job,
            candidate,
            proposal,
            parent_memory,
        )

        if record.get("staged") != expected_stage:
            raise RuntimeError("Cached update stage mismatch")

        if record.get("active_memory_changed") is not False:
            raise RuntimeError("Unexpected memory activation")

        return record

    def process(
        self,
        job,
        candidate,
        parent_memory,
        output_path,
    ):
        if not isinstance(candidate, LessonCandidate):
            raise TypeError("Expected LessonCandidate")

        if candidate.decision != "CANDIDATE":
            raise ValueError("ABSTAIN cannot enter memory update")

        output_path = Path(output_path)

        if output_path.is_file():
            return self.validate(
                read_json(output_path),
                job,
                candidate,
                parent_memory,
            )

        if (
            job.get("route") != "MEMORY_REVIEW"
            or job.get("source_scope")
            != "ONLINE_TRAIN_DEVELOPMENT"
            or not job.get("game_id", "").startswith("train:")
        ):
            raise ValueError("Unauthorized memory update source")

        proposal = ExperienceUpdater(self.llm).propose(
            candidate,
            parent_memory,
            top_k=self.top_k,
        )

        staged = stage_memory_candidate(
            job,
            candidate,
            proposal,
            parent_memory,
        )

        record = {
            "schema": "online_memory_update_proposal_v1",
            "job_id": job["job_id"],
            "trajectory_sha256": job["trajectory_sha256"],
            "event_sha256": job["event_sha256"],
            "candidate_sha256": fingerprint(asdict(candidate)),
            "parent_memory_sha256": memory_hash(parent_memory),
            "proposal": asdict(proposal),
            "staged": staged,
            "active_memory_changed": False,
        }

        write_immutable(output_path, record)

        return self.validate(
            read_json(output_path),
            job,
            candidate,
            parent_memory,
        )
