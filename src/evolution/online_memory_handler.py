import hashlib
import json
import os
from dataclasses import asdict
from pathlib import Path

from src.analysis.failure_analyzer import FailureAnalysis, FailureAnalyzer
from src.core.trajectory import Trajectory, TrajectoryStep
from src.counterfactual.verifier import (
    CounterfactualBatchResult,
    CounterfactualResult,
    verify_counterfactual,
)
from src.evolution.capabilities import build_active_agent
from src.evolution.online_scheduler import OnlineReviewScheduler
from src.evolution.review_consumer import write_immutable
from src.llm.client import LLMClient
from src.memory.updater import ComparativeExtractor


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


class BudgetedLLM:
    def __init__(self, factory, limit):
        self.factory = factory
        self.limit = limit
        self.calls = 0
        self.client = None

    def chat(self, messages):
        if self.calls >= self.limit:
            raise RuntimeError("LLM call budget exhausted")

        if self.client is None:
            self.client = self.factory()

        self.calls += 1
        return self.client.chat(messages)


class OnlineMemoryHandler:
    def __init__(
        self,
        capabilities,
        *,
        model,
        max_steps=50,
        top_k=3,
        max_continuation_decisions=12,
        max_llm_calls=16,
        llm_factory=None,
    ):
        if (
            max_steps < 1
            or top_k < 1
            or max_continuation_decisions < 1
            or max_llm_calls < 1
        ):
            raise ValueError("Invalid Memory Review budget")

        self.capabilities = capabilities
        self.model = model
        self.max_steps = max_steps
        self.top_k = top_k
        self.max_continuation_decisions = max_continuation_decisions
        self.max_llm_calls = max_llm_calls
        self.llm_factory = llm_factory or self._create_llm

    def _create_llm(self):
        if os.getenv("OPENAI_MODEL") != self.model:
            raise RuntimeError("Unexpected OPENAI_MODEL")
        return LLMClient()

    def source(self, job):
        return {
            "job_id": job["job_id"],
            "trajectory_sha256": job["trajectory_sha256"],
            "event_sha256": job["event_sha256"],
            "memory_version": self.capabilities.memory_info.version,
            "memory_sha256": self.capabilities.memory_info.memory_hash,
            "active_skill_id": self.capabilities.skill_id,
            "model": self.model,
        }

    @staticmethod
    def check_record(record, expected):
        if any(record.get(k) != v for k, v in expected.items()):
            raise RuntimeError("Cached Memory Review provenance mismatch")

    @staticmethod
    def restore_result(data):
        raw = data["result"]
        trajectory = raw["trajectory"]

        if trajectory is not None:
            trajectory = Trajectory(
                task=trajectory["task"],
                gamefile=trajectory["gamefile"],
                success=trajectory["success"],
                truncated=trajectory["truncated"],
                steps=[
                    TrajectoryStep(**step)
                    for step in trajectory["steps"]
                ],
                messages=trajectory["messages"],
            )

        return CounterfactualResult(
            critical_step=raw["critical_step"],
            original_action=raw["original_action"],
            counterfactual_action=raw["counterfactual_action"],
            success=raw["success"],
            continuation_steps=raw["continuation_steps"],
            trajectory=trajectory,
        )

    def validate_outcome(self, job, path):
        OnlineReviewScheduler.verify_source(job)
        record = read_json(path)
        self.check_record(record, self.source(job))

        if (
            record.get("schema") != "online_memory_outcome_v2"
            or record.get("memory_operation") != "NO_CHANGE"
            or record.get("memory_updated") is not False
            or record.get("skill_updated") is not False
        ):
            raise RuntimeError("Invalid Memory Review outcome")

        folder = Path(path).parent

        if (
            record["analysis_sha256"]
            != digest(folder / "automated_analysis.json")
            or record["branch_sha256"]
            != digest(folder / "automated_branch_00.json")
        ):
            raise RuntimeError("Memory Review evidence changed")

        branch = read_json(folder / "automated_branch_00.json")
        self.check_record(branch, self.source(job))

        result = branch["result"]
        candidate_path = folder / "automated_candidate.json"

        if result["success"]:
            if (
                not candidate_path.is_file()
                or record["candidate_sha256"]
                != digest(candidate_path)
            ):
                raise RuntimeError("Missing verified candidate evidence")

            candidate = read_json(candidate_path)
            self.check_record(candidate, self.source(job))

            expected = (
                "CANDIDATE_PENDING_GATE"
                if candidate["candidate"]["decision"] == "CANDIDATE"
                else "ABSTAIN"
            )
        else:
            if (
                candidate_path.exists()
                or record["candidate_sha256"] is not None
            ):
                raise RuntimeError("Unverified lesson candidate")

            expected = "NO_SUCCESS_WITHIN_BUDGET"

        if record["status"] != expected:
            raise RuntimeError("Memory Review status mismatch")

        return record["status"]

    def process(self, job, output_path):
        OnlineReviewScheduler.verify_source(job)

        if job["route"] != "MEMORY_REVIEW":
            raise ValueError("Memory Handler received wrong route")

        output_path = Path(output_path)

        if output_path.exists():
            return self.validate_outcome(job, output_path)

        event = read_json(job["event_path"])
        source = self.source(job)

        if (
            event["memory_version"] != source["memory_version"]
            or event["memory_sha256"] != source["memory_sha256"]
            or event["active_skill_id"] != source["active_skill_id"]
        ):
            raise RuntimeError("Active capability mismatch")

        trace = Trajectory.load_json(job["trajectory_path"])

        if trace.success or not trace.steps:
            raise ValueError("Memory Review requires a failed trajectory")

        llm = BudgetedLLM(
            self.llm_factory,
            self.max_llm_calls,
        )

        folder = output_path.parent
        analysis_path = folder / "automated_analysis.json"
        branch_path = folder / "automated_branch_00.json"
        candidate_path = folder / "automated_candidate.json"

        analysis_source = {
            **source,
            "schema": "automated_failure_analysis_v1",
        }

        if analysis_path.exists():
            saved = read_json(analysis_path)
            self.check_record(saved, analysis_source)

            raw = dict(saved["analysis"])
            raw["counterfactual_actions"] = tuple(
                raw["counterfactual_actions"]
            )
            analysis = FailureAnalysis(**raw)
        else:
            analysis = FailureAnalyzer(llm).analyze(
                trace,
                max_retries=0,
            )

            write_immutable(
                analysis_path,
                {
                    **analysis_source,
                    "analysis": asdict(analysis),
                },
            )

        FailureAnalyzer._validate_analysis(trace, analysis)

        action = analysis.counterfactual_actions[0]
        limit = min(
            self.max_steps,
            analysis.critical_step
            + 1
            + self.max_continuation_decisions,
        )

        branch_source = {
            **source,
            "schema": "automated_counterfactual_v1",
            "analysis_sha256": digest(analysis_path),
            "candidate_index": 0,
            "candidate_action": action,
            "max_steps_used": limit,
        }

        if branch_path.exists():
            saved = read_json(branch_path)
            self.check_record(saved, branch_source)
            result = self.restore_result(saved)
        else:
            agent = build_active_agent(
                llm,
                self.capabilities,
                max_steps=limit,
                top_k=self.top_k,
            )

            result = verify_counterfactual(
                trace,
                analysis.critical_step,
                action,
                agent,
                max_steps=limit,
            )

            write_immutable(
                branch_path,
                {
                    **branch_source,
                    "result": asdict(result),
                },
            )

        if (
            result.critical_step != analysis.critical_step
            or result.original_action != analysis.original_action
            or result.counterfactual_action != action
            or (
                result.success
                and (
                    result.trajectory is None
                    or not result.trajectory.success
                )
            )
        ):
            raise RuntimeError("Counterfactual result mismatch")

        candidate_hash = None

        if result.success:
            candidate_source = {
                **source,
                "schema": "automated_lesson_candidate_v1",
                "branch_sha256": digest(branch_path),
            }

            if candidate_path.exists():
                saved = read_json(candidate_path)
                self.check_record(saved, candidate_source)
                candidate = saved["candidate"]
            else:
                batch = CounterfactualBatchResult(
                    critical_step=analysis.critical_step,
                    original_action=analysis.original_action,
                    results=(result,),
                )

                candidate = asdict(
                    ComparativeExtractor(llm).extract(
                        trace,
                        batch,
                        candidate_index=0,
                    )
                )

                write_immutable(
                    candidate_path,
                    {
                        **candidate_source,
                        "candidate": candidate,
                    },
                )

            if candidate["decision"] not in ("CANDIDATE", "ABSTAIN"):
                raise RuntimeError("Invalid lesson extraction decision")

            status = (
                "CANDIDATE_PENDING_GATE"
                if candidate["decision"] == "CANDIDATE"
                else "ABSTAIN"
            )
            candidate_hash = digest(candidate_path)
        else:
            if candidate_path.exists():
                raise RuntimeError(
                    "Failed branch cannot have a lesson candidate"
                )

            status = "NO_SUCCESS_WITHIN_BUDGET"

        outcome = {
            **source,
            "schema": "online_memory_outcome_v2",
            "status": status,
            "analysis_sha256": digest(analysis_path),
            "branch_sha256": digest(branch_path),
            "candidate_sha256": candidate_hash,
            "verified_successes": int(result.success),
            "tested_candidates": [action],
            "untested_candidates": list(
                analysis.counterfactual_actions[1:]
            ),
            "memory_operation": "NO_CHANGE",
            "memory_updated": False,
            "skill_updated": False,
            "llm_calls_this_attempt": llm.calls,
        }

        write_immutable(output_path, outcome)
        return status
