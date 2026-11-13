
"""Single-process continual ALFWorld evolution controller.

Reuses the existing ReAct, counterfactual, extraction and updater modules.
Versions are immutable. The active pointer changes only after paired validation.
"""

import hashlib
import json
import os
from dataclasses import asdict
from pathlib import Path
from statistics import mean

from scripts.memory.run_evolution import (
    apply_proposal,
    atomic_json,
    memory_hash,
)
from src.agents.react_agent import ReActAgent
from src.analysis.failure_analyzer import (
    FailureAnalysis,
    FailureAnalyzer,
)
from src.core.trajectory import Trajectory
from src.counterfactual.verifier import (
    CounterfactualBatchResult,
    CounterfactualResult,
    verify_counterfactual_candidates,
)
from src.data.manifest import load_manifest
from src.envs.environment import ALFWorldEnvironment
from src.llm.client import LLMClient
from src.memory.experience import Experience
from src.memory.store import ExperienceStore
from src.memory.updater import (
    ComparativeExtractor,
    ExperienceUpdater,
    LessonCandidate,
    MemoryItem,
    UpdateProposal,
)


ROOT = Path("outputs/continual_v1")

CODE_FILES = [
    "src/core/trajectory.py",
    "src/agents/react_agent.py",
    "src/core/protocol.py",
    "src/memory/state_tracker.py",
    "src/counterfactual/verifier.py",
    "src/counterfactual/replay.py",
    "src/analysis/failure_analyzer.py",
    "src/memory/updater.py",
    "src/memory/store.py",
    "scripts/memory/run_evolution.py",
    "src/evolution/controller.py",
    "scripts/run_continual_evolution.py",
]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def save_trace(path: Path, trajectory: Trajectory) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    temporary = path.with_name(path.name + ".tmp")
    trajectory.save_json(str(temporary))

    os.replace(temporary, path)


def initial_memory(path: Path) -> list[MemoryItem]:
    raw = load_json(path)
    entries = raw.get("memory") if isinstance(raw, dict) else raw

    if not isinstance(entries, list) or not entries:
        raise ValueError("Invalid initial memory")

    result = []

    for i, entry in enumerate(entries):
        lesson = entry["lesson"].strip()
        task = entry["source_task"].strip()

        if not lesson or not task:
            raise ValueError(f"Invalid memory at {i}")

        result.append(
            MemoryItem(
                id=f"base_{i:04d}",
                lesson=lesson,
                source_task=task,
            )
        )

    return result


def experience_store(
    memory: list[MemoryItem],
) -> ExperienceStore:
    store = ExperienceStore()

    for item in memory:
        store.add(
            Experience(
                "memory_evolution",
                item.lesson,
                item.source_task,
            ),
            is_new=item.id.startswith("evo_"),
        )

    return store


def gate(
    base: list[dict],
    cand: list[dict],
    *,
    max_regressions=1,
    min_step_saving=1.0,
    max_step_penalty=2.0,
) -> dict:
    if not base or len(base) != len(cand):
        raise ValueError("Incomplete paired validation")

    if (
        [x["game_id"] for x in base]
        != [x["game_id"] for x in cand]
    ):
        raise ValueError("Validation tasks not aligned")

    improved = sum(
        not a["success"] and b["success"]
        for a, b in zip(base, cand)
    )

    regressed = sum(
        a["success"] and not b["success"]
        for a, b in zip(base, cand)
    )

    base_success = sum(a["success"] for a in base)
    cand_success = sum(b["success"] for b in cand)

    delta = mean(
        b["decision_steps"] - a["decision_steps"]
        for a, b in zip(base, cand)
    )

    pass_success = (
        cand_success > base_success
        and regressed <= max_regressions
        and delta <= max_step_penalty
    )

    pass_efficiency = (
        cand_success == base_success
        and improved == 0
        and regressed == 0
        and delta <= -min_step_saving
    )

    return {
        "accept": pass_success or pass_efficiency,
        "baseline_successes": base_success,
        "candidate_successes": cand_success,
        "improved": improved,
        "regressed": regressed,
        "average_baseline_steps": mean(
            r["decision_steps"] for r in base
        ),
        "average_candidate_steps": mean(
            r["decision_steps"] for r in cand
        ),
        "average_step_delta": delta,
        "reason": (
            "success_gain"
            if pass_success
            else (
                "efficiency_gain"
                if pass_efficiency
                else "no_sufficient_gain"
            )
        ),
    }


class EvolutionController:
    def __init__(
        self,
        *,
        run_name: str,
        mode: str,
        pilot: bool,
    ):
        if mode not in {"fixed", "append", "dynamic"}:
            raise ValueError("Unsupported evolution mode")

        if (
            not run_name
            or any(
                c not in (
                    "abcdefghijklmnopqrstuvwxyz"
                    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                    "0123456789_-."
                )
                for c in run_name
            )
        ):
            raise ValueError("Invalid run name")

        self.mode = mode
        self.pilot = pilot

        self.home = ROOT / "runs" / run_name
        self.protocol_path = ROOT / "protocol.json"
        self.protocol = load_json(self.protocol_path)

        self.model = self.protocol["model"]
        self.max_steps = self.protocol["max_steps"]
        self.top_k = self.protocol["top_k"]

        self.max_failures = self.protocol[
            "max_failures_per_round"
        ]
        self.max_actions = self.protocol[
            "max_counterfactual_actions_per_failure"
        ]

        self.rounds = (
            1 if pilot else self.protocol["rounds"]
        )

        self.learn_limit = (
            2 if pilot
            else self.protocol["learning_tasks_per_round"]
        )

        self.validation_limit = (
            2 if pilot
            else self.protocol["validation_tasks"]
        )

        if (
            self.protocol["retrieval"] != "source_task_bm25_type_priority"
            or not self.protocol["state_tracking"]
        ):
            raise ValueError(
                "Unexpected frozen retrieval/state configuration"
            )

        if self.protocol["methods"] != [
            "fixed",
            "append_only",
            "dynamic",
        ]:
            raise ValueError(
                "Unexpected frozen experiment methods"
            )

        if self.max_actions < 1 or self.max_failures < 0:
            raise ValueError("Invalid budgets")

        self.learn_sets = []

        for index in range(1, self.rounds + 1):
            file = ROOT / f"learning_round_{index:02d}.json"

            if (
                digest(file)
                != self.protocol["manifests"][file.name]
            ):
                raise RuntimeError(
                    "Learning manifest changed"
                )

            self.learn_sets.append(
                load_manifest(file)[:self.learn_limit]
            )

        validation_path = ROOT / "validation.json"

        if (
            digest(validation_path)
            != self.protocol["manifests"][validation_path.name]
        ):
            raise RuntimeError(
                "Validation manifest changed"
            )

        self.validation_games = load_manifest(
            validation_path
        )[:self.validation_limit]

        final_path = ROOT / "final_holdout.json"

        if (
            digest(final_path)
            != self.protocol["manifests"][final_path.name]
        ):
            raise RuntimeError(
                "Final holdout manifest changed"
            )

        self.initial_path = Path(
            self.protocol["initial_memory"]
        )

        if (
            digest(self.initial_path)
            != self.protocol["initial_memory_sha256"]
        ):
            raise RuntimeError(
                "Original memory has changed"
            )

        self.frozen_config = {
            "schema": "continual_run_v1",
            "protocol_sha256": digest(
                self.protocol_path
            ),
            "mode": mode,
            "pilot": pilot,
            "code_sha256": {
                path: digest(Path(path))
                for path in CODE_FILES
            },
        }

        self._llm = None

    @property
    def llm(self):
        if self._llm is None:
            if os.getenv("OPENAI_MODEL") != self.model:
                raise RuntimeError(
                    f"OPENAI_MODEL must be {self.model}"
                )

            self._llm = LLMClient()

        return self._llm

    def initialize(self):
        config_file = self.home / "run_config.json"
        state_file = self.home / "state.json"

        if config_file.exists():
            if (
                load_json(config_file)
                != self.frozen_config
            ):
                raise RuntimeError(
                    "Run configuration/code changed; "
                    "choose new run-name"
                )

        else:
            if state_file.exists():
                raise RuntimeError(
                    "Run state exists without configuration"
                )

            self.home.mkdir(
                parents=True,
                exist_ok=True,
            )

            atomic_json(
                config_file,
                self.frozen_config,
            )

        if not state_file.exists():
            memory = initial_memory(
                self.initial_path
            )

            self.save_version(
                "v000",
                None,
                0,
                memory,
            )

            atomic_json(
                state_file,
                {
                    "active_version": "v000",
                    "completed_rounds": 0,
                },
            )

        return load_json(state_file)

    def save_version(
        self,
        version: str,
        parent: str | None,
        round_id: int,
        memory: list[MemoryItem],
    ):
        path = (
            self.home
            / "versions"
            / f"{version}.json"
        )

        record = {
            "version": version,
            "parent": parent,
            "round": round_id,
            "memory_hash": memory_hash(memory),
            "memory_size": len(memory),
            "memory": [
                asdict(item)
                for item in memory
            ],
        }

        if path.exists():
            if load_json(path) != record:
                raise RuntimeError(
                    "Existing version snapshot differs"
                )

        else:
            atomic_json(path, record)

    def load_version(
        self,
        version: str,
    ) -> list[MemoryItem]:
        raw = load_json(
            self.home
            / "versions"
            / f"{version}.json"
        )

        memory = [
            MemoryItem(**item)
            for item in raw["memory"]
        ]

        if (
            memory_hash(memory)
            != raw["memory_hash"]
        ):
            raise RuntimeError(
                "Corrupt memory version"
            )

        return memory

    def run_episode(
        self,
        game,
        memory,
        path: Path,
    ) -> Trajectory:
        if path.exists():
            trace = Trajectory.load_json(
                str(path)
            )

        else:
            env = ALFWorldEnvironment(
                str(game.game_file),
                max_steps=self.max_steps,
            )

            agent = ReActAgent(
                self.llm,
                max_steps=self.max_steps,
                experience_store=experience_store(memory),
                experience_top_k=self.top_k,
                use_state_tracking=True,
            )

            try:
                trace = agent.run(env)
            finally:
                env.close()

            save_trace(path, trace)

        if (
            Path(trace.gamefile).resolve()
            != Path(game.game_file).resolve()
        ):
            raise RuntimeError(
                "Cached episode game mismatch"
            )

        return trace

    def do_learning(
        self,
        rd: Path,
        games,
        memory,
    ) -> list[tuple[int, object, Trajectory]]:
        failed = []

        for index, game in enumerate(games):
            path = (
                rd
                / "learning"
                / f"episode_{index:03d}.json"
            )

            trace = self.run_episode(
                game,
                memory,
                path,
            )

            print(
                f"[learn] {index + 1}/{len(games)} "
                f"success={trace.success} "
                f"steps={len(trace)}",
                flush=True,
            )

            if (
                not trace.success
                and len(failed) < self.max_failures
            ):
                failed.append(
                    (index, game, trace)
                )

        return failed

    def read_verification(
        self,
        data,
        directory: Path,
    ) -> CounterfactualBatchResult:
        results = []

        for idx, item in enumerate(data["results"]):
            trace_path = (
                directory
                / f"branch_{idx:02d}.json"
            )

            continuation = (
                Trajectory.load_json(str(trace_path))
                if item["has_trajectory"]
                else None
            )

            results.append(
                CounterfactualResult(
                    critical_step=data["critical_step"],
                    original_action=data["original_action"],
                    counterfactual_action=item["action"],
                    success=item["success"],
                    continuation_steps=item["steps"],
                    trajectory=continuation,
                )
            )

        return CounterfactualBatchResult(
            data["critical_step"],
            data["original_action"],
            tuple(results),
        )

    def process_failure(
        self,
        rd: Path,
        index: int,
        trace: Trajectory,
        memory: list[MemoryItem],
    ) -> list[MemoryItem]:
        folder = (
            rd
            / "failures"
            / f"episode_{index:03d}"
        )

        event_file = folder / "event.json"

        event_id = (
            f"r{rd.name[-2:]}_e{index:03d}"
        )

        before = memory_hash(memory)

        if event_file.exists():
            event = load_json(event_file)

            if event["before_hash"] != before:
                raise RuntimeError(
                    "Memory replay divergence"
                )

            updated = list(memory)

            if event["proposal"] is not None:
                apply_proposal(
                    updated,
                    LessonCandidate(**event["candidate"]),
                    UpdateProposal(**event["proposal"]),
                    event_id,
                )

            if (
                memory_hash(updated)
                != event["after_hash"]
            ):
                raise RuntimeError(
                    "Memory replay after-hash mismatch"
                )

            return updated

        analysis_file = folder / "analysis.json"

        if analysis_file.exists():
            raw = load_json(analysis_file)

            raw["counterfactual_actions"] = tuple(
                raw["counterfactual_actions"]
            )

            analysis = FailureAnalysis(**raw)

        else:
            analysis = FailureAnalyzer(
                self.llm
            ).analyze(trace)

            atomic_json(
                analysis_file,
                asdict(analysis),
            )

        verification_file = folder / "verification.json"

        if verification_file.exists():
            verification = self.read_verification(
                load_json(verification_file),
                folder,
            )

        else:
            cf_agent = ReActAgent(
                self.llm,
                max_steps=self.max_steps,
                experience_store=experience_store(memory),
                experience_top_k=self.top_k,
                use_state_tracking=True,
            )

            verification = verify_counterfactual_candidates(
                trajectory=trace,
                critical_step=analysis.critical_step,
                counterfactual_actions=tuple(
                    analysis.counterfactual_actions[
                        :self.max_actions
                    ]
                ),
                agent=cf_agent,
                max_steps=self.max_steps,
            )

            results = []

            for j, item in enumerate(
                verification.results
            ):
                has_trajectory = (
                    item.trajectory is not None
                )

                if has_trajectory:
                    save_trace(
                        folder / f"branch_{j:02d}.json",
                        item.trajectory,
                    )

                results.append({
                    "action": item.counterfactual_action,
                    "success": item.success,
                    "steps": item.continuation_steps,
                    "has_trajectory": has_trajectory,
                })

            atomic_json(
                verification_file,
                {
                    "critical_step": verification.critical_step,
                    "original_action": verification.original_action,
                    "results": results,
                },
            )

        successful = [
            i
            for i, item in enumerate(verification.results)
            if item.success
            and item.trajectory is not None
            and len(item.trajectory.steps) > 0
        ]

        candidate = None
        proposal = None

        if successful:
            candidate_file = folder / "candidate.json"

            if candidate_file.exists():
                candidate = LessonCandidate(
                    **load_json(candidate_file)
                )

            else:
                extractor = ComparativeExtractor(self.llm)

                for branch_index in successful:
                    result = extractor.extract(
                        trace,
                        verification,
                        candidate_index=branch_index,
                    )

                    candidate = result

                    if result.decision == "CANDIDATE":
                        break

                atomic_json(
                    candidate_file,
                    asdict(candidate),
                )

            if (
                candidate.decision == "CANDIDATE"
                and self.mode != "fixed"
            ):
                duplicate = any(
                    item.lesson.strip().casefold()
                    == candidate.lesson.casefold()
                    for item in memory
                )

                if not duplicate:
                    proposal_file = (
                        folder / "proposal.json"
                    )

                    if proposal_file.exists():
                        proposal = UpdateProposal(
                            **load_json(proposal_file)
                        )

                    elif self.mode == "append":
                        proposal = UpdateProposal(
                            operation="ADD",
                            target_id=None,
                            new_lesson=candidate.lesson,
                            reason="Append-only control group",
                            evidence_refs=candidate.evidence_refs,
                            memory_snapshot_sha256=before,
                        )

                        atomic_json(
                            proposal_file,
                            asdict(proposal),
                        )

                    else:
                        proposal = ExperienceUpdater(
                            self.llm
                        ).propose(
                            candidate,
                            memory,
                            top_k=8,
                        )

                        atomic_json(
                            proposal_file,
                            asdict(proposal),
                        )

        updated = list(memory)

        if proposal is not None:
            apply_proposal(
                updated,
                candidate,
                proposal,
                event_id,
            )

        atomic_json(
            event_file,
            {
                "before_hash": before,
                "after_hash": memory_hash(updated),
                "candidate": (
                    asdict(candidate)
                    if candidate is not None
                    else None
                ),
                "proposal": (
                    asdict(proposal)
                    if proposal is not None
                    else None
                ),
                "outcome": (
                    proposal.operation
                    if proposal
                    else "NO_CHANGE"
                ),
            },
        )

        print(
            f"[evolve] failure={index:03d} "
            f"result={proposal.operation if proposal else 'NO_CHANGE'} "
            f"memory={len(updated)}",
            flush=True,
        )

        return updated

    def retrieval_changes(
        self,
        parent_memory,
        candidate_memory,
    ) -> int:
        base_store = experience_store(
            parent_memory
        )
        cand_store = experience_store(
            candidate_memory
        )

        changed = 0

        for game in self.validation_games:
            env = ALFWorldEnvironment(
                str(game.game_file),
                max_steps=self.max_steps,
            )

            try:
                task = env.reset().task
            finally:
                env.close()

            before = [
                x.lesson
                for x in base_store.retrieve(
                    task,
                    self.top_k,
                )
            ]

            after = [
                x.lesson
                for x in cand_store.retrieve(
                    task,
                    self.top_k,
                )
            ]

            changed += before != after

        print(
            f"[precheck] changed validation prompts: "
            f"{changed}/{len(self.validation_games)}"
        )

        return changed

    def eval_version(
        self,
        version: str,
        memory: list[MemoryItem],
    ) -> list[dict]:
        result = []

        for i, game in enumerate(
            self.validation_games
        ):
            trace = self.run_episode(
                game,
                memory,
                (
                    self.home
                    / "validation"
                    / version
                    / f"episode_{i:03d}.json"
                ),
            )

            result.append({
                "game_id": game.id,
                "success": bool(trace.success),
                "decision_steps": len(trace.steps),
            })

            print(
                f"[validation] {version} "
                f"{i + 1}/{len(self.validation_games)} "
                f"success={trace.success}",
                flush=True,
            )

        return result

    def run_round(
        self,
        round_id: int,
        parent: str,
    ) -> str:
        rd = (
            self.home
            / "rounds"
            / f"round_{round_id:02d}"
        )

        plan_path = rd / "plan.json"

        plan = {
            "round": round_id,
            "parent": parent,
            "games": [
                game.id
                for game in self.learn_sets[
                    round_id - 1
                ]
            ],
        }

        if plan_path.exists():
            if load_json(plan_path) != plan:
                raise RuntimeError(
                    "Round plan changed"
                )

        else:
            atomic_json(plan_path, plan)

        decision_path = rd / "decision.json"

        if decision_path.exists():
            previous = load_json(
                decision_path
            )
            return previous["active_after"]

        parent_memory = self.load_version(
            parent
        )

        failures = self.do_learning(
            rd,
            self.learn_sets[round_id - 1],
            parent_memory,
        )

        candidate_memory = list(
            parent_memory
        )

        if self.mode != "fixed":
            for index, _, trace in failures:
                candidate_memory = self.process_failure(
                    rd,
                    index,
                    trace,
                    candidate_memory,
                )

        changed = (
            memory_hash(candidate_memory)
            != memory_hash(parent_memory)
        )

        if not changed:
            outcome = {
                "accept": False,
                "reason": "no_memory_change",
            }

            next_version = parent

        else:
            candidate_id = f"v{round_id:03d}"

            self.save_version(
                candidate_id,
                parent,
                round_id,
                candidate_memory,
            )

            if (
                self.retrieval_changes(
                    parent_memory,
                    candidate_memory,
                ) == 0
            ):
                outcome = {
                    "accept": False,
                    "reason": "same_validation_prompts",
                }

                next_version = parent

            else:
                base_eval = self.eval_version(
                    parent,
                    parent_memory,
                )

                cand_eval = self.eval_version(
                    candidate_id,
                    candidate_memory,
                )

                outcome = gate(
                    base_eval,
                    cand_eval,
                    max_regressions=self.protocol[
                        "selection_policy"
                    ]["max_regressions"],
                    min_step_saving=self.protocol[
                        "selection_policy"
                    ]["min_step_saving_if_tied"],
                )

                next_version = (
                    candidate_id
                    if outcome["accept"]
                    else parent
                )

        atomic_json(
            decision_path,
            {
                "round": round_id,
                "parent": parent,
                "candidate": (
                    f"v{round_id:03d}"
                    if changed
                    else None
                ),
                "active_after": next_version,
                "learning_failures": sum(
                    not Trajectory.load_json(
                        str(
                            rd
                            / "learning"
                            / f"episode_{i:03d}.json"
                        )
                    ).success
                    for i in range(
                        len(self.learn_sets[round_id - 1])
                    )
                ),
                "result": outcome,
            },
        )

        print(
            f"[round {round_id}] {outcome} "
            f"active={next_version}",
            flush=True,
        )

        return next_version

    def run(
        self,
        max_rounds: int | None = None,
    ):
        state = self.initialize()

        target = (
            self.rounds
            if max_rounds is None
            else min(max_rounds, self.rounds)
        )

        for round_id in range(
            state["completed_rounds"] + 1,
            target + 1,
        ):
            active = self.run_round(
                round_id,
                state["active_version"],
            )

            state = {
                **state,
                "active_version": active,
                "completed_rounds": round_id,
            }

            atomic_json(
                self.home / "state.json",
                state,
            )

        print(
            f"Current active version: "
            f"{state['active_version']}; "
            f"completed rounds={state['completed_rounds']}"
        )

    def status(self):
        state_path = self.home / "state.json"
        state = (
            load_json(state_path)
            if state_path.exists()
            else self.initialize()
        )

        print(
            json.dumps(
                state,
                ensure_ascii=False,
                indent=2,
            )
        )

        for path in sorted(
            (self.home / "rounds").glob(
                "round_*/decision.json"
            )
        ):
            row = load_json(path)

            print(
                f"Round {row['round']}: "
                f"{row['result']['reason']} "
                f"-> {row['active_after']}"
            )

    def rollback(self, version: str):
        state = self.initialize()

        path = (
            self.home
            / "versions"
            / f"{version}.json"
        )

        if not path.exists():
            raise ValueError(
                "Unknown version"
            )

        current = state["active_version"]
        ancestor = current
        ancestors = set()

        while ancestor:
            ancestors.add(ancestor)

            ancestor = load_json(
                self.home
                / "versions"
                / f"{ancestor}.json"
            )["parent"]

        if version not in ancestors:
            raise ValueError(
                "Rollback target is not an accepted ancestor"
            )

        if version == current:
            print(f"Version {version} is already active")
            return

        pending = (
            self.home
            / "rounds"
            / f"round_{state['completed_rounds'] + 1:02d}"
            / "decision.json"
        )

        if pending.exists():
            raise RuntimeError(
                "A completed round decision is pending state recovery; "
                "resume the run before rollback"
            )

        events = state.setdefault("rollback_events", [])

        if not isinstance(events, list):
            raise ValueError("Invalid rollback event history")

        events.append({
            "after_round": state["completed_rounds"],
            "from": current,
            "to": version,
        })

        state["active_version"] = version

        atomic_json(
            self.home / "state.json",
            state,
        )

        print(
            f"Active version rolled back to {version}"
        )

