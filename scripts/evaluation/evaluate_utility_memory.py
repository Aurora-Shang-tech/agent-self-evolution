
"""Feedback-based memory retrieval with held-out Train evaluation."""

import argparse
import json
import os
import random
from pathlib import Path

from src.agents.react_agent import ReActAgent
from src.core.trajectory import Trajectory
from src.data.alfworld_data import (
    GameExample,
    SUPPORTED_TASK_TYPES,
    default_data_root,
    discover_games,
)
from src.data.manifest import load_manifest
from src.envs.environment import ALFWorldEnvironment
from src.llm.client import LLMClient
from src.memory.utility import UtilityTracker
from src.memory.utility_store import HybridUtilityStore

from scripts.evaluation.evaluate_memory_evolution import (
    build_record,
    build_summary,
    file_hash,
    load_store,
    memory_file,
    paired_comparison,
    save_json,
    save_trajectory,
)


MAX_STEPS = 50
TOP_K = 3

TRAIN_MANIFEST = Path(
    "outputs/manifests/train_seed42_1000.json"
)

OUTPUT_ROOT = Path(
    "outputs/memory_utility/feedback_v1"
)

SOURCE_PATHS = [
    Path("src/agents/react_agent.py"),
    Path("src/core/protocol.py"),
    Path("src/core/trajectory.py"),
    Path("src/envs/environment.py"),
    Path("src/llm/client.py"),
    Path("src/memory/store.py"),
    Path("src/memory/utility.py"),
    Path("src/memory/utility_store.py"),
    Path("scripts/evaluation/evaluate_memory_evolution.py"),
    Path(__file__),
]


def source_hashes():
    return {
        str(p): file_hash(p)
        for p in SOURCE_PATHS
    }


def to_record(game):
    return {
        "id": game.id,
        "split": game.split,
        "game_file": str(game.game_file),
        "task_type": game.task_type,
    }


def from_record(record):
    return GameExample(
        id=record["id"],
        split=record["split"],
        game_file=Path(record["game_file"]),
        task_type=record["task_type"],
    )


def build_plan(model, seed, memory_path, data_root):
    if not TRAIN_MANIFEST.is_file():
        raise FileNotFoundError(TRAIN_MANIFEST)

    used = load_manifest(TRAIN_MANIFEST)

    if (
        len(used) != 1000
        or len({g.id for g in used}) != 1000
        or any(g.split != "train" for g in used)
    ):
        raise RuntimeError(
            "Expected 1000 distinct memory-construction Train games"
        )

    excluded = {g.id for g in used}
    games = discover_games(data_root, "train")

    rng = random.Random(seed)
    learning = []
    evaluation = []

    for task_type in sorted(SUPPORTED_TASK_TYPES):
        candidates = sorted(
            (
                g for g in games
                if g.task_type == task_type
                and g.id not in excluded
            ),
            key=lambda g: g.id,
        )

        rng.shuffle(candidates)

        if len(candidates) < 4:
            raise RuntimeError(
                f"Not enough held-out Train games of type {task_type}"
            )

        learning.extend(candidates[:2])
        evaluation.extend(candidates[2:4])

    rng.shuffle(learning)
    rng.shuffle(evaluation)

    if {g.id for g in learning} & {g.id for g in evaluation}:
        raise RuntimeError("Learning/evaluation games overlap")

    return {
        "schema": "utility_feedback_v1",
        "model": model,
        "seed": seed,
        "max_steps": MAX_STEPS,
        "top_k": TOP_K,
        "pool_size": 8,
        "utility_weight": 0.35,
        "alpha": 0.3,
        "state_tracking": False,
        "memory_path": str(memory_path),
        "memory_hash": file_hash(memory_path),
        "construction_manifest": str(TRAIN_MANIFEST),
        "construction_manifest_hash": file_hash(TRAIN_MANIFEST),
        "source_hashes": source_hashes(),
        "learning_games": [to_record(g) for g in learning],
        "evaluation_games": [to_record(g) for g in evaluation],
    }


def get_plan(directory, model, seed, memory_path, data_root):
    path = directory / "experiment_plan.json"

    if path.exists():
        plan = json.loads(
            path.read_text(encoding="utf-8")
        )

        if (
            plan["model"] != model
            or plan["seed"] != seed
            or plan["memory_hash"] != file_hash(memory_path)
            or plan["source_hashes"] != source_hashes()
            or plan["construction_manifest_hash"]
            != file_hash(TRAIN_MANIFEST)
        ):
            raise RuntimeError(
                "Frozen experiment inputs changed; "
                "choose a new --seed"
            )

        return plan

    if directory.exists() and list(
        directory.rglob("episode_*.json")
    ):
        raise RuntimeError(
            "Found trajectories without an experiment plan"
        )

    plan = build_plan(
        model,
        seed,
        memory_path,
        data_root,
    )

    save_json(path, plan)
    return plan


def run_agent(game, store, llm):
    env = ALFWorldEnvironment(
        str(game.game_file),
        max_steps=MAX_STEPS,
    )

    agent = ReActAgent(
        llm=llm,
        max_steps=MAX_STEPS,
        experience_store=store,
        experience_top_k=TOP_K,
        use_state_tracking=False,
    )

    try:
        return agent.run(env)
    finally:
        env.close()


def read_or_run(
    game,
    store,
    llm,
    trajectory_path,
    metadata_path,
):
    if trajectory_path.is_file():
        trace = Trajectory.load_json(
            str(trajectory_path)
        )

        created = False
        selected = store.rank_indices(
            trace.task,
            TOP_K,
        )

    else:
        trace = run_agent(
            game,
            store,
            llm,
        )

        selected = list(store.last_indices)

        if store.last_query != trace.task:
            raise RuntimeError(
                "Memory retrieval was not recorded for this run"
            )

        if selected != store.rank_indices(trace.task, TOP_K):
            raise RuntimeError(
                "Retrieval changed unexpectedly during the episode"
            )

        save_trajectory(
            trajectory_path,
            trace,
        )

        created = True

    record = build_record(
        0,
        game,
        trace,
    )

    metadata = {
        "game_id": game.id,
        "task": trace.task,
        "selected_indices": selected,
    }

    if metadata_path.is_file():
        old = json.loads(
            metadata_path.read_text(encoding="utf-8")
        )

        if old != metadata:
            raise RuntimeError(
                f"Retrieval provenance mismatch: {metadata_path}"
            )
    else:
        save_json(metadata_path, metadata)

    return record, selected, created


def learn(directory, plan, experiences, max_new):
    games = [
        from_record(x)
        for x in plan["learning_games"]
    ]

    tracker = UtilityTracker(
        len(experiences),
        alpha=plan["alpha"],
    )

    store = HybridUtilityStore(
        experiences,
        tracker=tracker,
        weight=plan["utility_weight"],
        pool_size=plan["pool_size"],
    )

    llm = None
    records = []
    created_count = 0

    for index, game in enumerate(games):
        trace_path = (
            directory
            / "learn"
            / f"episode_{index:03d}.json"
        )

        meta_path = (
            directory
            / "learn"
            / f"retrieval_{index:03d}.json"
        )

        if (
            not trace_path.is_file()
            and created_count >= max_new
        ):
            break

        if not trace_path.is_file() and llm is None:
            llm = LLMClient()

        record, indices, created = read_or_run(
            game,
            store,
            llm,
            trace_path,
            meta_path,
        )

        record["episode"] = index
        record["retrieved_indices"] = indices

        tracker.observe(
            indices,
            record["success"],
        )

        records.append(record)
        created_count += int(created)

        print(
            f"[LEARN] {index + 1}/{len(games)} "
            f"success={record['success']} "
            f"steps={record['decision_steps']} "
            f"ids={indices} "
            f"{'NEW' if created else 'CACHE'}",
            flush=True,
        )

    summary = build_summary(
        records,
        len(games),
    )

    save_json(
        directory / "learn_summary.json",
        summary,
    )

    if len(records) == len(games):
        snapshot = {
            "memory_hash": plan["memory_hash"],
            "plan_seed": plan["seed"],
            "tracker": tracker.dump(),
        }

        path = directory / "frozen_utility.json"

        if (
            path.exists()
            and json.loads(
                path.read_text(encoding="utf-8")
            ) != snapshot
        ):
            raise RuntimeError(
                "Frozen utility changed unexpectedly"
            )

        save_json(path, snapshot)

        print(
            f"Learning complete. Frozen utility: {path}"
        )

        for index, state in enumerate(tracker.stats):
            if state.uses:
                print(
                    f"  lesson {index:02d}: "
                    f"q={state.q:+.3f} "
                    f"uses={state.uses} "
                    f"success={state.successes} "
                    f"fail={state.failures}"
                )

    else:
        print(
            f"Learning: {len(records)}/{len(games)} "
            "complete; no evaluation snapshot yet"
        )


def evaluate(directory, plan, experiences, max_new):
    path = directory / "frozen_utility.json"

    if not path.exists():
        raise RuntimeError(
            "Finish learning before evaluation"
        )

    snapshot = json.loads(
        path.read_text(encoding="utf-8")
    )

    if (
        snapshot["memory_hash"] != plan["memory_hash"]
        or snapshot["plan_seed"] != plan["seed"]
    ):
        raise RuntimeError(
            "Frozen utility does not match the experiment"
        )

    tracker = UtilityTracker.load(
        snapshot["tracker"]
    )

    stores = {
        "hybrid": HybridUtilityStore(
            experiences,
            pool_size=plan["pool_size"],
        ),
        "utility": HybridUtilityStore(
            experiences,
            tracker=tracker,
            pool_size=plan["pool_size"],
            weight=plan["utility_weight"],
        ),
    }

    games = [
        from_record(x)
        for x in plan["evaluation_games"]
    ]

    rows = {
        "hybrid": [],
        "utility": [],
    }

    llm = None
    new_pairs = 0
    changed_retrieval = 0

    for index, game in enumerate(games):
        paths = {
            arm: (
                directory
                / "eval"
                / arm
                / f"episode_{index:03d}.json"
            )
            for arm in stores
        }

        missing_before = any(
            not p.exists()
            for p in paths.values()
        )

        if missing_before and new_pairs >= max_new:
            break

        pair_indices = {}

        for arm, store in stores.items():
            if not paths[arm].exists() and llm is None:
                llm = LLMClient()

            record, selected, created = read_or_run(
                game,
                store,
                llm,
                paths[arm],
                (
                    directory
                    / "eval"
                    / arm
                    / f"retrieval_{index:03d}.json"
                ),
            )

            record["episode"] = index
            record["retrieved_indices"] = selected

            rows[arm].append(record)
            pair_indices[arm] = selected

            print(
                f"[EVAL] {index + 1}/{len(games)} "
                f"{arm} "
                f"success={record['success']} "
                f"steps={record['decision_steps']} "
                f"ids={selected} "
                f"{'NEW' if created else 'CACHE'}",
                flush=True,
            )

        changed_retrieval += int(
            pair_indices["hybrid"]
            != pair_indices["utility"]
        )

        new_pairs += int(missing_before)

        summaries = {
            arm: build_summary(
                rows[arm],
                len(games),
            )
            for arm in stores
        }

        comparison = paired_comparison(
            "hybrid",
            summaries["hybrid"],
            "utility",
            summaries["utility"],
        )

        save_json(
            directory / "evaluation_summary.json",
            {
                "finished": all(
                    s["finished"]
                    for s in summaries.values()
                ),
                "completed_pairs": len(rows["hybrid"]),
                "retrieval_changed_pairs": changed_retrieval,
                "arms": summaries,
                "comparison": comparison,
            },
        )

    if rows["hybrid"]:
        result = paired_comparison(
            "hybrid",
            build_summary(
                rows["hybrid"],
                len(games),
            ),
            "utility",
            build_summary(
                rows["utility"],
                len(games),
            ),
        )

        print(
            f"[PAIRED] "
            f"+{result['improved']}/"
            f"-{result['regressed']} "
            f"net={result['net']:+d} "
            f"avg_steps="
            f"{result['average_steps_change']:+.2f} "
            f"p={result['mcnemar_exact_p']:.6f} "
            f"retrieval_changed="
            f"{changed_retrieval}/{len(rows['hybrid'])}"
        )
    else:
        print("No completed evaluation pairs")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "--stage",
        choices=(
            "plan",
            "learn",
            "eval",
            "report",
        ),
        default="plan",
    )

    parser.add_argument(
        "--model",
        default=os.getenv(
            "OPENAI_MODEL",
            "qwen3.8-chat",
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=20261110,
    )

    parser.add_argument(
        "--data-root",
        type=Path,
        default=default_data_root(),
    )

    parser.add_argument(
        "--max-new",
        type=int,
        default=0,
        help=(
            "Max new learning episodes or evaluation pairs; "
            "default 0"
        ),
    )

    args = parser.parse_args()

    if args.max_new < 0:
        parser.error(
            "--max-new must be nonnegative"
        )

    if (
        args.max_new
        and os.getenv("OPENAI_MODEL") != args.model
    ):
        raise RuntimeError(
            "OPENAI_MODEL must match --model for actual execution"
        )

    memory_path = memory_file(
        "direct_update",
        args.model,
    )

    experiences = load_store(
        "direct_update",
        memory_path,
    ).experiences

    if len(experiences) != 18:
        raise RuntimeError(
            "Expected the frozen Direct Update 18 memories"
        )

    directory = (
        OUTPUT_ROOT
        / args.model.replace("/", "_")
        / f"seed_{args.seed}"
    )

    plan = get_plan(
        directory,
        args.model,
        args.seed,
        memory_path,
        args.data_root,
    )

    print(
        f"Frozen plan: "
        f"{directory / 'experiment_plan.json'}"
    )

    print(
        f"Learning games: {len(plan['learning_games'])}; "
        f"evaluation games: {len(plan['evaluation_games'])}"
    )

    if args.stage == "plan":
        print(
            "Preflight only. No LLM calls."
        )

    elif args.stage == "learn":
        learn(
            directory,
            plan,
            experiences,
            args.max_new,
        )

    elif args.stage == "eval":
        evaluate(
            directory,
            plan,
            experiences,
            args.max_new,
        )

    else:
        path = (
            directory
            / "evaluation_summary.json"
        )

        if not path.exists():
            print(
                "No evaluation results yet"
            )
        else:
            result = json.loads(
                path.read_text(encoding="utf-8")
            )

            print(
                json.dumps(
                    {
                        "finished": result["finished"],
                        "completed_pairs": (
                            result["completed_pairs"]
                        ),
                        "retrieval_changed_pairs": (
                            result["retrieval_changed_pairs"]
                        ),
                        "comparison": result["comparison"],
                    },
                    indent=2,
                )
            )


if __name__ == "__main__":
    main()

