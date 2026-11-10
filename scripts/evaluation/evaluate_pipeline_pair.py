
"""Paired Hybrid vs Utility evaluation with a frozen MemoryPipeline + State."""

import argparse
import json
import os
from pathlib import Path

from scripts.evaluation.evaluate_memory_evolution import (
    build_record,
    build_summary,
    file_hash,
    paired_comparison,
    save_json,
    save_trajectory,
)
from src.agents.react_agent import ReActAgent
from src.core.trajectory import Trajectory
from src.data.alfworld_data import GameExample
from src.envs.environment import ALFWorldEnvironment
from src.llm.client import LLMClient
from src.memory.pipeline import MemoryPipeline


SOURCE = Path(
    "outputs/memory_evolution/train_full_v1/working_memory.json"
)

STATE = Path(
    "outputs/memory_pipeline/state.json"
)

CANDIDATES = Path(
    "outputs/memory_utility/contextual_v2/qwen3.8-chat/"
    "seed_20261112/experiment_plan.json"
)

ROOT = Path("outputs/memory_pipeline/paired_state_v1")

MAX_STEPS = 50
TOP_K = 3

CODE_PATHS = [
    Path(__file__),
    Path("src/memory/pipeline.py"),
    Path("src/memory/utility.py"),
    Path("src/memory/utility_store.py"),
    Path("src/memory/store.py"),
    Path("src/agents/react_agent.py"),
    Path("src/memory/state_tracker.py"),
    Path("src/core/protocol.py"),
    Path("src/core/trajectory.py"),
    Path("src/envs/environment.py"),
    Path("src/llm/client.py"),
    Path("scripts/evaluation/evaluate_memory_evolution.py"),
]


def read_json(path: Path) -> dict:
    return json.loads(
        path.read_text(encoding="utf-8")
    )


def fingerprints(source, state, candidates):
    paths = [source, state, candidates, *CODE_PATHS]
    return {
        str(path): file_hash(path)
        for path in paths
    }


def to_game(info):
    return GameExample(
        id=info["id"],
        split=info["split"],
        game_file=Path(info["game_file"]),
        task_type=info["task_type"],
    )


def previously_executed_ids():
    root = Path("outputs/memory_pipeline/executions")

    if not root.is_dir():
        return set()

    return {
        read_json(path)["game_id"]
        for path in root.rglob("episode_*_meta.json")
    }


def prepare(directory, model, source, state, candidates):
    current = fingerprints(source, state, candidates)
    plan_file = directory / "experiment_plan.json"

    if plan_file.is_file():
        plan = read_json(plan_file)

        if (
            plan["fingerprints"] != current
            or plan["model"] != model
        ):
            raise RuntimeError(
                "Frozen source/Q/code changed. "
                "Do not mix evaluation runs; use a fresh "
                "output directory for a new snapshot."
            )

        return plan

    if directory.is_dir() and list(
        directory.rglob("episode_*.json")
    ):
        raise RuntimeError(
            "Episode files found without a frozen plan"
        )

    old_plan = read_json(candidates)

    if (
        old_plan.get("model") != model
        or old_plan.get("schema") != "contextual_utility_v2"
    ):
        raise RuntimeError(
            "Unexpected candidate plan/model"
        )

    excluded = previously_executed_ids()

    ids = [
        game["id"]
        for game in old_plan["games"]
    ]

    if len(ids) != len(set(ids)):
        raise RuntimeError(
            "Duplicate candidate games"
        )

    pipeline = MemoryPipeline(
        source,
        state,
        mode="utility",
    )

    rows = []

    for position, info in enumerate(old_plan["games"]):
        game = to_game(info)

        if game.split != "train":
            raise RuntimeError(
                "Only held-out Train games allowed"
            )

        if game.id in excluded:
            print(
                f"[SKIP] candidate {position:02d}: "
                "previously executed"
            )
            continue

        env = ALFWorldEnvironment(
            str(game.game_file),
            max_steps=MAX_STEPS,
        )

        try:
            task = env.reset().task
        finally:
            env.close()

        hybrid = pipeline.rank_ids(
            task,
            mode="hybrid",
            top_k=TOP_K,
        )

        utility = pipeline.rank_ids(
            task,
            mode="utility",
            top_k=TOP_K,
        )

        changed = hybrid != utility

        rows.append({
            "candidate_index": position,
            "game": info,
            "task": task,
            "hybrid_ids": hybrid,
            "utility_ids": utility,
            "changed": changed,
        })

        status = (
            "CHANGED"
            if changed
            else "SAME"
        )

        print(
            f"[PLAN] {position:02d} {status}: "
            f"{hybrid} -> {utility}"
        )

    plan = {
        "schema": "pipeline_paired_state_v1",
        "model": model,
        "fingerprints": current,
        "max_steps": MAX_STEPS,
        "top_k": TOP_K,
        "state_tracking": True,
        "scope": "changed_retrieval_only",
        "rows": rows,
    }

    save_json(plan_file, plan)
    return plan


def check_frozen(plan, source, state, candidates):
    if plan["fingerprints"] != fingerprints(
        source,
        state,
        candidates,
    ):
        raise RuntimeError(
            "Source/Q/code changed during evaluation"
        )


def episode_file(directory, arm, index):
    return (
        directory
        / "eval"
        / arm
        / f"episode_{index:03d}.json"
    )


def run_or_load(
    directory,
    arm,
    row,
    model,
    source,
    state,
    llm,
):
    index = row["candidate_index"]
    game = to_game(row["game"])
    expected = row[f"{arm}_ids"]

    path = episode_file(directory, arm, index)

    meta_path = path.with_name(
        f"episode_{index:03d}_meta.json"
    )

    pipeline = MemoryPipeline(
        source,
        state,
        mode=arm,
    )

    if pipeline.rank_ids(
        row["task"],
        mode=arm,
        top_k=TOP_K,
    ) != expected:
        raise RuntimeError(
            "Memory ranking does not match "
            "the frozen plan"
        )

    if path.is_file():
        trace = Trajectory.load_json(
            str(path)
        )
        tag = "CACHE"

    else:
        if llm is None:
            raise RuntimeError(
                "LLM is required for a new episode"
            )

        env = ALFWorldEnvironment(
            str(game.game_file),
            max_steps=MAX_STEPS,
        )

        agent = ReActAgent(
            llm=llm,
            max_steps=MAX_STEPS,
            experience_store=pipeline,
            experience_top_k=TOP_K,
            use_state_tracking=True,
        )

        try:
            trace = agent.run(env)
        finally:
            env.close()

        if (
            pipeline.last_task != row["task"]
            or pipeline.last_ids != expected
        ):
            raise RuntimeError(
                "Agent used different memory "
                "than expected"
            )

        save_trajectory(
            path,
            trace,
        )

        tag = "NEW"

    record = build_record(
        index,
        game,
        trace,
    )

    if trace.task != row["task"]:
        raise RuntimeError(
            "Task changed from the frozen plan"
        )

    meta = {
        "model": model,
        "game_id": game.id,
        "arm": arm,
        "task": trace.task,
        "selected_ids": expected,
        "success": record["success"],
        "decision_steps": record["decision_steps"],
        "state_tracking": True,
    }

    if meta_path.is_file():
        if read_json(meta_path) != meta:
            raise RuntimeError(
                "Cached metadata conflict"
            )
    else:
        save_json(
            meta_path,
            meta,
        )

    record["selected_ids"] = expected

    print(
        f"[EVAL] game={index:02d} "
        f"{arm} {tag} "
        f"success={record['success']} "
        f"steps={record['decision_steps']}"
    )

    return record


def summarize(directory, plan):
    targets = [
        row
        for row in plan["rows"]
        if row["changed"]
    ]

    results = {
        "hybrid": [],
        "utility": [],
    }

    for row in targets:
        index = row["candidate_index"]
        records = {}

        for arm in results:
            path = episode_file(
                directory,
                arm,
                index,
            )

            meta_path = path.with_name(
                f"episode_{index:03d}_meta.json"
            )

            if (
                not path.is_file()
                or not meta_path.is_file()
            ):
                break

            meta = read_json(meta_path)
            game = to_game(row["game"])

            trace = Trajectory.load_json(
                str(path)
            )

            record = build_record(
                index,
                game,
                trace,
            )

            if (
                trace.task != row["task"]
                or meta["selected_ids"] != row[f"{arm}_ids"]
                or meta["game_id"] != game.id
                or meta["success"] != record["success"]
                or meta["decision_steps"] != record["decision_steps"]
                or meta["arm"] != arm
                or meta["model"] != plan["model"]
                or meta["state_tracking"] is not True
            ):
                raise RuntimeError(
                    "Invalid cached result"
                )

            record["selected_ids"] = meta["selected_ids"]
            records[arm] = record

        if len(records) != 2:
            break

        for arm in results:
            results[arm].append(
                records[arm]
            )

    summaries = {
        arm: build_summary(
            rows,
            len(targets),
        )
        for arm, rows in results.items()
    }

    comparison = (
        paired_comparison(
            "hybrid",
            summaries["hybrid"],
            "utility",
            summaries["utility"],
        )
        if results["hybrid"]
        else None
    )

    output = {
        "scope": "changed_retrieval_only",
        "finished": (
            bool(targets)
            and len(results["hybrid"]) == len(targets)
        ),
        "candidate_games": len(plan["rows"]),
        "changed_pairs": len(targets),
        "completed_pairs": len(results["hybrid"]),
        "arms": summaries,
        "comparison": comparison,
    }

    save_json(
        directory / "summary.json",
        output,
    )

    return output


def report(result):
    compact = {
        "scope": result["scope"],
        "finished": result["finished"],
        "candidate_games": result["candidate_games"],
        "changed_pairs": result["changed_pairs"],
        "completed_pairs": result["completed_pairs"],
        "hybrid_successes": (
            result["arms"]["hybrid"]["successes"]
        ),
        "utility_successes": (
            result["arms"]["utility"]["successes"]
        ),
        "comparison": result["comparison"],
    }

    print(
        json.dumps(
            compact,
            ensure_ascii=False,
            indent=2,
        )
    )


def main():
    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "--stage",
        choices=("plan", "eval", "report"),
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
        "--source",
        type=Path,
        default=SOURCE,
    )

    parser.add_argument(
        "--state-file",
        type=Path,
        default=STATE,
    )

    parser.add_argument(
        "--candidate-plan",
        type=Path,
        default=CANDIDATES,
    )

    parser.add_argument(
        "--max-new",
        type=int,
        default=0,
        help="Maximum new pairs, not episodes",
    )

    args = parser.parse_args()

    if args.max_new < 0:
        parser.error(
            "--max-new must be >= 0"
        )

    if (
        args.stage == "eval"
        and args.max_new
        and os.getenv("OPENAI_MODEL") != args.model
    ):
        raise RuntimeError(
            "OPENAI_MODEL must equal --model "
            "for model calls"
        )

    directory = (
        ROOT
        / args.model.replace("/", "_")
        / "v2_seed_20261112"
    )

    plan = prepare(
        directory,
        args.model,
        args.source,
        args.state_file,
        args.candidate_plan,
    )

    targets = [
        row
        for row in plan["rows"]
        if row["changed"]
    ]

    print(
        f"Frozen eligible games: {len(plan['rows'])}"
    )

    print(
        f"Changed Top-3 games: {len(targets)}"
    )

    print(
        f"Output: {directory}"
    )

    if args.stage == "plan":
        print(
            "PLAN ONLY: no LLM calls"
        )
        return

    if args.stage == "report":
        report(
            summarize(directory, plan)
        )
        return

    if not targets:
        print(
            "No changed retrievals; "
            "no LLM comparison is useful"
        )

        report(
            summarize(directory, plan)
        )
        return

    new_pairs = 0
    llm = None

    for row in targets:
        index = row["candidate_index"]

        missing = any(
            not episode_file(
                directory,
                arm,
                index,
            ).is_file()
            for arm in ("hybrid", "utility")
        )

        if (
            missing
            and new_pairs >= args.max_new
        ):
            break

        if missing:
            llm = llm or LLMClient()
            new_pairs += 1

        for arm in ("hybrid", "utility"):
            check_frozen(
                plan,
                args.source,
                args.state_file,
                args.candidate_plan,
            )

            run_or_load(
                directory,
                arm,
                row,
                args.model,
                args.source,
                args.state_file,
                llm,
            )

            check_frozen(
                plan,
                args.source,
                args.state_file,
                args.candidate_plan,
            )

        report(
            summarize(directory, plan)
        )

    if not new_pairs:
        print(
            "No new episodes executed"
        )

    report(
        summarize(directory, plan)
    )


if __name__ == "__main__":
    main()

