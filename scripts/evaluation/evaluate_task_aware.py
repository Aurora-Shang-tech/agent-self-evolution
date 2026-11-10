
import argparse
import hashlib
import json
import os
import random
from dataclasses import asdict
from pathlib import Path

from src.agents.react_agent import ReActAgent
from src.core.trajectory import Trajectory
from src.data.manifest import load_manifest
from src.envs.environment import ALFWorldEnvironment
from src.llm.client import LLMClient
from src.memory.task_aware import TaskAwareExperienceStore

from scripts.evaluation.evaluate_memory_evolution import (
    SPLIT_SIZES,
    MAX_STEPS,
    TOP_K,
    build_record,
    build_summary,
    file_hash,
    load_store,
    memory_file,
    paired_comparison,
    save_json,
    save_trajectory,
)


METHODS = (
    "original_18",
    "direct_update",
)

OUTPUT_ROOT = Path(
    "outputs/task_aware/evaluations"
)

BASELINE_ROOT = Path(
    "outputs/memory_retrieval/evaluations"
)

SOURCE_FILES = (
    "src/agents/react_agent.py",
    "src/core/trajectory.py",
    "src/core/protocol.py",
    "src/envs/environment.py",
    "src/llm/client.py",
    "src/memory/experience.py",
    "src/memory/store.py",
    "src/memory/task_aware.py",
    "scripts/evaluation/evaluate_memory_evolution.py",
    "scripts/evaluation/evaluate_task_aware.py",
)


def load_memories(method: str, model: str):
    source_method = (
        "original_130"
        if method == "original_18"
        else method
    )

    path = memory_file(source_method, model)
    store = load_store(source_method, path)

    selected_indices = None

    if method == "original_18":
        indices = list(
            range(len(store.experiences))
        )

        random.Random(42).shuffle(indices)

        selected_indices = sorted(
            indices[:18]
        )

        memories = [
            store.experiences[index]
            for index in selected_indices
        ]
    else:
        memories = list(store.experiences)

    if len(memories) != 18:
        raise RuntimeError(
            f"Expected 18 memories for {method}, "
            f"got {len(memories)}"
        )

    fingerprint_input = {
        "memory_file_hash": file_hash(path),
        "selected_indices": selected_indices,
        "memories": [
            asdict(memory)
            for memory in memories
        ],
    }

    fingerprint = hashlib.sha256(
        json.dumps(
            fingerprint_input,
            sort_keys=True,
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()

    return memories, path, fingerprint


def get_output_dir(model, split, method):
    return (
        OUTPUT_ROOT
        / model.replace("/", "_")
        / split
        / method
    )


def validate_config(directory, config):
    path = directory / "experiment_config.json"

    if path.exists():
        old = json.loads(
            path.read_text(encoding="utf-8")
        )

        if old != config:
            raise RuntimeError(
                "Experiment configuration changed. "
                "Use a new output directory rather "
                f"than reusing old trajectories: {directory}"
            )

        return

    if list(directory.glob("episode_*.json")):
        raise RuntimeError(
            "Existing episodes without experiment "
            f"configuration: {directory}"
        )

    save_json(path, config)


def evaluate(
    method,
    model,
    split,
    games,
    max_new,
    llm,
):
    memories, memory_path, fingerprint = (
        load_memories(method, model)
    )

    store = TaskAwareExperienceStore(
        memories,
        pool_size=8,
    )

    output_dir = get_output_dir(
        model,
        split,
        method,
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest_path = Path(
        f"outputs/manifests/"
        f"{split}_seed42_{len(games)}.json"
    )

    config = {
        "model": model,
        "split": split,
        "method": method,
        "retrieval": "task_aware_v2",
        "state_tracking": False,
        "top_k": TOP_K,
        "pool_size": 8,
        "max_steps": MAX_STEPS,
        "memory_path": str(memory_path),
        "memory_fingerprint": fingerprint,
        "manifest_hash": file_hash(
            manifest_path
        ),
        "source_hashes": {
            name: file_hash(Path(name))
            for name in SOURCE_FILES
        },
    }

    validate_config(
        output_dir,
        config,
    )

    records = []
    new_count = 0

    print("\n" + "=" * 65)
    print(f"Method: {method}")
    print(f"Split: {split}")
    print("Retrieval: task_aware_v2")
    print("State tracking: OFF")
    print(f"Memory size: {len(store)}")
    print(f"Max new episodes: {max_new}")
    print(f"Output: {output_dir}")
    print("=" * 65)

    for index, game in enumerate(games):
        episode_path = (
            output_dir
            / f"episode_{index:03d}.json"
        )

        if episode_path.exists():
            trajectory = Trajectory.load_json(
                str(episode_path)
            )
        else:
            if new_count >= max_new:
                continue

            env = ALFWorldEnvironment(
                gamefile=str(game.game_file),
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
                trajectory = agent.run(env)
            finally:
                env.close()

            save_trajectory(
                episode_path,
                trajectory,
            )

            new_count += 1

        record = build_record(
            index,
            game,
            trajectory,
        )

        retrieval_detail = store.explain(
            record["task"],
            top_k=TOP_K,
        )

        record["retrieved_memory_indices"] = [
            item["index"]
            for item in retrieval_detail["selected"]
        ]

        records.append(record)

        summary = build_summary(
            records,
            len(games),
        )

        summary.update({
            "method": method,
            "model": model,
            "split": split,
            "retrieval": "task_aware_v2",
            "state_tracking": False,
            "memory_size": len(store),
        })

        save_json(
            output_dir / "summary.json",
            summary,
        )

        print(
            f"[{method}][{split}] "
            f"episode={index + 1}/{len(games)} "
            f"success={record['success']} "
            f"steps={record['decision_steps']}",
            flush=True,
        )

    summary = build_summary(
        records,
        len(games),
    )

    summary.update({
        "method": method,
        "model": model,
        "split": split,
        "retrieval": "task_aware_v2",
        "state_tracking": False,
        "memory_size": len(store),
    })

    save_json(
        output_dir / "summary.json",
        summary,
    )

    print(
        f"\nCompleted: "
        f"{summary['completed']}/{summary['total']}"
    )
    print(
        f"Successes: {summary['successes']}"
    )

    if summary["average_decision_steps"] is not None:
        print(
            f"Average steps: "
            f"{summary['average_decision_steps']:.2f}"
        )

    return summary


def compare_with_hybrid(method, model, split, result):
    if not result["finished"]:
        print(
            f"[{method}] Incomplete evaluation. "
            "Skipping paired comparison."
        )
        return

    baseline_path = (
        BASELINE_ROOT
        / model.replace("/", "_")
        / split
        / method
        / "hybrid"
        / "summary.json"
    )

    if not baseline_path.is_file():
        raise FileNotFoundError(
            baseline_path
        )

    baseline = json.loads(
        baseline_path.read_text(encoding="utf-8")
    )

    if not baseline["finished"]:
        raise RuntimeError(
            f"Baseline is incomplete: {baseline_path}"
        )

    if (
        baseline.get("model") != model
        or baseline.get("split") != split
        or baseline.get("memory_size") != 18
    ):
        raise RuntimeError(
            f"Baseline configuration mismatch: "
            f"{baseline_path}"
        )

    row = paired_comparison(
        f"{method}:hybrid",
        baseline,
        f"{method}:task_aware_v2",
        result,
    )

    output_dir = get_output_dir(
        model,
        split,
        method,
    )

    save_json(
        output_dir / "comparison.json",
        row,
    )

    print("\nPAIRED COMPARISON")
    print(
        f"Hybrid -> TaskAware: "
        f"+{row['improved']}/-{row['regressed']}, "
        f"net={row['net']:+d}, "
        f"steps={row['average_steps_change']:+.2f}, "
        f"p={row['mcnemar_exact_p']:.6f}"
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--split",
        required=True,
        choices=SPLIT_SIZES,
    )

    parser.add_argument(
        "--methods",
        nargs="+",
        choices=METHODS,
        default=list(METHODS),
    )

    parser.add_argument(
        "--max-new",
        type=int,
        default=0,
    )

    args = parser.parse_args()

    if args.max_new < 0:
        parser.error(
            "--max-new must be >= 0"
        )

    model = os.getenv("OPENAI_MODEL")

    if not model:
        raise RuntimeError(
            "OPENAI_MODEL is not set"
        )

    size = SPLIT_SIZES[args.split]

    manifest_path = Path(
        f"outputs/manifests/"
        f"{args.split}_seed42_{size}.json"
    )

    games = load_manifest(
        manifest_path
    )

    if len(games) != size:
        raise RuntimeError(
            f"Expected {size} tasks, got {len(games)}"
        )

    if args.max_new == 0:
        print(
            "Preflight only. No LLM calls."
        )

        for method in args.methods:
            memories, path, _ = load_memories(
                method, model
            )

            print(
                f"{method}: {len(memories)} memories "
                f"from {path}"
            )

        return

    llm = LLMClient()

    for method in dict.fromkeys(args.methods):
        result = evaluate(
            method=method,
            model=model,
            split=args.split,
            games=games,
            max_new=args.max_new,
            llm=llm,
        )

        compare_with_hybrid(
            method,
            model,
            args.split,
            result,
        )


if __name__ == "__main__":
    main()

