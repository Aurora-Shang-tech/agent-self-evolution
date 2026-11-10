
"""Manage the integrated Memory Evolution and Utility Retrieval pipeline."""

import argparse
import hashlib
import json
import os
import re
from pathlib import Path

from src.agents.react_agent import ReActAgent
from src.data.manifest import load_manifest
from src.envs.environment import ALFWorldEnvironment
from src.llm.client import LLMClient
from src.memory.pipeline import MemoryPipeline, write_json

from scripts.evaluation.evaluate_memory_evolution import (
    save_trajectory,
)


DEFAULT_SOURCE = Path(
    "outputs/memory_evolution/train_full_v1/working_memory.json"
)

DEFAULT_STATE = Path(
    "outputs/memory_pipeline/state.json"
)

DEFAULT_V1 = Path(
    "outputs/memory_utility/feedback_v1/qwen3.8-chat/"
    "seed_20261110/frozen_utility.json"
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--stage",
        choices=("sync", "inspect", "run"),
        default="sync",
    )

    parser.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_SOURCE,
    )

    parser.add_argument(
        "--state-file",
        type=Path,
        default=DEFAULT_STATE,
    )

    parser.add_argument(
        "--import-v1",
        action="store_true",
    )

    parser.add_argument(
        "--task",
        default="",
    )

    parser.add_argument(
        "--mode",
        choices=("hybrid", "utility"),
        default="utility",
    )

    parser.add_argument(
        "--manifest",
        type=Path,
    )

    parser.add_argument(
        "--episode",
        type=int,
    )

    parser.add_argument(
        "--state-tracking",
        action="store_true",
    )

    parser.add_argument(
        "--learn",
        action="store_true",
    )

    parser.add_argument(
        "--run-name",
        default="manual",
    )

    args = parser.parse_args()

    if not re.fullmatch(
        r"[a-zA-Z0-9][a-zA-Z0-9_.-]*",
        args.run_name,
    ):
        parser.error("Invalid --run-name")

    pipeline = MemoryPipeline(
        args.source,
        args.state_file,
        mode=args.mode,
        import_v1=(
            DEFAULT_V1
            if args.import_v1
            else None
        ),
        sync=args.stage == "sync",
    )

    if args.stage == "sync":
        print(
            f"Memory records: {len(pipeline.catalog)}"
        )

        print(
            f"ID reconciliation: {pipeline.sync_counts}"
        )

        print(
            f"Existing feedback events: "
            f"{len(pipeline.state['history'])}"
        )

        print(
            f"v1 imported: "
            f"{pipeline.state['imported_v1'] is not None}"
        )

        print(f"State: {args.state_file}")
        return

    if args.stage == "inspect":
        if not args.task.strip():
            parser.error(
                "--task is required for inspect"
            )

        for mode in ("hybrid", "utility"):
            ids = pipeline.rank_ids(
                args.task,
                mode=mode,
            )

            print(f"{mode}: {ids}")

            for memory_id in ids:
                index = pipeline.id_to_index[memory_id]
                stat = pipeline.tracker.stats[index]

                print(
                    f"  {memory_id} "
                    f"q={stat.q:+.3f} "
                    f"uses={stat.uses}: "
                    f"{pipeline.catalog[index]['lesson']}"
                )

        print("Inspect only. No LLM calls.")
        return

    if args.manifest is None or args.episode is None:
        parser.error(
            "--manifest and --episode are required for run"
        )

    if args.episode < 0:
        parser.error(
            "--episode must be nonnegative"
        )

    games = load_manifest(
        args.manifest
    )

    if args.episode >= len(games):
        parser.error(
            "Episode outside manifest"
        )

    game = games[args.episode]

    if args.learn:
        if game.split != "train":
            raise RuntimeError(
                "Feedback learning is restricted to Train games"
            )

        construction = load_manifest(
            Path(
                "outputs/manifests/train_seed42_1000.json"
            )
        )

        if game.id in {
            item.id
            for item in construction
        }:
            raise RuntimeError(
                "This game was already used to build the memory"
            )

    model = os.getenv(
        "OPENAI_MODEL"
    )

    if not model:
        raise RuntimeError(
            "OPENAI_MODEL is missing"
        )

    key = hashlib.sha256(
        game.id.encode()
    ).hexdigest()[:12]

    tracking = (
        "state_on"
        if args.state_tracking
        else "state_off"
    )

    output_dir = (
        Path("outputs/memory_pipeline/executions")
        / args.run_name
        / model.replace("/", "_")
        / args.mode
        / tracking
    )

    trace_path = (
        output_dir
        / f"episode_{args.episode:03d}_{key}.json"
    )

    info_path = (
        output_dir
        / f"episode_{args.episode:03d}_{key}_meta.json"
    )

    event_id = (
        f"{args.run_name}:"
        f"{model}:"
        f"{args.mode}:"
        f"{tracking}:"
        f"{game.id}"
    )

    if trace_path.exists() != info_path.exists():
        raise RuntimeError(
            "Incomplete existing run files"
        )

    if info_path.exists():
        info = json.loads(
            info_path.read_text(
                encoding="utf-8"
            )
        )

        if info["source_hash"] != pipeline.source_hash:
            raise RuntimeError(
                "Cached run belongs to a different memory snapshot"
            )

        if (
            info["game_id"] != game.id
            or info["model"] != model
        ):
            raise RuntimeError(
                "Cached run metadata mismatch"
            )

        print(f"CACHE: {trace_path}")

    else:
        env = ALFWorldEnvironment(
            str(game.game_file),
            max_steps=50,
        )

        agent = ReActAgent(
            llm=LLMClient(),
            max_steps=50,
            experience_store=pipeline,
            experience_top_k=3,
            use_state_tracking=args.state_tracking,
        )

        try:
            trajectory = agent.run(env)
        finally:
            env.close()

        info = {
            "game_id": game.id,
            "model": model,
            "source_hash": pipeline.source_hash,
            "task": trajectory.task,
            "success": trajectory.success,
            "decision_steps": len(trajectory.steps),
            "selected_ids": list(pipeline.last_ids),
        }

        save_trajectory(
            trace_path,
            trajectory,
        )

        write_json(
            info_path,
            info,
        )

        print(f"NEW: {trace_path}")

    if args.learn:
        applied = pipeline.record_feedback(
            event_id=event_id,
            task=info["task"],
            success=info["success"],
            decision_steps=info["decision_steps"],
            selected_ids=info["selected_ids"],
        )

        print(
            f"Feedback applied: {applied}"
        )

    else:
        print(
            "Frozen evaluation: no Q update"
        )

    print(
        f"Success={info['success']} "
        f"steps={info['decision_steps']} "
        f"lessons={info['selected_ids']}"
    )


if __name__ == "__main__":
    main()

