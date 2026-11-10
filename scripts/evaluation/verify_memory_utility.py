
"""Auditable single-lesson ablation on Train games excluded from memory construction.

Without --max-new, only builds a frozen plan; never calls the LLM.
"""

import argparse
import hashlib
import json
import os
import random
from pathlib import Path
from math import comb

from src.agents.react_agent import ReActAgent
from src.core.trajectory import Trajectory
from src.data.alfworld_data import (
    default_data_root,
    discover_games,
    SUPPORTED_TASK_TYPES,
)
from src.data.manifest import load_manifest
from src.envs.environment import ALFWorldEnvironment
from src.llm.client import LLMClient

from scripts.evaluation.evaluate_memory_evolution import (
    build_record,
    file_hash,
    load_store,
    memory_file,
    save_json,
    save_trajectory,
)


MODEL_DEFAULT = "qwen3.8-chat"

TRAIN_MANIFEST = Path(
    "outputs/manifests/train_seed42_1000.json"
)

ROOT = Path(
    "outputs/memory_utility/holdout_train_v1"
)

MAX_STEPS = 50

SOURCE_FILES = (
    Path(__file__),
    Path("src/agents/react_agent.py"),
    Path("src/core/protocol.py"),
    Path("src/core/trajectory.py"),
    Path("src/envs/environment.py"),
    Path("src/llm/client.py"),
    Path("src/memory/experience.py"),
    Path("src/memory/store.py"),
    Path(
        "scripts/evaluation/"
        "evaluate_memory_evolution.py"
    ),
)


class OneLessonStore:
    """Expose exactly one lesson to ReActAgent."""

    def __init__(self, experience):
        self.experience = experience

    def retrieve(
        self,
        task: str,
        top_k: int = 3,
    ):
        if top_k < 1:
            raise ValueError(
                "top_k must be >= 1"
            )

        return [self.experience]


def exact_mcnemar(
    improved: int,
    regressed: int,
) -> float:

    n = improved + regressed

    if not n:
        return 1.0

    k = min(
        improved,
        regressed,
    )

    probability = (
        2.0
        * sum(
            comb(n, i)
            for i in range(k + 1)
        )
        / (2 ** n)
    )

    return min(
        1.0,
        probability,
    )


def paired_stats(
    records: list[dict],
) -> dict:

    if not records:
        return {
            "completed_pairs": 0,
        }

    controls = [
        row["control"]
        for row in records
    ]

    treatments = [
        row["with_lesson"]
        for row in records
    ]

    improved = sum(
        not a["success"] and b["success"]
        for a, b in zip(
            controls,
            treatments,
        )
    )

    regressed = sum(
        a["success"] and not b["success"]
        for a, b in zip(
            controls,
            treatments,
        )
    )

    count = len(records)

    return {
        "completed_pairs": count,

        "control_successes": sum(
            int(row["success"])
            for row in controls
        ),

        "with_lesson_successes": sum(
            int(row["success"])
            for row in treatments
        ),

        "improved": improved,
        "regressed": regressed,
        "net": improved - regressed,

        "exact_mcnemar_p": exact_mcnemar(
            improved,
            regressed,
        ),

        "control_avg_steps": (
            sum(
                row["decision_steps"]
                for row in controls
            )
            / count
        ),

        "with_lesson_avg_steps": (
            sum(
                row["decision_steps"]
                for row in treatments
            )
            / count
        ),

        "avg_step_delta": (
            sum(
                b["decision_steps"]
                - a["decision_steps"]
                for a, b in zip(
                    controls,
                    treatments,
                )
            )
            / count
        ),
    }


def read_and_validate(
    path: Path,
    config: dict,
):

    if path.exists():
        old = json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )

        if old != config:
            raise RuntimeError(
                "Frozen configuration changed: "
                f"{path}. Use a new experiment "
                "directory instead of reusing "
                "existing trajectories."
            )

        return

    if list(
        path.parent.glob(
            "episode_*.json"
        )
    ):
        raise RuntimeError(
            "Existing trajectories without "
            f"configuration: {path.parent}"
        )

    save_json(
        path,
        config,
    )


def frozen_selection(
    data_root: Path,
    task_type: str,
    seed: int,
    num_games: int,
):

    if not TRAIN_MANIFEST.is_file():
        raise FileNotFoundError(
            f"Training manifest missing: "
            f"{TRAIN_MANIFEST}"
        )

    used = load_manifest(
        TRAIN_MANIFEST
    )

    if (
        len(used) != 1000
        or any(
            game.split != "train"
            for game in used
        )
    ):
        raise RuntimeError(
            "Expected exactly 1000 Train "
            "games in memory-construction manifest"
        )

    used_ids = {
        game.id
        for game in used
    }

    if len(used_ids) != 1000:
        raise RuntimeError(
            "Duplicate game IDs in "
            "training manifest"
        )

    candidates = [
        game
        for game in discover_games(
            data_root,
            "train",
        )
        if (
            game.id not in used_ids
            and game.task_type == task_type
        )
    ]

    candidates.sort(
        key=lambda game: game.id
    )

    random.Random(seed).shuffle(
        candidates
    )

    if len(candidates) < num_games:
        raise RuntimeError(
            f"Only {len(candidates)} remaining "
            f"Train games of type {task_type}; "
            f"requested {num_games}."
        )

    return (
        candidates[:num_games],
        len(candidates),
    )


def run_one(
    game,
    llm,
    experience,
    arm: str,
    trajectory_file: Path,
):

    if trajectory_file.exists():
        return (
            Trajectory.load_json(
                str(trajectory_file)
            ),
            False,
        )

    env = ALFWorldEnvironment(
        str(game.game_file),
        max_steps=MAX_STEPS,
    )

    store = (
        None
        if arm == "control"
        else OneLessonStore(experience)
    )

    agent = ReActAgent(
        llm=llm,
        max_steps=MAX_STEPS,
        experience_store=store,
        experience_top_k=1,
        use_state_tracking=False,
    )

    try:
        trajectory = agent.run(
            env
        )
    finally:
        env.close()

    save_trajectory(
        trajectory_file,
        trajectory,
    )

    return trajectory, True


def main():

    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "--model",
        default=os.getenv(
            "OPENAI_MODEL",
            MODEL_DEFAULT,
        ),
    )

    parser.add_argument(
        "--list-lessons",
        action="store_true",
    )

    parser.add_argument(
        "--lesson-index",
        type=int,
    )

    parser.add_argument(
        "--task-type",
        choices=sorted(
            SUPPORTED_TASK_TYPES
        ),
    )

    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(
            os.getenv("ALFWORLD_DATA")
            or default_data_root()
        ),
    )

    parser.add_argument(
        "--num-games",
        type=int,
        default=6,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=20261110,
    )

    parser.add_argument(
        "--max-new",
        type=int,
        default=0,
    )

    args = parser.parse_args()

    if (
        args.num_games < 1
        or args.max_new < 0
    ):
        parser.error(
            "--num-games >= 1 and "
            "--max-new >= 0 required"
        )

    memory_path = memory_file(
        "direct_update",
        args.model,
    )

    source_store = load_store(
        "direct_update",
        memory_path,
    )

    experiences = (
        source_store.experiences
    )

    if args.list_lessons:
        print(
            "Direct Update memories:"
        )

        for index, experience in enumerate(
            experiences
        ):
            print(
                f"  [{index:02d}] "
                f"{experience.lesson}"
            )

            print(
                f"       source: "
                f"{experience.source_task}"
            )

        return

    if (
        args.lesson_index is None
        or args.task_type is None
    ):
        parser.error(
            "Specify --lesson-index and "
            "--task-type, or use --list-lessons"
        )

    if not (
        0 <= args.lesson_index
        < len(experiences)
    ):
        parser.error(
            "--lesson-index must be in "
            f"[0, {len(experiences) - 1}]"
        )

    experience = experiences[
        args.lesson_index
    ]

    games, n_candidates = (
        frozen_selection(
            args.data_root,
            args.task_type,
            args.seed,
            args.num_games,
        )
    )

    output_dir = (
        ROOT
        / args.model.replace("/", "_")
        / args.task_type
        / f"lesson_{args.lesson_index:02d}"
        / (
            f"seed_{args.seed}"
            f"_n_{args.num_games}"
        )
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    config = {
        "schema": (
            "single_lesson_ablation_v1"
        ),
        "model": args.model,
        "task_type": args.task_type,
        "lesson_index": (
            args.lesson_index
        ),
        "lesson": experience.lesson,
        "source_task": (
            experience.source_task
        ),
        "arms": [
            "control",
            "with_lesson",
        ],
        "max_steps": MAX_STEPS,
        "state_tracking": False,

        "memory_path": str(
            memory_path
        ),

        "memory_hash": file_hash(
            memory_path
        ),

        "train_manifest": str(
            TRAIN_MANIFEST
        ),

        "train_manifest_hash": (
            file_hash(TRAIN_MANIFEST)
        ),

        "eligible_holdout_count": (
            n_candidates
        ),

        "seed": args.seed,

        "games": [
            {
                "id": game.id,
                "game_file": str(
                    game.game_file
                ),
                "task_type": game.task_type,
            }
            for game in games
        ],

        "source_hashes": {
            str(path): file_hash(path)
            for path in SOURCE_FILES
        },
    }

    read_and_validate(
        output_dir
        / "experiment_config.json",
        config,
    )

    print(
        f"\nLesson [{args.lesson_index}] "
        f"{experience.lesson}"
    )

    print(
        f"Task type: {args.task_type}"
    )

    print(
        f"Eligible held-out Train games: "
        f"{n_candidates}"
    )

    print(
        f"Frozen pairs: {len(games)}"
    )

    print(
        f"Output: {output_dir}"
    )

    for index, game in enumerate(
        games
    ):
        print(
            f"  pair {index:02d}: "
            f"{game.id}"
        )

    if args.max_new == 0:
        print(
            "\nPREFLIGHT ONLY. "
            "No LLM calls or ALFWorld executions."
        )
        return

    if (
        os.getenv("OPENAI_MODEL")
        != args.model
    ):
        raise RuntimeError(
            "OPENAI_MODEL must equal --model "
            "before LLM execution."
        )

    llm = LLMClient()

    new_pairs = 0
    pairs = []

    for index, game in enumerate(
        games
    ):

        paths = {
            arm: (
                output_dir
                / (
                    f"episode_{index:03d}"
                    f"_{arm}.json"
                )
            )
            for arm in (
                "control",
                "with_lesson",
            )
        }

        any_missing = any(
            not path.is_file()
            for path in paths.values()
        )

        if (
            any_missing
            and new_pairs >= args.max_new
        ):
            continue

        if any_missing:
            new_pairs += 1

        traces = {}

        for arm in (
            "control",
            "with_lesson",
        ):

            traces[arm], created = (
                run_one(
                    game,
                    llm,
                    experience,
                    arm,
                    paths[arm],
                )
            )

            print(
                f"pair={index:02d} "
                f"arm={arm} "
                f"success="
                f"{traces[arm].success} "
                f"steps="
                f"{len(traces[arm].steps)} "
                f"{'NEW' if created else 'CACHE'}",
                flush=True,
            )

        if (
            not traces["control"].steps
            or not traces[
                "with_lesson"
            ].steps
        ):
            raise RuntimeError(
                f"Empty trajectory at pair "
                f"{index}"
            )

        if (
            traces["control"]
            .steps[0].observation
            != traces["with_lesson"]
            .steps[0].observation
        ):
            raise RuntimeError(
                "Initial observations differ "
                f"for game {game.id}"
            )

        pairs.append({
            "pair": index,
            "game_id": game.id,

            "control": build_record(
                index,
                game,
                traces["control"],
            ),

            "with_lesson": build_record(
                index,
                game,
                traces["with_lesson"],
            ),
        })

        save_json(
            output_dir / "summary.json",
            {
                "model": args.model,
                "lesson_index": (
                    args.lesson_index
                ),
                "task_type": args.task_type,
                "finished": (
                    len(pairs) == len(games)
                ),
                "total_pairs": len(games),
                "stats": paired_stats(
                    pairs
                ),
                "pairs": pairs,
            },
        )

    stats = paired_stats(
        pairs
    )

    print(
        f"\nCompleted pairs: "
        f"{stats['completed_pairs']}"
        f"/{len(games)}"
    )

    if stats["completed_pairs"]:
        print(
            "Control successes: "
            f"{stats['control_successes']}"
            f"/{stats['completed_pairs']}"
        )

        print(
            "With lesson: "
            f"{stats['with_lesson_successes']}"
            f"/{stats['completed_pairs']}"
        )

        print(
            "Improved/Regressed: "
            f"+{stats['improved']}"
            f"/-{stats['regressed']}"
        )

        print(
            "Mean steps delta: "
            f"{stats['avg_step_delta']:+.2f}"
        )

        print(
            "Exploratory exact McNemar p: "
            f"{stats['exact_mcnemar_p']:.6f}"
        )


if __name__ == "__main__":
    main()

