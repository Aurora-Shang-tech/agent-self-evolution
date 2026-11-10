
"""离线比较 Hybrid BM25 与 Task-Aware Retrieval，不执行 ALFWorld 或 LLM。"""

import argparse
import json
import os
import random
from collections import Counter
from pathlib import Path

from scripts.evaluation.evaluate_memory_evolution import (
    load_store,
    memory_file,
)
from src.memory.task_aware import (
    TaskAwareExperienceStore,
)


ROOT = Path(
    "outputs/memory_retrieval/evaluations"
)

OUTPUT = Path(
    "outputs/memory_retrieval/task_aware_diagnostics"
)


def load_experiences(method, model):
    source = (
        "original_130"
        if method == "original_18"
        else method
    )

    store = load_store(
        source,
        memory_file(source, model),
    )

    if method == "original_18":
        indices = list(
            range(len(store.experiences))
        )

        random.Random(42).shuffle(indices)

        return [
            store.experiences[i]
            for i in sorted(indices[:18])
        ]

    return store.experiences


def read_summary(path):
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing completed evaluation: {path}"
        )

    result = json.loads(
        path.read_text(encoding="utf-8")
    )

    if (
        not result.get("finished")
        or len(result.get("episodes", []))
        != result.get("total")
    ):
        raise RuntimeError(
            f"Evaluation is incomplete: {path}"
        )

    return result


def save_json(path, obj):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temp = path.with_name(
        path.name + ".tmp"
    )

    temp.write_text(
        json.dumps(
            obj,
            ensure_ascii=False,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )

    os.replace(temp, path)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        default=os.getenv(
            "OPENAI_MODEL",
            "qwen3.8-chat",
        ),
    )

    parser.add_argument(
        "--splits",
        nargs="+",
        choices=(
            "valid_seen",
            "valid_unseen",
        ),
        default=[
            "valid_seen",
            "valid_unseen",
        ],
    )

    parser.add_argument(
        "--methods",
        nargs="+",
        choices=(
            "original_18",
            "direct_update",
            "rule_strength_voting",
            "source_aware_revision",
        ),
        default=[
            "original_18",
            "direct_update",
        ],
    )

    parser.add_argument(
        "--pool-size",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
    )

    args = parser.parse_args()

    if (
        args.pool_size < args.top_k
        or args.top_k < 1
    ):
        parser.error(
            "Require pool-size >= top-k >= 1"
        )

    model_key = args.model.replace(
        "/",
        "_",
    )

    stores = {}

    for method in dict.fromkeys(
        args.methods
    ):
        experiences = load_experiences(
            method,
            args.model,
        )

        stores[method] = (
            TaskAwareExperienceStore(
                experiences,
                args.pool_size,
            )
        )

    for split in args.splits:
        summaries = {}

        for method in stores:
            path = (
                ROOT
                / model_key
                / split
                / method
                / "hybrid"
                / "summary.json"
            )

            summaries[method] = (
                read_summary(path)
            )

        first = next(
            iter(summaries.values())
        )

        ids = [
            row["game_id"]
            for row in first["episodes"]
        ]

        for method, summary in summaries.items():
            current_ids = [
                row["game_id"]
                for row in summary["episodes"]
            ]

            if current_ids != ids:
                raise RuntimeError(
                    f"Tasks are not aligned: "
                    f"{split}/{method}"
                )

        print(
            f"\n{'=' * 65}\n"
            f"{split} | "
            f"Top-{args.top_k} | "
            f"Pool-{args.pool_size}\n"
            f"{'=' * 65}"
        )

        for method, store in stores.items():
            diagnostics = []

            rejection_counts = Counter()
            changes = Counter()
            empty = 0

            for episode in (
                summaries[method]["episodes"]
            ):
                detail = store.explain(
                    episode["task"],
                    args.top_k,
                )

                detail.update({
                    "episode": episode["episode"],
                    "game_id": episode["game_id"],
                    "task_type": episode["task_type"],
                    "baseline_success": episode["success"],
                })

                old = [
                    row["index"]
                    for row in detail["baseline"]
                ]

                new = [
                    row["index"]
                    for row in detail["selected"]
                ]

                if old != new:
                    changes[
                        episode["task_type"]
                    ] += 1

                if not new:
                    empty += 1

                rejection_counts.update(
                    item["reason"]
                    for item in detail["rejected"]
                )

                diagnostics.append(detail)

            path = (
                OUTPUT
                / model_key
                / split
                / f"{method}.json"
            )

            save_json(
                path,
                {
                    "model": args.model,
                    "split": split,
                    "method": method,
                    "pool_size": args.pool_size,
                    "top_k": args.top_k,
                    "memory_size": len(store),
                    "changed_by_type": dict(
                        changes
                    ),
                    "rejections": dict(
                        rejection_counts
                    ),
                    "empty_selections": empty,
                    "episodes": diagnostics,
                },
            )

            print(
                f"{method}: "
                f"changed={sum(changes.values())}"
                f"/{len(diagnostics)}, "
                f"empty={empty}, "
                f"rejected={dict(rejection_counts)}"
            )

            print(
                f"  changed_by_type="
                f"{dict(changes)}"
            )

            print(
                f"  saved: {path}"
            )

        if {
            "original_18",
            "direct_update",
        }.issubset(summaries):

            before = (
                summaries["original_18"]["episodes"]
            )

            after = (
                summaries["direct_update"]["episodes"]
            )

            changed = [
                (a, b)
                for a, b in zip(
                    before,
                    after,
                )
                if a["success"] != b["success"]
            ]

            print(
                f"\nOutcome-changed episodes: "
                f"{len(changed)}"
            )

            for a, b in changed[:5]:
                tag = (
                    "IMPROVED"
                    if b["success"]
                    else "REGRESSED"
                )

                detail = (
                    stores["direct_update"].explain(
                        b["task"],
                        args.top_k,
                    )
                )

                print(
                    f"  [{tag}] "
                    f"episode={b['episode']} "
                    f"task={b['task']}"
                )

                print(
                    "    Hybrid:",
                    [
                        item["lesson"]
                        for item in detail["baseline"]
                    ],
                )

                print(
                    "    TaskAware:",
                    [
                        item["lesson"]
                        for item in detail["selected"]
                    ],
                )

                print(
                    "    Rejected:",
                    [
                        item["reason"]
                        for item in detail["rejected"]
                    ],
                )

    print(
        "\nOffline analysis finished. "
        "No LLM or environment calls were made."
    )


if __name__ == "__main__":
    main()

