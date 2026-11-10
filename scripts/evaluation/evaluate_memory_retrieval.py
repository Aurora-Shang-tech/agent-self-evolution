
import argparse
import hashlib
import json
import os
import random
from dataclasses import asdict
from pathlib import Path

from rank_bm25 import BM25Okapi

from src.agents.react_agent import ReActAgent
from src.core.trajectory import Trajectory
from src.data.manifest import load_manifest
from src.envs.environment import ALFWorldEnvironment
from src.llm.client import LLMClient
from src.memory.store import ExperienceStore

from scripts.evaluation.evaluate_memory_evolution import (
    SPLIT_SIZES,
    TOP_K,
    MAX_STEPS,
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
    "original_130",
    "original_18",
    "original_21",
    "original_22",
    "direct_update",
    "rule_strength_voting",
    "source_aware_revision",
)

DEFAULT_METHODS = (
    "original_130",
    "original_21",
    "direct_update",
    "rule_strength_voting",
    "source_aware_revision",
)

OUTPUT_ROOT = Path(
    "outputs/memory_retrieval/evaluations"
)

OLD_OUTPUT_ROOT = Path(
    "outputs/memory_evolution/evaluations"
)

SOURCE_FILES = (
    "src/agents/react_agent.py",
    "src/core/trajectory.py",
    "src/core/protocol.py",
    "src/envs/environment.py",
    "src/memory/store.py",
    "src/memory/experience.py",
    "src/memory/state_tracker.py",
    "src/llm/client.py",
)


def hash_data(data):
    content = json.dumps(
        data,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )

    return hashlib.sha256(
        content.encode("utf-8")
    ).hexdigest()


def load_memory(method, model):
    if method.startswith("original_"):
        path = memory_file("original_130", model)
        store = load_store("original_130", path)

        count = int(method.split("_")[1])

        if count > len(store):
            raise ValueError(
                f"Requested {count} memories, "
                f"only {len(store)} available"
            )

        if count == len(store):
            experiences = list(store.experiences)
            selected_indices = list(
                range(len(experiences))
            )
        else:
            indices = list(
                range(len(store.experiences))
            )

            random.Random(42).shuffle(indices)

            selected_indices = sorted(
                indices[:count]
            )

            experiences = [
                store.experiences[i]
                for i in selected_indices
            ]

    else:
        path = memory_file(method, model)
        store = load_store(method, path)
        experiences = list(store.experiences)
        selected_indices = None

    fingerprint = hash_data({
        "source_file_hash": file_hash(path),
        "selected_indices": selected_indices,
        "experiences": [
            asdict(item)
            for item in experiences
        ],
    })

    return experiences, path, fingerprint


class HybridExperienceStore(ExperienceStore):
    def __init__(self, experiences):
        super().__init__()

        self.experiences = list(experiences)

        self.source_bm25 = BM25Okapi([
            self._tokenize(item.source_task)
            for item in self.experiences
        ])

        self.lesson_bm25 = BM25Okapi([
            self._tokenize(item.lesson)
            for item in self.experiences
        ])

    @staticmethod
    def normalize(scores):
        values = [
            max(0.0, float(score))
            for score in scores
        ]

        maximum = max(values, default=0.0)

        if maximum <= 0:
            return [
                0.0 for _ in values
            ]

        return [
            value / maximum
            for value in values
        ]

    def retrieve(self, task, top_k=3):
        if top_k < 1:
            raise ValueError("top_k must be >= 1")

        if not self.experiences:
            return []

        query = self._tokenize(task)

        source_scores = self.normalize(
            self.source_bm25.get_scores(query)
        )

        lesson_scores = self.normalize(
            self.lesson_bm25.get_scores(query)
        )

        scores = [
            0.5 * source_scores[i]
            + 0.5 * lesson_scores[i]
            for i in range(len(self.experiences))
        ]

        ranked = sorted(
            range(len(self.experiences)),
            key=lambda i: (-scores[i], i),
        )

        return [
            self.experiences[i]
            for i in ranked[:top_k]
        ]


def make_store(experiences, retrieval):
    if retrieval == "hybrid":
        return HybridExperienceStore(
            experiences
        )

    store = ExperienceStore()

    for item in experiences:
        store.add(item)

    return store


def validate_config(directory, config):
    path = directory / "experiment_config.json"

    if path.exists():
        old = json.loads(
            path.read_text(encoding="utf-8")
        )

        if old != config:
            raise RuntimeError(
                f"Experiment config changed: {directory}"
            )
    else:
        if list(directory.glob("episode_*.json")):
            raise RuntimeError(
                f"Trajectories without config: {directory}"
            )

        save_json(path, config)


def load_cached_trajectory(path, game):
    if not path.exists():
        return None

    try:
        trajectory = Trajectory.load_json(
            str(path)
        )
    except (
        json.JSONDecodeError,
        UnicodeDecodeError,
        OSError,
        ValueError,
        TypeError,
        KeyError,
    ) as exc:
        print(
            f"Invalid cache, deleting {path}: {exc}",
            flush=True,
        )
        path.unlink(missing_ok=True)
        return None

    if (
        Path(trajectory.gamefile).resolve()
        != Path(game.game_file).resolve()
    ):
        raise RuntimeError(
            f"Game mismatch: {path}"
        )

    return trajectory


def load_existing_source(method, model, split):
    size = SPLIT_SIZES[split]

    root = (
        OLD_OUTPUT_ROOT
        / model.replace("/", "_")
        / split
        / "no_state"
        / method
    )

    summary_path = root / "summary.json"
    config_path = root / "experiment_config.json"

    if not summary_path.is_file():
        raise FileNotFoundError(summary_path)

    if not config_path.is_file():
        raise FileNotFoundError(config_path)

    old_config = json.loads(
        config_path.read_text(encoding="utf-8")
    )

    manifest_path = Path(
        f"outputs/manifests/{split}_seed42_{size}.json"
    )

    memory_path = memory_file(method, model)

    checks = {
        "model": model,
        "split": split,
        "method": method,
        "use_state_tracking": False,
        "top_k": TOP_K,
        "max_steps": MAX_STEPS,
        "manifest_hash": file_hash(manifest_path),
        "memory_hash": file_hash(memory_path),
    }

    for key, expected in checks.items():
        if old_config.get(key) != expected:
            raise RuntimeError(
                f"Existing source experiment mismatch: "
                f"{root} ({key})"
            )

    old_hashes = old_config.get(
        "source_hashes", {}
    )

    for name in SOURCE_FILES:
        old_hash = old_hashes.get(name)

        if old_hash is None:
            raise RuntimeError(
                f"Missing source fingerprint: {name}"
            )

        if old_hash != file_hash(Path(name)):
            raise RuntimeError(
                f"Source code changed since old "
                f"evaluation: {name}"
            )

    summary = json.loads(
        summary_path.read_text(encoding="utf-8")
    )

    if (
        not summary.get("finished")
        or summary.get("completed") != size
        or len(summary.get("episodes", [])) != size
    ):
        raise RuntimeError(
            f"Old evaluation is incomplete: {root}"
        )

    print(
        f"[REUSE] {method}/source: "
        f"{summary['successes']}/{size}, "
        f"steps={summary['average_decision_steps']:.2f}"
    )

    summary["retrieval"] = "source"
    summary["source_summary_path"] = str(
        summary_path
    )

    return summary


def evaluate(
    method,
    retrieval,
    model,
    split,
    games,
    llm,
    max_new,
):
    experiences, memory_path, fingerprint = (
        load_memory(method, model)
    )

    store = make_store(
        experiences, retrieval
    )

    output_dir = (
        OUTPUT_ROOT
        / model.replace("/", "_")
        / split
        / method
        / retrieval
    )

    manifest_path = Path(
        f"outputs/manifests/"
        f"{split}_seed42_{len(games)}.json"
    )

    config = {
        "model": model,
        "split": split,
        "method": method,
        "retrieval": retrieval,
        "state_tracking": False,
        "top_k": TOP_K,
        "max_steps": MAX_STEPS,
        "memory_file": str(memory_path),
        "memory_fingerprint": fingerprint,
        "memory_size": len(store),
        "manifest_hash": file_hash(
            manifest_path
        ),
        "source_hashes": {
            name: file_hash(Path(name))
            for name in (
                *SOURCE_FILES,
                "scripts/evaluation/"
                "evaluate_memory_retrieval.py",
                "scripts/evaluation/"
                "evaluate_memory_evolution.py",
            )
        },
    }

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    validate_config(
        output_dir, config
    )

    records = []
    new_count = 0

    print(
        f"\n[{method}][{retrieval}] "
        f"Memory={len(store)} "
        f"Tasks={len(games)}",
        flush=True,
    )

    for index, game in enumerate(games):
        episode_path = (
            output_dir / f"episode_{index:03d}.json"
        )

        trajectory = load_cached_trajectory(
            episode_path, game
        )

        if trajectory is None:
            if (
                max_new is not None
                and new_count >= max_new
            ):
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
                episode_path, trajectory
            )

            new_count += 1

        record = build_record(
            index, game, trajectory
        )

        records.append(record)

        summary = build_summary(
            records, len(games)
        )

        summary.update({
            "model": model,
            "split": split,
            "method": method,
            "retrieval": retrieval,
            "memory_size": len(store),
            "memory_fingerprint": fingerprint,
        })

        save_json(
            output_dir / "summary.json",
            summary,
        )

        print(
            f"[{method}][{retrieval}] "
            f"{index + 1}/{len(games)} "
            f"success={record['success']} "
            f"steps={record['decision_steps']} "
            f"rate={summary['success_rate']:.2%}",
            flush=True,
        )

    return summary


def compare_results(results, model, split):
    comparisons = []

    for method in METHODS:
        source_key = (method, "source")
        hybrid_key = (method, "hybrid")

        if (
            source_key in results
            and hybrid_key in results
            and results[source_key]["finished"]
            and results[hybrid_key]["finished"]
        ):
            comparisons.append(
                paired_comparison(
                    f"{method}:source",
                    results[source_key],
                    f"{method}:hybrid",
                    results[hybrid_key],
                )
            )

    for method in (
        "direct_update",
        "rule_strength_voting",
        "source_aware_revision",
    ):
        for retrieval in ("source", "hybrid"):
            baseline = (
                "original_21", retrieval
            )

            target = (
                method, retrieval
            )

            if (
                baseline in results
                and target in results
                and results[baseline]["finished"]
                and results[target]["finished"]
            ):
                comparisons.append(
                    paired_comparison(
                        f"original_21:{retrieval}",
                        results[baseline],
                        f"{method}:{retrieval}",
                        results[target],
                    )
                )

    root = (
        OUTPUT_ROOT
        / model.replace("/", "_")
        / split
    )

    save_json(
        root / "comparison.json",
        {
            "split": split,
            "model": model,
            "state_tracking": False,
            "summaries": {
                f"{method}:{retrieval}": {
                    key: value
                    for key, value in summary.items()
                    if key != "episodes"
                }
                for (method, retrieval), summary
                in results.items()
            },
            "paired_comparisons": comparisons,
        },
    )

    print("\n" + "=" * 65)
    print("RETRIEVAL COMPARISON")
    print("=" * 65)

    for row in comparisons:
        print(
            f"{row['from']} -> {row['to']}: "
            f"+{row['improved']}/-{row['regressed']}, "
            f"net={row['net']:+d}, "
            f"steps={row['average_steps_change']:+.2f}, "
            f"p={row['mcnemar_exact_p']:.6f}"
        )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--split",
        choices=SPLIT_SIZES,
        required=True,
    )

    parser.add_argument(
        "--methods",
        nargs="+",
        choices=METHODS,
        default=list(DEFAULT_METHODS),
    )

    parser.add_argument(
        "--retrievals",
        nargs="+",
        choices=("source", "hybrid"),
        default=["source", "hybrid"],
    )

    parser.add_argument(
        "--max-new",
        type=int,
        default=None,
    )

    args = parser.parse_args()

    if (
        args.max_new is not None
        and args.max_new < 1
    ):
        parser.error("--max-new must be >= 1")

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

    games = load_manifest(manifest_path)

    if len(games) != size:
        raise RuntimeError(
            f"Expected {size} games, got {len(games)}"
        )

    methods = list(dict.fromkeys(
        args.methods
    ))

    retrievals = list(dict.fromkeys(
        args.retrievals
    ))

    results = {}
    llm = None

    for method in methods:
        for retrieval in retrievals:
            key = (method, retrieval)

            if (
                retrieval == "source"
                and method != "original_21"
                and method not in (
                    "original_18",
                    "original_22",
                )
            ):
                results[key] = load_existing_source(
                    method, model, args.split
                )

                continue

            if llm is None:
                llm = LLMClient()

            results[key] = evaluate(
                method=method,
                retrieval=retrieval,
                model=model,
                split=args.split,
                games=games,
                llm=llm,
                max_new=args.max_new,
            )

    compare_results(
        results, model, args.split
    )


if __name__ == "__main__":
    main()

