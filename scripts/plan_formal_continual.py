import argparse
import random
from collections import Counter
from pathlib import Path

from scripts.run_unified_pilot import (
    AUDIT_PATH,
    TASK_TYPES,
    collect_train_ids,
    exclusion_sources,
    read_json,
    save_immutable,
    sha,
)
from src.data.alfworld_data import (
    default_data_root,
    discover_games,
)
from src.data.manifest import load_manifest, save_manifest
from src.evolution.version_store import load_memory_version


ROOT = Path("outputs/continual_formal_120_60")
PROTOCOL = ROOT / "protocol.json"

TEMPLATE = Path("outputs/continual_32/protocol.json")
INITIAL = Path(
    "outputs/continual_32/runs/fixed_32/versions/v000.json"
)

SEED = 20261117
ROUNDS = 5
TRAIN_PER_TYPE = 20
VALIDATION_PER_TYPE = 10
LEARNING_PER_ROUND = 24


def source_paths():
    paths = set(exclusion_sources())

    paths.update(
        Path("outputs").glob(
            "unified_evolution_v*/**/protocol.json"
        )
    )

    paths.update(
        Path("outputs").glob(
            "unified_evolution_v*/**/validation*.json"
        )
    )

    return sorted(
        p for p in paths
        if p.is_file() and not p.is_relative_to(ROOT)
    )


def selected_games():
    audit = read_json(AUDIT_PATH)

    excluded = {
        row["game_id"]
        for row in audit["records"]
    }

    source_hashes = {}

    for path in source_paths():
        data = read_json(path)

        source_hashes[str(path)] = sha(path)

        collect_train_ids(data, excluded)

    available = {
        task_type: []
        for task_type in TASK_TYPES
    }

    for game in discover_games(
        default_data_root(), "train"
    ):
        if (
            game.id not in excluded
            and game.task_type in available
        ):
            available[game.task_type].append(game)

    learning_by_type = {}
    validation = []

    for i, task_type in enumerate(TASK_TYPES):
        games = sorted(
            available[task_type],
            key=lambda g: g.id,
        )

        random.Random(SEED + i).shuffle(games)

        required = (
            TRAIN_PER_TYPE + VALIDATION_PER_TYPE
        )

        if len(games) < required:
            raise RuntimeError(
                f"Insufficient unused {task_type}: "
                f"{len(games)} < {required}"
            )

        learning_by_type[task_type] = (
            games[:TRAIN_PER_TYPE]
        )

        validation.extend(
            games[TRAIN_PER_TYPE:required]
        )

    rounds = []

    for i in range(ROUNDS):
        current = []

        for task_type in TASK_TYPES:
            start = i * (
                TRAIN_PER_TYPE // ROUNDS
            )
            stop = (i + 1) * (
                TRAIN_PER_TYPE // ROUNDS
            )

            current.extend(
                learning_by_type[task_type][start:stop]
            )

        random.Random(
            SEED + 100 + i
        ).shuffle(current)

        if len(current) != LEARNING_PER_ROUND:
            raise RuntimeError("Invalid round size")

        rounds.append(current)

    random.Random(SEED + 200).shuffle(validation)

    all_ids = (
        [g.id for group in rounds for g in group]
        + [g.id for g in validation]
    )

    if (
        len(all_ids) != 180
        or len(set(all_ids)) != 180
    ):
        raise RuntimeError(
            "Learning and validation are not disjoint"
        )

    return (
        rounds,
        validation,
        source_hashes,
        len(excluded),
    )


def create():
    if PROTOCOL.exists():
        audit_protocol()
        return

    if ROOT.exists() and any(ROOT.iterdir()):
        raise RuntimeError(
            "Partially initialized experiment directory"
        )

    if not TEMPLATE.is_file() or not INITIAL.is_file():
        raise FileNotFoundError(
            "Missing original continual protocol "
            "or v000 snapshot"
        )

    template = read_json(TEMPLATE)

    _, info = load_memory_version(
        INITIAL.parent.parent,
        "v000",
    )

    if info.memory_size != 32:
        raise RuntimeError(
            "Expected 32-lesson starting memory"
        )

    (
        rounds,
        validation,
        source_hashes,
        excluded_count,
    ) = selected_games()

    final_seen = discover_games(
        default_data_root(), "valid_seen"
    )

    final_unseen = discover_games(
        default_data_root(), "valid_unseen"
    )

    final_games = final_seen + final_unseen

    ROOT.mkdir(parents=True, exist_ok=False)

    for i, games in enumerate(rounds, 1):
        save_manifest(
            games,
            ROOT / f"learning_round_{i:02d}.json",
            seed=SEED,
        )

    save_manifest(
        validation,
        ROOT / "validation.json",
        seed=SEED + 1,
    )

    save_manifest(
        final_games,
        ROOT / "final_holdout.json",
        seed=SEED + 2,
    )

    manifests = {
        path.name: sha(path)
        for path in sorted(ROOT.glob("*.json"))
        if path.name != "protocol.json"
    }

    assets = {
        str(game.game_file): sha(game.game_file)
        for game in (
            [g for group in rounds for g in group]
            + validation
            + final_games
        )
    }

    protocol = {
        "schema": "continual_formal_120_60_v1",
        "status": "FROZEN",
        "seed": SEED,
        "model": template["model"],
        "max_steps": template["max_steps"],
        "top_k": template["top_k"],
        "max_failures_per_round": min(
            2,
            template["max_failures_per_round"],
        ),
        "max_counterfactual_actions_per_failure": min(
            2,
            template[
                "max_counterfactual_actions_per_failure"
            ],
        ),
        "rounds": ROUNDS,
        "learning_tasks_per_round": LEARNING_PER_ROUND,
        "validation_tasks": len(validation),
        "retrieval": "source_task_bm25_type_priority",
        "state_tracking": True,
        "methods": [
            "fixed",
            "append_only",
            "dynamic",
        ],
        "selection_policy": template["selection_policy"],
        "initial_memory": str(INITIAL),
        "initial_memory_sha256": sha(INITIAL),
        "manifests": manifests,
        "game_asset_sha256": assets,
        "source_protocol_sha256": sha(TEMPLATE),
        "historical_audit_sha256": sha(AUDIT_PATH),
        "source_exclusion_hashes": source_hashes,
        "historical_excluded_ids": excluded_count,
        "final_split_sizes": {
            "valid_seen": len(final_seen),
            "valid_unseen": len(final_unseen),
        },
    }

    save_immutable(PROTOCOL, protocol)

    audit_protocol()


def audit_protocol():
    p = read_json(PROTOCOL)

    if (
        p.get("schema")
        != "continual_formal_120_60_v1"
        or p.get("status") != "FROZEN"
        or p.get("rounds") != 5
        or p.get("learning_tasks_per_round") != 24
        or p.get("validation_tasks") != 60
        or p.get("initial_memory") != str(INITIAL)
    ):
        raise RuntimeError(
            "Frozen formal protocol mismatch"
        )

    if sha(TEMPLATE) != p["source_protocol_sha256"]:
        raise RuntimeError(
            "Historical controller protocol changed"
        )

    if sha(AUDIT_PATH) != p["historical_audit_sha256"]:
        raise RuntimeError(
            "Historical feedback audit changed"
        )

    if sha(INITIAL) != p["initial_memory_sha256"]:
        raise RuntimeError(
            "Frozen starting memory changed"
        )

    for filename, expected in p["manifests"].items():
        if sha(ROOT / filename) != expected:
            raise RuntimeError(
                f"Manifest modified: {filename}"
            )

    for path, expected in (
        p["source_exclusion_hashes"].items()
    ):
        if sha(path) != expected:
            raise RuntimeError(
                f"Historical source modified: {path}"
            )

    for path, expected in (
        p["game_asset_sha256"].items()
    ):
        if sha(path) != expected:
            raise RuntimeError(
                f"ALFWorld game modified: {path}"
            )

    all_ids = []
    types_learning = Counter()
    types_validation = Counter()

    for i in range(1, 6):
        games = load_manifest(
            ROOT / f"learning_round_{i:02d}.json"
        )

        if (
            len(games) != 24
            or any(
                game.split != "train"
                for game in games
            )
        ):
            raise RuntimeError(
                "Invalid learning round"
            )

        all_ids.extend(g.id for g in games)

        types_learning.update(
            g.task_type for g in games
        )

    validation = load_manifest(
        ROOT / "validation.json"
    )

    if (
        len(validation) != 60
        or any(
            g.split != "train"
            for g in validation
        )
    ):
        raise RuntimeError(
            "Invalid validation split"
        )

    all_ids.extend(g.id for g in validation)

    types_validation.update(
        g.task_type for g in validation
    )

    if (
        len(set(all_ids)) != 180
        or any(
            types_learning[t] != 20
            for t in TASK_TYPES
        )
        or any(
            types_validation[t] != 10
            for t in TASK_TYPES
        )
    ):
        raise RuntimeError(
            "Overlapping or unbalanced Train tasks"
        )

    excluded_now = {
        row["game_id"]
        for row in read_json(AUDIT_PATH)["records"]
    }

    for path in p["source_exclusion_hashes"]:
        collect_train_ids(
            read_json(path),
            excluded_now,
        )

    if any(
        game_id in excluded_now
        for game_id in all_ids
    ):
        raise RuntimeError(
            "Formal tasks overlap historical Train tasks"
        )

    final = load_manifest(
        ROOT / "final_holdout.json"
    )

    split_counts = Counter(
        g.split for g in final
    )

    if dict(split_counts) != p["final_split_sizes"]:
        raise RuntimeError(
            "Final evaluation split mismatch"
        )

    print(
        "========== FORMAL CONTINUAL PROTOCOL =========="
    )

    print(
        "Learning: 120 Train (5 rounds x 24; 20 per type)"
    )
    print(
        "Validation: 60 independent Train (10 per type)"
    )
    print(
        "Final evaluation:",
        dict(split_counts),
    )
    print(
        "Excluded historical Train IDs:",
        p["historical_excluded_ids"],
    )
    print("Starting memory: fixed32 v000")
    print("Model:", p["model"])
    print(
        "Counterfactual failures per round:",
        p["max_failures_per_round"],
    )
    print(
        "Actions per failure:",
        p["max_counterfactual_actions_per_failure"],
    )

    print("AUDIT PASS | LLM calls: 0")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--stage",
        required=True,
        choices=("plan", "audit"),
    )

    args = parser.parse_args()

    if args.stage == "plan":
        create()
    else:
        audit_protocol()


if __name__ == "__main__":
    main()
