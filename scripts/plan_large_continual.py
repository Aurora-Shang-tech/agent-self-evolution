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


ROOT = Path("outputs/continual_formal_large_v1")
PROTOCOL = ROOT / "protocol.json"
TEMPLATE = Path("outputs/continual_32/protocol.json")
INITIAL = Path(
    "outputs/continual_32/runs/fixed_32/versions/v000.json"
)
OLD_FINAL = Path(
    "outputs/continual_formal_120_60/final_holdout.json"
)

SEED = 20261118
ROUNDS = 10
MAX_LEARN_PER_TYPE = 200
VALID_PER_TYPE = 20


def source_files():
    files = set(exclusion_sources())

    for pattern in (
        "unified_evolution_v*/**/protocol.json",
        "unified_evolution_v*/**/validation*.json",
        "continual_formal_*/learning_round_*.json",
        "continual_formal_*/validation.json",
    ):
        files.update(Path("outputs").glob(pattern))

    return sorted(
        p for p in files
        if p.is_file()
        and not p.is_relative_to(ROOT)
    )


def excluded_games():
    audit = read_json(AUDIT_PATH)

    excluded = {
        row["game_id"]
        for row in audit["records"]
    }

    sources = {}

    for path in source_files():
        sources[str(path)] = sha(path)
        collect_train_ids(read_json(path), excluded)

    return excluded, sources


def make_plan():
    if PROTOCOL.exists():
        audit_plan()
        return

    if ROOT.exists():
        raise RuntimeError(
            "Experiment directory already exists without protocol"
        )

    template = read_json(TEMPLATE)

    _, info = load_memory_version(
        INITIAL.parent.parent, "v000"
    )

    if info.memory_size != 32:
        raise RuntimeError("Expected 32 initial lessons")

    if not OLD_FINAL.is_file():
        raise FileNotFoundError(OLD_FINAL)

    excluded, sources = excluded_games()

    available = {name: [] for name in TASK_TYPES}

    for game in discover_games(
        default_data_root(), "train"
    ):
        if (
            game.id not in excluded
            and game.task_type in available
        ):
            available[game.task_type].append(game)

    counts = {
        name: len(games)
        for name, games in available.items()
    }

    capacity = min(counts.values()) - VALID_PER_TYPE

    learn_per_type = min(
        MAX_LEARN_PER_TYPE,
        (capacity // ROUNDS) * ROUNDS,
    )

    if learn_per_type < 100:
        raise RuntimeError(
            f"Insufficient independent Train games: {counts}"
        )

    per_round_per_type = learn_per_type // ROUNDS

    rounds = [[] for _ in range(ROUNDS)]
    validation = []

    for i, task_type in enumerate(TASK_TYPES):
        games = sorted(
            available[task_type],
            key=lambda x: x.id,
        )

        random.Random(SEED + i).shuffle(games)

        learn = games[:learn_per_type]

        validation.extend(
            games[
                learn_per_type:
                learn_per_type + VALID_PER_TYPE
            ]
        )

        for j in range(ROUNDS):
            start = j * per_round_per_type
            stop = start + per_round_per_type
            rounds[j].extend(learn[start:stop])

    for i, games in enumerate(rounds):
        random.Random(SEED + 100 + i).shuffle(games)

    random.Random(SEED + 200).shuffle(validation)

    final_games = load_manifest(OLD_FINAL)

    train_ids = [
        game.id
        for group in rounds
        for game in group
    ] + [game.id for game in validation]

    if (
        len(set(train_ids)) != len(train_ids)
        or any(game_id in excluded for game_id in train_ids)
    ):
        raise RuntimeError("Train task overlap")

    ROOT.mkdir(parents=True)

    for i, games in enumerate(rounds, 1):
        save_manifest(
            games,
            ROOT / f"learning_round_{i:02d}.json",
            seed=SEED + i,
        )

    save_manifest(
        validation,
        ROOT / "validation.json",
        seed=SEED + 100,
    )

    save_manifest(
        final_games,
        ROOT / "final_holdout.json",
        seed=SEED + 200,
    )

    manifests = {
        path.name: sha(path)
        for path in ROOT.glob("*.json")
        if path.name != "protocol.json"
    }

    all_games = [
        game
        for group in rounds
        for game in group
    ] + validation + final_games

    assets = {
        str(game.game_file): sha(game.game_file)
        for game in all_games
    }

    protocol = {
        "schema": "continual_formal_large_v1",
        "status": "FROZEN",
        "seed": SEED,
        "model": template["model"],
        "max_steps": template["max_steps"],
        "top_k": template["top_k"],
        "retrieval": "source_task_bm25_type_priority",
        "state_tracking": True,
        "methods": [
            "fixed", "append_only", "dynamic"
        ],
        "selection_policy": template["selection_policy"],
        "rounds": ROUNDS,
        "learning_tasks_per_round": len(rounds[0]),
        "validation_tasks": len(validation),
        "max_failures_per_round": 10,
        "max_counterfactual_actions_per_failure": 3,
        "initial_memory": str(INITIAL),
        "initial_memory_sha256": sha(INITIAL),
        "manifests": manifests,
        "game_asset_sha256": assets,
        "source_exclusion_hashes": sources,
        "historical_audit_sha256": sha(AUDIT_PATH),
        "source_protocol_sha256": sha(TEMPLATE),
        "planner_sha256": sha(Path(__file__)),
        "excluded_train_games": len(excluded),
        "learning_per_type": learn_per_type,
        "validation_per_type": VALID_PER_TYPE,
        "final_split_sizes": dict(
            Counter(game.split for game in final_games)
        ),
    }

    save_immutable(PROTOCOL, protocol)
    audit_plan()


def audit_plan():
    p = read_json(PROTOCOL)

    if (
        p.get("schema") != "continual_formal_large_v1"
        or p.get("status") != "FROZEN"
        or p.get("rounds") != ROUNDS
        or p.get("validation_tasks") != 120
        or p.get("initial_memory") != str(INITIAL)
        or p.get("max_failures_per_round") != 10
        or p.get("max_counterfactual_actions_per_failure") != 3
    ):
        raise RuntimeError("Invalid formal protocol")

    if (
        sha(Path(__file__)) != p["planner_sha256"]
        or sha(INITIAL) != p["initial_memory_sha256"]
        or sha(TEMPLATE) != p["source_protocol_sha256"]
        or sha(AUDIT_PATH) != p["historical_audit_sha256"]
    ):
        raise RuntimeError("Frozen experiment source changed")

    for path, expected in p["source_exclusion_hashes"].items():
        if sha(path) != expected:
            raise RuntimeError(f"Exclusion source changed: {path}")

    for filename, expected in p["manifests"].items():
        if sha(ROOT / filename) != expected:
            raise RuntimeError(f"Manifest changed: {filename}")

    for path, expected in p["game_asset_sha256"].items():
        if sha(path) != expected:
            raise RuntimeError(f"ALFWorld game changed: {path}")

    excluded = {
        row["game_id"]
        for row in read_json(AUDIT_PATH)["records"]
    }

    for path in p["source_exclusion_hashes"]:
        collect_train_ids(read_json(path), excluded)

    per_type = p["learning_per_type"]
    per_round = per_type // ROUNDS

    if (
        not 100 <= per_type <= MAX_LEARN_PER_TYPE
        or per_type % ROUNDS != 0
        or p["learning_tasks_per_round"] != 6 * per_round
    ):
        raise RuntimeError("Invalid learning size")

    all_ids = []
    learning_types = Counter()

    for i in range(1, ROUNDS + 1):
        games = load_manifest(
            ROOT / f"learning_round_{i:02d}.json"
        )

        counts = Counter(g.task_type for g in games)

        if (
            len(games) != 6 * per_round
            or any(
                counts[name] != per_round
                for name in TASK_TYPES
            )
            or any(g.split != "train" for g in games)
        ):
            raise RuntimeError(f"Invalid learning round {i}")

        learning_types.update(counts)
        all_ids.extend(g.id for g in games)

    validation = load_manifest(
        ROOT / "validation.json"
    )

    validation_types = Counter(
        g.task_type for g in validation
    )

    if (
        len(validation) != 120
        or any(g.split != "train" for g in validation)
        or any(
            validation_types[name] != VALID_PER_TYPE
            for name in TASK_TYPES
        )
        or any(
            learning_types[name] != per_type
            for name in TASK_TYPES
        )
    ):
        raise RuntimeError("Invalid validation distribution")

    all_ids.extend(g.id for g in validation)

    if (
        len(set(all_ids)) != len(all_ids)
        or any(game_id in excluded for game_id in all_ids)
    ):
        raise RuntimeError("Historical Train overlap")

    final = load_manifest(ROOT / "final_holdout.json")

    if dict(Counter(g.split for g in final)) != (
        p["final_split_sizes"]
    ):
        raise RuntimeError("Final evaluation manifest changed")

    print("\n========== LARGE CONTINUAL PROTOCOL ==========")
    print("Learning Train:", ROUNDS * 6 * per_round)
    print("Rounds:", ROUNDS)
    print("Learning per round:", 6 * per_round)
    print("Validation Train:", len(validation))
    print("Learning per task type:", per_type)
    print("Historical excluded:", len(excluded))
    print("Initial memory: 32 lessons")
    print("Model:", p["model"])
    print("Max failures per round: 10")
    print("Counterfactual actions per failure: 3")
    print("Final evaluation:", p["final_split_sizes"])
    print("AUDIT: PASS")
    print("New LLM calls: 0")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        required=True,
        choices=("plan", "audit"),
    )
    args = parser.parse_args()

    if args.stage == "plan":
        make_plan()
    else:
        audit_plan()


if __name__ == "__main__":
    main()
