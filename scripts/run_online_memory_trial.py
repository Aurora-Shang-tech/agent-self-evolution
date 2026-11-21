import argparse
from collections import Counter
from pathlib import Path

from scripts import run_unified_batch as batch
from scripts.run_unified_v3 import make_scheduler
from src.data.manifest import load_manifest


ROOT = Path(
    "outputs/unified_evolution_v2/online_memory_trial_01"
)
PROTOCOL = ROOT / "protocol.json"
VALIDATION = Path(
    "outputs/manifests/"
    "online_memory_trial_01_validation.json"
)
LOCK = ROOT / "trial_lock.json"
STATE_ROOT = Path(
    "outputs/unified_evolution_v3/online_memory_trial_01"
)
SEED = 20261117


def configure():
    batch.ROOT = ROOT
    batch.PROTOCOL_PATH = PROTOCOL
    batch.SEED = SEED


def validation_record(games):
    return {
        "seed": SEED + 1,
        "num_games": len(games),
        "games": [
            {
                "id": game.id,
                "split": game.split,
                "game_file": str(game.game_file),
                "task_type": game.task_type,
                "game_sha256": batch.sha(game.game_file),
            }
            for game in games
        ],
    }


def create_plan():
    configure()

    if PROTOCOL.exists():
        protocol, caps = batch.verify_batch()
    else:
        protocol, caps = batch.create_plan()

    if not VALIDATION.exists():
        old, _ = batch.verify_old_protocol()
        excluded, _ = batch.collect_sources(old)

        excluded.update(
            game["id"] for game in protocol["games"]
        )

        available = batch.discover_games(
            batch.default_data_root(),
            "train",
        )

        group_a = batch.select_games(
            available,
            excluded,
            seed=SEED + 1,
        )

        excluded.update(game.id for game in group_a)

        group_b = batch.select_games(
            available,
            excluded,
            seed=SEED + 2,
        )

        selected = group_a + group_b

        if len({game.id for game in selected}) != 12:
            raise RuntimeError("Duplicate validation game")

        batch.save_immutable(
            VALIDATION,
            validation_record(selected),
        )

    lock = {
        "schema": "online_memory_trial_lock_v1",
        "seed": SEED,
        "protocol_sha256": batch.sha(PROTOCOL),
        "validation_sha256": batch.sha(VALIDATION),
        "entrypoint_sha256": batch.sha(Path(__file__)),
    }

    batch.save_immutable(LOCK, lock)
    return verify()


def verify():
    configure()

    protocol, caps = batch.verify_batch()
    lock = batch.read_json(LOCK)

    if lock != {
        "schema": "online_memory_trial_lock_v1",
        "seed": SEED,
        "protocol_sha256": batch.sha(PROTOCOL),
        "validation_sha256": batch.sha(VALIDATION),
        "entrypoint_sha256": batch.sha(Path(__file__)),
    }:
        raise RuntimeError("Trial protocol or code changed")

    data = batch.read_json(VALIDATION)
    games = load_manifest(VALIDATION)

    if (
        data["num_games"] != 12
        or len(games) != 12
        or len(data["games"]) != 12
    ):
        raise RuntimeError("Invalid validation size")

    train_ids = {
        item["id"] for item in protocol["games"]
    }

    validation_ids = [game.id for game in games]

    if (
        len(set(validation_ids)) != 12
        or train_ids.intersection(validation_ids)
    ):
        raise RuntimeError("Learning/validation overlap")

    for game, item in zip(games, data["games"]):
        if (
            game.id != item["id"]
            or game.split != "train"
            or game.task_type != item["task_type"]
            or str(game.game_file) != item["game_file"]
            or batch.sha(game.game_file)
            != item["game_sha256"]
        ):
            raise RuntimeError(
                f"Validation game changed: {game.id}"
            )

    return protocol, caps


def show_status(protocol, caps):
    jobs, saved = batch.collect_jobs(protocol, caps)

    scheduler = make_scheduler(
        jobs,
        caps,
        protocol,
        batch_root=ROOT,
        state_root=STATE_ROOT,
        execute=False,
    )

    results = scheduler.run(max_new_jobs=0)
    counts = Counter(
        item["state"] for item in results
    )

    print("\n========== ONLINE MEMORY TRIAL ==========")
    print("Saved episodes:", saved)
    print("Planned learning games:", len(protocol["games"]))
    print("Frozen validation games: 12")
    print("Memory:", caps.memory_info.version)
    print("Active skill:", caps.skill_id)
    print("Review jobs:", len(jobs))

    for row in results:
        print(
            row["job_id"],
            row["route"],
            row["state"],
            row["outcome_status"],
        )

    print("Cached:", counts["CACHED"])
    print("Pending:", counts["PENDING"])
    print("Executed:", counts["EXECUTED"])


def run_review(
    protocol,
    caps,
    *,
    max_new_jobs,
    max_llm_calls,
):
    jobs, saved = batch.collect_jobs(protocol, caps)

    scheduler = make_scheduler(
        jobs,
        caps,
        protocol,
        batch_root=ROOT,
        state_root=STATE_ROOT,
        execute=max_new_jobs > 0,
        max_llm_calls=max_llm_calls,
    )

    results = scheduler.run(
        max_new_jobs=max_new_jobs
    )

    print("\n========== TRIAL REVIEW ==========")
    print("Saved episodes:", saved)

    for row in results:
        print(
            row["job_id"],
            row["route"],
            row["state"],
            row["outcome_status"],
        )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--stage",
        required=True,
        choices=("plan", "run", "report", "review"),
    )
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument(
        "--max-new-jobs", type=int, default=0
    )
    parser.add_argument(
        "--max-llm-calls", type=int, default=16
    )

    args = parser.parse_args()

    if args.max_new_jobs < 0:
        parser.error("Invalid Review job budget")

    if args.stage == "plan":
        protocol, caps = create_plan()

        print("========== MEMORY TRIAL PLAN ==========")
        print("Learning games:", len(protocol["games"]))
        print("Validation games: 12")
        print("Active memory:", caps.memory_info.version)

        for index, game in enumerate(protocol["games"]):
            print(
                index,
                game["task_type"],
                game["id"],
            )

        print("Protocol:", PROTOCOL)
        print("Validation:", VALIDATION)
        print("New LLM calls: 0")
        return

    protocol, caps = verify()

    if args.stage == "run":
        if args.max_new_jobs != 0:
            parser.error(
                "Review execution requires --stage review"
            )

        batch.run_episodes(
            protocol,
            caps,
            args.limit,
        )
        show_status(protocol, caps)

    elif args.stage == "review":
        run_review(
            protocol,
            caps,
            max_new_jobs=args.max_new_jobs,
            max_llm_calls=args.max_llm_calls,
        )

    else:
        show_status(protocol, caps)


if __name__ == "__main__":
    main()
