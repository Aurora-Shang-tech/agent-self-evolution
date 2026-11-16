import argparse
import os
import random
import re
from collections import Counter
from pathlib import Path

from scripts.run_unified_pilot import (
    AUDIT_PATH,
    PROTOCOL_PATH as OLD_PROTOCOL_PATH,
    TASK_TYPES,
    collect_train_ids,
    exclusion_sources,
    read_json,
    save_immutable,
    sha,
    verify_protocol as verify_old_protocol,
)
from scripts.run_unified_online import build_scheduler
from src.data.alfworld_data import (
    GameExample,
    default_data_root,
    discover_games,
)
from src.envs.environment import ALFWorldEnvironment
from src.evolution.capabilities import build_active_agent
from src.evolution.feedback_router import FeedbackRouter
from src.evolution.unified_controller import UnifiedFeedbackController


ROOT = Path("outputs/unified_evolution_v2/online_batch_01")
PROTOCOL_PATH = ROOT / "protocol.json"
SEED = 20261116

CODE_PATHS = (
    "scripts/run_unified_batch.py",
    "scripts/run_unified_online.py",
    "src/agents/react_agent.py",
    "src/evolution/capabilities.py",
    "src/evolution/feedback_router.py",
    "src/evolution/unified_controller.py",
    "src/evolution/online_scheduler.py",
    "src/evolution/online_memory_handler.py",
    "src/evolution/online_skill_handler.py",
    "src/analysis/failure_analyzer.py",
    "src/counterfactual/verifier.py",
    "src/memory/updater.py",
)


def select_games(games, excluded, seed):
    available = [
        game
        for game in games
        if (
            game.split == "train"
            and game.id.startswith("train:")
            and game.id not in excluded
            and game.task_type in TASK_TYPES
        )
    ]

    available.sort(key=lambda game: game.id)
    random.Random(seed).shuffle(available)

    selected = []
    seen_types = set()
    seen_ids = set()

    for game in available:
        if (
            game.task_type in seen_types
            or game.id in seen_ids
        ):
            continue

        selected.append(game)
        seen_types.add(game.task_type)
        seen_ids.add(game.id)

        if len(selected) == len(TASK_TYPES):
            break

    if seen_types != set(TASK_TYPES):
        raise RuntimeError(
            "Insufficient unused Train games for all task types"
        )

    return selected


def collect_sources(old):
    audit = read_json(AUDIT_PATH)

    excluded = {
        row["game_id"]
        for row in audit["records"]
    }

    excluded.update(
        game["id"] for game in old["games"]
    )

    sources = set(exclusion_sources())

    for path in Path("outputs/unified_evolution_v2").glob(
        "*/protocol.json"
    ):
        if path.resolve() != PROTOCOL_PATH.resolve():
            sources.add(path)

    hashes = {}

    for path in sorted(sources):
        if not path.is_file():
            continue

        hashes[str(path)] = sha(path)
        collect_train_ids(read_json(path), excluded)

    return excluded, hashes


def create_plan():
    old, caps = verify_old_protocol()

    if PROTOCOL_PATH.exists():
        return verify_batch()

    if old["source_split"] != "train":
        raise RuntimeError("Unexpected historical source split")

    excluded, source_hashes = collect_sources(old)

    games = discover_games(
        default_data_root(),
        "train",
    )

    selected = select_games(games, excluded, SEED)

    protocol = {
        "schema": "unified_online_batch_v1",
        "status": "FROZEN",
        "source_split": "train",
        "seed": SEED,
        "model": old["model"],
        "max_steps": old["max_steps"],
        "top_k": old["top_k"],
        "state_tracking": True,
        "memory_version": caps.memory_info.version,
        "memory_sha256": caps.memory_info.memory_hash,
        "active_skill_id": caps.skill_id,
        "old_protocol_sha256": sha(OLD_PROTOCOL_PATH),
        "historical_audit_sha256": sha(AUDIT_PATH),
        "source_hashes": source_hashes,
        "code_hashes": {
            path: sha(path)
            for path in CODE_PATHS
        },
        "reference_actions_by_type":
            old["reference_actions_by_type"],
        "excluded_game_count": len(excluded),
        "games": [
            {
                "id": game.id,
                "split": game.split,
                "task_type": game.task_type,
                "game_file": str(game.game_file),
                "game_sha256": sha(game.game_file),
            }
            for game in selected
        ],
    }

    save_immutable(PROTOCOL_PATH, protocol)
    return verify_batch()


def verify_batch():
    old, caps = verify_old_protocol()
    protocol = read_json(PROTOCOL_PATH)

    if (
        protocol["schema"] != "unified_online_batch_v1"
        or protocol["status"] != "FROZEN"
        or protocol["source_split"] != "train"
        or len(protocol["games"]) != len(TASK_TYPES)
    ):
        raise RuntimeError("Invalid batch protocol")

    if (
        protocol["old_protocol_sha256"] != sha(OLD_PROTOCOL_PATH)
        or protocol["historical_audit_sha256"]
        != sha(AUDIT_PATH)
        or protocol["model"] != old["model"]
        or protocol["max_steps"] != old["max_steps"]
        or protocol["top_k"] != old["top_k"]
        or protocol["reference_actions_by_type"]
        != old["reference_actions_by_type"]
    ):
        raise RuntimeError("Reference protocol changed")

    if (
        protocol["memory_version"]
        != caps.memory_info.version
        or protocol["memory_sha256"]
        != caps.memory_info.memory_hash
        or protocol["active_skill_id"] != caps.skill_id
    ):
        raise RuntimeError("Active capabilities changed")

    for path, expected in protocol["source_hashes"].items():
        if sha(path) != expected:
            raise RuntimeError(f"Source changed: {path}")

    for path, expected in protocol["code_hashes"].items():
        if sha(path) != expected:
            raise RuntimeError(f"Code changed: {path}")

    excluded, _ = collect_sources(old)
    ids = set()
    task_types = set()

    for game in protocol["games"]:
        if (
            game["split"] != "train"
            or not game["id"].startswith("train:")
            or game["id"] in ids
            or game["id"] in excluded
            or game["task_type"] in task_types
            or sha(game["game_file"]) != game["game_sha256"]
        ):
            raise RuntimeError(
                f"Frozen game mismatch: {game['id']}"
            )

        ids.add(game["id"])
        task_types.add(game["task_type"])

    if task_types != set(TASK_TYPES):
        raise RuntimeError("Task type coverage mismatch")

    return protocol, caps


def game_example(raw):
    return GameExample(
        id=raw["id"],
        split=raw["split"],
        game_file=Path(raw["game_file"]),
        task_type=raw["task_type"],
    )


def run_episodes(protocol, caps, limit):
    if not 1 <= limit <= len(protocol["games"]):
        raise ValueError("Invalid episode limit")

    controller = UnifiedFeedbackController(caps, ROOT)

    router = FeedbackRouter(
        protocol["reference_actions_by_type"]
    )

    llm = None
    created = 0

    for index, raw in enumerate(protocol["games"][:limit]):
        game = game_example(raw)

        path = (
            ROOT / "episodes"
            / f"episode_{index:02d}.json"
        )

        fresh = not path.is_file()

        if fresh:
            if os.getenv("OPENAI_MODEL") != protocol["model"]:
                raise RuntimeError("Unexpected LLM model")

            if llm is None:
                from src.llm.client import LLMClient
                llm = LLMClient()

            env = ALFWorldEnvironment(
                str(game.game_file),
                max_steps=protocol["max_steps"],
            )

            agent = build_active_agent(
                llm,
                caps,
                max_steps=protocol["max_steps"],
                top_k=protocol["top_k"],
            )

            try:
                trace = agent.run(env)
            finally:
                env.close()

            temporary = path.with_suffix(".json.tmp")
            temporary.parent.mkdir(
                parents=True, exist_ok=True
            )

            trace.save_json(str(temporary))
            os.replace(temporary, path)
            created += 1

        event = controller.process_saved_episode(
            episode_id=f"batch_{index:02d}",
            game=game,
            trajectory_path=path,
            router=router,
            phase="learning",
        )

        decision = event["decision"]

        print(
            f"[{'new' if fresh else 'cached'}] "
            f"{index:02d} "
            f"success={decision['success']} "
            f"env={decision['environment_actions']} "
            f"route={decision['route']}"
        )

    print("New episodes executed:", created)


def collect_jobs(protocol, caps):
    controller = UnifiedFeedbackController(caps, ROOT)

    controller.review_queue()

    jobs = []
    events = sorted(
        controller.event_dir.glob("*.json")
    )

    for path in events:
        event = read_json(path)
        episode_id = event["episode_id"]

        match = re.fullmatch(
            r"batch_(\d{2})",
            episode_id,
        )

        if not match or path.stem != episode_id:
            raise RuntimeError("Unknown batch event")

        index = int(match.group(1))

        if index >= len(protocol["games"]):
            raise RuntimeError("Unexpected batch episode")

        game = protocol["games"][index]

        trace_path = (
            ROOT / "episodes"
            / f"episode_{index:02d}.json"
        )

        if (
            event["game_id"] != game["id"]
            or event["task_type"] != game["task_type"]
            or event["phase"] != "learning"
            or event["memory_version"]
            != caps.memory_info.version
            or event["memory_sha256"]
            != caps.memory_info.memory_hash
            or event["active_skill_id"] != caps.skill_id
            or Path(event["trajectory_path"]).resolve()
            != trace_path.resolve()
            or event["trajectory_sha256"] != sha(trace_path)
        ):
            raise RuntimeError(
                f"Batch event mismatch: {episode_id}"
            )

        route = event["decision"]["route"]

        if route == "ARCHIVE":
            continue

        if route not in ("MEMORY_REVIEW", "SKILL_REVIEW"):
            raise RuntimeError("Unknown learning route")

        jobs.append({
            "job_id": episode_id,
            "game_id": game["id"],
            "route": route,
            "handler": (
                "failure_analyzer"
                if route == "MEMORY_REVIEW"
                else "skill_inefficiency_analysis"
            ),
            "source_scope": "ONLINE_TRAIN_DEVELOPMENT",
            "status": "PENDING",
            "trajectory_path": str(trace_path),
            "trajectory_sha256": sha(trace_path),
            "event_path": str(path),
            "event_sha256": sha(path),
        })

    return jobs, len(events)


def review(protocol, caps, max_new_jobs, max_decisions, max_calls):
    jobs, event_count = collect_jobs(protocol, caps)

    scheduler = build_scheduler(
        jobs,
        caps,
        protocol,
        ROOT,
        enable_handlers=max_new_jobs > 0,
        max_continuation_decisions=max_decisions,
        max_llm_calls=max_calls,
    )

    results = scheduler.run(
        max_new_jobs=max_new_jobs
    )

    counts = Counter(row["state"] for row in results)

    print("\n========== UNIFIED BATCH REVIEW ==========")
    print("Saved episodes:", event_count)
    print("Review jobs:", len(jobs))

    for item in results:
        print(
            item["job_id"],
            item["route"],
            item["state"],
            item["outcome_status"],
        )

    print("Cached:", counts["CACHED"])
    print("Pending:", counts["PENDING"])
    print("Executed:", counts["EXECUTED"])
    print("Automatic promotion: disabled")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--stage",
        required=True,
        choices=("plan", "run", "review", "report"),
    )
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument(
        "--max-new-jobs", type=int, default=0
    )
    parser.add_argument(
        "--max-continuation-decisions",
        type=int,
        default=12,
    )
    parser.add_argument(
        "--max-llm-calls",
        type=int,
        default=16,
    )

    args = parser.parse_args()

    if args.stage == "plan":
        protocol, caps = create_plan()

        print("========== NEW ONLINE BATCH PLAN ==========")
        print("Source: Train")
        print("Memory:", caps.memory_info.version)
        print("Active skill:", caps.skill_id)
        print(
            "Excluded historical IDs:",
            protocol["excluded_game_count"],
        )

        for index, game in enumerate(protocol["games"]):
            print(
                index,
                game["task_type"],
                game["id"],
            )

        print("Frozen protocol:", PROTOCOL_PATH)
        print("LLM calls: 0")
        return

    protocol, caps = verify_batch()

    if args.stage == "run":
        if args.max_new_jobs != 0:
            parser.error(
                "Use --stage review to process new Review jobs"
            )
        run_episodes(protocol, caps, args.limit)

    elif args.stage == "review":
        review(
            protocol,
            caps,
            args.max_new_jobs,
            args.max_continuation_decisions,
            args.max_llm_calls,
        )

    else:
        review(
            protocol,
            caps,
            0,
            args.max_continuation_decisions,
            args.max_llm_calls,
        )


if __name__ == "__main__":
    main()
