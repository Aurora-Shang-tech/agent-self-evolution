import argparse
import hashlib
import json
import os
import random
from collections import defaultdict
from pathlib import Path

from src.data.alfworld_data import (
    GameExample,
    default_data_root,
    discover_games,
)
from src.evolution.capabilities import (
    build_active_agent,
    load_active_capabilities,
)
from src.evolution.feedback_router import FeedbackRouter
from src.evolution.unified_controller import UnifiedFeedbackController
from src.envs.environment import ALFWorldEnvironment


ROOT = Path("outputs/unified_evolution_v1/online_pilot_v1")
PROTOCOL_PATH = ROOT / "protocol.json"
AUDIT_PATH = Path(
    "outputs/unified_evolution_v1/feedback_routing_audit.json"
)
MEMORY_DIR = Path("outputs/continual_32/runs/fixed_32")
REGISTRY_PATH = Path(
    "outputs/skill_learning_v1/skills/registry/index.json"
)

TASK_TYPES = (
    "look_at_obj_in_light",
    "pick_and_place_simple",
    "pick_clean_then_place_in_recep",
    "pick_cool_then_place_in_recep",
    "pick_heat_then_place_in_recep",
    "pick_two_obj_and_place",
)

CODE_PATHS = (
    "scripts/run_unified_pilot.py",
    "src/agents/react_agent.py",
    "src/evolution/capabilities.py",
    "src/evolution/feedback_router.py",
    "src/evolution/unified_controller.py",
    "src/evolution/version_store.py",
    "src/skills/search_skill.py",
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_immutable(path, data):
    path = Path(path)
    content = json.dumps(
        data, ensure_ascii=False, indent=2
    ) + "\n"

    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        if path.read_text(encoding="utf-8") != content:
            raise RuntimeError(f"Frozen file differs: {path}")
        return

    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def collect_train_ids(data, result):
    if isinstance(data, dict):
        for value in data.values():
            collect_train_ids(value, result)
    elif isinstance(data, list):
        for value in data:
            collect_train_ids(value, result)
    elif isinstance(data, str) and data.startswith("train:"):
        result.add(data)


def exclusion_sources():
    paths = set()

    manifest_dir = Path("outputs/manifests")
    paths.update(manifest_dir.glob("*.json"))

    for directory in Path("outputs").glob("continual*"):
        for name in ("validation.json", "final_holdout.json"):
            path = directory / name
            if path.is_file():
                paths.add(path)

        paths.update(directory.glob("learning_round_*.json"))

    skill_root = Path("outputs/skill_learning_v1")
    paths.update(skill_root.rglob("protocol.json"))

    return sorted(
        path for path in paths if path.is_file()
    )


def capabilities():
    return load_active_capabilities(
        MEMORY_DIR, REGISTRY_PATH
    )


def plan():
    if PROTOCOL_PATH.exists():
        raise RuntimeError(
            "Protocol already frozen; use --stage report"
        )

    caps = capabilities()

    if (
        caps.memory_info.version != "v000"
        or caps.memory_info.memory_size != 32
        or caps.skill_id is not None
    ):
        raise RuntimeError("Unexpected active capabilities")

    audit = read_json(AUDIT_PATH)

    if (
        audit["summary"]["total"] != 60
        or audit["summary"]["successes"] != 57
    ):
        raise RuntimeError("Unexpected historical audit")

    references = defaultdict(list)

    for record in audit["records"]:
        if record["success"]:
            references[record["task_type"]].append(
                record["environment_actions"]
            )

    if any(
        len(references[task_type]) < 5
        for task_type in TASK_TYPES
    ):
        raise RuntimeError("Insufficient frozen references")

    excluded = {
        record["game_id"]
        for record in audit["records"]
    }

    source_hashes = {}

    for path in exclusion_sources():
        source_hashes[str(path)] = sha(path)
        collect_train_ids(read_json(path), excluded)

    games = discover_games(default_data_root(), "train")

    available = [
        game for game in games
        if game.id not in excluded
    ]

    random.Random(20261115).shuffle(available)

    selected = []
    selected_types = set()

    for game in available:
        if game.task_type in selected_types:
            continue

        selected.append(game)
        selected_types.add(game.task_type)

        if len(selected) == len(TASK_TYPES):
            break

    if selected_types != set(TASK_TYPES):
        raise RuntimeError("Not enough unused Train games")

    protocol = {
        "schema": "unified_online_pilot_v1",
        "status": "FROZEN",
        "seed": 20261115,
        "model": "qwen3.8-chat",
        "max_steps": 50,
        "top_k": 3,
        "state_tracking": True,
        "source_split": "train",
        "memory_version": caps.memory_info.version,
        "memory_hash": caps.memory_info.memory_hash,
        "active_skill_id": caps.skill_id,
        "audit_sha256": sha(AUDIT_PATH),
        "memory_snapshot_sha256": sha(
            MEMORY_DIR / "versions/v000.json"
        ),
        "memory_state_sha256": sha(
            MEMORY_DIR / "state.json"
        ),
        "skill_registry_sha256": sha(REGISTRY_PATH),
        "source_hashes": source_hashes,
        "code_hashes": {
            path: sha(path)
            for path in CODE_PATHS
        },
        "reference_actions_by_type": dict(references),
        "games": [
            {
                "id": game.id,
                "split": game.split,
                "game_file": str(game.game_file),
                "game_sha256": sha(game.game_file),
                "task_type": game.task_type,
            }
            for game in selected
        ],
    }

    save_immutable(PROTOCOL_PATH, protocol)

    print("========== UNIFIED PILOT PLAN ==========")
    print("Memory:", caps.memory_info.version)
    print("Active skill:", caps.skill_id)
    print("Selected games:", len(selected))
    print("Excluded historical games:", len(excluded))

    for index, game in enumerate(selected):
        print(index, game.task_type, game.id)

    print("Protocol:", PROTOCOL_PATH)
    print("LLM calls: 0")


def verify_protocol():
    protocol = read_json(PROTOCOL_PATH)

    if (
        protocol["schema"] != "unified_online_pilot_v1"
        or protocol["status"] != "FROZEN"
        or len(protocol["games"]) != 6
    ):
        raise ValueError("Invalid frozen protocol")

    if protocol["audit_sha256"] != sha(AUDIT_PATH):
        raise RuntimeError("Historical audit changed")

    for path, expected in protocol["source_hashes"].items():
        if sha(path) != expected:
            raise RuntimeError(f"Manifest changed: {path}")

    for path, expected in protocol["code_hashes"].items():
        if sha(path) != expected:
            raise RuntimeError(f"Code changed: {path}")

    if sha(MEMORY_DIR / "versions/v000.json") != (
        protocol["memory_snapshot_sha256"]
    ):
        raise RuntimeError("Memory snapshot changed")

    if sha(MEMORY_DIR / "state.json") != (
        protocol["memory_state_sha256"]
    ):
        raise RuntimeError("Memory state changed")

    if sha(REGISTRY_PATH) != protocol["skill_registry_sha256"]:
        raise RuntimeError("Skill registry changed")

    caps = capabilities()

    if (
        caps.memory_info.version != protocol["memory_version"]
        or caps.memory_info.memory_hash != protocol["memory_hash"]
        or caps.skill_id != protocol["active_skill_id"]
    ):
        raise RuntimeError("Active capabilities changed")

    ids = set()

    for game in protocol["games"]:
        if (
            game["split"] != "train"
            or game["id"] in ids
            or sha(game["game_file"]) != game["game_sha256"]
        ):
            raise RuntimeError("Frozen game mismatch")

        ids.add(game["id"])

    return protocol, caps


def run(limit):
    protocol, caps = verify_protocol()

    if os.getenv("OPENAI_MODEL") != protocol["model"]:
        raise RuntimeError("Unexpected LLM model")

    if not 1 <= limit <= 6:
        raise ValueError("limit must be between 1 and 6")

    router = FeedbackRouter(
        protocol["reference_actions_by_type"]
    )

    controller = UnifiedFeedbackController(caps, ROOT)

    llm = None
    created = 0

    for index, raw in enumerate(protocol["games"][:limit]):
        game = GameExample(
            id=raw["id"],
            split="train",
            game_file=Path(raw["game_file"]),
            task_type=raw["task_type"],
        )

        trace_path = (
            ROOT / "episodes"
            / f"episode_{index:02d}.json"
        )

        if not trace_path.exists():
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

            trace_path.parent.mkdir(
                parents=True, exist_ok=True
            )

            temporary = trace_path.with_name(
                trace_path.name + ".tmp"
            )

            trace.save_json(str(temporary))
            os.replace(temporary, trace_path)
            created += 1

        event = controller.process_saved_episode(
            episode_id=f"fresh_{index:02d}",
            game=game,
            trajectory_path=trace_path,
            router=router,
            phase="learning",
        )

        decision = event["decision"]

        print(
            f"[{'new' if created else 'cached'}] "
            f"{index:02d} "
            f"success={decision['success']} "
            f"env={decision['environment_actions']} "
            f"route={decision['route']}"
        )

    print("New episodes executed:", created)
    print("Review queue:", len(controller.review_queue()))


def report():
    protocol, caps = verify_protocol()

    controller = UnifiedFeedbackController(caps, ROOT)

    events = sorted(
        (ROOT / "events").glob("fresh_*.json")
    ) if (ROOT / "events").exists() else []

    print("========== UNIFIED ONLINE PILOT ==========")
    print("Planned games:", len(protocol["games"]))
    print("Saved events:", len(events))

    for event_path in events:
        event = read_json(event_path)
        print(
            event["episode_id"],
            event["decision"]["route"],
            event["decision"]["environment_actions"],
        )

    print("Review jobs:", len(controller.review_queue()))
    print("Active skill:", caps.skill_id)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=("plan", "run", "report"),
        required=True,
    )
    parser.add_argument("--limit", type=int, default=2)

    args = parser.parse_args()

    if args.stage == "plan":
        plan()
    elif args.stage == "run":
        run(args.limit)
    else:
        report()


if __name__ == "__main__":
    main()
