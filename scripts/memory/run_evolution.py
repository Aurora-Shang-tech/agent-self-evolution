
import argparse
import hashlib
import json
import os
import re
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from scripts.extract_experiences import load_verification
from src.core.trajectory import Trajectory
from src.llm.client import LLMClient
from src.memory.updater import (
    ComparativeExtractor,
    ExperienceUpdater,
    LessonCandidate,
    MemoryItem,
    UpdateProposal,
)

BASELINE_DIR = Path("outputs/baseline_qwen3.8-chat_train_seed42_1000")
VERIFICATION_DIR = Path("outputs/counterfactual_qwen3.8-chat_train_seed42_1000")
OUTPUT_ROOT = Path("outputs/memory_evolution")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def memory_hash(memory):
    return hashlib.sha256(
        json.dumps(
            [asdict(item) for item in memory],
            ensure_ascii=False,
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temp, path)


def source_hashes(verification_path, index):
    baseline_path = BASELINE_DIR / verification_path.name
    paths = [baseline_path, verification_path]

    data = json.loads(
        verification_path.read_text(encoding="utf-8")
    )

    if index is not None:
        saved_path = data["candidates"][index].get(
            "trajectory_path"
        )

        if saved_path:
            paths.append(Path(saved_path))

    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)

    return {
        str(path): digest(path)
        for path in paths
    }


def apply_proposal(memory, candidate, proposal, event_id):
    if proposal.memory_snapshot_sha256 != memory_hash(memory):
        raise ValueError(f"Stale memory snapshot: {event_id}")

    operation = proposal.operation

    target = next(
        (
            i for i, item in enumerate(memory)
            if item.id == proposal.target_id
        ),
        None,
    )

    if operation == "ADD":
        new_id = f"evo_{event_id}"

        if any(item.id == new_id for item in memory):
            raise ValueError(f"Duplicate memory ID: {event_id}")

        if any(
            item.lesson.strip().casefold()
            == proposal.new_lesson.strip().casefold()
            for item in memory
        ):
            raise ValueError(
                f"ADD duplicates an existing lesson: {event_id}"
            )

        memory.append(
            MemoryItem(
                id=new_id,
                lesson=proposal.new_lesson,
                source_task=candidate.source_task,
            )
        )

    elif operation == "EDIT":
        if target is None:
            raise ValueError(f"EDIT target missing: {event_id}")

        old = memory[target]

        memory[target] = MemoryItem(
            id=old.id,
            lesson=proposal.new_lesson,
            source_task=old.source_task,
        )

    elif operation == "REMOVE":
        if target is None:
            raise ValueError(f"REMOVE target missing: {event_id}")

        memory.pop(target)

    elif operation != "KEEP":
        raise ValueError(f"Invalid operation: {operation}")


def catalog():
    paths = sorted(
        path
        for path in VERIFICATION_DIR.glob("episode_*.json")
        if re.fullmatch(r"episode_\d+", path.stem)
    )

    if not paths:
        raise RuntimeError(
            "No train verification episode files found"
        )

    entries = []

    for path in paths:
        data = json.loads(
            path.read_text(encoding="utf-8")
        )

        indices = (
            range(len(data["candidates"]))
            if data["candidates"]
            else [None]
        )

        for index in indices:
            key = (
                f"{path.stem}_c{index + 1:03d}"
                if index is not None
                else f"{path.stem}_c000"
            )

            entries.append((path, index, key))

    return paths, entries


def replay_event(memory, event, verification_path, index, key):
    if event.get("key") != key:
        raise ValueError(f"Event ID mismatch: {key}")

    if event.get("source_hashes") != source_hashes(
        verification_path, index
    ):
        raise ValueError(f"Input data changed: {key}")

    if event.get("memory_before_sha256") != memory_hash(memory):
        raise ValueError(
            f"Event order or memory mismatch: {key}"
        )

    if event["status"] in {"PROVISIONAL", "KEEP"}:
        candidate = LessonCandidate(**event["candidate"])
        proposal = UpdateProposal(**event["proposal"])

        if (
            event["status"] == "KEEP"
        ) != (
            proposal.operation == "KEEP"
        ):
            raise ValueError(
                f"Event status inconsistent: {key}"
            )

        apply_proposal(memory, candidate, proposal, key)

    elif event["status"] not in {"ABSTAIN", "SKIPPED"}:
        raise ValueError(
            f"Unknown event status: {key}"
        )

    if event.get("memory_after_sha256") != memory_hash(memory):
        raise ValueError(
            f"Replay result differs: {key}"
        )


def produce_event(
    memory,
    verification_path,
    index,
    key,
    extractor,
    updater,
    top_k,
):
    before = memory_hash(memory)

    event = {
        "key": key,
        "source_hashes": source_hashes(
            verification_path, index
        ),
        "memory_before_sha256": before,
    }

    if index is None:
        event.update(
            status="SKIPPED",
            reason="No counterfactual candidates",
        )

    else:
        verification = load_verification(
            verification_path
        )

        branch = verification.results[index]

        if (
            not branch.success
            or branch.trajectory is None
            or not branch.trajectory.steps
        ):
            event.update(
                status="SKIPPED",
                reason=(
                    "No successful saved continuation "
                    "for this branch"
                ),
                branch_success=branch.success,
            )

        else:
            baseline = Trajectory.load_json(
                str(
                    BASELINE_DIR
                    / verification_path.name
                )
            )

            candidate = extractor.extract(
                baseline,
                verification,
                candidate_index=index,
            )

            event["candidate"] = asdict(candidate)

            if candidate.decision == "ABSTAIN":
                event.update(
                    status="ABSTAIN",
                    reason=candidate.reason,
                )

            else:
                proposal = updater.propose(
                    candidate,
                    memory,
                    top_k=top_k,
                )

                event["proposal"] = asdict(proposal)

                apply_proposal(
                    memory,
                    candidate,
                    proposal,
                    key,
                )

                event["status"] = (
                    "KEEP"
                    if proposal.operation == "KEEP"
                    else "PROVISIONAL"
                )

    event["memory_after_sha256"] = memory_hash(memory)

    return event


def write_snapshot(output_dir, memory, events):
    counts = Counter(
        event["status"]
        for event in events
    )

    operations = Counter(
        event["proposal"]["operation"]
        for event in events
        if event.get("proposal") is not None
    )

    atomic_json(
        output_dir / "working_memory.json",
        {
            "status": "PROVISIONAL_UNVALIDATED",
            "processed_branches": len(events),
            "memory_size": len(memory),
            "event_counts": dict(counts),
            "operation_counts": dict(operations),
            "memory": [
                asdict(item)
                for item in memory
            ],
            "memory_sha256": memory_hash(memory),
        },
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--run-name",
        default="dev",
    )

    parser.add_argument(
        "--limit",
        type=int,
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--replay-only",
        action="store_true",
    )

    args = parser.parse_args()

    if not re.fullmatch(
        r"[a-zA-Z0-9][a-zA-Z0-9_.-]*",
        args.run_name,
    ):
        raise ValueError("Invalid run name")

    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be >= 1")

    if args.top_k < 1:
        raise ValueError("--top-k must be >= 1")

    output_dir = OUTPUT_ROOT / args.run_name
    event_dir = output_dir / "events"
    manifest_path = output_dir / "manifest.json"

    paths, entries = catalog()

    entry_map = {
        key: (path, index)
        for path, index, key in entries
    }

    all_keys = [
        key
        for _, _, key in entries
    ]

    config = {
        "schema": "evolution_provisional_v1",
        "model": os.environ.get("OPENAI_MODEL"),
        "top_k": args.top_k,
        "code_hashes": {
            "src/memory/updater.py": digest(
                Path("src/memory/updater.py")
            ),
            "scripts/memory/run_evolution.py": digest(
                Path(__file__)
            ),
        },
        "baseline_dir": str(BASELINE_DIR),
        "verification_dir": str(VERIFICATION_DIR),
    }

    if manifest_path.exists():
        old_config = json.loads(
            manifest_path.read_text(encoding="utf-8")
        )

        if old_config != config:
            raise ValueError(
                "Run configuration or code changed; "
                "use another --run-name"
            )

    else:
        if args.replay_only:
            raise FileNotFoundError(manifest_path)

        if event_dir.exists() and list(
            event_dir.glob("*.json")
        ):
            raise ValueError(
                "Existing events without manifest"
            )

        atomic_json(manifest_path, config)

    event_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    existing = sorted(
        path.stem
        for path in event_dir.glob("*.json")
    )

    if existing != all_keys[:len(existing)]:
        raise ValueError(
            "Saved event sequence is not a prefix "
            "of current train data"
        )

    memory = []
    events = []

    for key in existing:
        path, index = entry_map[key]

        event = json.loads(
            (event_dir / f"{key}.json").read_text(
                encoding="utf-8"
            )
        )

        replay_event(
            memory,
            event,
            path,
            index,
            key,
        )

        events.append(event)

    write_snapshot(
        output_dir,
        memory,
        events,
    )

    print(
        f"Replayed {len(existing)} branches. "
        f"Working memory: {len(memory)}"
    )

    if args.replay_only:
        return

    selected_episodes = {
        path.stem
        for path in (
            paths[:args.limit]
            if args.limit is not None
            else paths
        )
    }

    processed = set(existing)

    pending = [
        entry
        for entry in entries
        if entry[2] not in processed
        and entry[0].stem in selected_episodes
    ]

    if not pending:
        print("No new branches to process")
        return

    llm = LLMClient()
    extractor = ComparativeExtractor(llm)
    updater = ExperienceUpdater(llm)

    for i, (path, index, key) in enumerate(
        pending, 1
    ):
        event = produce_event(
            memory,
            path,
            index,
            key,
            extractor,
            updater,
            args.top_k,
        )

        atomic_json(
            event_dir / f"{key}.json",
            event,
        )

        events.append(event)

        write_snapshot(
            output_dir,
            memory,
            events,
        )

        operation = (
            event.get("proposal") or {}
        ).get("operation", "-")

        print(
            f"[{i}/{len(pending)}] "
            f"{key}: "
            f"{event['status']} "
            f"{operation}; "
            f"memory={len(memory)}"
        )

    print(
        f"Saved: "
        f"{output_dir / 'working_memory.json'}"
    )

    print(
        "All proposed changes remain unvalidated "
        "and must not be deployed."
    )


if __name__ == "__main__":
    main()

