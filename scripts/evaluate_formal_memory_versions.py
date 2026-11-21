import argparse
import hashlib
import json
import os
from pathlib import Path

from scripts.evaluation.evaluate_memory_evolution import (
    build_record,
    build_summary,
    paired_comparison,
)
from src.agents.react_agent import ReActAgent
from src.core.trajectory import Trajectory
from src.data.manifest import load_manifest
from src.envs.environment import ALFWorldEnvironment
from src.evolution.version_store import (
    load_memory_version,
)
from src.llm.client import LLMClient


ROOT = Path("outputs/continual_formal_large_v1")
RUN = ROOT / "runs/dynamic_large_v1"
OUTPUT = ROOT / "final_pair_v000_v001"

VERSIONS = {
    "fixed": "v000",
    "dynamic": "v001",
}

SPLITS = ("valid_seen", "valid_unseen")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp, path)


def context():
    protocol = read(ROOT / "protocol.json")
    state = read(RUN / "state.json")
    config = read(RUN / "run_config.json")

    if (
        state["completed_rounds"] != 10
        or state["active_version"] != "v001"
    ):
        raise RuntimeError("Unexpected final active version")

    if sha(ROOT / "protocol.json") != config["protocol_sha256"]:
        raise RuntimeError("Experiment protocol changed")

    if os.getenv("OPENAI_MODEL") != protocol["model"]:
        raise RuntimeError("OPENAI_MODEL mismatch")

    stores = {}

    for name, version in VERSIONS.items():
        store, info = load_memory_version(RUN, version)
        stores[name] = (store, info)

    return protocol, stores


def evaluate(split, name, protocol, stores, execute):
    version = VERSIONS[name]
    store, info = stores[name]

    manifest = ROOT / "final_holdout.json"

    if sha(manifest) != protocol["manifests"][manifest.name]:
        raise RuntimeError("Final manifest changed")

    games = [
        game for game in load_manifest(manifest)
        if game.split == split
    ]

    if len(games) != protocol["final_split_sizes"][split]:
        raise RuntimeError("Final task count mismatch")

    folder = OUTPUT / split / name

    config = {
        "schema": "formal_memory_final_pair_v1",
        "model": protocol["model"],
        "split": split,
        "method": name,
        "memory_version": version,
        "memory_sha256": info.memory_hash,
        "memory_file_sha256": sha(
            RUN / "versions" / f"{version}.json"
        ),
        "manifest_sha256": sha(manifest),
        "max_steps": protocol["max_steps"],
        "top_k": protocol["top_k"],
        "state_tracking": True,
        "source_hashes": {
            path: sha(path)
            for path in (
                "src/agents/react_agent.py",
                "src/core/protocol.py",
                "src/memory/store.py",
                "src/memory/state_tracker.py",
                "src/envs/environment.py",
            )
        },
    }

    config_path = folder / "config.json"

    if config_path.exists():
        if read(config_path) != config:
            raise RuntimeError("Evaluation config changed")
    elif execute:
        save(config_path, config)

    llm = None
    records = []

    for index, game in enumerate(games):
        path = folder / f"episode_{index:03d}.json"

        if path.exists():
            trace = Trajectory.load_json(str(path))
        elif not execute:
            continue
        else:
            expected = protocol["game_asset_sha256"].get(
                str(game.game_file)
            )

            if expected is None or sha(game.game_file) != expected:
                raise RuntimeError("Final game file changed")

            if llm is None:
                llm = LLMClient()

            env = ALFWorldEnvironment(
                str(game.game_file),
                max_steps=protocol["max_steps"],
            )

            agent = ReActAgent(
                llm,
                max_steps=protocol["max_steps"],
                experience_store=store,
                experience_top_k=protocol["top_k"],
                use_state_tracking=True,
            )

            try:
                trace = agent.run(env)
            finally:
                env.close()

            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".json.tmp")
            trace.save_json(str(tmp))
            os.replace(tmp, path)

        if (
            Path(trace.gamefile).resolve()
            != Path(game.game_file).resolve()
        ):
            raise RuntimeError("Cached trajectory game mismatch")

        records.append(build_record(index, game, trace))

        if execute and (index + 1) % 20 == 0:
            print(
                f"[{split}][{name}] {index + 1}/{len(games)}",
                flush=True,
            )

    summary = build_summary(records, len(games))

    if execute:
        save(folder / "summary.json", summary)

    print(
        f"{split:12} {name:7} "
        f"{summary['completed']}/{summary['total']} "
        f"success={summary['successes']}"
    )

    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=("run", "report"),
        required=True,
    )
    args = parser.parse_args()

    protocol, stores = context()
    results = {}

    for split in SPLITS:
        results[split] = {}

        for name in VERSIONS:
            results[split][name] = evaluate(
                split,
                name,
                protocol,
                stores,
                execute=args.stage == "run",
            )

        fixed = results[split]["fixed"]
        dynamic = results[split]["dynamic"]

        if fixed["finished"] and dynamic["finished"]:
            comparison = paired_comparison(
                "fixed_v000",
                fixed,
                "dynamic_v001",
                dynamic,
            )

            print("\nPAIR:", split)
            print(json.dumps(comparison, indent=2))

            if args.stage == "run":
                save(
                    OUTPUT / split / "comparison.json",
                    comparison,
                )


if __name__ == "__main__":
    main()
