import argparse
import hashlib
import json
import os
from dataclasses import asdict
from pathlib import Path
from statistics import mean

from src.agents.react_agent import ReActAgent
from src.envs.environment import ALFWorldEnvironment
from src.evolution.controller import experience_store
from src.llm.client import LLMClient
from src.memory.updater import MemoryItem


ROOT = Path("outputs/skill_learning_v1/online_abc_v1")
PROTOCOL_PATH = ROOT / "protocol.json"

EXPECTED_PROTOCOL_SHA = (
    "8d384c8d302daa489f72ea9367859f51cb904"
    "aab9b82da9a0c75ae2ae9e13844"
)

MODES = ("baseline", "hint", "execute")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)

    content = json.dumps(
        data, ensure_ascii=False, indent=2
    ) + "\n"

    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def prepare():
    assert sha(PROTOCOL_PATH) == EXPECTED_PROTOCOL_SHA, (
        "Frozen experiment protocol changed"
    )

    protocol = load(PROTOCOL_PATH)

    assert protocol["status"] == "FROZEN"
    assert protocol["methods"] == list(MODES)
    assert len(protocol["dev"]) == 12
    assert len(protocol["final_holdout"]) == 30

    dev_ids = {
        game["game_id"]
        for game in protocol["dev"]
    }
    holdout_ids = {
        game["game_id"]
        for game in protocol["final_holdout"]
    }

    assert len(dev_ids) == 12
    assert len(holdout_ids) == 30
    assert not dev_ids.intersection(holdout_ids)

    memory_path = Path(protocol["memory_path"])
    skill_path = Path(protocol["skill_path"])

    assert sha(memory_path) == protocol["memory_sha256"]
    assert sha(skill_path) == protocol["skill_sha256"]

    for name, expected in protocol["code_sha256"].items():
        assert sha(Path(name)) == expected, (
            f"Frozen code changed: {name}"
        )

    for game in protocol["dev"]:
        assert game["split"] == "train"
        assert Path(game["game_file"]).is_file()

    memory = load(memory_path)
    skill = load(skill_path)["proposal"]

    assert memory["memory_size"] == 32
    assert skill["decision"] == "CANDIDATE"

    config = {
        "experiment": "online_abc_v1",
        "protocol_sha256": sha(PROTOCOL_PATH),
        "runner_sha256": sha(Path(__file__)),
        "model": protocol["model"],
        "max_steps": protocol["max_steps"],
        "top_k": protocol["top_k"],
        "state_tracking": protocol["state_tracking"],
        "memory_sha256": protocol["memory_sha256"],
        "skill_sha256": protocol["skill_sha256"],
        "code_sha256": protocol["code_sha256"],
    }

    config_path = ROOT / "run_config.json"

    if config_path.exists():
        assert load(config_path) == config, (
            "Frozen online experiment configuration changed"
        )
    else:
        save_json(config_path, config)

    return protocol, memory, skill, sha(config_path)


def selected_indices(protocol, subset):
    if subset == "dev":
        return list(range(len(protocol["dev"])))

    selected = []
    seen_types = set()

    for index, game in enumerate(protocol["dev"]):
        task_type = game["task_type"]

        if task_type not in seen_types:
            selected.append(index)
            seen_types.add(task_type)

    assert len(selected) == 3

    return selected


def result_path(mode, index):
    return ROOT / "runs" / mode / f"task_{index:02d}.json"


def run_one(
    protocol,
    memory,
    skill,
    config_sha,
    mode,
    index,
    llm,
):
    game = protocol["dev"][index]
    path = result_path(mode, index)

    if path.exists():
        record = load(path)

        assert record["config_sha256"] == config_sha
        assert record["game_id"] == game["game_id"]
        assert record["mode"] == mode
        assert record["trajectory"]["gamefile"] == (
            str(Path(game["game_file"]).resolve())
        )

        print(
            f"[cached] task={index:02d} mode={mode}",
            flush=True,
        )

        return record

    kwargs = {}

    if mode != "baseline":
        kwargs = {
            "search_skill_mode": mode,
            "search_skill_proposal": skill,
        }

    agent = ReActAgent(
        llm,
        max_steps=protocol["max_steps"],
        experience_store=experience_store(memory),
        experience_top_k=protocol["top_k"],
        use_state_tracking=protocol["state_tracking"],
        **kwargs,
    )

    env = ALFWorldEnvironment(
        game["game_file"],
        max_steps=protocol["max_steps"],
    )

    try:
        trajectory = agent.run(env)
    finally:
        env.close()

    assert Path(trajectory.gamefile).resolve() == (
        Path(game["game_file"]).resolve()
    )

    events = agent.skill_events

    assert len(events) <= 1

    if mode == "baseline":
        assert not events

    decisions = len(trajectory.steps)
    llm_calls = agent.llm_calls

    assert 0 <= llm_calls <= decisions

    if mode in ("baseline", "hint"):
        assert llm_calls == decisions

    if mode == "execute":
        assert llm_calls == decisions - len(events)

    record = {
        "config_sha256": config_sha,
        "index": index,
        "game_id": game["game_id"],
        "task_type": game["task_type"],
        "mode": mode,
        "task": trajectory.task,
        "success": bool(trajectory.success),
        "truncated": bool(trajectory.truncated),
        "decisions": decisions,
        "environment_actions": sum(
            bool(step.valid_format)
            for step in trajectory.steps
        ),
        "llm_calls": llm_calls,
        "skill_triggered": bool(events),
        "skill_followed": (
            bool(events[0]["followed"])
            if events else False
        ),
        "skill_events": events,
        "trajectory": asdict(trajectory),
    }

    save_json(path, record)

    print(
        f"[done] task={index:02d} mode={mode} "
        f"success={record['success']} "
        f"env={record['environment_actions']} "
        f"llm={llm_calls} "
        f"triggered={record['skill_triggered']} "
        f"followed={record['skill_followed']}",
        flush=True,
    )

    return record


def report(protocol, config_sha, indices):
    records = {}

    for index in indices:
        games = {}

        for mode in MODES:
            path = result_path(mode, index)

            if not path.is_file():
                print(
                    f"Missing: task={index:02d}, mode={mode}"
                )
                return

            record = load(path)

            assert record["config_sha256"] == config_sha
            assert record["game_id"] == (
                protocol["dev"][index]["game_id"]
            )

            games[mode] = record

        assert len({
            row["task"] for row in games.values()
        }) == 1

        records[index] = games

    print("\n========== ONLINE A/B/C RESULTS ==========")

    for index, games in records.items():
        print(
            f"\n#{index:02d} {games['baseline']['task']}"
        )

        for mode in MODES:
            row = games[mode]

            print(
                f"  {mode:8s} "
                f"success={str(row['success']):5s} "
                f"env={row['environment_actions']:2d} "
                f"llm={row['llm_calls']:2d} "
                f"trigger={row['skill_triggered']} "
                f"followed={row['skill_followed']}"
            )

    summary = {
        "task_indices": indices,
        "modes": {},
        "paired": {},
    }

    for mode in MODES:
        items = [
            records[index][mode]
            for index in indices
        ]

        stats = {
            "successes": sum(
                item["success"] for item in items
            ),
            "mean_environment_actions": mean(
                item["environment_actions"]
                for item in items
            ),
            "mean_llm_calls": mean(
                item["llm_calls"]
                for item in items
            ),
            "triggers": sum(
                item["skill_triggered"] for item in items
            ),
            "followed": sum(
                item["skill_followed"] for item in items
            ),
        }

        summary["modes"][mode] = stats

    for mode in ("hint", "execute"):
        pairs = [
            (
                records[index]["baseline"],
                records[index][mode],
            )
            for index in indices
        ]

        both_success = [
            (a, b)
            for a, b in pairs
            if a["success"] and b["success"]
        ]

        summary["paired"][mode] = {
            "improved": sum(
                not a["success"] and b["success"]
                for a, b in pairs
            ),
            "regressed": sum(
                a["success"] and not b["success"]
                for a, b in pairs
            ),
            "mean_env_delta_all": mean(
                b["environment_actions"]
                - a["environment_actions"]
                for a, b in pairs
            ),
            "mean_env_delta_both_success": (
                mean(
                    b["environment_actions"]
                    - a["environment_actions"]
                    for a, b in both_success
                )
                if both_success else None
            ),
        }

    print("\n========== SUMMARY ==========")

    for mode, stats in summary["modes"].items():
        print(
            f"{mode:8s} "
            f"success={stats['successes']}/{len(indices)} "
            f"avg_env={stats['mean_environment_actions']:.2f} "
            f"avg_llm={stats['mean_llm_calls']:.2f} "
            f"triggers={stats['triggers']} "
            f"followed={stats['followed']}"
        )

    print("\nPaired comparisons:")
    print(
        json.dumps(
            summary["paired"],
            ensure_ascii=False,
            indent=2,
        )
    )

    name = (
        "pilot_summary.json"
        if len(indices) == 3
        else "dev_summary.json"
    )

    path = ROOT / name
    save_json(path, summary)
    print("Saved:", path)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--stage",
        choices=("plan", "run", "report"),
        default="plan",
    )
    parser.add_argument(
        "--set",
        choices=("pilot", "dev"),
        default="pilot",
    )

    args = parser.parse_args()

    protocol, raw_memory, skill, config_sha = prepare()
    indices = selected_indices(protocol, args.set)

    print("Protocol:", sha(PROTOCOL_PATH))
    print("Dataset:", args.set)
    print("Selected tasks:", indices)

    if args.stage == "plan":
        for index in indices:
            game = protocol["dev"][index]

            print(
                f"#{index:02d} "
                f"{game['task_type']} "
                f"{game['game_id']}"
            )

        print("LLM calls: 0")
        return

    if args.stage == "report":
        report(protocol, config_sha, indices)
        return

    if os.getenv("OPENAI_MODEL") != protocol["model"]:
        raise RuntimeError(
            f"OPENAI_MODEL must be {protocol['model']}"
        )

    memory = [
        MemoryItem(**item)
        for item in raw_memory["memory"]
    ]

    llm = LLMClient()

    for index in indices:
        for mode in MODES:
            run_one(
                protocol,
                memory,
                skill,
                config_sha,
                mode,
                index,
                llm,
            )

    report(protocol, config_sha, indices)


if __name__ == "__main__":
    main()
