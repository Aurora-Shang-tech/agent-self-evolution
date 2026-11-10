
"""Offline contextual credit from v1 paired results; fresh held-out evaluation.

prepare: no LLM, builds a frozen experiment from v1 evaluation as TRAINING data.
inspect: no LLM, checks retrieval differences on new games via ALFWorld reset.
eval: executes new paired episodes; disabled unless --max-new > 0.
report: prints persisted evaluation results.
"""

import argparse
import json
import os
import random
from pathlib import Path

from src.agents.react_agent import ReActAgent
from src.core.trajectory import Trajectory
from src.data.alfworld_data import (
    GameExample,
    SUPPORTED_TASK_TYPES,
    default_data_root,
    discover_games,
)
from src.data.manifest import load_manifest
from src.envs.environment import ALFWorldEnvironment
from src.llm.client import LLMClient
from src.memory.contextual_utility import ContextualCredit
from src.memory.contextual_store import ContextualHybridStore
from scripts.evaluation.evaluate_memory_evolution import (
    build_record,
    build_summary,
    file_hash,
    load_store,
    memory_file,
    paired_comparison,
    save_json,
    save_trajectory,
)

MAX_STEPS = 50
TOP_K = 3
POOL_SIZE = 8
WEIGHT = 0.45
PRIOR_STRENGTH = 1.0
STEP_MARGIN = 5
STEP_WEIGHT = 0.25
CONSTRUCTION_MANIFEST = Path("outputs/manifests/train_seed42_1000.json")
V1_ROOT = Path("outputs/memory_utility/feedback_v1")
OUTPUT_ROOT = Path("outputs/memory_utility/contextual_v2")
SOURCE_FILES = (
    Path(__file__),
    Path("src/memory/contextual_utility.py"),
    Path("src/memory/contextual_store.py"),
    Path("src/memory/store.py"),
    Path("src/agents/react_agent.py"),
    Path("src/core/protocol.py"),
    Path("src/core/trajectory.py"),
    Path("src/envs/environment.py"),
    Path("src/llm/client.py"),
    Path("scripts/evaluation/evaluate_memory_evolution.py"),
)


def read_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def game_dict(game: GameExample) -> dict:
    return {
        "id": game.id,
        "split": game.split,
        "game_file": str(game.game_file),
        "task_type": game.task_type,
    }


def restore_game(item: dict) -> GameExample:
    return GameExample(
        id=item["id"],
        split=item["split"],
        game_file=Path(item["game_file"]),
        task_type=item["task_type"],
    )


def build_credit(v1_summary: dict, memory_size: int):
    arms = v1_summary["arms"]
    hybrid = arms["hybrid"]["episodes"]
    utility = arms["utility"]["episodes"]
    if not v1_summary.get("finished") or len(hybrid) != len(utility):
        raise RuntimeError("v1 evaluation must be complete and paired")

    credit = ContextualCredit(memory_size, PRIOR_STRENGTH)
    events = []
    seen = set()

    for h, u in zip(hybrid, utility):
        game_id = h["game_id"]
        if game_id in seen or game_id != u["game_id"]:
            raise RuntimeError("Unmatched or duplicate v1 evaluation games")
        seen.add(game_id)
        if h["task_type"] != u["task_type"]:
            raise RuntimeError("Task type differs in a v1 pair")

        h_ids = list(h["retrieved_indices"])
        u_ids = list(u["retrieved_indices"])
        if (
            len(h_ids) != TOP_K or len(u_ids) != TOP_K
            or len(set(h_ids)) != TOP_K or len(set(u_ids)) != TOP_K
            or any(not isinstance(i, int) or not 0 <= i < memory_size
                   for i in h_ids + u_ids)
        ):
            raise RuntimeError(f"Invalid retrieval indices for {game_id}")

        added = sorted(set(u_ids) - set(h_ids))
        removed = sorted(set(h_ids) - set(u_ids))

        h_ok = bool(h["success"])
        u_ok = bool(u["success"])
        h_steps = int(h["decision_steps"])
        u_steps = int(u["decision_steps"])

        if h_ok != u_ok:
            signal = 1.0 if u_ok else -1.0
            basis = "success_difference"
        elif h_ok and abs(h_steps - u_steps) >= STEP_MARGIN:
            signal = STEP_WEIGHT * (h_steps - u_steps) / MAX_STEPS
            basis = "successful_step_difference"
        else:
            signal = 0.0
            basis = "no_informative_difference"

        # Reordering the same lessons cannot identify individual lesson effects.
        if not added and not removed:
            signal = 0.0
            basis = "same_lesson_set"

        divisor = max(len(added), len(removed), 1)
        assigned = []
        if signal:
            for i in added:
                value = signal / divisor
                credit.add(h["task_type"], i, value)
                assigned.append({
                    "index": i,
                    "credit": value,
                    "arm": "utility",
                })

            for i in removed:
                value = -signal / divisor
                credit.add(h["task_type"], i, value)
                assigned.append({
                    "index": i,
                    "credit": value,
                    "arm": "hybrid",
                })

        events.append({
            "game_id": game_id,
            "task_type": h["task_type"],
            "hybrid_ids": h_ids,
            "utility_ids": u_ids,
            "hybrid_success": h_ok,
            "utility_success": u_ok,
            "hybrid_steps": h_steps,
            "utility_steps": u_steps,
            "added": added,
            "removed": removed,
            "signal": signal,
            "basis": basis,
            "assigned": assigned,
        })

    return credit, events


def excluded_game_ids(v1_plan: dict) -> set[str]:
    used_games = load_manifest(CONSTRUCTION_MANIFEST)
    if (
        len(used_games) != 1000
        or len({game.id for game in used_games}) != 1000
        or any(game.split != "train" for game in used_games)
    ):
        raise RuntimeError(
            "Expected 1,000 distinct memory-construction Train games"
        )

    excluded = {game.id for game in used_games}

    for label in ("learning_games", "evaluation_games"):
        excluded.update(row["id"] for row in v1_plan[label])

    prior = Path("outputs/memory_utility/holdout_train_v1")
    for path in prior.rglob("experiment_config.json") if prior.is_dir() else []:
        data = read_json(path)
        excluded.update(row["id"] for row in data.get("games", []))

    return excluded


def create_plan(directory: Path, args, memory_path: Path, memory_size: int):
    model_dir = args.model.replace("/", "_")
    v1_dir = V1_ROOT / model_dir / f"seed_{args.v1_seed}"

    v1_plan_path = v1_dir / "experiment_plan.json"
    v1_eval_path = v1_dir / "evaluation_summary.json"

    v1_plan = read_json(v1_plan_path)
    v1_eval = read_json(v1_eval_path)

    if (
        v1_plan["model"] != args.model
        or v1_plan["memory_hash"] != file_hash(memory_path)
    ):
        raise RuntimeError(
            "v1 data and current memory catalog do not match"
        )

    files = {
        "memory": file_hash(memory_path),
        "construction": file_hash(CONSTRUCTION_MANIFEST),
        "v1_plan": file_hash(v1_plan_path),
        "v1_evaluation": file_hash(v1_eval_path),
        "code": {
            str(p): file_hash(p)
            for p in SOURCE_FILES
        },
    }

    path = directory / "experiment_plan.json"

    if path.is_file():
        plan = read_json(path)

        if plan["fingerprints"] != files or plan["seed"] != args.seed:
            raise RuntimeError(
                "Frozen inputs changed; choose a new --seed"
            )

        return plan

    if directory.exists() and list(directory.rglob("episode_*.json")):
        raise RuntimeError(
            "Trajectories exist without a frozen plan"
        )

    credit, events = build_credit(v1_eval, memory_size)
    excluded = excluded_game_ids(v1_plan)
    candidates = discover_games(args.data_root, "train")

    rng = random.Random(args.seed)
    selected = []

    for task_type in sorted(SUPPORTED_TASK_TYPES):
        eligible = sorted(
            (
                game
                for game in candidates
                if game.task_type == task_type
                and game.id not in excluded
            ),
            key=lambda g: g.id,
        )

        rng.shuffle(eligible)

        if len(eligible) < 2:
            raise RuntimeError(
                f"Not enough unused Train games for {task_type}"
            )

        selected.extend(eligible[:2])

    rng.shuffle(selected)

    plan = {
        "schema": "contextual_utility_v2",
        "model": args.model,
        "seed": args.seed,
        "v1_seed": args.v1_seed,
        "max_steps": MAX_STEPS,
        "top_k": TOP_K,
        "pool_size": POOL_SIZE,
        "weight": WEIGHT,
        "prior_strength": PRIOR_STRENGTH,
        "step_margin": STEP_MARGIN,
        "step_weight": STEP_WEIGHT,
        "state_tracking": False,
        "fingerprints": files,
        "credit": credit.to_dict(),
        "credit_events": events,
        "games": [game_dict(g) for g in selected],
    }

    save_json(path, plan)
    return plan


def make_store(experiences, task_type: str, credit, plan):
    return ContextualHybridStore(
        experiences,
        task_type=task_type,
        credit=credit,
        weight=plan["weight"],
        pool_size=plan["pool_size"],
    )


def inspect(directory, plan, experiences):
    path = directory / "retrieval_inspection.json"

    if path.is_file():
        rows = read_json(path)["games"]

        if [r["id"] for r in rows] != [
            g["id"] for g in plan["games"]
        ]:
            raise RuntimeError(
                "Inspection game IDs do not match plan"
            )

        return rows

    credit = ContextualCredit.from_dict(plan["credit"])
    rows = []

    for index, item in enumerate(plan["games"]):
        game = restore_game(item)

        env = ALFWorldEnvironment(
            str(game.game_file),
            max_steps=MAX_STEPS,
        )

        try:
            state = env.reset()
        finally:
            env.close()

        hybrid = make_store(
            experiences,
            game.task_type,
            None,
            plan,
        )

        contextual = make_store(
            experiences,
            game.task_type,
            credit,
            plan,
        )

        a = hybrid.rank_indices(state.task, TOP_K)
        b = contextual.rank_indices(state.task, TOP_K)

        rows.append({
            "id": game.id,
            "task": state.task,
            "task_type": game.task_type,
            "hybrid": a,
            "contextual": b,
        })

        print(
            f"[INSPECT] {index + 1}/12 "
            f"{game.task_type} {a} -> {b}",
            flush=True,
        )

    save_json(path, {"games": rows})
    return rows


def run_episode(game, store, llm, path):
    if path.is_file():
        return Trajectory.load_json(str(path)), False

    env = ALFWorldEnvironment(
        str(game.game_file),
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
        trace = agent.run(env)
    finally:
        env.close()

    if store.last_query != trace.task:
        raise RuntimeError(
            "Agent did not retrieve memory for this task"
        )

    save_trajectory(path, trace)
    return trace, True


def evaluate(directory, plan, experiences, max_new: int):
    inspected = inspect(
        directory,
        plan,
        experiences,
    )

    changed = sum(
        x["hybrid"] != x["contextual"]
        for x in inspected
    )

    print(
        f"Different top-3 rankings: "
        f"{changed}/{len(inspected)}"
    )

    if changed == 0:
        print(
            "No retrieval changes. "
            "Skipping costly LLM evaluation."
        )
        return

    if max_new == 0:
        print(
            "No LLM calls (default --max-new 0)."
        )
        return

    targets = [
        (i, item, inspected[i])
        for i, item in enumerate(plan["games"])
        if inspected[i]["hybrid"] != inspected[i]["contextual"]
    ]

    print(
        f"Changed-only exploratory evaluation: {len(targets)} of "
        f"{len(plan['games'])} frozen games."
    )

    credit = ContextualCredit.from_dict(
        plan["credit"]
    )

    arms = {
        "hybrid": None,
        "contextual": credit,
    }

    results = {
        name: []
        for name in arms
    }

    new_pairs = 0
    llm = None

    for i, item, check in targets:
        game = restore_game(item)

        paths = {
            name: (
                directory
                / "eval"
                / name
                / f"episode_{i:03d}.json"
            )
            for name in arms
        }

        incomplete = any(
            not p.is_file()
            for p in paths.values()
        )

        if incomplete and new_pairs >= max_new:
            break

        new_pairs += int(incomplete)

        for name, arm_credit in arms.items():
            store = make_store(
                experiences,
                game.task_type,
                arm_credit,
                plan,
            )

            expected = check[name]

            if store.rank_indices(
                check["task"],
                TOP_K,
            ) != expected:
                raise RuntimeError(
                    "Frozen retrieval no longer matches inspection"
                )

            if not paths[name].is_file() and llm is None:
                llm = LLMClient()

            trace, created = run_episode(
                game,
                store,
                llm,
                paths[name],
            )

            if trace.task != check["task"]:
                raise RuntimeError(
                    "Trajectory task differs from frozen inspection"
                )

            if created and store.last_indices != expected:
                raise RuntimeError(
                    "Actual agent memory does not match inspection"
                )

            record = build_record(
                i,
                game,
                trace,
            )

            record["retrieved_indices"] = expected
            results[name].append(record)

            print(
                f"[EVAL] game_index={i} "
                f"{name} "
                f"success={record['success']} "
                f"steps={record['decision_steps']} "
                f"ids={expected} "
                f"{'NEW' if created else 'CACHE'}",
                flush=True,
            )

        summaries = {
            name: build_summary(
                rows,
                len(targets),
            )
            for name, rows in results.items()
        }

        comparison = paired_comparison(
            "hybrid",
            summaries["hybrid"],
            "contextual",
            summaries["contextual"],
        )

        save_json(
            directory / "evaluation_summary.json",
            {
                "scope": "changed_retrieval_only",
                "finished": all(
                    x["finished"]
                    for x in summaries.values()
                ),
                "frozen_game_count": len(plan["games"]),
                "target_pairs": len(targets),
                "completed_pairs": len(results["hybrid"]),
                "retrieval_changed_pairs": len(results["hybrid"]),
                "arms": summaries,
                "comparison": comparison,
            },
        )

    print(
        f"New evaluation pairs in this invocation: {new_pairs}"
    )

    show_report(directory)


def show_report(directory):
    path = directory / "evaluation_summary.json"

    if not path.is_file():
        print(
            "No paired evaluation results yet"
        )
        return

    summary = read_json(path)

    print(json.dumps({
        "finished": summary["finished"],
        "scope": summary["scope"],
        "target_pairs": summary["target_pairs"],
        "completed_pairs": summary["completed_pairs"],
        "retrieval_changed_pairs": summary["retrieval_changed_pairs"],
        "hybrid_successes": summary["arms"]["hybrid"]["successes"],
        "contextual_successes": summary["arms"]["contextual"]["successes"],
        "comparison": summary["comparison"],
    }, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "--stage",
        choices=(
            "prepare",
            "inspect",
            "eval",
            "report",
        ),
        default="prepare",
    )

    parser.add_argument(
        "--model",
        default=os.getenv(
            "OPENAI_MODEL",
            "qwen3.8-chat",
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=20261112,
    )

    parser.add_argument(
        "--v1-seed",
        type=int,
        default=20261110,
    )

    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(
            os.getenv("ALFWORLD_DATA")
            or default_data_root()
        ),
    )

    parser.add_argument(
        "--max-new",
        type=int,
        default=0,
    )

    args = parser.parse_args()

    if args.max_new < 0:
        parser.error(
            "--max-new must be >= 0"
        )

    if (
        args.max_new
        and args.stage == "eval"
        and os.getenv("OPENAI_MODEL") != args.model
    ):
        raise RuntimeError(
            "OPENAI_MODEL must match --model "
            "before LLM execution"
        )

    memory_path = memory_file(
        "direct_update",
        args.model,
    )

    experiences = load_store(
        "direct_update",
        memory_path,
    ).experiences

    if len(experiences) != 18:
        raise RuntimeError(
            "Expected Direct Update 18 memories"
        )

    directory = (
        OUTPUT_ROOT
        / args.model.replace("/", "_")
        / f"seed_{args.seed}"
    )

    plan = create_plan(
        directory,
        args,
        memory_path,
        len(experiences),
    )

    credit = ContextualCredit.from_dict(
        plan["credit"]
    )

    events = plan["credit_events"]

    print(
        f"Plan: {directory / 'experiment_plan.json'}"
    )

    print(
        f"Old paired data reused: {len(events)} pairs"
    )

    print(
        "Pairs with nonzero weak credit: "
        f"{sum(bool(e['assigned']) for e in events)}"
    )

    for task_type, states in sorted(
        credit.by_type.items()
    ):
        for index, item in sorted(
            states.items()
        ):
            print(
                f"  {task_type} "
                f"lesson {index:02d} "
                f"credit={credit.bonus(task_type, index):+.3f} "
                f"n={item.count}"
            )

    if args.stage == "prepare":
        print(
            "Prepared with existing data. No LLM calls."
        )

    elif args.stage == "inspect":
        inspected = inspect(
            directory,
            plan,
            experiences,
        )

        n = sum(
            x["hybrid"] != x["contextual"]
            for x in inspected
        )

        print(
            f"Different top-3 rankings: "
            f"{n}/{len(inspected)}; no LLM calls"
        )

    elif args.stage == "eval":
        evaluate(
            directory,
            plan,
            experiences,
            args.max_new,
        )

    else:
        show_report(directory)


if __name__ == "__main__":
    main()

