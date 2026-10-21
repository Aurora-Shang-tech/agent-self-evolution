"""分析Unverified与Verified的不一致Episode中的失败模式"""

import json
from collections import Counter
from pathlib import Path

from src.core.trajectory import Trajectory

MODEL = "qwen3.8-chat"
SEED = 42

CASES_PATH = Path(
    "outputs/analysis/unverified_vs_verified_cases.json"
)

OUTPUT_PATH = Path(
    "outputs/analysis/case_study_patterns.json"
)

SPLIT_SIZES = {
    "valid_seen": 140,
    "valid_unseen": 134,
}

# ALFWorld 状态转换任务的动作关键词
STATE_ACTIONS = {
    "clean": "clean ",
    "cool": "cool ",
    "heat": "heat ",
}

OBSERVATION_ACTIONS = (
    "look",
    "examine ",
    "inventory",
)


def get_required_state(task: str) -> str | None:
    """从任务描述识别所需状态"""

    task = task.lower()

    for state in STATE_ACTIONS:
        if f" a {state} " in task or f" some {state} " in task:
            return state

    return None

def longest_run(actions: list[str]) -> tuple[str, int]:
    """统计连续执行同一动作的最长次数"""

    if not actions:
        return "", 0
    best_action = actions[0]
    best_count = 1

    current_action = actions[0]
    current_count = 1

    for action in actions[1:]:
        if action == current_action:
            current_count += 1
        else:
            current_action = action
            current_count = 1

        if current_count > best_count:
            best_action = current_action
            best_count = current_count

    return best_action, best_count

def longest_observation_run(actions: list[str]) -> int:
    """统计连续观察类动作的最长长度。"""

    best = 0
    current = 0

    for action in actions:
        if (
            action == "look"
            or action == "inventory"
            or action.startswith("examine ")
        ):
            current += 1
            best = max(best, current)
        else:
            current = 0

    return best

def analyze_trajectory(trajectory: Trajectory) -> dict:
    """分析单条轨迹的状态操作和循环行为。"""

    actions = [
        step.action.strip().lower()
        for step in trajectory.steps
    ]

    required_state = get_required_state(trajectory.task)

    # 是否执行了任务要求的状态转换
    state_indices = []

    if required_state:
        prefix = STATE_ACTIONS[required_state]

        state_indices = [
            i for i, action in enumerate(actions)
            if action.startswith(prefix)
        ]

    # ALFWorld 放置动作通常为 move X to Y
    placement_indices = [
        i for i, action in enumerate(actions)
        if action.startswith("move ")
    ]

    repeated_action, repeated_count = longest_run(actions)

    observation_run = longest_observation_run(actions)

    missing_state = (
        required_state is not None
        and not state_indices
    )

    premature_placement = (
        required_state is not None
        and bool(placement_indices)
        and (
            not state_indices
            or placement_indices[0] < state_indices[0]
        )
    )

    return {
        "success": trajectory.success,
        "steps": len(actions),
        "required_state": required_state,
        "state_operation_count": len(state_indices),
        "missing_state_transition": missing_state,
        "premature_placement": premature_placement,
        "longest_repeated_action": repeated_action,
        "longest_repeated_action_count": repeated_count,
        "repeated_action_loop": repeated_count >= 5,
        "longest_observation_run": observation_run,
        "repeated_observation": observation_run >= 5,
    }

def load_trajectory(
    method: str,
    split: str,
    episode: str,
) -> Trajectory:
    """根据方法、数据集和 Episode 读取轨迹。"""

    size = SPLIT_SIZES[split]

    prefix = {
        "unverified": "unverified_memory",
        "verified": "memory",
    }[method]

    path = Path(
        f"outputs/{prefix}_{MODEL}_{split}_seed{SEED}_{size}"
    ) / episode

    if not path.exists():
        raise FileNotFoundError(path)

    return Trajectory.load_json(str(path))

def main():
    with CASES_PATH.open(encoding="utf-8") as f:
        cases = json.load(f)

    results = []

    for case in cases:
        split = case["split"]
        episode = case["episode"]

        unverified = load_trajectory(
            "unverified", split, episode
        )
        verified = load_trajectory(
            "verified", split, episode
        )

        results.append({
            "split": split,
            "episode": episode,
            "task": case["task"],
            "transition": case["transition"],
            "unverified": analyze_trajectory(unverified),
            "verified": analyze_trajectory(verified),
        })

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Analyzed cases: {len(results)}")

    for transition in ("improved", "regressed"):
        subset = [
            item for item in results
            if item["transition"] == transition
        ]

        print(f"\n{'=' * 65}")
        print(f"{transition.upper()} ({len(subset)})")
        print("=" * 65)

        for method in ("unverified", "verified"):
            counts = Counter()

            for item in subset:
                data = item[method]

                for key in (
                    "missing_state_transition",
                    "premature_placement",
                    "repeated_action_loop",
                    "repeated_observation",
                ):
                    if data[key]:
                        counts[key] += 1

            print(f"\n{method}:")
            for key, value in counts.items():
                print(f"  {key}: {value}/{len(subset)}")

        for item in subset:
            a = item["unverified"]
            b = item["verified"]

            print(
                f"\n{item['split']}/{item['episode']}"
                f" | {item['task']}"
            )
            print(
                f"  Unverified: {a['steps']} steps, "
                f"missing_state={a['missing_state_transition']}, "
                f"observation_run={a['longest_observation_run']}"
            )
            print(
                f"  Verified:   {b['steps']} steps, "
                f"missing_state={b['missing_state_transition']}, "
                f"observation_run={b['longest_observation_run']}"
            )

    print(f"\nSaved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
