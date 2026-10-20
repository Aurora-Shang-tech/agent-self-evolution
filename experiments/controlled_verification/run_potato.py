import hashlib
import json
import re
from pathlib import Path

from src.core.trajectory import Trajectory
from src.counterfactual.replay import replay_to_step


BASELINE_PATH = Path(
    "outputs/baseline_qwen3.8-chat_train_seed42_1000/episode_921.json"
)
VERIFICATION_PATH = Path(
    "outputs/counterfactual_qwen3.8-chat_train_seed42_1000/episode_921.json"
)
OUTPUT_PATH = Path(
    "outputs/analysis/controlled_verification/episode_921_potato_2x2_v1.json"
)

EPISODE = "episode_921"
CRITICAL_STEP = 19
MAX_STEPS = 50

POTATO_IDS = (2, 3)
FRIDGE = "fridge 1"
MICROWAVE = "microwave 1"


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def state_hash(observation, actions):
    data = json.dumps(
        {
            "observation": observation,
            "admissible_actions": list(actions),
        },
        ensure_ascii=False,
        sort_keys=True,
    )

    return sha256_bytes(data.encode("utf-8"))


def require_action(actions, wanted):
    if wanted not in actions:
        raise RuntimeError(
            f"Action unavailable: {wanted!r}\n"
            f"Current admissible actions: "
            f"{json.dumps(list(actions), ensure_ascii=False)}"
        )

    return wanted


def choose_take(actions, potato_id):
    pattern = re.compile(
        rf"take potato {potato_id} from .+"
    )

    choices = [
        action
        for action in actions
        if pattern.fullmatch(action)
    ]

    if len(choices) != 1:
        raise RuntimeError(
            f"Expected exactly one take action for "
            f"potato {potato_id}, found {choices}.\n"
            f"Available: "
            f"{json.dumps(list(actions), ensure_ascii=False)}"
        )

    return choices[0]


def run_trial(trajectory, potato_id, do_cool):
    branch = (
        f"potato_{potato_id}_"
        f"{'cool' if do_cool else 'no_cool'}"
    )

    record = {
        "branch": branch,
        "potato_id": potato_id,
        "cool": do_cool,
        "status": "error",
        "actions": [],
    }

    env = None

    try:
        env, restored = replay_to_step(
            trajectory,
            CRITICAL_STEP,
            max_steps=MAX_STEPS,
        )

        observation = restored.observation
        available = restored.admissible_actions

        record["replay_state_sha256"] = state_hash(
            observation,
            available,
        )

        def execute(action, allow_terminal=False):
            nonlocal observation, available

            require_action(available, action)

            result = env.step(action)

            item = {
                "evidence_id": (
                    f"{EPISODE}/controlled_v1/"
                    f"{branch}/"
                    f"action_{len(record['actions']) + 1:03d}"
                ),
                "action": action,
                "observation_before": observation,
                "admissible_actions_before": list(available),
                "observation_after": result.observation,
                "admissible": result.admissible,
                "done": result.done,
                "won": result.won,
                "score": result.score,
            }

            record["actions"].append(item)

            observation = result.observation
            available = result.admissible_actions

            if not result.admissible:
                raise RuntimeError(
                    f"Environment rejected admissible action: "
                    f"{action!r}"
                )

            if result.done and not allow_terminal:
                raise RuntimeError(
                    f"Episode terminated before controlled "
                    f"script ended: {action!r}, "
                    f"won={result.won}"
                )

        execute(
            choose_take(available, potato_id)
        )

        execute(
            require_action(
                available,
                f"go to {FRIDGE}",
            )
        )

        cooling_action = (
            f"cool potato {potato_id} with {FRIDGE}"
        )

        if cooling_action not in available:
            execute(
                require_action(
                    available,
                    f"open {FRIDGE}",
                )
            )

        require_action(
            available,
            cooling_action,
        )

        record["pre_intervention_sha256"] = state_hash(
            observation,
            available,
        )

        record["cool_action_admissible_at_intervention"] = True

        if do_cool:
            execute(cooling_action)

        execute(
            require_action(
                available,
                f"go to {MICROWAVE}",
            )
        )

        placing_action = (
            f"move potato {potato_id} to {MICROWAVE}"
        )

        if placing_action not in available:
            execute(
                require_action(
                    available,
                    f"open {MICROWAVE}",
                )
            )

        require_action(
            available,
            placing_action,
        )

        before_placement = state_hash(
            observation,
            available,
        )

        execute(
            placing_action,
            allow_terminal=True,
        )

        record["pre_placement_sha256"] = before_placement
        record["status"] = "completed"
        record["goal_reached"] = record["actions"][-1]["won"]
        record["episode_done"] = record["actions"][-1]["done"]
        record["end_observation"] = observation

    except Exception as exc:
        record["error"] = str(exc)

    finally:
        if env is not None:
            env.close()

    return record


def comparable_actions(trial):
    return [
        item["action"]
        for item in trial["actions"]
        if not item["action"].startswith("cool potato ")
    ]


def main():
    if not BASELINE_PATH.is_file():
        raise FileNotFoundError(BASELINE_PATH)

    if not VERIFICATION_PATH.is_file():
        raise FileNotFoundError(VERIFICATION_PATH)

    trajectory = Trajectory.load_json(
        str(BASELINE_PATH)
    )

    if not 0 <= CRITICAL_STEP < len(trajectory.steps):
        raise ValueError(
            "Critical step is out of range"
        )

    verification = json.loads(
        VERIFICATION_PATH.read_text(
            encoding="utf-8"
        )
    )

    if verification["critical_step"] != CRITICAL_STEP:
        raise ValueError(
            f"Unexpected critical step: "
            f"{verification['critical_step']}"
        )

    task = trajectory.task.lower()

    if "cool potato" not in task or "microwave" not in task:
        raise ValueError(
            f"Unexpected task for episode_921: "
            f"{trajectory.task}"
        )

    trials = [
        run_trial(
            trajectory,
            potato_id,
            do_cool,
        )
        for potato_id in POTATO_IDS
        for do_cool in (False, True)
    ]

    matrix = {}

    for potato_id in POTATO_IDS:
        no_cool = next(
            trial
            for trial in trials
            if (
                trial["potato_id"] == potato_id
                and not trial["cool"]
            )
        )

        cooled = next(
            trial
            for trial in trials
            if (
                trial["potato_id"] == potato_id
                and trial["cool"]
            )
        )

        route_equal = (
            no_cool["status"] == "completed"
            and cooled["status"] == "completed"
            and comparable_actions(no_cool)
            == comparable_actions(cooled)
        )

        pre_state_equal = (
            no_cool.get("pre_intervention_sha256") is not None
            and no_cool.get("pre_intervention_sha256")
            == cooled.get("pre_intervention_sha256")
        )

        shared_replay_state = (
            no_cool.get("replay_state_sha256") is not None
            and no_cool.get("replay_state_sha256")
            == cooled.get("replay_state_sha256")
        )

        cooling_effect_supported = (
            shared_replay_state
            and route_equal
            and pre_state_equal
            and no_cool.get("goal_reached") is False
            and cooled.get("goal_reached") is True
        )

        matrix[f"potato_{potato_id}"] = {
            "no_cool_goal_reached": no_cool.get(
                "goal_reached"
            ),
            "cool_goal_reached": cooled.get(
                "goal_reached"
            ),
            "shared_replay_state": shared_replay_state,
            "shared_pre_intervention_state": pre_state_equal,
            "same_actions_except_cooling": route_equal,
            "cooling_effect_supported_in_this_state": (
                cooling_effect_supported
            ),
        }

    result = {
        "experiment": "controlled_potato_cooling_2x2_v1",
        "source_episode": EPISODE,
        "experience_id": "08c7e7c441bd",
        "target_claim_ids": [
            "C1",
            "C2",
        ],
        "task": trajectory.task,
        "critical_step_zero_based": CRITICAL_STEP,
        "max_steps": MAX_STEPS,
        "baseline_path": str(BASELINE_PATH),
        "baseline_sha256": sha256_bytes(
            BASELINE_PATH.read_bytes()
        ),
        "verification_path": str(VERIFICATION_PATH),
        "verification_sha256": sha256_bytes(
            VERIFICATION_PATH.read_bytes()
        ),
        "intervention": "cool potato with fridge",
        "trials": trials,
        "paired_comparisons": matrix,
        "all_trials_completed": all(
            trial["status"] == "completed"
            for trial in trials
        ),
    }

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_PATH.write_text(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("=" * 60)
    print("Controlled Counterfactual Verification")
    print("=" * 60)

    print("Task:", result["task"])
    print("Critical step:", CRITICAL_STEP)

    print("\nTrials:")

    for trial in trials:
        print(
            trial["branch"],
            "status=", trial["status"],
            "goal_reached=", trial.get("goal_reached"),
            "actions=", len(trial["actions"]),
        )

        if "error" in trial:
            print(
                "  error:",
                trial["error"],
            )

    print("\nPaired comparisons:")

    print(
        json.dumps(
            matrix,
            ensure_ascii=False,
            indent=2,
        )
    )

    print("\nAll trials completed:", result["all_trials_completed"])
    print("Saved:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
