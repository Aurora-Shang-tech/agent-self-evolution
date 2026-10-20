from collections import Counter
from pathlib import Path

from src.core.trajectory import Trajectory
from src.memory.state_tracker import TaskStateTracker

ROOT = Path("outputs")

EXPERIMENTS = {
    "Unverified Seen": "unverified_memory_qwen3.8-chat_valid_seen_seed42_140",
    "Verified Seen": "memory_qwen3.8-chat_valid_seen_seed42_140",
    "Unverified Unseen": "unverified_memory_qwen3.8-chat_valid_unseen_seed42_134",
    "Verified Unseen": "memory_qwen3.8-chat_valid_unseen_seed42_134",
}

def audit(directory: Path):
    stats = Counter()
    suspicious = []

    for path in sorted(directory.glob("episode_*.json")):
        trajectory = Trajectory.load_json(path)
        tracker = TaskStateTracker(trajectory.task)

        if tracker.operation is None:
            stats["untracked_tasks"] += 1
            continue

        stats["tracked_tasks"] += 1
        stats[f"task_{tracker.operation}"] += 1

        for step in trajectory.steps:
            tracker.observe(step)

        if tracker.action_observed:
            stats["action_observed"] += 1
        if tracker.confirmed:
            stats["confirmed"] += 1
        if trajectory.success:
            stats["successful_tasks"] += 1

        if trajectory.success and not tracker.confirmed:
            stats["success_without_confirmation"] += 1
            suspicious.append({
                "episode": path.name,
                "task": trajectory.task,
                "operation": tracker.operation,
                "success": trajectory.success,
                "action_observed": tracker.action_observed,
                "confirmed": tracker.confirmed,
            })

        if tracker.action_observed and not tracker.confirmed:
            stats["action_without_confirmation"] += 1
            suspicious.append({
                "episode": path.name,
                "task": trajectory.task,
                "operation": tracker.operation,
                "success": trajectory.success,
                "action_observed": True,
                "confirmed": False,
            })

    return stats, suspicious

def main():
    for name, dirname in EXPERIMENTS.items():
        directory = ROOT / dirname
        
        print(f"\n{'=' * 65}")
        print(name)
        print(f"{'=' * 65}")

        if not directory.exists():
            print("Directory not found:", directory)
            continue

        stats, suspicious = audit(directory)

        for key, value in sorted(stats.items()):
            print(f"{key}: {value}")

        print("\nSuspicious cases:")

        for case in suspicious[:15]:
            print(case)

        if not suspicious:
            print("None")


if __name__ == "__main__":
    main()





