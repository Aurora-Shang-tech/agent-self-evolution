import json
import os
from pathlib import Path
METHODS = {
    "ReAct": "baseline",
    "Ordinary": "ordinary_memory",
    "Unverified": "unverified_memory",
    "Verified": "memory",
}

SPLITS = {
    "valid_seen": 140,
    "valid_unseen": 134,
}

def load_summary(path):
    if not path.exists():
        raise FileNotFoundError(path)

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def main():
    model = os.environ["OPENAI_MODEL"].replace("/", "_")

    output_dir = Path("outputs/analysis")
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}

    for split, size in SPLITS.items():
        print(f"\n{'=' * 80}")
        print(f"Split: {split}")
        print(f"{'=' * 80}")

        summaries = {}

        for method, prefix in METHODS.items():
            path = Path(
                f"outputs/{prefix}_{model}_{split}_seed42_{size}/summary.json"
            )
            summaries[method] = load_summary(path)

        task_types = sorted(
            set().union(
                *[
                    summary["task_type_stats"].keys()
                    for summary in summaries.values()
                ]
            )
        )

        split_results = {}

        for task_type in task_types:
            print(f"\nTask type: {task_type}")
            print(
                f"{'Method':<15}"
                f"{'Success':<15}"
                f"{'Rate':<12}"
            )

            task_results = {}

            for method, summary in summaries.items():
                stats = summary["task_type_stats"].get(task_type)

                if stats is None:
                    continue

                successes = stats["successes"]
                episodes = stats["episodes"]
                rate = successes / episodes if episodes else 0.0

                task_results[method] = {
                    "successes": successes,
                    "episodes": episodes,
                    "success_rate": rate,
                }

                print(
                    f"{method:<15}"
                    f"{successes}/{episodes:<12}"
                    f"{rate:<12.2%}"
                )

            split_results[task_type] = task_results

        all_results[split] = split_results

    output_path = output_dir / "task_type_comparison.json"
    output_path.write_text(
        json.dumps(all_results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
