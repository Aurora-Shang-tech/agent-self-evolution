"""比较不同方法在相同 ALFWorld episode 上成功/失败变化"""

import json
from pathlib import Path
from math import comb

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

MODEL = "qwen3.8-chat"
SEED = 42

def load_results(directory: Path) -> dict[str, bool]:
    """读取每一个 episode 的成功状态"""

    results = {}
    for path in sorted(directory.glob("episode_*.json")):
        with path.open(encoding="utf-8") as f:
            data = json.load(f)

        results[path.name] = data["success"]

    return results

def compare(
    before: dict[str, bool],
    after: dict[str, bool],
) -> dict[str, int]:
    """统计两个方法在相同 episode 上的成功状态变化"""
    
    if before.keys() != after.keys():
        raise ValueError("Episode sets do not match")

    counts = {
        "fail -> fail": 0,
        "fail -> success": 0,
        "success -> fail": 0,
        "success -> success": 0,
    }

    for episode in before:
        a = before[episode]
        b = after[episode]
        
        if not a and not b:
            counts["fail -> fail"] += 1
        elif not a and b:
            counts["fail -> success"] += 1
        elif a and not b:
            counts["success -> fail"] += 1
        else:
            counts["success -> success"] += 1

    return counts

def exact_mcnemar(b: int, c: int) -> float:
    """计算双侧 Exact McNemar 检验的p值"""

    n = b + c

    if n == 0:
        return 1.0

    k = min(b, c)

    p_value = 2 * sum(comb(n, i) for i in range(k + 1)) / (2 ** n)

    return min(1.0, p_value)
                

def main():
    comparisons = [
        ("ReAct", "Ordinary"),
        ("Ordinary", "Unverified"),
        ("Unverified", "Verified"),
        ("ReAct", "Verified"),
    ]

    for split, num_games in SPLITS.items():
        print()
        print("=" * 70)
        print(split)
        print("=" * 70)

        results = {}

        for method_name, prefix in METHODS.items():
            directory = Path(
                f"outputs/{prefix}_{MODEL}_{split}_seed{SEED}_{num_games}"
            )

            method_results = load_results(directory)

            if len(method_results) != num_games:
                raise ValueError(
                    f"{method_name} {split}: "
                    f"expected {num_games} episodes, "
                    f"found {len(method_results)}"
                )

            results[method_name] = method_results

            successes = sum(method_results.values())

            print(
                f"{method_name:<12}: "
                f"{successes}/{num_games} "
                f"({successes / num_games:.2%})"
            )

        for before_name, after_name in comparisons:
            counts = compare(
                results[before_name],
                results[after_name],
            )

            print()
            print(f"{before_name} -> {after_name}")

            for key, value in counts.items():
                print(f"  {key:<20}: {value}")

            improved = counts["fail -> success"]
            regressed = counts["success -> fail"]

            p_value = exact_mcnemar(regressed, improved)

            print(f"  net improvement     : {improved - regressed}")
            print(f"  McNemar exact p     : {p_value:.6f}")


if __name__ == "__main__":
    main()






                       
