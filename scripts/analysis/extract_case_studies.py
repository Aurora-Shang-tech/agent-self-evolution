"""提取 Unverified 和 Verified 结果不一致的ALFWorld任务"""

import json
from pathlib import Path

MODEL = "qwen3.8-chat"
SEED = 42
SPLITS = {
    "valid_seen": 140,
    "valid_unseen": 134,
}

OUTPUT_PATH = Path("outputs/analysis/unverified_vs_verified_cases.json")

def load_episodes(directory: Path) -> dict[str, dict]:
    """读取目录下的所有episode"""

    episodes = {}
    for path in sorted(directory.glob("episode_*.json")):
        with path.open(encoding="utf-8") as f:
            episodes[path.name] = json.load(f)

    return episodes

def main():
    cases = []
    for split, num_games in SPLITS.items():
        unverified_dir = Path(
            f"outputs/unverified_memory_{MODEL}_{split}_seed{SEED}_{num_games}"
        )
        verified_dir = Path(
            f"outputs/memory_{MODEL}_{split}_seed{SEED}_{num_games}"
        )

        unverified = load_episodes(unverified_dir)
        verified = load_episodes(verified_dir)

        # 必须是相同的 episode，才能进行配对比较
        if len(unverified) != num_games or len(verified) != num_games:
            raise ValueError(f"{split}: episode count mismatch")

        if unverified.keys() != verified.keys():
            raise ValueError(f"{split}: episode sets do not match")

        for episode_name in sorted(unverified):
            a = unverified[episode_name]
            b = verified[episode_name]
            
            # 两种方法成功状态一致的任务暂不分析
            if a["success"] == b["success"]:
                continue

            # 校验是否对应同一个 ALFWorld 任务
            if a["gamefile"] != b["gamefile"]:
                raise ValueError(
                    f"{split}/{episode_name}: gamefile mismatch"
                )

            transition = (
                "improved"
                if not a["success"] and b["success"]
                else "regressed"
            )

            cases.append({
                "split": split,
                "episode": episode_name,
                "task": a["task"],
                "gamefile": a["gamefile"],
                "transition": transition,
                "unverified_success": a["success"],
                "verified_success": b["success"],
                "unverified_steps": len(a["steps"]),
                "verified_steps": len(b["steps"]),
            })

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(cases, f, ensure_ascii=False, indent=2)

    improved = sum(c["transition"] == "improved" for c in cases)
    regressed = sum(c["transition"] == "regressed" for c in cases)

    print(f"Total cases: {len(cases)}")
    print(f"Improved: {improved}")
    print(f"Regressed: {regressed}")
    print(f"Saved: {OUTPUT_PATH}")

    for case in cases:
        print(
            f"[{case['split']}] "
            f"{case['episode']} "
            f"{case['transition']}: "
            f"{case['task']}"
        )


if __name__ == "__main__":
    main()

    
