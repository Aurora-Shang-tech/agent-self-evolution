"""Audit whether a frozen Hybrid/Utility Top-K change is a reorder or a replacement.

Reads existing paired evaluation plans and metadata only. Never calls an LLM,
modifies pipeline state, or re-runs ALFWorld episodes.
"""

import argparse
import json
from pathlib import Path
from statistics import mean


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def change_kind(hybrid_ids: list[str], utility_ids: list[str]) -> str:
    if not hybrid_ids or not utility_ids:
        raise ValueError("Empty retrieved memory IDs")
    if len(hybrid_ids) != len(set(hybrid_ids)):
        raise ValueError("Duplicate Hybrid memory IDs")
    if len(utility_ids) != len(set(utility_ids)):
        raise ValueError("Duplicate Utility memory IDs")
    if hybrid_ids == utility_ids:
        return "identical"
    if set(hybrid_ids) == set(utility_ids):
        return "reordered_only"
    return "membership_changed"


def summary(rows: list[dict]) -> dict:
    if not rows:
        return {
            "completed_pairs": 0,
            "hybrid_successes": 0,
            "utility_successes": 0,
            "improved": 0,
            "regressed": 0,
            "mean_hybrid_steps": None,
            "mean_utility_steps": None,
            "mean_step_delta": None,
            "both_success_pairs": 0,
            "mean_step_delta_both_success": None,
        }

    both_succeeded = [
        row for row in rows
        if row["hybrid_success"] and row["utility_success"]
    ]
    return {
        "completed_pairs": len(rows),
        "hybrid_successes": sum(row["hybrid_success"] for row in rows),
        "utility_successes": sum(row["utility_success"] for row in rows),
        "improved": sum(
            not row["hybrid_success"] and row["utility_success"]
            for row in rows
        ),
        "regressed": sum(
            row["hybrid_success"] and not row["utility_success"]
            for row in rows
        ),
        "mean_hybrid_steps": mean(row["hybrid_steps"] for row in rows),
        "mean_utility_steps": mean(row["utility_steps"] for row in rows),
        "mean_step_delta": mean(row["step_delta"] for row in rows),
        "both_success_pairs": len(both_succeeded),
        "mean_step_delta_both_success": (
            mean(row["step_delta"] for row in both_succeeded)
            if both_succeeded else None
        ),
    }


def load_metadata(directory: Path, arm: str, index: int) -> dict | None:
    path = directory / "eval" / arm / f"episode_{index:03d}_meta.json"
    return read_json(path) if path.is_file() else None


def validate_metadata(meta: dict, row: dict, arm: str) -> None:
    expected_ids = row[f"{arm}_ids"]
    if (
        meta.get("arm") != arm
        or meta.get("game_id") != row["game"]["id"]
        or meta.get("task") != row["task"]
        or meta.get("selected_ids") != expected_ids
    ):
        raise ValueError(
            f"Frozen evaluation metadata mismatch: "
            f"candidate={row['candidate_index']}, arm={arm}"
        )
    if type(meta.get("success")) is not bool:
        raise ValueError("Metadata success must be boolean")
    if (
        type(meta.get("decision_steps")) is not int
        or meta["decision_steps"] < 1
    ):
        raise ValueError("Metadata decision_steps must be a positive integer")


def audit(directory: Path) -> dict:
    plan = read_json(directory / "experiment_plan.json")
    if plan.get("schema") != "pipeline_paired_state_v1":
        raise ValueError("Unexpected paired pipeline plan schema")
    rows = plan.get("rows")
    if not isinstance(rows, list):
        raise ValueError("Plan rows must be a list")

    counts = {
        "identical": 0,
        "reordered_only": 0,
        "membership_changed": 0,
    }
    completed = {key: [] for key in counts}
    details = []
    seen_indices = set()
    incomplete = []

    for row in rows:
        index = row["candidate_index"]
        if type(index) is not int or index < 0 or index in seen_indices:
            raise ValueError("Invalid or duplicate candidate_index")
        seen_indices.add(index)

        hybrid_ids = row["hybrid_ids"]
        utility_ids = row["utility_ids"]
        kind = change_kind(hybrid_ids, utility_ids)
        if row["changed"] != (kind != "identical"):
            raise ValueError(f"Plan changed flag disagrees at index {index}")
        counts[kind] += 1

        detail = {
            "candidate_index": index,
            "game_id": row["game"]["id"],
            "task": row["task"],
            "kind": kind,
            "hybrid_ids": hybrid_ids,
            "utility_ids": utility_ids,
            "added_by_utility": [x for x in utility_ids if x not in hybrid_ids],
            "removed_by_utility": [x for x in hybrid_ids if x not in utility_ids],
            "evaluated": False,
        }
        details.append(detail)

        # The paired runner deliberately evaluates changed retrievals only.
        if kind == "identical":
            continue

        hybrid = load_metadata(directory, "hybrid", index)
        utility = load_metadata(directory, "utility", index)
        if hybrid is None or utility is None:
            incomplete.append(index)
            continue

        validate_metadata(hybrid, row, "hybrid")
        validate_metadata(utility, row, "utility")
        h_success = hybrid["success"]
        u_success = utility["success"]
        delta = utility["decision_steps"] - hybrid["decision_steps"]
        observation = {
            "candidate_index": index,
            "hybrid_success": h_success,
            "utility_success": u_success,
            "hybrid_steps": hybrid["decision_steps"],
            "utility_steps": utility["decision_steps"],
            "step_delta": delta,
        }
        completed[kind].append(observation)
        detail.update(observation)
        detail["evaluated"] = True

    return {
        "schema": "retrieval_composition_audit_v1",
        "directory": str(directory),
        "plan_rows": len(rows),
        "changed_rows": counts["reordered_only"] + counts["membership_changed"],
        "change_counts": counts,
        "incomplete_changed_indices": incomplete,
        "overall": summary(completed["reordered_only"] + completed["membership_changed"]),
        "by_change_kind": {
            kind: {
                "planned": counts[kind],
                **summary(completed[kind]),
            }
            for kind in counts
        },
        "pairs": details,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--directory", type=Path, required=True,
        help="Directory containing experiment_plan.json and eval/{hybrid,utility}/",
    )
    parser.add_argument(
        "--output", type=Path,
        help="Optional JSON report path; omitted by default for read-only inspection",
    )
    args = parser.parse_args()

    result = audit(args.directory)
    print(
        f"Plan: {result['plan_rows']} rows; "
        f"{result['changed_rows']} changed retrievals"
    )
    for kind, count in result["change_counts"].items():
        group = result["by_change_kind"][kind]
        print(
            f"  {kind}: {count} planned, "
            f"{group['completed_pairs']} completed"
        )
        if group["completed_pairs"]:
            print(
                f"    successes: {group['hybrid_successes']} -> "
                f"{group['utility_successes']}; "
                f"improved={group['improved']} regressed={group['regressed']}; "
                f"mean step delta (Utility-Hybrid)="
                f"{group['mean_step_delta']:+.2f}"
            )
    if result["incomplete_changed_indices"]:
        print(
            "  Incomplete changed pairs: "
            f"{result['incomplete_changed_indices']}"
        )
    for pair in result["pairs"]:
        if pair["kind"] == "identical":
            continue
        status = "done" if pair["evaluated"] else "missing"
        suffix = f" step_delta={pair['step_delta']:+d}" if pair["evaluated"] else ""
        print(
            f"  game={pair['candidate_index']:02d} {pair['kind']} "
            f"status={status} +{pair['added_by_utility']} "
            f"-{pair['removed_by_utility']}{suffix}"
        )

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"Saved audit: {args.output}")


if __name__ == "__main__":
    main()
