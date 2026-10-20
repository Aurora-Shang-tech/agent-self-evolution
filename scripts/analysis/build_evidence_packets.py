import hashlib
import json
from collections import defaultdict
from pathlib import Path


CLAIMS_PATH = Path("outputs/analysis/claims_dev_v2.jsonl")
EXPERIENCE_DIR = Path("outputs/experiences_qwen3.8-chat_train_seed42_1000")
BASELINE_DIR = Path("outputs/baseline_qwen3.8-chat_train_seed42_1000")
VERIFICATION_DIR = Path("outputs/counterfactual_qwen3.8-chat_train_seed42_1000")

OUTPUT_DIR = Path("outputs/analysis/evidence_packets_dev_v1")
OUTPUT_JSONL = Path("outputs/analysis/evidence_packets_dev_v1.jsonl")


def read_json(path):
    if not path.is_file():
        raise FileNotFoundError(path)

    return json.loads(path.read_text(encoding="utf-8"))


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compact_steps(steps, episode, branch):
    result = []

    for index, step in enumerate(steps, start=1):
        result.append({
            "evidence_id": f"{episode}/{branch}/step_{index:03d}",
            "step": index,
            "action": step["action"],
            "feedback": step["next_observation"],
            "valid_format": step["valid_format"],
            "admissible": step["admissible"],
            "done": step["done"],
            "won": step["won"],
        })

    return result


def build_index():
    index = defaultdict(list)

    for path in sorted(EXPERIENCE_DIR.glob("episode_*.json")):
        record = read_json(path)
        lesson = record["lesson"].strip()
        index[lesson].append(path.stem)

    return index


def build_episode(episode):
    baseline_path = BASELINE_DIR / f"{episode}.json"
    verification_path = VERIFICATION_DIR / f"{episode}.json"

    baseline = read_json(baseline_path)
    verification = read_json(verification_path)

    original_steps = baseline["steps"]
    critical_step = verification["critical_step"]

    if not 0 <= critical_step < len(original_steps):
        raise ValueError(f"Invalid critical step: {episode}")

    original_action = original_steps[critical_step]["action"]

    if original_action != verification["original_action"]:
        raise ValueError(f"Original action mismatch: {episode}")

    candidates = []

    for index, candidate in enumerate(
        verification["candidates"],
        start=1,
    ):
        branch = f"cf_{index}"
        trajectory_path = candidate.get("trajectory_path")
        continuation = []
        continuation_hash = None
        replacement_observation = None

        if trajectory_path is not None:
            path = Path(trajectory_path)
            trajectory = read_json(path)
            continuation = trajectory["steps"]
            continuation_hash = file_hash(path)

            if len(continuation) != candidate["continuation_steps"]:
                raise ValueError(
                    f"Continuation length mismatch: {episode}/{branch}"
                )

            if bool(trajectory["success"]) != bool(candidate["success"]):
                raise ValueError(
                    f"Continuation success mismatch: {episode}/{branch}"
                )

            if continuation:
                replacement_observation = continuation[0]["observation"]

        elif candidate["continuation_steps"] != 0:
            raise ValueError(
                f"Missing continuation trajectory: {episode}/{branch}"
            )

        immediate_success = (
            bool(candidate["success"])
            and candidate["continuation_steps"] == 0
        )

        candidates.append({
            "branch": branch,
            "replacement": {
                "evidence_id": f"{episode}/{branch}/replacement",
                "action": candidate["action"],
                "observation_after_action": replacement_observation,
                "immediate_success": immediate_success,
                "won": True if immediate_success else None,
            },
            "success": bool(candidate["success"]),
            "continuation_steps": candidate["continuation_steps"],
            "continuation": compact_steps(
                continuation,
                episode,
                branch,
            ),
            "trajectory_path": trajectory_path,
            "trajectory_sha256": continuation_hash,
        })

    return {
        "source_episode": episode,
        "task": baseline["task"],
        "critical_step_zero_based": critical_step,
        "critical_step_one_based": critical_step + 1,
        "original_action": original_action,
        "critical_observation": original_steps[
            critical_step
        ]["observation"],
        "baseline": {
            "success": bool(baseline["success"]),
            "truncated": bool(baseline.get("truncated", False)),
            "steps": compact_steps(
                original_steps,
                episode,
                "baseline",
            ),
        },
        "counterfactuals": candidates,
        "provenance": {
            "baseline_path": str(baseline_path),
            "baseline_sha256": file_hash(baseline_path),
            "verification_path": str(verification_path),
            "verification_sha256": file_hash(verification_path),
        },
    }


def main():
    records = [
        json.loads(line)
        for line in CLAIMS_PATH.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]

    if len(records) != 10:
        raise ValueError("Expected exactly 10 Dev records")

    ids = [record["experience_id"] for record in records]

    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate experience IDs")

    for record in records:
        if any(
            key in record
            for key in ("label", "rationale", "gold_label")
        ):
            raise ValueError("Gold labels must not appear in inputs")

    index = build_index()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    outputs = []
    total_claims = 0
    total_episodes = 0
    total_candidates = 0

    for number, record in enumerate(records, start=1):
        experience_id = record["experience_id"]
        lesson = record["lesson"].strip()

        episodes = index.get(lesson, [])

        if not episodes:
            raise ValueError(
                f"No source episodes found: {experience_id}"
            )

        evidence_episodes = [
            build_episode(episode)
            for episode in episodes
        ]

        packet = {
            "experience_id": experience_id,
            "task": record["task"],
            "lesson": record["lesson"],
            "claims": record["claims"],
            "source_episodes": evidence_episodes,
        }

        output_path = OUTPUT_DIR / f"{experience_id}.json"
        temporary_path = output_path.with_suffix(".tmp")

        temporary_path.write_text(
            json.dumps(
                packet,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        temporary_path.replace(output_path)

        outputs.append(packet)

        claim_count = len(record["claims"])
        candidate_count = sum(
            len(episode["counterfactuals"])
            for episode in evidence_episodes
        )

        total_claims += claim_count
        total_episodes += len(evidence_episodes)
        total_candidates += candidate_count

        print(
            f"[{number}/{len(records)}] "
            f"{experience_id} | "
            f"claims={claim_count} | "
            f"episodes={len(evidence_episodes)} | "
            f"candidates={candidate_count}"
        )

    with OUTPUT_JSONL.open("w", encoding="utf-8") as f:
        for packet in outputs:
            f.write(
                json.dumps(packet, ensure_ascii=False) + "\n"
            )

    print("\n" + "=" * 60)
    print("Evidence Packet Summary")
    print("=" * 60)
    print("Experiences:", len(outputs))
    print("Claims:", total_claims)
    print("Source episodes:", total_episodes)
    print("Counterfactual candidates:", total_candidates)
    print("Output:", OUTPUT_JSONL)


if __name__ == "__main__":
    main()
