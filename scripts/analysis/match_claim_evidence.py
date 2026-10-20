
import hashlib
import json
from pathlib import Path

from src.llm.client import LLMClient


INPUT_PATH = Path("outputs/analysis/evidence_packets_dev_v1.jsonl")
OUTPUT_DIR = Path("outputs/analysis/evidence_matches_dev_v2")
OUTPUT_JSONL = Path("outputs/analysis/evidence_matches_dev_v2.jsonl")

SCHEMA_VERSION = "evidence_matching_v2"
MAX_RETRIES = 3

VERDICTS = {
    "SUPPORTED",
    "CONTRADICTED",
    "INSUFFICIENT_EVIDENCE",
}

SYSTEM_PROMPT = """
You are a conservative evidence verifier for ALFWorld agent experiences.

Evaluate exactly one atomic claim using recorded environment evidence.

Rules:
1. Treat the claim as a hypothesis, not as established knowledge.
2. Use only the supplied environment actions, feedback, and task outcomes.
3. An agent's belief or the original lesson is not environmental evidence.
4. An admissible action is not necessarily a successful task completion.
5. A successful counterfactual continuation does not automatically validate
   the lesson or the claim.
6. A failed continuation does not establish why the task failed.
7. Distinguish object identity, object state, destination, and action order.
8. Do not infer that two object types are interchangeable from their names.
9. Do not infer the reverse direction of a substitution from one tested direction.
10. Preserve quantifiers and conditions in the claim.
11. A single successful example cannot prove a universal assertion.
12. When multiple factors change together, do not claim a unique causal cause.
13. For replacement actions without a recorded immediate result, do not invent
    an observation or a won value.
14. For claims that need an unperformed action or a controlled comparison,
    choose INSUFFICIENT_EVIDENCE and describe the missing experiment.
15. Use SUPPORTED only when recorded evidence directly supports the claim
    at the scope being evaluated.
16. Use CONTRADICTED only when recorded evidence directly conflicts with
    the claim. Otherwise use INSUFFICIENT_EVIDENCE.
17. Cite only exact evidence_id values provided in the input.
18. Explain any limitation caused by the tested environment state.
19. Do not output lesson-level judgments.
20. done=True means the episode ended; ONLY won=True means task success.
21. done=True and won=False is a failed episode, never a win.
22. Do not invent a later action that the claim does not require.
23. Do not infer that an action order is necessary merely because
    the final action changes the task outcome.
24. Distinguish evidence supporting one task instance from evidence
    validating a general rule across tasks.
25. A failed task after an attempted substitution may support a
    counterexample, but consider other unsatisfied goal conditions.
26. Distinguish a specific counterexample from a universal statement
    about all symbolic object types.
27. Keep statements such as often, typically, reliably, always,
    and never at their original logical strength.
28. Do not infer an object's physical state from its location alone.
29. Never invent an environment action, feedback, or successful outcome.
30. The evidence packet may omit additional experiments performed
    outside the recorded trajectories. Do not assume they occurred.
31. When evidence is insufficient, explain the smallest additional
    comparison needed to resolve the uncertainty.
32. Do not equate correlation with a uniquely established causal effect.
33. Return only one JSON object.

Required output:
{
  "claim_id": "C1",
  "verdict": "INSUFFICIENT_EVIDENCE",
  "supporting_evidence_ids": [],
  "contradicting_evidence_ids": [],
  "missing_evidence": [
    "A controlled substitution experiment is needed."
  ],
  "reason": "The available trajectories do not test the stated substitution."
}
"""


def fingerprint(payload, model):
    content = json.dumps(
        {
            "schema": SCHEMA_VERSION,
            "model": model,
            "prompt": SYSTEM_PROMPT,
            "payload": payload,
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def compact_steps(steps):
    result = []
    previous_observations = {}

    for step in steps:
        action = step["action"].strip().lower()
        read_only = (
            action in {"look", "inventory"}
            or action.startswith("examine ")
        )

        item = {
            "evidence_id": step["evidence_id"],
            "action": step["action"],
            "feedback": step["feedback"],
            "admissible": step["admissible"],
            "won": step["won"],
            "done": step["done"],
        }

        if not read_only:
            previous_observations = {}
            result.append(item)
            continue

        key = json.dumps(
            [
                item["action"],
                item["feedback"],
                item["admissible"],
                item["won"],
                item["done"],
            ],
            ensure_ascii=False,
        )

        if key in previous_observations:
            existing = result[previous_observations[key]]
            existing["repeated_count"] = (
                existing.get("repeated_count", 1) + 1
            )
            existing["last_repeated_evidence_id"] = item["evidence_id"]
            continue

        previous_observations[key] = len(result)
        result.append(item)

    return result


def build_model_input(packet, claim):
    episodes = []

    for episode in packet["source_episodes"]:
        branches = []

        for candidate in episode["counterfactuals"]:
            branches.append({
                "branch": candidate["branch"],
                "replacement": candidate["replacement"],
                "success": candidate["success"],
                "continuation_steps": candidate["continuation_steps"],
                "continuation": compact_steps(candidate["continuation"]),
            })

        episodes.append({
            "source_episode": episode["source_episode"],
            "critical_step_zero_based": episode["critical_step_zero_based"],
            "critical_observation": episode["critical_observation"],
            "baseline": {
                "success": episode["baseline"]["success"],
                "steps": compact_steps(episode["baseline"]["steps"]),
            },
            "counterfactuals": branches,
        })

    return {
        "experience_id": packet["experience_id"],
        "task": packet["task"],
        "claim": claim,
        "source_episodes": episodes,
    }


def collect_evidence_ids(packet):
    ids = set()

    for episode in packet["source_episodes"]:
        ids.update(
            step["evidence_id"]
            for step in episode["baseline"]["steps"]
        )

        for candidate in episode["counterfactuals"]:
            ids.add(candidate["replacement"]["evidence_id"])
            ids.update(
                step["evidence_id"]
                for step in candidate["continuation"]
            )

    return ids


def parse_response(response):
    if not isinstance(response, str):
        raise ValueError("Expected a string response")

    content = response.strip()

    if content.startswith("```"):
        lines = content.splitlines()

        if len(lines) < 3 or lines[-1].strip() != "```":
            raise ValueError("Invalid JSON code fence")

        content = "\n".join(lines[1:-1]).strip()

    try:
        return json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON: {exc}") from exc


def validate_match(data, claim_id, evidence_ids):
    required = {
        "claim_id",
        "verdict",
        "supporting_evidence_ids",
        "contradicting_evidence_ids",
        "missing_evidence",
        "reason",
    }

    if not isinstance(data, dict) or set(data) != required:
        raise ValueError("Invalid match object fields")

    if data["claim_id"] != claim_id:
        raise ValueError("Claim ID mismatch")

    if data["verdict"] not in VERDICTS:
        raise ValueError("Invalid verdict")

    for field in (
        "supporting_evidence_ids",
        "contradicting_evidence_ids",
    ):
        values = data[field]

        if not isinstance(values, list):
            raise ValueError(f"Invalid {field}")

        if any(
            not isinstance(value, str)
            or value not in evidence_ids
            for value in values
        ):
            raise ValueError(f"Unknown evidence ID in {field}")

        if len(values) != len(set(values)):
            raise ValueError(f"Duplicate evidence ID in {field}")

    if set(data["supporting_evidence_ids"]) & set(
        data["contradicting_evidence_ids"]
    ):
        raise ValueError("Same evidence cited as support and contradiction")

    missing = data["missing_evidence"]

    if (
        not isinstance(missing, list)
        or not all(
            isinstance(value, str) and value.strip()
            for value in missing
        )
    ):
        raise ValueError("Invalid missing_evidence")

    if (
        not isinstance(data["reason"], str)
        or not data["reason"].strip()
    ):
        raise ValueError("Empty reason")

    if (
        data["verdict"] == "SUPPORTED"
        and not data["supporting_evidence_ids"]
    ):
        raise ValueError("SUPPORTED requires supporting evidence")

    if (
        data["verdict"] == "CONTRADICTED"
        and not data["contradicting_evidence_ids"]
    ):
        raise ValueError("CONTRADICTED requires contradicting evidence")

    if (
        data["verdict"] == "INSUFFICIENT_EVIDENCE"
        and not data["missing_evidence"]
    ):
        raise ValueError("Insufficient evidence requires missing_evidence")

    return data


def judge_claim(llm, payload, evidence_ids):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": json.dumps(payload, ensure_ascii=False),
        },
    ]

    claim_id = payload["claim"]["claim_id"]

    for attempt in range(1, MAX_RETRIES + 1):
        response = llm.chat(messages)

        try:
            result = parse_response(response)
            return validate_match(
                result,
                claim_id,
                evidence_ids,
            )

        except (ValueError, TypeError) as exc:
            print(f"Validation failed {attempt}/{MAX_RETRIES}: {exc}")

            if attempt == MAX_RETRIES:
                raise

            messages.append({
                "role": "assistant",
                "content": response if isinstance(response, str) else "",
            })
            messages.append({
                "role": "user",
                "content": (
                    f"Validation error: {exc}. "
                    "Return a corrected, complete JSON object. "
                    "Use only evidence IDs included in the input."
                ),
            })

    raise RuntimeError("Evidence matching failed")


def main():
    records = [
        json.loads(line)
        for line in INPUT_PATH.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]

    if len(records) != 10:
        raise ValueError("Expected 10 Dev evidence packets")

    experience_ids = [record["experience_id"] for record in records]

    if len(experience_ids) != len(set(experience_ids)):
        raise ValueError("Duplicate experience IDs")

    llm = LLMClient()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_results = []
    counts = {verdict: 0 for verdict in VERDICTS}

    for index, packet in enumerate(records, start=1):
        experience_id = packet["experience_id"]
        evidence_ids = collect_evidence_ids(packet)
        matches = []

        print(f"\n[{index}/{len(records)}] {experience_id}")

        for claim in packet["claims"]:
            claim_id = claim["claim_id"]
            payload = build_model_input(packet, claim)
            current_hash = fingerprint(payload, llm.model)

            output_path = (
                OUTPUT_DIR / experience_id / f"{claim_id}.json"
            )
            output_path.parent.mkdir(parents=True, exist_ok=True)

            if output_path.exists():
                saved = json.loads(
                    output_path.read_text(encoding="utf-8")
                )

                if (
                    saved.get("input_hash") != current_hash
                    or saved.get("model") != llm.model
                ):
                    raise ValueError(
                        f"Cached result is incompatible: {output_path}"
                    )

                match = validate_match(
                    saved["match"],
                    claim_id,
                    evidence_ids,
                )

                print(f"  {claim_id}: cached {match['verdict']}")

            else:
                match = judge_claim(llm, payload, evidence_ids)

                saved = {
                    "experience_id": experience_id,
                    "claim_id": claim_id,
                    "model": llm.model,
                    "schema_version": SCHEMA_VERSION,
                    "input_hash": current_hash,
                    "claim": claim,
                    "match": match,
                }

                temporary = output_path.with_suffix(".tmp")
                temporary.write_text(
                    json.dumps(saved, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                temporary.replace(output_path)

                print(f"  {claim_id}: {match['verdict']}")

            counts[match["verdict"]] += 1
            matches.append(match)

        all_results.append({
            "experience_id": experience_id,
            "model": llm.model,
            "matches": matches,
        })

    with OUTPUT_JSONL.open("w", encoding="utf-8") as f:
        for record in all_results:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print("\n" + "=" * 60)
    print("Evidence Matching Summary")
    print("=" * 60)
    print("Experiences:", len(all_results))
    print("Claims:", sum(len(x["matches"]) for x in all_results))

    for verdict in sorted(VERDICTS):
        print(verdict, counts[verdict])

    print("Output:", OUTPUT_JSONL)


if __name__ == "__main__":
    main()

