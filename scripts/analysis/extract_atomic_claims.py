
import hashlib
import json
from pathlib import Path

from src.llm.client import LLMClient


INPUT_PATH = Path("outputs/analysis/claim_extraction_dev_v1.jsonl")
OUTPUT_DIR = Path("outputs/analysis/claims_dev_v2")
OUTPUT_JSONL = Path("outputs/analysis/claims_dev_v2.jsonl")

SCHEMA_VERSION = "claim_extraction_v2"
MAX_RETRIES = 3

CLAIM_TYPES = {
    "object_type",
    "object_state",
    "action_effect",
    "action_order",
    "search_strategy",
    "success_condition",
    "other",
}

SYSTEM_PROMPT = """
You extract atomic, testable claims from reusable ALFWorld lessons.

Your job is claim extraction, NOT fact checking.

Rules:
1. Extract all distinct, meaningful claims explicitly asserted by the lesson.
2. Split independent assertions into separate claims.
3. Each claim must express one independently testable proposition.
4. Preserve the original meaning, scope, conditions, and quantifiers.
5. Preserve strong claims such as "always" and "never"; do not silently weaken them.
6. Do not add claims based on your own ALFWorld knowledge.
7. Do not infer facts from the task description that the lesson does not assert.
8. Do not assume any claim is correct.
9. Keep claims in English.
10. Extract between 1 and 6 claims.
11. Preserve logical strength precisely. "Do not assume X" or
    "X is not guaranteed" must NEVER become "X is false" or "X never happens".
12. Preserve qualifiers such as typically, often, sometimes, may,
    can, always, never, only, and must.
13. Treat warnings against making an assumption as claims about
    insufficient guarantees, not universal claims about object states.
14. Split independently testable requirements such as "locate and use"
    or "check availability and select" into separate claims when meaningful.
15. Do not split a single conditional dependency into disconnected
    unconditional claims.
16. Every statement must be logically entailed by the original lesson.
17. Before returning, check whether each statement is stronger than
    its source_span. If so, rewrite it to preserve the original scope.
18. Never silently correct an incorrect claim using world knowledge.

For each claim, provide:
- claim_id: C1, C2, ...
- claim_type: one of object_type, object_state, action_effect,
  action_order, search_strategy, success_condition, other
- statement: one standalone, testable proposition
- preconditions: explicit conditions stated or implied by the lesson
- source_span: an exact contiguous substring copied from the lesson

The task description is context only, not evidence.

Return only a JSON object in this format:

{
  "claims": [
    {
      "claim_id": "C1",
      "claim_type": "object_type",
      "statement": "A mug cannot substitute for a cup in ALFWorld tasks.",
      "preconditions": ["The task requires a cup."],
      "source_span": "a mug is not a cup"
    }
  ]
}

Do not output verification results, confidence scores, or evidence labels.
"""


def input_hash(record):
    payload = {
        "task": record["task"],
        "lesson": record["lesson"],
        "schema_version": SCHEMA_VERSION,
        "prompt": SYSTEM_PROMPT,
    }
    content = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def parse_response(response):
    if not isinstance(response, str):
        raise ValueError("Model response must be a string")

    content = response.strip()

    if content.startswith("```"):
        lines = content.splitlines()

        if lines[-1].strip() != "```":
            raise ValueError("Unclosed JSON code fence")

        content = "\n".join(lines[1:-1]).strip()

    try:
        return json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON: {exc}") from exc


def validate_claims(data, lesson):
    if not isinstance(data, dict) or set(data) != {"claims"}:
        raise ValueError("Expected an object containing only claims")

    claims = data["claims"]

    if not isinstance(claims, list) or not 1 <= len(claims) <= 6:
        raise ValueError("Expected 1 to 6 claims")

    expected_fields = {
        "claim_id",
        "claim_type",
        "statement",
        "preconditions",
        "source_span",
    }

    statements = set()

    for index, claim in enumerate(claims, start=1):
        if not isinstance(claim, dict):
            raise ValueError(f"Claim {index} is not an object")

        if set(claim) != expected_fields:
            raise ValueError(f"Claim {index} has invalid fields")

        if claim["claim_id"] != f"C{index}":
            raise ValueError(f"Claim {index} has invalid ID")

        if claim["claim_type"] not in CLAIM_TYPES:
            raise ValueError(f"Claim {index} has invalid claim_type")

        statement = claim["statement"]

        if not isinstance(statement, str) or not statement.strip():
            raise ValueError(f"Claim {index} has empty statement")

        normalized = statement.strip().casefold()

        if normalized in statements:
            raise ValueError("Duplicate claim statements")

        statements.add(normalized)

        preconditions = claim["preconditions"]

        if not isinstance(preconditions, list):
            raise ValueError(f"Claim {index} has invalid preconditions")

        if not all(
            isinstance(item, str) and item.strip()
            for item in preconditions
        ):
            raise ValueError(f"Claim {index} has empty preconditions")

        span = claim["source_span"]

        if (
            not isinstance(span, str)
            or not span.strip()
            or span not in lesson
        ):
            raise ValueError(
                f"Claim {index} source_span is not an exact lesson substring"
            )

    return claims


def extract_claims(llm, record):
    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": json.dumps(
                {
                    "task": record["task"],
                    "lesson": record["lesson"],
                },
                ensure_ascii=False,
            ),
        },
    ]

    for attempt in range(1, MAX_RETRIES + 1):
        response = llm.chat(messages)

        try:
            data = parse_response(response)
            return validate_claims(data, record["lesson"])

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
                    f"Your previous output failed validation: {exc}. "
                    "Correct it and return the entire JSON object again. "
                    "Each source_span must be copied exactly from the lesson."
                ),
            })

    raise RuntimeError("Claim extraction failed")


def main():
    with INPUT_PATH.open("r", encoding="utf-8") as f:
        records = [
            json.loads(line)
            for line in f
            if line.strip()
        ]

    if len(records) != 10:
        raise ValueError("Expected exactly 10 development records")

    ids = [record["experience_id"] for record in records]

    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate experience IDs")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    llm = LLMClient()
    outputs = []

    for index, record in enumerate(records, start=1):
        experience_id = record["experience_id"]
        output_path = OUTPUT_DIR / f"{experience_id}.json"
        fingerprint = input_hash(record)

        print(f"\n[{index}/{len(records)}] {experience_id}")

        if output_path.exists():
            result = json.loads(
                output_path.read_text(encoding="utf-8")
            )

            if (
                result.get("input_hash") != fingerprint
                or result.get("schema_version") != SCHEMA_VERSION
            ):
                raise ValueError(
                    f"Existing output does not match input: {output_path}"
                )

            validate_claims(
                {"claims": result["claims"]},
                record["lesson"],
            )
            print("Existing valid result, skipping")

        else:
            claims = extract_claims(llm, record)

            result = {
                "experience_id": experience_id,
                "schema_version": SCHEMA_VERSION,
                "input_hash": fingerprint,
                "task": record["task"],
                "lesson": record["lesson"],
                "claims": claims,
            }

            temporary = output_path.with_suffix(".tmp")
            temporary.write_text(
                json.dumps(result, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            temporary.replace(output_path)

        outputs.append(result)

        for claim in result["claims"]:
            print(
                f"  {claim['claim_id']} "
                f"[{claim['claim_type']}] "
                f"{claim['statement']}"
            )

    with OUTPUT_JSONL.open("w", encoding="utf-8") as f:
        for result in outputs:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

    total_claims = sum(len(x["claims"]) for x in outputs)

    print("\n" + "=" * 60)
    print("Claim Extraction Summary")
    print("=" * 60)
    print("Experiences:", len(outputs))
    print("Total claims:", total_claims)
    print("Average claims:", round(total_claims / len(outputs), 2))
    print("Output:", OUTPUT_JSONL)


if __name__ == "__main__":
    main()

