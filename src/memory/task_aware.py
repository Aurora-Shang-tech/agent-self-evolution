
"""Conservative task-aware filtering for experience retrieval."""

import re
from dataclasses import asdict, dataclass

from rank_bm25 import BM25Okapi

from src.memory.store import ExperienceStore


@dataclass(frozen=True)
class TaskFeatures:
    operation: str | None
    quantity: int
    destination: str | None
    target_object: str | None


@dataclass(frozen=True)
class LessonScope:
    operations: tuple[str, ...]
    multiple_only: bool
    specific_object: str | None
    requires_prior_failure: bool


_OPERATION_PATTERN = {
    "clean": r"\bclean(?:ed|ing)?\b",
    "cool": r"\bcool(?:ed|ing)?\b|\bcold\b",
    "heat": r"\bheat(?:ed|ing)?\b|\bhot\b",
}


def parse_task(task: str) -> TaskFeatures:
    text = task.strip().lower()

    operation = next(
        (
            name
            for name, pattern in _OPERATION_PATTERN.items()
            if re.search(pattern, text)
        ),
        None,
    )

    quantity = (
        2
        if re.search(r"\b(?:two|2)\b", text)
        else 1
    )

    place = re.search(
        r"\b(?:put|place)\b.*?\b(?:in|on)\s+"
        r"(?:(?:the|a|an)\s+)?([a-z][\w-]*)",
        text,
    )

    destination = (
        place.group(1)
        if place
        else None
    )

    obj = re.search(
        r"\b(?:put|place|heat|cool|clean)\s+"
        r"(?:(?:a|an|the|some|two|2)\s+)?"
        r"(?:(?:clean|cool|cold|hot|heated)\s+)?"
        r"([a-z][\w-]*)",
        text,
    )

    target_object = (
        obj.group(1)
        if obj
        else None
    )

    if target_object in {
        "it",
        "them",
        "some",
        "one",
        "two",
    }:
        target_object = None

    return TaskFeatures(
        operation=operation,
        quantity=quantity,
        destination=destination,
        target_object=target_object,
    )


def _strip_examples(text: str) -> str:
    return re.sub(
        r"\([^()]*(?:e\.g\.|for example|such as)[^()]*\)",
        "",
        text,
        flags=re.IGNORECASE,
    )


def parse_lesson(lesson: str) -> LessonScope:
    text = _strip_examples(
        lesson.lower()
    )

    condition = text.split(",", 1)[0].strip()

    is_conditional = bool(
        re.match(
            r"^(?:when|if)\b",
            condition,
        )
    )

    operations = ()

    if is_conditional:
        is_generic_state = (
            "specific state" in condition
        )

        if not is_generic_state:
            operations = tuple(
                name
                for name, pattern
                in _OPERATION_PATTERN.items()
                if re.search(
                    pattern,
                    condition,
                )
            )

    elif re.match(
        r"^(?:always|before|do not|never|ensure)\b",
        condition,
    ):
        operations = tuple(
            name
            for name, pattern
            in _OPERATION_PATTERN.items()
            if re.search(
                pattern,
                condition,
            )
        )

    multiple_only = (
        is_conditional
        and bool(
            re.search(
                r"\b(?:multiple|two|both)\s+"
                r"(?:specific\s+)?"
                r"(?:items|objects)\b",
                condition,
            )
        )
    )

    specific_object = None

    if is_conditional:
        explicit = re.search(
            r"\b(?:requires|specifies|calls for|asks for)\b"
            r"[^,]{0,100}?"
            r"\b(?:item|object)\s+like\s+"
            r"['\"]([a-z][a-z0-9_-]*)['\"]",
            condition,
        )

        if explicit:
            specific_object = (
                explicit.group(1)
            )

    requires_prior_failure = (
        is_conditional
        and bool(
            re.search(
                r"\b(?:fails?|failed|invalid)\b",
                condition,
            )
        )
        and bool(
            re.search(
                r"\b(?:action|attempt|command)\b",
                condition,
            )
        )
    )

    return LessonScope(
        operations=operations,
        multiple_only=multiple_only,
        specific_object=specific_object,
        requires_prior_failure=requires_prior_failure,
    )


def rejection_reason(
    task: TaskFeatures,
    scope: LessonScope,
) -> str | None:

    if scope.requires_prior_failure:
        return "requires_prior_failure"

    if (
        scope.multiple_only
        and task.quantity < 2
    ):
        return "requires_multiple_objects"

    if (
        scope.operations
        and task.operation not in scope.operations
    ):
        return "operation_mismatch"

    if (
        scope.specific_object
        and task.target_object is not None
        and scope.specific_object
        != task.target_object
    ):
        return "object_mismatch"

    return None


def _normalize(scores) -> list[float]:
    values = [
        max(0.0, float(value))
        for value in scores
    ]

    largest = max(
        values,
        default=0.0,
    )

    if largest <= 0:
        return [0.0] * len(values)

    return [
        value / largest
        for value in values
    ]


class TaskAwareExperienceStore(
    ExperienceStore
):

    def __init__(
        self,
        experiences,
        pool_size: int = 8,
    ):
        super().__init__()

        self.experiences = list(
            experiences
        )

        if not self.experiences:
            raise ValueError(
                "Memory store must not be empty"
            )

        if pool_size < 1:
            raise ValueError(
                "pool_size must be >= 1"
            )

        self.pool_size = pool_size

        self.scopes = [
            parse_lesson(item.lesson)
            for item in self.experiences
        ]

        self.source_bm25 = BM25Okapi([
            self._tokenize(
                item.source_task
            )
            for item in self.experiences
        ])

        self.lesson_bm25 = BM25Okapi([
            self._tokenize(
                item.lesson
            )
            for item in self.experiences
        ])

    def explain(
        self,
        task: str,
        top_k: int = 3,
    ) -> dict:

        if top_k < 1:
            raise ValueError(
                "top_k must be >= 1"
            )

        features = parse_task(task)

        query = self._tokenize(task)

        source_scores = _normalize(
            self.source_bm25.get_scores(
                query
            )
        )

        lesson_scores = _normalize(
            self.lesson_bm25.get_scores(
                query
            )
        )

        scores = [
            0.5 * (source + lesson)
            for source, lesson in zip(
                source_scores,
                lesson_scores,
            )
        ]

        ranking = sorted(
            range(len(self.experiences)),
            key=lambda i: (
                -scores[i],
                i,
            ),
        )

        def entry(index: int) -> dict:
            item = self.experiences[index]

            return {
                "index": index,
                "score": round(
                    scores[index],
                    6,
                ),
                "lesson": item.lesson,
                "source_task": item.source_task,
                "scope": asdict(
                    self.scopes[index]
                ),
            }

        baseline = [
            entry(i)
            for i in ranking[:top_k]
        ]

        candidates = ranking[
            :max(
                self.pool_size,
                top_k,
            )
        ]

        selected = []
        rejected = []

        for index in candidates:
            reason = rejection_reason(
                features,
                self.scopes[index],
            )

            if reason is not None:
                rejected.append({
                    **entry(index),
                    "reason": reason,
                })
                continue

            if len(selected) < top_k:
                selected.append(
                    entry(index)
                )

        return {
            "task": task,
            "features": asdict(
                features
            ),
            "baseline": baseline,
            "selected": selected,
            "rejected": rejected,
            "candidate_count": len(
                candidates
            ),
        }

    def retrieve(
        self,
        task: str,
        top_k: int = 3,
    ):
        result = self.explain(
            task,
            top_k,
        )

        return [
            self.experiences[
                item["index"]
            ]
            for item in result["selected"]
        ]

