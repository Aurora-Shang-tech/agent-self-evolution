
"""Hybrid BM25 candidate recall with task-type-specific credit reranking."""

from rank_bm25 import BM25Okapi

from src.memory.store import ExperienceStore
from src.memory.contextual_utility import ContextualCredit


def normalize(scores) -> list[float]:
    positive = [max(0.0, float(x)) for x in scores]
    maximum = max(positive, default=0.0)
    if maximum == 0:
        return [0.0] * len(positive)
    return [x / maximum for x in positive]


class ContextualHybridStore(ExperienceStore):
    def __init__(
        self,
        experiences,
        task_type: str,
        credit: ContextualCredit | None = None,
        weight: float = 0.45,
        pool_size: int = 8,
    ):
        super().__init__()
        self.experiences = list(experiences)
        if not self.experiences or not task_type or weight < 0 or pool_size < 1:
            raise ValueError("Invalid store configuration")
        if credit is not None and credit.memory_size != len(self.experiences):
            raise ValueError("Credit model and memory catalog differ")

        self.task_type = task_type
        self.credit = credit
        self.weight = weight
        self.pool_size = pool_size
        self.last_query: str | None = None
        self.last_indices: list[int] = []

        self.source_bm25 = BM25Okapi([
            self._tokenize(x.source_task) for x in self.experiences
        ])
        self.lesson_bm25 = BM25Okapi([
            self._tokenize(x.lesson) for x in self.experiences
        ])

    def rank_indices(self, task: str, top_k: int = 3) -> list[int]:
        if top_k < 1:
            raise ValueError("top_k must be positive")
        query = self._tokenize(task)
        source = normalize(self.source_bm25.get_scores(query))
        lesson = normalize(self.lesson_bm25.get_scores(query))
        relevance = [0.5 * (s + l) for s, l in zip(source, lesson)]

        ranked = sorted(
            range(len(self.experiences)),
            key=lambda i: (-relevance[i], i),
        )
        candidates = ranked[:max(top_k, self.pool_size)]
        if self.credit is not None:
            candidates.sort(
                key=lambda i: (
                    -(relevance[i] + self.weight * self.credit.bonus(self.task_type, i)),
                    -relevance[i],
                    i,
                )
            )
        return candidates[:top_k]

    def retrieve(self, task: str, top_k: int = 3):
        self.last_query = task
        self.last_indices = self.rank_indices(task, top_k)
        return [self.experiences[i] for i in self.last_indices]

