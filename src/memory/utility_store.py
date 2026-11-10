
"""Hybrid BM25 recall, optional utility reranking."""

from rank_bm25 import BM25Okapi

from src.memory.store import ExperienceStore
from src.memory.utility import UtilityTracker


def normalize(scores):
    values = [max(0.0, float(x)) for x in scores]
    maximum = max(values, default=0.0)

    if maximum <= 0:
        return [0.0] * len(values)

    return [x / maximum for x in values]


class HybridUtilityStore(ExperienceStore):
    def __init__(
        self,
        experiences,
        tracker: UtilityTracker | None = None,
        weight: float = 0.35,
        pool_size: int = 8,
    ):
        super().__init__()

        self.experiences = list(experiences)

        if not self.experiences or pool_size < 1 or weight < 0:
            raise ValueError("Invalid memory store settings")

        if (
            tracker is not None
            and len(tracker.stats) != len(self.experiences)
        ):
            raise ValueError("Tracker and memory size mismatch")

        self.tracker = tracker
        self.weight = weight
        self.pool_size = pool_size

        self.source_index = BM25Okapi([
            self._tokenize(item.source_task)
            for item in self.experiences
        ])

        self.lesson_index = BM25Okapi([
            self._tokenize(item.lesson)
            for item in self.experiences
        ])

        self.last_indices = []
        self.last_query = None

    def rank_indices(self, task: str, top_k: int = 3) -> list[int]:
        if top_k < 1:
            raise ValueError("top_k must be positive")

        query = self._tokenize(task)

        sources = normalize(
            self.source_index.get_scores(query)
        )

        lessons = normalize(
            self.lesson_index.get_scores(query)
        )

        relevance = [
            0.5 * (s + l)
            for s, l in zip(sources, lessons)
        ]

        ranked = sorted(
            range(len(relevance)),
            key=lambda i: (-relevance[i], i),
        )

        pool = ranked[:max(top_k, self.pool_size)]

        if self.tracker is not None:
            pool.sort(
                key=lambda i: (
                    -(
                        relevance[i]
                        + self.weight * self.tracker.bonus(i)
                    ),
                    -relevance[i],
                    i,
                )
            )

        return pool[:top_k]

    def retrieve(self, task: str, top_k: int = 3):
        self.last_query = task
        self.last_indices = self.rank_indices(task, top_k)

        return [
            self.experiences[i]
            for i in self.last_indices
        ]

