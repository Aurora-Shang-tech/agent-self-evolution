
"""Weak, exposure-conditioned utility estimates; not causal effects."""

from dataclasses import asdict, dataclass


@dataclass
class UtilityStat:
    q: float = 0.0
    uses: int = 0
    successes: int = 0
    failures: int = 0


class UtilityTracker:
    def __init__(self, size: int, alpha: float = 0.3):
        if size <= 0 or not 0 < alpha <= 1:
            raise ValueError("Invalid size or alpha")

        self.alpha = alpha
        self.stats = [UtilityStat() for _ in range(size)]

    def observe(self, indices: list[int], success: bool) -> None:
        if len(indices) != len(set(indices)):
            raise ValueError("Duplicate lesson indices")

        reward = 1.0 if success else -1.0

        for index in indices:
            state = self.stats[index]
            state.q += self.alpha * (reward - state.q)
            state.uses += 1
            state.successes += int(success)
            state.failures += int(not success)

    def bonus(self, index: int, min_uses: int = 2) -> float:
        state = self.stats[index]
        return state.q * min(1.0, state.uses / min_uses)

    def dump(self) -> dict:
        return {
            "alpha": self.alpha,
            "stats": [asdict(s) for s in self.stats],
        }

    @classmethod
    def load(cls, data: dict) -> "UtilityTracker":
        result = cls(len(data["stats"]), float(data["alpha"]))
        result.stats = [
            UtilityStat(**item)
            for item in data["stats"]
        ]
        return result

