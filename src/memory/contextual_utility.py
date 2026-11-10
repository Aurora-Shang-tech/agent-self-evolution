
"""Conservative, task-type-conditioned signals from paired trajectories."""

from dataclasses import asdict, dataclass


@dataclass
class CreditStat:
    total: float = 0.0
    count: int = 0
    positive: int = 0
    negative: int = 0


class ContextualCredit:
    def __init__(self, memory_size: int, prior_strength: float = 1.0):
        if memory_size < 1 or prior_strength <= 0:
            raise ValueError("Invalid memory size or prior strength")
        self.memory_size = memory_size
        self.prior_strength = float(prior_strength)
        self.by_type: dict[str, dict[int, CreditStat]] = {}

    def add(self, task_type: str, memory_index: int, signal: float) -> None:
        if not task_type or not 0 <= memory_index < self.memory_size:
            raise ValueError("Invalid task type or memory index")
        if not -1.0 <= signal <= 1.0:
            raise ValueError("Signal must be in [-1, 1]")
        if abs(signal) < 1e-12:
            return

        items = self.by_type.setdefault(task_type, {})
        item = items.setdefault(memory_index, CreditStat())
        item.total += signal
        item.count += 1
        item.positive += int(signal > 0)
        item.negative += int(signal < 0)

    def bonus(self, task_type: str, memory_index: int) -> float:
        if not 0 <= memory_index < self.memory_size:
            raise ValueError("Invalid memory index")
        item = self.by_type.get(task_type, {}).get(memory_index)
        if item is None:
            return 0.0
        return item.total / (self.prior_strength + item.count)

    def to_dict(self) -> dict:
        return {
            "memory_size": self.memory_size,
            "prior_strength": self.prior_strength,
            "by_type": {
                task_type: {
                    str(index): asdict(item)
                    for index, item in sorted(items.items())
                }
                for task_type, items in sorted(self.by_type.items())
            },
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ContextualCredit":
        result = cls(
            int(data["memory_size"]),
            float(data["prior_strength"]),
        )
        for task_type, items in data["by_type"].items():
            for index, stat in items.items():
                i = int(index)
                if not 0 <= i < result.memory_size:
                    raise ValueError("Memory index outside frozen catalog")
                result.by_type.setdefault(task_type, {})[i] = CreditStat(**stat)
        return result

