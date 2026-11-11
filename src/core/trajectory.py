
"""ALFWorld episode trajectory and replayable LLM context."""

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class TrajectoryStep:
    observation: str
    admissible_actions: list[str]
    thought: str
    action: str
    valid_format: bool
    admissible: bool
    next_observation: str
    score: float
    done: bool
    won: bool


@dataclass
class Trajectory:
    task: str
    gamefile: str
    success: bool = False
    truncated: bool = False
    steps: list[TrajectoryStep] = field(default_factory=list)
    messages: list[dict] = field(default_factory=list)

    def add_step(self, step: TrajectoryStep) -> None:
        self.steps.append(step)

    def __len__(self) -> int:
        return len(self.steps)

    def save_json(self, path: str) -> None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(asdict(self), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load_json(cls, path: str) -> "Trajectory":
        data = json.loads(Path(path).read_text(encoding="utf-8"))

        return cls(
            task=data["task"],
            gamefile=data["gamefile"],
            success=data.get("success", False),
            truncated=data.get("truncated", False),
            steps=[
                TrajectoryStep(**step)
                for step in data.get("steps", [])
            ],
            messages=data.get("messages", []),
        )

