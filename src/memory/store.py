
"""Experience storage and task-aware retrieval."""

import json
import re
from pathlib import Path

from rank_bm25 import BM25Okapi

from src.memory.experience import Experience


class ExperienceStore:
    def __init__(self):
        self.experiences: list[Experience] = []
        self._new_flags: list[bool] = []

    def add(
        self,
        experience: Experience,
        *,
        is_new: bool = False,
    ) -> None:
        self.experiences.append(experience)
        self._new_flags.append(is_new)

    def __len__(self) -> int:
        return len(self.experiences)

    def _flags(self) -> list[bool]:
        if len(self._new_flags) == len(self.experiences):
            return self._new_flags
        return [False] * len(self.experiences)

    def save(self, path: str | Path) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)

        records = []

        for item, is_new in zip(
            self.experiences,
            self._flags(),
        ):
            record = {
                "failure_type": item.failure_type,
                "lesson": item.lesson,
                "source_task": item.source_task,
            }

            if is_new:
                record["is_new"] = True

            records.append(record)

        output.write_text(
            json.dumps(records, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    def load(self, path: str | Path) -> None:
        records = json.loads(
            Path(path).read_text(encoding="utf-8")
        )

        self.experiences = []
        self._new_flags = []

        for item in records:
            self.add(
                Experience(
                    failure_type=item["failure_type"],
                    lesson=item["lesson"],
                    source_task=item["source_task"],
                ),
                is_new=bool(item.get("is_new", False)),
            )

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return re.findall(r"[a-z0-9]+", text.lower())

    @staticmethod
    def _task_type(task: str) -> str | None:
        text = task.lower().strip()

        if (
            re.search(r"\b(?:desklamp|floorlamp|lamp)\b", text)
            and re.match(r"^(?:look at|examine)\b", text)
        ):
            return "look_at_obj_in_light"

        if (
            re.search(r"\b(?:put|place)\b", text)
            and re.search(r"\b(?:two|2)\b", text)
        ):
            return "pick_two_obj_and_place"

        if re.search(r"\b(?:clean|cleaned)\b", text):
            return "pick_clean_then_place_in_recep"

        if re.search(r"\b(?:cool|cold|cooled)\b", text):
            return "pick_cool_then_place_in_recep"

        if re.search(r"\b(?:heat|hot|heated)\b", text):
            return "pick_heat_then_place_in_recep"

        if re.search(r"\b(?:put|place)\b", text):
            return "pick_and_place_simple"

        return None

    def _rank(
        self,
        task: str,
        indices: list[int],
    ) -> list[int]:
        if not indices:
            return []

        corpus = [
            self._tokenize(
                self.experiences[index].source_task
            )
            for index in indices
        ]

        scores = BM25Okapi(corpus).get_scores(
            self._tokenize(task)
        )

        return [
            index
            for index, _ in sorted(
                zip(indices, scores),
                key=lambda pair: (
                    -float(pair[1]),
                    pair[0],
                ),
            )
        ]

    def retrieve(
        self,
        task: str,
        top_k: int = 3,
    ) -> list[Experience]:
        if top_k < 1:
            raise ValueError("top_k must be >= 1")

        if not self.experiences:
            return []

        flags = self._flags()

        original_indices = [
            index
            for index, is_new in enumerate(flags)
            if not is_new
        ]

        family = self._task_type(task)

        new_indices = [
            index
            for index, is_new in enumerate(flags)
            if is_new
            and family is not None
            and self._task_type(
                self.experiences[index].source_task
            ) == family
        ]

        if not new_indices:
            indices = (
                original_indices
                if original_indices
                else list(range(len(self.experiences)))
            )

            ranked = self._rank(task, indices)

            return [
                self.experiences[index]
                for index in ranked[:top_k]
            ]

        ranked_original = self._rank(
            task,
            original_indices,
        )

        ranked_new = self._rank(
            task,
            new_indices,
        )

        selected = (
            ranked_original[:max(0, top_k - 1)]
            + ranked_new[:1]
        )

        for index in ranked_original + ranked_new[1:]:
            if len(selected) >= top_k:
                break
            if index not in selected:
                selected.append(index)

        return [
            self.experiences[index]
            for index in selected
        ]
