import unittest
from pathlib import Path

from scripts.run_unified_batch import (
    TASK_TYPES,
    select_games,
)
from src.data.alfworld_data import GameExample


class UnifiedBatchTests(unittest.TestCase):
    def make_games(self):
        games = []

        for index, task_type in enumerate(TASK_TYPES):
            for variant in range(2):
                game_id = f"train:game_{index}_{variant}"

                games.append(
                    GameExample(
                        id=game_id,
                        split="train",
                        game_file=Path(
                            f"mock_{index}_{variant}.json"
                        ),
                        task_type=task_type,
                    )
                )

        return games

    def test_selects_six_unused_train_games(self):
        games = self.make_games()

        excluded = {
            f"train:game_{i}_0"
            for i in range(len(TASK_TYPES))
        }

        selected = select_games(
            games,
            excluded,
            seed=20261116,
        )

        self.assertEqual(len(selected), 6)
        self.assertEqual(
            {g.task_type for g in selected},
            set(TASK_TYPES),
        )
        self.assertFalse(
            {g.id for g in selected} & excluded
        )
        self.assertTrue(
            all(g.split == "train" for g in selected)
        )

    def test_insufficient_games_rejected(self):
        games = self.make_games()

        excluded = {
            g.id
            for g in games
            if g.task_type == TASK_TYPES[0]
        }

        with self.assertRaises(RuntimeError):
            select_games(games, excluded, seed=42)


if __name__ == "__main__":
    unittest.main()
