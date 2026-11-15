import unittest

from src.core.trajectory import TrajectoryStep
from src.evolution.controller import gate
from src.skills.search_skill import ConditionalSearchSkill


PROPOSAL = {
    "decision": "CANDIDATE",
    "trigger": {
        "min_unproductive_search_locations": 3,
        "require_target_not_found": True,
        "require_unvisited_location": True,
    },
    "preferred_location_types": ["countertop"],
    "action_template": "go to <selected unvisited countertop>",
}


def observed_step(location, objects):
    return TrajectoryStep(
        observation="",
        admissible_actions=[],
        thought="",
        action=f"go to {location}",
        valid_format=True,
        admissible=True,
        next_observation=(
            f"You arrive at {location}. "
            f"On the {location}, you see {objects}."
        ),
        score=0.0,
        done=False,
        won=False,
    )


class SkillRejectionTests(unittest.TestCase):
    def test_three_unproductive_locations_required(self):
        skill = ConditionalSearchSkill(
            PROPOSAL,
            "clean some butterknife and put it in drawer.",
        )

        actions = (
            "go to countertop 1",
            "go to cabinet 4",
        )

        for index in range(1, 3):
            skill.observe(
                observed_step(
                    f"cabinet {index}",
                    f"a knife {index}",
                )
            )
            self.assertIsNone(skill.suggest(actions))

        skill.observe(
            observed_step("cabinet 3", "a knife 3")
        )

        self.assertEqual(
            skill.suggest(actions),
            "go to countertop 1",
        )

    def test_exact_target_found_prevents_trigger(self):
        skill = ConditionalSearchSkill(
            PROPOSAL,
            "clean some butterknife and put it in drawer.",
        )

        for index in range(1, 4):
            skill.observe(
                observed_step(
                    f"cabinet {index}",
                    f"a knife {index}",
                )
            )

        skill.observe(
            observed_step(
                "diningtable 1",
                "a butterknife 1",
            )
        )

        self.assertIsNone(
            skill.suggest(("go to countertop 1",))
        )

    def test_below_efficiency_gate_rejected(self):
        baseline = [
            {
                "game_id": f"test_{i}",
                "success": True,
                "decision_steps": 20,
            }
            for i in range(12)
        ]

        candidate = [
            {
                **record,
                "decision_steps": 19 if i < 11 else 22,
            }
            for i, record in enumerate(baseline)
        ]

        result = gate(
            baseline,
            candidate,
            max_regressions=0,
            min_step_saving=1.0,
        )

        self.assertFalse(result["accept"])
        self.assertAlmostEqual(
            result["average_step_delta"],
            -0.75,
        )
        self.assertEqual(
            result["reason"],
            "no_sufficient_gain",
        )


if __name__ == "__main__":
    unittest.main()
