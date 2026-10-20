
import copy
import unittest
from types import SimpleNamespace

from src.agents.react_agent import ReActAgent


class MockLLM:
    def __init__(self, actions):
        self.actions = actions
        self.calls = []

    def chat(self, messages):
        self.calls.append(copy.deepcopy(messages))
        index = len(self.calls) - 1

        return (
            "Thought: Execute the next planned action.\n"
            f"Action: {self.actions[index]}"
        )


class MockEnvironment:
    def __init__(self, task, transitions):
        self.task = task
        self.transitions = transitions
        self.index = 0

    def reset(self):
        self.index = 0

        return SimpleNamespace(
            task=self.task,
            gamefile="mock_game",
            observation="You are in the kitchen.",
            admissible_actions=(self.transitions[0][0],),
        )

    def step(self, action):
        expected_action, observation, won = self.transitions[self.index]

        if action != expected_action:
            raise AssertionError(
                f"Expected {expected_action!r}, got {action!r}"
            )

        self.index += 1

        remaining = self.index < len(self.transitions)

        return SimpleNamespace(
            observation=observation,
            score=float(won),
            won=won,
            done=won or not remaining,
            admissible=True,
            admissible_actions=(
                (self.transitions[self.index][0],)
                if remaining
                else ()
            ),
        )


class TestAgentStateTracking(unittest.TestCase):

    def run_agent(self, task, transitions, enabled):
        actions = [item[0] for item in transitions]

        llm = MockLLM(actions)
        env = MockEnvironment(task, transitions)

        agent = ReActAgent(
            llm=llm,
            max_steps=50,
            use_state_tracking=enabled,
        )

        trajectory = agent.run(env)

        return llm, trajectory

    def test_tracking_disabled(self):
        transitions = [
            (
                "clean egg 1 with sinkbasin 1",
                "You clean the egg 1 using the sinkbasin 1.",
                False,
            ),
            (
                "move egg 1 to microwave 1",
                "You move the egg 1 to the microwave 1.",
                True,
            ),
        ]

        llm, trajectory = self.run_agent(
            "put a clean egg in microwave.",
            transitions,
            enabled=False,
        )

        self.assertTrue(trajectory.success)
        self.assertEqual(len(trajectory.steps), 2)

        for call in llm.calls:
            for message in call:
                self.assertNotIn(
                    "Task-state evidence",
                    message["content"],
                )

    def test_tracking_enabled(self):
        transitions = [
            (
                "clean egg 1 with sinkbasin 1",
                "You clean the egg 1 using the sinkbasin 1.",
                False,
            ),
            (
                "move egg 1 to microwave 1",
                "You move the egg 1 to the microwave 1.",
                True,
            ),
        ]

        llm, trajectory = self.run_agent(
            "put a clean egg in microwave.",
            transitions,
            enabled=True,
        )

        self.assertTrue(trajectory.success)
        self.assertEqual(len(llm.calls), 2)

        first_prompt = llm.calls[0][-1]["content"]
        second_prompt = llm.calls[1][-1]["content"]

        self.assertIn(
            "Confirmed instances: none confirmed by environment",
            first_prompt,
        )

        self.assertIn(
            "Confirmed instances: egg 1",
            second_prompt,
        )

    def test_state_invalidation(self):
        transitions = [
            (
                "cool potato 1 with fridge 1",
                "You cool the potato 1 using the fridge 1.",
                False,
            ),
            (
                "heat potato 1 with microwave 1",
                "You heat the potato 1 using the microwave 1.",
                False,
            ),
            (
                "look",
                "You are facing the microwave 1.",
                False,
            ),
        ]

        llm, trajectory = self.run_agent(
            "cool some potato and put it in microwave.",
            transitions,
            enabled=True,
        )

        self.assertFalse(trajectory.success)
        self.assertEqual(len(llm.calls), 3)

        after_cooling = llm.calls[1][-1]["content"]
        after_heating = llm.calls[2][-1]["content"]

        self.assertIn(
            "Confirmed instances: potato 1",
            after_cooling,
        )

        self.assertIn(
            "Confirmed instances: none confirmed by environment",
            after_heating,
        )


    def test_no_state_requirement_is_noop(self):
        transitions = [
            (
                "take apple 1 from table 1",
                "You pick up the apple 1 from the table 1.",
                False,
            ),
            (
                "move apple 1 to countertop 1",
                "You move the apple 1 to the countertop 1.",
                True,
            ),
        ]

        task = "put an apple in countertop."

        original_llm, original_trajectory = self.run_agent(
            task,
            transitions,
            enabled=False,
        )

        tracking_llm, tracking_trajectory = self.run_agent(
            task,
            transitions,
            enabled=True,
        )

        self.assertEqual(
            original_llm.calls,
            tracking_llm.calls,
        )

        self.assertEqual(
            [step.action for step in original_trajectory.steps],
            [step.action for step in tracking_trajectory.steps],
        )

        self.assertEqual(
            original_trajectory.success,
            tracking_trajectory.success,
        )



if __name__ == "__main__":
    unittest.main()

