import unittest
from types import SimpleNamespace

from src.agents.react_agent import ReActAgent


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

ACTIONS = (
    "go to drawer 1",
    "go to drawer 2",
    "go to cabinet 1",
    "go to countertop 1",
)


class FakeLLM:
    def __init__(self, actions):
        self.actions = iter(actions)
        self.prompts = []

    def chat(self, messages):
        self.prompts.append(messages[-1]["content"])
        action = next(self.actions)
        return f"Thought: Test decision.\nAction: {action}"


class FakeEnvironment:
    def reset(self):
        return SimpleNamespace(
            task="clean some pan and put it in drawer.",
            gamefile="/tmp/skill-runtime-test",
            observation="Your task is to: clean some pan and put it in drawer.",
            admissible_actions=ACTIONS,
        )

    def step(self, action):
        location = action.removeprefix("go to ")
        won = action == "go to countertop 1"

        return SimpleNamespace(
            observation=(
                f"You arrive at {location}. "
                f"On the {location}, you see nothing."
            ),
            admissible_actions=ACTIONS,
            admissible=True,
            score=1.0 if won else 0.0,
            done=won,
            won=won,
        )


class SkillRuntimeTests(unittest.TestCase):
    def build_agent(self, mode, actions):
        llm = FakeLLM(actions)

        kwargs = {}

        if mode is not None:
            kwargs = {
                "search_skill_mode": mode,
                "search_skill_proposal": PROPOSAL,
            }

        agent = ReActAgent(
            llm,
            max_steps=4,
            use_state_tracking=True,
            **kwargs,
        )

        return agent, llm

    def test_baseline_has_no_skill_event(self):
        agent, llm = self.build_agent(None, ACTIONS)
        trace = agent.run(FakeEnvironment())

        self.assertTrue(trace.success)
        self.assertEqual(len(trace.steps), 4)
        self.assertEqual(agent.llm_calls, 4)
        self.assertEqual(agent.skill_events, [])
        self.assertEqual(len(llm.prompts), 4)

    def test_hint_is_given_to_llm(self):
        agent, llm = self.build_agent("hint", ACTIONS)
        trace = agent.run(FakeEnvironment())

        self.assertTrue(trace.success)
        self.assertEqual(agent.llm_calls, 4)
        self.assertEqual(len(agent.skill_events), 1)
        self.assertTrue(agent.skill_events[0]["followed"])
        self.assertEqual(
            agent.skill_events[0]["step_index"], 3
        )
        self.assertIn(
            "Optional learned search suggestion",
            llm.prompts[3],
        )

    def test_execute_skips_one_llm_call(self):
        agent, llm = self.build_agent(
            "execute", ACTIONS[:3]
        )
        trace = agent.run(FakeEnvironment())

        self.assertTrue(trace.success)
        self.assertEqual(len(trace.steps), 4)
        self.assertEqual(agent.llm_calls, 3)
        self.assertEqual(len(llm.prompts), 3)
        self.assertEqual(len(agent.skill_events), 1)
        self.assertEqual(
            agent.skill_events[0]["chosen_action"],
            "go to countertop 1",
        )


if __name__ == "__main__":
    unittest.main()
