"""检查temperature=0时LLM API的输出一致性"""

from collections import Counter
from src.llm.client import LLMClient

def main():
    llm = LLMClient()

    messages = [
        {
            "role": "system",
            "content": "You are an ALFWorld agent. Return only one action.",
        },
        {
            "role": "user",
            "content": (
                "Task: put a clean egg in microwave.\n"
                "Observation: You are at fridge 1. "
                "You see egg 1.\n"
                "Available actions: take egg 1 from fridge 1, "
                "go to sinkbasin 1, go to microwave 1.\n"
                "What is your next action?"
            ),
        },
    ]

    outputs = []

    for i in range(10):
        result = llm.chat(messages)
        outputs.append(result)

        print(f"[{i + 1:02d}] {result!r}")

    counts = Counter(outputs)

    print("\nUnique outputs:", len(counts))

    for output, count in counts.items():
        print(f"Count: {count}, Output: {output!r}")


if __name__ == "__main__":
    main()
