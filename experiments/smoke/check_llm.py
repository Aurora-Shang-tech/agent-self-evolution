from src.llm.client import LLMClient


def main():
    # 创建 LLM 客户端
    client = LLMClient()

    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant.",
        },
        {
            "role": "user",
            "content": "Reply with exactly: hello",
        },
    ]

    response = client.chat(messages)

    print("==== LLM Response ====")
    print(response)


if __name__ == "__main__":
    main()
