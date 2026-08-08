import ollama


SYSTEM_PROMPT = """
You are a helpful AI assistant.
Give clear, accurate, and concise answers.
Explain technical concepts in simple language.
"""


def ask_model(prompt):
    response = ollama.chat(
        model="llama3.2:3b",
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response["message"]["content"]


prompts = [
    "What is artificial intelligence?",
    "Explain machine learning in simple terms.",
    "What is the difference between AI and ML?",
    "What is a neural network?",
    "What are the advantages of running an LLM locally?"
]


if __name__ == "__main__":
    print("Running Llama 3.2 3B locally with Ollama\n")

    for i, prompt in enumerate(prompts, start=1):
        print(f"Prompt {i}: {prompt}")
        answer = ask_model(prompt)
        print("Response:")
        print(answer)
        print("-" * 70)