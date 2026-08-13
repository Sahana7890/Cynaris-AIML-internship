import ollama


questions = [
    "Explain artificial intelligence in simple terms.",
    "What is the difference between supervised and unsupervised learning?",
    "Explain how a neural network works."
]


models = [
    "llama3.2:3b",
    "qwen2.5:3b"
]


def ask_model(model, question):
    response = ollama.chat(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "You are a helpful AI assistant. Give clear and concise answers."
            },
            {
                "role": "user",
                "content": question
            }
        ]
    )

    return response["message"]["content"]


if __name__ == "__main__":

    for question_number, question in enumerate(questions, start=1):

        print("=" * 80)
        print(f"QUESTION {question_number}")
        print(question)
        print("=" * 80)

        for model in models:

            print(f"\nMODEL: {model}")
            print("-" * 40)

            answer = ask_model(model, question)

            print(answer)
            print()