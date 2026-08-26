import requests

OLLAMA_URL = "http://localhost:11434/api/chat"

MODELS = [
    "llama3.2:3b",
    "qwen2.5:3b"
]

QUESTIONS = [
    "Explain artificial intelligence in simple words.",
    "Write a Python function to check whether a number is prime.",
    "What are the main differences between supervised and unsupervised learning?"
]


def ask_model(model, question):

    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful technical assistant. Give clear and accurate answers."
            },
            {
                "role": "user",
                "content": question
            }
        ],
        "stream": False
    }

    response = requests.post(
        OLLAMA_URL,
        json=payload
    )

    response.raise_for_status()

    return response.json()["message"]["content"]


for question_number, question in enumerate(QUESTIONS, 1):

    print("=" * 80)
    print(f"QUESTION {question_number}")
    print(question)
    print("=" * 80)

    for model in MODELS:

        print(f"\nMODEL: {model}")
        print("-" * 50)

        answer = ask_model(model, question)

        print(answer)

    print("\n")