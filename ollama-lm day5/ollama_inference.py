import requests

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "llama3.2:3b"

SYSTEM_PROMPT = """
You are a helpful AI assistant.
Give clear, accurate, and beginner-friendly answers.
For programming questions, provide simple examples.
"""


def ask_ollama(prompt):
    payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        "stream": False
    }

    response = requests.post(OLLAMA_URL, json=payload)
    response.raise_for_status()

    return response.json()["message"]["content"]


prompts = [
    "Explain machine learning in simple words.",
    "Write a Python function to reverse a string.",
    "What is the difference between AI and machine learning?",
    "Explain cybersecurity to a beginner.",
    "Give three real-world applications of Large Language Models."
]


for i, prompt in enumerate(prompts, 1):

    print("=" * 70)
    print(f"PROMPT {i}: {prompt}")
    print("=" * 70)

    answer = ask_ollama(prompt)

    print(answer)
    print()