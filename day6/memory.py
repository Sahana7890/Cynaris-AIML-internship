from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

llm = OllamaLLM(model="llama3.2:3b")

prompt = PromptTemplate(
    input_variables=["history", "input"],
    template="""
You are a helpful assistant.

Conversation history:
{history}

User: {input}

Assistant:
"""
)

parser = StrOutputParser()

chain = prompt | llm | parser

# ConversationBufferMemory-style history
history = []

turns = [
    "My name is Sahana.",
    "What is my name?",
    "I am learning LangChain.",
    "What am I learning?",
    "Can you summarize what you know about me from this conversation?"
]

for i, user_input in enumerate(turns, 1):

    history_text = "\n".join(
        f"User: {user_msg}\nAssistant: {assistant_msg}"
        for user_msg, assistant_msg in history
    )

    response = chain.invoke({
        "history": history_text,
        "input": user_input
    })

    print(f"\n--- Turn {i} ---")
    print("User:", user_input)
    print("Assistant:", response)

    history.append((user_input, response))

print("\n==============================")
print("Conversation history maintained")
print("==============================")

for user_msg, assistant_msg in history:
    print(f"User: {user_msg}")
    print(f"Assistant: {assistant_msg}\n")