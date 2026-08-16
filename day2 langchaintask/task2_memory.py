from langchain_ollama import OllamaLLM
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Create Ollama model
llm = OllamaLLM(
    model="llama3.2:3b"
)

# Create conversation memory
memory = ConversationBufferMemory(
    return_messages=True
)

# Create prompt
prompt = PromptTemplate(
    input_variables=["history", "input"],
    template="""
You are a helpful assistant.

Conversation history:
{history}

User:
{input}

Assistant:
"""
)

# Create output parser
parser = StrOutputParser()

# Create chain
chain = prompt | llm | parser

# 5 conversation turns
questions = [
    "My name is Sahana.",
    "What is my name?",
    "I am learning Python and AI.",
    "What am I learning?",
    "Can you summarize what you know about me from this conversation?"
]

for i, user_input in enumerate(questions, 1):

    print(f"\n--- Turn {i} ---")
    print("User:", user_input)

    # Get conversation history
    history_messages = memory.load_memory_variables({})["history"]

    # Convert messages to text
    history_text = "\n".join(
        [
            f"{message.type}: {message.content}"
            for message in history_messages
        ]
    )

    # Send to chain
    response = chain.invoke({
        "history": history_text,
        "input": user_input
    })

    print("Assistant:", response)

    # Save conversation
    memory.save_context(
        {"input": user_input},
        {"output": response}
    )

print("\n===== FINAL CONVERSATION HISTORY =====")

history = memory.load_memory_variables({})["history"]

for message in history:
    print(f"{message.type}: {message.content}")