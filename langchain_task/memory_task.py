from langchain_ollama import OllamaLLM
from langchain_core.messages import HumanMessage, AIMessage

# Create Ollama LLM
llm = OllamaLLM(model="llama3.2:3b")

# Conversation history
history = []

def chat(user_message):
    # Add user message to history
    history.append(HumanMessage(content=user_message))

    # Send complete conversation to Ollama
    response = llm.invoke(history)

    # Add AI response to history
    history.append(AIMessage(content=response))

    return response


print("===== Conversation Memory Test =====")

# Turn 1
print("\nTurn 1")
print("User: My name is Sahana.")
print("AI:", chat("My name is Sahana."))

# Turn 2
print("\nTurn 2")
print("User: I am learning Python and LangChain.")
print("AI:", chat("I am learning Python and LangChain."))

# Turn 3
print("\nTurn 3")
print("User: What is my name?")
print("AI:", chat("What is my name?"))

# Turn 4
print("\nTurn 4")
print("User: What am I learning?")
print("AI:", chat("What am I learning?"))

# Turn 5
print("\nTurn 5")
print("User: What have I told you so far?")
print("AI:", chat("What have I told you so far?"))

print("\n===== Memory Test Completed =====")