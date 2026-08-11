from langchain_ollama import OllamaLLM
from langchain_core.tools import tool

# Create Ollama model
llm = OllamaLLM(model="llama3.2:3b")


# Tool 1: Calculator
@tool
def calculator(expression: str) -> str:
    """Calculate a mathematical expression."""
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return str(result)
    except Exception:
        return "Invalid mathematical expression."


# Tool 2: Web Search Stub
@tool
def web_search(query: str) -> str:
    """A simple web search stub for demonstration."""
    
    fake_results = {
        "python": "Python is a popular programming language used for AI, data science, and web development.",
        "langchain": "LangChain is a framework for developing applications powered by language models.",
        "ollama": "Ollama allows you to run large language models locally."
    }

    query_lower = query.lower()

    for keyword, result in fake_results.items():
        if keyword in query_lower:
            return result

    return "No web results found for this query."


# Display available tools
print("===== LangChain Agent Tool Test =====")

print("\nAvailable Tools:")
print("1. Calculator")
print("2. Web Search Stub")


# TASK 1 - Calculator
print("\n===== Task 1 =====")
question = "Calculate 25 * 4 + 10"

print("User:", question)

result = calculator.invoke({
    "expression": "25 * 4 + 10"
})

print("Calculator:", result)


# TASK 2 - Web Search
print("\n===== Task 2 =====")
question = "Search for information about Python"

print("User:", question)

result = web_search.invoke({
    "query": "Python"
})

print("Web Search:", result)


# TASK 3 - Web Search
print("\n===== Task 3 =====")
question = "Search for information about LangChain"

print("User:", question)

result = web_search.invoke({
    "query": "LangChain"
})

print("Web Search:", result)


print("\n===== Agent Tool Tasks Completed =====")