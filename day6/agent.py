from langchain_ollama import OllamaLLM
from langchain_core.tools import tool


# Tool 1: Calculator
@tool
def calculator(expression: str) -> str:
    """Calculate a mathematical expression."""

    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return str(result)
    except Exception as e:
        return f"Calculation error: {e}"


# Tool 2: Web search stub
@tool
def web_search(query: str) -> str:
    """Simulate a web search and return sample results."""

    return (
        f"Web search results for '{query}': "
        "LangChain is a framework for building applications "
        "powered by language models."
    )


tools = [calculator, web_search]

llm = OllamaLLM(model="llama3.2:3b")


def run_agent(task):
    task_lower = task.lower()

    if any(symbol in task for symbol in ["+", "-", "*", "/"]):
        return calculator.invoke(task)

    return web_search.invoke(task)


tasks = [
    "25 * 4",
    "What is LangChain?",
    "Latest information about artificial intelligence"
]

for i, task in enumerate(tasks, 1):

    print(f"\n--- Agent Task {i} ---")
    print("Task:", task)

    result = run_agent(task)

    print("Result:", result)