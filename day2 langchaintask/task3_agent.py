from langchain_ollama import OllamaLLM
from langchain_core.tools import tool

# Create Ollama model
llm = OllamaLLM(
    model="llama3.2:3b"
)


# Tool 1: Calculator
@tool
def calculator(expression: str) -> str:
    """Calculate a mathematical expression."""

    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return str(result)

    except Exception:
        return "Invalid mathematical expression."


# Tool 2: Web search stub
@tool
def web_search(query: str) -> str:
    """Search the web. This is a demonstration stub and does not perform a real search."""

    return f"Web search result for '{query}': Python is a popular programming language used for AI, data science, automation, and web development."


# List tools
tools = [
    calculator,
    web_search
]


def run_agent(task):

    task_lower = task.lower()

    # Decide which tool to use
    if any(symbol in task_lower for symbol in ["+", "-", "*", "/", "calculate"]):

        expression = (
            task_lower
            .replace("calculate", "")
            .replace("what is", "")
            .replace("?", "")
            .strip()
        )

        result = calculator.invoke(expression)

        return f"Calculator result: {result}"

    elif any(word in task_lower for word in ["search", "web", "internet"]):

        result = web_search.invoke(task)

        return f"Web search result: {result}"

    else:

        prompt = f"""
Answer this question clearly:

{task}
"""

        return llm.invoke(prompt)


# Test 3 tasks
tasks = [
    "What is 25 * 8?",
    "Search the web for information about Python.",
    "What is 100 / 4?"
]


for i, task in enumerate(tasks, 1):

    print(f"\n--- Agent Task {i} ---")
    print("Task:", task)

    result = run_agent(task)

    print("Result:", result)