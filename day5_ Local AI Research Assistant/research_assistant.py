"""
Local AI Research Assistant.

Technology stack:
- LangGraph: workflow orchestration
- Ollama: local LLM inference
- MLflow: experiment tracking
"""

from typing import TypedDict

import mlflow
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph


class ResearchState(TypedDict):
    """State used by the LangGraph research workflow."""

    question: str
    answer: str


# Connect LangChain to the locally running Ollama model.
llm = ChatOllama(
    model="llama3.2:3b",
    temperature=0,
)


def validate_question(state: ResearchState) -> ResearchState:
    """
    Validate and clean the research question.

    Args:
        state: Current workflow state.

    Returns:
        Updated workflow state.

    Raises:
        ValueError: If the question is empty.
    """
    question = state["question"].strip()

    if not question:
        raise ValueError("Research question cannot be empty.")

    state["question"] = question

    return state


def generate_answer(state: ResearchState) -> ResearchState:
    """
    Generate an answer using the local Ollama LLM.

    Args:
        state: Current workflow state.

    Returns:
        State containing the generated answer.
    """
    prompt = f"""
You are a Local AI Research Assistant.

Research question:
{state["question"]}

Provide a clear and structured answer.

Use this format:

1. Introduction
2. Key Points
3. Explanation
4. Conclusion

Important:
- Be concise and factual.
- Do not claim that you searched the internet.
- Do not invent references or sources.
"""

    response = llm.invoke(prompt)

    state["answer"] = response.content

    return state


def build_workflow():
    """
    Build the LangGraph workflow.

    Workflow:

    START
      ↓
    Validate Question
      ↓
    Generate Answer
      ↓
    END
    """
    workflow = StateGraph(ResearchState)

    workflow.add_node("validate", validate_question)
    workflow.add_node("research", generate_answer)

    workflow.add_edge(START, "validate")
    workflow.add_edge("validate", "research")
    workflow.add_edge("research", END)

    return workflow.compile()


def run_research(question: str) -> str:
    """
    Run the complete research assistant.

    Args:
        question: User's research question.

    Returns:
        AI-generated research answer.
    """
    # Create an MLflow experiment for tracking.
    mlflow.set_experiment("local-ai-research-assistant")

    with mlflow.start_run():
        # Record useful experiment information.
        mlflow.log_param("model", "llama3.2:3b")
        mlflow.log_param("workflow", "LangGraph")

        # Build and execute the LangGraph workflow.
        graph = build_workflow()

        result = graph.invoke(
            {
                "question": question,
                "answer": "",
            }
        )

        # Track the generated answer length.
        mlflow.log_metric(
            "answer_length",
            len(result["answer"]),
        )

        return result["answer"]


def main() -> None:
    """Run the command-line research assistant."""

    print("=" * 50)
    print("       LOCAL AI RESEARCH ASSISTANT")
    print("=" * 50)

    question = input("\nEnter your research question: ")

    try:
        answer = run_research(question)

        print("\n===== RESEARCH ANSWER =====\n")
        print(answer)

    except ValueError as error:
        print(f"\nInput error: {error}")

    except Exception as error:
        print(f"\nApplication error: {error}")


if __name__ == "__main__":
    main()