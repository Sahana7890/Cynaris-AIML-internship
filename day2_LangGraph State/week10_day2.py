from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command


# -------------------------------------------------
# 1. Define State
# -------------------------------------------------

class AgentState(TypedDict):
    user_input: str
    category: str
    response: str
    human_feedback: str


# -------------------------------------------------
# 2. CLASSIFY NODE
# -------------------------------------------------

def classify(state: AgentState):
    text = state["user_input"].lower()

    if any(word in text for word in ["python", "java", "code", "programming"]):
        category = "technical"

    elif any(word in text for word in ["ticket", "incident", "laptop", "network"]):
        category = "support"

    elif any(word in text for word in ["hello", "hi", "hey"]):
        category = "general"

    else:
        category = "unknown"

    print(f"[CLASSIFY] Input: {state['user_input']}")
    print(f"[CLASSIFY] Category: {category}")

    return {
        "category": category
    }


# -------------------------------------------------
# 3. ROUTE NODE
# -------------------------------------------------

def route(state: AgentState):
    category = state["category"]

    print(f"[ROUTE] Routing to: {category}")

    return {}


# -------------------------------------------------
# 4. RESPOND NODE
# -------------------------------------------------

def respond(state: AgentState):
    category = state["category"]

    if category == "technical":
        response = "This is a technical question. Technical support response selected."

    elif category == "support":
        response = "This is an IT support issue. Support response selected."

    elif category == "general":
        response = "Hello! How can I help you?"

    else:
        response = "I need more information to understand your request."

    print(f"[RESPOND] {response}")

    return {
        "response": response
    }


# -------------------------------------------------
# 5. HUMAN-IN-THE-LOOP NODE
# -------------------------------------------------

def human_review(state: AgentState):

    print("\n--------------------------------")
    print("HUMAN REVIEW REQUIRED")
    print("--------------------------------")

    feedback = interrupt(
        {
            "message": "Please review the generated response.",
            "response": state["response"],
            "category": state["category"]
        }
    )

    return {
        "human_feedback": feedback
    }


# -------------------------------------------------
# 6. FINAL RESPONSE NODE
# -------------------------------------------------

def final_response(state: AgentState):

    feedback = state.get("human_feedback", "")

    if feedback:
        response = (
            f"Final response after human review: {feedback}"
        )
    else:
        response = state["response"]

    print(f"[FINAL] {response}")

    return {
        "response": response
    }


# -------------------------------------------------
# 7. Build Graph
# -------------------------------------------------

builder = StateGraph(AgentState)


builder.add_node("classify", classify)
builder.add_node("route", route)
builder.add_node("respond", respond)
builder.add_node("human_review", human_review)
builder.add_node("final_response", final_response)


# -------------------------------------------------
# 8. Conditional Routing
# -------------------------------------------------

def routing_decision(state: AgentState):

    category = state["category"]

    if category == "technical":
        return "technical"

    elif category == "support":
        return "support"

    elif category == "general":
        return "general"

    else:
        return "unknown"


builder.add_edge(START, "classify")

builder.add_conditional_edges(
    "classify",
    routing_decision,
    {
        "technical": "route",
        "support": "route",
        "general": "route",
        "unknown": "route"
    }
)

builder.add_edge("route", "respond")
builder.add_edge("respond", "human_review")
builder.add_edge("human_review", "final_response")
builder.add_edge("final_response", END)


# -------------------------------------------------
# 9. Add Checkpointer
# -------------------------------------------------

memory = MemorySaver()

graph = builder.compile(
    checkpointer=memory
)


# -------------------------------------------------
# 10. Test One Input with Human Interrupt
# -------------------------------------------------

if __name__ == "__main__":

    config = {
        "configurable": {
            "thread_id": "week10-day2"
        }
    }

    initial_state = {
        "user_input": "How do I fix my Python code?",
        "category": "",
        "response": "",
        "human_feedback": ""
    }

    print("\nStarting LangGraph...\n")

    result = graph.invoke(
        initial_state,
        config
    )

    print("\nGraph paused for human review.")

    print("\nEnter human feedback:")
    feedback = input("> ")

    result = graph.invoke(
        Command(resume=feedback),
        config
    )

    print("\nGraph completed.")
    print("Final Output:", result["response"])