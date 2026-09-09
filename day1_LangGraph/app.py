from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import MemorySaver


# --------------------------------------------------
# STATE
# --------------------------------------------------

class AgentState(TypedDict):
    user_input: str
    classification: str
    response: str
    human_approval: str


# --------------------------------------------------
# NODE 1: CLASSIFY
# --------------------------------------------------

def classify(state: AgentState):
    text = state["user_input"].lower()

    if any(word in text for word in ["error", "bug", "crash", "not working"]):
        category = "technical"

    elif any(word in text for word in ["price", "cost", "payment", "refund"]):
        category = "billing"

    elif any(word in text for word in ["hello", "hi", "hey"]):
        category = "general"

    else:
        category = "general"

    print(f"[CLASSIFY] {category}")

    return {
        "classification": category
    }


# --------------------------------------------------
# NODE 2: ROUTE
# --------------------------------------------------

def route(state: AgentState):
    category = state["classification"]

    print(f"[ROUTE] Sending request to: {category}")

    return {}


# --------------------------------------------------
# NODE 3: RESPOND
# --------------------------------------------------

def respond(state: AgentState):
    category = state["classification"]
    user_input = state["user_input"]

    if category == "technical":
        response = (
            f"Technical support: I will help troubleshoot this issue: "
            f"{user_input}"
        )

    elif category == "billing":
        response = (
            f"Billing support: I will help with your payment-related "
            f"request: {user_input}"
        )

    else:
        response = (
            f"General support: Thanks for your message. "
            f"How can I help you with: {user_input}?"
        )

    print(f"[RESPOND] {response}")

    return {
        "response": response
    }


# --------------------------------------------------
# HUMAN-IN-THE-LOOP NODE
# --------------------------------------------------

def human_review(state: AgentState):
    approval = interrupt(
        {
            "message": "Human approval required.",
            "draft_response": state["response"],
            "question": "Approve this response? Type yes or no."
        }
    )

    return {
        "human_approval": approval
    }


# --------------------------------------------------
# CONDITIONAL ROUTING
# --------------------------------------------------

def classification_router(state: AgentState):
    category = state["classification"]

    if category == "technical":
        return "technical"

    elif category == "billing":
        return "billing"

    else:
        return "general"


# --------------------------------------------------
# BUILD GRAPH
# --------------------------------------------------

builder = StateGraph(AgentState)

builder.add_node("classify", classify)
builder.add_node("route", route)
builder.add_node("respond", respond)
builder.add_node("human_review", human_review)

builder.add_edge(START, "classify")

# Conditional edge based on classification
builder.add_conditional_edges(
    "classify",
    classification_router,
    {
        "technical": "route",
        "billing": "route",
        "general": "route",
    }
)

builder.add_edge("route", "respond")
builder.add_edge("respond", "human_review")
builder.add_edge("human_review", END)

# Memory/checkpointer enables pause/resume
memory = MemorySaver()

graph = builder.compile(checkpointer=memory)


# --------------------------------------------------
# TEST 5 INPUTS
# --------------------------------------------------

test_inputs = [
    "My laptop is not working",
    "I need a refund for my payment",
    "Hello, how are you?",
    "The application crashes when I open it",
    "What is the price of the service?"
]


def run_test(user_input, thread_id):
    print("\n" + "=" * 60)
    print(f"INPUT: {user_input}")

    config = {
        "configurable": {
            "thread_id": thread_id
        }
    }

    result = graph.invoke(
        {
            "user_input": user_input,
            "classification": "",
            "response": "",
            "human_approval": ""
        },
        config
    )

    print("\nGraph paused for human approval.")
    print("Draft:", result.get("response"))

    # Simulate human approval
    human_input = input("Human input (yes/no): ")

    result = graph.invoke(
        Command(resume=human_input),
        config
    )

    print("FINAL RESULT:")
    print(result)


# --------------------------------------------------
# MAIN
# --------------------------------------------------

if __name__ == "__main__":

    for index, user_input in enumerate(test_inputs, start=1):
        run_test(
            user_input,
            f"week10-day1-test-{index}"
        )