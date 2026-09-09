from typing import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command


# -----------------------------------------
# STATE
# -----------------------------------------

class AgentState(TypedDict):
    user_input: str
    category: str
    response: str
    human_feedback: str


# -----------------------------------------
# NODE 1: CLASSIFY
# -----------------------------------------

def classify(state: AgentState):

    user_input = state["user_input"].lower()

    if any(word in user_input for word in
           ["python", "java", "code", "programming"]):
        category = "technical"

    elif any(word in user_input for word in
             ["laptop", "network", "ticket", "incident"]):
        category = "support"

    elif any(word in user_input for word in
             ["hello", "hi", "hey"]):
        category = "general"

    else:
        category = "unknown"

    print("\n[CLASSIFY]")
    print("Input:", state["user_input"])
    print("Category:", category)

    return {
        "category": category
    }


# -----------------------------------------
# NODE 2: ROUTE
# -----------------------------------------

def route(state: AgentState):

    print("\n[ROUTE]")
    print("Routing category:", state["category"])

    return {}


# -----------------------------------------
# NODE 3: RESPOND
# -----------------------------------------

def respond(state: AgentState):

    category = state["category"]

    if category == "technical":
        response = "Technical response: This is a programming-related question."

    elif category == "support":
        response = "Support response: This is an IT support issue."

    elif category == "general":
        response = "General response: Hello! How can I help you?"

    else:
        response = "Unknown response: More information is required."

    print("\n[RESPOND]")
    print(response)

    return {
        "response": response
    }


# -----------------------------------------
# HUMAN-IN-THE-LOOP
# -----------------------------------------

def human_review(state: AgentState):

    print("\n--------------------------------")
    print("HUMAN REVIEW")
    print("--------------------------------")

    print("Generated response:")
    print(state["response"])

    feedback = interrupt(
        {
            "message": "Human review required",
            "response": state["response"]
        }
    )

    return {
        "human_feedback": feedback
    }


# -----------------------------------------
# FINAL RESPONSE
# -----------------------------------------

def final_response(state: AgentState):

    feedback = state.get("human_feedback", "")

    if feedback:
        final = (
            state["response"]
            + "\nHuman feedback: "
            + str(feedback)
        )
    else:
        final = state["response"]

    print("\n[FINAL RESPONSE]")
    print(final)

    return {
        "response": final
    }


# -----------------------------------------
# CONDITIONAL ROUTING
# -----------------------------------------

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


# -----------------------------------------
# BUILD GRAPH
# -----------------------------------------

builder = StateGraph(AgentState)

builder.add_node("classify", classify)
builder.add_node("route", route)
builder.add_node("respond", respond)
builder.add_node("human_review", human_review)
builder.add_node("final_response", final_response)


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


# -----------------------------------------
# CHECKPOINT
# -----------------------------------------

memory = MemorySaver()

graph = builder.compile(
    checkpointer=memory
)


# -----------------------------------------
# TEST
# -----------------------------------------

if __name__ == "__main__":

    print("\n===================================")
    print("WEEK 10 - DAY 3 LANGGRAPH")
    print("===================================")

    config = {
        "configurable": {
            "thread_id": "week10-day3"
        }
    }

    user_input = input(
        "\nEnter your question: "
    )

    initial_state = {
        "user_input": user_input,
        "category": "",
        "response": "",
        "human_feedback": ""
    }

    result = graph.invoke(
        initial_state,
        config
    )

    print("\nGraph paused for human review.")

    feedback = input(
        "\nEnter human feedback: "
    )

    result = graph.invoke(
        Command(resume=feedback),
        config
    )

    print("\n===================================")
    print("GRAPH COMPLETED")
    print("===================================")