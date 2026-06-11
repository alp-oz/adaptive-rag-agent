from __future__ import annotations

from typing import Any, Literal, TypedDict

from langgraph.graph import END, START, StateGraph

MAX_REWRITES = 3


class RAGState(TypedDict):
    query: str
    documents: list[str]
    confidence_score: float
    answer: str
    rewrite_count: int
    status: Literal["pending", "answered", "flagged"]
    # internal keys used between nodes (not exposed to callers)
    _similarity_scores: list[float]
    _confidence_result: Any


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------

CONFIDENCE_THRESHOLD = 0.5


def route_after_evaluate(
    state: RAGState,
) -> Literal["answer", "rewrite", "flag"]:
    if state["confidence_score"] >= CONFIDENCE_THRESHOLD:
        return "answer"
    if state["rewrite_count"] < MAX_REWRITES:
        return "rewrite"
    return "flag"


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------

def build_graph() -> StateGraph:
    from nodes import answer, evaluate, flag, retrieve, rewrite

    builder = StateGraph(RAGState)

    builder.add_node("retrieve", retrieve)
    builder.add_node("evaluate", evaluate)
    builder.add_node("rewrite", rewrite)
    builder.add_node("answer", answer)
    builder.add_node("flag", flag)

    builder.add_edge(START, "retrieve")
    builder.add_edge("retrieve", "evaluate")
    builder.add_conditional_edges(
        "evaluate",
        route_after_evaluate,
        {"answer": "answer", "rewrite": "rewrite", "flag": "flag"},
    )
    builder.add_edge("rewrite", "retrieve")
    builder.add_edge("answer", END)
    builder.add_edge("flag", END)

    return builder


graph = build_graph().compile()


if __name__ == "__main__":
    initial_state: RAGState = {
        "query": "What is the capital adequacy ratio?",
        "documents": [],
        "confidence_score": 0.0,
        "answer": "",
        "rewrite_count": 0,
        "status": "pending",
    }
    result = graph.invoke(initial_state)
    print(result)
