"""
LangGraph node implementations for adaptive-rag-agent.

Each function matches the stub signature in graph.py and is wired into the
StateGraph. The LLM call (answer node) uses claude-sonnet-4-6 via
langchain-anthropic.
"""
from __future__ import annotations

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage

from evaluator import ConfidenceResult, compute_confidence
from graph import RAGState
from metrics import MetricsLogger
from retriever import Retriever

_ANSWER_MODEL = "claude-sonnet-4-6"
_TOP_K = 5

# Module-level singletons — created once, reused across invocations.
_retriever: Retriever | None = None
_llm: ChatAnthropic | None = None
_logger: MetricsLogger = MetricsLogger()


def get_retriever() -> Retriever:
    global _retriever
    if _retriever is None:
        _retriever = Retriever()
    return _retriever


def get_llm() -> ChatAnthropic:
    global _llm
    if _llm is None:
        _llm = ChatAnthropic(model=_ANSWER_MODEL)
    return _llm


def get_logger() -> MetricsLogger:
    return _logger


# ---------------------------------------------------------------------------
# Node: retrieve
# ---------------------------------------------------------------------------

def retrieve(state: RAGState) -> RAGState:
    """Query ChromaDB for documents relevant to state['query']."""
    result = get_retriever().query(state["query"], k=_TOP_K)
    return {
        **state,
        "documents": result.documents,
        # stash similarity scores in a side-channel key for the evaluate node
        "_similarity_scores": result.similarity_scores,
    }


# ---------------------------------------------------------------------------
# Node: evaluate
# ---------------------------------------------------------------------------

def evaluate(state: RAGState) -> RAGState:
    """Score retrieval confidence via concentration inequalities."""
    scores: list[float] = state.get("_similarity_scores", [])
    result: ConfidenceResult = compute_confidence(scores, confidence_level=0.80, bound="adaptive")
    return {
        **state,
        "confidence_score": result.score,
        "_confidence_result": result,
    }


# ---------------------------------------------------------------------------
# Node: rewrite
# ---------------------------------------------------------------------------

_REWRITE_SYSTEM = (
    "You are a query optimisation assistant. "
    "Rewrite the user's question to improve retrieval from a financial document database. "
    "Return only the rewritten question, no explanation."
)


def rewrite(state: RAGState) -> RAGState:
    """Use the LLM to rephrase the query for better retrieval."""
    response = get_llm().invoke(
        [
            SystemMessage(content=_REWRITE_SYSTEM),
            HumanMessage(content=state["query"]),
        ]
    )
    new_query = response.content.strip()
    return {
        **state,
        "query": new_query,
        "rewrite_count": state["rewrite_count"] + 1,
        "documents": [],
        "confidence_score": 0.0,
    }


# ---------------------------------------------------------------------------
# Node: answer
# ---------------------------------------------------------------------------

_ANSWER_SYSTEM = (
    "You are a financial analyst assistant. "
    "Answer the user's question using only the provided document excerpts. "
    "Be concise and cite page numbers where relevant. "
    "If the excerpts do not contain enough information, say so explicitly."
)


def answer(state: RAGState) -> RAGState:
    """Generate an answer from retrieved documents."""
    context = "\n\n---\n\n".join(state["documents"])
    response = get_llm().invoke(
        [
            SystemMessage(content=_ANSWER_SYSTEM),
            HumanMessage(
                content=f"Documents:\n{context}\n\nQuestion: {state['query']}"
            ),
        ]
    )
    cr: ConfidenceResult | None = state.get("_confidence_result")
    _logger.log(
        query=state["query"],
        route="answer",
        confidence_score=state["confidence_score"],
        mean_similarity=cr.mean_similarity if cr else 0.0,
        bound_used=cr.bound_used if cr else "none",
        n_docs=len(state["documents"]),
        rewrite_count=state["rewrite_count"],
    )
    return {
        **state,
        "answer": response.content.strip(),
        "status": "answered",
    }


# ---------------------------------------------------------------------------
# Node: flag
# ---------------------------------------------------------------------------

def flag(state: RAGState) -> RAGState:
    """Mark the query as unresolvable; log and surface for human review."""
    cr: ConfidenceResult | None = state.get("_confidence_result")
    _logger.log(
        query=state["query"],
        route="flag",
        confidence_score=state["confidence_score"],
        mean_similarity=cr.mean_similarity if cr else 0.0,
        bound_used=cr.bound_used if cr else "none",
        n_docs=len(state["documents"]),
        rewrite_count=state["rewrite_count"],
    )
    return {
        **state,
        "answer": (
            f"Could not retrieve sufficient evidence after "
            f"{state['rewrite_count']} rewrite(s). "
            f"Confidence lower bound: {state['confidence_score']:.3f}. "
            "Query flagged for human review."
        ),
        "status": "flagged",
    }
