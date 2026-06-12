"""
Run-time metric logging for the adaptive-rag-agent graph.

Tracks per-query events and accumulates session-level stats, following the
structure from rag-metrics/cautious_rag/utils/metrics.py (EvaluationResult /
MetricsCalculator) but adapted for LangGraph node callbacks.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Literal

Route = Literal["answer", "rewrite", "flag"]


@dataclass
class QueryRecord:
    query: str
    route: Route
    confidence_score: float
    mean_similarity: float
    bound_used: str
    n_docs: int
    rewrite_count: int
    latency_ms: float
    entropy: float = 0.0
    timestamp: float = field(default_factory=time.time)


@dataclass
class SessionMetrics:
    records: list[QueryRecord] = field(default_factory=list)

    # Counters
    total: int = 0
    answered: int = 0
    rewritten: int = 0
    flagged: int = 0

    # Accumulators (divided on read)
    _sum_confidence: float = 0.0
    _sum_mean_sim: float = 0.0
    _sum_rewrites: float = 0.0
    _sum_latency_ms: float = 0.0

    @property
    def answer_rate(self) -> float:
        return self.answered / self.total if self.total else 0.0

    @property
    def flag_rate(self) -> float:
        return self.flagged / self.total if self.total else 0.0

    @property
    def avg_confidence(self) -> float:
        return self._sum_confidence / self.total if self.total else 0.0

    @property
    def avg_mean_similarity(self) -> float:
        return self._sum_mean_sim / self.total if self.total else 0.0

    @property
    def avg_rewrites(self) -> float:
        return self._sum_rewrites / self.total if self.total else 0.0

    @property
    def avg_latency_ms(self) -> float:
        return self._sum_latency_ms / self.total if self.total else 0.0

    def summary(self) -> dict[str, Any]:
        return {
            "total_queries": self.total,
            "answer_rate": round(self.answer_rate, 4),
            "flag_rate": round(self.flag_rate, 4),
            "avg_confidence": round(self.avg_confidence, 4),
            "avg_mean_similarity": round(self.avg_mean_similarity, 4),
            "avg_rewrites_per_query": round(self.avg_rewrites, 4),
            "avg_latency_ms": round(self.avg_latency_ms, 2),
        }

    def print_summary(self) -> None:
        s = self.summary()
        print("=" * 48)
        print("SESSION METRICS")
        print("=" * 48)
        print(f"Total queries:          {s['total_queries']}")
        print(f"Answer rate:            {s['answer_rate']*100:.1f}%")
        print(f"Flag rate:              {s['flag_rate']*100:.1f}%")
        print(f"Avg confidence (lb):    {s['avg_confidence']:.4f}")
        print(f"Avg mean similarity:    {s['avg_mean_similarity']:.4f}")
        print(f"Avg rewrites/query:     {s['avg_rewrites_per_query']:.2f}")
        print(f"Avg latency:            {s['avg_latency_ms']:.1f} ms")
        print("=" * 48)


class MetricsLogger:
    """
    Thin stateful logger passed into graph nodes.

    Usage:
        logger = MetricsLogger()
        logger.log(query, route, confidence_result, rewrite_count, latency_ms)
        logger.session.print_summary()
    """

    def __init__(self) -> None:
        self.session = SessionMetrics()
        self._query_start: float = 0.0

    def start_query(self) -> None:
        self._query_start = time.monotonic()

    def log(
        self,
        query: str,
        route: Route,
        confidence_score: float,
        mean_similarity: float,
        bound_used: str,
        n_docs: int,
        rewrite_count: int,
        entropy: float = 0.0,
        latency_ms: float | None = None,
    ) -> QueryRecord:
        if latency_ms is None:
            latency_ms = (time.monotonic() - self._query_start) * 1000

        record = QueryRecord(
            query=query,
            route=route,
            confidence_score=confidence_score,
            mean_similarity=mean_similarity,
            bound_used=bound_used,
            n_docs=n_docs,
            rewrite_count=rewrite_count,
            entropy=entropy,
            latency_ms=latency_ms,
        )

        s = self.session
        s.records.append(record)
        s.total += 1
        if route == "answer":
            s.answered += 1
        elif route == "rewrite":
            s.rewritten += 1
        else:
            s.flagged += 1

        s._sum_confidence += confidence_score
        s._sum_mean_sim += mean_similarity
        s._sum_rewrites += rewrite_count
        s._sum_latency_ms += latency_ms

        return record
