"""Entry point for the adaptive-rag-agent."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from graph import RAGState, build_graph
from nodes import get_logger, get_retriever


def ingest(paths: list[str]) -> None:
    r = get_retriever()
    for p in paths:
        n = r.ingest(p)
        print(f"Ingested {n} chunks from {Path(p).name}  (total: {r.doc_count})")


def run_query(query: str, show_docs: bool = False) -> None:
    get_logger().start_query()
    graph = build_graph().compile()
    initial: RAGState = {
        "query": query,
        "documents": [],
        "confidence_score": 0.0,
        "answer": "",
        "rewrite_count": 0,
        "status": "pending",
        "_similarity_scores": [],
        "_confidence_result": None,
    }
    result = graph.invoke(initial)
    print(f"\nStatus:    {result['status']}")
    print(f"Rewrites:  {result['rewrite_count']}")
    print(f"Confidence:{result['confidence_score']:.4f}")
    if show_docs:
        for i, doc in enumerate(result["documents"], 1):
            print(f"\n[Doc {i}]\n{doc[:300]}...")
    print(f"\nAnswer:\n{result['answer']}")
    get_logger().session.print_summary()


def main() -> None:
    parser = argparse.ArgumentParser(description="Adaptive RAG Agent")
    sub = parser.add_subparsers(dest="cmd", required=True)

    ingest_p = sub.add_parser("ingest", help="Load PDF(s) into the vector store")
    ingest_p.add_argument("files", nargs="+")

    query_p = sub.add_parser("query", help="Run a query against ingested documents")
    query_p.add_argument("query")
    query_p.add_argument("--show-docs", action="store_true")

    args = parser.parse_args()

    if args.cmd == "ingest":
        ingest(args.files)
    elif args.cmd == "query":
        run_query(args.query, show_docs=args.show_docs)


if __name__ == "__main__":
    main()
