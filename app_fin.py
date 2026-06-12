from __future__ import annotations

try:
    __import__("pysqlite3")
    import sys
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    pass

import math
import os
import shutil
import tempfile
import time
from pathlib import Path

import streamlit as st

st.set_page_config(
    page_title="Adaptive RAG Agent",
    page_icon="📄",
    layout="centered",
)

if not os.environ.get("ANTHROPIC_API_KEY"):
    try:
        os.environ["ANTHROPIC_API_KEY"] = st.secrets["ANTHROPIC_API_KEY"]
    except Exception:
        env_file = Path(__file__).parent / ".env"
        if env_file.exists():
            for line in env_file.read_text().splitlines():
                if line.startswith("ANTHROPIC_API_KEY="):
                    os.environ["ANTHROPIC_API_KEY"] = line.split("=", 1)[1].strip()
                    break

EMBED_MODEL = "BAAI/bge-base-en-v1.5"
MODEL_STAMP = Path("chroma_db/.embed_model")
FINANCE_DATA_FILES = [
    "data/ecb_stress_test_2023.txt",
    "data/basel3_capital_requirements.txt",
    "data/ecb_srep_methodology.txt",
    "data/swedish_banking_capital_2023.txt",
]
ARISTOTLE_DATA_FILES = [
    "data/aristotle_rhetoric_book3.txt",
]
POINCARE_DATA_FILES = [
    "data/poincare_science_and_method.txt",
]
PUBLIC_DATA_FILES = ARISTOTLE_DATA_FILES + POINCARE_DATA_FILES
MIN_RECOMMENDED_CHUNKS = 20


@st.cache_resource(show_spinner="Loading embedding model and vector store...")
def get_default_retriever(domain: str = "finance"):
    from retriever import Retriever

    persist_dir = f"./chroma_db_{domain}"
    stamp = Path(f"{persist_dir}/.embed_model")

    if stamp.exists() and stamp.read_text().strip() != EMBED_MODEL:
        shutil.rmtree(persist_dir, ignore_errors=True)

    r = Retriever(persist_dir=persist_dir, embed_model=EMBED_MODEL)

    if r.doc_count == 0:
        if domain == "finance":
            files = FINANCE_DATA_FILES
        elif domain == "aristotle":
            files = ARISTOTLE_DATA_FILES
        elif domain == "poincare":
            files = POINCARE_DATA_FILES
        else:
            files = PUBLIC_DATA_FILES
        for f in files:
            if Path(f).exists():
                r.ingest_txt(f)
        stamp.parent.mkdir(exist_ok=True)
        stamp.write_text(EMBED_MODEL)

    return r


@st.cache_resource(show_spinner="Loading language model...")
def get_graph():
    from graph import build_graph
    return build_graph().compile()


with st.sidebar:
    st.header("Retrieval settings")
    top_k = st.slider(
        "Chunks retrieved (TOP_K)",
        min_value=5,
        max_value=50,
        value=20,
        step=5,
        help=(
            "How many document chunks to retrieve per query. "
            "More chunks tighten the confidence bound (penalty shrinks as 1/sqrt(n)) "
            "but increase latency and context length."
        ),
    )
    confidence_level = st.select_slider(
        "Confidence level",
        options=[0.80, 0.90, 0.95, 0.99],
        value=0.95,
        format_func=lambda x: f"{int(x * 100)}%",
        help=(
            "The confidence level for the Empirical Bernstein lower bound. "
            "95% means: in repeated sampling, at least 95% of such bounds will lie "
            "below the true mean similarity. Higher confidence widens the penalty, "
            "lowering the bound."
        ),
    )
    st.divider()
    st.caption(
        f"With n={top_k} chunks and {int(confidence_level * 100)}% confidence, "
        "the Bernstein penalty is approximately "
        f"{(2 * 0.04 * __import__('math').log(2 / (1 - confidence_level)) / top_k) ** 0.5:.3f} "
        "(assuming variance 0.04)."
    )

st.title("📄 Adaptive RAG Agent")
st.caption(
    "Document Q&A with statistically-guaranteed retrieval confidence. "
    "Built with [LangGraph](https://github.com/langchain-ai/langgraph), "
    "[cautious-rag](https://github.com/alp-oz/cautious-rag) and "
    "[rag-metrics](https://github.com/alp-oz/rag-metrics)."
)

st.markdown(
    f"The agent retrieves the top {top_k} chunks from your documents, then computes a "
    f"{int(confidence_level * 100)}% confidence lower bound on the mean cosine similarity "
    "using the Empirical Bernstein inequality. "
    "If that lower bound clears the threshold the query is answered with Claude. "
    "Otherwise the query is rewritten and retried up to three times, "
    "then flagged for human review if retrieval remains insufficient."
)

st.divider()

SAMPLE_DOMAIN = "finance"

if "query_input" not in st.session_state:
    st.session_state.query_input = ""
if "custom_retriever" not in st.session_state:
    st.session_state.custom_retriever = None
if "custom_doc_count" not in st.session_state:
    st.session_state.custom_doc_count = 0


def _upload_section() -> tuple:
    """Render file upload UI. Returns (retriever | None, show_examples: bool)."""
    st.markdown("**Upload one or more PDF or TXT files.**")
    st.info(
        "For the confidence bound to be meaningful your documents should together "
        "produce at least 20 text chunks, roughly 2000 words or 4 pages. "
        "With fewer chunks the statistical penalty grows large and most queries will "
        "be flagged even when the content is actually relevant."
    )
    uploaded_files = st.file_uploader(
        "Drop files here",
        type=["pdf", "txt"],
        accept_multiple_files=True,
    )
    if not uploaded_files:
        return None, False

    file_key = "_".join(sorted(f.name for f in uploaded_files))
    if file_key != st.session_state.get("uploaded_file_key"):
        from retriever import Retriever
        with st.spinner("Ingesting documents..."):
            tmp_dir = tempfile.mkdtemp()
            r = Retriever(persist_dir=os.path.join(tmp_dir, "chroma"), embed_model=EMBED_MODEL)
            total_chunks = 0
            for uf in uploaded_files:
                suffix = Path(uf.name).suffix.lower()
                tmp_path = os.path.join(tmp_dir, uf.name)
                with open(tmp_path, "wb") as fh:
                    fh.write(uf.read())
                total_chunks += r.ingest_pdf(tmp_path) if suffix == ".pdf" else r.ingest_txt(tmp_path)
        st.session_state.custom_retriever = r
        st.session_state.custom_doc_count = total_chunks
        st.session_state.uploaded_file_key = file_key

    doc_count = st.session_state.custom_doc_count
    if doc_count < MIN_RECOMMENDED_CHUNKS:
        st.warning(
            f"Only {doc_count} chunks ingested. "
            f"At least {MIN_RECOMMENDED_CHUNKS} are recommended for reliable scoring."
        )
    else:
        st.success(f"{doc_count} chunks ingested across {len(uploaded_files)} file(s). Ready.")

    return st.session_state.custom_retriever, False


SOURCE_OPTIONS = {
    "finance": "Built-in: Banking & Finance (EBA stress test, Basel III, ECB SREP)",
    "aristotle": "Aristotle's Rhetoric — Style, metaphor and persuasion",
    "poincare": "Poincaré's Science and Method — Mathematical discovery and intuition",
    "upload": "Upload your own documents",
}

DOMAIN_EXAMPLES = {
    "finance": [
        "What is the CET1 capital ratio under the adverse scenario?",
        "How much did credit losses accumulate across EU banks in the stress test?",
        "What happens to the leverage ratio under adverse conditions?",
        "Was the ECB ever elected as the central bank of Sweden?",
    ],
    "aristotle": [
        "What makes a metaphor good according to Aristotle?",
        "What is an example of a good metaphor according to Aristotle?",
        "How does word choice affect the believability of an argument?",
        "Was Aristotle ever elected to public office in Athens?",
    ],
    "poincare": [
        "What is the cognitive process behind mathematical insight?",
        "What does Poincaré say about the selection of facts in science?",
        "What role does elegance play in mathematical discovery?",
        "What was Poincaré's opinion on the Treaty of Versailles as prime minister?",
    ],
}

if SAMPLE_DOMAIN == "finance":
    available = ["finance", "upload"]
else:
    available = ["aristotle", "poincare", "upload"]

source = st.radio(
    "Document source:",
    available,
    format_func=lambda x: SOURCE_OPTIONS[x],
)

if source == "upload":
    active_retriever, show_examples = _upload_section()
    if active_retriever is None:
        st.caption("No files uploaded yet.")
        active_retriever = None
    show_examples = False
else:
    active_retriever = get_default_retriever(source)
    show_examples = True

st.divider()

EXAMPLES = DOMAIN_EXAMPLES.get(source, [])

if show_examples:
    st.markdown("**Example questions for the built-in documents:**")
    cols = st.columns(2)
    for i, example in enumerate(EXAMPLES):
        if cols[i % 2].button(example, use_container_width=True):
            st.session_state.query_input = example
            st.rerun()
    st.divider()

query = st.text_input(
    "Your question:",
    value=st.session_state.query_input,
    placeholder="Ask anything about your documents...",
    key="query_input",
    disabled=active_retriever is None,
)

run = st.button("Ask", type="primary", disabled=not query.strip() or active_retriever is None)

if run and query.strip():
    graph = get_graph()

    from graph import RAGState

    import nodes
    nodes._retriever = active_retriever
    nodes._TOP_K = top_k
    nodes._CONFIDENCE_LEVEL = confidence_level

    initial: RAGState = {
        "query": query.strip(),
        "documents": [],
        "confidence_score": 0.0,
        "answer": "",
        "rewrite_count": 0,
        "status": "pending",
        "_similarity_scores": [],
        "_confidence_result": None,
        "_rewrite_history": [query.strip()],
        "_confidence_history": [],
    }

    with st.spinner("Running agent..."):
        t0 = time.monotonic()
        result = graph.invoke(initial)
        latency = (time.monotonic() - t0) * 1000

    status = result["status"]
    rewrites = result["rewrite_count"]
    confidence = result["confidence_score"]
    cr = result.get("_confidence_result")
    entropy = cr.entropy if cr else 0.0
    mean_sim = cr.mean_similarity if cr else 0.0

    if status == "answered":
        st.success(f"Answered ({rewrites} rewrite{'s' if rewrites != 1 else ''})")
    elif status == "flagged":
        st.error(
            f"Flagged after {rewrites} rewrite{'s' if rewrites != 1 else ''}. "
            "The question appears to be outside the scope of the loaded documents."
        )

    st.markdown("### Answer")
    st.markdown(result["answer"])

    rewrite_history = result.get("_rewrite_history", [])
    if len(rewrite_history) > 1:
        with st.expander(f"How the query was rewritten ({len(rewrite_history) - 1} attempt{'s' if len(rewrite_history) - 1 != 1 else ''})"):
            for i, q in enumerate(rewrite_history):
                label = "Your question" if i == 0 else f"Rewrite {i}"
                st.markdown(f"**{label}:** {q}")

    confidence_history = result.get("_confidence_history", [])
    if len(confidence_history) > 1:
        import pandas as pd
        with st.expander("Did rewriting improve retrieval?"):
            st.caption(
                f"Each rewrite fetches a different set of chunks from your documents. "
                f"This chart shows the {int(confidence_level * 100)}% confidence lower bound "
                "on mean similarity after each attempt. "
                "The agent answers once that lower bound crosses 0.45."
            )
            labels = ["Original"] + [f"Rewrite {i}" for i in range(1, len(confidence_history))]
            df = pd.DataFrame({
                "Query version": labels,
                "P(chunks are relevant)": confidence_history,
                "Answer threshold (0.45)": [0.45] * len(confidence_history),
            })
            st.line_chart(df.set_index("Query version"), color=["#2196F3", "#FF5722"])

    st.divider()
    st.markdown("### Retrieval diagnostics")

    m1, m2, m3, m4 = st.columns(4)
    n_docs = cr.n_docs if cr else 5
    m1.metric(
        f"Mean similarity lower bound ({int(confidence_level * 100)}% CI)",
        f"{confidence:.3f}",
        help=(
            f"With {int(confidence_level * 100)}% confidence, the true mean cosine similarity "
            f"between your query and the {top_k} retrieved chunks is at least this value. "
            "Computed via the Empirical Bernstein inequality: mean similarity minus a penalty "
            "that grows with the variance across chunk scores and shrinks as more chunks are retrieved. "
            "The agent answers if this lower bound exceeds 0.45."
        ),
    )
    m2.metric(
        "Mean chunk similarity",
        f"{mean_sim:.3f}",
        help=(
            "Average cosine similarity between your query and the retrieved chunks. "
            "This is the raw score before the statistical penalty is applied."
        ),
    )
    m3.metric(
        "Retrieval ambiguity",
        f"{entropy:.3f}",
        help=(
            "How evenly the similarity scores are spread across the retrieved chunks. "
            "Low means one chunk clearly dominates, which is focused retrieval. "
            f"High means all chunks scored similarly, which signals ambiguity. "
            f"Maximum possible for {n_docs} chunks is {math.log(n_docs):.2f}."
        ),
    )
    m4.metric("Latency", f"{latency:.0f} ms")

    bound_label = cr.bound_used if cr else "n/a"
    st.caption(
        f"Inequality: {bound_label} "
        f"| Chunks retrieved: {n_docs} "
        f"| Confidence level: {int(confidence_level * 100)}% "
        f"| Rewrites: {rewrites} "
        f"| Answer threshold: 0.45 (calibrated empirically on this corpus, see README)"
    )

    if result["documents"]:
        with st.expander("View retrieved document chunks"):
            sims = result.get("_similarity_scores", [])
            for i, (doc, sim) in enumerate(zip(result["documents"], sims), 1):
                st.markdown(f"**Chunk {i}** (similarity {sim:.3f})")
                st.text(doc[:400] + ("..." if len(doc) > 400 else ""))
                if i < len(result["documents"]):
                    st.divider()

st.divider()
default_docs = (
    "EBA 2023 stress test, Basel III framework, ECB SREP methodology, Swedish banking sector report"
    if SAMPLE_DOMAIN == "finance"
    else "Aristotle's Rhetoric Book III, Poincaré's Science and Method (selected chapters)"
)
st.caption(
    f"Default documents: {default_docs}. "
    "Model: claude-sonnet-4-6. "
    "Embeddings: BAAI/bge-base-en-v1.5. "
    "Source: [github.com/alp-oz/adaptive-rag-agent](https://github.com/alp-oz/adaptive-rag-agent)."
)
