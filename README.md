# adaptive-rag-agent

A LangGraph agent that performs RAG on documents and routes based on statistically-guaranteed retrieval confidence. Designed as a portfolio project demonstrating production-ready agentic patterns.

## Part of a series

This project is the third in a series building toward principled, measurable RAG systems:

- [cautious-rag](https://github.com/alp-oz/cautious-rag) — concentration inequality bounds for retrieval confidence
- [rag-metrics](https://github.com/alp-oz/rag-metrics) — intrinsic RAG evaluation metrics
- **adaptive-rag-agent** (this repo) — LangGraph agentic routing layer combining both

## What it does

1. **Retrieves** the top-K document chunks from a ChromaDB vector store using cosine similarity
2. **Evaluates** retrieval quality using the Empirical Bernstein inequality to compute a 95% confidence lower bound on the true mean similarity between the query and retrieved chunks
3. **Routes** based on that lower bound:
   - Lower bound ≥ 0.45 → generate answer with Claude
   - Lower bound < 0.45, rewrites remaining → rewrite query with Claude and retry
   - Exhausted rewrites → flag for human review
4. **Logs** per-query and session-level metrics (confidence, similarity, entropy, latency, answer/flag rates)

```
query
  │
  ▼
retrieve ──► evaluate ──► answer ──► END
                │
                ├─(low confidence, rewrites left)──► rewrite ──► retrieve
                │
                └─(low confidence, no rewrites left)──► flag ──► END
```

## Live demo

The Streamlit app lets you query built-in documents (Aristotle's Rhetoric, Poincaré's Science and Method) or upload your own PDF/TXT files. Sidebar controls let you tune TOP_K and confidence level interactively.

## Stack

| Component | Library |
|---|---|
| Agent graph | `langgraph` 1.2.4 |
| LLM | `langchain-anthropic` → Claude Sonnet 4.6 |
| Vector store | `chromadb` 1.5.9 |
| Embeddings | `sentence-transformers` (BAAI/bge-base-en-v1.5) |
| PDF loading | `pypdf` 6.13.2 |
| UI | `streamlit` |
| Concentration bounds | custom (`evaluator.py`, ported from [cautious-rag](https://github.com/alp-oz/cautious-rag)) |
| Metrics | custom (`metrics.py`, ported from [rag-metrics](https://github.com/alp-oz/rag-metrics)) |

## Project structure

```
adaptive-rag-agent/
├── graph.py        # LangGraph StateGraph: state type, routing logic, compiled graph
├── nodes.py        # Node implementations: retrieve, evaluate, rewrite, answer, flag
├── evaluator.py    # Confidence scoring via Empirical Bernstein bound + retrieval entropy
├── metrics.py      # Per-query QueryRecord + SessionMetrics logger
├── retriever.py    # ChromaDB retriever: ingest PDF/TXT, cosine similarity search
├── main.py         # CLI entry point
├── app.py          # Streamlit UI
├── data/
│   ├── aristotle_rhetoric_book3.txt     # Aristotle's Rhetoric, Book III
│   └── poincare_science_and_method.txt  # Poincaré's Science and Method (selected chapters)
├── requirements.txt
└── .env.example
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# Edit .env and add your Anthropic API key
```

## Usage

**Ingest a document** (PDF or TXT):

```bash
python main.py ingest data/aristotle_rhetoric_book3.txt
```

**Query the agent:**

```bash
python main.py query "What makes a metaphor good according to Aristotle?"
python main.py query "What is the cognitive process behind mathematical insight?"
```

Example output:

```
Status:    answered
Rewrites:  0
Confidence:0.5801

Answer:
The genesis of mathematical discovery is described by Poincaré as unfolding
in several phases: initial conscious effort, unconscious incubation where the
subliminal ego generates combinations, sudden illumination when the decisive
idea surfaces, and a final verification phase...

================================================
SESSION METRICS
================================================
Total queries:          1
Answer rate:            100.0%
Flag rate:              0.0%
Avg confidence (lb):    0.5801
Avg mean similarity:    0.6550
Avg rewrites/query:     0.00
Avg latency:            13327.2 ms
================================================
```

**Run the Streamlit app locally:**

```bash
streamlit run app.py
```

## How confidence scoring works

Instead of a raw similarity score, the agent computes a **95% confidence lower bound** on the mean cosine similarity between the query embedding and the retrieved chunk embeddings, using the Empirical Bernstein inequality:

```
lower_bound = mean_similarity - penalty
```

where the penalty grows with the variance across chunk scores and shrinks as more chunks are retrieved (proportional to 1/√n). This gives a conservative but rigorous floor: with 95% confidence, the true mean similarity is at least this value.

The agent answers only when even this pessimistic estimate clears the threshold of 0.45.

### Why Empirical Bernstein over Hoeffding?

Hoeffding uses only the range of the scores; Bernstein also uses the observed variance. When scores cluster tightly, the Bernstein penalty is much smaller, giving a tighter (higher) lower bound. The agent uses the adaptive strategy: pick whichever bound is tighter.

### Embedding model

`BAAI/bge-base-en-v1.5` produces well-separated similarity distributions on technical text:

| Query type | Mean similarity | Lower bound (Bernstein, 95%, n=20) |
|---|---|---|
| In-domain | 0.58–0.75 | 0.50–0.67 |
| Out-of-domain | 0.28–0.44 | 0.23–0.37 |

The routing threshold of **0.45** sits in the gap between the two distributions.

### Threshold calibration

The threshold of 0.45 was set empirically by measuring lower bounds on in-domain and out-of-domain queries against the built-in corpus. In production you would calibrate against labelled relevance judgements, choosing the threshold that maximises F1 on the routing decision. The concentration inequality guarantee holds regardless of threshold choice.

## Configuration

| Parameter | Location | Default | Effect |
|---|---|---|---|
| Confidence threshold | `graph.py:CONFIDENCE_THRESHOLD` | 0.45 | Route to answer vs. rewrite |
| Max rewrites | `graph.py:MAX_REWRITES` | 3 | Rewrite attempts before flagging |
| Confidence level | `nodes.py:_CONFIDENCE_LEVEL` | 0.95 | Statistical confidence of the bound |
| Top-k retrieval | `nodes.py:_TOP_K` | 20 | Chunks retrieved per query |
| Embedding model | `retriever.py:_EMBED_MODEL` | BAAI/bge-base-en-v1.5 | Sentence-transformers model |
| LLM model | `nodes.py:_ANSWER_MODEL` | claude-sonnet-4-6 | Anthropic model for answer/rewrite |

Both TOP_K and confidence level are also tunable at runtime via the Streamlit sidebar.

## Deployment

The app is deployable on Streamlit Community Cloud. Set the following in the Secrets panel:

```toml
ANTHROPIC_API_KEY = "sk-ant-..."
SAMPLE_DOMAIN = "public"   # or "finance" for a finance-specific deployment
```

See `.streamlit/secrets.toml.example` for the full template.
