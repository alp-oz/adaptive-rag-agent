# adaptive-rag-agent

A LangGraph agent that performs RAG on financial documents and routes based on statistically-guaranteed retrieval confidence. Designed as a portfolio project demonstrating production-ready agentic patterns.

## What it does

1. **Retrieves** relevant document chunks from a ChromaDB vector store
2. **Evaluates** retrieval quality using concentration inequalities (Hoeffding / Empirical Bernstein bounds) to compute a lower bound on true relevance with 80% statistical confidence
3. **Routes** based on that bound:
   - Confidence ≥ 0.5 → generate answer with Claude
   - Confidence < 0.5, rewrites remaining → rewrite query with Claude and retry
   - Exhausted rewrites → flag for human review
4. **Logs** per-query and session-level metrics (confidence, similarity, latency, answer/flag rates)

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

## Stack

| Component | Library |
|---|---|
| Agent graph | `langgraph` 1.2.4 |
| LLM | `langchain-anthropic` → Claude Sonnet 4.6 |
| Vector store | `chromadb` 1.5.9 |
| Embeddings | `sentence-transformers` (all-MiniLM-L6-v2) |
| PDF loading | `pypdf` 6.13.2 |
| Concentration bounds | custom (`evaluator.py`, ported from [cautious-rag](https://github.com/alp-oz/cautious-rag)) |
| Metrics | custom (`metrics.py`, ported from [rag-metrics](https://github.com/alp-oz/rag-metrics)) |

## Project structure

```
adaptive-rag-agent/
├── graph.py        # LangGraph StateGraph: state type, routing logic, compiled graph
├── nodes.py        # Node implementations: retrieve, evaluate, rewrite, answer, flag
├── evaluator.py    # Confidence scoring via Hoeffding / Empirical Bernstein bounds
├── metrics.py      # Per-query QueryRecord + SessionMetrics logger
├── retriever.py    # ChromaDB retriever: ingest PDF/TXT, cosine similarity search
├── main.py         # CLI entry point
├── data/           # Drop financial PDFs or TXT files here for ingestion
│   └── ecb_stress_test_2023.txt   # EBA 2023 EU-wide stress test summary
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
python main.py ingest data/ecb_stress_test_2023.txt
python main.py ingest data/annual_report.pdf
```

**Query the agent:**

```bash
python main.py query "What is the CET1 capital ratio under the adverse scenario?"
python main.py query "What were the main credit risk losses?" --show-docs
```

Example output:

```
Status:    answered
Rewrites:  0
Confidence:0.5055

Answer:
Based on the document excerpts, the CET1 ratio under the adverse scenario
(end-2025) is 10.4%, compared to a starting CET1 ratio of 15.0% (fully
loaded, end-2022), representing a depletion of 4.6 percentage points.
...

================================================
SESSION METRICS
================================================
Total queries:          1
Answer rate:            100.0%
Flag rate:              0.0%
Avg confidence (lb):    0.5055
Avg mean similarity:    0.6973
Avg rewrites/query:     0.00
Avg latency:            7190.8 ms
================================================
```

## How confidence scoring works

Instead of a raw similarity score, the agent applies a **concentration inequality lower bound** on the mean relevance of retrieved documents. This gives a statistically-guaranteed floor: with 80% confidence, the true mean relevance is at least this value.

- With low variance across retrieved scores → **Empirical Bernstein** gives a tighter bound
- With high variance or few documents → falls back to **Hoeffding**
- The agent picks whichever bound is tighter (higher lower bound)

This means the agent will refuse to answer (and rewrite the query instead) when retrieval is genuinely uncertain, not just when raw similarity looks low. The approach is ported from the [cautious-rag](https://github.com/alp-oz/cautious-rag) repository.

### Embedding model choice

The embedding model matters significantly for threshold calibration. `all-MiniLM-L6-v2` compresses financial text similarities into a narrow band (0.45–0.70), making the lower bound uninformative as a routing signal. This project uses `BAAI/bge-base-en-v1.5`, which produces well-separated scores on technical/financial text:

| Query type | Mean similarity | Lower bound (Bernstein, 80%) |
|---|---|---|
| In-domain (financial) | 0.73–0.80 | 0.52–0.61 |
| Out-of-domain | 0.42–0.47 | 0.26–0.31 |

The routing threshold of **0.45** sits in the centre of a ~0.2 gap between in- and out-of-domain queries.

### Threshold calibration

The threshold of 0.45 was set empirically by measuring lower bounds on a small set of in-domain and out-of-domain queries against this corpus. In production you would calibrate against labelled relevance judgements — a set of (query, document, relevant: yes/no) triples — choosing the threshold that maximises F1 on the routing decision. The concentration inequality guarantee holds regardless of threshold choice; the threshold determines the operating point on the precision/recall curve for deciding when to answer vs. rewrite.

## Configuration

| Parameter | Location | Default | Effect |
|---|---|---|---|
| Confidence threshold | `graph.py:CONFIDENCE_THRESHOLD` | 0.5 | Route to answer vs. rewrite |
| Max rewrites | `graph.py:MAX_REWRITES` | 3 | Rewrite attempts before flagging |
| Confidence level (δ) | `nodes.py:evaluate` | 0.80 | Statistical confidence of the bound |
| Top-k retrieval | `nodes.py:_TOP_K` | 5 | Documents retrieved per query |
| Embedding model | `retriever.py:_EMBED_MODEL` | all-MiniLM-L6-v2 | Sentence-transformers model |
| LLM model | `nodes.py:_ANSWER_MODEL` | claude-sonnet-4-6 | Anthropic model for answer/rewrite |
