"""
ChromaDB-backed retriever with sentence-transformers embeddings.

Handles PDF ingestion (ported from rag-metrics/pipeline/{ingestion,chunking})
and similarity search returning (documents, cosine_similarity_scores).
"""
from __future__ import annotations

import re
import uuid
from pathlib import Path
from typing import NamedTuple

import chromadb
import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

_EMBED_MODEL = "all-MiniLM-L6-v2"
_CHUNK_SIZE = 500
_CHUNK_OVERLAP = 100


class RetrievalResult(NamedTuple):
    documents: list[str]
    similarity_scores: list[float]   # cosine similarity in [0, 1]
    metadatas: list[dict]


class Retriever:
    """
    Wraps a ChromaDB collection with a sentence-transformers embedding function.

    Usage:
        r = Retriever()
        r.ingest_pdf("data/annual_report.pdf")
        result = r.query("What is the capital adequacy ratio?", k=5)
    """

    def __init__(
        self,
        collection_name: str = "financial_docs",
        persist_dir: str = "./chroma_db",
        embed_model: str = _EMBED_MODEL,
    ) -> None:
        self._model = SentenceTransformer(embed_model)
        self._client = chromadb.PersistentClient(path=persist_dir)
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def ingest_txt(self, path: str | Path) -> int:
        """Load, chunk, embed, and store a plain-text file."""
        path = Path(path)
        text = path.read_text(encoding="utf-8")
        pages = [{"text": text, "metadata": {"source": path.name, "page": 1}}]
        chunks = _chunk_documents(pages, _CHUNK_SIZE, _CHUNK_OVERLAP)

        texts = [c["text"] for c in chunks]
        metadatas = [c["metadata"] for c in chunks]
        ids = [str(uuid.uuid4()) for _ in chunks]
        embeddings = self._model.encode(texts, show_progress_bar=False).tolist()

        self._collection.add(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
        )
        return len(chunks)

    def ingest(self, path: str | Path) -> int:
        """Ingest a file by extension (.pdf or .txt)."""
        path = Path(path)
        if path.suffix.lower() == ".pdf":
            return self.ingest_pdf(path)
        return self.ingest_txt(path)

    def ingest_pdf(self, path: str | Path) -> int:
        """Load, chunk, embed, and store a PDF. Returns number of chunks added."""
        path = Path(path)
        pages = _load_pdf(path)
        chunks = _chunk_documents(pages, _CHUNK_SIZE, _CHUNK_OVERLAP)

        texts = [c["text"] for c in chunks]
        metadatas = [c["metadata"] for c in chunks]
        ids = [str(uuid.uuid4()) for _ in chunks]
        embeddings = self._model.encode(texts, show_progress_bar=False).tolist()

        self._collection.add(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
        )
        return len(chunks)

    def ingest_texts(self, texts: list[str], metadatas: list[dict] | None = None) -> int:
        """Embed and store raw text chunks directly."""
        if metadatas is None:
            metadatas = [{}] * len(texts)
        ids = [str(uuid.uuid4()) for _ in texts]
        embeddings = self._model.encode(texts, show_progress_bar=False).tolist()
        self._collection.add(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
        )
        return len(texts)

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def query(self, query: str, k: int = 5) -> RetrievalResult:
        """
        Retrieve top-k chunks for a query.

        ChromaDB returns L2 distances when hnsw:space="cosine" — these are
        actually 1 - cosine_similarity, so we convert: similarity = 1 - distance.
        """
        n_results = min(k, self._collection.count())
        if n_results == 0:
            return RetrievalResult([], [], [])

        query_embedding = self._model.encode([query], show_progress_bar=False).tolist()
        results = self._collection.query(
            query_embeddings=query_embedding,
            n_results=n_results,
            include=["documents", "distances", "metadatas"],
        )

        documents = results["documents"][0]
        distances = results["distances"][0]
        metadatas = results["metadatas"][0]

        # Convert cosine distance → cosine similarity, clip to [0, 1]
        similarities = [float(np.clip(1.0 - d, 0.0, 1.0)) for d in distances]

        return RetrievalResult(
            documents=documents,
            similarity_scores=similarities,
            metadatas=metadatas,
        )

    @property
    def doc_count(self) -> int:
        return self._collection.count()


# ------------------------------------------------------------------
# PDF helpers (ported from rag-metrics/pipeline/{ingestion,chunking})
# ------------------------------------------------------------------

def _load_pdf(path: Path) -> list[dict]:
    reader = PdfReader(str(path))
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if not text:
            continue
        cleaned = re.sub(r"\s([a-zA-Z])\s", r"\1", text.strip())
        pages.append({"text": cleaned, "metadata": {"source": path.name, "page": i + 1}})
    return pages


def _chunk_documents(
    docs: list[dict],
    chunk_size: int = _CHUNK_SIZE,
    overlap: int = _CHUNK_OVERLAP,
) -> list[dict]:
    if not docs:
        return []
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    full_text = ""
    page_offsets: list[tuple[int, int, int]] = []
    pos = 0
    for page in docs:
        t = page["text"] + "\n"
        page_offsets.append((pos, pos + len(t), page["metadata"]["page"]))
        full_text += t
        pos += len(t)

    chunks = []
    start = 0
    source = docs[0]["metadata"]["source"]
    while start < len(full_text):
        end = start + chunk_size
        text = full_text[start:end].strip()
        if not text:
            start += chunk_size - overlap
            continue

        pages_in = [
            pn for ps, pe, pn in page_offsets if not (end <= ps or start >= pe)
        ]
        page_range = (min(pages_in), max(pages_in)) if pages_in else (None, None)

        page_range_str = f"{page_range[0]}-{page_range[1]}" if page_range[0] is not None else "unknown"
        chunks.append({"text": text, "metadata": {"source": source, "page_range": page_range_str}})
        start += chunk_size - overlap

    return chunks
