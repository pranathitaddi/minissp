"""
minissp/retrieval.py

Phase 2 of plan.md (§2.2): in-process dense retrieval over the scoped wiki-18
corpus built by `minissp.data build-index`.

    retriever = Retriever(index_dir, embed_name)
    retriever.load()                       # e5 + faiss.read_index + docs.jsonl
    docs = retriever.search(["who wrote hamlet"], topk=3)
    block = format_information(docs[0])

Lazy loading is not optional: `import minissp.retrieval` must not touch disk,
the network, faiss or sentence-transformers. Everything heavy happens inside
`.load()`, mirroring data.py's discipline.

e5 requires asymmetric prefixing: passages were embedded by
`data.build_index` as "passage: {title}\\n{text}", so queries must be encoded
as "query: {text}" or retrieval quality collapses.
"""

from __future__ import annotations

import json
from pathlib import Path

QUERY_PREFIX = "query: "

INDEX_FILENAME = "corpus.faiss"
DOCS_FILENAME = "docs.jsonl"


class Retriever:
    """Dense retriever over a FAISS IndexFlatIP of e5-embedded passages."""

    def __init__(self, index_dir: str | Path, embed_name: str = "intfloat/e5-base-v2",
                 device: str = "cpu") -> None:
        # Nothing is loaded here on purpose (plan §2.2).
        self.index_dir = Path(index_dir)
        self.embed_name = embed_name
        self.device = device
        self.model = None
        self.index = None
        self.docs: list[dict] = []

    # -- loading ----------------------------------------------------------

    @property
    def index_path(self) -> Path:
        return self.index_dir / INDEX_FILENAME

    @property
    def docs_path(self) -> Path:
        return self.index_dir / DOCS_FILENAME

    def is_loaded(self) -> bool:
        return self.model is not None and self.index is not None

    def load(self) -> "Retriever":
        """Loads the e5 encoder, the FAISS index and docs.jsonl. Idempotent."""
        if self.is_loaded():
            return self

        import faiss
        from sentence_transformers import SentenceTransformer

        if not self.index_path.exists():
            raise FileNotFoundError(
                f"FAISS index not found at {self.index_path}. "
                f"Build it with `python -m minissp.data build-index --out {self.index_dir}`."
            )
        if not self.docs_path.exists():
            raise FileNotFoundError(
                f"docs.jsonl not found at {self.docs_path}. "
                f"It is written alongside the index by `minissp.data build-index`."
            )

        self.model = SentenceTransformer(self.embed_name, device=self.device)
        self.index = faiss.read_index(str(self.index_path))

        docs: list[dict] = []
        with open(self.docs_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    docs.append(json.loads(line))
        self.docs = docs

        if self.index.ntotal != len(self.docs):
            raise ValueError(
                f"index/docs mismatch: {self.index_path} holds {self.index.ntotal} "
                f"vectors but {self.docs_path} has {len(self.docs)} lines. "
                f"They must be written by the same build-index run."
            )
        return self

    # -- search -----------------------------------------------------------

    def search(self, queries: list[str], topk: int = 3) -> list[list[dict]]:
        """
        Batched search. Returns one list of `topk` docs per query, each doc a
        {"id", "title", "text", "score"} dict, best first.
        """
        if not self.is_loaded():
            raise RuntimeError(
                "Retriever.load() must be called before Retriever.search(); "
                "the encoder and FAISS index are loaded lazily on purpose."
            )
        if not queries:
            return []

        prefixed = [QUERY_PREFIX + (q or "").strip() for q in queries]
        embeddings = self.model.encode(
            prefixed,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        k = min(topk, len(self.docs))
        scores, indices = self.index.search(embeddings, k)

        results: list[list[dict]] = []
        for row_scores, row_indices in zip(scores, indices):
            hits: list[dict] = []
            for score, idx in zip(row_scores, row_indices):
                if idx < 0:  # faiss pads with -1 when fewer than k are found
                    continue
                doc = self.docs[int(idx)]
                hits.append({
                    "id": doc.get("id"),
                    "title": doc.get("title", ""),
                    "text": doc.get("text", ""),
                    "score": float(score),
                })
            results.append(hits)
        return results


def format_information(docs: list[dict]) -> str:
    """
    Renders retrieved docs for an <information> block, paper Tables 7-12 style:

        (Title: "Hamlet") Hamlet is a tragedy written by ...

    one doc per line.
    """
    return "\n".join(
        f'(Title: "{d.get("title", "")}") {d.get("text", "")}' for d in docs
    )
