from dataclasses import dataclass
import numpy as np
from src.chunking import Chunk
from src.embedders import Embedder
from src.vectorstore import QdrantStore
from src.config import RetrieverConfig


@dataclass
class RetrievedChunk:
    text: str
    doc_id: str
    doc_title: str
    doc_url: str
    chunk_id: str
    score: float
    rank: int


class DenseRetriever:
    def __init__(self, store: QdrantStore, embedder: Embedder, top_k: int):
        self.store = store
        self.embedder = embedder
        self.top_k = top_k

    def retrieve(self, query: str) -> list[RetrievedChunk]:
        qvec = self.embedder.encode([query], is_query=True)[0]
        hits = self.store.search(qvec, top_k=self.top_k)
        out = []
        for rank, h in enumerate(hits):
            out.append(RetrievedChunk(
                text=h["text"],
                doc_id=h["doc_id"],
                doc_title=h["doc_title"],
                doc_url=h["doc_url"],
                chunk_id=h["chunk_id"],
                score=float(h["score"]),
                rank=rank,
            ))
        return out


class BM25Retriever:
    def __init__(self, chunks: list[Chunk], top_k: int):
        from rank_bm25 import BM25Okapi
        self.chunks = chunks
        self.tokenized = [self._tokenize(c.text) for c in chunks]
        self.bm25 = BM25Okapi(self.tokenized)
        self.top_k = top_k

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        import re
        return re.findall(r"\w+", text.lower())

    def retrieve(self, query: str) -> list[RetrievedChunk]:
        scores = self.bm25.get_scores(self._tokenize(query))
        top_idx = np.argsort(scores)[::-1][:self.top_k]
        out = []
        for rank, i in enumerate(top_idx):
            c = self.chunks[i]
            out.append(RetrievedChunk(
                text=c.text,
                doc_id=c.doc_id,
                doc_title=c.doc_title,
                doc_url=c.doc_url,
                chunk_id=c.chunk_id,
                score=float(scores[i]),
                rank=rank,
            ))
        return out


class EnsembleRetriever:
    def __init__(self, dense: DenseRetriever, sparse: BM25Retriever, dense_weight: float, top_k: int):
        self.dense = dense
        self.sparse = sparse
        self.dense_weight = dense_weight
        self.top_k = top_k

    def retrieve(self, query: str) -> list[RetrievedChunk]:
        dense_hits = self.dense.retrieve(query)
        sparse_hits = self.sparse.retrieve(query)

        def to_rank_score(hits: list[RetrievedChunk]) -> dict[str, float]:
            return {h.chunk_id: 1.0 / (h.rank + 60) for h in hits}

        dense_rs = to_rank_score(dense_hits)
        sparse_rs = to_rank_score(sparse_hits)

        all_chunks: dict[str, RetrievedChunk] = {}
        for h in dense_hits + sparse_hits:
            all_chunks[h.chunk_id] = h

        combined = {}
        for cid in all_chunks:
            d = dense_rs.get(cid, 0.0)
            s = sparse_rs.get(cid, 0.0)
            combined[cid] = self.dense_weight * d + (1.0 - self.dense_weight) * s

        sorted_ids = sorted(combined.keys(), key=lambda x: combined[x], reverse=True)[:self.top_k]
        out = []
        for rank, cid in enumerate(sorted_ids):
            base = all_chunks[cid]
            out.append(RetrievedChunk(
                text=base.text,
                doc_id=base.doc_id,
                doc_title=base.doc_title,
                doc_url=base.doc_url,
                chunk_id=base.chunk_id,
                score=float(combined[cid]),
                rank=rank,
            ))
        return out
