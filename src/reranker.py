from typing import Protocol
from src.retrieval import RetrievedChunk
from src.config import RerankerConfig


class Reranker(Protocol):
    def rerank(self, query: str, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]: ...


class NoopReranker:
    def rerank(self, query: str, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        return chunks


class CrossEncoderReranker:
    def __init__(self, model_name: str, top_k: int):
        from sentence_transformers import CrossEncoder
        self.model = CrossEncoder(model_name)
        self.top_k = top_k
        self.model_name = model_name

    def rerank(self, query: str, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        if not chunks:
            return chunks
        pairs = [(query, c.text) for c in chunks]
        scores = self.model.predict(pairs).tolist()
        scored = list(zip(chunks, scores))
        scored.sort(key=lambda x: x[1], reverse=True)
        out = []
        for rank, (c, s) in enumerate(scored[:self.top_k]):
            out.append(RetrievedChunk(
                text=c.text,
                doc_id=c.doc_id,
                doc_title=c.doc_title,
                doc_url=c.doc_url,
                chunk_id=c.chunk_id,
                score=float(s),
                rank=rank,
            ))
        return out


def build_reranker(cfg: RerankerConfig) -> Reranker:
    if cfg.type == "none":
        return NoopReranker()
    if cfg.type == "cross_encoder":
        return CrossEncoderReranker(cfg.model, cfg.top_k)
    raise ValueError(f"Unknown reranker type: {cfg.type}")
