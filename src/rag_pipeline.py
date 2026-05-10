from dataclasses import dataclass
from src.config import ExperimentConfig
from src.ingest import ingest_all, SourceDocument
from src.preprocessing import apply_preprocessing
from src.chunking import chunk_all, Chunk
from src.embedders import build_embedder, TfidfEmbedder
from src.vectorstore import QdrantStore
from src.retrieval import DenseRetriever, BM25Retriever, EnsembleRetriever, RetrievedChunk
from src.reranker import build_reranker
from src.generator import build_generator, Generation


@dataclass
class IndexedPipeline:
    cfg: ExperimentConfig
    chunks: list[Chunk]
    embedder: object
    store: QdrantStore | None
    retriever: object
    reranker: object
    generator: object


def build_index(cfg: ExperimentConfig) -> IndexedPipeline:
    docs = ingest_all()
    for d in docs:
        d.raw_text = apply_preprocessing(d.raw_text, cfg.preprocessing)
        d.pages = [apply_preprocessing(p, cfg.preprocessing) for p in d.pages]

    chunks = chunk_all(docs, cfg.chunker)

    embedder = build_embedder(cfg.embedder)
    if isinstance(embedder, TfidfEmbedder):
        embedder.fit([c.text for c in chunks])

    store = None
    if cfg.retriever.type in ("dense", "ensemble"):
        vectors = embedder.encode([c.text for c in chunks], is_query=False)
        store = QdrantStore(
            host=cfg.qdrant_host,
            port=cfg.qdrant_port,
            collection_name=cfg.collection_name,
            dim=embedder.dim,
        )
        store.recreate_collection()
        store.upsert_chunks(chunks, vectors)

    if cfg.retriever.type == "dense":
        retriever = DenseRetriever(store, embedder, cfg.retriever.top_k)
    elif cfg.retriever.type == "bm25":
        retriever = BM25Retriever(chunks, cfg.retriever.top_k)
    elif cfg.retriever.type == "ensemble":
        dense = DenseRetriever(store, embedder, cfg.retriever.top_k * 2)
        sparse = BM25Retriever(chunks, cfg.retriever.top_k * 2)
        retriever = EnsembleRetriever(dense, sparse, cfg.retriever.bm25_weight, cfg.retriever.top_k)
    else:
        raise ValueError(f"Unknown retriever type: {cfg.retriever.type}")

    reranker = build_reranker(cfg.reranker)
    generator = build_generator(cfg.generator)

    return IndexedPipeline(
        cfg=cfg,
        chunks=chunks,
        embedder=embedder,
        store=store,
        retriever=retriever,
        reranker=reranker,
        generator=generator,
    )


def query_pipeline(pipeline: IndexedPipeline, query: str) -> tuple[list[RetrievedChunk], Generation]:
    initial = pipeline.retriever.retrieve(query)
    reranked = pipeline.reranker.rerank(query, initial)
    generation = pipeline.generator.generate(query, reranked)
    return reranked, generation
