import numpy as np
from src.chunking import Chunk


class QdrantStore:
    def __init__(self, host: str, port: int, collection_name: str, dim: int, path: str = ""):
        from qdrant_client import QdrantClient
        from qdrant_client.http import models as qm
        self.qm = qm
        if path and path.strip():
            from pathlib import Path
            Path(path).mkdir(parents=True, exist_ok=True)
            self.client = QdrantClient(path=path)
        else:
            self.client = QdrantClient(host=host, port=port, prefer_grpc=False, https=False, timeout=60)
        self.collection_name = collection_name
        self.dim = dim

    def recreate_collection(self):
        if self.client.collection_exists(self.collection_name):
            self.client.delete_collection(self.collection_name)
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=self.qm.VectorParams(size=self.dim, distance=self.qm.Distance.COSINE),
        )

    def upsert_chunks(self, chunks: list[Chunk], vectors: np.ndarray, batch_size: int = 128):
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors length mismatch")
        points = []
        for i, (chunk, vec) in enumerate(zip(chunks, vectors)):
            payload = {
                "chunk_id": chunk.chunk_id,
                "text": chunk.text,
                "doc_id": chunk.doc_id,
                "doc_title": chunk.doc_title,
                "doc_url": chunk.doc_url,
                "doc_category": chunk.doc_category,
                "chunk_idx": chunk.chunk_idx,
                "char_start": chunk.char_start,
                "char_end": chunk.char_end,
            }
            points.append(self.qm.PointStruct(id=i, vector=vec.tolist(), payload=payload))
            if len(points) >= batch_size:
                self.client.upsert(collection_name=self.collection_name, points=points)
                points = []
        if points:
            self.client.upsert(collection_name=self.collection_name, points=points)

    def search(self, query_vector, top_k: int = 5) -> list[dict]:
        result = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector.tolist(),
            limit=top_k,
            with_payload=True,
        )
        return [{"score": p.score, **p.payload} for p in result.points]
