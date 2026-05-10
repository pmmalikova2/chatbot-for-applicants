from typing import Protocol
import numpy as np
from src.config import EmbedderConfig


class Embedder(Protocol):
    dim: int
    name: str

    def encode(self, texts: list[str], is_query: bool = False) -> np.ndarray: ...


class TfidfEmbedder:
    def __init__(self, max_features: int = 4096):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.vectorizer = TfidfVectorizer(max_features=max_features)
        self.fitted = False
        self.max_features = max_features
        self.name = f"tfidf-{max_features}"
        self.dim = max_features

    def fit(self, texts: list[str]):
        self.vectorizer.fit(texts)
        self.fitted = True
        self.dim = len(self.vectorizer.get_feature_names_out())

    def encode(self, texts: list[str], is_query: bool = False) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("TfidfEmbedder must be fitted before encode")
        m = self.vectorizer.transform(texts).toarray().astype(np.float32)
        norms = np.linalg.norm(m, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return m / norms


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)
        self.name = model_name
        self.dim = self.model.get_sentence_embedding_dimension()
        self._is_e5 = "e5" in model_name.lower()

    def encode(self, texts: list[str], is_query: bool = False) -> np.ndarray:
        if self._is_e5:
            prefix = "query: " if is_query else "passage: "
            texts = [prefix + t for t in texts]
        emb = self.model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return emb.astype(np.float32)


class OllamaEmbedder:
    def __init__(self, model_name: str = "bge-m3", base_url: str = "http://localhost:11434"):
        import requests
        self.requests = requests
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.name = f"ollama-{model_name}"
        probe = self._call("ping")
        self.dim = len(probe)

    def _call(self, text: str) -> list[float]:
        r = self.requests.post(
            f"{self.base_url}/api/embeddings",
            json={"model": self.model_name, "prompt": text},
            timeout=120,
        )
        r.raise_for_status()
        return r.json()["embedding"]

    def encode(self, texts: list[str], is_query: bool = False) -> np.ndarray:
        out = []
        for t in texts:
            v = self._call(t)
            out.append(v)
        arr = np.array(out, dtype=np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return arr / norms


def build_embedder(cfg: EmbedderConfig) -> Embedder:
    if cfg.type == "tfidf":
        return TfidfEmbedder(max_features=cfg.max_features)
    if cfg.type == "sentence_transformer":
        return SentenceTransformerEmbedder(cfg.model)
    if cfg.type == "ollama":
        return OllamaEmbedder(model_name=cfg.model)
    raise ValueError(f"Unknown embedder type: {cfg.type}")
