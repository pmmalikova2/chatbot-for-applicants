from dataclasses import dataclass, field
from pathlib import Path
import yaml


@dataclass
class ChunkerConfig:
    type: str = "recursive"
    size: int = 1200
    overlap: int = 200


@dataclass
class EmbedderConfig:
    type: str = "tfidf"
    model: str = ""
    max_features: int = 4096


@dataclass
class RetrieverConfig:
    type: str = "dense"
    top_k: int = 5
    bm25_weight: float = 0.5


@dataclass
class RerankerConfig:
    type: str = "none"
    model: str = ""
    top_k: int = 3


@dataclass
class GeneratorConfig:
    type: str = "none"
    model: str = "qwen2.5:7b-instruct"
    temperature: float = 0.2
    max_tokens: int = 400
    base_url: str = "http://localhost:11434"


@dataclass
class PreprocessingConfig:
    lowercase: bool = False
    remove_extra_whitespace: bool = True
    remove_page_numbers: bool = True
    remove_headers_footers: bool = False


@dataclass
class ExperimentConfig:
    name: str = "baseline"
    collection_name: str = "hse_admission_baseline"
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_path: str = ""
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    chunker: ChunkerConfig = field(default_factory=ChunkerConfig)
    embedder: EmbedderConfig = field(default_factory=EmbedderConfig)
    retriever: RetrieverConfig = field(default_factory=RetrieverConfig)
    reranker: RerankerConfig = field(default_factory=RerankerConfig)
    generator: GeneratorConfig = field(default_factory=GeneratorConfig)


def load_config(path: str | Path) -> ExperimentConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    cfg = ExperimentConfig()
    cfg.name = raw.get("name", cfg.name)
    cfg.collection_name = raw.get("collection_name", cfg.collection_name)
    cfg.qdrant_host = raw.get("qdrant_host", cfg.qdrant_host)
    cfg.qdrant_port = raw.get("qdrant_port", cfg.qdrant_port)
    cfg.qdrant_path = raw.get("qdrant_path", cfg.qdrant_path)

    if "preprocessing" in raw:
        cfg.preprocessing = PreprocessingConfig(**raw["preprocessing"])
    if "chunker" in raw:
        cfg.chunker = ChunkerConfig(**raw["chunker"])
    if "embedder" in raw:
        cfg.embedder = EmbedderConfig(**raw["embedder"])
    if "retriever" in raw:
        cfg.retriever = RetrieverConfig(**raw["retriever"])
    if "reranker" in raw:
        cfg.reranker = RerankerConfig(**raw["reranker"])
    if "generator" in raw:
        cfg.generator = GeneratorConfig(**raw["generator"])

    return cfg
