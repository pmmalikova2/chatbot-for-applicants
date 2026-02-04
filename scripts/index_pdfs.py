import os
import glob
import uuid
import pickle

import numpy as np
from tqdm import tqdm
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct


RAW_DIR = "data/raw"
COLLECTION = "hse_admission_moscow_ba"
QDRANT_URL = "http://127.0.0.1:6333"

CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200

VECTORIZER_PATH = "data/processed/tfidf.pkl"


def read_pdf_text(path: str) -> str:
    reader = PdfReader(path)
    parts = []
    for page in reader.pages:
        parts.append(page.extract_text() or "")
    return "\n".join(parts)


def chunk_text(text: str, chunk_size: int, overlap: int):
    text = " ".join(text.split())
    if not text:
        return []
    chunks = []
    i = 0
    step = max(1, chunk_size - overlap)
    while i < len(text):
        chunks.append(text[i:i + chunk_size])
        i += step
    return chunks


def main():
    os.makedirs("data/processed", exist_ok=True)

    client = QdrantClient(url=QDRANT_URL, check_compatibility=False)

    pdfs = sorted(glob.glob(os.path.join(RAW_DIR, "*.pdf")))
    if not pdfs:
        raise SystemExit(f"No PDFs in {RAW_DIR}")

    texts = []
    payloads = []

    for pdf in tqdm(pdfs, desc="Read PDFs"):
        text = read_pdf_text(pdf)
        chunks = chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)
        base = os.path.basename(pdf)
        for j, ch in enumerate(chunks):
            texts.append(ch)
            payloads.append({"source_file": base, "chunk_id": j, "text": ch})

    if not texts:
        raise SystemExit("No extracted text")

    vectorizer = TfidfVectorizer(max_features=4096)
    X = vectorizer.fit_transform(texts).astype(np.float32)
    vectors = X.toarray()
    dim = vectors.shape[1]

    with open(VECTORIZER_PATH, "wb") as f:
        pickle.dump(vectorizer, f)

    if client.collection_exists(COLLECTION):
        client.delete_collection(COLLECTION)

    client.create_collection(
        collection_name=COLLECTION,
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
    )

    points = [
        PointStruct(
            id=str(uuid.uuid4()),
            vector=v.tolist(),
            payload=pl
        )
        for v, pl in zip(vectors, payloads)
    ]

    client.upsert(collection_name=COLLECTION, points=points)
    print(f"OK: pdfs={len(pdfs)} chunks={len(points)} dim={dim}")
    print(f"Saved vectorizer: {VECTORIZER_PATH}")


if __name__ == "__main__":
    main()
