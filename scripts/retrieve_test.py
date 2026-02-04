import csv
import pickle
from pathlib import Path

from qdrant_client import QdrantClient

QDRANT_URL = "http://127.0.0.1:6333"
COLLECTION = "hse_admission_moscow_ba"
QA_PATH = Path("data/qa.csv")
VEC_PATH = Path("data/processed/tfidf.pkl")
TOPK = 3

def load_vectorizer():
    with open(VEC_PATH, "rb") as f:
        return pickle.load(f)

def read_questions(limit=None):
    rows = []
    with open(QA_PATH, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            q = (row.get("question") or "").strip()
            ctx = (row.get("gold_context") or "").strip()
            if q:
                rows.append((q, ctx))
            if limit and len(rows) >= limit:
                break
    return rows

def qdrant_query(client, vec, topk):
    res = client.query_points(
        collection_name=COLLECTION,
        query=vec,
        limit=topk,
        with_payload=True,
    )
    return list(res.points)

def normalize(s):
    return " ".join((s or "").lower().split())

def main():
    vec = load_vectorizer()
    client = QdrantClient(url=QDRANT_URL, check_compatibility=False)
    items = read_questions()
    total = 0
    hit1 = 0
    hit3 = 0

    for i, (q, gold) in enumerate(items, 1):
        qv = vec.transform([q]).toarray()[0].tolist()
        points = qdrant_query(client, qv, TOPK)
        texts = []
        for p in points:
            payload = p.payload or {}
            texts.append(payload.get("text", "") or "")

        g = normalize(gold)
        ok1 = False
        ok3 = False
        if g:
            if texts:
                ok1 = g in normalize(texts[0])
                ok3 = any(g in normalize(t) for t in texts)
        else:
            ok1 = True if texts else False
            ok3 = True if texts else False

        total += 1
        hit1 += 1 if ok1 else 0
        hit3 += 1 if ok3 else 0

        print(f"Q{i}: {q}")
        for j, t in enumerate(texts, 1):
            t_short = " ".join(t.split())[:300]
            print(f"  top{j}: {t_short}")
        if g:
            print(f"  gold_in_top1: {int(ok1)} gold_in_top3: {int(ok3)}")
        print()

    r1 = hit1 / total if total else 0.0
    r3 = hit3 / total if total else 0.0
    print(f"SUMMARY: n={total} recall@1={r1:.3f} recall@3={r3:.3f}")

if __name__ == "__main__":
    main()
