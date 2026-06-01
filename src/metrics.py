from dataclasses import dataclass
import csv
import re


@dataclass
class GoldQuestion:
    q_id: str
    question: str
    category: str
    formulation_type: str
    difficulty: str
    expected_doc_id: str
    expected_chunk_quote: str
    gold_answer: str
    notes: str


def load_gold_set(path: str) -> list[GoldQuestion]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            out.append(GoldQuestion(
                q_id=row["q_id"],
                question=row["question"],
                category=row["category"],
                formulation_type=row["formulation_type"],
                difficulty=row["difficulty"],
                expected_doc_id=row["expected_doc_id"],
                expected_chunk_quote=row["expected_chunk_quote"],
                gold_answer=row["gold_answer"],
                notes=row["notes"],
            ))
    return out


def doc_hit_at_k(retrieved_doc_ids: list[str], expected_doc_id: str, k: int) -> int:
    return int(expected_doc_id in retrieved_doc_ids[:k])


def normalize_for_match(text: str) -> str:
    t = text.lower()
    t = re.sub(r"\s+", " ", t)
    return t.strip()


def chunk_hit_at_k(retrieved_texts: list[str], expected_quote: str, k: int) -> int:
    if not expected_quote or not expected_quote.strip():
        return 0
    target = normalize_for_match(expected_quote)
    target_tokens = target.split()
    if len(target_tokens) < 3:
        return 0
    for text in retrieved_texts[:k]:
        normalized = normalize_for_match(text)
        if target in normalized:
            return 1
        matched = sum(1 for tok in target_tokens if tok in normalized)
        if matched / len(target_tokens) >= 0.7:
            return 1
    return 0


def reciprocal_rank(retrieved_doc_ids: list[str], expected_doc_id: str) -> float:
    for i, d in enumerate(retrieved_doc_ids):
        if d == expected_doc_id:
            return 1.0 / (i + 1)
    return 0.0


def aggregate_retrieval_metrics(per_question: list[dict]) -> dict:
    if not per_question:
        return {}
    in_scope = [r for r in per_question if r["expected_doc_id"] != "NONE"]
    n = len(in_scope) if in_scope else 1
    sum_doc_at1 = sum(r["doc_recall_at_1"] for r in in_scope)
    sum_doc_at3 = sum(r["doc_recall_at_3"] for r in in_scope)
    sum_doc_at5 = sum(r["doc_recall_at_5"] for r in in_scope)
    sum_chunk_at3 = sum(r["chunk_recall_at_3"] for r in in_scope)
    sum_chunk_at5 = sum(r["chunk_recall_at_5"] for r in in_scope)
    sum_mrr = sum(r["mrr"] for r in in_scope)
    return {
        "n_in_scope": len(in_scope),
        "doc_recall_at_1": sum_doc_at1 / n,
        "doc_recall_at_3": sum_doc_at3 / n,
        "doc_recall_at_5": sum_doc_at5 / n,
        "chunk_recall_at_3": sum_chunk_at3 / n,
        "chunk_recall_at_5": sum_chunk_at5 / n,
        "mrr": sum_mrr / n,
    }


def breakdown_by_field(per_question: list[dict], field: str) -> dict[str, dict]:
    groups: dict[str, list[dict]] = {}
    for r in per_question:
        if r["expected_doc_id"] == "NONE":
            continue
        key = r.get(field, "")
        groups.setdefault(key, []).append(r)
    out = {}
    for key, rows in groups.items():
        out[key] = aggregate_retrieval_metrics(rows)
    return out
