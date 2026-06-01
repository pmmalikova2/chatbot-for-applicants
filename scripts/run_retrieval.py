import argparse
import json
import time
from pathlib import Path
from src.config import load_config
from src.metrics import (
    load_gold_set,
    doc_hit_at_k,
    chunk_hit_at_k,
    reciprocal_rank,
    aggregate_retrieval_metrics,
    breakdown_by_field,
)
from src.rag_pipeline import build_index, query_pipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--qa", default="data/qa.csv")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    print(f"[*] Building index for {cfg.name}")
    t0 = time.time()
    pipeline = build_index(cfg)
    build_time = time.time() - t0
    print(f"[+] Index built in {build_time:.1f}s, total chunks: {len(pipeline.chunks)}")

    gold = load_gold_set(args.qa)
    print(f"[+] Loaded {len(gold)} gold questions")

    per_question = []
    t1 = time.time()
    for gq in gold:
        retrieved, _ = query_pipeline(pipeline, gq.question)
        retrieved_doc_ids = [r.doc_id for r in retrieved]
        retrieved_texts = [r.text for r in retrieved]

        if gq.expected_doc_id == "NONE":
            per_question.append({
                "q_id": gq.q_id,
                "question": gq.question,
                "category": gq.category,
                "formulation_type": gq.formulation_type,
                "difficulty": gq.difficulty,
                "expected_doc_id": "NONE",
                "retrieved_doc_ids": retrieved_doc_ids,
                "doc_recall_at_1": 0,
                "doc_recall_at_3": 0,
                "doc_recall_at_5": 0,
                "chunk_recall_at_3": 0,
                "chunk_recall_at_5": 0,
                "mrr": 0.0,
            })
            continue

        per_question.append({
            "q_id": gq.q_id,
            "question": gq.question,
            "category": gq.category,
            "formulation_type": gq.formulation_type,
            "difficulty": gq.difficulty,
            "expected_doc_id": gq.expected_doc_id,
            "retrieved_doc_ids": retrieved_doc_ids,
            "doc_recall_at_1": doc_hit_at_k(retrieved_doc_ids, gq.expected_doc_id, 1),
            "doc_recall_at_3": doc_hit_at_k(retrieved_doc_ids, gq.expected_doc_id, 3),
            "doc_recall_at_5": doc_hit_at_k(retrieved_doc_ids, gq.expected_doc_id, 5),
            "chunk_recall_at_3": chunk_hit_at_k(retrieved_texts, gq.expected_chunk_quote, 3),
            "chunk_recall_at_5": chunk_hit_at_k(retrieved_texts, gq.expected_chunk_quote, 5),
            "mrr": reciprocal_rank(retrieved_doc_ids, gq.expected_doc_id),
        })
    eval_time = time.time() - t1

    overall = aggregate_retrieval_metrics(per_question)
    by_category = breakdown_by_field(per_question, "category")
    by_formulation = breakdown_by_field(per_question, "formulation_type")
    by_difficulty = breakdown_by_field(per_question, "difficulty")

    summary = {
        "config_name": cfg.name,
        "n_chunks": len(pipeline.chunks),
        "build_time_sec": round(build_time, 2),
        "eval_time_sec": round(eval_time, 2),
        "overall": overall,
        "by_category": by_category,
        "by_formulation_type": by_formulation,
        "by_difficulty": by_difficulty,
    }

    out_path = Path(args.out) if args.out else Path("results/retrieval") / f"{cfg.name}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "per_question": per_question}, f, ensure_ascii=False, indent=2)

    print()
    print(f"=== {cfg.name} ===")
    print(f"chunks: {len(pipeline.chunks)}")
    print(f"in-scope questions: {overall.get('n_in_scope', 0)}")
    print(f"doc_recall@1: {overall.get('doc_recall_at_1', 0):.3f}")
    print(f"doc_recall@3: {overall.get('doc_recall_at_3', 0):.3f}")
    print(f"doc_recall@5: {overall.get('doc_recall_at_5', 0):.3f}")
    print(f"chunk_recall@3: {overall.get('chunk_recall_at_3', 0):.3f}")
    print(f"chunk_recall@5: {overall.get('chunk_recall_at_5', 0):.3f}")
    print(f"MRR: {overall.get('mrr', 0):.3f}")
    print(f"saved to: {out_path}")


if __name__ == "__main__":
    main()
