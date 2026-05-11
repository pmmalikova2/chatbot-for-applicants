import argparse
import json
import time
from pathlib import Path
from src.config import load_config
from src.metrics import load_gold_set
from src.rag_pipeline import build_index, query_pipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--qa", default="data/qa.csv")
    parser.add_argument("--out", default=None)
    parser.add_argument("--sample-size", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    print(f"[*] Building pipeline for {cfg.name}")
    t0 = time.time()
    pipeline = build_index(cfg)
    build_time = time.time() - t0
    print(f"[+] Pipeline built in {build_time:.1f}s, chunks: {len(pipeline.chunks)}")

    gold = load_gold_set(args.qa)
    print(f"[+] Loaded {len(gold)} gold questions")
    if args.sample_size and args.sample_size < len(gold):
        import random
        random.seed(42)
        by_category = {}
        for q in gold:
            by_category.setdefault(q.category, []).append(q)
        per_cat = max(1, args.sample_size // len(by_category))
        sampled = []
        for cat, items in by_category.items():
            random.shuffle(items)
            sampled.extend(items[:per_cat])
        if len(sampled) > args.sample_size:
            random.shuffle(sampled)
            sampled = sampled[:args.sample_size]
        gold = sampled
        print(f"[+] Sampled down to {len(gold)} questions")

    results = []
    total_latency = 0.0
    t1 = time.time()
    for i, gq in enumerate(gold, 1):
        q_t0 = time.time()
        retrieved, generation = query_pipeline(pipeline, gq.question)
        q_latency = time.time() - q_t0
        total_latency += q_latency

        results.append({
            "q_id": gq.q_id,
            "question": gq.question,
            "category": gq.category,
            "formulation_type": gq.formulation_type,
            "difficulty": gq.difficulty,
            "expected_doc_id": gq.expected_doc_id,
            "gold_answer": gq.gold_answer,
            "notes": gq.notes,
            "answer": generation.answer,
            "retrieved_chunks": [
                {
                    "chunk_id": r.chunk_id,
                    "doc_id": r.doc_id,
                    "doc_title": r.doc_title,
                    "doc_url": r.doc_url,
                    "score": r.score,
                    "rank": r.rank,
                    "text_preview": r.text[:200],
                }
                for r in retrieved
            ],
            "latency_sec": round(q_latency, 2),
        })
        if i % 10 == 0:
            avg = total_latency / i
            print(f"  [{i}/{len(gold)}] avg latency: {avg:.2f}s")

    eval_time = time.time() - t1
    avg_latency = total_latency / len(gold)

    summary = {
        "config_name": cfg.name,
        "n_chunks": len(pipeline.chunks),
        "n_questions": len(gold),
        "build_time_sec": round(build_time, 2),
        "eval_time_sec": round(eval_time, 2),
        "avg_latency_sec": round(avg_latency, 2),
    }

    out_path = Path(args.out) if args.out else Path("results/generation") / f"{cfg.name}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "per_question": results}, f, ensure_ascii=False, indent=2)

    print()
    print(f"=== {cfg.name} ===")
    print(f"chunks: {len(pipeline.chunks)}")
    print(f"questions: {len(gold)}")
    print(f"avg latency: {avg_latency:.2f}s")
    print(f"total time: {eval_time:.1f}s")
    print(f"saved to: {out_path}")


if __name__ == "__main__":
    main()
