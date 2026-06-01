import argparse
import json
import os
import time
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

JUDGE_PROMPT = (
    "Ты оцениваешь качество ответов чат-бота приемной комиссии НИУ ВШЭ. "
    "Тебе дан вопрос пользователя, контекст (найденные фрагменты документов), "
    "ответ бота и эталонный ответ. Оцени ответ бота по трем критериям:\n\n"
    "1. FAITHFULNESS (1-5): опирается ли ответ бота на предоставленный контекст? "
    "5 = полностью опирается на контекст, 1 = полностью выдуман.\n"
    "2. RELEVANCE (1-5): отвечает ли бот на заданный вопрос? "
    "5 = точно отвечает, 1 = не имеет отношения к вопросу.\n"
    "3. COMPLETENESS (1-5): насколько полон ответ по сравнению с эталоном? "
    "5 = содержит всю нужную информацию, 1 = пустой или бесполезный.\n\n"
    "Если бот корректно отказался отвечать (вопрос вне темы или нет информации "
    "в контексте) - это нормально, ставь высокие оценки за faithfulness и relevance.\n\n"
    "Ответь СТРОГО в формате JSON без пояснений:\n"
    "{\"faithfulness\": N, \"relevance\": N, \"completeness\": N}"
)


def judge_answer(client, question, context, bot_answer, gold_answer):
    user_msg = (
        f"Вопрос: {question}\n\n"
        f"Контекст (фрагменты документов):\n{context}\n\n"
        f"Ответ бота: {bot_answer}\n\n"
        f"Эталонный ответ: {gold_answer}"
    )
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": JUDGE_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.0,
            max_tokens=100,
        )
        raw = response.choices[0].message.content.strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        return json.loads(raw)
    except Exception:
        return {"faithfulness": 0, "relevance": 0, "completeness": 0}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    from openai import OpenAI
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    data = json.load(open(args.input, encoding="utf-8"))
    questions = data["per_question"]

    print(f"[*] Judging {len(questions)} answers")
    results = []
    t0 = time.time()

    for i, q in enumerate(questions, 1):
        context = "\n".join([
            f"[{j+1}] {ch.get('text_preview', '')}"
            for j, ch in enumerate(q.get("retrieved_chunks", []))
        ])
        scores = judge_answer(
            client,
            q["question"],
            context,
            q["answer"],
            q.get("gold_answer", ""),
        )
        results.append({
            "q_id": q["q_id"],
            "question": q["question"],
            "category": q["category"],
            "scores": scores,
        })
        if i % 10 == 0:
            print(f"  [{i}/{len(questions)}]")

    elapsed = time.time() - t0

    all_f = [r["scores"]["faithfulness"] for r in results if r["scores"]["faithfulness"] > 0]
    all_r = [r["scores"]["relevance"] for r in results if r["scores"]["relevance"] > 0]
    all_c = [r["scores"]["completeness"] for r in results if r["scores"]["completeness"] > 0]

    summary = {
        "input_file": args.input,
        "n_questions": len(questions),
        "eval_time_sec": round(elapsed, 1),
        "avg_faithfulness": round(sum(all_f) / len(all_f), 2) if all_f else 0,
        "avg_relevance": round(sum(all_r) / len(all_r), 2) if all_r else 0,
        "avg_completeness": round(sum(all_c) / len(all_c), 2) if all_c else 0,
    }

    out_path = Path(args.out) if args.out else Path(args.input).with_name(
        Path(args.input).stem + "_judge.json"
    )
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "per_question": results}, f, ensure_ascii=False, indent=2)

    print()
    print(f"=== LLM-as-Judge ===")
    print(f"questions: {len(questions)}")
    print(f"avg faithfulness: {summary['avg_faithfulness']}")
    print(f"avg relevance: {summary['avg_relevance']}")
    print(f"avg completeness: {summary['avg_completeness']}")
    print(f"eval time: {summary['eval_time_sec']}s")
    print(f"saved to: {out_path}")


if __name__ == "__main__":
    main()
