import os
from dotenv import load_dotenv
from src.config import ExperimentConfig

load_dotenv()

REWRITE_PROMPT = (
    "Ты помощник, который переформулирует вопросы абитуриентов. "
    "Пользователь может писать с опечатками, сокращениями или разговорно. "
    "Твоя задача - переписать вопрос в формальном стиле, раскрыв сокращения "
    "и исправив опечатки. Контекст: поступление в НИУ ВШЭ, Москва, бакалавриат, 2026 год. "
    "Известные сокращения: ПМИ = Прикладная математика и информатика, "
    "ПИ = Программная инженерия, КН = Компьютерные науки и анализ данных, "
    "БВИ = прием без вступительных испытаний, ИД = индивидуальные достижения, "
    "КМС = кандидат в мастера спорта, ГТО = Готов к труду и обороне, "
    "ВсОШ = Всероссийская олимпиада школьников, ФДП = факультет довузовской подготовки, "
    "ФМШ = физико-математическая школа, DANO = национальная олимпиада по анализу данных. "
    "Верни ТОЛЬКО переформулированный вопрос, без пояснений."
)


class QueryRewriter:
    def __init__(self, model: str = "gpt-4o-mini"):
        from openai import OpenAI
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = model

    def rewrite(self, query: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": REWRITE_PROMPT},
                {"role": "user", "content": query},
            ],
            temperature=0.0,
            max_tokens=200,
        )
        return response.choices[0].message.content.strip()


class NoopRewriter:
    def rewrite(self, query: str) -> str:
        return query


def build_rewriter(cfg: ExperimentConfig):
    if hasattr(cfg, "query_rewriter") and cfg.query_rewriter:
        return QueryRewriter(model=cfg.query_rewriter.get("model", "gpt-4o-mini"))
    return NoopRewriter()
