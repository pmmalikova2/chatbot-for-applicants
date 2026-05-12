import os
import json
from dataclasses import dataclass
from dotenv import load_dotenv
from src.retrieval import RetrievedChunk

load_dotenv()

ROUTER_PROMPT = (
    "Ты классификатор запросов для чат-бота приемной комиссии НИУ ВШЭ. "
    "Определи тип запроса пользователя. Ответь ТОЛЬКО одним словом.\n\n"
    "OUT_OF_SCOPE - ТОЛЬКО если вопрос ТОЧНО не имеет отношения к поступлению. "
    "Примеры: погода, рецепты, развлечения, знаменитости, день недели. "
    "ВАЖНО: вопросы про олимпиады, баллы, ГТО, скидки, стоимость обучения, "
    "индивидуальные достижения, КМС, БВИ, аттестат, волонтерство, общежитие, "
    "апелляцию, документы - это ВСЕ про поступление, НЕ out-of-scope!\n\n"
    "NEEDS_REWRITE - если вопрос содержит сокращения (ПМИ, ПИ, БВИ, ИД, КМС, "
    "ГТО, ВсОШ, ФДП, ФМШ, КН, БИ), опечатки, разговорный стиль "
    "или неформальные выражения.\n\n"
    "DIRECT - если вопрос сформулирован четко и формально.\n\n"
    "Если сомневаешься между OUT_OF_SCOPE и другим вариантом - выбирай DIRECT. "
    "Лучше лишний раз поискать, чем пропустить вопрос про поступление."
)

REFLECTION_PROMPT = (
    "Ты оцениваешь качество результатов поиска для RAG-системы. "
    "Тебе дан вопрос пользователя и найденные фрагменты документов. "
    "Определи, содержат ли фрагменты ответ на вопрос. "
    "Ответь ТОЛЬКО одним словом:\n"
    "- GOOD - фрагменты содержат релевантную информацию для ответа\n"
    "- BAD - фрагменты не содержат нужной информации, стоит попробовать другую формулировку"
)

REWRITE_PROMPT = (
    "Ты помощник, который переформулирует вопросы абитуриентов. "
    "Пользователь может писать с опечатками, сокращениями или разговорно. "
    "Твоя задача - переписать вопрос в формальном стиле, раскрыв сокращения "
    "и исправив опечатки. Контекст: поступление в НИУ ВШЭ, Москва, бакалавриат, 2026 год. "
    "Известные сокращения: ПМИ = Прикладная математика и информатика, "
    "ПИ = Программная инженерия, БИ = Бизнес-информатика, "
    "КН = Компьютерные науки и анализ данных, "
    "БВИ = прием без вступительных испытаний, ИД = индивидуальные достижения, "
    "КМС = кандидат в мастера спорта, ГТО = Готов к труду и обороне, "
    "ВсОШ = Всероссийская олимпиада школьников, ФДП = факультет довузовской подготовки, "
    "ФМШ = физико-математическая школа, DANO = национальная олимпиада по анализу данных, "
    "МЭФ = Международная программа по экономике и финансам, "
    "ФКН = факультет компьютерных наук, ФЭН = факультет экономических наук, "
    "ДВИ = дополнительные вступительные испытания, "
    "СПО = среднее профессиональное образование, ЕПГУ = портал Госуслуг. "
    "Верни ТОЛЬКО переформулированный вопрос, без пояснений."
)


@dataclass
class AgentDecision:
    original_query: str
    intent: str
    rewritten_query: str
    search_query: str
    reflection: str
    retried: bool


class RAGAgent:
    def __init__(self, model: str = "gpt-4o-mini"):
        from openai import OpenAI
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = model

    def _call_llm(self, system: str, user: str, max_tokens: int = 50) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.0,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content.strip()

    def classify_intent(self, query: str) -> str:
        result = self._call_llm(ROUTER_PROMPT, query)
        result = result.upper().strip()
        if "OUT" in result:
            return "OUT_OF_SCOPE"
        if "REWRITE" in result or "NEEDS" in result:
            return "NEEDS_REWRITE"
        return "DIRECT"

    def rewrite_query(self, query: str) -> str:
        return self._call_llm(REWRITE_PROMPT, query, max_tokens=200)

    def reflect_on_results(self, query: str, chunks: list[RetrievedChunk]) -> str:
        if not chunks:
            return "BAD"
        context = "\n".join([f"[{i+1}] {c.text[:300]}" for i, c in enumerate(chunks[:3])])
        user_msg = f"Вопрос: {query}\n\nНайденные фрагменты:\n{context}"
        result = self._call_llm(REFLECTION_PROMPT, user_msg)
        return "GOOD" if "GOOD" in result.upper() else "BAD"

    def run(self, query: str, retrieve_fn, top_k: int = 5) -> tuple[list[RetrievedChunk], AgentDecision]:
        intent = self.classify_intent(query)

        if intent == "OUT_OF_SCOPE":
            return [], AgentDecision(
                original_query=query,
                intent=intent,
                rewritten_query="",
                search_query="",
                reflection="SKIPPED",
                retried=False,
            )

        if intent == "NEEDS_REWRITE":
            rewritten = self.rewrite_query(query)
            search_query = rewritten
        else:
            rewritten = query
            search_query = query

        chunks = retrieve_fn(search_query)

        reflection = self.reflect_on_results(query, chunks)

        if reflection == "BAD" and intent == "DIRECT":
            rewritten = self.rewrite_query(query)
            search_query = rewritten
            chunks = retrieve_fn(search_query)
            reflection = self.reflect_on_results(query, chunks)
            retried = True
        elif reflection == "BAD" and intent == "NEEDS_REWRITE":
            alt_query = f"НИУ ВШЭ Москва бакалавриат 2026 {query}"
            chunks = retrieve_fn(alt_query)
            reflection = self.reflect_on_results(query, chunks)
            retried = True
        else:
            retried = False

        return chunks, AgentDecision(
            original_query=query,
            intent=intent,
            rewritten_query=rewritten,
            search_query=search_query,
            reflection=reflection,
            retried=retried,
        )


class NoopAgent:
    def run(self, query: str, retrieve_fn, top_k: int = 5) -> tuple[list, AgentDecision]:
        chunks = retrieve_fn(query)
        return chunks, AgentDecision(
            original_query=query,
            intent="DIRECT",
            rewritten_query=query,
            search_query=query,
            reflection="SKIPPED",
            retried=False,
        )


def build_agent(cfg):
    if hasattr(cfg, "agent") and cfg.agent and cfg.agent.get("enabled", False):
        return RAGAgent(model=cfg.agent.get("model", "gpt-4o-mini"))
    return NoopAgent()
