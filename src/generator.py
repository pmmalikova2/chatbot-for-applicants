from dataclasses import dataclass
import requests
import os
from dotenv import load_dotenv
from src.retrieval import RetrievedChunk
from src.config import GeneratorConfig


load_dotenv()


SYSTEM_PROMPT = (
    "Ты ассистент приемной комиссии НИУ ВШЭ. "
    "Отвечаешь абитуриентам бакалавриата на вопросы о поступлении в Москву в 2026 году. "
    "Отвечай на основе предоставленного контекста из официальных документов ВШЭ. "
    "Используй цифры, баллы, даты и стоимость только если они есть в контексте, не выдумывай их. "
    "Различай минимальные баллы ЕГЭ и пороги для олимпиад или особых прав: это разные вещи, не путай их. "
    "В таблицах каждая строка относится к одной программе: не переноси число из соседней строки. "
    "Символ прочерка (-) или (-*) означает, что мест по этому виду нет или данных нет, "
    "не заменяй прочерк числом из другой строки. "
    "Если в контексте нет ответа, честно скажи что информации нет "
    "и предложи обратиться по адресу abitur@hse.ru. "
    "Если вопрос не относится к поступлению в ВШЭ, вежливо откажись отвечать. "
    "Если пользователь грубит или оскорбляет тебя, не оскорбляй его в ответ. "
    "Ответь спокойно, с легкой иронией, и верни разговор к теме поступления. "
    "Отвечай на русском языке коротко и по делу, не более 150 слов."
)


def build_user_prompt(query: str, chunks: list[RetrievedChunk]) -> str:
    if not chunks:
        context = "(контекст не найден)"
    else:
        parts = []
        for i, c in enumerate(chunks, start=1):
            parts.append(f"[Источник {i}: {c.doc_title}]\n{c.text}")
        context = "\n\n".join(parts)
    return f"Контекст:\n{context}\n\nВопрос пользователя: {query}\n\nОтвет:"


@dataclass
class Generation:
    answer: str
    used_chunks: list[RetrievedChunk]
    raw_response: dict


class NoopGenerator:
    def generate(self, query: str, chunks: list[RetrievedChunk]) -> Generation:
        return Generation(answer="", used_chunks=chunks, raw_response={})


class OllamaGenerator:
    def __init__(self, model: str, base_url: str, temperature: float, max_tokens: int):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.max_tokens = max_tokens

    def generate(self, query: str, chunks: list[RetrievedChunk]) -> Generation:
        user = build_user_prompt(query, chunks)
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
            },
        }
        r = requests.post(f"{self.base_url}/api/chat", json=payload, timeout=300)
        r.raise_for_status()
        data = r.json()
        answer = data.get("message", {}).get("content", "").strip()
        return Generation(answer=answer, used_chunks=chunks, raw_response=data)


class OpenAIGenerator:
    def __init__(self, model: str, temperature: float, max_tokens: int):
        from openai import OpenAI
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def generate(self, query: str, chunks: list[RetrievedChunk]) -> Generation:
        user = build_user_prompt(query, chunks)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user},
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        answer = response.choices[0].message.content.strip()
        return Generation(answer=answer, used_chunks=chunks, raw_response={"id": response.id})


def build_generator(cfg: GeneratorConfig):
    if cfg.type == "none":
        return NoopGenerator()
    if cfg.type == "ollama":
        return OllamaGenerator(
            model=cfg.model,
            base_url=cfg.base_url,
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
        )
    if cfg.type == "openai":
        return OpenAIGenerator(
            model=cfg.model,
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
        )
    raise ValueError(f"Unknown generator type: {cfg.type}")
