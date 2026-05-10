from dataclasses import dataclass
import requests
from src.retrieval import RetrievedChunk
from src.config import GeneratorConfig


SYSTEM_PROMPT = (
    "Ты ассистент приемной комиссии НИУ ВШЭ. "
    "Отвечаешь абитуриентам бакалавриата на вопросы о поступлении в Москву в 2026 году. "
    "Отвечай только на основе предоставленного контекста из официальных документов ВШЭ. "
    "Если в контексте нет ответа на вопрос, честно скажи что не знаешь и предложи "
    "обратиться по адресу abitur@hse.ru. "
    "Если вопрос не относится к поступлению в ВШЭ, вежливо откажись отвечать. "
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
    raise ValueError(f"Unknown generator type: {cfg.type}")
