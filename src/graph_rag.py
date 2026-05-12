import os
import json
import re
from dataclasses import dataclass, field
from dotenv import load_dotenv
from src.chunking import Chunk
from src.retrieval import RetrievedChunk

load_dotenv()

EXTRACT_PROMPT = (
    "Извлеки ключевые сущности из фрагмента документа о поступлении в НИУ ВШЭ. "
    "Типы сущностей: PROGRAM (образовательная программа), OLYMPIAD (олимпиада), "
    "EXAM (экзамен/испытание), ACHIEVEMENT (индивидуальное достижение), "
    "DISCOUNT (скидка), DEADLINE (срок), SCORE (баллы), DOCUMENT (документ), "
    "QUOTA (квота), RIGHT (право при поступлении). "
    "Верни JSON-список объектов: [{\"name\": \"...\", \"type\": \"...\"}]. "
    "Только JSON, без пояснений. Максимум 10 сущностей."
)


@dataclass
class Entity:
    name: str
    entity_type: str
    chunk_ids: list[str] = field(default_factory=list)


class KnowledgeGraph:
    def __init__(self):
        import networkx as nx
        self.nx = nx
        self.graph = nx.Graph()
        self.entity_to_chunks: dict[str, set[str]] = {}

    def add_entities_from_chunk(self, chunk_id: str, entities: list[Entity]):
        for ent in entities:
            key = f"{ent.entity_type}:{ent.name}".lower()
            if not self.graph.has_node(key):
                self.graph.add_node(key, name=ent.name, entity_type=ent.entity_type)
            if key not in self.entity_to_chunks:
                self.entity_to_chunks[key] = set()
            self.entity_to_chunks[key].add(chunk_id)

        for i, e1 in enumerate(entities):
            for e2 in entities[i+1:]:
                k1 = f"{e1.entity_type}:{e1.name}".lower()
                k2 = f"{e2.entity_type}:{e2.name}".lower()
                if self.graph.has_edge(k1, k2):
                    self.graph[k1][k2]["weight"] += 1
                else:
                    self.graph.add_edge(k1, k2, weight=1)

    def find_related_chunks(self, query_entities: list[Entity], max_hops: int = 2) -> set[str]:
        chunk_ids = set()
        for ent in query_entities:
            key = f"{ent.entity_type}:{ent.name}".lower()
            if key in self.entity_to_chunks:
                chunk_ids.update(self.entity_to_chunks[key])
            for node in self.graph.nodes():
                if ent.name.lower() in node:
                    chunk_ids.update(self.entity_to_chunks.get(node, set()))
                    for neighbor in self.graph.neighbors(node):
                        chunk_ids.update(self.entity_to_chunks.get(neighbor, set()))
        return chunk_ids


class GraphRAGBuilder:
    def __init__(self, model: str = "gpt-4o-mini"):
        from openai import OpenAI
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = model

    def extract_entities(self, text: str) -> list[Entity]:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": EXTRACT_PROMPT},
                    {"role": "user", "content": text[:2000]},
                ],
                temperature=0.0,
                max_tokens=500,
            )
            raw = response.choices[0].message.content.strip()
            raw = re.sub(r"```json\s*", "", raw)
            raw = re.sub(r"```\s*", "", raw)
            items = json.loads(raw)
            return [Entity(name=it["name"], entity_type=it.get("type", "OTHER")) for it in items]
        except Exception:
            return []

    def build_graph(self, chunks: list[Chunk], batch_size: int = 5) -> KnowledgeGraph:
        kg = KnowledgeGraph()
        for i, chunk in enumerate(chunks):
            entities = self.extract_entities(chunk.text)
            kg.add_entities_from_chunk(chunk.chunk_id, entities)
            if (i + 1) % 50 == 0:
                print(f"  [graph] processed {i+1}/{len(chunks)} chunks, "
                      f"nodes: {kg.graph.number_of_nodes()}, edges: {kg.graph.number_of_edges()}")
        print(f"  [graph] done: {kg.graph.number_of_nodes()} nodes, {kg.graph.number_of_edges()} edges")
        return kg

    def extract_query_entities(self, query: str) -> list[Entity]:
        return self.extract_entities(query)


class GraphEnhancedRetriever:
    def __init__(self, base_retriever, kg: KnowledgeGraph, builder: GraphRAGBuilder,
                 chunks: list[Chunk], top_k: int = 5):
        self.base_retriever = base_retriever
        self.kg = kg
        self.builder = builder
        self.chunk_map = {c.chunk_id: c for c in chunks}
        self.top_k = top_k

    def retrieve(self, query: str) -> list[RetrievedChunk]:
        base_results = self.base_retriever.retrieve(query)

        query_entities = self.builder.extract_query_entities(query)
        graph_chunk_ids = self.kg.find_related_chunks(query_entities)

        boosted = []
        seen = set()
        for r in base_results:
            boost = 0.3 if r.chunk_id in graph_chunk_ids else 0.0
            boosted.append(RetrievedChunk(
                text=r.text,
                doc_id=r.doc_id,
                doc_title=r.doc_title,
                doc_url=r.doc_url,
                chunk_id=r.chunk_id,
                score=r.score + boost,
                rank=0,
            ))
            seen.add(r.chunk_id)

        for cid in graph_chunk_ids:
            if cid not in seen and cid in self.chunk_map:
                c = self.chunk_map[cid]
                boosted.append(RetrievedChunk(
                    text=c.text,
                    doc_id=c.doc_id,
                    doc_title=c.doc_title,
                    doc_url=c.doc_url,
                    chunk_id=c.chunk_id,
                    score=0.2,
                    rank=0,
                ))
                seen.add(cid)

        boosted.sort(key=lambda x: x.score, reverse=True)
        for i, r in enumerate(boosted):
            r.rank = i
        return boosted[:self.top_k]
