from dataclasses import dataclass, field
from src.config import ChunkerConfig
from src.ingest import SourceDocument


@dataclass
class Chunk:
    chunk_id: str
    text: str
    doc_id: str
    doc_title: str
    doc_url: str
    doc_category: str
    chunk_idx: int
    char_start: int
    char_end: int
    metadata: dict = field(default_factory=dict)


def fixed_size_chunks(text: str, size: int, overlap: int) -> list[tuple[int, int, str]]:
    if size <= 0:
        raise ValueError("size must be positive")
    if overlap >= size:
        raise ValueError("overlap must be less than size")
    out = []
    step = size - overlap
    pos = 0
    n = len(text)
    while pos < n:
        end = min(pos + size, n)
        out.append((pos, end, text[pos:end]))
        if end == n:
            break
        pos += step
    return out


def recursive_chunks(text: str, size: int, overlap: int) -> list[tuple[int, int, str]]:
    separators = ["\n\n", "\n", ". ", " "]

    def split_recursive(segment: str, seps: list[str]) -> list[str]:
        if len(segment) <= size:
            return [segment]
        if not seps:
            return [segment[i:i + size] for i in range(0, len(segment), size - overlap)]
        sep = seps[0]
        parts = segment.split(sep)
        merged = []
        current = ""
        for part in parts:
            candidate = current + (sep if current else "") + part
            if len(candidate) <= size:
                current = candidate
            else:
                if current:
                    merged.append(current)
                if len(part) > size:
                    merged.extend(split_recursive(part, seps[1:]))
                    current = ""
                else:
                    current = part
        if current:
            merged.append(current)
        return merged

    pieces = split_recursive(text, separators)
    out = []
    pos = 0
    for piece in pieces:
        idx = text.find(piece, max(0, pos - overlap))
        if idx == -1:
            idx = pos
        end = idx + len(piece)
        out.append((idx, end, piece))
        pos = end
    return out


def sentence_chunks(text: str, size: int, overlap: int) -> list[tuple[int, int, str]]:
    import re
    sentences = re.split(r"(?<=[.!?])\s+", text)
    out = []
    current_parts = []
    current_len = 0
    pos = 0
    for sent in sentences:
        if current_len + len(sent) > size and current_parts:
            chunk = " ".join(current_parts)
            start = text.find(chunk, pos)
            if start == -1:
                start = pos
            end = start + len(chunk)
            out.append((start, end, chunk))
            tail_chars = 0
            keep = []
            for s in reversed(current_parts):
                if tail_chars + len(s) > overlap:
                    break
                keep.insert(0, s)
                tail_chars += len(s)
            current_parts = keep + [sent]
            current_len = sum(len(s) for s in current_parts)
            pos = end - tail_chars
        else:
            current_parts.append(sent)
            current_len += len(sent)
    if current_parts:
        chunk = " ".join(current_parts)
        start = text.find(chunk, pos)
        if start == -1:
            start = pos
        out.append((start, start + len(chunk), chunk))
    return out


def chunk_document(doc: SourceDocument, cfg: ChunkerConfig) -> list[Chunk]:
    text = doc.raw_text
    if cfg.type == "fixed":
        spans = fixed_size_chunks(text, cfg.size, cfg.overlap)
    elif cfg.type == "recursive":
        spans = recursive_chunks(text, cfg.size, cfg.overlap)
    elif cfg.type == "sentence":
        spans = sentence_chunks(text, cfg.size, cfg.overlap)
    else:
        raise ValueError(f"Unknown chunker type: {cfg.type}")

    chunks = []
    for i, (start, end, chunk_text) in enumerate(spans):
        chunks.append(Chunk(
            chunk_id=f"{doc.doc_id}_c{i:04d}",
            text=chunk_text,
            doc_id=doc.doc_id,
            doc_title=doc.title,
            doc_url=doc.url,
            doc_category=doc.category,
            chunk_idx=i,
            char_start=start,
            char_end=end,
            metadata={"chunker": cfg.type, "size": cfg.size, "overlap": cfg.overlap},
        ))
    return chunks


def chunk_all(docs: list[SourceDocument], cfg: ChunkerConfig) -> list[Chunk]:
    out = []
    for doc in docs:
        out.extend(chunk_document(doc, cfg))
    return out
