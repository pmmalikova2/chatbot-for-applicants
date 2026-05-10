import re
from src.config import PreprocessingConfig


def normalize_whitespace(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def strip_page_numbers(text: str) -> str:
    lines = text.split("\n")
    cleaned = []
    for line in lines:
        stripped = line.strip()
        if stripped.isdigit() and len(stripped) <= 3:
            continue
        cleaned.append(line)
    return "\n".join(cleaned)


def strip_repeated_lines(text: str, min_repeats: int = 5) -> str:
    lines = text.split("\n")
    counts: dict[str, int] = {}
    for line in lines:
        stripped = line.strip()
        if len(stripped) < 10:
            continue
        counts[stripped] = counts.get(stripped, 0) + 1
    repeated = {line for line, c in counts.items() if c >= min_repeats}
    return "\n".join(line for line in lines if line.strip() not in repeated)


def apply_preprocessing(text: str, cfg: PreprocessingConfig) -> str:
    out = text
    if cfg.remove_page_numbers:
        out = strip_page_numbers(out)
    if cfg.remove_headers_footers:
        out = strip_repeated_lines(out)
    if cfg.remove_extra_whitespace:
        out = normalize_whitespace(out)
    if cfg.lowercase:
        out = out.lower()
    return out
