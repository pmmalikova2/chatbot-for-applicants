from dataclasses import dataclass
from pathlib import Path
from typing import Union
import csv
import pypdf


@dataclass
class SourceDocument:
    doc_id: str
    filename: str
    url: str
    title: str
    category: str
    raw_text: str = ""
    pages: list[str] = None

    def __post_init__(self):
        if self.pages is None:
            self.pages = []


def load_sources(sources_path: Union[str, Path]) -> list[SourceDocument]:
    docs = []
    with open(sources_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            docs.append(SourceDocument(
                doc_id=row["doc_id"],
                filename=row["filename"],
                url=row["url"],
                title=row["title"],
                category=row["category"],
            ))
    return docs


def extract_pdf_text(pdf_path: Union[str, Path]) -> tuple[str, list[str]]:
    pdf_path = Path(pdf_path)
    reader = pypdf.PdfReader(str(pdf_path))
    pages = []
    for page in reader.pages:
        text = page.extract_text() or ""
        pages.append(text)
    full_text = "\n\n".join(pages)
    return full_text, pages


def ingest_all(
    sources_path: Union[str, Path] = "data/sources.csv",
    pdfs_dir: Union[str, Path] = "data/raw_pdfs",
) -> list[SourceDocument]:
    docs = load_sources(sources_path)
    pdfs_dir = Path(pdfs_dir)
    for doc in docs:
        pdf_path = pdfs_dir / doc.filename
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        full_text, pages = extract_pdf_text(pdf_path)
        doc.raw_text = full_text
        doc.pages = pages
    return docs
