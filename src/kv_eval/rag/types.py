"""Types shared by ingestion, Qdrant indexing and retrieval.

LayoutElement is an ingestion-internal structure. It never reaches the Qdrant
payload or the retriever output, both of which keep their existing contract.
"""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator


class TextBlock(BaseModel):
    text: str
    bbox: tuple[float, float, float, float]


ElementType = Literal[
    "title",
    "front_matter",
    "abstract",
    "section_title",
    "paragraph",
    "equation",
    "table",
    "table_caption",
    "figure_caption",
    "other",
]

# Element kinds that carry no retrievable meaning on their own and are dropped
# before chunking: author lists, affiliations, figure axis labels, page numbers.
NON_BODY_ELEMENT_TYPES: frozenset[str] = frozenset({"front_matter", "other"})


class LayoutElement(BaseModel):
    page: int
    text: str
    bbox: tuple[float, float, float, float]
    element_type: ElementType = "paragraph"
    font_size: float = 0.0
    is_bold: bool = False


class PageText(BaseModel):
    doc_id: str
    page: int
    text: str
    blocks: list[TextBlock] = Field(default_factory=list)
    elements: list[LayoutElement] = Field(default_factory=list)


class DocumentRecord(BaseModel):
    doc_id: str
    file: str
    title: str
    tech_id: str
    camp: str
    doc_type: str
    page_start: int = 1
    page_end: int | None = None

    @field_validator("tech_id")
    @classmethod
    def normalize_tech_id(cls, value: str) -> str:
        return value.lower()


class Manifest(BaseModel):
    documents: list[DocumentRecord]


class DocumentChunk(BaseModel):
    text: str
    doc_id: str
    title: str
    page: int
    tech_id: str
    camp: str
    doc_type: str
    chunk_index: int
    chunk_id: str | None = None

    def payload(self) -> dict[str, str | int]:
        return {
            "chunk_id": self.chunk_id or f"{self.doc_id}:p{self.page}:c{self.chunk_index}",
            "doc_id": self.doc_id,
            "title": self.title,
            "page": self.page,
            "tech_id": self.tech_id,
            "camp": self.camp,
            "doc_type": self.doc_type,
            "chunk_index": self.chunk_index,
            "text": self.text,
        }


class RetrievedChunk(BaseModel):
    text: str
    chunk_id: str | None = None
    doc_id: str
    page: int
    tech_id: str
    camp: str
    doc_type: str
    chunk_index: int
    score: float


def resolve_pdf_path(pdf_root: Path, file_name: str) -> Path:
    path = Path(file_name)
    if path.is_absolute():
        return path
    return pdf_root / path
