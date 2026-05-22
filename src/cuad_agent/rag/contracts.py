"""Contract records and CUAD text adapter for sentence RAG."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ContractDocument:
    document_row_id: int
    title: str
    text: str
    source: str = "cuad_json"
    page_source: str = "unavailable_in_cuad_json"

    def to_dict(self) -> dict[str, Any]:
        return {
            "document_row_id": self.document_row_id,
            "title": self.title,
            "text": self.text,
            "source": self.source,
            "page_source": self.page_source,
        }


def contract_from_row(row: dict[str, Any]) -> ContractDocument:
    return ContractDocument(
        document_row_id=int(row.get("document_row_id", 0)),
        title=str(row.get("title") or row.get("document_id") or ""),
        text=str(row.get("context") or row.get("paragraphs.context") or ""),
    )


def contracts_from_lookup(
    contract_lookup: dict[int, dict[str, Any]],
) -> dict[int, ContractDocument]:
    return {
        int(document_row_id): contract_from_row(
            {"document_row_id": document_row_id, **row}
        )
        for document_row_id, row in contract_lookup.items()
    }
