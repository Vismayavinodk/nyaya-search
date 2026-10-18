"""
Standardized schemas for legal judgments, queries, citations, qrels, and citation graph edges.
Unified across IL-PCR, FIRE AILA, and custom legal corpora.
"""

from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional


@dataclass
class CitationMatch:
    """Represents an extracted citation from legal text."""
    raw_text: str
    normalized_form: str
    citation_type: str  # 'case_precedent', 'statute', or 'constitutional_article'
    reporter: Optional[str] = None  # e.g., 'AIR', 'SCC', 'SCR'
    year: Optional[int] = None
    start_char: int = -1
    end_char: int = -1

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CitationEdge:
    """Represents a directed citation edge from source case to target precedent or statute."""
    source_doc_id: str
    target_doc_id: str
    citation_type: str = "case_precedent"  # 'case_precedent' or 'statute'

    def to_tuple(self) -> tuple:
        return (self.source_doc_id, self.target_doc_id, self.citation_type)

    def to_tsv_line(self) -> str:
        return f"{self.source_doc_id}\t{self.target_doc_id}\t{self.citation_type}\n"

    @classmethod
    def from_tsv_line(cls, line: str) -> "CitationEdge":
        parts = line.strip().split("\t")
        if len(parts) >= 2:
            ctype = parts[2] if len(parts) > 2 else "case_precedent"
            return cls(source_doc_id=parts[0], target_doc_id=parts[1], citation_type=ctype)
        raise ValueError(f"Invalid citation edge TSV line: '{line}'")


@dataclass
class LegalCase:
    """
    Standardized representation of a candidate precedent case in the judicial corpus.
    Outputs: doc_id, text, cleaned_text, length.
    """
    doc_id: str
    text: str
    cleaned_text: str = ""
    length: int = 0
    title: str = ""
    court: Optional[str] = "Supreme Court of India"
    year: Optional[int] = None
    cited_cases: List[str] = field(default_factory=list)
    cited_statutes: List[str] = field(default_factory=list)
    ratio_summary: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.cleaned_text:
            self.cleaned_text = self.text
        if self.length <= 0 and self.cleaned_text:
            self.length = len(self.cleaned_text.split())
        if not self.title:
            self.title = self.doc_id

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LegalCase":
        text_val = data.get("text", "")
        cleaned = data.get("cleaned_text", text_val)
        length_val = int(data.get("length", len(cleaned.split()) if cleaned else 0))
        return cls(
            doc_id=str(data["doc_id"]),
            text=text_val,
            cleaned_text=cleaned,
            length=length_val,
            title=data.get("title", str(data["doc_id"])),
            court=data.get("court", "Supreme Court of India"),
            year=data.get("year"),
            cited_cases=list(data.get("cited_cases", [])),
            cited_statutes=list(data.get("cited_statutes", [])),
            ratio_summary=data.get("ratio_summary"),
            metadata=dict(data.get("metadata", {}))
        )


@dataclass
class LegalQuery:
    """
    Standardized representation of an input query case (facts, legal issues).
    """
    query_id: str
    text: str = ""
    cleaned_text: str = ""
    length: int = 0
    facts: str = ""
    issues: Optional[str] = None
    cited_statutes: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.text:
            if self.issues:
                self.text = f"{self.facts} Issues: {self.issues}".strip()
            else:
                self.text = self.facts
        if not self.cleaned_text:
            self.cleaned_text = self.text
        if self.length <= 0 and self.cleaned_text:
            self.length = len(self.cleaned_text.split())

    @property
    def full_query_text(self) -> str:
        """Combines facts and issues for retrieval indexing."""
        return self.text

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LegalQuery":
        text_val = data.get("text", data.get("facts", ""))
        cleaned = data.get("cleaned_text", text_val)
        length_val = int(data.get("length", len(cleaned.split()) if cleaned else 0))
        return cls(
            query_id=str(data["query_id"]),
            text=text_val,
            cleaned_text=cleaned,
            length=length_val,
            facts=data.get("facts", text_val),
            issues=data.get("issues"),
            cited_statutes=list(data.get("cited_statutes", [])),
            metadata=dict(data.get("metadata", {}))
        )


@dataclass
class Qrel:
    """
    Relevance judgment ground truth mapping a query to a relevant precedent case.
    Compatible with TREC qrels format: query_id\titeration\tdoc_id\trelevance
    """
    query_id: str
    doc_id: str
    relevance: int  # 0: non-relevant, 1: relevant, 2: highly relevant
    iteration: str = "0"

    def to_trec_line(self) -> str:
        """Formats the qrel as a standard TREC qrel line."""
        return f"{self.query_id}\t{self.iteration}\t{self.doc_id}\t{self.relevance}\n"

    @classmethod
    def from_trec_line(cls, line: str) -> "Qrel":
        """Parses a whitespace- or tab-delimited TREC qrel line."""
        parts = line.strip().split()
        if len(parts) == 4:
            return cls(
                query_id=parts[0],
                iteration=parts[1],
                doc_id=parts[2],
                relevance=int(parts[3])
            )
        elif len(parts) == 3:
            return cls(
                query_id=parts[0],
                iteration="0",
                doc_id=parts[1],
                relevance=int(parts[2])
            )
        raise ValueError(f"Invalid TREC qrel line format: '{line}'")
