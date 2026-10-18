"""
Phase 6: Legal RAG Layer.
Separated cleanly from the core retrieval and ranking pipeline.
"""

from src.rag.schema import LegalQAExample, RAGPassage, RAGResponse
from src.rag.qa_dataset import RAGQADataset
from src.rag.prompts import format_rag_prompt, extract_cited_doc_ids, REFUSAL_STRING
from src.rag.rag_engine import LegalRAGEngine
from src.rag.evaluator import RAGEvaluator

__all__ = [
    "LegalQAExample",
    "RAGPassage",
    "RAGResponse",
    "RAGQADataset",
    "format_rag_prompt",
    "extract_cited_doc_ids",
    "REFUSAL_STRING",
    "LegalRAGEngine",
    "RAGEvaluator"
]
