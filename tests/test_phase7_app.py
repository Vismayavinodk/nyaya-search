"""
Test suite for Phase 7 Streamlit demonstration and explainability logic.
"""

import os
import pytest
from src.data.dataset_loaders import DatasetLoader
from src.retrieval.sparse_bm25 import BM25Retriever
from src.graph.citation_graph import CitationGraphEngine
from src.rag.rag_engine import LegalRAGEngine
from src.rag.schema import RAGResponse
from src.data.schema import LegalQuery


def test_app_resources_and_indexing():
    """Verifies that the demo resources load and index seamlessly."""
    corpus_path = "data/sample/corpus.jsonl"
    queries_path = "data/sample/queries.jsonl"
    qrels_path = "data/sample/qrels.tsv"

    loader = DatasetLoader()
    corpus, queries, qrels = loader.load_standard_jsonl(corpus_path, queries_path, qrels_path)

    assert len(corpus) >= 10
    assert "SC_1973_KESHAVANANDA" in corpus
    assert "SC_2017_PUTTASWAMY" in corpus

    # Verify BM25 indexing
    bm25 = BM25Retriever(algorithm="bm25_plus", k1=2.2, b=0.85, delta=0.25, backend="rank_bm25")
    bm25.build_index(corpus)

    test_q = LegalQuery(query_id="Q_TEST", text="basic structure amendment article 368")
    results = bm25.retrieve_for_query(test_q, top_k=5)
    assert len(results) > 0
    top_doc_id = results[0][0]
    assert top_doc_id == "SC_1973_KESHAVANANDA"


def test_app_attribution_and_graph_authority():
    """Verifies that component-level scores and why-panel logic compute accurately."""
    corpus_path = "data/sample/corpus.jsonl"
    queries_path = "data/sample/queries.jsonl"
    qrels_path = "data/sample/qrels.tsv"

    loader = DatasetLoader()
    corpus, queries, qrels = loader.load_standard_jsonl(corpus_path, queries_path, qrels_path)

    graph_engine = CitationGraphEngine()
    graph_engine.build_graph(corpus)

    assert "SC_1973_KESHAVANANDA" in graph_engine.in_degree_cache
    assert graph_engine.in_degree_cache["SC_1973_KESHAVANANDA"] >= 0.1
    assert graph_engine.pagerank_cache["SC_1973_KESHAVANANDA"] > 0.0


def test_app_rag_grounding_and_refusal_execution():
    """Verifies that RAG answer generation and refusal guardrail operate correctly in demo workflow."""
    rag_engine = LegalRAGEngine(min_evidence_threshold=0.30, use_open_model=False)

    passages = [
        {
            "doc_id": "SC_1973_KESHAVANANDA",
            "title": "Kesavananda Bharati v. State of Kerala",
            "text": "The basic structure doctrine is the bedrock of constitutional jurisprudence in India. Article 368 amending power is limited.",
            "score": 0.85
        }
    ]

    # Valid supported question
    resp = rag_engine.answer_question(
        "QA_001",
        "Does Article 368 allow altering the basic structure of the Constitution?",
        passages
    )
    assert not resp.is_refusal
    assert "SC_1973_KESHAVANANDA" in resp.cited_doc_ids or len(resp.cited_doc_ids) > 0

    # Weak/adversarial question
    weak_resp = rag_engine.answer_question(
        "QA_002",
        "What is the statutory speed limit for hovercrafts operating on Mars under Motor Vehicles Act?",
        passages
    )
    assert weak_resp.is_refusal
    assert "cannot answer" in weak_resp.answer.lower()
