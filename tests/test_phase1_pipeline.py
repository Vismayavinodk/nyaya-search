"""
Comprehensive sanity checks and verification tests for Phase 1 outputs:
- corpus.jsonl (doc_id, text, cleaned_text, length)
- queries.jsonl
- qrels.tsv (standard TREC format)
- citation_edges.tsv
- Zero test query citation leakage assertion
- Truncation & chunking logic verification
"""

import os
import json
import pytest
import networkx as nx

from src.data.schema import LegalCase, LegalQuery, Qrel, CitationEdge
from src.data.download_and_build import CorpusBuilder
from src.data.legal_preprocessor import LegalPreprocessor
from src.data.citation_extractor import IndianLegalCitationExtractor
from src.data.sample_generator import generate_synthetic_legal_dataset
from src.graph.citation_graph import CitationGraphEngine


@pytest.fixture
def build_sample_processed(tmp_path):
    """Builds a temporary standardized processed dataset directory."""
    output_dir = str(tmp_path / "processed_test")
    builder = CorpusBuilder()
    cases, queries, qrels = generate_synthetic_legal_dataset()
    paths = builder.process_and_export(cases, queries, qrels, output_dir)
    return output_dir, paths, cases, queries, qrels


def test_corpus_jsonl_schema(build_sample_processed):
    """Verify corpus.jsonl contains doc_id, text, cleaned_text, length."""
    _, paths, _, _, _ = build_sample_processed
    corpus_file = paths["corpus"]
    assert os.path.exists(corpus_file), f"corpus.jsonl missing at {corpus_file}"

    with open(corpus_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
        assert len(lines) > 0, "corpus.jsonl must not be empty"
        for line in lines:
            data = json.loads(line)
            assert "doc_id" in data and isinstance(data["doc_id"], str)
            assert "text" in data and isinstance(data["text"], str)
            assert "cleaned_text" in data and isinstance(data["cleaned_text"], str)
            assert "length" in data and isinstance(data["length"], int)
            assert data["length"] > 0, "Document word length must be positive"


def test_queries_jsonl_schema(build_sample_processed):
    """Verify queries.jsonl schema and length field."""
    _, paths, _, _, _ = build_sample_processed
    queries_file = paths["queries"]
    assert os.path.exists(queries_file)

    with open(queries_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
        assert len(lines) > 0
        for line in lines:
            data = json.loads(line)
            assert "query_id" in data
            assert "text" in data
            assert "cleaned_text" in data
            assert "length" in data
            assert data["length"] > 0


def test_qrels_trec_format(build_sample_processed):
    """Verify qrels.tsv adheres strictly to standard TREC 4-column format."""
    _, paths, _, _, _ = build_sample_processed
    qrels_file = paths["qrels"]
    assert os.path.exists(qrels_file)

    with open(qrels_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
        assert len(lines) > 0
        for line in lines:
            parts = line.strip().split()
            assert len(parts) == 4, f"TREC line must have 4 columns: '{line}'"
            qid, iter_id, did, rel = parts
            assert iter_id == "0", "Iteration field in TREC qrels should be '0'"
            assert rel.isdigit() and int(rel) >= 0, "Relevance must be a non-negative integer"


def test_citation_edges_schema(build_sample_processed):
    """Verify citation_edges.tsv structure."""
    _, paths, _, _, _ = build_sample_processed
    edges_file = paths["edges"]
    assert os.path.exists(edges_file)

    with open(edges_file, "r", encoding="utf-8") as f:
        header = f.readline().strip().split("\t")
        assert header == ["source_doc_id", "target_doc_id", "citation_type"]
        lines = f.readlines()
        assert len(lines) > 0
        for line in lines:
            parts = line.strip().split("\t")
            assert len(parts) == 3


def test_zero_leakage_for_test_queries():
    """
    CRITICAL TEST: Asserts that held-out test queries have zero outgoing edges
    in the feature graph to prevent evaluation data leakage.
    """
    cases, queries, _ = generate_synthetic_legal_dataset()
    corpus_dict = {c.doc_id: c for c in cases}
    query_dict = {q.query_id: q for q in queries}

    test_qids = {"QRY_01_DEATH_PENALTY", "QRY_02_PRIVACY_SURVEILLANCE"}

    # Simulate query cases that have citations
    query_dict["QRY_01_DEATH_PENALTY"].cited_statutes = ["STATUTE:IPC:SEC_302"]
    # Even if a test query had outgoing case citations:
    corpus_dict["QRY_01_DEATH_PENALTY"] = LegalCase(
        doc_id="QRY_01_DEATH_PENALTY",
        text="Sample query text",
        cited_cases=["SC_1980_BACHAN_SINGH"]
    )

    graph_engine = CitationGraphEngine()
    graph_engine.build_graph(
        corpus=corpus_dict,
        queries=query_dict,
        exclude_outgoing_for=test_qids
    )

    # Assert test query has node in graph but out_degree is strictly 0
    assert graph_engine.graph.has_node("QRY_01_DEATH_PENALTY")
    assert graph_engine.graph.out_degree("QRY_01_DEATH_PENALTY") == 0, (
        "LEAKAGE DETECTED: Test query must have zero outgoing edges in feature graph!"
    )


def test_head_tail_truncation_preserves_both_ends():
    """Verify dual-anchor head-tail truncation preserves both factual context and legal holding."""
    preprocessor = LegalPreprocessor()
    # Create synthetic long document of 1,000 words
    first_words = "FACT_START " * 100
    middle_words = "PROCEDURAL_DETAIL " * 800
    last_words = "RATIO_HOLDING_END " * 100
    long_doc = f"{first_words} {middle_words} {last_words}"

    truncated = preprocessor.head_tail_truncate(long_doc, max_words=200, head_ratio=0.5)

    # Verify word limit
    words = truncated.split()
    # 200 words + 1 break marker
    assert len(words) <= 201

    # Verify both ends are preserved
    assert "FACT_START" in truncated, "Head factual anchor must be retained"
    assert "RATIO_HOLDING_END" in truncated, "Tail ratio decidendi anchor must be retained"


def test_chunking_sliding_window():
    """Verify sliding-window chunking divides text into overlapping chunks."""
    preprocessor = LegalPreprocessor()
    text = "word " * 1000
    chunks = preprocessor.chunk_text(text, chunk_size=400, chunk_overlap=50)

    assert len(chunks) == 3
    for c in chunks:
        assert len(c.split()) <= 400
