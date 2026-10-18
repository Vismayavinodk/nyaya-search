"""
Unit tests for Phase 1: schemas, citation extraction, text preprocessing, and dataset loading.
"""

import os
import shutil
import pytest
from src.data.schema import LegalCase, LegalQuery, Qrel, CitationMatch
from src.data.citation_extractor import IndianLegalCitationExtractor
from src.data.legal_preprocessor import LegalPreprocessor
from src.data.dataset_loaders import DatasetLoader
from src.data.sample_generator import generate_synthetic_legal_dataset, save_dataset_to_disk


def test_citation_extractor():
    """Verify regex extraction of Indian legal citations and statutes."""
    extractor = IndianLegalCitationExtractor()
    sample_text = (
        "In Kesavananda Bharati (1973) 4 SCC 225, also reported in AIR 1973 SC 1461, "
        "the court examined Article 21 of the Constitution of India and Section 302 of the Indian Penal Code."
    )
    cases, statutes = extractor.get_citation_keys(sample_text)

    # Validate SCC extraction
    assert any("SCC" in c for c in cases), "SCC citation must be detected"
    # Validate AIR extraction
    assert any("AIR" in c for c in cases), "AIR citation must be detected"
    # Validate Constitutional article
    assert "CONST:ART_21" in statutes, "Article 21 must be extracted and normalized"
    # Validate Statutory section
    assert "STATUTE:IPC:SEC_302" in statutes, "Section 302 IPC must be extracted and normalized"


def test_legal_preprocessor():
    """Verify boilerplate removal, legal stopword filtering, and tokenization."""
    preprocessor = LegalPreprocessor(
        lower_case=True,
        remove_boilerplate=True,
        remove_stopwords=True
    )
    raw_judgment = (
        "IN THE SUPREME COURT OF INDIA CIVIL APPELLATE JURISDICTION "
        "The learned counsel for the appellant argued that the impugned judgment violates Article 21 and Section 302 IPC."
    )
    cleaned = preprocessor.clean_text(raw_judgment)
    # Boilerplate removed
    assert "CIVIL APPELLATE JURISDICTION" not in cleaned

    tokens = preprocessor.tokenize(raw_judgment)
    # Generic legal words like 'learned', 'counsel', 'appellant', 'impugned' should be stripped
    assert "learned" not in tokens
    assert "counsel" not in tokens
    assert "appellant" not in tokens
    # Essential legal tokens must be preserved
    assert "article" in tokens or "21" in tokens or "art_21" in tokens
    assert "section" in tokens or "302" in tokens or "sec_302" in tokens


def test_schema_roundtrip():
    """Verify dataclass serialization and deserialization."""
    case = LegalCase(
        doc_id="SC_TEST_01",
        title="Test Case v. Union of India",
        text="Sample judgment text",
        year=2024,
        cited_cases=["SC_1973_KESHAVANANDA"],
        cited_statutes=["CONST:ART_21"]
    )
    case_dict = case.to_dict()
    restored = LegalCase.from_dict(case_dict)
    assert restored.doc_id == case.doc_id
    assert restored.year == 2024
    assert restored.cited_cases == ["SC_1973_KESHAVANANDA"]

    qrel = Qrel(query_id="Q1", doc_id="D1", relevance=2)
    trec_line = qrel.to_trec_line()
    qrel_restored = Qrel.from_trec_line(trec_line)
    assert qrel_restored.query_id == "Q1"
    assert qrel_restored.doc_id == "D1"
    assert qrel_restored.relevance == 2


def test_sample_generation_and_loading(tmp_path):
    """Verify synthetic dataset generation, disk saving, and loading."""
    test_dir = str(tmp_path / "sample_test_data")
    cases, queries, qrels = generate_synthetic_legal_dataset()
    save_dataset_to_disk(cases, queries, qrels, test_dir)

    loader = DatasetLoader()
    loaded_cases, loaded_queries, loaded_qrels = loader.load_or_create_sample(sample_dir=test_dir)

    assert len(loaded_cases) == len(cases)
    assert len(loaded_queries) == len(queries)
    assert len(loaded_qrels) == len(qrels)

    train_ids, val_ids, test_ids = loader.split_queries(
        loaded_queries, loaded_qrels, train_ratio=0.5, val_ratio=0.25, test_ratio=0.25, seed=42
    )
    assert len(train_ids) + len(val_ids) + len(test_ids) == len(loaded_queries)
    assert len(set(train_ids).intersection(set(test_ids))) == 0


def test_il_pcr_data_loading():
    """Verify loading and schema normalization of real IL-PCR dataset files."""
    raw_dir = "data/raw/il_pcr"
    if not os.path.exists(raw_dir):
        pytest.skip("IL-PCR raw data directory not present")

    loader = DatasetLoader()
    proc_dir = "data/processed/il_pcr"
    corpus, queries, qrels = loader.load_il_pcr(raw_dir, processed_dir=proc_dir)

    assert len(corpus) == 3183, f"Expected 3183 candidate precedent cases, got {len(corpus)}"
    assert len(queries) == 6271, f"Expected 6271 queries, got {len(queries)}"
    assert len(qrels) > 15000, f"Expected >15000 qrels, got {len(qrels)}"

    # Check official split preservation
    train_ids, val_ids, test_ids = loader.split_queries(queries, qrels)
    assert len(train_ids) == 5017, f"Expected 5017 train queries, got {len(train_ids)}"
    assert len(val_ids) == 627, f"Expected 627 val queries, got {len(val_ids)}"
    assert len(test_ids) == 627, f"Expected 627 test queries, got {len(test_ids)}"

    # Check a sample query and case
    first_q = next(iter(queries.values()))
    assert first_q.facts, "Query facts must not be empty"
    assert first_q.text, "Query text must not be empty"
    assert "split" in first_q.metadata, "Query must carry official split metadata"

    first_case = next(iter(corpus.values()))
    assert first_case.doc_id, "Case doc_id must not be empty"
    assert len(first_case.text) > 100, "Case text must contain judgment content"
