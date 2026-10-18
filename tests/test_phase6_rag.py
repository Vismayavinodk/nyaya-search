"""
Unit tests for Phase 6:
- Legal RAG Layer (natural-language question in, short answer out, citing retrieved judgments)
- Refusal mechanism when evidence is weak or absent
- Self-written test set (30-50 questions) loading & validation
- Faithfulness and citation correctness evaluation
- Clean separation from core pipeline
"""

import pytest
import numpy as np
import pandas as pd

from src.rag.schema import LegalQAExample, RAGResponse
from src.rag.qa_dataset import RAGQADataset
from src.rag.prompts import extract_cited_doc_ids, REFUSAL_STRING
from src.rag.rag_engine import LegalRAGEngine
from src.rag.evaluator import RAGEvaluator


def test_qa_dataset_loading_and_schema():
    """Verify loading and schema of the 35-question legal QA benchmark."""
    benchmark = RAGQADataset.load_benchmark()

    # Must contain 30-50 questions
    assert 30 <= len(benchmark) <= 50
    assert len(benchmark) == 35

    answerable = [q for q in benchmark if q.is_answerable]
    unanswerable = [q for q in benchmark if not q.is_answerable]

    assert len(answerable) >= 20
    assert len(unanswerable) >= 8

    # Validate schema fields
    for ex in benchmark:
        assert ex.question_id.startswith("QA_")
        assert len(ex.question) > 15
        assert len(ex.gold_answer) > 5
        if ex.is_answerable:
            assert len(ex.gold_doc_ids) > 0


def test_citation_parser_regex():
    """Verify citation extraction across bracketed formats."""
    text_1 = "Parliament's amending power is limited by the basic structure doctrine. [Doc: case_kesavananda]"
    text_2 = "Privacy is a fundamental right. [Doc ID: case_puttaswamy] and [case_maneka_gandhi]."
    text_3 = "The practice was declared unconstitutional. [Precedent: case_shayara_bano]"

    cites_1 = extract_cited_doc_ids(text_1)
    cites_2 = extract_cited_doc_ids(text_2)
    cites_3 = extract_cited_doc_ids(text_3)

    assert "case_kesavananda" in cites_1
    assert "case_puttaswamy" in cites_2
    assert "case_maneka_gandhi" in cites_2
    assert "case_shayara_bano" in cites_3


def test_grounded_answer_generation_with_citations():
    """Verify question answering with grounded context and citations."""
    engine = LegalRAGEngine(use_open_model=False, min_evidence_threshold=0.30)

    question = "Does Parliament have unlimited amending power under Article 368?"
    passages = [
        {
            "doc_id": "case_kesavananda",
            "title": "Kesavananda Bharati v. State of Kerala",
            "text": "The Supreme Court held that Article 368 does not enable Parliament to alter the basic structure or framework of the Constitution.",
            "score": 0.95
        }
    ]

    response = engine.answer_question(
        question_id="QA_001",
        question=question,
        retrieved_passages=passages
    )

    assert not response.is_refusal
    assert len(response.answer) > 20
    assert "case_kesavananda" in response.cited_doc_ids
    assert response.evidence_score >= 0.30


def test_refusal_guardrail_on_weak_or_absent_evidence():
    """Verify that the system strictly refuses when evidence is weak or unanswerable."""
    engine = LegalRAGEngine(use_open_model=False, min_evidence_threshold=0.40)

    # 1. Empty evidence
    resp_empty = engine.answer_question(
        question_id="QA_REFUSAL_001",
        question="What is the speed limit on Mars?",
        retrieved_passages=[]
    )
    assert resp_empty.is_refusal
    assert resp_empty.answer == REFUSAL_STRING
    assert len(resp_empty.cited_doc_ids) == 0

    # 2. Weak/Irrelevant evidence
    weak_passages = [
        {
            "doc_id": "case_unrelated",
            "title": "State Excise Department v. Distillery",
            "text": "The taxation of denatured industrial spirit falls under the legislative jurisdiction of the state.",
            "score": 0.15
        }
    ]
    resp_weak = engine.answer_question(
        question_id="QA_REFUSAL_002",
        question="Can citizens own private nuclear attack submarines under Article 14?",
        retrieved_passages=weak_passages
    )
    assert resp_weak.is_refusal
    assert resp_weak.answer == REFUSAL_STRING
    assert len(resp_weak.cited_doc_ids) == 0


def test_rag_evaluator_metrics():
    """Verify faithfulness, citation correctness, and refusal metrics."""
    evaluator = RAGEvaluator()

    ex_answerable = LegalQAExample(
        question_id="QA_001",
        question="Is privacy a fundamental right?",
        gold_answer="Yes, privacy is a fundamental right under Article 21.",
        gold_doc_ids=["case_puttaswamy"],
        is_answerable=True
    )
    ex_unanswerable = LegalQAExample(
        question_id="QA_002",
        question="What is the traffic penalty on Mars?",
        gold_answer="REFUSAL",
        gold_doc_ids=[],
        is_answerable=False
    )

    # 1. Good answer
    resp_good = RAGResponse(
        question_id="QA_001",
        question="Is privacy a fundamental right?",
        answer="The Supreme Court held privacy is a fundamental right under Article 21. [Doc: case_puttaswamy]",
        cited_doc_ids=["case_puttaswamy"],
        is_refusal=False,
        evidence_score=0.85,
        retrieved_passages=[{"doc_id": "case_puttaswamy", "text": "Privacy is a fundamental right under Article 21."}]
    )

    # 2. Correct refusal
    resp_refused = RAGResponse(
        question_id="QA_002",
        question="What is the traffic penalty on Mars?",
        answer=REFUSAL_STRING,
        cited_doc_ids=[],
        is_refusal=True,
        evidence_score=0.10,
        retrieved_passages=[]
    )

    summary, df_results = evaluator.evaluate_benchmark(
        examples=[ex_answerable, ex_unanswerable],
        responses=[resp_good, resp_refused]
    )

    assert summary["total_questions"] == 2
    assert summary["mean_faithfulness"] > 0.70
    assert summary["citation_precision"] == 1.0
    assert summary["citation_recall"] == 1.0
    assert summary["refusal_accuracy"] == 1.0
    assert summary["refusal_precision"] == 1.0
    assert summary["refusal_recall"] == 1.0
