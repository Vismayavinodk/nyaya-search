"""
Unit tests for Phase 5:
- Feature extraction using top-100 candidates from hybrid retrieval
- Core features: bm25_score, dense_score, pagerank, in_degree, doc_length
- Option A: LightGBM LambdaMART training on train split, tuning on dev split, feature importances
- Option B: Cross-encoder reranker
- Paired statistical significance testing (paired t-test and paired bootstrap test)
- Strict split isolation (tune on dev, test evaluated only once)
"""

import os
import pytest
import numpy as np
import pandas as pd

from src.data.schema import LegalCase, LegalQuery, Qrel
from src.graph.citation_graph import CitationGraphEngine
from src.rerank.feature_extractor import LegalFeatureExtractor
from src.rerank.lightgbm_ranker import LightGBMLambdaMART
from src.rerank.cross_encoder_reranker import CrossEncoderReranker
from src.evaluation.eval_harness import EvaluationHarness


@pytest.fixture
def synthetic_phase5_dataset():
    corpus = {
        f"case_{i:03d}": LegalCase(
            doc_id=f"case_{i:03d}",
            text=f"Legal precedent text for case {i} discussing constitutional law and statutory interpretations. " * (i + 1),
            title=f"Precedent Case {i}",
            court="Supreme Court of India" if i % 2 == 0 else "Delhi High Court",
            year=2000 + i,
            cited_cases=[f"case_{j:03d}" for j in range(max(0, i - 2), i)],
            cited_statutes=["Article 368", "Article 21"] if i % 2 == 0 else ["Section 302 IPC"]
        )
        for i in range(12)
    }

    # Train, Dev, Test queries
    queries = {
        "q_train_1": LegalQuery(query_id="q_train_1", text="Constitutional amendment basic structure", cited_statutes=["Article 368"]),
        "q_train_2": LegalQuery(query_id="q_train_2", text="Criminal intent under Indian Penal Code", cited_statutes=["Section 302 IPC"]),
        "q_dev_1": LegalQuery(query_id="q_dev_1", text="Scope of judicial review and amendment power", cited_statutes=["Article 368"]),
        "q_test_1": LegalQuery(query_id="q_test_1", text="Supreme Court guidelines on Article 21 and life", cited_statutes=["Article 21"]),
    }

    qrels = [
        # Train 1
        Qrel(query_id="q_train_1", doc_id="case_000", relevance=2),
        Qrel(query_id="q_train_1", doc_id="case_002", relevance=1),
        Qrel(query_id="q_train_1", doc_id="case_004", relevance=0),
        # Train 2
        Qrel(query_id="q_train_2", doc_id="case_001", relevance=2),
        Qrel(query_id="q_train_2", doc_id="case_003", relevance=1),
        # Dev 1
        Qrel(query_id="q_dev_1", doc_id="case_002", relevance=2),
        Qrel(query_id="q_dev_1", doc_id="case_000", relevance=1),
        # Test 1
        Qrel(query_id="q_test_1", doc_id="case_004", relevance=2),
        Qrel(query_id="q_test_1", doc_id="case_002", relevance=1),
    ]

    return corpus, queries, qrels


def test_feature_extractor_core_and_hybrid_pool(synthetic_phase5_dataset):
    """Test feature extractor with top-100 hybrid candidate pool and core feature names."""
    corpus, queries, _ = synthetic_phase5_dataset
    graph = CitationGraphEngine()
    graph.build_graph(corpus=corpus, queries=queries, exclude_outgoing_for={"q_test_1"})

    extractor = LegalFeatureExtractor(graph_engine=graph)

    # Verify CORE_FEATURE_NAMES are present
    assert "bm25_score" in extractor.CORE_FEATURE_NAMES
    assert "dense_score" in extractor.CORE_FEATURE_NAMES
    assert "pagerank" in extractor.CORE_FEATURE_NAMES
    assert "in_degree" in extractor.CORE_FEATURE_NAMES
    assert "doc_length" in extractor.CORE_FEATURE_NAMES

    # Synthetic hybrid candidate pool
    hybrid_candidates = {
        "q_train_1": [("case_000", 0.95), ("case_002", 0.85), ("case_004", 0.50), ("case_006", 0.30)],
        "q_dev_1": [("case_002", 0.90), ("case_000", 0.80), ("case_004", 0.40)],
        "q_test_1": [("case_004", 0.88), ("case_002", 0.70), ("case_006", 0.45)]
    }

    sparse_scores = {
        "q_train_1": [("case_000", 12.5), ("case_002", 9.0)],
        "q_dev_1": [("case_002", 11.0), ("case_000", 8.5)],
        "q_test_1": [("case_004", 10.0), ("case_002", 7.5)]
    }

    dense_scores = {
        "q_train_1": [("case_002", 0.88), ("case_000", 0.82)],
        "q_dev_1": [("case_000", 0.85), ("case_002", 0.83)],
        "q_test_1": [("case_004", 0.91), ("case_002", 0.79)]
    }

    df_feats = extractor.build_dataset_features_from_candidates(
        queries=queries,
        corpus=corpus,
        candidate_rankings=hybrid_candidates,
        sparse_results=sparse_scores,
        dense_results=dense_scores,
        top_k=100
    )

    assert not df_feats.empty
    for feat in extractor.CORE_FEATURE_NAMES:
        assert feat in df_feats.columns, f"Missing core feature: {feat}"
        assert not df_feats[feat].isna().any(), f"NaN in feature: {feat}"

    assert "hybrid_score" in df_feats.columns
    assert "hybrid_rank" in df_feats.columns
    assert len(df_feats) == 4 + 3 + 3


def test_lightgbm_lambdamart_train_tune_predict(synthetic_phase5_dataset, tmp_path):
    """Test LightGBM LambdaMART ranking model fitting, dev-tuning, and feature importance."""
    corpus, queries, qrels = synthetic_phase5_dataset
    graph = CitationGraphEngine()
    graph.build_graph(corpus=corpus, queries=queries, exclude_outgoing_for={"q_test_1"})
    extractor = LegalFeatureExtractor(graph_engine=graph)

    train_queries = {qid: queries[qid] for qid in ["q_train_1", "q_train_2"]}
    dev_queries = {qid: queries[qid] for qid in ["q_dev_1"]}
    test_queries = {qid: queries[qid] for qid in ["q_test_1"]}

    cands = {
        "q_train_1": [("case_000", 0.9), ("case_002", 0.8), ("case_004", 0.4)],
        "q_train_2": [("case_001", 0.95), ("case_003", 0.75), ("case_005", 0.3)],
        "q_dev_1": [("case_002", 0.9), ("case_000", 0.7), ("case_004", 0.3)],
        "q_test_1": [("case_004", 0.85), ("case_002", 0.65), ("case_000", 0.4)]
    }

    train_df = extractor.build_dataset_features_from_candidates(train_queries, corpus, cands, top_k=100)
    dev_df = extractor.build_dataset_features_from_candidates(dev_queries, corpus, cands, top_k=100)
    test_df = extractor.build_dataset_features_from_candidates(test_queries, corpus, cands, top_k=100)

    train_qrels = [q for q in qrels if q.query_id in train_queries]
    dev_qrels = [q for q in qrels if q.query_id in dev_queries]

    ranker = LightGBMLambdaMART(
        n_estimators=30,
        learning_rate=0.1,
        num_leaves=7,
        max_depth=3,
        min_child_samples=1,
        random_state=42,
        feature_cols=extractor.CORE_FEATURE_NAMES
    )

    ranker.fit(
        train_df=train_df,
        train_qrels=train_qrels,
        val_df=dev_df,
        val_qrels=dev_qrels,
        early_stopping_rounds=10
    )

    assert ranker.is_fitted
    assert ranker.best_iteration_ is not None

    # Predict rankings for test split
    test_rankings = ranker.predict_rankings(test_df, top_k=10)
    assert "q_test_1" in test_rankings
    assert len(test_rankings["q_test_1"]) == 3

    # Feature importance check
    importances = ranker.get_feature_importances(importance_type="gain")
    assert len(importances) == len(extractor.CORE_FEATURE_NAMES)
    for feat in extractor.CORE_FEATURE_NAMES:
        assert feat in importances

    # Model persistence check
    model_path = str(tmp_path / "lightgbm_lambdamart.pkl")
    ranker.save_model(model_path)
    assert os.path.exists(model_path)

    loaded_ranker = LightGBMLambdaMART()
    loaded_ranker.load_model(model_path)
    assert loaded_ranker.is_fitted
    loaded_preds = loaded_ranker.predict_rankings(test_df, top_k=10)
    assert loaded_preds["q_test_1"] == test_rankings["q_test_1"]


def test_cross_encoder_reranker(synthetic_phase5_dataset):
    """Verify Cross-Encoder reranker candidate scoring and batch reranking."""
    corpus, queries, _ = synthetic_phase5_dataset

    # Initialize cross encoder with lightweight model or fallback
    reranker = CrossEncoderReranker(
        model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
        max_length=64,
        max_doc_chars=300
    )

    candidate_rankings = {
        "q_dev_1": [("case_004", 0.5), ("case_002", 0.6), ("case_000", 0.7)]
    }

    reranked = reranker.rerank_batch(
        queries=queries,
        corpus=corpus,
        candidate_rankings=candidate_rankings,
        top_k=3
    )

    assert "q_dev_1" in reranked
    assert len(reranked["q_dev_1"]) == 3
    doc_ids = [d for d, _ in reranked["q_dev_1"]]
    assert set(doc_ids) == {"case_004", "case_002", "case_000"}


def test_paired_significance_bootstrap_and_ttest():
    """Verify paired t-test, Wilcoxon, and paired bootstrap significance calculations."""
    harness = EvaluationHarness()

    baseline_data = [
        {"query_id": f"q_{i}", "ndcg@10": 0.30 + (i % 5) * 0.02}
        for i in range(20)
    ]
    # Treatment with consistent improvement
    treatment_data = [
        {"query_id": f"q_{i}", "ndcg@10": 0.40 + (i % 5) * 0.02}
        for i in range(20)
    ]

    df_base = pd.DataFrame(baseline_data)
    df_treat = pd.DataFrame(treatment_data)

    sig = harness.compute_statistical_significance(df_base, df_treat, metric="ndcg@10")

    assert "t_statistic" in sig
    assert "t_pvalue" in sig
    assert "bootstrap_pvalue" in sig
    assert "bootstrap_ci_lower" in sig
    assert "bootstrap_ci_upper" in sig
    assert sig["is_significant_p05"] is True
    assert sig["bootstrap_pvalue"] < 0.05
    assert sig["bootstrap_ci_lower"] > 0.0
