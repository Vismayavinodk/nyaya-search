"""
Unit tests for Phase 4:
- NetworkX citation graph construction
- In-degree, PageRank, HITS, and Node2Vec embeddings
- Strict zero-leakage test proving no test-query citation edges exist in the graph
- Feature-relevance correlation analysis
- GraphScoreInterpolator dev tuning and batch interpolation
"""

import os
import pytest
import numpy as np
import pandas as pd

from src.data.schema import LegalCase, LegalQuery
from src.graph.citation_graph import CitationGraphEngine
from src.graph.graph_retriever import GraphScoreInterpolator
from src.evaluation.eval_harness import EvaluationHarness


@pytest.fixture
def synthetic_graph_dataset():
    # Construct a directed citation network:
    # case_001 is a seminal Supreme Court landmark cited by case_002, case_003, case_004
    # case_002 cites case_001
    # case_003 cites case_001 and case_002
    # case_004 cites case_001
    corpus = {
        "case_001": LegalCase(
            doc_id="case_001",
            text="Landmark judgment on constitutional basic structure.",
            title="Kesavananda v. State",
            cited_cases=[],
            cited_statutes=["Article 368", "Article 13"]
        ),
        "case_002": LegalCase(
            doc_id="case_002",
            text="Subsequent judgment applying basic structure doctrine.",
            title="Minerva Mills v. Union of India",
            cited_cases=["case_001"],
            cited_statutes=["Article 368", "Article 31C"]
        ),
        "case_003": LegalCase(
            doc_id="case_003",
            text="Judgment analyzing judicial review and amendment power.",
            title="Waman Rao v. Union of India",
            cited_cases=["case_001", "case_002"],
            cited_statutes=["Article 368"]
        ),
        "case_004": LegalCase(
            doc_id="case_004",
            text="Recent application of precedent.",
            title="Kumar v. State",
            cited_cases=["case_001"],
            cited_statutes=["Article 21"]
        ),
    }

    # Queries: q_train, q_dev, q_test
    # Note: q_test has citations in its raw form to test leakage prevention
    queries = {
        "q_train": LegalQuery(
            query_id="q_train",
            text="Constitutional validity of constitutional amendments",
            cited_statutes=["Article 368"],
            metadata={"cited_cases": ["case_001"]}
        ),
        "q_dev": LegalQuery(
            query_id="q_dev",
            text="Scope of judicial review under Article 368",
            cited_statutes=["Article 368"],
            metadata={"cited_cases": ["case_001", "case_002"]}
        ),
        "q_test": LegalQuery(
            query_id="q_test",
            text="Basic structure limitations on Parliament",
            cited_statutes=["Article 368", "Article 13"],
            metadata={"cited_cases": ["case_001", "case_003"]}
        ),
    }

    qrels = {
        "q_train": {"case_001": 2, "case_002": 1, "case_003": 0, "case_004": 0},
        "q_dev": {"case_001": 2, "case_002": 2, "case_003": 1, "case_004": 0},
        "q_test": {"case_001": 2, "case_002": 1, "case_003": 2, "case_004": 0},
    }

    return corpus, queries, qrels


def test_citation_graph_centrality_and_hits(synthetic_graph_dataset):
    """Test in-degree, PageRank, HITS authority/hub scores, and PPR."""
    corpus, queries, _ = synthetic_graph_dataset
    graph = CitationGraphEngine(damping_factor=0.85)
    graph.build_graph(corpus=corpus, queries=queries, exclude_outgoing_for={"q_test"})

    # case_001 is cited by case_002, case_003, case_004 -> highest in-degree
    assert graph.get_in_degree("case_001") > graph.get_in_degree("case_002")
    assert graph.get_in_degree("case_002") > graph.get_in_degree("case_004")

    # Global PageRank: case_001 should have highest PageRank
    pr_001 = graph.get_pagerank("case_001")
    pr_004 = graph.get_pagerank("case_004")
    assert pr_001 > pr_004

    # HITS Authority: case_001 is the authority cited by multiple hubs
    auth_001 = graph.get_hits_authority("case_001")
    auth_004 = graph.get_hits_authority("case_004")
    assert auth_001 > auth_004

    # HITS Hub: case_003 cites both case_001 and case_002 -> higher hub score
    hub_003 = graph.get_hits_hub("case_003")
    hub_001 = graph.get_hits_hub("case_001")
    assert hub_003 > hub_001

    # Personalized PageRank teleporting to case_002
    ppr_res = graph.compute_personalized_pagerank(seed_nodes=["case_002"])
    assert "case_001" in ppr_res
    assert ppr_res["case_001"] > 0.0

    # Statute Jaccard
    jaccard = graph.compute_statute_jaccard(["Article 368", "Article 13"], "case_001")
    assert np.isclose(jaccard, 1.0)


def test_node2vec_embeddings(synthetic_graph_dataset):
    """Verify Node2Vec random walk embedding generation."""
    corpus, _, _ = synthetic_graph_dataset
    graph = CitationGraphEngine()
    graph.build_graph(corpus=corpus)

    embs = graph.compute_node2vec_embeddings(
        dimensions=16,
        walk_length=5,
        num_walks=5,
        window_size=2,
        seed=42
    )

    assert len(embs) > 0
    assert "case_001" in embs
    assert embs["case_001"].shape == (16,)
    assert graph.get_node2vec_embedding("case_001") is not None


def test_zero_test_query_leakage_in_graph(synthetic_graph_dataset):
    """
    CRITICAL LEAKAGE TEST:
    Proves that no test-query citation edges are present in the graph,
    out-degree of test queries is zero, and graph metrics on corpus cases
    are strictly invariant to the presence of test query citation metadata.
    """
    corpus, queries, _ = synthetic_graph_dataset
    test_qids = {"q_test"}

    # 1. Build graph with test query in exclude_outgoing_for
    graph_leakage_safe = CitationGraphEngine(damping_factor=0.85)
    graph_leakage_safe.build_graph(
        corpus=corpus,
        queries=queries,
        exclude_outgoing_for=test_qids
    )

    # 1a. Assert NO outgoing edge from any test query exists
    for u, v in graph_leakage_safe.graph.edges():
        assert u not in test_qids, f"LEAKAGE DETECTED: Found outgoing edge ({u} -> {v}) from test query!"

    # 1b. Assert test query node has out-degree of exactly 0
    assert graph_leakage_safe.graph.has_node("q_test")
    assert graph_leakage_safe.graph.out_degree("q_test") == 0

    # 2. Counterfactual invariance test:
    # Build a second graph where q_test citations are manually deleted before building
    queries_counterfactual = {
        qid: LegalQuery(
            query_id=q.query_id,
            text=q.text,
            cited_statutes=q.cited_statutes,
            metadata={} if qid in test_qids else dict(q.metadata)
        )
        for qid, q in queries.items()
    }
    graph_counterfactual = CitationGraphEngine(damping_factor=0.85)
    graph_counterfactual.build_graph(
        corpus=corpus,
        queries=queries_counterfactual,
        exclude_outgoing_for=test_qids
    )

    # Assert edge sets are identical
    assert set(graph_leakage_safe.graph.edges()) == set(graph_counterfactual.graph.edges())

    # Assert corpus in-degrees, PageRank, and HITS authority scores are identical
    for did in corpus:
        assert np.isclose(
            graph_leakage_safe.get_in_degree(did),
            graph_counterfactual.get_in_degree(did),
            atol=1e-7
        )
        assert np.isclose(
            graph_leakage_safe.get_pagerank(did),
            graph_counterfactual.get_pagerank(did),
            atol=1e-7
        )
        assert np.isclose(
            graph_leakage_safe.get_hits_authority(did),
            graph_counterfactual.get_hits_authority(did),
            atol=1e-7
        )


def test_feature_relevance_correlations(synthetic_graph_dataset):
    """Test correlation computation between graph features and relevance."""
    corpus, queries, qrels = synthetic_graph_dataset
    graph = CitationGraphEngine()
    graph.build_graph(corpus=corpus, queries=queries, exclude_outgoing_for={"q_test"})

    candidates = {
        "q_train": [("case_001", 10.0), ("case_002", 5.0), ("case_003", 2.0), ("case_004", 1.0)],
        "q_dev": [("case_001", 8.0), ("case_002", 9.0), ("case_003", 4.0), ("case_004", 1.0)],
    }

    df_corr = graph.compute_feature_relevance_correlations(candidates, qrels, queries=queries)
    assert not df_corr.empty
    assert "Feature" in df_corr.columns
    assert "Spearman_rho" in df_corr.columns
    assert "Pearson_r" in df_corr.columns
    assert "Statistically_Significant" in df_corr.columns
    assert len(df_corr) >= 4


def test_graph_score_interpolator_dev_tuning(synthetic_graph_dataset, tmp_path):
    """Test GraphScoreInterpolator dev tuning and TREC run export."""
    corpus, queries, qrels = synthetic_graph_dataset
    graph = CitationGraphEngine()
    graph.build_graph(corpus=corpus, queries=queries, exclude_outgoing_for={"q_test"})

    interpolator = GraphScoreInterpolator(graph_engine=graph)
    harness = EvaluationHarness()

    dev_base = {
        "q_dev": [("case_004", 10.0), ("case_002", 5.0), ("case_001", 2.0)]
    }

    # Interpolate single query with PageRank
    interpolated_dev = interpolator.interpolate_batch(
        dev_base,
        beta=0.4,
        signal_name="pagerank",
        top_k=3
    )
    assert len(interpolated_dev["q_dev"]) == 3
    # With beta=0.4, case_001 (landmark with high PageRank) should gain rank over case_004 (low PageRank)
    res_docs = [d for d, _ in interpolated_dev["q_dev"]]
    assert "case_001" in res_docs

    # Dev parameter sweep
    best_cfg, df_grid = interpolator.tune_interpolation_on_dev(
        dev_base_results=dev_base,
        dev_qrels=qrels,
        harness=harness,
        beta_candidates=[0.0, 0.2, 0.4],
        graph_signals=["pagerank", "hits_authority", "in_degree"],
        top_k=3
    )

    assert "best_beta" in best_cfg
    assert "best_signal" in best_cfg
    assert len(df_grid) == 3 * 3

    # TREC run export
    run_file = str(tmp_path / "graph_tuned.run")
    interpolator.save_run_file(interpolated_dev, run_file, run_name="GRAPH_TUNED")
    assert os.path.exists(run_file)
