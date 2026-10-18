"""
Unit tests for retrieval modules, citation graph, fusion, reranking, and IR metrics.
"""

import pytest
import numpy as np
from src.data.sample_generator import generate_synthetic_legal_dataset
from src.retrieval.sparse_bm25 import BM25Retriever
from src.retrieval.hybrid_fusion import HybridFusionRetriever
from src.graph.citation_graph import CitationGraphEngine
from src.rerank.feature_extractor import LegalFeatureExtractor
from src.rerank.ranker import LegalReranker
from src.evaluation.metrics import IRMetricsEvaluator
from src.data.schema import Qrel


def test_bm25_retrieval():
    """Verify BM25 indexing and query retrieval on landmark cases."""
    cases, queries, _ = generate_synthetic_legal_dataset()
    corpus = {c.doc_id: c for c in cases}
    bm25 = BM25Retriever(algorithm="bm25_plus")
    bm25.build_index(corpus)

    death_penalty_q = [q for q in queries if "DEATH" in q.query_id][0]
    results = bm25.retrieve_for_query(death_penalty_q, top_k=3)
    retrieved_ids = [did for did, _ in results]

    # Bachan Singh or Machhi Singh should be in top 3
    assert "SC_1980_BACHAN_SINGH" in retrieved_ids or "SC_1983_MACHHI_SINGH" in retrieved_ids


def test_citation_graph_metrics():
    """Verify in-degree, PageRank, and Personalized PageRank."""
    cases, _, _ = generate_synthetic_legal_dataset()
    corpus = {c.doc_id: c for c in cases}
    graph = CitationGraphEngine()
    graph.build_graph(corpus)

    # Kesavananda Bharati and Maneka Gandhi are heavily cited
    kb_in = graph.get_in_degree("SC_1973_KESHAVANANDA")
    kb_pr = graph.get_pagerank("SC_1973_KESHAVANANDA")
    assert kb_in > 0.0, "Kesavananda Bharati should have positive in-degree"
    assert kb_pr > 0.0, "Kesavananda Bharati should have positive PageRank"

    # Test PPR seeded from Bachan Singh
    ppr = graph.compute_personalized_pagerank(["SC_1980_BACHAN_SINGH"])
    assert "SC_1980_BACHAN_SINGH" in ppr


def test_hybrid_fusion():
    """Verify Reciprocal Rank Fusion combines two ranked lists."""
    fusion = HybridFusionRetriever(method="rrf", rrf_k=60)
    sparse_res = [("doc_A", 10.0), ("doc_B", 8.0), ("doc_C", 5.0)]
    dense_res = [("doc_B", 0.95), ("doc_A", 0.85), ("doc_D", 0.70)]

    fused = fusion.reciprocal_rank_fusion(sparse_res, dense_res, top_k=3)
    fused_docs = [did for did, _ in fused]

    # doc_A and doc_B appear in both, so they must rank higher than single-list docs
    assert set(fused_docs[:2]) == {"doc_A", "doc_B"}


def test_ir_metrics_numerical_correctness():
    """Verify exact mathematical values of nDCG, MAP, and MRR."""
    evaluator = IRMetricsEvaluator(relevance_threshold=1)

    retrieved = ["doc_1", "doc_2", "doc_3", "doc_4"]
    ground_truth = {
        "doc_1": 2,
        "doc_2": 1,
        "doc_3": 0,
        "doc_4": 0
    }

    metrics = evaluator.evaluate_query(retrieved, ground_truth)
    assert metrics["ndcg@5"] == 1.0, "Perfect ranking must yield nDCG=1.0"
    assert metrics["map"] == 1.0, "Perfect precision must yield MAP=1.0"
    assert metrics["mrr"] == 1.0, "First position hit must yield MRR=1.0"
    assert metrics["recall@5"] == 1.0, "Both relevant docs in top 5 yields Recall=1.0"


def test_reranker_pipeline():
    """Verify feature extraction and LambdaMART model training."""
    cases, queries, qrels = generate_synthetic_legal_dataset()
    corpus = {c.doc_id: c for c in cases}
    query_dict = {q.query_id: q for q in queries}

    graph = CitationGraphEngine()
    graph.build_graph(corpus)
    extractor = LegalFeatureExtractor(graph_engine=graph)

    bm25 = BM25Retriever()
    bm25.build_index(corpus)
    sparse_res = bm25.batch_search(query_dict, top_k=10)

    # Dummy dense results for fast unit test
    dense_res = {qid: [(c, 0.5) for c in list(corpus.keys())[:10]] for qid in query_dict}

    df = extractor.build_dataset_features(
        query_dict, corpus, sparse_res, dense_res, candidate_pool_size=10
    )
    assert not df.empty
    assert "bm25_score" in df.columns
    assert "pagerank" in df.columns

    reranker = LegalReranker(n_estimators=10, max_depth=3)
    reranker.fit(df, qrels)
    assert reranker.is_fitted

    preds = reranker.predict_rankings(df, top_k=5)
    assert len(preds) == len(query_dict)
