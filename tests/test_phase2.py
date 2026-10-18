"""
Unit tests for Phase 2:
- TF-IDF baseline indexing, retrieval, and TREC run export
- BM25 indexing, Okapi vs BM25+ scoring, dynamic parameter updates
- EvaluationHarness parsing of TREC run and qrels files, per-query scores, and statistical significance tests
- BM25DevTuner grid search parameter optimization on dev split
"""

import os
import pytest
import numpy as np
import pandas as pd

from src.data.sample_generator import generate_synthetic_legal_dataset
from src.retrieval.tfidf_retriever import TFIDFRetriever
from src.retrieval.sparse_bm25 import BM25Retriever
from src.retrieval.bm25_tuner import BM25DevTuner
from src.evaluation.eval_harness import EvaluationHarness


@pytest.fixture
def sample_benchmark():
    cases, queries, qrels = generate_synthetic_legal_dataset()
    corpus = {c.doc_id: c for c in cases}
    q_dict = {q.query_id: q for q in queries}
    qrel_dict = {}
    for q in qrels:
        if q.query_id not in qrel_dict:
            qrel_dict[q.query_id] = {}
        qrel_dict[q.query_id][q.doc_id] = q.relevance
    return corpus, q_dict, qrels, qrel_dict


def test_tfidf_retriever(sample_benchmark, tmp_path):
    """Verify TF-IDF indexing, query search, and TREC run export."""
    corpus, queries, _, _ = sample_benchmark
    tfidf = TFIDFRetriever(sublinear_tf=True)
    tfidf.build_index(corpus)

    results = tfidf.batch_search(queries, top_k=5)
    assert len(results) == len(queries)
    for qid, scored_list in results.items():
        assert len(scored_list) <= 5
        assert all(isinstance(score, float) for _, score in scored_list)

    # Test run file export
    run_file = str(tmp_path / "tfidf_test.run")
    tfidf.save_run_file(results, run_file, run_name="TEST_TFIDF")
    assert os.path.exists(run_file)

    # Verify run file contents
    with open(run_file, "r") as f:
        lines = f.readlines()
        assert len(lines) > 0
        parts = lines[0].strip().split()
        assert len(parts) == 6
        assert parts[1] == "Q0"
        assert parts[5] == "TEST_TFIDF"


def test_bm25_retriever(sample_benchmark, tmp_path):
    """Verify BM25 vectorized engine, dynamic set_params, and Okapi vs BM25+."""
    corpus, queries, _, _ = sample_benchmark
    bm25 = BM25Retriever(algorithm="bm25_okapi", k1=1.5, b=0.75)
    bm25.build_index(corpus)

    # Search with default
    res_okapi = bm25.batch_search(queries, top_k=5)
    assert len(res_okapi) == len(queries)

    # Reconfigure parameters dynamically
    bm25.algorithm = "bm25_plus"
    bm25.set_params(k1=1.2, b=0.6, delta=0.5)
    res_plus = bm25.batch_search(queries, top_k=5)
    assert len(res_plus) == len(queries)

    # Test run export
    run_file = str(tmp_path / "bm25_test.run")
    bm25.save_run_file(res_plus, run_file, run_name="TEST_BM25")
    assert os.path.exists(run_file)


def test_evaluation_harness_trec_files(sample_benchmark, tmp_path):
    """Verify EvaluationHarness parsing, metric calculation, per-query export, and significance tests."""
    corpus, queries, qrels, qrel_dict = sample_benchmark
    harness = EvaluationHarness(relevance_threshold=1)

    # Generate synthetic run files for Model A and Model B
    bm25 = BM25Retriever(algorithm="bm25_okapi")
    bm25.build_index(corpus)
    res_a = bm25.batch_search(queries, top_k=5)

    tfidf = TFIDFRetriever()
    tfidf.build_index(corpus)
    res_b = tfidf.batch_search(queries, top_k=5)

    run_a_file = str(tmp_path / "model_a.run")
    run_b_file = str(tmp_path / "model_b.run")
    qrels_file = str(tmp_path / "qrels.tsv")
    per_query_a_file = str(tmp_path / "per_query_a.csv")

    bm25.save_run_file(res_a, run_a_file, run_name="MODEL_A")
    tfidf.save_run_file(res_b, run_b_file, run_name="MODEL_B")

    with open(qrels_file, "w") as f:
        for q in qrels:
            f.write(q.to_trec_line())

    # Evaluate Model A via harness
    metrics_a, df_a = harness.evaluate_run_file(
        run_file_path=run_a_file,
        qrels_file_path=qrels_file,
        per_query_output_path=per_query_a_file
    )
    assert "ndcg@10" in metrics_a
    assert "map" in metrics_a
    assert "mrr" in metrics_a
    assert os.path.exists(per_query_a_file)
    assert len(df_a) == len(queries)

    # Evaluate Model B
    metrics_b, df_b = harness.evaluate_run_file(run_b_file, qrels_file)

    # Test statistical significance computation
    sig_res = harness.compute_statistical_significance(df_b, df_a, metric="ndcg@10")
    assert "t_statistic" in sig_res
    assert "t_pvalue" in sig_res
    assert "relative_gain_percent" in sig_res


def test_bm25_dev_tuner(sample_benchmark, tmp_path):
    """Verify BM25DevTuner parameter sweep on dev queries."""
    corpus, queries, _, qrel_dict = sample_benchmark
    bm25 = BM25Retriever()
    bm25.build_index(corpus)

    tuner = BM25DevTuner(
        retriever=bm25,
        dev_queries=queries,
        dev_qrels=qrel_dict
    )

    best_params, df_grid = tuner.tune_bm25_okapi(
        k1_values=[1.2, 1.5],
        b_values=[0.6, 0.75],
        top_k=5
    )

    assert "k1" in best_params
    assert "b" in best_params
    assert len(df_grid) == 4
    assert df_grid.iloc[0]["ndcg@10"] >= df_grid.iloc[-1]["ndcg@10"]

    summary_file = str(tmp_path / "tuning_summary.csv")
    tuner.save_tuning_summary(df_grid, summary_file)
    assert os.path.exists(summary_file)
