"""
Unit tests for Phase 3:
- FAISS vector index (build, search, serialize, reload)
- DenseLegalRetriever chunking, max-pooling vs. mean-pooling, query encoding
- Checkpoint resumption to protect long Kaggle/Colab sessions
- Hybrid fusion (RRF and convex score combination)
- Dev-only parameter tuning for hybrid weights and RRF-k
"""

import os
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import pytest
import numpy as np
import pandas as pd

from src.data.schema import LegalCase, LegalQuery
from src.retrieval.faiss_index import LegalFAISSIndex
from src.retrieval.dense_encoder import DenseLegalRetriever
from src.retrieval.hybrid_fusion import HybridFusionRetriever
from src.evaluation.eval_harness import EvaluationHarness


@pytest.fixture
def sample_corpus_and_queries():
    cases = [
        LegalCase(
            doc_id="case_001",
            text="The Supreme Court observed that murder under Section 302 IPC requires clear intention. " * 50,
            title="State v. Kumar",
            cited_statutes=["IPC Section 302"]
        ),
        LegalCase(
            doc_id="case_002",
            text="Dishonour of cheque under Section 138 of Negotiable Instruments Act requires statutory notice. " * 50,
            title="Sharma v. Verma",
            cited_statutes=["NI Act Section 138"]
        ),
        LegalCase(
            doc_id="case_003",
            text="Arbitration and Conciliation Act Section 34 provides limited scope for judicial intervention in awards. " * 50,
            title="Tech Corp v. Builders Ltd",
            cited_statutes=["Arbitration Act Section 34"]
        ),
    ]
    queries = [
        LegalQuery(query_id="q1", text="Section 302 Indian Penal Code murder intention requirement", cited_statutes=["IPC Section 302"]),
        LegalQuery(query_id="q2", text="Negotiable Instruments Act Section 138 legal notice dishonour", cited_statutes=["NI Act Section 138"]),
    ]
    qrels = {
        "q1": {"case_001": 2, "case_002": 0, "case_003": 0},
        "q2": {"case_002": 2, "case_001": 0, "case_003": 0},
    }
    corpus_dict = {c.doc_id: c for c in cases}
    query_dict = {q.query_id: q for q in queries}
    return corpus_dict, query_dict, qrels


def test_faiss_index_lifecycle(tmp_path):
    """Test FAISS IndexFlatIP building, searching, saving, and loading."""
    dim = 32
    num_docs = 10
    np.random.seed(42)

    # Generate synthetic normalized embeddings
    embs = np.random.randn(num_docs, dim).astype(np.float32)
    embs /= np.linalg.norm(embs, axis=1, keepdims=True)
    doc_ids = [f"doc_{i:03d}" for i in range(num_docs)]

    index = LegalFAISSIndex(dimension=dim, index_type="flat_ip", use_gpu=False)
    index.build(embs, doc_ids)
    assert index.is_built
    assert index.index.ntotal == num_docs

    # Query search
    q_vec = embs[2:3]  # Query identical to doc_002
    scores, indices = index.search(q_vec, top_k=3)
    assert indices.shape == (1, 3)
    # The nearest neighbor should be doc_002 itself with cosine similarity ~ 1.0
    assert indices[0, 0] == 2
    assert np.isclose(scores[0, 0], 1.0, atol=1e-5)

    # Search queries mapping
    res_dict = index.search_queries(q_vec, query_ids=["test_q"], top_k=3)
    assert "test_q" in res_dict
    assert res_dict["test_q"][0][0] == "doc_002"

    # Save and Load roundtrip
    index_file = str(tmp_path / "test.faiss")
    index.save(index_file)
    assert os.path.exists(index_file)
    assert os.path.exists(index_file + ".meta")

    loaded_index = LegalFAISSIndex(dimension=dim, index_type="flat_ip", use_gpu=False)
    loaded_index.load(index_file)
    assert loaded_index.is_built
    assert loaded_index.doc_ids == doc_ids

    scores_loaded, indices_loaded = loaded_index.search(q_vec, top_k=3)
    np.testing.assert_array_almost_equal(scores, scores_loaded)
    np.testing.assert_array_equal(indices, indices_loaded)


def test_dense_retriever_chunking_and_pooling(sample_corpus_and_queries, tmp_path):
    """Test document chunking, max-pooling vs mean-pooling, and query encoding."""
    corpus, queries, _ = sample_corpus_and_queries
    case = corpus["case_001"]

    retriever_max = DenseLegalRetriever(
        model_name="BAAI/bge-small-en-v1.5",
        chunk_size=50,
        chunk_overlap=10,
        max_chunks_per_case=4,
        pooling_strategy="max",
        device="cpu"
    )

    # Test chunking
    chunks = retriever_max._chunk_text(case.text)
    assert len(chunks) <= 4
    assert len(chunks) > 1

    # Test case encoding with max pooling
    emb_max = retriever_max.encode_case(case)
    assert emb_max.ndim == 1
    assert emb_max.shape[0] == retriever_max.embedding_dimension
    assert np.isclose(np.linalg.norm(emb_max), 1.0, atol=1e-4)

    # Test case encoding with mean pooling
    retriever_mean = DenseLegalRetriever(
        model_name="BAAI/bge-small-en-v1.5",
        chunk_size=50,
        chunk_overlap=10,
        max_chunks_per_case=4,
        pooling_strategy="mean",
        device="cpu"
    )
    emb_mean = retriever_mean.encode_case(case)
    assert np.isclose(np.linalg.norm(emb_mean), 1.0, atol=1e-4)

    # Max pooling and mean pooling should yield distinct representations
    assert not np.allclose(emb_max, emb_mean, atol=1e-3)

    # Test query encoding
    q_emb = retriever_max.encode_query(queries["q1"])
    assert q_emb.shape == (retriever_max.embedding_dimension,)
    assert np.isclose(np.linalg.norm(q_emb), 1.0, atol=1e-4)


def test_dense_retriever_checkpoint_resumption(sample_corpus_and_queries, tmp_path):
    """Verify that DenseLegalRetriever can resume indexing from a partial checkpoint."""
    corpus, queries, _ = sample_corpus_and_queries
    ckpt_dir = str(tmp_path / "checkpoints")
    cache_path = str(tmp_path / "cache" / "test_corpus.npz")
    os.makedirs(ckpt_dir, exist_ok=True)

    retriever = DenseLegalRetriever(
        model_name="BAAI/bge-small-en-v1.5",
        device="cpu",
        use_faiss=True
    )

    # Simulate an interrupted session by pre-saving a checkpoint with case_001 only
    emb_001 = retriever.encode_case(corpus["case_001"])
    ckpt_path = os.path.join(ckpt_dir, "test_corpus_ckpt.npz")
    np.savez_compressed(
        ckpt_path,
        doc_ids=np.array(["case_001"]),
        embeddings=np.array([emb_001])
    )
    assert os.path.exists(ckpt_path)

    # Build index: should detect checkpoint, resume, and encode remaining 2 cases
    retriever.build_index(
        corpus=corpus,
        cache_path=cache_path,
        checkpoint_dir=ckpt_dir,
        checkpoint_interval=1
    )

    assert retriever.corpus_embeddings is not None
    assert retriever.corpus_embeddings.shape == (3, retriever.embedding_dimension)
    assert retriever.doc_ids == list(corpus.keys())
    assert os.path.exists(cache_path)
    # Checkpoint should be cleaned up after successful final cache write
    assert not os.path.exists(ckpt_path)

    # Verify search operates correctly
    search_res = retriever.batch_search(queries, top_k=3)
    assert len(search_res) == 2
    assert len(search_res["q1"]) == 3


def test_hybrid_fusion_methods():
    """Verify Reciprocal Rank Fusion and Convex Combination logic."""
    fusion = HybridFusionRetriever(method="rrf", rrf_k=60)

    sparse_rank = [("docA", 15.0), ("docB", 10.0), ("docC", 5.0)]
    dense_rank = [("docB", 0.95), ("docA", 0.85), ("docD", 0.70)]

    # 1. Test RRF
    rrf_res = fusion.reciprocal_rank_fusion(sparse_rank, dense_rank, top_k=4, rrf_k=60)
    rrf_dict = dict(rrf_res)
    # docA rank_sparse=1, rank_dense=2 -> 1/61 + 1/62 = 0.01639 + 0.01613 = 0.03252
    # docB rank_sparse=2, rank_dense=1 -> 1/62 + 1/61 = 0.03252 (equal sum)
    # docC rank_sparse=3, rank_dense=None -> 1/63 = 0.01587
    # docD rank_sparse=None, rank_dense=3 -> 1/63 = 0.01587
    assert set(rrf_dict.keys()) == {"docA", "docB", "docC", "docD"}
    assert rrf_dict["docA"] > rrf_dict["docC"]
    assert rrf_dict["docB"] > rrf_dict["docD"]

    # 2. Test Convex Combination
    # alpha=1.0 -> purely dense ranking
    convex_dense = fusion.convex_combination(sparse_rank, dense_rank, top_k=4, alpha=1.0)
    assert convex_dense[0][0] == "docB"  # Highest dense score

    # alpha=0.0 -> purely sparse ranking
    convex_sparse = fusion.convex_combination(sparse_rank, dense_rank, top_k=4, alpha=0.0)
    assert convex_sparse[0][0] == "docA"  # Highest sparse score

    # alpha=0.5 -> balanced combination
    convex_mid = fusion.convex_combination(sparse_rank, dense_rank, top_k=4, alpha=0.5)
    assert len(convex_mid) == 4


def test_hybrid_fusion_dev_tuning(sample_corpus_and_queries, tmp_path):
    """Test that HybridFusionRetriever tunes parameters on dev queries and exports TREC run."""
    corpus, queries, qrels = sample_corpus_and_queries
    fusion = HybridFusionRetriever()
    harness = EvaluationHarness()

    sparse_dev = {
        "q1": [("case_001", 10.0), ("case_002", 2.0)],
        "q2": [("case_001", 5.0), ("case_002", 8.0)],
    }
    dense_dev = {
        "q1": [("case_001", 0.9), ("case_002", 0.2)],
        "q2": [("case_002", 0.85), ("case_001", 0.3)],
    }

    best_cfg, df_grid = fusion.tune_weights_on_dev(
        dev_sparse_results=sparse_dev,
        dev_dense_results=dense_dev,
        dev_qrels=qrels,
        harness=harness,
        alpha_steps=5,
        rrf_k_candidates=[10, 60],
        top_k=2
    )

    assert "best_alpha" in best_cfg
    assert "best_rrf_k" in best_cfg
    assert 0.0 <= best_cfg["best_alpha"] <= 1.0
    assert len(df_grid) == 5 + 2  # 5 alpha steps + 2 RRF trials

    # Test run file export
    fused_results = fusion.fuse_batch(sparse_dev, dense_dev, top_k=2, method="convex", alpha=best_cfg["best_alpha"])
    run_file = str(tmp_path / "hybrid_tuned.run")
    fusion.save_run_file(fused_results, run_file, run_name="HYBRID_TUNED")
    assert os.path.exists(run_file)
