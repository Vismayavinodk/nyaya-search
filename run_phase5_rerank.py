"""
Phase 5 Runner: First-Stage Top-100 Candidate Generation from Hybrid Retrieval,
Option A (LightGBM LambdaMART with BM25, Dense, PageRank, In-Degree, Doc Length),
Option B (Cross-Encoder / BGE-Reranker on Top-K),
Trained on Train split, Tuned on Dev split, Evaluated on Held-out Test split only once.
Includes Feature Importance Analysis and Paired Statistical Significance Testing (Bootstrap & Paired t-test).
"""

from __future__ import annotations

import os
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import argparse
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
from tabulate import tabulate

from src.utils.seed import seed_everything
from src.utils.logger import get_logger
from src.utils.config_parser import load_config
from src.data.dataset_loaders import DatasetLoader
from src.data.schema import Qrel
from src.graph.citation_graph import CitationGraphEngine
from src.retrieval.sparse_bm25 import BM25Retriever
from src.retrieval.dense_encoder import DenseLegalRetriever
from src.retrieval.hybrid_fusion import HybridFusionRetriever
from src.rerank.feature_extractor import LegalFeatureExtractor
from src.rerank.lightgbm_ranker import LightGBMLambdaMART
from src.rerank.cross_encoder_reranker import CrossEncoderReranker
from src.evaluation.eval_harness import EvaluationHarness


def parse_args():
    parser = argparse.ArgumentParser(description="Phase 5: LightGBM LambdaMART vs Cross-Encoder Reranking")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config YAML")
    parser.add_argument("--dataset", type=str, default="il_pcr", help="Dataset name ('il_pcr' or 'sample')")
    parser.add_argument("--top_k", type=int, default=100, help="Candidate pool size from hybrid retrieval")
    parser.add_argument("--cross_encoder_top_k", type=int, default=50, help="Top-K candidates to cross-encoder rerank")
    parser.add_argument("--max_cases", type=int, default=250, help="Corpus cases (defaults to 250 matching Phase 3)")
    parser.add_argument("--max_queries", type=int, default=50, help="Dev/Test query count (defaults to 50 matching Phase 3)")
    parser.add_argument("--max_train_queries", type=int, default=100, help="Train query count for reranker")
    parser.add_argument("--cross_encoder_model", type=str, default="BAAI/bge-reranker-base", help="Cross-encoder model name")
    return parser.parse_args()


def verify_zero_test_leakage(graph_engine: CitationGraphEngine, test_qids: set) -> Dict[str, Any]:
    """Formally verifies that zero outgoing citation edges originate from test queries."""
    g = graph_engine.graph
    test_out_edges = [(u, v) for u, v in g.edges() if u in test_qids]
    max_out_deg = max((g.out_degree(qid) for qid in test_qids if g.has_node(qid)), default=0)
    is_leakage_free = (len(test_out_edges) == 0) and (max_out_deg == 0)
    return {
        "is_leakage_free": is_leakage_free,
        "test_nodes_evaluated": len(test_qids),
        "test_outgoing_edges_found": len(test_out_edges),
        "max_test_out_degree": max_out_deg
    }


def main():
    args = parse_args()
    cfg = load_config(args.config)
    seed = seed_everything(cfg.project.seed)
    logger = get_logger("Phase5Reranker", log_dir=cfg.paths.log_dir)

    logger.info("=" * 85)
    logger.info("PHASE 5: MULTI-MODAL RERANKING (OPTION A: LIGHTGBM LAMBDAMART vs OPTION B: CROSS-ENCODER)")
    logger.info(f"Dataset: {args.dataset} | Hybrid Candidate Top-K: {args.top_k} | Cross-Encoder Top-K: {args.cross_encoder_top_k}")
    logger.info("=" * 85)

    # 1. Load Dataset
    loader = DatasetLoader()
    raw_dir = os.path.join(cfg.paths.raw_data_dir, args.dataset)
    proc_dir = os.path.join(cfg.paths.processed_data_dir, args.dataset)

    if args.dataset == "il_pcr":
        corpus, queries, qrels = loader.load_il_pcr(raw_dir, processed_dir=proc_dir)
    elif args.dataset == "sample":
        corpus, queries, qrels = loader.load_or_create_sample(sample_dir=cfg.paths.sample_data_dir)
    else:
        corpus, queries, qrels = loader.load_il_pcr(raw_dir, processed_dir=proc_dir)

    if args.max_cases and len(corpus) > args.max_cases:
        subset_keys = list(corpus.keys())[:args.max_cases]
        corpus = {k: corpus[k] for k in subset_keys}
        logger.info(f"Subsampled corpus to {len(corpus)} cases for benchmark evaluation.")

    train_qids, val_qids, test_qids = loader.split_queries(queries, qrels, seed=cfg.project.seed)

    if args.max_train_queries and len(train_qids) > args.max_train_queries:
        train_qids = train_qids[:args.max_train_queries]
    if args.max_queries:
        val_qids = val_qids[:args.max_queries]
        test_qids = test_qids[:args.max_queries]

    logger.info(f"Splits for Phase 5: Train={len(train_qids)}, Dev={len(val_qids)}, Test={len(test_qids)}")

    train_queries = {qid: queries[qid] for qid in train_qids if qid in queries}
    dev_queries = {qid: queries[qid] for qid in val_qids if qid in queries}
    test_queries = {qid: queries[qid] for qid in test_qids if qid in queries}

    # Format Qrels
    train_qrels_list = [q for q in qrels if q.query_id in train_queries]
    dev_qrels_list = [q for q in qrels if q.query_id in dev_queries]
    test_qrels_list = [q for q in qrels if q.query_id in test_queries]

    qrel_dict: Dict[str, Dict[str, int]] = {}
    for q in qrels:
        if q.query_id not in qrel_dict:
            qrel_dict[q.query_id] = {}
        qrel_dict[q.query_id][q.doc_id] = q.relevance

    dev_qrels = {qid: qrel_dict[qid] for qid in dev_queries if qid in qrel_dict}
    test_qrels = {qid: qrel_dict[qid] for qid in test_queries if qid in qrel_dict}

    runs_dir = "experiments/runs"
    results_dir = cfg.paths.output_dir
    os.makedirs(runs_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    harness = EvaluationHarness(relevance_threshold=cfg.evaluation.relevance_threshold)
    qrels_file = os.path.join(proc_dir, "qrels.tsv") if os.path.exists(os.path.join(proc_dir, "qrels.tsv")) else os.path.join(cfg.paths.sample_data_dir, "qrels.tsv")

    # =========================================================================
    # Step 1: Build Citation Graph (Strictly Leakage-Safe)
    # =========================================================================
    logger.info("\n--- [Step 1] Building Leakage-Safe Citation Graph (NetworkX) ---")
    graph = CitationGraphEngine(
        damping_factor=cfg.citation_graph.damping_factor,
        max_iter=cfg.citation_graph.max_iter,
        tol=cfg.citation_graph.tol
    )
    graph.build_graph(corpus=corpus, queries=queries, exclude_outgoing_for=set(test_qids))
    leakage_audit = verify_zero_test_leakage(graph, set(test_qids))
    logger.info(f"Leakage Audit: Safe={leakage_audit['is_leakage_free']} (0 test outgoing edges)")
    assert leakage_audit["is_leakage_free"], "Fatal test data leakage in citation graph!"

    # =========================================================================
    # Step 2: First-Stage Hybrid Retrieval Candidates (Top-100)
    # =========================================================================
    logger.info(f"\n--- [Step 2] Generating First-Stage Hybrid Candidate Pool (Top-{args.top_k}) ---")
    # 2a. BM25 Retrieval
    bm25 = BM25Retriever(algorithm="bm25_okapi", k1=2.2, b=0.85)
    bm25.build_index(corpus)
    train_sparse = bm25.batch_search(train_queries, top_k=args.top_k)
    dev_sparse = bm25.batch_search(dev_queries, top_k=args.top_k)
    test_sparse = bm25.batch_search(test_queries, top_k=args.top_k)

    # 2b. Dense Semantic Retrieval
    dense = DenseLegalRetriever(
        model_name="BAAI/bge-small-en-v1.5",
        pooling_strategy="mean",
        batch_size=16,
        device="auto"
    )
    cache_path = os.path.join(cfg.paths.cache_dir, f"{args.dataset}_BAAI_bge-small-en-v1.5_mean.npz")
    dense.build_index(corpus, cache_path=cache_path)
    train_dense = dense.batch_search(train_queries, top_k=args.top_k)
    dev_dense = dense.batch_search(dev_queries, top_k=args.top_k)
    test_dense = dense.batch_search(test_queries, top_k=args.top_k)

    # 2c. Hybrid Fusion (Phase 3 Optimal Weighted Score, alpha=0.35)
    fusion = HybridFusionRetriever(method="convex", convex_dense_weight=0.35)
    hybrid_train = fusion.fuse_batch(train_sparse, train_dense, top_k=args.top_k)
    hybrid_dev = fusion.fuse_batch(dev_sparse, dev_dense, top_k=args.top_k)
    hybrid_test = fusion.fuse_batch(test_sparse, test_dense, top_k=args.top_k)

    hybrid_run_path = os.path.join(runs_dir, "phase5_hybrid_baseline_test.run")
    hybrid_per_query_file = os.path.join(results_dir, "per_query_phase5_hybrid_baseline_test.csv")
    fusion.save_run_file(hybrid_test, hybrid_run_path, run_name="HYBRID_BASELINE")
    hybrid_dev_metrics, _ = harness.evaluate_rankings(hybrid_dev, dev_qrels)
    hybrid_test_metrics, df_hybrid_test = harness.evaluate_rankings(hybrid_test, test_qrels)
    df_hybrid_test.to_csv(hybrid_per_query_file, index=False)

    comparison_records = [{
        "Configuration": "Hybrid Baseline (BM25 + Dense BGE, alpha=0.35)",
        "Type": "First-Stage Hybrid Pool",
        "Dev nDCG@10": hybrid_dev_metrics.get("ndcg@10", 0.0),
        "Test nDCG@10": hybrid_test_metrics.get("ndcg@10", 0.0),
        "Test MAP": hybrid_test_metrics.get("map", 0.0),
        "Test Recall@10": hybrid_test_metrics.get("recall@10", 0.0),
        "Test Recall@20": hybrid_test_metrics.get("recall@20", 0.0),
        "Test MRR": hybrid_test_metrics.get("mrr", 0.0)
    }]

    # =========================================================================
    # Step 3: Option A - Feature Extraction & LightGBM LambdaMART
    # =========================================================================
    logger.info("\n--- [Step 3] Option A: LightGBM LambdaMART Training & Dev Tuning ---")
    feat_extractor = LegalFeatureExtractor(graph_engine=graph)
    core_features = LegalFeatureExtractor.CORE_FEATURE_NAMES
    logger.info(f"Target Feature Set ({len(core_features)} features): {core_features}")

    logger.info("Extracting candidate pair features for Train, Dev, and Test splits...")
    train_df = feat_extractor.build_dataset_features_from_candidates(
        queries=train_queries,
        corpus=corpus,
        candidate_rankings=hybrid_train,
        sparse_results=train_sparse,
        dense_results=train_dense,
        top_k=args.top_k
    )
    dev_df = feat_extractor.build_dataset_features_from_candidates(
        queries=dev_queries,
        corpus=corpus,
        candidate_rankings=hybrid_dev,
        sparse_results=dev_sparse,
        dense_results=dev_dense,
        top_k=args.top_k
    )
    test_df = feat_extractor.build_dataset_features_from_candidates(
        queries=test_queries,
        corpus=corpus,
        candidate_rankings=hybrid_test,
        sparse_results=test_sparse,
        dense_results=test_dense,
        top_k=args.top_k
    )

    logger.info(f"Feature Matrices: Train={len(train_df)}, Dev={len(dev_df)}, Test={len(test_df)}")

    # Initialize LightGBM LambdaMART
    lgbm_ranker = LightGBMLambdaMART(
        n_estimators=150,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_samples=5,
        random_state=cfg.project.seed,
        feature_cols=core_features
    )

    # Train on Train split, tune stopping on Dev split
    lgbm_ranker.fit(
        train_df=train_df,
        train_qrels=train_qrels_list,
        val_df=dev_df,
        val_qrels=dev_qrels_list,
        early_stopping_rounds=15
    )

    # Feature Importance Analysis
    feat_imp_gain = lgbm_ranker.get_feature_importances(importance_type="gain")
    feat_imp_split = lgbm_ranker.get_feature_importances(importance_type="split")

    feat_imp_df = pd.DataFrame([
        {
            "Feature": feat,
            "Importance_Gain": feat_imp_gain.get(feat, 0.0),
            "Importance_Split": feat_imp_split.get(feat, 0.0)
        }
        for feat in core_features
    ]).sort_values(by="Importance_Gain", ascending=False)

    feat_imp_csv = os.path.join(results_dir, "feature_importance_lightgbm.csv")
    feat_imp_df.to_csv(feat_imp_csv, index=False)
    logger.info(f"Saved LightGBM feature importance to: {feat_imp_csv}")

    # Dev and Test Ranking Predictions
    lgbm_dev_rankings = lgbm_ranker.predict_rankings(dev_df, top_k=args.top_k)
    lgbm_test_rankings = lgbm_ranker.predict_rankings(test_df, top_k=args.top_k)

    lgbm_run_path = os.path.join(runs_dir, "rerank_lightgbm_tuned_test.run")
    lgbm_per_query_file = os.path.join(results_dir, "per_query_rerank_lightgbm_test.csv")
    fusion.save_run_file(lgbm_test_rankings, lgbm_run_path, run_name="LIGHTGBM_LAMBDAMART")

    lgbm_dev_metrics, _ = harness.evaluate_rankings(lgbm_dev_rankings, dev_qrels)
    lgbm_test_metrics, df_lgbm_test = harness.evaluate_rankings(lgbm_test_rankings, test_qrels)
    df_lgbm_test.to_csv(lgbm_per_query_file, index=False)

    comparison_records.append({
        "Configuration": f"Option A: LightGBM LambdaMART (Best Iter={lgbm_ranker.best_iteration_})",
        "Type": "Supervised LTR (LambdaMART)",
        "Dev nDCG@10": lgbm_dev_metrics.get("ndcg@10", 0.0),
        "Test nDCG@10": lgbm_test_metrics.get("ndcg@10", 0.0),
        "Test MAP": lgbm_test_metrics.get("map", 0.0),
        "Test Recall@10": lgbm_test_metrics.get("recall@10", 0.0),
        "Test Recall@20": lgbm_test_metrics.get("recall@20", 0.0),
        "Test MRR": lgbm_test_metrics.get("mrr", 0.0)
    })

    # =========================================================================
    # Step 4: Option B - Cross-Encoder Reranker (BGE-Reranker on Top-K)
    # =========================================================================
    logger.info(f"\n--- [Step 4] Option B: Cross-Encoder Reranking (Top-{args.cross_encoder_top_k}) ---")
    ce_reranker = CrossEncoderReranker(
        model_name=args.cross_encoder_model,
        max_length=512,
        batch_size=16,
        max_doc_chars=1500
    )

    ce_dev_rankings = ce_reranker.rerank_batch(
        queries=dev_queries,
        corpus=corpus,
        candidate_rankings=hybrid_dev,
        top_k=args.cross_encoder_top_k
    )
    ce_test_rankings = ce_reranker.rerank_batch(
        queries=test_queries,
        corpus=corpus,
        candidate_rankings=hybrid_test,
        top_k=args.cross_encoder_top_k
    )

    ce_run_path = os.path.join(runs_dir, "rerank_cross_encoder_test.run")
    ce_per_query_file = os.path.join(results_dir, "per_query_rerank_cross_encoder_test.csv")
    fusion.save_run_file(ce_test_rankings, ce_run_path, run_name="CROSS_ENCODER")

    ce_dev_metrics, _ = harness.evaluate_rankings(ce_dev_rankings, dev_qrels)
    ce_test_metrics, df_ce_test = harness.evaluate_rankings(ce_test_rankings, test_qrels)
    df_ce_test.to_csv(ce_per_query_file, index=False)

    comparison_records.append({
        "Configuration": f"Option B: Cross-Encoder ({os.path.basename(ce_reranker.model_name)}, top-{args.cross_encoder_top_k})",
        "Type": "Cross-Encoder Transformer",
        "Dev nDCG@10": ce_dev_metrics.get("ndcg@10", 0.0),
        "Test nDCG@10": ce_test_metrics.get("ndcg@10", 0.0),
        "Test MAP": ce_test_metrics.get("map", 0.0),
        "Test Recall@10": ce_test_metrics.get("recall@10", 0.0),
        "Test Recall@20": ce_test_metrics.get("recall@20", 0.0),
        "Test MRR": ce_test_metrics.get("mrr", 0.0)
    })

    # =========================================================================
    # Step 5: Paired Statistical Significance Testing (Bootstrap & Paired t-test)
    # =========================================================================
    logger.info("\n--- [Step 5] Computing Paired Statistical Significance Tests (nDCG@10) ---")
    sig_records = []

    # 1. Option A (LightGBM) vs Hybrid Baseline
    sig_a_vs_base = harness.compute_statistical_significance(
        df_baseline=df_hybrid_test,
        df_treatment=df_lgbm_test,
        metric="ndcg@10"
    )
    sig_records.append({"Comparison": "Option A (LightGBM) vs Hybrid Baseline", **sig_a_vs_base})

    # 2. Option B (Cross-Encoder) vs Hybrid Baseline
    sig_b_vs_base = harness.compute_statistical_significance(
        df_baseline=df_hybrid_test,
        df_treatment=df_ce_test,
        metric="ndcg@10"
    )
    sig_records.append({"Comparison": "Option B (Cross-Encoder) vs Hybrid Baseline", **sig_b_vs_base})

    # 3. Option A vs Option B
    sig_a_vs_b = harness.compute_statistical_significance(
        df_baseline=df_ce_test,
        df_treatment=df_lgbm_test,
        metric="ndcg@10"
    )
    sig_records.append({"Comparison": "Option A (LightGBM) vs Option B (Cross-Encoder)", **sig_a_vs_b})

    sig_df = pd.DataFrame(sig_records)
    sig_csv_path = os.path.join(results_dir, "statistical_significance_phase5.csv")
    sig_df.to_csv(sig_csv_path, index=False)

    # =========================================================================
    # Step 6: Display & Save Comprehensive Phase 5 Report
    # =========================================================================
    results_df = pd.DataFrame(comparison_records)

    print("\n" + "=" * 105)
    print("PHASE 5: MULTI-MODAL RERANKING RESULTS TABLE (HELD-OUT TEST SET EVALUATED ONCE)")
    print("=" * 105)
    print(tabulate(results_df, headers="keys", tablefmt="github", floatfmt=".4f"))
    print("=" * 105)

    print("\n" + "=" * 105)
    print("LIGHTGBM LAMBDAMART FEATURE IMPORTANCE (GAIN & SPLIT)")
    print("=" * 105)
    print(tabulate(feat_imp_df, headers="keys", tablefmt="github", floatfmt=".4f"))
    print("=" * 105)

    print("\n" + "=" * 105)
    print("PAIRED STATISTICAL SIGNIFICANCE TESTS (nDCG@10: Paired t-test, Wilcoxon & Bootstrap 95% CI)")
    print("=" * 105)
    sig_display_cols = [
        "Comparison", "mean_baseline", "mean_treatment", "relative_gain_percent",
        "t_statistic", "t_pvalue", "bootstrap_pvalue", "bootstrap_ci_lower", "bootstrap_ci_upper", "is_significant_p05"
    ]
    print(tabulate(sig_df[sig_display_cols], headers="keys", tablefmt="github", floatfmt=".4f"))
    print("=" * 105 + "\n")

    summary_md_path = os.path.join(results_dir, "phase5_rerank_summary.md")
    with open(summary_md_path, "w", encoding="utf-8") as f:
        f.write("# Phase 5: Multi-Modal Candidate Reranking Evaluation Report\n\n")
        f.write(f"**Dataset**: `{args.dataset}` ({len(corpus)} cases, {len(dev_queries)} dev queries, {len(test_queries)} test queries)\n")
        f.write(f"**Candidate Pool Size**: Top-{args.top_k} from Hybrid System\n")
        f.write(f"**Cross-Encoder Model**: `{ce_reranker.model_name}` (Reranking Top-{args.cross_encoder_top_k})\n")
        f.write(f"**Zero-Leakage Audit**: Safe={leakage_audit['is_leakage_free']}\n\n")

        f.write("## 1. Comparative Results on Held-out Test Split\n\n")
        f.write(tabulate(results_df, headers="keys", tablefmt="github", floatfmt=".4f"))
        f.write("\n\n## 2. LightGBM LambdaMART Feature Importances\n\n")
        f.write(tabulate(feat_imp_df, headers="keys", tablefmt="github", floatfmt=".4f"))
        f.write("\n\n## 3. Paired Statistical Significance Testing (nDCG@10)\n\n")
        f.write(tabulate(sig_df[sig_display_cols], headers="keys", tablefmt="github", floatfmt=".4f"))
        f.write("\n\n## 4. Key Findings & Insights\n\n")
        f.write("- **Supervised Listwise Ranking**: LightGBM LambdaMART directly optimizes NDCG@10 across lexical, dense, and graph authority signals.\n")
        f.write(f"- **Tuned on Dev, Evaluated Once on Test**: Early stopping determined the optimal iteration count (`best_iteration = {lgbm_ranker.best_iteration_}`) exclusively on the validation split before scoring the test split.\n")
        f.write("- **Option A vs. Option B**: Compares tabular feature combinations (Option A) against deep sentence-pair cross-attention (Option B).\n\n")
        f.write("## 5. Generated Artifacts\n\n")
        f.write(f"- LightGBM Test Run: `{lgbm_run_path}`\n")
        f.write(f"- Cross-Encoder Test Run: `{ce_run_path}`\n")
        f.write(f"- Feature Importance CSV: `{feat_imp_csv}`\n")
        f.write(f"- Statistical Significance CSV: `{sig_csv_path}`\n")
        f.write(f"- Per-Query CSVs: `{lgbm_per_query_file}`, `{ce_per_query_file}`\n")

    results_df.to_json(os.path.join(results_dir, "phase5_rerank_summary.json"), orient="records", indent=2)
    logger.info(f"Phase 5 summary report saved to: {summary_md_path}")


if __name__ == "__main__":
    main()
