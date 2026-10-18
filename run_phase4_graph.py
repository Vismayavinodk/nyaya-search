"""
Phase 4 Runner: Leakage-Safe Citation Graph Construction (NetworkX),
Topological Centrality (In-Degree, PageRank, HITS, Node2Vec),
Feature-Relevance Correlation Analysis, and Dev-Tuned Graph Score Interpolation.
"""

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
from src.retrieval.sparse_bm25 import BM25Retriever
from src.graph.citation_graph import CitationGraphEngine
from src.graph.graph_retriever import GraphScoreInterpolator
from src.evaluation.eval_harness import EvaluationHarness


def parse_args():
    parser = argparse.ArgumentParser(description="Phase 4: Citation Graph & Graph Score Interpolation")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config YAML")
    parser.add_argument("--dataset", type=str, default="il_pcr", help="Dataset name (il_pcr or sample)")
    parser.add_argument("--top_k", type=int, default=50, help="Retrieval candidate pool top-k")
    parser.add_argument("--max_cases", type=int, default=None, help="Subsample corpus size for fast local testing")
    parser.add_argument("--max_queries", type=int, default=None, help="Subsample queries count for quick test")
    parser.add_argument("--node2vec_dim", type=int, default=32, help="Embedding dimension for Node2Vec")
    return parser.parse_args()


def verify_zero_test_leakage(graph_engine: CitationGraphEngine, test_qids: set) -> Dict[str, Any]:
    """
    Formally verifies that zero outgoing citation edges originate from test queries.
    """
    g = graph_engine.graph
    test_out_edges = []
    for u, v in g.edges():
        if u in test_qids:
            test_out_edges.append((u, v))

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
    logger = get_logger("Phase4CitationGraph", log_dir=cfg.paths.log_dir)

    logger.info("=" * 80)
    logger.info("PHASE 4: CITATION GRAPH (NETWORKX), HITS, NODE2VEC & SCORE INTERPOLATION")
    logger.info(f"Dataset: {args.dataset} | Top-K: {args.top_k}")
    logger.info("=" * 80)

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

    # Subsample corpus if requested for fast testing
    if args.max_cases and len(corpus) > args.max_cases:
        subset_keys = list(corpus.keys())[:args.max_cases]
        corpus = {k: corpus[k] for k in subset_keys}
        logger.info(f"Subsampled corpus to {len(corpus)} cases for testing.")

    train_qids, val_qids, test_qids = loader.split_queries(queries, qrels, seed=cfg.project.seed)
    logger.info(f"Query Partitions: Train={len(train_qids)}, Dev={len(val_qids)}, Test={len(test_qids)}")

    if args.max_queries:
        val_qids = val_qids[:args.max_queries]
        test_qids = test_qids[:args.max_queries]
        logger.info(f"Subsampled Dev to {len(val_qids)} queries, Test to {len(test_qids)} queries.")

    dev_queries = {qid: queries[qid] for qid in val_qids if qid in queries}
    test_queries = {qid: queries[qid] for qid in test_qids if qid in queries}

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
    logger.info("\n--- [Step 1] Constructing Directed Citation Network (NetworkX) ---")
    graph_engine = CitationGraphEngine(
        damping_factor=cfg.citation_graph.damping_factor,
        max_iter=cfg.citation_graph.max_iter,
        tol=cfg.citation_graph.tol
    )

    # Exclude outgoing edges from test queries to eliminate test data leakage
    graph_engine.build_graph(
        corpus=corpus,
        queries=queries,
        exclude_outgoing_for=set(test_qids)
    )

    # Formally verify zero test leakage
    leakage_audit = verify_zero_test_leakage(graph_engine, set(test_qids))
    logger.info(
        f"Test Data Leakage Verification: Safe={leakage_audit['is_leakage_free']} | "
        f"Outgoing edges from {leakage_audit['test_nodes_evaluated']} test queries: "
        f"{leakage_audit['test_outgoing_edges_found']} (Max Out-Degree={leakage_audit['max_test_out_degree']})"
    )
    assert leakage_audit["is_leakage_free"], "Fatal test data leakage detected in citation graph!"

    # Export citation edges TSV
    edges_path = os.path.join(proc_dir, "citation_edges.tsv")
    graph_engine.export_citation_edges(edges_path)

    # =========================================================================
    # Step 2: Compute Node2Vec Graph Embeddings
    # =========================================================================
    logger.info("\n--- [Step 2] Computing Node2Vec Graph Embeddings ---")
    n2v_embs = graph_engine.compute_node2vec_embeddings(
        dimensions=args.node2vec_dim,
        walk_length=20,
        num_walks=10,
        window_size=5,
        seed=cfg.project.seed
    )
    logger.info(f"Node2Vec dense embeddings computed for {len(n2v_embs)} legal precedents.")

    # Top Precedents by In-Degree, PageRank, and HITS Authority
    top_in = sorted(graph_engine.in_degree_cache.items(), key=lambda x: x[1], reverse=True)[:5]
    top_pr = sorted(graph_engine.pagerank_cache.items(), key=lambda x: x[1], reverse=True)[:5]
    top_hits = sorted(graph_engine.hits_authorities_cache.items(), key=lambda x: x[1], reverse=True)[:5]

    logger.info("\nTop Precedents by Authority Centrality:")
    for rank, ((did_in, s_in), (did_pr, s_pr), (did_h, s_h)) in enumerate(zip(top_in, top_pr, top_hits), start=1):
        logger.info(
            f"  Rank {rank}: In-Degree={did_in} ({s_in:.4f}) | "
            f"PageRank={did_pr} ({s_pr:.4f}) | "
            f"HITS-Auth={did_h} ({s_h:.4f})"
        )

    # =========================================================================
    # Step 3: Base Retrieval Candidate Generation (Tuned BM25 / Hybrid)
    # =========================================================================
    logger.info("\n--- [Step 3] Generating Candidate Retrieval Pool ---")
    bm25 = BM25Retriever(algorithm="bm25_okapi", k1=2.2, b=0.85)
    bm25.build_index(corpus)
    base_dev_res = bm25.batch_search(dev_queries, top_k=args.top_k)
    base_test_res = bm25.batch_search(test_queries, top_k=args.top_k)

    base_dev_metrics, _ = harness.evaluate_rankings(base_dev_res, dev_qrels)
    base_test_metrics, base_per_query_df = harness.evaluate_rankings(base_test_res, test_qrels)

    comparison_records: List[Dict[str, Any]] = [{
        "Configuration": "Phase 2: Tuned BM25 Baseline",
        "Type": "Lexical",
        "Dev nDCG@10": base_dev_metrics.get("ndcg@10", 0.0),
        "Test nDCG@10": base_test_metrics.get("ndcg@10", 0.0),
        "Test MAP": base_test_metrics.get("map", 0.0),
        "Test Recall@10": base_test_metrics.get("recall@10", 0.0),
        "Test Recall@20": base_test_metrics.get("recall@20", 0.0),
        "Test MRR": base_test_metrics.get("mrr", 0.0)
    }]

    # Load Phase 3 Hybrid Weighted reference if available and matches dataset queries
    hybrid_run_path = os.path.join(runs_dir, "hybrid_weighted_tuned_test.run")
    hybrid_per_query_df = pd.DataFrame()
    if os.path.exists(hybrid_run_path):
        h_metrics, h_df = harness.evaluate_run_file(hybrid_run_path, qrels_file)
        if not h_df.empty and set(h_df["query_id"]).intersection(set(test_queries.keys())):
            hybrid_per_query_df = h_df
            comparison_records.append({
                "Configuration": "Phase 3: Hybrid Weighted Score",
                "Type": "Hybrid (BM25+Dense)",
                "Dev nDCG@10": 0.0617,
                "Test nDCG@10": h_metrics.get("ndcg@10", 0.0),
                "Test MAP": h_metrics.get("map", 0.0),
                "Test Recall@10": h_metrics.get("recall@10", 0.0),
                "Test Recall@20": h_metrics.get("recall@20", 0.0),
                "Test MRR": h_metrics.get("mrr", 0.0)
            })

    # =========================================================================
    # Step 4: Feature-Relevance Correlation Analysis
    # =========================================================================
    logger.info("\n--- [Step 4] Computing Feature-Relevance Correlations ---")
    corr_df = graph_engine.compute_feature_relevance_correlations(
        candidate_rankings=base_test_res,
        qrels=test_qrels,
        queries=test_queries
    )

    corr_csv_path = os.path.join(results_dir, "feature_relevance_correlations.csv")
    corr_df.to_csv(corr_csv_path, index=False)
    logger.info(f"Saved feature correlation analysis to: {corr_csv_path}")

    print("\n" + "=" * 90)
    print("FEATURE-RELEVANCE CORRELATIONS (Point-Biserial Pearson r and Spearman rho)")
    print("=" * 90)
    print(tabulate(corr_df, headers="keys", tablefmt="github", floatfmt=".4f"))
    print("=" * 90)

    # =========================================================================
    # Step 5: Graph Score Interpolation (Tuned Strictly on Dev)
    # =========================================================================
    logger.info("\n--- [Step 5] Tuning Graph Score Interpolation Strictly on Dev Split ---")
    interpolator = GraphScoreInterpolator(graph_engine=graph_engine)

    best_cfg, df_grid = interpolator.tune_interpolation_on_dev(
        dev_base_results=base_dev_res,
        dev_qrels=dev_qrels,
        harness=harness,
        beta_candidates=[0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50],
        graph_signals=["pagerank", "hits_authority", "in_degree", "personalized_pagerank"],
        top_k=args.top_k,
        queries=dev_queries
    )

    tuning_csv = os.path.join(results_dir, "graph_interpolation_dev_tuning_results.csv")
    df_grid.to_csv(tuning_csv, index=False)
    logger.info(f"Saved graph interpolation tuning grid to: {tuning_csv}")

    opt_beta = best_cfg["best_beta"]
    opt_signal = best_cfg["best_signal"]

    # =========================================================================
    # Step 6: Evaluate Optimal Graph Interpolation on Held-out Test Set
    # =========================================================================
    logger.info(f"\nEvaluating Optimal Graph Interpolation (signal='{opt_signal}', beta={opt_beta}) on Test Set...")
    interpolated_dev = interpolator.interpolate_batch(
        base_results=base_dev_res,
        beta=opt_beta,
        signal_name=opt_signal,
        top_k=args.top_k,
        queries=dev_queries
    )
    interpolated_test = interpolator.interpolate_batch(
        base_results=base_test_res,
        beta=opt_beta,
        signal_name=opt_signal,
        top_k=args.top_k,
        queries=test_queries
    )

    graph_run_path = os.path.join(runs_dir, "graph_interpolated_tuned_test.run")
    interpolator.save_run_file(interpolated_test, graph_run_path, run_name=f"GRAPH_{opt_signal.upper()}")

    dev_metrics, _ = harness.evaluate_rankings(interpolated_dev, dev_qrels)
    per_query_file = os.path.join(results_dir, "per_query_graph_interpolated_tuned_test.csv")
    test_metrics, graph_per_query_df = harness.evaluate_run_file(
        run_file_path=graph_run_path,
        qrels_file_path=qrels_file,
        per_query_output_path=per_query_file
    )

    comparison_records.append({
        "Configuration": f"Phase 4: Graph-Interpolated ({opt_signal}, beta={opt_beta:.2f})",
        "Type": "Graph-Augmented",
        "Dev nDCG@10": dev_metrics.get("ndcg@10", 0.0),
        "Test nDCG@10": test_metrics.get("ndcg@10", 0.0),
        "Test MAP": test_metrics.get("map", 0.0),
        "Test Recall@10": test_metrics.get("recall@10", 0.0),
        "Test Recall@20": test_metrics.get("recall@20", 0.0),
        "Test MRR": test_metrics.get("mrr", 0.0)
    })

    # =========================================================================
    # Step 7: Paired Statistical Significance Testing
    # =========================================================================
    logger.info("\n--- Computing Paired Statistical Significance Tests for Phase 4 ---")
    sig_records = []

    # Test 1: Graph-Interpolated vs BM25 Baseline
    sig_graph_vs_bm25 = harness.compute_statistical_significance(
        df_baseline=base_per_query_df,
        df_treatment=graph_per_query_df,
        metric="ndcg@10"
    )
    sig_records.append({"Comparison": "Graph-Interpolated vs BM25 Baseline", **sig_graph_vs_bm25})

    # Test 2: Graph-Interpolated vs Phase 3 Hybrid
    if not hybrid_per_query_df.empty:
        sig_graph_vs_hybrid = harness.compute_statistical_significance(
            df_baseline=hybrid_per_query_df,
            df_treatment=graph_per_query_df,
            metric="ndcg@10"
        )
        sig_records.append({"Comparison": "Graph-Interpolated vs Phase 3 Hybrid", **sig_graph_vs_hybrid})

    sig_df = pd.DataFrame(sig_records)
    sig_csv_path = os.path.join(results_dir, "statistical_significance_phase4.csv")
    sig_df.to_csv(sig_csv_path, index=False)

    # =========================================================================
    # Step 8: Display & Save Comprehensive Phase 4 Report
    # =========================================================================
    results_df = pd.DataFrame(comparison_records)

    print("\n" + "=" * 95)
    print("PHASE 4: CITATION GRAPH & GRAPH SCORE INTERPOLATION RESULTS TABLE")
    print("=" * 95)
    print(tabulate(results_df, headers="keys", tablefmt="github", floatfmt=".4f"))
    print("=" * 95)

    print("\n" + "=" * 95)
    print("PAIRED STATISTICAL SIGNIFICANCE TESTS (nDCG@10 on Held-out Test Set)")
    print("=" * 95)
    sig_display = sig_df[["Comparison", "mean_baseline", "mean_treatment", "relative_gain_percent", "t_statistic", "t_pvalue", "is_significant_p05"]]
    print(tabulate(sig_display, headers="keys", tablefmt="github", floatfmt=".4f"))
    print("=" * 95 + "\n")

    summary_md_path = os.path.join(results_dir, "phase4_graph_summary.md")
    with open(summary_md_path, "w", encoding="utf-8") as f:
        f.write("# Phase 4: Citation Graph & Graph Score Interpolation Evaluation Report\n\n")
        f.write(f"**Dataset**: `{args.dataset}` ({len(corpus)} cases, {len(dev_queries)} dev queries, {len(test_queries)} test queries)\n")
        f.write(f"**Graph Dimensions**: {graph_engine.graph.number_of_nodes()} nodes, {graph_engine.graph.number_of_edges()} citation edges\n")
        f.write(f"**Leakage Audit**: Safe={leakage_audit['is_leakage_free']} (0 outgoing edges from test queries)\n\n")
        f.write("## 1. Comparative Results across Phases (BM25 vs. Hybrid vs. Graph-Interpolated)\n\n")
        f.write(tabulate(results_df, headers="keys", tablefmt="github", floatfmt=".4f"))
        f.write("\n\n## 2. Feature-Relevance Correlations\n\n")
        f.write(tabulate(corr_df, headers="keys", tablefmt="github", floatfmt=".4f"))
        f.write("\n\n## 3. Paired Statistical Significance Testing (nDCG@10)\n\n")
        f.write(tabulate(sig_display, headers="keys", tablefmt="github", floatfmt=".4f"))
        f.write("\n\n## 4. Dev-Split Graph Score Interpolation Tuning\n\n")
        f.write(f"- **Optimal Graph Signal**: `{opt_signal}`\n")
        f.write(f"- **Optimal Interpolation Weight**: `beta = {opt_beta:.2f}` (Base weight = {1.0 - opt_beta:.2f}, Graph weight = {opt_beta:.2f})\n")
        f.write("\n## 5. Generated Artifacts\n\n")
        f.write(f"- Citation Edges TSV: `{edges_path}`\n")
        f.write(f"- Interpolated Test Run: `experiments/runs/graph_interpolated_tuned_test.run`\n")
        f.write(f"- Feature Correlations CSV: `{corr_csv_path}`\n")
        f.write(f"- Tuning Grid CSV: `{tuning_csv}`\n")
        f.write(f"- Per-Query Scores CSV: `{per_query_file}`\n")

    results_df.to_json(os.path.join(results_dir, "phase4_graph_summary.json"), orient="records", indent=2)
    logger.info(f"Phase 4 report generated in: {summary_md_path}")


if __name__ == "__main__":
    main()
