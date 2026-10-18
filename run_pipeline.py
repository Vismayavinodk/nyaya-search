"""
Main CLI entry point for Citation-Aware Legal Case Retrieval.
Executes data ingestion, sparse/dense indexing, citation graph construction,
reranker training, and full ablation evaluation.
"""

import os
import argparse
from typing import Dict, List, Tuple
from tabulate import tabulate

from src.utils.seed import seed_everything
from src.utils.logger import get_logger
from src.utils.config_parser import load_config, Config
from src.data.citation_extractor import IndianLegalCitationExtractor
from src.data.legal_preprocessor import LegalPreprocessor
from src.data.dataset_loaders import DatasetLoader
from src.data.download_and_build import CorpusBuilder
from src.retrieval.sparse_bm25 import BM25Retriever
from src.retrieval.dense_encoder import DenseLegalRetriever
from src.retrieval.hybrid_fusion import HybridFusionRetriever
from src.graph.citation_graph import CitationGraphEngine
from src.rerank.feature_extractor import LegalFeatureExtractor
from src.rerank.ranker import LegalReranker
from src.evaluation.metrics import IRMetricsEvaluator
from src.evaluation.ablation import LegalRetrievalAblationStudy


def parse_args():
    parser = argparse.ArgumentParser(description="Legal Case Retrieval Pipeline CLI")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config YAML")
    parser.add_argument("--mode", type=str, default="all",
                        choices=["all", "data", "sparse", "dense", "graph", "rerank", "ablation"],
                        help="Execution phase mode")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Override dataset name (sample, il_pcr, fire_aila)")
    parser.add_argument("--force_rebuild", action="store_true",
                        help="Force re-parsing raw files even if processed cache exists")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    seed = seed_everything(cfg.project.seed)
    logger = get_logger("MainPipeline", log_dir=cfg.paths.log_dir)

    logger.info("=" * 60)
    logger.info(f"Starting Legal Case Retrieval Pipeline (Mode: {args.mode})")
    logger.info(f"Random seed locked to {seed}")
    logger.info("=" * 60)

    # Determine dataset to use
    if args.dataset:
        dataset_name = args.dataset
    else:
        # Auto-detect IL-PCR dataset in raw folder
        ilpcr_path = os.path.join(cfg.paths.raw_data_dir, "il_pcr")
        if os.path.isdir(ilpcr_path):
            dataset_name = "il_pcr"
            logger.info("Auto-detected IL-PCR dataset in raw data; using 'il_pcr'.")
        else:
            dataset_name = cfg.data.dataset_name

    # 1. Initialize Preprocessor & Extractor
    extractor = IndianLegalCitationExtractor()
    preprocessor = LegalPreprocessor(
        lower_case=cfg.preprocessing.lower_case,
        remove_boilerplate=cfg.data.remove_boilerplate,
        remove_stopwords=cfg.preprocessing.legal_stopword_removal,
        min_token_len=cfg.preprocessing.min_token_length,
        max_token_len=cfg.preprocessing.max_token_length
    )

    # 2. Data Loading & Partitioning
    loader = DatasetLoader(citation_extractor=extractor)
    try:
        if dataset_name == "sample":
            corpus, queries, qrels = loader.load_or_create_sample(sample_dir=cfg.paths.sample_data_dir)
        elif dataset_name == "fire_aila":
            raw_fire_dir = os.path.join(cfg.paths.raw_data_dir, "fire_aila")
            corpus, queries, qrels = loader.load_fire_aila(raw_fire_dir)
        elif dataset_name == "il_pcr":
            raw_il_dir = os.path.join(cfg.paths.raw_data_dir, "il_pcr")
            proc_il_dir = None if getattr(args, "force_rebuild", False) else os.path.join(cfg.paths.processed_data_dir, "il_pcr")
            corpus, queries, qrels = loader.load_il_pcr(raw_il_dir, processed_dir=proc_il_dir)
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
    except FileNotFoundError as fnf_err:
        logger.error(f"\n{'='*70}\nDATASET ERROR: {fnf_err}\n{'='*70}")
        return

    if not corpus or not queries:
        logger.error(f"Cannot run pipeline: dataset '{dataset_name}' contains {len(corpus)} documents and {len(queries)} queries.")
        return

    train_qids, val_qids, test_qids = loader.split_queries(
        queries=queries,
        qrels=qrels,
        train_ratio=cfg.data.train_split_ratio,
        val_ratio=cfg.data.val_split_ratio,
        test_ratio=cfg.data.test_split_ratio,
        seed=cfg.project.seed
    )
    logger.info(f"Splits created: Train={len(train_qids)}, Val={len(val_qids)}, Test={len(test_qids)} queries.")

    # Phase 1: Build and export standardized corpus, queries, qrels, and edges
    processed_dir = os.path.join(cfg.paths.processed_data_dir, dataset_name)
    builder = CorpusBuilder(preprocessor=preprocessor, citation_extractor=extractor)
    exported_paths = builder.process_and_export(
        cases=list(corpus.values()),
        queries=list(queries.values()),
        qrels=qrels,
        output_dir=processed_dir
    )
    logger.info(f"Phase 1 artifacts generated in '{processed_dir}':")
    for k, p in exported_paths.items():
        logger.info(f"  - {k}: {p}")

    if args.mode == "data":
        logger.info("Phase 1: Data ingestion and standardization complete.")
        return

    # Citation Graph Construction (Phase 4)
    # Strictly remove outgoing edges from test queries to eliminate test data leakage
    graph = CitationGraphEngine(
        damping_factor=cfg.citation_graph.damping_factor,
        max_iter=cfg.citation_graph.max_iter,
        tol=cfg.citation_graph.tol
    )
    graph.build_graph(
        corpus=corpus,
        queries=queries,
        exclude_outgoing_for=set(test_qids)
    )
    edges_out = os.path.join(processed_dir, "citation_edges.tsv")
    graph.export_citation_edges(edges_out)

    if args.mode == "graph":
        logger.info("Top Precedents by In-Degree Centrality & PageRank (Test Leakage Free):")
        sorted_in = sorted(graph.in_degree_cache.items(), key=lambda x: x[1], reverse=True)[:5]
        for did, score in sorted_in:
            logger.info(f"  {did}: In-Degree = {score:.4f}, PageRank = {graph.get_pagerank(did):.4f}")
        logger.info("Phase 4: Citation graph analysis complete.")
        return

    # Sparse BM25 Indexing (Phase 2)
    bm25 = BM25Retriever(
        algorithm=cfg.sparse_retrieval.algorithm,
        k1=cfg.sparse_retrieval.k1,
        b=cfg.sparse_retrieval.b,
        delta=cfg.sparse_retrieval.delta,
        preprocessor=preprocessor
    )
    bm25.build_index(corpus)

    if args.mode == "sparse":
        evaluator = IRMetricsEvaluator(relevance_threshold=cfg.evaluation.relevance_threshold)
        test_queries = {qid: queries[qid] for qid in test_qids if qid in queries}
        test_qrels = [q for q in qrels if q.query_id in test_qids]
        bm25_res = bm25.batch_search(test_queries, top_k=cfg.sparse_retrieval.top_k)
        metrics = evaluator.evaluate_run(bm25_res, test_qrels)
        logger.info(f"BM25 Test Metrics: {metrics}")
        return

    # Dense Semantic Indexing (Phase 3)
    dense = DenseLegalRetriever(
        model_name=cfg.dense_retrieval.model_name,
        chunk_size=cfg.dense_retrieval.chunk_size,
        chunk_overlap=cfg.dense_retrieval.chunk_overlap,
        max_chunks_per_case=getattr(cfg.dense_retrieval, "max_chunks_per_case", 6),
        pooling_strategy=cfg.dense_retrieval.pooling_strategy,
        batch_size=cfg.dense_retrieval.batch_size,
        device=cfg.dense_retrieval.device,
        normalize_embeddings=cfg.dense_retrieval.normalize_embeddings
    )
    emb_cache = os.path.join(cfg.paths.cache_dir, f"{dataset_name}_embeddings.npz")
    dense.build_index(corpus, cache_path=emb_cache)

    if args.mode == "dense":
        evaluator = IRMetricsEvaluator(relevance_threshold=cfg.evaluation.relevance_threshold)
        test_queries = {qid: queries[qid] for qid in test_qids if qid in queries}
        test_qrels = [q for q in qrels if q.query_id in test_qids]
        dense_res = dense.batch_search(test_queries, top_k=cfg.dense_retrieval.top_k)
        metrics = evaluator.evaluate_run(dense_res, test_qrels)
        logger.info(f"Dense Semantic Test Metrics: {metrics}")
        return

    # 6. Hybrid Fusion Setup
    fusion = HybridFusionRetriever(
        method=cfg.hybrid_fusion.method,
        rrf_k=cfg.hybrid_fusion.rrf_k,
        convex_dense_weight=cfg.hybrid_fusion.convex_dense_weight
    )

    # 7. Feature Extraction & Reranker Training
    feat_extractor = LegalFeatureExtractor(graph_engine=graph)
    reranker = LegalReranker(
        n_estimators=cfg.reranker.n_estimators,
        learning_rate=cfg.reranker.learning_rate,
        max_depth=cfg.reranker.max_depth,
        subsample=cfg.reranker.subsample,
        colsample_bytree=cfg.reranker.colsample_bytree,
        random_state=cfg.project.seed
    )

    # Generate sparse and dense candidate pools for train queries
    max_train_q = getattr(cfg.reranker, "max_train_queries", None)
    if max_train_q and len(train_qids) > max_train_q:
        import random
        rng = random.Random(cfg.project.seed)
        selected_train_qids = rng.sample(train_qids, max_train_q)
        logger.info(f"Subsampled {len(selected_train_qids)} / {len(train_qids)} train queries for reranker training.")
    else:
        selected_train_qids = train_qids

    train_queries = {qid: queries[qid] for qid in selected_train_qids if qid in queries}
    train_qrels = [q for q in qrels if q.query_id in selected_train_qids]
    val_sample_qids = val_qids[:200] if len(val_qids) > 200 else val_qids
    val_queries = {qid: queries[qid] for qid in val_sample_qids if qid in queries}
    val_qrels = [q for q in qrels if q.query_id in val_sample_qids]

    train_sparse = bm25.batch_search(train_queries, top_k=50)
    train_dense = dense.batch_search(train_queries, top_k=50)
    train_df = feat_extractor.build_dataset_features(
        train_queries, corpus, train_sparse, train_dense, candidate_pool_size=50
    )

    val_sparse = bm25.batch_search(val_queries, top_k=50) if val_queries else {}
    val_dense = dense.batch_search(val_queries, top_k=50) if val_queries else {}
    val_df = feat_extractor.build_dataset_features(
        val_queries, corpus, val_sparse, val_dense, candidate_pool_size=50
    ) if val_queries else None

    reranker.fit(train_df, train_qrels, val_df, val_qrels)
    importances = reranker.get_feature_importances()
    logger.info(f"Reranker Top Features: {list(importances.items())[:5]}")

    if args.mode == "rerank":
        reranker.save_model(os.path.join(cfg.paths.output_dir, "reranker_xgboost.pkl"))
        return

    # 8. Full Ablation Study
    ablation = LegalRetrievalAblationStudy(
        corpus=corpus,
        queries=queries,
        qrels=qrels,
        test_query_ids=test_qids,
        train_query_ids=train_qids,
        val_query_ids=val_qids,
        evaluator=IRMetricsEvaluator(relevance_threshold=cfg.evaluation.relevance_threshold)
    )

    results_df = ablation.run_full_ablation(
        bm25_retriever=bm25,
        dense_retriever=dense,
        graph_engine=graph,
        fusion_retriever=fusion,
        reranker=reranker,
        feature_extractor=feat_extractor,
        top_k=50
    )

    print("\n" + "=" * 80)
    print("FINAL EXPERIMENTAL ABLATION RESULTS")
    print("=" * 80)
    print(tabulate(results_df, headers="keys", tablefmt="github", floatfmt=".4f"))
    print("=" * 80 + "\n")

    summary_file = ablation.save_ablation_summary(results_df, output_dir=cfg.paths.output_dir)
    logger.info(f"Pipeline successfully completed! Summary saved to {summary_file}")


if __name__ == "__main__":
    main()
