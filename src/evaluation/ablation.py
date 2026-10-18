"""
Ablation experiment harness for Indian legal case retrieval.
Executes systematic comparisons across BM25, Dense, Hybrid, Citation-Graph, and LTR Reranker.
"""

import os
import json
from typing import Dict, List, Tuple, Any, Optional
import pandas as pd

from src.data.schema import LegalCase, LegalQuery, Qrel
from src.retrieval.sparse_bm25 import BM25Retriever
from src.retrieval.dense_encoder import DenseLegalRetriever
from src.retrieval.hybrid_fusion import HybridFusionRetriever
from src.graph.citation_graph import CitationGraphEngine
from src.rerank.feature_extractor import LegalFeatureExtractor
from src.rerank.ranker import LegalReranker
from src.evaluation.metrics import IRMetricsEvaluator
from src.utils.logger import get_logger

logger = get_logger("AblationStudy")


class LegalRetrievalAblationStudy:
    """
    Automated experimental harness running the full ablation suite:
    1. BM25 Baseline
    2. Dense Semantic Retrieval
    3. Hybrid Fusion (BM25 + Dense RRF)
    4. Hybrid + Citation Graph (Personalized PageRank)
    5. Full Learning-to-Rank Reranker (LambdaMART)
    """

    def __init__(
        self,
        corpus: Dict[str, LegalCase],
        queries: Dict[str, LegalQuery],
        qrels: List[Qrel],
        test_query_ids: List[str],
        train_query_ids: List[str],
        val_query_ids: List[str],
        evaluator: Optional[IRMetricsEvaluator] = None
    ):
        self.corpus = corpus
        self.queries = queries
        self.qrels = qrels
        self.test_query_ids = test_query_ids
        self.train_query_ids = train_query_ids
        self.val_query_ids = val_query_ids
        self.evaluator = evaluator or IRMetricsEvaluator()

        self.test_queries = {qid: queries[qid] for qid in test_query_ids if qid in queries}
        self.test_qrels = [q for q in qrels if q.query_id in self.test_query_ids]

    def run_full_ablation(
        self,
        bm25_retriever: BM25Retriever,
        dense_retriever: DenseLegalRetriever,
        graph_engine: CitationGraphEngine,
        fusion_retriever: HybridFusionRetriever,
        reranker: LegalReranker,
        feature_extractor: LegalFeatureExtractor,
        top_k: int = 50
    ) -> pd.DataFrame:
        """
        Executes all 5 ablation models and compiles comparative metric results.
        """
        logger.info("Executing Retrieval Ablation Suite...")
        results_records = []

        # 1. Sparse BM25
        logger.info("[1/5] Evaluating BM25 Baseline...")
        bm25_test_results = bm25_retriever.batch_search(self.test_queries, top_k=top_k)
        bm25_metrics = self.evaluator.evaluate_run(bm25_test_results, self.test_qrels)
        results_records.append({"Configuration": "1. BM25 Alone", **bm25_metrics})

        # 2. Dense Semantic
        logger.info("[2/5] Evaluating Dense Semantic Baseline...")
        dense_test_results = dense_retriever.batch_search(self.test_queries, top_k=top_k)
        dense_metrics = self.evaluator.evaluate_run(dense_test_results, self.test_qrels)
        results_records.append({"Configuration": "2. Dense Alone", **dense_metrics})

        # 3. Hybrid Fusion (BM25 + Dense RRF)
        logger.info("[3/5] Evaluating Hybrid Fusion (RRF)...")
        hybrid_test_results: Dict[str, List[Tuple[str, float]]] = {}
        for qid in self.test_query_ids:
            s_cand = bm25_test_results.get(qid, [])
            d_cand = dense_test_results.get(qid, [])
            hybrid_test_results[qid] = fusion_retriever.fuse(s_cand, d_cand, top_k=top_k)
        hybrid_metrics = self.evaluator.evaluate_run(hybrid_test_results, self.test_qrels)
        results_records.append({"Configuration": "3. Hybrid (BM25 + Dense RRF)", **hybrid_metrics})

        # 4. Hybrid + Citation Graph (Re-scored by Personalized PageRank)
        logger.info("[4/5] Evaluating Hybrid + Citation Graph (PPR)...")
        graph_augmented_results: Dict[str, List[Tuple[str, float]]] = {}
        for qid in self.test_query_ids:
            h_cand = hybrid_test_results.get(qid, [])
            top_seeds = [did for did, _ in h_cand[:3]]
            ppr_scores = graph_engine.compute_personalized_pagerank(top_seeds)

            # Rerank hybrid candidates using weighted combination of fusion rank and PPR
            rescaled = []
            for rank_0, (did, score) in enumerate(h_cand):
                ppr_val = ppr_scores.get(did, 0.0)
                # Boost candidate if strongly connected to top seeds
                boosted_score = score + 0.3 * ppr_val
                rescaled.append((did, boosted_score))
            rescaled.sort(key=lambda x: x[1], reverse=True)
            graph_augmented_results[qid] = rescaled[:top_k]

        graph_metrics = self.evaluator.evaluate_run(graph_augmented_results, self.test_qrels)
        results_records.append({"Configuration": "4. Hybrid + Citation Graph", **graph_metrics})

        # 5. Full LTR Reranker (LambdaMART)
        logger.info("[5/5] Evaluating Full LambdaMART Reranker...")
        # Prepare test features
        test_df = feature_extractor.build_dataset_features(
            queries=self.test_queries,
            corpus=self.corpus,
            sparse_results=bm25_test_results,
            dense_results=dense_test_results,
            candidate_pool_size=top_k
        )
        reranked_test_results = reranker.predict_rankings(test_df, top_k=top_k)
        rerank_metrics = self.evaluator.evaluate_run(reranked_test_results, self.test_qrels)
        results_records.append({"Configuration": "5. Full LTR Reranker (LambdaMART)", **rerank_metrics})

        df_results = pd.DataFrame(results_records)
        return df_results

    @staticmethod
    def save_ablation_summary(df_results: pd.DataFrame, output_dir: str = "experiments/results") -> str:
        """
        Saves tabular summary as JSON and Markdown.
        """
        os.makedirs(output_dir, exist_ok=True)
        json_path = os.path.join(output_dir, "ablation_results.json")
        df_results.to_json(json_path, orient="records", indent=2)

        md_path = os.path.join(output_dir, "ablation_summary.md")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write("# Ablation Experiment Results\n\n")
            f.write(df_results.to_markdown(index=False))
            f.write("\n\n*Evaluation conducted across nDCG@5, nDCG@10, MAP, Recall@k, and MRR.*\n")

        logger.info(f"Saved ablation summary to {md_path} and {json_path}")
        return md_path
