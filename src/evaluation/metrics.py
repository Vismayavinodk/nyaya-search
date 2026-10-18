"""
Information Retrieval Evaluation Metrics.
Strict implementations of nDCG@k, MAP, Recall@k, and MRR.
"""

import math
from typing import Dict, List, Tuple, Set
import numpy as np

from src.data.schema import Qrel


class IRMetricsEvaluator:
    """
    Computes standard academic IR evaluation metrics across queries.
    """

    def __init__(self, relevance_threshold: int = 1):
        self.relevance_threshold = relevance_threshold

    @staticmethod
    def dcg_at_k(relevances: List[int], k: int) -> float:
        """
        Computes Discounted Cumulative Gain at rank k using 2^rel - 1 formulation.
        """
        dcg = 0.0
        for i, rel in enumerate(relevances[:k]):
            dcg += (math.pow(2.0, rel) - 1.0) / math.log2(i + 2.0)
        return dcg

    def ndcg_at_k(self, retrieved_doc_ids: List[str], ground_truth: Dict[str, int], k: int) -> float:
        """
        Computes Normalized Discounted Cumulative Gain at rank k.
        """
        if not ground_truth:
            return 0.0

        # Actual relevance vector
        actual_rels = [ground_truth.get(did, 0) for did in retrieved_doc_ids[:k]]
        dcg = self.dcg_at_k(actual_rels, k)

        # Ideal relevance vector
        ideal_rels = sorted(list(ground_truth.values()), reverse=True)[:k]
        idcg = self.dcg_at_k(ideal_rels, k)

        if idcg <= 0.0:
            return 0.0
        return float(dcg / idcg)

    def average_precision(self, retrieved_doc_ids: List[str], relevant_docs: Set[str]) -> float:
        """
        Computes Average Precision (AP) for a single query.
        """
        if not relevant_docs:
            return 0.0

        num_hits = 0
        sum_precisions = 0.0

        for i, did in enumerate(retrieved_doc_ids):
            if did in relevant_docs:
                num_hits += 1
                precision_at_i = num_hits / (i + 1.0)
                sum_precisions += precision_at_i

        return float(sum_precisions / len(relevant_docs))

    def recall_at_k(self, retrieved_doc_ids: List[str], relevant_docs: Set[str], k: int) -> float:
        """
        Computes Recall at rank k.
        """
        if not relevant_docs:
            return 0.0

        retrieved_set = set(retrieved_doc_ids[:k])
        hits = len(retrieved_set.intersection(relevant_docs))
        return float(hits / len(relevant_docs))

    def reciprocal_rank(self, retrieved_doc_ids: List[str], relevant_docs: Set[str]) -> float:
        """
        Computes Reciprocal Rank (RR).
        """
        if not relevant_docs:
            return 0.0

        for i, did in enumerate(retrieved_doc_ids):
            if did in relevant_docs:
                return float(1.0 / (i + 1.0))
        return 0.0

    def evaluate_query(
        self,
        retrieved_doc_ids: List[str],
        query_qrels: Dict[str, int]
    ) -> Dict[str, float]:
        """
        Computes all standard metrics for a single query's retrieved ranked list.
        """
        relevant_docs = {did for did, rel in query_qrels.items() if rel >= self.relevance_threshold}

        return {
            "ndcg@5": self.ndcg_at_k(retrieved_doc_ids, query_qrels, k=5),
            "ndcg@10": self.ndcg_at_k(retrieved_doc_ids, query_qrels, k=10),
            "map": self.average_precision(retrieved_doc_ids, relevant_docs),
            "recall@5": self.recall_at_k(retrieved_doc_ids, relevant_docs, k=5),
            "recall@10": self.recall_at_k(retrieved_doc_ids, relevant_docs, k=10),
            "recall@20": self.recall_at_k(retrieved_doc_ids, relevant_docs, k=20),
            "mrr": self.reciprocal_rank(retrieved_doc_ids, relevant_docs)
        }

    def evaluate_run(
        self,
        run_results: Dict[str, List[Tuple[str, float]]],
        all_qrels: List[Qrel]
    ) -> Dict[str, float]:
        """
        Macro-averages evaluation metrics over all evaluated queries.

        Args:
            run_results: Map of query_id -> List of (doc_id, score).
            all_qrels: List of ground-truth Qrel objects.

        Returns:
            Dictionary of mean metric scores.
        """
        # Group qrels by query_id
        qrel_by_query: Dict[str, Dict[str, int]] = {}
        for q in all_qrels:
            if q.query_id not in qrel_by_query:
                qrel_by_query[q.query_id] = {}
            qrel_by_query[q.query_id][q.doc_id] = q.relevance

        per_query_metrics: List[Dict[str, float]] = []

        for qid, scored_pairs in run_results.items():
            if qid not in qrel_by_query:
                continue
            retrieved_ids = [did for did, _ in scored_pairs]
            q_metrics = self.evaluate_query(retrieved_ids, qrel_by_query[qid])
            per_query_metrics.append(q_metrics)

        if not per_query_metrics:
            return {
                "ndcg@5": 0.0, "ndcg@10": 0.0, "map": 0.0,
                "recall@5": 0.0, "recall@10": 0.0, "recall@20": 0.0, "mrr": 0.0
            }

        # Average across all queries
        mean_metrics: Dict[str, float] = {}
        metric_keys = per_query_metrics[0].keys()
        for k in metric_keys:
            mean_metrics[k] = float(np.mean([m[k] for m in per_query_metrics]))

        return mean_metrics
