"""
Hybrid retrieval fusion module.
Implements Reciprocal Rank Fusion (RRF) and Weighted Score Fusion (Convex Combination)
to merge sparse (BM25) and dense (semantic) candidate rankings, with dev-only parameter tuning.
"""

from __future__ import annotations

import os
from typing import Dict, List, Tuple, Any, Optional, TYPE_CHECKING
import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from src.evaluation.eval_harness import EvaluationHarness

from src.utils.logger import get_logger

logger = get_logger("HybridFusion")


class HybridFusionRetriever:
    """
    Combines sparse BM25 and dense semantic retrieval rankings.
    Supports Reciprocal Rank Fusion (RRF) and normalized convex combination,
    with automated dev-split weight optimization.
    """

    def __init__(
        self,
        method: str = "rrf",
        rrf_k: int = 60,
        convex_dense_weight: float = 0.5
    ):
        self.method = method.lower()
        self.rrf_k = rrf_k
        self.convex_dense_weight = convex_dense_weight

    def reciprocal_rank_fusion(
        self,
        sparse_ranked: List[Tuple[str, float]],
        dense_ranked: List[Tuple[str, float]],
        top_k: int = 50,
        rrf_k: Optional[int] = None
    ) -> List[Tuple[str, float]]:
        """
        Computes Reciprocal Rank Fusion score across sparse and dense rankings:
        RRF_Score(d) = 1 / (k + rank_sparse(d)) + 1 / (k + rank_dense(d))
        """
        k = rrf_k if rrf_k is not None else self.rrf_k
        scores: Dict[str, float] = {}

        # 1. Process sparse ranks
        for rank_0, (doc_id, _) in enumerate(sparse_ranked):
            rank = rank_0 + 1
            scores[doc_id] = scores.get(doc_id, 0.0) + (1.0 / (k + rank))

        # 2. Process dense ranks
        for rank_0, (doc_id, _) in enumerate(dense_ranked):
            rank = rank_0 + 1
            scores[doc_id] = scores.get(doc_id, 0.0) + (1.0 / (k + rank))

        sorted_candidates = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_candidates[:top_k]

    def convex_combination(
        self,
        sparse_ranked: List[Tuple[str, float]],
        dense_ranked: List[Tuple[str, float]],
        top_k: int = 50,
        alpha: Optional[float] = None
    ) -> List[Tuple[str, float]]:
        """
        Combines min-max normalized sparse and dense scores linearly:
        Score(d) = alpha * norm_dense(d) + (1 - alpha) * norm_sparse(d)
        """
        dense_weight = alpha if alpha is not None else self.convex_dense_weight
        sparse_dict = dict(sparse_ranked)
        dense_dict = dict(dense_ranked)
        all_doc_ids = set(sparse_dict.keys()).union(set(dense_dict.keys()))

        # Normalize sparse scores per query
        if sparse_dict:
            s_vals = list(sparse_dict.values())
            s_min, s_max = min(s_vals), max(s_vals)
            s_range = (s_max - s_min) if (s_max - s_min) > 1e-9 else 1.0
            norm_sparse = {d: (s - s_min) / s_range for d, s in sparse_dict.items()}
        else:
            norm_sparse = {}

        # Normalize dense scores per query
        if dense_dict:
            d_vals = list(dense_dict.values())
            d_min, d_max = min(d_vals), max(d_vals)
            d_range = (d_max - d_min) if (d_max - d_min) > 1e-9 else 1.0
            norm_dense = {d: (s - d_min) / d_range for d, s in dense_dict.items()}
        else:
            norm_dense = {}

        combined_scores: Dict[str, float] = {}
        for doc_id in all_doc_ids:
            s_score = norm_sparse.get(doc_id, 0.0)
            d_score = norm_dense.get(doc_id, 0.0)
            combined_scores[doc_id] = (dense_weight * d_score) + ((1.0 - dense_weight) * s_score)

        sorted_candidates = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_candidates[:top_k]

    def fuse(
        self,
        sparse_ranked: List[Tuple[str, float]],
        dense_ranked: List[Tuple[str, float]],
        top_k: int = 50
    ) -> List[Tuple[str, float]]:
        """Fuses candidate lists according to configured fusion method."""
        if self.method == "rrf":
            return self.reciprocal_rank_fusion(sparse_ranked, dense_ranked, top_k=top_k)
        elif self.method == "convex":
            return self.convex_combination(sparse_ranked, dense_ranked, top_k=top_k)
        else:
            raise ValueError(f"Unknown fusion method '{self.method}'. Choose 'rrf' or 'convex'.")

    def fuse_batch(
        self,
        sparse_batch: Dict[str, List[Tuple[str, float]]],
        dense_batch: Dict[str, List[Tuple[str, float]]],
        top_k: int = 50,
        method: Optional[str] = None,
        alpha: Optional[float] = None,
        rrf_k: Optional[int] = None
    ) -> Dict[str, List[Tuple[str, float]]]:
        """
        Batch executes hybrid fusion across all queries.
        """
        chosen_method = method.lower() if method else self.method
        all_qids = set(sparse_batch.keys()).union(set(dense_batch.keys()))
        fused_results: Dict[str, List[Tuple[str, float]]] = {}

        for qid in all_qids:
            s_cand = sparse_batch.get(qid, [])
            d_cand = dense_batch.get(qid, [])
            if chosen_method == "rrf":
                fused_results[qid] = self.reciprocal_rank_fusion(s_cand, d_cand, top_k=top_k, rrf_k=rrf_k)
            elif chosen_method == "convex":
                fused_results[qid] = self.convex_combination(s_cand, d_cand, top_k=top_k, alpha=alpha)
            else:
                fused_results[qid] = self.reciprocal_rank_fusion(s_cand, d_cand, top_k=top_k, rrf_k=rrf_k)

        return fused_results

    def tune_weights_on_dev(
        self,
        dev_sparse_results: Dict[str, List[Tuple[str, float]]],
        dev_dense_results: Dict[str, List[Tuple[str, float]]],
        dev_qrels: Dict[str, Dict[str, int]],
        harness: Optional[EvaluationHarness] = None,
        alpha_steps: int = 21,
        rrf_k_candidates: List[int] = [10, 20, 40, 60, 80, 100],
        top_k: int = 50
    ) -> Tuple[Dict[str, Any], pd.DataFrame]:
        """
        Tunes weighted score fusion (alpha) and RRF smoothing constant (k) on dev queries only.

        Returns:
            Tuple of:
              - best_config: Dict with optimal 'alpha' and 'rrf_k'
              - grid_results_df: pd.DataFrame with all trial evaluations
        """
        if harness is None:
            from src.evaluation.eval_harness import EvaluationHarness
            eval_harness = EvaluationHarness()
        else:
            eval_harness = harness
        logger.info(
            f"Starting Hybrid Fusion tuning on {len(dev_sparse_results)} dev queries "
            f"({alpha_steps} alpha steps + {len(rrf_k_candidates)} RRF-k trials)..."
        )
        records: List[Dict[str, Any]] = []

        # 1. Sweep Convex Combination Dense Weight alpha in [0.0, 1.0]
        alphas = np.linspace(0.0, 1.0, alpha_steps)
        for a in alphas:
            a_val = float(round(a, 4))
            fused = self.fuse_batch(
                dev_sparse_results, dev_dense_results,
                top_k=top_k, method="convex", alpha=a_val
            )
            metrics, _ = eval_harness.evaluate_rankings(fused, dev_qrels)
            records.append({
                "method": "convex_weighted",
                "param_name": "alpha_dense_weight",
                "param_value": a_val,
                "ndcg@10": metrics.get("ndcg@10", 0.0),
                "map": metrics.get("map", 0.0),
                "recall@10": metrics.get("recall@10", 0.0),
                "recall@20": metrics.get("recall@20", 0.0),
                "mrr": metrics.get("mrr", 0.0)
            })

        # 2. Sweep RRF smoothing constant k in candidates
        for k_val in rrf_k_candidates:
            fused = self.fuse_batch(
                dev_sparse_results, dev_dense_results,
                top_k=top_k, method="rrf", rrf_k=k_val
            )
            metrics, _ = eval_harness.evaluate_rankings(fused, dev_qrels)
            records.append({
                "method": "reciprocal_rank_fusion",
                "param_name": "rrf_k",
                "param_value": k_val,
                "ndcg@10": metrics.get("ndcg@10", 0.0),
                "map": metrics.get("map", 0.0),
                "recall@10": metrics.get("recall@10", 0.0),
                "recall@20": metrics.get("recall@20", 0.0),
                "mrr": metrics.get("mrr", 0.0)
            })

        df_grid = pd.DataFrame(records).sort_values(by="ndcg@10", ascending=False).reset_index(drop=True)

        # Extract best alpha and best rrf_k
        convex_rows = df_grid[df_grid["method"] == "convex_weighted"]
        best_alpha = float(convex_rows.iloc[0]["param_value"]) if not convex_rows.empty else 0.5
        best_alpha_ndcg = float(convex_rows.iloc[0]["ndcg@10"]) if not convex_rows.empty else 0.0

        rrf_rows = df_grid[df_grid["method"] == "reciprocal_rank_fusion"]
        best_rrf_k = int(rrf_rows.iloc[0]["param_value"]) if not rrf_rows.empty else 60
        best_rrf_ndcg = float(rrf_rows.iloc[0]["ndcg@10"]) if not rrf_rows.empty else 0.0

        best_config = {
            "best_alpha": best_alpha,
            "best_alpha_ndcg10": best_alpha_ndcg,
            "best_rrf_k": best_rrf_k,
            "best_rrf_ndcg10": best_rrf_ndcg
        }

        logger.info(
            f"Hybrid Fusion Tuning Complete on Dev! Best Convex: alpha={best_alpha} (nDCG@10={best_alpha_ndcg:.4f}) | "
            f"Best RRF: k={best_rrf_k} (nDCG@10={best_rrf_ndcg:.4f})"
        )
        return best_config, df_grid

    @staticmethod
    def save_run_file(
        results: Dict[str, List[Tuple[str, float]]],
        filepath: str,
        run_name: str = "HYBRID"
    ) -> None:
        """
        Exports fused retrieval rankings to standard TREC 6-column run format:
        query_id Q0 doc_id rank score run_name
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            for qid in sorted(results.keys()):
                for rank, (doc_id, score) in enumerate(results[qid], start=1):
                    f.write(f"{qid}\tQ0\t{doc_id}\t{rank}\t{score:.6f}\t{run_name}\n")
        logger.info(f"Saved TREC run file with {sum(len(v) for v in results.values())} rows to {filepath}")
