"""
Graph Score Interpolator for Legal Case Retrieval.
Combines textual/hybrid retrieval rankings with structural citation graph authority signals
(PageRank, HITS Authority, In-Degree Centrality, and Personalized PageRank),
with dev-only weight parameter optimization.
"""

from __future__ import annotations

import os
from typing import Dict, List, Tuple, Any, Optional, TYPE_CHECKING
import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from src.evaluation.eval_harness import EvaluationHarness
    from src.graph.citation_graph import CitationGraphEngine
    from src.data.schema import LegalQuery

from src.utils.logger import get_logger

logger = get_logger("GraphScoreInterpolator")


class GraphScoreInterpolator:
    """
    Interpolates base retrieval candidate scores (e.g. Phase 3 Hybrid or BM25)
    with structural citation authority metrics (PageRank, HITS Authority, In-Degree, PPR).
    """

    def __init__(
        self,
        graph_engine: Optional[CitationGraphEngine] = None,
        beta: float = 0.15,
        graph_signal: str = "pagerank"
    ):
        self.graph = graph_engine
        self.beta = beta
        self.graph_signal = graph_signal.lower()

    def get_graph_score(
        self,
        doc_id: str,
        signal_name: str,
        ppr_cache: Optional[Dict[str, float]] = None
    ) -> float:
        """Retrieves raw graph authority score for a candidate document."""
        if self.graph is None:
            return 0.0

        sig = signal_name.lower()
        if sig == "pagerank":
            return self.graph.get_pagerank(doc_id)
        elif sig in ("hits_authority", "authority"):
            return self.graph.get_hits_authority(doc_id)
        elif sig in ("in_degree", "indegree"):
            return self.graph.get_in_degree(doc_id)
        elif sig in ("personalized_pagerank", "ppr"):
            if ppr_cache is not None:
                return ppr_cache.get(doc_id, 0.0)
            return self.graph.get_pagerank(doc_id)
        elif sig in ("hits_hub", "hub"):
            return self.graph.get_hits_hub(doc_id)
        else:
            return self.graph.get_pagerank(doc_id)

    def interpolate_query(
        self,
        ranked_candidates: List[Tuple[str, float]],
        beta: float,
        signal_name: str,
        top_k: int = 50,
        ppr_cache: Optional[Dict[str, float]] = None
    ) -> List[Tuple[str, float]]:
        """
        Interpolates scores for a single query's candidate list:
        Score(d) = (1 - beta) * norm_base(d) + beta * norm_graph(d)
        """
        if not ranked_candidates:
            return []

        # If beta is 0.0, return original ranking
        if beta <= 0.0:
            return ranked_candidates[:top_k]

        doc_ids = [d for d, _ in ranked_candidates]
        base_scores = np.array([s for _, s in ranked_candidates], dtype=np.float64)
        graph_raw = np.array([
            self.get_graph_score(d, signal_name, ppr_cache=ppr_cache)
            for d in doc_ids
        ], dtype=np.float64)

        # 1. Min-Max normalization for base retrieval scores
        b_min, b_max = base_scores.min(), base_scores.max()
        b_range = (b_max - b_min) if (b_max - b_min) > 1e-9 else 1.0
        norm_base = (base_scores - b_min) / b_range

        # 2. Min-Max normalization for graph authority scores
        g_min, g_max = graph_raw.min(), graph_raw.max()
        g_range = (g_max - g_min) if (g_max - g_min) > 1e-9 else 1.0
        norm_graph = (graph_raw - g_min) / g_range

        # 3. Convex score interpolation
        combined_scores = (1.0 - beta) * norm_base + beta * norm_graph

        # 4. Rank candidates descending by interpolated score
        scored_pairs = list(zip(doc_ids, combined_scores.tolist()))
        scored_pairs.sort(key=lambda x: x[1], reverse=True)
        return scored_pairs[:top_k]

    def interpolate_batch(
        self,
        base_results: Dict[str, List[Tuple[str, float]]],
        beta: Optional[float] = None,
        signal_name: Optional[str] = None,
        top_k: int = 50,
        queries: Optional[Dict[str, LegalQuery]] = None
    ) -> Dict[str, List[Tuple[str, float]]]:
        """
        Applies graph score interpolation across all queries.
        """
        use_beta = beta if beta is not None else self.beta
        use_sig = (signal_name or self.graph_signal).lower()

        interpolated_results: Dict[str, List[Tuple[str, float]]] = {}
        for qid, candidates in base_results.items():
            ppr_dict = None
            if use_sig in ("personalized_pagerank", "ppr") and self.graph is not None:
                top_seed = [candidates[0][0]] if candidates else []
                ppr_dict = self.graph.compute_personalized_pagerank(top_seed)

            interpolated_results[qid] = self.interpolate_query(
                ranked_candidates=candidates,
                beta=use_beta,
                signal_name=use_sig,
                top_k=top_k,
                ppr_cache=ppr_dict
            )

        return interpolated_results

    def tune_interpolation_on_dev(
        self,
        dev_base_results: Dict[str, List[Tuple[str, float]]],
        dev_qrels: Dict[str, Dict[str, int]],
        harness: Optional[EvaluationHarness] = None,
        beta_candidates: List[float] = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50],
        graph_signals: List[str] = ["pagerank", "hits_authority", "in_degree", "personalized_pagerank"],
        top_k: int = 50,
        queries: Optional[Dict[str, LegalQuery]] = None
    ) -> Tuple[Dict[str, Any], pd.DataFrame]:
        """
        Optimizes interpolation weight beta and graph authority signal strictly on dev queries.

        Returns:
            Tuple of:
              - best_config: Dict with optimal 'beta', 'signal', and 'dev_ndcg10'
              - grid_results_df: pd.DataFrame with all trial evaluations
        """
        if harness is None:
            from src.evaluation.eval_harness import EvaluationHarness
            eval_harness = EvaluationHarness()
        else:
            eval_harness = harness

        logger.info(
            f"Starting Graph Score Interpolation tuning on {len(dev_base_results)} dev queries "
            f"({len(beta_candidates)} beta trials × {len(graph_signals)} signals)..."
        )
        records: List[Dict[str, Any]] = []

        for sig in graph_signals:
            for b in beta_candidates:
                b_val = float(round(b, 4))
                interpolated = self.interpolate_batch(
                    base_results=dev_base_results,
                    beta=b_val,
                    signal_name=sig,
                    top_k=top_k,
                    queries=queries
                )
                metrics, _ = eval_harness.evaluate_rankings(interpolated, dev_qrels)
                records.append({
                    "graph_signal": sig,
                    "beta": b_val,
                    "ndcg@10": metrics.get("ndcg@10", 0.0),
                    "map": metrics.get("map", 0.0),
                    "recall@10": metrics.get("recall@10", 0.0),
                    "recall@20": metrics.get("recall@20", 0.0),
                    "mrr": metrics.get("mrr", 0.0)
                })

        df_grid = pd.DataFrame(records).sort_values(by="ndcg@10", ascending=False).reset_index(drop=True)

        best_row = df_grid.iloc[0]
        best_config = {
            "best_beta": float(best_row["beta"]),
            "best_signal": str(best_row["graph_signal"]),
            "best_dev_ndcg10": float(best_row["ndcg@10"]),
            "best_dev_map": float(best_row["map"]),
            "best_dev_recall20": float(best_row["recall@20"]),
            "best_dev_mrr": float(best_row["mrr"])
        }

        logger.info(
            f"Graph Interpolation Tuning Complete on Dev! Best: signal='{best_config['best_signal']}', "
            f"beta={best_config['best_beta']:.2f} (Dev nDCG@10={best_config['best_dev_ndcg10']:.4f})"
        )
        return best_config, df_grid

    @staticmethod
    def save_run_file(
        results: Dict[str, List[Tuple[str, float]]],
        filepath: str,
        run_name: str = "GRAPH_INTERPOLATED"
    ) -> None:
        """
        Exports interpolated retrieval rankings to standard TREC 6-column run format:
        query_id Q0 doc_id rank score run_name
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            for qid in sorted(results.keys()):
                for rank, (doc_id, score) in enumerate(results[qid], start=1):
                    f.write(f"{qid}\tQ0\t{doc_id}\t{rank}\t{score:.6f}\t{run_name}\n")
        logger.info(f"Saved TREC run file with {sum(len(v) for v in results.values())} rows to {filepath}")
