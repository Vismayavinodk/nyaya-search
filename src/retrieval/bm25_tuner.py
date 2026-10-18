"""
BM25 Parameter Optimization Engine on Dev Queries.
Performs grid search over k1 and b parameters strictly on held-out dev split.
"""

from typing import Dict, List, Tuple, Any, Optional
import pandas as pd

from src.retrieval.sparse_bm25 import BM25Retriever
from src.evaluation.eval_harness import EvaluationHarness
from src.data.schema import LegalQuery
from src.utils.logger import get_logger

logger = get_logger("BM25DevTuner")


class BM25DevTuner:
    """
    Sweeps BM25 parameters on dev queries only.
    """

    def __init__(
        self,
        retriever: BM25Retriever,
        dev_queries: Dict[str, LegalQuery],
        dev_qrels: Dict[str, Dict[str, int]],
        harness: Optional[EvaluationHarness] = None
    ):
        self.retriever = retriever
        self.dev_queries = dev_queries
        self.dev_qrels = dev_qrels
        self.harness = harness or EvaluationHarness()

    def tune_bm25_okapi(
        self,
        k1_values: List[float] = [1.2, 1.5, 1.8, 2.0, 2.2],
        b_values: List[float] = [0.55, 0.65, 0.75, 0.85],
        top_k: int = 50
    ) -> Tuple[Dict[str, float], pd.DataFrame]:
        """Grid sweeps k1 and b for BM25 Okapi on dev queries."""
        records = []
        for k1 in k1_values:
            for b in b_values:
                self.retriever.set_params(k1=k1, b=b)
                results = self.retriever.batch_search(self.dev_queries, top_k=top_k)
                metrics, _ = self.harness.evaluate_rankings(results, self.dev_qrels)
                records.append({
                    "k1": k1,
                    "b": b,
                    "ndcg@10": metrics.get("ndcg@10", 0.0),
                    "map": metrics.get("map", 0.0),
                    "recall@10": metrics.get("recall@10", 0.0),
                    "recall@20": metrics.get("recall@20", 0.0),
                    "mrr": metrics.get("mrr", 0.0)
                })

        df_grid = pd.DataFrame(records).sort_values(by="ndcg@10", ascending=False).reset_index(drop=True)
        best_row = df_grid.iloc[0]
        best_params = {"k1": float(best_row["k1"]), "b": float(best_row["b"])}
        logger.info(f"Optimal BM25 Okapi parameters on Dev: {best_params} (Dev nDCG@10={best_row['ndcg@10']:.4f})")
        return best_params, df_grid

    def save_tuning_summary(self, df_grid: pd.DataFrame, filepath: str) -> None:
        """Saves grid evaluation results to CSV."""
        import os
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        df_grid.to_csv(filepath, index=False)
        logger.info(f"Saved BM25 tuning results to {filepath}")
