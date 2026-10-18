"""
Reusable Evaluation Harness for Legal Case Retrieval.
Reads standard TREC 6-column run files and qrels files, computes nDCG@10, MAP, Recall@k, MRR,
outputs per-query score CSVs for statistical significance tests, and runs paired significance testing.
"""

import os
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
from scipy import stats

from src.evaluation.metrics import IRMetricsEvaluator
from src.utils.logger import get_logger

logger = get_logger("EvaluationHarness")


class EvaluationHarness:
    """
    Standardized, reusable evaluation harness for comparing IR runs.
    """

    def __init__(self, relevance_threshold: int = 1):
        self.relevance_threshold = relevance_threshold
        self.evaluator = IRMetricsEvaluator(relevance_threshold=relevance_threshold)

    def load_run_file(self, filepath: str) -> Dict[str, List[Tuple[str, float]]]:
        """
        Parses a standard 6-column TREC run file:
        query_id Q0 doc_id rank score run_name
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Run file not found at: {filepath}")

        run_dict: Dict[str, List[Tuple[str, float]]] = {}
        row_count = 0
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) >= 6:
                    qid, _, did, _, score_str, _ = parts[:6]
                elif len(parts) >= 4:
                    qid, did, _, score_str = parts[:4]
                else:
                    continue

                score = float(score_str)
                if qid not in run_dict:
                    run_dict[qid] = []
                run_dict[qid].append((did, score))
                row_count += 1

        for qid in run_dict:
            run_dict[qid].sort(key=lambda x: x[1], reverse=True)

        logger.info(f"Loaded run file '{filepath}': {len(run_dict)} queries, {row_count} rankings.")
        return run_dict

    def load_qrels_file(self, filepath: str) -> Dict[str, Dict[str, int]]:
        """
        Parses a standard 4-column TREC qrels file:
        query_id 0 doc_id relevance
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Qrels file not found at: {filepath}")

        qrels_dict: Dict[str, Dict[str, int]] = {}
        count = 0
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) >= 4:
                    qid, _, did, rel_str = parts[:4]
                elif len(parts) == 3:
                    qid, did, rel_str = parts[:3]
                else:
                    continue

                rel = int(rel_str)
                if qid not in qrels_dict:
                    qrels_dict[qid] = {}
                qrels_dict[qid][did] = rel
                count += 1

        logger.info(f"Loaded qrels file '{filepath}': {len(qrels_dict)} queries, {count} ground-truth judgments.")
        return qrels_dict

    def evaluate_rankings(
        self,
        rankings: Dict[str, List[Tuple[str, float]]],
        qrels: Dict[str, Dict[str, int]]
    ) -> Tuple[Dict[str, float], pd.DataFrame]:
        """
        Evaluates retrieval rankings across queries.
        Returns:
            Tuple of (mean_metrics_dict, per_query_dataframe)
        """
        query_ids = [qid for qid in rankings.keys() if qid in qrels]
        if not query_ids:
            cols = ["query_id", "ndcg@5", "ndcg@10", "map", "recall@5", "recall@10", "recall@20", "mrr"]
            return {}, pd.DataFrame(columns=cols)

        per_query_records = []
        for qid in query_ids:
            cand_list = rankings[qid]
            q_gold = qrels[qid]
            ranked_doc_ids = [doc_id for doc_id, _ in cand_list]

            q_metrics = self.evaluator.evaluate_query(ranked_doc_ids, q_gold)
            per_query_records.append({
                "query_id": qid,
                **q_metrics
            })

        df_per_query = pd.DataFrame(per_query_records)
        mean_metrics = {
            col: float(df_per_query[col].mean())
            for col in df_per_query.columns if col != "query_id"
        }

        return mean_metrics, df_per_query

    def evaluate_run_file(
        self,
        run_file_path: str,
        qrels_file_path: str,
        per_query_output_path: Optional[str] = None
    ) -> Tuple[Dict[str, float], pd.DataFrame]:
        """Evaluates a run file against qrels file, optionally saving per-query CSV."""
        run_dict = self.load_run_file(run_file_path)
        qrels_dict = self.load_qrels_file(qrels_file_path)

        mean_metrics, df_per_query = self.evaluate_rankings(run_dict, qrels_dict)

        if per_query_output_path:
            os.makedirs(os.path.dirname(per_query_output_path), exist_ok=True)
            df_per_query.to_csv(per_query_output_path, index=False)
            logger.info(f"Saved per-query evaluation scores ({len(df_per_query)} queries) to: {per_query_output_path}")

        return mean_metrics, df_per_query

    @staticmethod
    def compute_statistical_significance(
        df_baseline: pd.DataFrame,
        df_treatment: pd.DataFrame,
        metric: str = "ndcg@10"
    ) -> Dict[str, Any]:
        """
        Computes paired Student's t-test and Wilcoxon signed-rank test.
        """
        if df_baseline.empty or df_treatment.empty or metric not in df_baseline.columns or metric not in df_treatment.columns:
            return {
                "metric": metric,
                "queries_evaluated": 0,
                "mean_baseline": 0.0,
                "mean_treatment": 0.0,
                "relative_gain_percent": 0.0,
                "t_statistic": 0.0,
                "t_pvalue": 1.0,
                "wilcoxon_statistic": 0.0,
                "wilcoxon_pvalue": 1.0,
                "is_significant_p05": False
            }

        merged = pd.merge(
            df_baseline[["query_id", metric]],
            df_treatment[["query_id", metric]],
            on="query_id",
            suffixes=("_baseline", "_treatment")
        )

        base_scores = merged[f"{metric}_baseline"].values
        treat_scores = merged[f"{metric}_treatment"].values

        base_mean = float(np.mean(base_scores))
        treat_mean = float(np.mean(treat_scores))
        rel_gain = float(((treat_mean - base_mean) / (base_mean + 1e-12)) * 100.0)

        # Paired t-test
        diffs = treat_scores - base_scores
        if np.all(diffs == 0.0) or len(diffs) <= 1:
            t_stat, t_pval = 0.0, 1.0
        else:
            try:
                t_res = stats.ttest_rel(treat_scores, base_scores)
                t_stat, t_pval = float(t_res.statistic), float(t_res.pvalue)
            except Exception:
                t_stat, t_pval = 0.0, 1.0

        # Wilcoxon test
        if np.all(diffs == 0.0) or len(diffs) <= 1:
            w_stat, w_pval = 0.0, 1.0
        else:
            try:
                w_res = stats.wilcoxon(treat_scores, base_scores)
                w_stat, w_pval = float(w_res.statistic), float(w_res.pvalue)
            except Exception:
                w_stat, w_pval = 0.0, 1.0

        # Paired Bootstrap Test (1,000 resamples)
        n_boot = 1000
        rng = np.random.RandomState(42)
        n_samples = len(diffs)
        if n_samples > 1 and not np.all(diffs == 0.0):
            sample_indices = rng.randint(0, n_samples, size=(n_boot, n_samples))
            boot_diffs = diffs[sample_indices].mean(axis=1)
            boot_p = float(2.0 * min(np.mean(boot_diffs <= 0.0), np.mean(boot_diffs >= 0.0)))
            boot_p = min(1.0, max(0.0, boot_p))
            ci_lower = float(np.percentile(boot_diffs, 2.5))
            ci_upper = float(np.percentile(boot_diffs, 97.5))
        else:
            boot_p, ci_lower, ci_upper = 1.0, 0.0, 0.0

        return {
            "metric": metric,
            "queries_evaluated": len(merged),
            "mean_baseline": round(base_mean, 4),
            "mean_treatment": round(treat_mean, 4),
            "relative_gain_percent": round(rel_gain, 4),
            "t_statistic": round(t_stat, 4),
            "t_pvalue": round(t_pval, 4),
            "wilcoxon_statistic": round(w_stat, 4),
            "wilcoxon_pvalue": round(w_pval, 4),
            "bootstrap_pvalue": round(boot_p, 4),
            "bootstrap_ci_lower": round(ci_lower, 4),
            "bootstrap_ci_upper": round(ci_upper, 4),
            "is_significant_p05": bool(t_pval < 0.05 or boot_p < 0.05)
        }
