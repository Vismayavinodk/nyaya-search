"""
LightGBM LambdaMART Learning-to-Rank (LTR) engine for Indian Legal Case Retrieval.
Optimizes listwise ranking (LambdaRank NDCG@10) using lexical BM25, dense semantic,
and citation graph authority features (In-Degree, PageRank, Doc Length).
Trained on train split and tuned on dev split with early stopping.
"""

from __future__ import annotations

import os
import joblib
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import pandas as pd
import lightgbm as lgb

from src.data.schema import Qrel
from src.rerank.feature_extractor import LegalFeatureExtractor
from src.utils.logger import get_logger

logger = get_logger("LightGBMLambdaMART")


class LightGBMLambdaMART:
    """
    LightGBM LambdaMART ranking model for candidate reranking.
    """

    DEFAULT_FEATURES = [
        "bm25_score",
        "dense_score",
        "pagerank",
        "in_degree",
        "doc_length"
    ]

    def __init__(
        self,
        n_estimators: int = 150,
        learning_rate: float = 0.05,
        num_leaves: int = 31,
        max_depth: int = 6,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        min_child_samples: int = 5,
        random_state: int = 42,
        feature_cols: Optional[List[str]] = None
    ):
        self.n_estimators = int(n_estimators)
        self.learning_rate = float(learning_rate)
        self.num_leaves = int(num_leaves)
        self.max_depth = int(max_depth)
        self.subsample = float(subsample)
        self.colsample_bytree = float(colsample_bytree)
        self.min_child_samples = int(min_child_samples)
        self.random_state = int(random_state)
        self.feature_cols = feature_cols or list(self.DEFAULT_FEATURES)

        self.model = lgb.LGBMRanker(
            objective="lambdarank",
            metric="ndcg",
            eval_at=[10],
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            num_leaves=self.num_leaves,
            max_depth=self.max_depth,
            subsample=self.subsample,
            subsample_freq=1 if self.subsample < 1.0 else 0,
            colsample_bytree=self.colsample_bytree,
            min_child_samples=self.min_child_samples,
            random_state=self.random_state,
            n_jobs=-1,
            verbose=-1
        )
        self.is_fitted = False
        self.best_iteration_: Optional[int] = None

    def _prepare_matrix(
        self,
        df: pd.DataFrame,
        qrels: Optional[List[Qrel]] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray], List[int], List[str]]:
        """
        Sorts dataframe by query_id, extracts feature matrix, group sizes, and relevance labels.
        """
        if df.empty:
            return np.empty((0, len(self.feature_cols))), None, [], []

        # Sort dataframe by query_id to ensure contiguous query groups
        df_sorted = df.sort_values(by="query_id").reset_index(drop=True)

        # Check for missing feature columns and impute with 0.0
        for col in self.feature_cols:
            if col not in df_sorted.columns:
                df_sorted[col] = 0.0

        X = df_sorted[self.feature_cols].values.astype(np.float32)

        # Contiguous group sizes (number of candidates per query)
        groups = df_sorted.groupby("query_id", sort=False).size().tolist()
        group_qids = df_sorted["query_id"].drop_duplicates().tolist()

        y = None
        if qrels is not None:
            # Map (query_id, doc_id) -> relevance
            qrel_map = {(q.query_id, q.doc_id): int(q.relevance) for q in qrels}
            y_list = [
                qrel_map.get((row["query_id"], row["doc_id"]), 0)
                for _, row in df_sorted.iterrows()
            ]
            y = np.array(y_list, dtype=np.int32)

        return X, y, groups, group_qids

    def fit(
        self,
        train_df: pd.DataFrame,
        train_qrels: List[Qrel],
        val_df: Optional[pd.DataFrame] = None,
        val_qrels: Optional[List[Qrel]] = None,
        early_stopping_rounds: int = 15
    ) -> "LightGBMLambdaMART":
        """
        Trains LightGBM LambdaMART on train split and tunes stopping iteration on dev split.
        """
        logger.info(
            f"Training LightGBM LambdaMART on {len(train_df)} candidate pairs using {len(self.feature_cols)} features: "
            f"{self.feature_cols}"
        )

        X_train, y_train, group_train, _ = self._prepare_matrix(train_df, train_qrels)
        if len(X_train) == 0:
            raise ValueError("Training DataFrame contains 0 samples.")

        fit_params: Dict[str, Any] = {
            "group": group_train
        }

        callbacks = []
        if val_df is not None and not val_df.empty and val_qrels is not None:
            X_val, y_val, group_val, _ = self._prepare_matrix(val_df, val_qrels)
            fit_params["eval_set"] = [(X_val, y_val)]
            fit_params["eval_group"] = [group_val]
            callbacks.append(lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False))
            logger.info(f"Tuning LambdaMART with early stopping ({early_stopping_rounds} rounds) on {len(val_df)} dev pairs.")

        if callbacks:
            fit_params["callbacks"] = callbacks

        self.model.fit(X_train, y_train, **fit_params)
        self.is_fitted = True
        self.best_iteration_ = getattr(self.model, "best_iteration_", self.n_estimators)
        logger.info(f"LambdaMART training finished. Best iteration: {self.best_iteration_}")
        return self

    def predict_rankings(
        self,
        df: pd.DataFrame,
        top_k: int = 100
    ) -> Dict[str, List[Tuple[str, float]]]:
        """
        Scores candidate pairs and returns ranked list per query sorted by predicted score descending.
        """
        if not self.is_fitted:
            raise RuntimeError("LightGBMLambdaMART model has not been fitted yet.")

        if df.empty:
            return {}

        df_sorted = df.sort_values(by="query_id").reset_index(drop=True)
        for col in self.feature_cols:
            if col not in df_sorted.columns:
                df_sorted[col] = 0.0

        X = df_sorted[self.feature_cols].values.astype(np.float32)
        scores = self.model.predict(X)
        df_sorted["rerank_score"] = scores

        results: Dict[str, List[Tuple[str, float]]] = {}
        for qid, group in df_sorted.groupby("query_id"):
            sorted_docs = group.sort_values(by="rerank_score", ascending=False)
            ranked_pairs = [
                (row["doc_id"], float(row["rerank_score"]))
                for _, row in sorted_docs.head(top_k).iterrows()
            ]
            results[qid] = ranked_pairs

        return results

    def get_feature_importances(self, importance_type: str = "gain") -> Dict[str, float]:
        """
        Returns feature importance scores sorted descending.
        importance_type can be 'gain' or 'split'.
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before obtaining feature importances.")

        booster = self.model.booster_
        raw_importances = booster.feature_importance(importance_type=importance_type)
        feat_map = dict(zip(self.feature_cols, [float(v) for v in raw_importances]))
        return dict(sorted(feat_map.items(), key=lambda x: x[1], reverse=True))

    def save_model(self, filepath: str) -> None:
        """Persists trained model to file."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(
            {
                "model": self.model,
                "feature_cols": self.feature_cols,
                "best_iteration": self.best_iteration_
            },
            filepath
        )
        logger.info(f"Saved LightGBMLambdaMART model to {filepath}")

    def load_model(self, filepath: str) -> None:
        """Loads trained model from file."""
        data = joblib.load(filepath)
        self.model = data["model"]
        self.feature_cols = data["feature_cols"]
        self.best_iteration_ = data.get("best_iteration")
        self.is_fitted = True
        logger.info(f"Loaded LightGBMLambdaMART model from {filepath}")
