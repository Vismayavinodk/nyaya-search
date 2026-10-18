"""
Learning-to-Rank (LTR) reranking engine using XGBoost LambdaMART.
Optimizes ranking using listwise NDCG loss, combining lexical, semantic, and citation graph signals.
"""

import os
import joblib
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import xgboost as xgb

from src.data.schema import Qrel
from src.rerank.feature_extractor import LegalFeatureExtractor
from src.utils.logger import get_logger

logger = get_logger("LegalReranker")


class LegalReranker:
    """
    Supervised Learning-to-Rank model using XGBRanker (LambdaMART).
    """

    def __init__(
        self,
        n_estimators: int = 150,
        learning_rate: float = 0.05,
        max_depth: int = 6,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        random_state: int = 42
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.random_state = random_state

        self.model = xgb.XGBRanker(
            objective="rank:ndcg",
            eval_metric="ndcg@10",
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            random_state=self.random_state
        )
        self.is_fitted = False

    def _prepare_matrix(
        self,
        df: pd.DataFrame,
        qrels: Optional[List[Qrel]] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray], List[int]]:
        """
        Sorts dataframe by query_id, aligns relevance labels, and computes group size vector.
        """
        if df.empty:
            return np.empty((0, len(LegalFeatureExtractor.FEATURE_NAMES))), None, []

        # Sort dataframe by query_id to ensure contiguous query groups
        df_sorted = df.sort_values(by="query_id").reset_index(drop=True)
        feature_cols = LegalFeatureExtractor.FEATURE_NAMES
        X = df_sorted[feature_cols].values

        # Group sizes (number of candidates per query)
        groups = df_sorted.groupby("query_id", sort=False).size().tolist()

        y = None
        if qrels is not None:
            # Map query_id, doc_id -> relevance
            qrel_map = {(q.query_id, q.doc_id): q.relevance for q in qrels}
            y_list = []
            for _, row in df_sorted.iterrows():
                key = (row["query_id"], row["doc_id"])
                y_list.append(qrel_map.get(key, 0))
            y = np.array(y_list, dtype=np.int32)

        return X, y, groups

    def fit(
        self,
        train_df: pd.DataFrame,
        train_qrels: List[Qrel],
        val_df: Optional[pd.DataFrame] = None,
        val_qrels: Optional[List[Qrel]] = None
    ) -> None:
        """
        Trains the LambdaMART ranker using train and validation query splits.
        """
        logger.info(f"Training XGBRanker on {len(train_df)} candidate pairs...")
        X_train, y_train, group_train = self._prepare_matrix(train_df, train_qrels)

        eval_set = None
        if val_df is not None and not val_df.empty and val_qrels is not None:
            X_val, y_val, group_val = self._prepare_matrix(val_df, val_qrels)
            # XGBRanker eval_set requires group information passed to fit
            eval_set = [(X_val, y_val)]
            eval_group = [group_val]
        else:
            eval_group = None

        fit_kwargs = {"group": group_train, "verbose": False}
        if eval_set:
            fit_kwargs["eval_set"] = eval_set
            fit_kwargs["eval_group"] = eval_group

        self.model.fit(X_train, y_train, **fit_kwargs)
        self.is_fitted = True
        logger.info("LambdaMART model training complete.")

    def predict_rankings(
        self,
        df: pd.DataFrame,
        top_k: int = 50
    ) -> Dict[str, List[Tuple[str, float]]]:
        """
        Scores candidate pairs and returns ranked list per query.
        """
        if not self.is_fitted:
            raise RuntimeError("Ranker model not fitted yet.")

        if df.empty:
            return {}

        df_sorted = df.sort_values(by="query_id").reset_index(drop=True)
        feature_cols = LegalFeatureExtractor.FEATURE_NAMES
        X = df_sorted[feature_cols].values

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

    def get_feature_importances(self) -> Dict[str, float]:
        """
        Returns feature importance scores sorted descending by importance.
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted first.")

        importances = self.model.feature_importances_
        feat_imp = dict(zip(LegalFeatureExtractor.FEATURE_NAMES, importances))
        return dict(sorted(feat_imp.items(), key=lambda x: x[1], reverse=True))

    def save_model(self, filepath: str) -> None:
        """Persists trained reranker to disk."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self.model, filepath)
        logger.info(f"Saved reranker model to {filepath}")

    def load_model(self, filepath: str) -> None:
        """Loads trained reranker from disk."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found at {filepath}")
        self.model = joblib.load(filepath)
        self.is_fitted = True
        logger.info(f"Loaded reranker model from {filepath}")
