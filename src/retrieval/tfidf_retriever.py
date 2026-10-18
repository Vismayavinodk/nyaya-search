"""
TF-IDF baseline retriever using scikit-learn TfidfVectorizer.
"""

import os
from typing import Dict, List, Tuple, Optional
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from src.data.schema import LegalCase, LegalQuery
from src.utils.logger import get_logger

logger = get_logger("TFIDFRetriever")


class TFIDFRetriever:
    """
    Standard TF-IDF vector space baseline for legal case retrieval.
    """

    def __init__(
        self,
        sublinear_tf: bool = True,
        max_features: int = 50000,
        ngram_range: Tuple[int, int] = (1, 2)
    ):
        self.vectorizer = TfidfVectorizer(
            sublinear_tf=sublinear_tf,
            max_features=max_features,
            ngram_range=ngram_range,
            lowercase=True
        )
        self.doc_ids: List[str] = []
        self.tfidf_matrix = None

    def build_index(self, corpus: Dict[str, LegalCase]) -> None:
        """Fits TF-IDF vectorizer and transforms corpus."""
        self.doc_ids = list(corpus.keys())
        texts = [corpus[did].text for did in self.doc_ids]
        self.tfidf_matrix = self.vectorizer.fit_transform(texts)
        logger.info(f"Built TF-IDF index for {len(self.doc_ids)} documents with {self.tfidf_matrix.shape[1]} features.")

    def batch_search(
        self,
        queries: Dict[str, LegalQuery],
        top_k: int = 50
    ) -> Dict[str, List[Tuple[str, float]]]:
        """Performs batch cosine similarity search."""
        if self.tfidf_matrix is None:
            raise RuntimeError("Index not built. Call build_index() first.")

        qids = list(queries.keys())
        q_texts = [queries[qid].full_query_text for qid in qids]
        q_vecs = self.vectorizer.transform(q_texts)

        sim_matrix = q_vecs.dot(self.tfidf_matrix.T).toarray()

        results: Dict[str, List[Tuple[str, float]]] = {}
        for i, qid in enumerate(qids):
            scores = sim_matrix[i]
            ranked_indices = np.argsort(scores)[::-1][:top_k]
            results[qid] = [
                (self.doc_ids[idx], float(scores[idx]))
                for idx in ranked_indices
            ]

        return results

    def save_run_file(
        self,
        results: Dict[str, List[Tuple[str, float]]],
        filepath: str,
        run_name: str = "TFIDF"
    ) -> None:
        """Saves rankings to TREC 6-column run file."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            for qid in sorted(results.keys()):
                for rank, (doc_id, score) in enumerate(results[qid], start=1):
                    f.write(f"{qid}\tQ0\t{doc_id}\t{rank}\t{score:.6f}\t{run_name}\n")
        logger.info(f"Saved TREC run file to {filepath}")
