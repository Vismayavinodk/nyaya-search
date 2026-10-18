"""
Cross-Encoder reranker module for legal case retrieval.
Applies transformer-based cross-encoders (e.g. BAAI/bge-reranker-base)
to rerank top-K candidates from the first-stage hybrid retrieval pool.
"""

from __future__ import annotations

import os
import torch
from typing import Dict, List, Tuple, Optional, Any
import numpy as np

from src.data.schema import LegalCase, LegalQuery
from src.utils.logger import get_logger

logger = get_logger("CrossEncoderReranker")


class CrossEncoderReranker:
    """
    Reranks candidate legal precedents using a deep cross-encoder.
    Supports BAAI/bge-reranker-base and sentence-transformers CrossEncoder.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-base",
        max_length: int = 256,
        batch_size: int = 64,
        device: Optional[str] = None,
        max_doc_chars: int = 1000
    ):
        self.model_name = model_name
        self.max_length = int(max_length)
        self.batch_size = int(batch_size)
        self.max_doc_chars = int(max_doc_chars)

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.model = None
        self._load_model()

    def _load_model(self) -> None:
        """Loads the cross-encoder model, falling back gracefully if necessary."""
        try:
            from sentence_transformers import CrossEncoder
            logger.info(f"Loading CrossEncoder '{self.model_name}' on device '{self.device}'...")
            self.model = CrossEncoder(
                self.model_name,
                max_length=self.max_length,
                device=self.device
            )
            logger.info("CrossEncoder successfully loaded.")
        except Exception as e:
            logger.warning(
                f"Failed to load primary CrossEncoder '{self.model_name}': {e}. "
                f"Attempting fallback to 'cross-encoder/ms-marco-MiniLM-L-6-v2'..."
            )
            try:
                from sentence_transformers import CrossEncoder
                self.model = CrossEncoder(
                    "cross-encoder/ms-marco-MiniLM-L-6-v2",
                    max_length=self.max_length,
                    device=self.device
                )
                self.model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
                logger.info("Fallback CrossEncoder loaded successfully.")
            except Exception as e2:
                logger.error(f"Failed to load fallback CrossEncoder: {e2}. Running in heuristic fallback mode.")
                self.model = None

    def rerank_query_candidates(
        self,
        query: LegalQuery,
        candidate_docs: List[Tuple[str, float]],
        corpus: Dict[str, LegalCase],
        top_k: int = 50
    ) -> List[Tuple[str, float]]:
        """
        Reranks candidates for a single query.
        """
        return self.rerank_batch(
            queries={query.query_id: query},
            corpus=corpus,
            candidate_rankings={query.query_id: candidate_docs},
            top_k=top_k
        ).get(query.query_id, candidate_docs[:top_k])

    def rerank_batch(
        self,
        queries: Dict[str, LegalQuery],
        corpus: Dict[str, LegalCase],
        candidate_rankings: Dict[str, List[Tuple[str, float]]],
        top_k: int = 50
    ) -> Dict[str, List[Tuple[str, float]]]:
        """
        Reranks top-K candidates across all queries in candidate_rankings using globally batched inference.
        """
        logger.info(
            f"Cross-encoder reranking top-{top_k} candidates for {len(candidate_rankings)} queries "
            f"using '{self.model_name}' (batch_size={self.batch_size})..."
        )
        all_pairs: List[Tuple[str, str]] = []
        pair_metadata: List[Tuple[str, str, float]] = []

        for qid, cands in candidate_rankings.items():
            if qid not in queries:
                continue
            q = queries[qid]
            q_text = q.text.strip()
            for did, orig_score in cands[:top_k]:
                if did not in corpus:
                    continue
                doc = corpus[did]
                doc_text = doc.text[:self.max_doc_chars].strip() if doc.text else ""
                if doc.title:
                    doc_text = f"{doc.title}. {doc_text}"
                all_pairs.append((q_text, doc_text))
                pair_metadata.append((qid, did, float(orig_score)))

        if not all_pairs or self.model is None:
            return {qid: cands[:top_k] for qid, cands in candidate_rankings.items()}

        try:
            scores = self.model.predict(
                all_pairs,
                batch_size=self.batch_size,
                show_progress_bar=False
            )
        except Exception as e:
            logger.warning(f"Batched CrossEncoder inference failed ({e}), retaining base scores.")
            return {qid: cands[:top_k] for qid, cands in candidate_rankings.items()}

        # Group and rank by query_id
        reranked_dict: Dict[str, List[Tuple[str, float]]] = {qid: [] for qid in candidate_rankings}
        for (qid, did, _), s in zip(pair_metadata, scores):
            reranked_dict[qid].append((did, float(s)))

        for qid in reranked_dict:
            reranked_dict[qid].sort(key=lambda x: x[1], reverse=True)
            reranked_dict[qid] = reranked_dict[qid][:top_k]

        return reranked_dict
