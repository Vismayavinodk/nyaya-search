"""
Feature engineering module for legal case reranking.
Constructs rich multi-modal feature vectors combining BM25 lexical signals,
dense semantic similarities, citation graph metrics, and statutory overlaps.
"""

import math
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd

from src.data.schema import LegalCase, LegalQuery
from src.graph.citation_graph import CitationGraphEngine


class LegalFeatureExtractor:
    """
    Extracts structured tabular features for query-candidate document pairs.
    """

    CORE_FEATURE_NAMES = [
        "bm25_score",
        "dense_score",
        "pagerank",
        "in_degree",
        "doc_length"
    ]

    FEATURE_NAMES = [
        "bm25_score",
        "bm25_reciprocal_rank",
        "dense_score",
        "dense_reciprocal_rank",
        "rrf_score",
        "in_degree",
        "in_degree_centrality",
        "pagerank",
        "personalized_pagerank",
        "statute_jaccard_overlap",
        "doc_length",
        "doc_length_log",
        "is_supreme_court",
        "year_delta"
    ]

    def __init__(self, graph_engine: CitationGraphEngine):
        self.graph_engine = graph_engine

    def extract_pair_features(
        self,
        query: LegalQuery,
        candidate_doc: LegalCase,
        bm25_score: float = 0.0,
        bm25_rank: int = 1000,
        dense_score: float = 0.0,
        dense_rank: int = 1000,
        ppr_scores: Optional[Dict[str, float]] = None
    ) -> Dict[str, float]:
        """
        Computes features for a single (query, candidate_case) pair.
        """
        doc_id = candidate_doc.doc_id

        # 1. Lexical retrieval signals
        bm25_recip_rank = 1.0 / (bm25_rank + 1) if bm25_rank < 1000 else 0.0

        # 2. Dense semantic signals
        dense_recip_rank = 1.0 / (dense_rank + 1) if dense_rank < 1000 else 0.0

        # 3. RRF score (k=60)
        rrf = 0.0
        if bm25_rank < 1000:
            rrf += 1.0 / (60.0 + bm25_rank + 1)
        if dense_rank < 1000:
            rrf += 1.0 / (60.0 + dense_rank + 1)

        # 4. Citation Graph Features
        in_degree = self.graph_engine.get_in_degree(doc_id)
        pagerank = self.graph_engine.get_pagerank(doc_id)
        ppr = ppr_scores.get(doc_id, 0.0) if ppr_scores else pagerank

        # 5. Statutory Overlap
        statute_jaccard = self.graph_engine.compute_statute_jaccard(
            query.cited_statutes, doc_id
        )

        # 6. Metadata Features
        text_len = len(candidate_doc.text) if candidate_doc.text else 0
        doc_length_log = math.log1p(text_len)

        is_sc = 1.0 if "supreme" in str(candidate_doc.court).lower() else 0.0
        doc_year = candidate_doc.year or 2000
        year_delta = float(abs(2026 - doc_year))

        return {
            "bm25_score": float(bm25_score),
            "bm25_reciprocal_rank": float(bm25_recip_rank),
            "dense_score": float(dense_score),
            "dense_reciprocal_rank": float(dense_recip_rank),
            "rrf_score": float(rrf),
            "in_degree": float(in_degree),
            "in_degree_centrality": float(in_degree),
            "pagerank": float(pagerank),
            "personalized_pagerank": float(ppr),
            "statute_jaccard_overlap": float(statute_jaccard),
            "doc_length": float(text_len),
            "doc_length_log": float(doc_length_log),
            "is_supreme_court": float(is_sc),
            "year_delta": float(year_delta)
        }

    def build_dataset_features(
        self,
        queries: Dict[str, LegalQuery],
        corpus: Dict[str, LegalCase],
        sparse_results: Dict[str, List[Tuple[str, float]]],
        dense_results: Dict[str, List[Tuple[str, float]]],
        candidate_pool_size: int = 100
    ) -> pd.DataFrame:
        """
        Builds a comprehensive tabular feature DataFrame for all queries.
        Merges top candidates from sparse and dense retrieval up to candidate_pool_size.
        """
        rows: List[Dict[str, Any]] = []

        for qid, query in queries.items():
            s_res = sparse_results.get(qid, [])
            d_res = dense_results.get(qid, [])

            s_dict = {doc_id: (score, rank) for rank, (doc_id, score) in enumerate(s_res)}
            d_dict = {doc_id: (score, rank) for rank, (doc_id, score) in enumerate(d_res)}

            # Candidate pool is the union of top-k sparse and top-k dense
            candidate_ids = list(dict.fromkeys(
                [doc_id for doc_id, _ in s_res[:candidate_pool_size]] +
                [doc_id for doc_id, _ in d_res[:candidate_pool_size]]
            ))[:candidate_pool_size]

            # Compute Personalized PageRank seeded by candidate seeds
            seed_nodes = [doc_id for doc_id, _ in s_res[:5]]
            ppr_scores = self.graph_engine.compute_personalized_pagerank(seed_nodes)

            for doc_id in candidate_ids:
                if doc_id not in corpus:
                    continue
                case = corpus[doc_id]

                s_score, s_rank = s_dict.get(doc_id, (0.0, 1000))
                d_score, d_rank = d_dict.get(doc_id, (0.0, 1000))

                feat_dict = self.extract_pair_features(
                    query=query,
                    candidate_doc=case,
                    bm25_score=s_score,
                    bm25_rank=s_rank,
                    dense_score=d_score,
                    dense_rank=d_rank,
                    ppr_scores=ppr_scores
                )

                feat_dict["query_id"] = qid
                feat_dict["doc_id"] = doc_id
                rows.append(feat_dict)

        df = pd.DataFrame(rows)
        return df

    def build_dataset_features_from_candidates(
        self,
        queries: Dict[str, LegalQuery],
        corpus: Dict[str, LegalCase],
        candidate_rankings: Dict[str, List[Tuple[str, float]]],
        sparse_results: Optional[Dict[str, List[Tuple[str, float]]]] = None,
        dense_results: Optional[Dict[str, List[Tuple[str, float]]]] = None,
        top_k: int = 100
    ) -> pd.DataFrame:
        """
        Builds tabular features directly from hybrid candidate rankings up to top_k (e.g. top-100).
        """
        rows: List[Dict[str, Any]] = []

        for qid, query in queries.items():
            cand_list = candidate_rankings.get(qid, [])[:top_k]
            s_res = (sparse_results or {}).get(qid, [])
            d_res = (dense_results or {}).get(qid, [])

            s_dict = {doc_id: (score, rank) for rank, (doc_id, score) in enumerate(s_res)}
            d_dict = {doc_id: (score, rank) for rank, (doc_id, score) in enumerate(d_res)}

            seed_nodes = [doc_id for doc_id, _ in cand_list[:5]]
            ppr_scores = self.graph_engine.compute_personalized_pagerank(seed_nodes) if seed_nodes else {}

            for rank, (doc_id, hybrid_score) in enumerate(cand_list):
                if doc_id not in corpus:
                    continue
                case = corpus[doc_id]

                s_score, s_rank = s_dict.get(doc_id, (0.0, 1000))
                d_score, d_rank = d_dict.get(doc_id, (0.0, 1000))

                feat_dict = self.extract_pair_features(
                    query=query,
                    candidate_doc=case,
                    bm25_score=s_score,
                    bm25_rank=s_rank,
                    dense_score=d_score,
                    dense_rank=d_rank,
                    ppr_scores=ppr_scores
                )

                feat_dict["query_id"] = qid
                feat_dict["doc_id"] = doc_id
                feat_dict["hybrid_score"] = float(hybrid_score)
                feat_dict["hybrid_rank"] = int(rank)
                rows.append(feat_dict)

        df = pd.DataFrame(rows)
        return df
