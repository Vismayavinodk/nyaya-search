"""
Information Retrieval engines: Sparse BM25, Dense Semantic Encoders, and Hybrid Fusion.
"""

from src.retrieval.sparse_bm25 import BM25Retriever
from src.retrieval.faiss_index import LegalFAISSIndex
from src.retrieval.dense_encoder import DenseLegalRetriever
from src.retrieval.hybrid_fusion import HybridFusionRetriever

__all__ = ["BM25Retriever", "LegalFAISSIndex", "DenseLegalRetriever", "HybridFusionRetriever"]
