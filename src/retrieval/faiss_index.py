"""
High-performance FAISS vector index wrapper for legal case retrieval.
Supports Inner Product (Cosine similarity for L2-normalized embeddings) and HNSW,
with CPU/GPU automatic device allocation and disk persistence.
"""

import os
import pickle
from typing import Dict, List, Tuple, Optional
import numpy as np

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

from src.utils.logger import get_logger

logger = get_logger("LegalFAISSIndex")


class LegalFAISSIndex:
    """
    FAISS vector index engine for fast similarity search over legal case embeddings.
    """

    def __init__(
        self,
        dimension: int,
        index_type: str = "flat_ip",
        use_gpu: bool = False
    ):
        self.dimension = dimension
        self.index_type = index_type.lower()
        self.use_gpu = use_gpu

        self.index: Optional[faiss.Index] = None if FAISS_AVAILABLE else None
        self.doc_ids: List[str] = []
        self.is_built: bool = False

    def build(self, embeddings: np.ndarray, doc_ids: List[str]) -> None:
        """
        Builds the FAISS index from normalized embedding vectors.

        Args:
            embeddings: Float32 array of shape (N, dimension).
            doc_ids: List of string document IDs of length N.
        """
        if not FAISS_AVAILABLE:
            raise RuntimeError("FAISS is not installed. Please install faiss-cpu or faiss-gpu.")

        N, D = embeddings.shape
        if D != self.dimension:
            raise ValueError(f"Embedding dimension ({D}) does not match index dimension ({self.dimension})")
        if len(doc_ids) != N:
            raise ValueError(f"doc_ids length ({len(doc_ids)}) does not match embeddings count ({N})")

        self.doc_ids = list(doc_ids)
        vecs = embeddings.astype(np.float32).copy()

        # Build index
        if self.index_type == "flat_ip":
            cpu_index = faiss.IndexFlatIP(self.dimension)
        elif self.index_type == "hnsw":
            cpu_index = faiss.IndexHNSWFlat(self.dimension, 32, faiss.METRIC_INNER_PRODUCT)
        else:
            cpu_index = faiss.IndexFlatIP(self.dimension)

        # GPU acceleration if requested and available
        if self.use_gpu and hasattr(faiss, "StandardGpuResources"):
            try:
                res = faiss.StandardGpuResources()
                self.index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
                logger.info(f"Initialized GPU FAISS ({self.index_type}) index on device 0.")
            except Exception as e:
                logger.warning(f"Could not initialize GPU FAISS ({e}), defaulting to CPU.")
                self.index = cpu_index
        else:
            self.index = cpu_index

        self.index.add(vecs)
        self.is_built = True
        logger.info(f"FAISS index built with {self.index.ntotal} vectors of dimension {self.dimension}.")

    def search(
        self,
        query_embeddings: np.ndarray,
        top_k: int = 100
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Searches FAISS index with a batch of query vectors.

        Args:
            query_embeddings: Float32 array of shape (num_queries, dimension).
            top_k: Number of nearest documents to retrieve.

        Returns:
            Tuple of (scores_matrix, indices_matrix) of shape (num_queries, top_k).
        """
        if not self.is_built or self.index is None:
            raise RuntimeError("FAISS index not built. Call build() or load() first.")

        q_vecs = query_embeddings.astype(np.float32).copy()
        scores, indices = self.index.search(q_vecs, top_k)
        return scores, indices

    def search_queries(
        self,
        query_embeddings: np.ndarray,
        query_ids: List[str],
        top_k: int = 100
    ) -> Dict[str, List[Tuple[str, float]]]:
        """
        Executes batch search and maps indices to document IDs.

        Returns:
            Dictionary mapping query_id -> List of (doc_id, cosine_score).
        """
        scores, indices = self.search(query_embeddings, top_k=top_k)

        results: Dict[str, List[Tuple[str, float]]] = {}
        for i, qid in enumerate(query_ids):
            q_results: List[Tuple[str, float]] = []
            for j in range(scores.shape[1]):
                idx = indices[i, j]
                if 0 <= idx < len(self.doc_ids):
                    doc_id = self.doc_ids[idx]
                    score = float(scores[i, j])
                    q_results.append((doc_id, score))
            results[qid] = q_results

        return results

    def save(self, filepath: str) -> None:
        """Saves FAISS index and doc_ids metadata to disk."""
        if not self.is_built or self.index is None:
            raise RuntimeError("Cannot save unbuilt FAISS index.")

        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        # Convert to CPU index for serialization if GPU
        cpu_index = faiss.index_gpu_to_cpu(self.index) if hasattr(faiss, "index_gpu_to_cpu") and self.use_gpu else self.index
        faiss.write_index(cpu_index, filepath)

        meta_path = filepath + ".meta"
        with open(meta_path, "wb") as f:
            pickle.dump({
                "dimension": self.dimension,
                "index_type": self.index_type,
                "doc_ids": self.doc_ids
            }, f)
        logger.info(f"Saved FAISS index ({self.index.ntotal} vectors) to {filepath} and metadata to {meta_path}")

    def load(self, filepath: str) -> None:
        """Loads FAISS index and doc_ids metadata from disk."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"FAISS index file not found at {filepath}")

        meta_path = filepath + ".meta"
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"FAISS metadata file not found at {meta_path}")

        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
            self.dimension = meta["dimension"]
            self.index_type = meta["index_type"]
            self.doc_ids = meta["doc_ids"]

        cpu_index = faiss.read_index(filepath)
        if self.use_gpu and hasattr(faiss, "StandardGpuResources"):
            try:
                res = faiss.StandardGpuResources()
                self.index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
            except Exception:
                self.index = cpu_index
        else:
            self.index = cpu_index

        self.is_built = True
        logger.info(f"Loaded FAISS index with {self.index.ntotal} vectors from {filepath}")
