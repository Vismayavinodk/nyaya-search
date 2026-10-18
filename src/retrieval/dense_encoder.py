"""
Dense semantic retrieval engine for Indian legal judgments.
Features:
- Multi-model support: BGE, InLegalBERT, and Multilingual-E5
- Document chunking with mean-pooling and max-pooling across judgment chunks
- Integrated FAISS index (IndexFlatIP / HNSW) with CPU/GPU allocation
- Checkpointing for Kaggle / Colab GPU execution so sessions never lose progress
- Model-specific prefix handling (E5 'query:/passage:', BGE instruction)
- Standard TREC run file export
"""

import os
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import re
import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Union
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

from src.data.schema import LegalCase, LegalQuery
from src.retrieval.faiss_index import LegalFAISSIndex, FAISS_AVAILABLE
from src.utils.logger import get_logger

logger = get_logger("DenseLegalRetriever")


class DenseLegalRetriever:
    """
    Dense semantic retriever supporting chunk-level encoding, hierarchical pooling,
    checkpoint-resumable encoding for Kaggle GPU sessions, and FAISS indexing.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        chunk_size: int = 400,
        chunk_overlap: int = 50,
        max_chunks_per_case: int = 6,
        pooling_strategy: str = "mean",  # 'mean', 'max', 'first_k'
        batch_size: int = 16,
        device: str = "auto",
        normalize_embeddings: bool = True,
        use_faiss: bool = True,
        faiss_index_type: str = "flat_ip"
    ):
        self.model_name = model_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.max_chunks_per_case = max_chunks_per_case
        self.pooling_strategy = pooling_strategy.lower()
        self.batch_size = batch_size
        self.normalize_embeddings = normalize_embeddings
        self.use_faiss = use_faiss and FAISS_AVAILABLE
        self.faiss_index_type = faiss_index_type

        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        logger.info(
            f"Initializing Dense Retriever: model='{model_name}', "
            f"pooling='{self.pooling_strategy}', device='{self.device}', FAISS={self.use_faiss}"
        )
        self.model = SentenceTransformer(model_name, device=self.device)
        self.embedding_dimension = self.model.get_sentence_embedding_dimension()

        self.doc_ids: List[str] = []
        self.corpus_embeddings: Optional[np.ndarray] = None  # Shape: (N, D)
        self.faiss_index: Optional[LegalFAISSIndex] = None

    @property
    def model_safe_name(self) -> str:
        """Sanitizes model name for directory and filename paths."""
        return self.model_name.replace("/", "_").replace("\\", "_")

    def _format_query_text(self, text: str) -> str:
        """Applies model-specific query prompts if required by the architecture."""
        text = text.strip()
        if "bge" in self.model_name.lower():
            return f"Represent this sentence for searching relevant passages: {text}"
        elif "e5" in self.model_name.lower():
            return f"query: {text}"
        return text

    def _format_passage_text(self, text: str) -> str:
        """Applies model-specific passage prefixes."""
        text = text.strip()
        if "e5" in self.model_name.lower():
            return f"passage: {text}"
        return text

    def _chunk_text(self, text: str) -> List[str]:
        """Splits long judgment text into overlapping word windows, sampling head and tail for long texts."""
        words = text.split()
        if len(words) <= self.chunk_size:
            return [text]

        chunks = []
        step = self.chunk_size - self.chunk_overlap
        for i in range(0, len(words), step):
            chunk = " ".join(words[i:i + self.chunk_size])
            chunks.append(chunk)
            if i + self.chunk_size >= len(words):
                break

        # If long, retain critical head (facts/procedural history) and tail (ratio decidendi/holding)
        if self.max_chunks_per_case and len(chunks) > self.max_chunks_per_case:
            half = self.max_chunks_per_case // 2
            chunks = chunks[:half] + chunks[-half:]

        return chunks

    def encode_case(self, case: LegalCase) -> np.ndarray:
        """
        Encodes a legal judgment by chunking and pooling chunk embeddings.

        Returns:
            Normalized 1D numpy array of shape (embedding_dim,).
        """
        raw_chunks = self._chunk_text(case.text)
        formatted_chunks = [self._format_passage_text(c) for c in raw_chunks]

        chunk_embeddings = self.model.encode(
            formatted_chunks,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize_embeddings
        )

        # Hierarchical pooling across judgment chunks
        if self.pooling_strategy == "max":
            # Element-wise max pooling: captures the maximum semantic activation across all chunks
            pooled = np.max(chunk_embeddings, axis=0)
        elif self.pooling_strategy == "first_k":
            pooled = chunk_embeddings[0]
        else:
            # Mean pooling: standard average across chunks
            pooled = np.mean(chunk_embeddings, axis=0)

        if self.normalize_embeddings:
            norm = np.linalg.norm(pooled)
            if norm > 1e-12:
                pooled = pooled / norm

        return pooled

    def encode_query(self, query: LegalQuery) -> np.ndarray:
        """
        Encodes a legal query statement into a semantic vector.

        Returns:
            Normalized 1D numpy array of shape (embedding_dim,).
        """
        raw_text = query.full_query_text
        formatted_text = self._format_query_text(raw_text)

        vec = self.model.encode(
            formatted_text,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize_embeddings
        )
        return vec

    def build_index(
        self,
        corpus: Dict[str, LegalCase],
        cache_path: Optional[str] = None,
        checkpoint_dir: Optional[str] = "data/cache/checkpoints",
        checkpoint_interval: int = 100
    ) -> None:
        """
        Encodes all cases in corpus with automatic checkpointing and FAISS indexing.
        Supports resume capability for long Kaggle/Colab GPU sessions.
        """
        self.doc_ids = list(corpus.keys())

        # 1. Check for complete final cache
        if cache_path and os.path.exists(cache_path):
            logger.info(f"Loading cached corpus embeddings from {cache_path}...")
            with np.load(cache_path) as data:
                cached_ids = list(data["doc_ids"])
                cached_embs = np.array(data["embeddings"]).astype(np.float32)
            if cached_ids == self.doc_ids:
                self.corpus_embeddings = cached_embs
                logger.info(f"Successfully loaded {self.corpus_embeddings.shape[0]} embeddings from cache.")
                self._build_or_load_faiss(cache_path)
                return
            logger.warning("Cached doc_ids do not match corpus. Recomputing embeddings...")

        # 2. Check for intermediate checkpoint
        ckpt_path = None
        encoded_dict: Dict[str, np.ndarray] = {}
        if checkpoint_dir:
            os.makedirs(checkpoint_dir, exist_ok=True)
            if cache_path:
                base_stem = os.path.splitext(os.path.basename(cache_path))[0]
                ckpt_path = os.path.join(checkpoint_dir, f"{base_stem}_ckpt.npz")
            else:
                ckpt_path = os.path.join(
                    checkpoint_dir,
                    f"{self.model_safe_name}_{self.pooling_strategy}_ckpt.npz"
                )
            if os.path.exists(ckpt_path):
                logger.info(f"Found checkpoint at {ckpt_path}. Resuming encoding session...")
                with np.load(ckpt_path) as ckpt_data:
                    ckpt_ids = list(ckpt_data["doc_ids"])
                    ckpt_embs = np.array(ckpt_data["embeddings"])
                for did, emb in zip(ckpt_ids, ckpt_embs):
                    if did in corpus:
                        encoded_dict[did] = emb
                logger.info(f"Resumed {len(encoded_dict)} / {len(self.doc_ids)} already encoded cases from checkpoint.")

        # 3. Encode remaining documents
        remaining_ids = [did for did in self.doc_ids if did not in encoded_dict]
        logger.info(
            f"Encoding {len(remaining_ids)} remaining legal cases with {self.model_name} "
            f"(pooling={self.pooling_strategy}, checkpoint_interval={checkpoint_interval})..."
        )

        pbar = tqdm(remaining_ids, desc=f"Encoding ({self.pooling_strategy})") if len(remaining_ids) > 10 else remaining_ids
        for count, did in enumerate(pbar, start=1):
            case = corpus[did]
            emb = self.encode_case(case)
            encoded_dict[did] = emb

            # Periodically write checkpoint to guard against Kaggle session disconnects
            if ckpt_path and (count % checkpoint_interval == 0 or count == len(remaining_ids)):
                current_ids = list(encoded_dict.keys())
                current_embs = np.vstack([encoded_dict[k] for k in current_ids])
                np.savez_compressed(ckpt_path, doc_ids=np.array(current_ids), embeddings=current_embs)

        # Assemble full corpus embedding matrix in canonical doc_ids order
        self.corpus_embeddings = np.vstack([encoded_dict[did] for did in self.doc_ids]).astype(np.float32)
        logger.info(f"Corpus embedding matrix completed: {self.corpus_embeddings.shape}")

        # 4. Save final compressed cache
        if cache_path:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            np.savez_compressed(
                cache_path,
                doc_ids=np.array(self.doc_ids),
                embeddings=self.corpus_embeddings
            )
            logger.info(f"Saved final corpus embeddings to {cache_path}")
            # Clean up intermediate checkpoint upon successful final serialization
            if ckpt_path and os.path.exists(ckpt_path):
                try:
                    os.remove(ckpt_path)
                except Exception:
                    pass

        # 5. Build and persist FAISS index
        self._build_or_load_faiss(cache_path)

    def _build_or_load_faiss(self, cache_path: Optional[str] = None) -> None:
        """Builds or loads the FAISS index."""
        if not self.use_faiss or self.corpus_embeddings is None:
            return

        faiss_path = os.path.splitext(cache_path)[0] + ".faiss" if cache_path else None
        self.faiss_index = LegalFAISSIndex(
            dimension=self.embedding_dimension,
            index_type=self.faiss_index_type,
            use_gpu=(self.device == "cuda")
        )

        if faiss_path and os.path.exists(faiss_path) and os.path.exists(faiss_path + ".meta"):
            try:
                self.faiss_index.load(faiss_path)
                logger.info(f"Loaded existing FAISS index from {faiss_path}")
                return
            except Exception as e:
                logger.warning(f"Could not load FAISS index ({e}), rebuilding...")

        self.faiss_index.build(self.corpus_embeddings, self.doc_ids)
        if faiss_path:
            try:
                self.faiss_index.save(faiss_path)
            except Exception as e:
                logger.warning(f"Could not persist FAISS index ({e})")

    def batch_search(
        self,
        queries: Dict[str, LegalQuery],
        top_k: int = 100
    ) -> Dict[str, List[Tuple[str, float]]]:
        """
        Batch dense retrieval for queries using FAISS or vectorized matrix multiplication.
        """
        if not queries:
            return {}
        if self.corpus_embeddings is None:
            raise RuntimeError("Corpus embeddings not computed. Call build_index() first.")

        qids = list(queries.keys())
        query_texts = [self._format_query_text(queries[qid].full_query_text) for qid in qids]

        # Batch encode query representations
        q_embs = self.model.encode(
            query_texts,
            batch_size=self.batch_size,
            show_progress_bar=len(queries) > 20,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize_embeddings
        ).astype(np.float32)

        # 1. Fast FAISS path if available
        if self.use_faiss and self.faiss_index is not None and self.faiss_index.is_built:
            return self.faiss_index.search_queries(q_embs, qids, top_k=top_k)

        # 2. Vectorized BLAS matrix multiplication path
        sim_matrix = np.dot(q_embs, self.corpus_embeddings.T)

        results: Dict[str, List[Tuple[str, float]]] = {}
        for i, qid in enumerate(qids):
            scores = sim_matrix[i]
            ranked_indices = np.argsort(scores)[::-1][:top_k]
            results[qid] = [(self.doc_ids[idx], float(scores[idx])) for idx in ranked_indices]

        return results

    def retrieve_for_query(self, query: LegalQuery, top_k: int = 100) -> List[Tuple[str, float]]:
        """Encodes single query and retrieves top candidates."""
        res = self.batch_search({query.query_id: query}, top_k=top_k)
        return res.get(query.query_id, [])

    @staticmethod
    def save_run_file(
        results: Dict[str, List[Tuple[str, float]]],
        filepath: str,
        run_name: str = "DENSE"
    ) -> None:
        """
        Exports retrieval rankings to standard TREC 6-column run format:
        query_id Q0 doc_id rank score run_name
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            for qid in sorted(results.keys()):
                for rank, (doc_id, score) in enumerate(results[qid], start=1):
                    f.write(f"{qid}\tQ0\t{doc_id}\t{rank}\t{score:.6f}\t{run_name}\n")
        logger.info(f"Saved TREC run file with {sum(len(v) for v in results.values())} rows to {filepath}")
