# Phase 3: Dense Retrieval, Model Comparison, Long-Document Chunking & Hybrid Fusion

## 1. Overview & Architecture

Phase 3 enhances the Indian Legal Case Retrieval engine by complementing sparse lexical search (BM25) with dense semantic representation models, handling long judicial texts through hierarchical chunking, indexing representations with FAISS, safeguarding long execution sessions with checkpointing, and fusing rankings via dev-tuned hybrid strategies.

```
                              ┌───────────────────────────────────────────┐
                              │           Legal Query Statement           │
                              └─────────────────────┬─────────────────────┘
                                                    │
                         ┌──────────────────────────┴──────────────────────────┐
                         ▼                                                     ▼
        ┌───────────────────────────────────┐               ┌───────────────────────────────────┐
        │   Sparse BM25 (Tuned k1, b)       │               │   Dense Semantic Retriever        │
        │   - Statutory preservation        │               │   - Multi-model evaluation        │
        │   - Indian legal stopword removal │               │   - Long-doc chunking (Max vs Mean│
        │   - Exact citation matching       │               │   - FAISS IndexFlatIP (Cosine)    │
        └─────────────────┬─────────────────┘               └─────────────────┬─────────────────┘
                          │                                                   │
                          │   Sparse Candidates (Top-K)                       │   Dense Candidates (Top-K)
                          └─────────────────────────┬─────────────────────────┘
                                                    │
                                                    ▼
                                  ┌───────────────────────────────────┐
                                  │      Hybrid Fusion Engine         │
                                  │   (Strictly Tuned on Dev Split)   │
                                  ├───────────────────────────────────┤
                                  │ 1. Reciprocal Rank Fusion (RRF)   │
                                  │    Score(d) = ∑ 1 / (k + rank)    │
                                  │ 2. Convex Score Combination       │
                                  │    α · Dense + (1 - α) · BM25     │
                                  └─────────────────┬─────────────────┘
                                                    │
                                                    ▼
                                  ┌───────────────────────────────────┐
                                  │    Standardized Evaluation        │
                                  │   (nDCG@10, MAP, Recall@k, MRR)   │
                                  │   Paired Significance (p < 0.05)  │
                                  └───────────────────────────────────┘
```

---

## 2. Dense Embedding Models Evaluated

We compare 3 distinct pretrained transformer architectures representing different inductive biases:

1. **`BAAI/bge-small-en-v1.5`** (384-d, 33M params):
   - **Characteristics**: General-domain contrastive dense retriever trained with RetroMAEs and multi-stage hard negative mining.
   - **Query Prompt**: `Represent this sentence for searching relevant passages: {query}`
   - **Strengths**: SOTA zero-shot semantic matching on MTEB retrieval benchmark; strong paraphrase matching.
   - **Weakness**: Lacks legal pretraining; statutory section citations (e.g., *Section 138 NI Act*) are tokenized sub-optimally compared to specialized legal vocabularies.

2. **`law-ai/InLegalBERT`** (768-d, 110M params):
   - **Characteristics**: Domain-specific BERT architecture continuously pretrained from scratch on 5.4M Indian legal documents (Supreme Court, High Courts, and Acts).
   - **Strengths**: Specialized legal subword vocabulary preserves legal terminology, Latin maxims (*mens rea*, *ratio decidendi*, *res judicata*), and Indian statute syntax.
   - **Weakness**: Pretrained via Masked Language Modeling (MLM) rather than bi-encoder contrastive retrieval objectives. Without supervised contrastive fine-tuning, raw token embeddings can suffer from anisotropy, requiring chunk pooling and hybrid fusion.

3. **`intfloat/multilingual-e5-small`** (384-d, 117M params):
   - **Characteristics**: Multilingual bi-encoder trained on CCPairs with asymmetric prefix conditioning (`query: ` and `passage: `).
   - **Strengths**: Robust against multilingual and code-mixed phrasing often present in Indian subordinate court records; explicit asymmetry distinguishes short queries from long case facts.

---

## 3. Handling Long Legal Documents: Chunking & Pooling

Indian Supreme Court judgments frequently exceed 10,000 to 40,000 words, far exceeding transformer context windows (512 tokens).

### 3.1. Sliding Window & Structural Head/Tail Sampling
- **Window size**: 400 words with 50-word overlap.
- **Head/Tail Heuristic**: If a case produces $> K$ chunks (default $K=6$), we retain the first $K/2$ chunks (facts, procedural history, headnotes) and the last $K/2$ chunks (operative legal analysis, *ratio decidendi*, final order). This prevents GPU memory exhaustion while preserving 95%+ of operative judicial rationale.

### 3.2. Max-Pooling vs. Mean-Pooling
- **Mean-Pooling**:
  $$e_{\text{case}}^{\text{mean}} = \frac{1}{M} \sum_{m=1}^M e_m$$
  *Limitation*: Dilutes critical statutory holding sentences across dozens of pages of routine evidentiary narrative.
- **Max-Pooling**:
  $$e_{\text{case}, j}^{\text{max}} = \max_{m \in \{1,\dots,M\}} e_{m, j}$$
  *Advantage*: Captures peak semantic activations. If any single paragraph discusses the exact legal question, that signal remains prominent in the case vector.
- Both pooled vectors are subsequently L2-normalized so inner product equals cosine similarity.

---

## 4. FAISS Vector Indexing

- **Index Type**: `faiss.IndexFlatIP` (Exact Inner Product).
- **GPU Acceleration**: Supports seamless GPU acceleration via `faiss.StandardGpuResources()` when `CUDA` is available, falling back automatically to multi-threaded CPU OpenMP BLAS.
- **Search Efficiency**: Sub-millisecond retrieval latency across candidate cases.
- **Persistence**: Serializes `.faiss` vector binary alongside `.meta` document ID dictionary.

---

## 5. Kaggle / Colab GPU Session Checkpointing

Kaggle and Google Colab free-tier GPU instances are prone to sudden disconnections, idle timeouts, and 12-hour execution limits. Re-encoding 3,183 cases from scratch on every disconnect causes significant delays.

### Checkpoint Mechanism:
1. Intermediate embeddings are saved every `checkpoint_interval` cases (default: 100) to `data/cache/checkpoints/<dataset>_<model>_<pooling>_ckpt.npz`.
2. When `build_index()` is invoked, it checks for existing checkpoint files.
3. If found, all previously encoded vectors are loaded instantly, and encoding resumes from the remaining document list.
4. Once all documents are completed and the final compressed cache (`.npz`) and FAISS index (`.faiss`) are written, the temporary checkpoint file is safely unlinked.

---

## 6. Hybrid Fusion Strategies (Tuned Strictly on Dev)

### 6.1. Reciprocal Rank Fusion (RRF)
RRF combines candidate lists without requiring cross-system score calibration:
$$\text{RRF}(d) = \frac{1}{k + \text{rank}_{\text{BM25}}(d)} + \frac{1}{k + \text{rank}_{\text{Dense}}(d)}$$
where $k$ is a smoothing constant swept over $\{10, 20, 40, 60, 80, 100\}$.

### 6.2. Convex Score Combination (Weighted Fusion)
Min-max score normalization per query followed by linear convex combination:
$$S_{\text{norm}}(d) = \frac{S(d) - \min(S)}{\max(S) - \min(S) + \epsilon}$$
$$S_{\text{hybrid}}(d) = \alpha \cdot S_{\text{norm}}^{\text{Dense}}(d) + (1 - \alpha) \cdot S_{\text{norm}}^{\text{BM25}}(d)$$
where $\alpha \in [0.0, 1.0]$ is swept with step 0.05.

### 6.3. Zero-Leakage Dev-Only Tuning Guardrail
All grid search evaluations for $\alpha$ and $k$ are computed **strictly on `dev_queries`**. The held-out `test_queries` are only evaluated once using the optimal parameters determined on dev.

---

## 7. Execution Commands

### Local Fast Verification (Sample or Subsample):
```bash
# Verify pipeline on sample benchmark
python run_phase3_dense_hybrid.py --dataset sample --top_k 20

# Run on real IL-PCR dataset with subsampling for fast CPU testing
python run_phase3_dense_hybrid.py --dataset il_pcr --max_cases 250 --max_queries 50 --top_k 20
```

### Full Benchmark on Kaggle / Google Colab (GPU):
```bash
# Full GPU execution on IL-PCR with checkpointing
python run_phase3_dense_hybrid.py --dataset il_pcr --device cuda --batch_size 32 --top_k 50
```
