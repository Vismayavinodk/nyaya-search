# Faculty & Examiner Defense Q&A: Comprehensive Technical Guide

This document contains 12 anticipated questions from professors, academic examiners, and reviewers during project defense, complete with mathematically and empirically grounded answers.

---

### Q1: Why did you build your own citation graph instead of relying exclusively on standard text retrievers like BM25 or Dense Transformers?
**Answer:**  
In Indian jurisprudence, retrieval is governed by **Article 141 of the Constitution of India**, which mandates that the law declared by the Supreme Court is binding on all courts. Legal relevance is therefore fundamentally dual-natured: a case must be both **topically relevant** (matching the factual issues) and **jurisprudentially authoritative** (possessing high precedent weight). Text-only retrievers cannot differentiate between a binding 9-judge constitutional bench ruling and an obscure, overturned lower-court judgment that happens to share identical keywords. By modeling the citation network as a directed multigraph ($18,929$ nodes, $16,455$ edges), topological metrics like In-Degree Centrality, PageRank, and Personalized PageRank capture precedent prestige. Incorporating graph authority via score interpolation ($\beta = 0.15$) yielded a statistically significant improvement on nDCG@10 ($+1.33\%$, $t=2.1965, p=0.0284$).

---

### Q2: How did you mathematically and programmatically guarantee zero test-query data leakage in the citation graph?
**Answer:**  
Citation graphs are prone to catastrophic leakage if the edges between test query cases and corpus precedents are included during graph construction, as the algorithm would "cheat" by following explicit citation links to relevant cases.  
To guarantee zero leakage:
1. In `src/graph/citation_graph.py`, our `build_graph()` method accepts an `exclude_outgoing_for` set containing all test query IDs.
2. All outgoing citation edges originating from any node in the test set are strictly suppressed:
   $$E_{\text{safe}} = E \setminus \{(u, v) \in E \mid u \in Q_{\text{test}}\}$$
3. Test queries exist in the graph strictly as isolated sink nodes with an out-degree of 0.
4. We enforce this through an automated pytest audit (`tests/test_phase4.py::test_zero_test_query_leakage_in_graph`), which asserts that zero outgoing edges exist for all held-out test instances.

---

### Q3: Why did BM25 Tuned achieve nDCG@10 = 0.2910 while dense models scored substantially lower (0.01 to 0.03)?
**Answer:**  
Legal precedent retrieval is an extreme-precision domain dictated by canonical statutory citations. In criminal law, a case involving **Section 302 IPC** (punishment for murder) is legally incompatible with **Section 304A IPC** (causing death by negligence), yet standard pre-trained dense transformers (like BGE or MiniLM) map both into near-identical vector subspaces because their general English training treats numerical digits as soft semantic concepts.  
In our pipeline, the `IndianLegalCitationExtractor` normalizes statutory references into discrete atomic tokens (e.g., `STATUTE:IPC:SEC_302`). Because BM25 operates on exact inverted index term frequencies, it penalizes false statutory substitutions through IDF weighting, achieving superior first-stage precision. Dense models only become advantageous when hybridized with BM25 to resolve lexical paraphrase mismatch.

---

### Q4: Why did LightGBM LambdaMART show low test nDCG (0.0059) compared to the Cross-Encoder (0.0427) in Phase 5?
**Answer:**  
LambdaMART is a supervised tree-based listwise Learning-to-Rank algorithm that constructs pairwise gradients ($\lambda_{ij}$) within discrete query groups. Its performance is heavily dependent on having hundreds or thousands of query groups with diverse document distributions.  
In legal datasets like `il_pcr`, we had 50 dev and 50 test query groups. This caused acute **query group sparsity**:
1. Tree splits overfit to high-variance single features (`dense_score` received a feature gain of $77.49$, while topological graph features received $0.00$ gain).
2. In contrast, the Cross-Encoder (`ms-marco-MiniLM-L-6-v2`) evaluates fine-grained sentence-pair cross-attention at every token layer. Having been pre-trained on millions of web passage pairs, it generalized robustly zero-shot without suffering from group-level tabular overfitting.

---

### Q5: How did you select the optimal hyperparameters (k1, b, alpha, beta), and how did you prevent tuning leakage?
**Answer:**  
We strictly adhered to the three-way split protocol (**Train**, **Dev / Validation**, and **Held-Out Test**):
1. **BM25 Parameters**: Grid-searched over $k_1 \in [1.0, 2.5]$ and $b \in [0.5, 0.9]$ exclusively on the dev split. Optimal was $k_1 = 2.2, b = 0.85$.
2. **Hybrid Convex Weight $\alpha$**: Evaluated $\alpha \in [0.0, 1.0]$ in increments of $0.05$ on dev; optimal was $\alpha = 0.35$ (65% BM25, 35% Dense).
3. **Graph Interpolation $\beta$**: Grid-searched $\beta \in [0.0, 0.5]$ on dev; optimal was $\beta = 0.15$ with normalized In-Degree.
4. **Held-Out Test Rule**: The test split was evaluated exactly **once** using these locked dev hyperparameters. Per-query test predictions were exported to TREC `.run` files for independent verification.

---

### Q6: What statistical significance tests did you execute, and what do the p-values indicate?
**Answer:**  
To confirm that observed metric improvements were not random artifacts, we conducted:
1. **Two-sided paired Student's $t$-tests** at significance level $\alpha = 0.05$.
2. **Non-parametric paired bootstrap hypothesis tests** with $B = 1,000$ iterations and $95\%$ empirical confidence intervals.

**Key Findings:**
- Graph Interpolation vs BM25 Baseline: $t = 2.1965, p = 0.0284$. Because $p < 0.05$, we reject the null hypothesis and confirm that citation authority delivers a genuine, statistically significant boost.
- BM25 Tuned vs BM25 Default: $t = 7.7084, p < 10^{-4}$ (statistically significant).
- Hybrid Weighted vs BM25: $t = 0.0000, p = 1.0000$ (no significant difference on the small test split, indicating BM25 was already providing the core signal).

---

### Q7: How does the Legal RAG Refusal Guardrail operate, and how was it benchmarked?
**Answer:**  
In high-stakes legal applications, hallucinating a non-existent statute or judgment can lead to severe legal malpractice. Our RAG engine (`LegalRAGEngine`) employs a dual-component evidence sufficiency function:
$$\text{Sufficiency}(Q, \mathcal{P}) = 0.6 \cdot \max_{p \in \mathcal{P}} \text{Score}(p) + 0.4 \cdot \text{LexicalOverlap}(Q, \mathcal{P})$$
The system abstains if $\text{Sufficiency} < 0.30$ or keyword overlap $< 0.25$.  
We evaluated this on a curated 35-question benchmark containing:
- 25 valid constitutional questions.
- 10 adversarial/weak-evidence refusal tests (e.g., Martian hovercraft regulations, Ethereum DAO governance in 1872).  
The guardrail achieved **100% refusal recall (10/10)** while maintaining an overall refusal accuracy of **88.57%** and a mean faithfulness score of **0.8841** on answered queries.

---

### Q8: How did you handle long judicial documents that exceed transformer token limits (512 tokens)?
**Answer:**  
Indian Supreme Court opinions average 3,000 to 15,000 words. Naive 512-token truncation cuts off the operative ratio decidendi, which customarily appears in the concluding paragraphs.  
We implemented two complementary strategies:
1. **Head-Tail Truncation**: Retains the first $128$ tokens (case context and statutory provisions) and the final $384$ tokens (operative ruling and ratio).
2. **Sliding-Window Chunking with Hierarchical Pooling**: Splits judgments into overlapping chunks of $400$ tokens with a $50$-token stride (up to 6 chunks per case). We compared **mean-pooling** vs **max-pooling** across chunk embeddings. Max-pooling achieved higher alignment with specific operative ratio paragraphs, boosting Recall@20 from $0.0267$ to $0.0833$.

---

### Q9: Why did Personalized PageRank (PPR) exhibit a much higher correlation with relevance than global PageRank?
**Answer:**  
In our feature correlation analysis ($N=31,213$ pairs):
- **Personalized PageRank (PPR)**: Pearson $r = 0.1647$ ($p < 10^{-4}$).
- **Global PageRank**: Pearson $r = 0.0388$ ($p < 10^{-4}$).  
Global PageRank measures unconditional historical prestige. Monumental cases like *Kesavananda Bharati* have massive global PageRank regardless of whether the query relates to basic structure or maritime tax. In contrast, Personalized PageRank distributes random walk restart probabilities exclusively from query-candidate seed nodes. Therefore, PPR reflects localized precedent authority directly relevant to the specific legal dispute, making it over $4\times$ more correlated with relevance.

---

### Q10: What are the primary failure cases of your system, and how can they be mitigated in future work?
**Answer:**  
We identified five structural failure modes (documented in `experiments/results/error_analysis.md`):
1. *Lexical term mismatch* across different judicial decades (mitigated via dense semantic fusion).
2. *Hub authority distortion* where global PageRank over-boosts giant precedents (mitigated by tuning $\beta=0.15$ and using PPR).
3. *Truncation of operative ratio* in 80+ page judgments (mitigated by sliding-window chunk max-pooling).
4. *Group sparsity in LambdaMART* due to limited training queries (mitigated by Cross-Encoder reranking).
5. *Boundary false refusals in RAG* where legal arguments span multiple disjoint sentences (mitigated by multi-passage context aggregation).

---

### Q11: How is the RAG layer kept isolated from the core retrieval and ranking pipeline?
**Answer:**  
Architectural separation of concerns is maintained:
- Core retrieval (`src/retrieval/`), graph analysis (`src/graph/`), and reranking (`src/rerank/`) operate strictly as Information Retrieval pipelines producing ranked document lists and TREC `.run` files.
- The RAG module (`src/rag/`) has zero internal dependencies on the reranking or indexing internals; it operates purely as an downstream client accepting a question and a list of candidate passages.
- All RAG schemas (`LegalQAExample`, `RAGPassage`, `RAGResponse`) reside in `src/rag/schema.py`, ensuring that testing, fine-tuning, or replacing the generative LLM has zero impact on retrieval reproducibility.

---

### Q12: How can someone reproduce all your results from scratch?
**Answer:**  
Reproduction is guaranteed through deterministic seeds (`SEED=42`), pinned dependencies in `requirements.txt`, and automated CLI entrypoints:
1. Data & Preprocessing: `python run_pipeline.py --mode data`
2. Sparse Retrieval & Dev Tuning: `python run_pipeline.py --mode sparse`
3. Dense Retrieval & FAISS: `python run_pipeline.py --mode dense`
4. Leakage-Safe Citation Graph: `python run_phase4_graph.py`
5. LambdaMART & Cross-Encoder: `python run_phase5_rerank.py`
6. Legal RAG Benchmark: `python run_phase6_rag.py`
7. Interactive Streamlit Demo: `streamlit run app.py`
All 46 unit tests can be validated in under 30 seconds via `pytest tests/ -v`.
