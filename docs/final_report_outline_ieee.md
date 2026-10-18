# IEEE-Style Final Project Report Outline

**Title**: Citation-Aware Multi-Stage Legal Precedent Retrieval and Grounded RAG with Calibrated Refusal for Indian Court Judgments  
**Format**: IEEE Transactions on Knowledge and Data Engineering (TKDE) / IEEE Access Style  
**Author**: Vismaya  
**Course**: Information Retrieval (CS9xxx)  

---

## Abstract
Indian legal case retrieval presents unique challenges characterized by extreme judgment length, vocabulary divergence, and the constitutional doctrine of binding precedent (*stare decisis* under Article 141). We present **Nyaya-IR**, an end-to-end multi-stage legal information retrieval and Retrieval-Augmented Generation (RAG) system. First, canonical Indian statutory anchors (`IPC`, `CrPC`, `Constitution`) and precedent citations are extracted and preserved as atomic tokens. Second, an optimized lexical BM25 retriever is coupled with chunk-pooled dense transformer encoders (InLegalBERT, BGE-Small) via dev-tuned convex score combination ($\alpha = 0.35$). Third, a directed citation multigraph ($18,929$ nodes, $16,455$ edges) computes topological authority metrics (In-Degree, PageRank, Personalized PageRank), enforced under a strict zero test-leakage audit. Interpolating citation authority with first-stage retrieval yields a statistically significant improvement on nDCG@10 ($+1.33\%$, $p = 0.0284$). Fourth, we evaluate candidate reranking comparing supervised listwise Learning-to-Rank (LightGBM LambdaMART) against deep sentence-pair cross-attention (Cross-Encoder), highlighting the impact of query-group sparsity in legal datasets. Finally, an isolated Legal RAG layer generates concise answers citing retrieved precedents in `[Doc: <doc_id>]` format, backed by a dual-component evidence sufficiency guardrail that achieves $100\%$ refusal recall on adversarial queries. All components are demonstrated via an interactive Streamlit application with transparent feature attribution.

**Index Terms**—Legal Information Retrieval, Citation Graph, PageRank, BM25+, Learning-to-Rank, Cross-Encoder, Retrieval-Augmented Generation (RAG), Hallucination Guardrails, Article 141 Indian Constitution.

---

## Section I: Introduction
1. **Background & Domain Motivation**:
   - Institutional reality of the Indian judicial hierarchy (Supreme Court of India, 25 High Courts, District Judiciary).
   - Article 141 of the Constitution of India: Law declared by the Supreme Court is binding on all courts.
   - Dual nature of legal relevance: Topical similarity (facts/issues) + Precedent authority (binding legal force).
2. **Distinct IR Challenges in Indian Case Law**:
   - *Judicial Verbosity*: Judgments frequently exceed 50–100 pages.
   - *Archaic & Divergent Phraseology*: Vocabulary mismatch across historical eras.
   - *Statutory Anchors*: Extreme sensitivity to specific sections (e.g., Section 302 vs 304 IPC).
3. **Research Questions (RQs)**:
   - **RQ1**: How do tuned sparse algorithms compare against modern dense bi-encoders when statutory citations are preserved as atomic identifiers?
   - **RQ2**: Can graph-theoretic authority signals from the precedent network improve retrieval without introducing test-query leakage?
   - **RQ3**: How does supervised listwise Learning-to-Rank (LambdaMART) compare against zero-shot deep Cross-Encoders under legal query-group sparsity?
   - **RQ4**: What calibrated guardrail mechanism effectively eliminates hallucination in legal RAG when evidence is weak or ungrounded?
4. **Summary of Contributions**:
   - Development of a reproducible, leakage-safe multi-stage pipeline.
   - Empirical evaluation across IL-PCR and a curated 35-query Indian legal QA benchmark.
   - Interactive Streamlit demonstration with component-level explainability.

---

## Section II: Related Work
1. **Legal Information Retrieval & Case Law Search**:
   - FIRE AILA (Artificial Intelligence for Legal Assistance) track.
   - IL-PCR (Indian Legal Precedent Case Retrieval) benchmark.
   - Lexical baselines vs domain-specific adaptations.
2. **Dense Semantic Retrieval & Long-Document Transformers**:
   - Bi-encoder limitations on legal documents.
   - Chunking and hierarchical pooling strategies (mean vs max pooling).
   - Domain-adapted language models: LegalBERT, InLegalBERT.
3. **Citation Network Analysis in Jurisprudence**:
   - Legal citation networks as directed multigraphs.
   - PageRank, HITS, Bibliographic Coupling, and Personalized PageRank in judicial prestige.
   - Risks and audit methodologies for data leakage in citation networks.
4. **Learning-to-Rank (LTR) & Neural Rerankers**:
   - Pointwise vs Pairwise vs Listwise LTR (LambdaMART, XGBoost).
   - Cross-Encoder architectures for passage reranking.
5. **Retrieval-Augmented Generation (RAG) & Guardrails in High-Stakes Domains**:
   - Hallucination risks in legal AI.
   - Faithfulness and citation attribution metrics.
   - Selective abstention and confidence-calibrated refusal mechanisms.

---

## Section III: System Architecture & Methodology
1. **Data Ingestion & Statutory Anchor Preservation**:
   - Regex-based legal citation extraction: SCC, AIR, SCR, SCALE, IPC, CrPC, CPC, Constitution.
   - Court boilerplate removal, head/tail truncation, and legal stopword elimination.
2. **First-Stage Retrieval Dual-Stream**:
   - *Stream A (Sparse)*: BM25Okapi and BM25+ with exact statutory token preservation.
   - *Stream B (Dense)*: Sliding-window chunk-based bi-encoder encoding (BGE-Small, InLegalBERT, Multilingual-E5) with FAISS IndexFlatIP.
   - *Fusion*: Dev-split tuned Reciprocal Rank Fusion ($k=80$) and convex combination ($\alpha=0.35$).
3. **Citation Graph Engine & Zero-Leakage Guarantee**:
   - Construction of directed DiGraph $(V, E)$ where $(u, v) \in E$ denotes judgment $u$ cites judgment $v$.
   - Mathematical proof of zero test-leakage: Formal suppression of outgoing citation edges $\forall u \in Q_{\text{test}}$.
   - Feature computation: In-degree centrality, global PageRank, Personalized PageRank (PPR), HITS authorities/hubs.
   - Linear score interpolation: $S_{\text{final}}(d) = (1 - \beta) S_{\text{hybrid}}(d) + \beta S_{\text{graph}}(d)$.
4. **Second-Stage Candidate Reranking**:
   - Top-100 candidate pool formulation.
   - *Option A*: LightGBM LambdaMART trained on train split, tuned with early stopping on dev split.
   - *Option B*: Cross-Encoder (`ms-marco-MiniLM-L-6-v2`) computing query-passage cross-attention over top-30 candidates.
5. **Isolated Legal RAG Layer with Calibrated Refusal**:
   - Architectural decoupling of `src/rag/` from core retrieval.
   - Open-source instruction model (`google/flan-t5-small`) with deterministic offline fallback.
   - Evidence sufficiency scoring function:
     $$\text{Sufficiency}(Q, \mathcal{P}) = 0.6 \cdot \max_{p \in \mathcal{P}} \text{Score}(p) + 0.4 \cdot \text{LexicalOverlap}(Q, \mathcal{P})$$
   - Refusal condition: $\text{Sufficiency} < \tau$ ($\tau = 0.30$) or $\text{Overlap} < 0.25$.
   - Regex citation extraction ensuring `[Doc: <doc_id>]` attribution.

---

## Section IV: Experimental Setup & Evaluation Protocol
1. **Datasets**:
   - *IL-PCR*: 3,183 Indian court judgments, 627 dev queries, 627 test queries.
   - *Curated Landmark Benchmark*: 10 landmark Supreme Court judgments with complete citation topologies.
   - *Legal QA Benchmark*: 35 curated questions (25 answerable landmark questions + 10 adversarial/weak-evidence refusal tests).
2. **Evaluation Metrics**:
   - Ranking: nDCG@5, nDCG@10, MAP, Recall@{5, 10, 20}, MRR.
   - RAG Grounding: Mean Faithfulness (unigram context containment).
   - Citation Quality: Citation Precision, Citation Recall, Citation F1.
   - Safety: Refusal Accuracy, Refusal Precision, Refusal Recall, Refusal F1.
3. **Statistical Hypothesis Testing Protocol**:
   - Paired Student's $t$-test ($H_0: \mu_{\Delta} = 0$, $\alpha = 0.05$).
   - 1,000-sample non-parametric bootstrap testing with $95\%$ confidence intervals.

---

## Section V: Empirical Results & Ablation Analysis
1. **Sparse vs Dense Retrieval Performance (RQ1)**:
   - Analysis of Table 1: Tuned BM25 ($nDCG@10=0.2910$) substantially outperforming dense models ($0.01-0.03$).
   - Discussion of why general dense bi-encoders suffer semantic drift on statutory numbers.
2. **Impact of Citation Graph Signals (RQ2)**:
   - Feature-relevance correlation analysis: PPR ($r=0.1647, p < 10^{-4}$), Statute Jaccard ($r=0.0936$).
   - Statistical significance of Graph Interpolation over BM25 Baseline ($+1.33\%$, $t=2.1965, p=0.0284$).
3. **Reranking: LightGBM LambdaMART vs Cross-Encoder (RQ3)**:
   - Analysis of LambdaMART group sparsity and feature gain monopoly.
   - Cross-Encoder superior generalization zero-shot ($nDCG@10=0.0427$).
4. **Legal RAG Grounding & Refusal Reliability (RQ4)**:
   - Faithfulness score ($0.8841$) and Citation F1 ($0.8571$).
   - $100\%$ refusal recall on adversarial out-of-scope queries.

---

## Section VI: Diagnostic Error Analysis
1. Case 1: Lexical Term Mismatch & Archaic Phrasing.
2. Case 2: Hub Authority Distortion from Monolithic Precedents.
3. Case 3: Truncation of Operative Ratio in Ultra-Long Judgments.
4. Case 4: Group Sparsity & Feature Overfitting in Supervised LambdaMART.
5. Case 5: Boundary False Refusals in Multi-Sentence Legal Arguments.

---

## Section VII: Threats to Validity & Limitations
1. **Construct Validity**: Evaluation of legal relevance vs real-world judicial citations.
2. **External Validity**: Generalizability to High Court and sub-ordinate tribunal records.
3. **Computational Constraints**: Scalability of Cross-Encoder inference over large candidate pools.

---

## Section VIII: Conclusion & Future Work
- Summary of findings confirming the indispensable role of statutory preservation, citation authority, and calibrated refusal.
- Future roadmap: Legal graph neural networks (GNNs), multi-turn dialogic legal assistants, and court-specific fine-tuning.

---

## References
*(Comprehensive academic citations including Robertson et al. (BM25), Page et al. (PageRank), Burges (LambdaMART), Reimers & Gurevych (SentenceTransformers), Malik et al. (IL-PCR), Bhattacharya et al. (FIRE AILA), Lewis et al. (RAG).)*
