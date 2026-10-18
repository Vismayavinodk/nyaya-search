# Presentation Slide Deck Outline (12 Slides)

**Project Title**: Nyaya-IR: Citation-Aware Multi-Stage Legal Precedent Retrieval & Grounded RAG with Calibrated Refusal  
**Presenter**: Vismaya  
**Target Duration**: 15–20 Minutes (including 5 min Q&A and Demo)  

---

### Slide 1: Title & Introduction
- **Slide Title**: Nyaya-IR: Citation-Aware Hybrid Legal Precedent Retrieval & Grounded RAG
- **Sub-header**: Multi-Stage Ranking, Network Authority, and Calibrated Refusal for Indian Court Judgments
- **Presenter**: Vismaya | Information Retrieval Course Project
- **Visual Elements**:
  - Scales of Justice icon / Court diagram.
  - Bullet highlights: BM25+ • Chunk-Pooled Transformers • NetworkX Citation Graph • LambdaMART & Cross-Encoder • Grounded RAG.
- **Speaking Points**:
  - "Welcome everyone. Today I am presenting Nyaya-IR, an information retrieval and grounded generative answering system engineered specifically for Indian judicial precedents."

---

### Slide 2: Motivation & Legal Domain Realities
- **Slide Title**: The Indian Legal Context: Why Generic IR Fails
- **Key Concepts**:
  - **Article 141 & *Stare Decisis***: Supreme Court law is binding on all courts in India. Relevance is dual: factual topicality + judicial authority.
  - **Judicial Verbosity**: Opinions span 50 to 100+ pages; operative *ratio decidendi* is often buried on the final pages.
  - **Archaic Vocabulary & Phrase Mismatch**: Precedents from different decades use divergent phraseology for the same legal concepts.
  - **Statutory Anchors**: Critical dependence on exact sections (e.g., Section 302 IPC vs Section 304 IPC).
- **Visual Elements**:
  - Side-by-side comparison: Short Web Query vs 80-page Judicial Case Document.
- **Speaking Points**:
  - "In legal search, retrieving a topically similar High Court case is insufficient if an authoritative, binding Supreme Court ruling directly overrides it."

---

### Slide 3: Research Questions & Objectives
- **Slide Title**: Core Research Questions
- **Content**:
  - **RQ1**: How do tuned sparse models compare against dense bi-encoders when statutory code citations are preserved?
  - **RQ2**: Can citation graph authority signals improve ranking without introducing test-query data leakage?
  - **RQ3**: How does supervised listwise Learning-to-Rank (LambdaMART) compare against zero-shot deep Cross-Encoders on legal datasets?
  - **RQ4**: How can we ensure a legal RAG system refuses to answer when precedent evidence is weak or ungrounded?
- **Speaking Points**:
  - "We set out to investigate whether domain-adapted retrieval, graph network structure, and calibrated guardrails can systematically outperform off-the-shelf retrieval pipelines."

---

### Slide 4: System Architecture Overview
- **Slide Title**: Nyaya-IR Multi-Stage Architecture
- **Visual Diagram**:
  ```
  Query / Case Summary
         │
         ├──► Stream A: BM25+ (Statutory Anchor Tokens) ──┐
         │                                                ├──► Hybrid Convex Score Fusion (α=0.35)
         └──► Stream B: Chunk-Pooled Dense (BGE-Small) ───┘                │
                                                                           ▼
                                                                Citation Graph DiGraph
                                                            (In-Degree, PageRank, PPR: β=0.15)
                                                                           │
                                                                           ▼
                                                             Top-100 Multi-Modal Pool
                                                                           │
                                                                           ▼
                                                               Cross-Encoder / LambdaMART
                                                                           │
                                                                           ▼
                                                           Isolated Legal RAG Engine
                                                      (Sufficiency ≥ 0.30? Answer : Refuse)
  ```
- **Speaking Points**:
  - "Notice our strict architectural decoupling: retrieval and ranking feed the candidate pool, while the RAG engine remains an independent consumer."

---

### Slide 5: Data Ingestion & Statutory Preservation
- **Slide Title**: Preserving the Legal Fabric: Ingestion & Citation Extraction
- **Key Features**:
  - High-precision regex extractor for Indian legal citations: SCC, AIR, SCR, SCALE.
  - Preservation of statutory codes (`CONST:ART_21`, `STATUTE:IPC:SEC_302`) as atomic tokens to prevent tokenizer splitting.
  - Boilerplate cleaning (removing court headers, bench rosters, advocate lists) and legal stopword filtering.
  - Head-tail truncation and sliding-window chunking.
- **Speaking Points**:
  - "Standard Lucene tokenizers split 'STATUTE:IPC:SEC_302' into four meaningless tokens. Our pipeline preserves them atomically."

---

### Slide 6: Citation Network & Zero-Leakage Guarantee
- **Slide Title**: Precedent Graph: Topological Authority without Leakage
- **Key Metrics**:
  - Directed DiGraph built with NetworkX: **18,929 nodes, 16,455 edges**.
  - Features: Normalized In-degree, global PageRank, Personalized PageRank (PPR), HITS.
- **The Zero-Leakage Audit**:
  - Strict suppression of outgoing citation edges from all test-split queries and documents.
  - Verified via automated unit tests (`test_zero_test_query_leakage_in_graph`).
- **Feature-Relevance Correlation**:
  - Personalized PageRank: $r = 0.1647$ ($p < 10^{-4}$).
  - Statute Jaccard Overlap: $r = 0.0936$ ($p < 10^{-4}$).
- **Speaking Points**:
  - "We mathematically proved zero test-query leakage by ensuring all test cases exist purely as isolated sink nodes without outgoing edges."

---

### Slide 7: Sparse vs Dense Retrieval Results
- **Slide Title**: First-Stage Benchmark: The Power of Tuned BM25
- **Results Table**:
  - Tuned BM25: **nDCG@10 = 0.2910**, MAP = 0.2361 (+10.23% over default BM25, $p < 10^{-4}$).
  - Dense BGE-Small (Max-Pooling): nDCG@10 = 0.0099.
  - Dense InLegalBERT: nDCG@10 = 0.0098.
  - Hybrid Weighted Score ($\alpha = 0.35$): nDCG@10 = 0.0631.
- **Key Takeaway**:
  - General dense embeddings suffer from semantic drift on numerical statute sections. Tuned BM25 with statutory tokens dominates first-stage lexical precision.
- **Speaking Points**:
  - "In law, Section 302 (Murder) and Section 304 (Culpable Homicide) are worlds apart, yet dense models treat them as near-synonyms. BM25 provides exact statutory filtering."

---

### Slide 8: Citation Authority Augmentation
- **Slide Title**: Enhancing Retrieval via Citation Graph Interpolation
- **Interpolation Formulation**:
  $$S_{\text{final}}(d) = (1 - \beta) S_{\text{base}}(d) + \beta S_{\text{graph}}(d)$$
- **Empirical Results**:
  - Optimal dev-split parameter: $\beta = 0.15$ with normalized In-Degree.
  - Test nDCG@10: **0.2949** (vs 0.2910 baseline, $+1.33\%$ relative gain).
  - Paired $t$-test confirmation: **$t = 2.1965, p = 0.0284$** (statistically significant at $\alpha = 0.05$).
  - Test Recall@10: Increases from $0.3892$ to **$0.3963$**.
- **Speaking Points**:
  - "By infusing a modest 15% citation authority weight, we achieved a statistically significant lift in nDCG@10, successfully prioritizing landmark precedents."

---

### Slide 9: Reranking: LambdaMART vs Cross-Encoder
- **Slide Title**: Stage Two Reranking: Option A vs Option B
- **Findings**:
  - **Option A (LightGBM LambdaMART)**: Test nDCG@10 = $0.0059$.
    - *Diagnosis*: Group sparsity (50 query groups) caused tree-based listwise rankers to overfit to dominant dense features (gain = $77.49$) while zeroing topological graph splits.
  - **Option B (Cross-Encoder `MiniLM-L6-v2`)**: Test nDCG@10 = **$0.0427$**.
    - *Diagnosis*: Deep token-level cross-attention generalizes substantially better zero-shot on long legal passages.
- **Speaking Points**:
  - "Our findings show that deep cross-encoders outperform tree-based tabular LTR when legal training query groups are limited."

---

### Slide 10: Legal RAG Layer & Calibrated Refusal Guardrail
- **Slide Title**: Legal RAG: Grounded Answering & Refusal Guardrail
- **Core Mechanism**:
  - Evidence Sufficiency: $0.6 \cdot \text{MaxScore} + 0.4 \cdot \text{LexicalOverlap}$.
  - Calibrated threshold $\tau = 0.30$.
- **Evaluation on 35-Question Benchmark**:
  - **Faithfulness Score**: **$0.8841$** (high context grounding).
  - **Citation F1 Score**: **$0.8571$** (accurate `[Doc: <doc_id>]` attribution).
  - **Refusal Recall**: **$1.0000$ (10/10 adversarial queries safely rejected)**.
- **Speaking Points**:
  - "A legal AI must know when not to answer. Our guardrail achieved a 100% refusal rate on out-of-scope and adversarial queries."

---

### Slide 11: Interactive Demo (Streamlit Tour)
- **Slide Title**: System Demonstration: Transparent Attribution & Q&A
- **Live Demo Highlights**:
  - 1. **Case Search & Attribution**: Paste legal facts, see component breakdown (BM25, Dense, In-Degree, PageRank).
  - 2. **'Why This Ranked High' Panel**: Explains the exact statutory overlap, semantic affinity, and citation authority.
  - 3. **Grounded Legal RAG**: Generates short answers citing precedent documents.
  - 4. **Guardrail In Action**: Demonstrates abstention on Martian hovercraft speed limits.
- **Speaking Points**:
  - "Let us transition to the live Streamlit demo to see the multi-stage attribution and guardrail in real time."

---

### Slide 12: Summary, Contributions & Q&A
- **Slide Title**: Conclusion & Key Contributions
- **Summary**:
  - Developed a complete 7-phase legal retrieval and RAG pipeline.
  - Proved that statutory preservation + citation graph authority significantly boosts retrieval ($p=0.0284$).
  - Evaluated Cross-Encoders vs LambdaMART under legal data sparsity.
  - Designed an adversarial-tested RAG refusal guardrail ($100\%$ refusal recall).
  - Built an interactive Streamlit application with transparent explainability.
- **Thank You!** • Questions & Faculty Discussion Invited.
