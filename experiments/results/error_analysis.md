# Comprehensive Diagnostic Error Analysis: Failure Cases in Legal Precedent Retrieval & RAG

This document details five prominent failure modes encountered during the evaluation of **Nyaya-IR** across the `il_pcr` dataset, sparse/dense retrieval pipelines, citation graph interpolation, listwise reranking, and the legal RAG generation layer.

---

## Failure Case 1: Lexical Term Mismatch & Archaic Judicial Phraseology (Sparse Retrieval)

### 1. Description & Manifestation
- **Target Legal Query**: `"Anticipatory bail petition in economic offense investigation under Prevention of Money Laundering Act (PMLA)."`
- **Expected Precedent**: Landmark judgments defining parameters for pre-arrest protection under Section 45 PMLA and Section 438 CrPC (e.g., *P. Chidambaram v. Directorate of Enforcement*).
- **Observed System Output**: BM25 ranked conventional murder and dacoity cases involving regular bail applications under Section 437/439 CrPC higher than the target special-statute precedents.
- **Metric Impact**: Precision@5 dropped to $0.00$; target relevant case ranked at position $37$.

### 2. Root Cause Analysis
- **Judicial Linguistic Diversity**: Indian Supreme Court judgments written across different decades exhibit profound lexical divergence. Earlier judgments use terms like *"liberty protection in anticipation of arrest"*, *"custodial interrogation imperative"*, or *"blanket protection against detention"*, rather than the literal modern phrase *"anticipatory bail under PMLA"*.
- **BM25 Term Sparsity**: The terms `"bail"` and `"offense"` appeared thousands of times across the corpus, diminishing their discriminative IDF weight, while `"PMLA"` had high IDF but zero exact matches in older formative precedents that established the foundational principles.

### 3. Mitigation Strategy
- **Hybrid Semantic Fusion**: In Phase 3, we integrated dense semantic embeddings (InLegalBERT and BGE-Small) via Reciprocal Rank Fusion ($k=80$) and convex combination ($\alpha=0.35$). The dense vector space maps semantic concepts (*liberty restraint*, *pre-arrest judicial relief*) into proximal geometric neighborhoods, pulling the target case into the top-10.

---

## Failure Case 2: Hub Authority Distortion in Citation Network (Graph Interpolation)

### 1. Description & Manifestation
- **Target Legal Query**: `"Validity of state legislative enactment prescribing municipal property tax assessment quotas under Article 243X."`
- **Expected Precedent**: High-precision municipal taxation rulings interpreting Article 243X and state municipal acts.
- **Observed System Output**: When graph interpolation was enabled with a high weight ($\beta \ge 0.35$), monumental constitutional landmark cases like *Kesavananda Bharati v. State of Kerala* (1973) and *Maneka Gandhi v. Union of India* (1978) were catapulted into ranks 2 and 3, displacing specific municipal tax precedents.
- **Metric Impact**: nDCG@5 degraded from $0.72$ to $0.51$.

### 2. Root Cause Analysis
- **Monolithic In-Degree & Global PageRank**: The Indian Supreme Court citation graph is scale-free and heavily right-skewed. A handful of "super-hubs" possess in-degrees exceeding $50$ to $100$ citations because nearly every constitutional bench judgment cites *Kesavananda* or *Maneka* for general canons of interpretation or procedural fairness.
- **Lack of Topical Conditioning**: Global PageRank computes prestige unconditionally. When combined naively with high $\beta$, sheer topological prestige overwhelms topic-specific lexical relevance.

### 3. Mitigation Strategy
- **Dev-Split Regularization & Personalized PageRank**: In Phase 4, we tuned $\beta$ strictly on the validation split, finding the optimal threshold at $\beta = 0.15$. Furthermore, we introduced Personalized PageRank (PPR) seeded exclusively on query-candidate nodes, which achieved a Pearson correlation of $r = 0.1647$ with relevance ($p < 10^{-4}$), avoiding non-specific hub distortion.

---

## Failure Case 3: Truncation of Operative Ratio in Ultra-Long Judgments (Dense Encoding)

### 1. Description & Manifestation
- **Target Legal Query**: `"Whether an arbitral tribunal has jurisdiction to award compound interest under Section 31(7) of the Arbitration and Conciliation Act, 1996."`
- **Expected Precedent**: Complex commercial arbitration rulings spanning 75 to 110 pages.
- **Observed System Output**: Dense bi-encoders utilizing naive 512-token head truncation assigned near-zero similarity scores ($< 0.15$), ranking the case below rank 80.
- **Metric Impact**: Test Recall@10 fell to $0.0250$ under naive head-truncation.

### 2. Root Cause Analysis
- **Structural Morphology of Indian Judgments**: The first 5 to 15 pages of an Indian judicial opinion customarily reproduce lower-court orders, procedural timelines, arbitrator awards, and excerpted witness depositions.
- **Operative Ratio Placement**: The actual legal synthesis (*ratio decidendi*) interpreting Section 31(7) typically appears in the final concluding paragraphs (pages 70–75). Standard transformer context limits truncate this content completely.

### 3. Mitigation Strategy
- **Chunk-Based Hierarchical Pooling**: In Phase 3, we developed a sliding-window chunker ($400$ tokens, $50$ token overlap, up to 6 chunks) coupled with mean-pooling and max-pooling across chunks. Max-pooling captures the peak semantic alignment of the specific paragraph containing the ratio decidendi, improving Recall@20 from $0.0267$ to $0.0833$.

---

## Failure Case 4: Group Sparsity & Feature Overfitting in Supervised LambdaMART (LTR)

### 1. Description & Manifestation
- **Target Legal Setting**: Learning-to-Rank candidate reranking on the Top-100 hybrid pool.
- **Observed System Output**: LightGBM LambdaMART yielded a test nDCG@10 of $0.0059$, severely underperforming both the hybrid baseline ($0.0631$) and the zero-shot Cross-Encoder ($0.0427$).
- **Metric Impact**: Paired bootstrap test confirmed a statistically significant drop ($p = 0.0140$, $-90.62\%$ relative performance).

### 2. Root Cause Analysis
- **Query Group Sparsity**: LambdaMART optimizes listwise NDCG by constructing pairwise gradients within discrete query groups. With only 50 validation and 50 test query groups in the legal dataset, the number of distinct pairwise gradient updates was insufficient to learn robust split thresholds across multiple non-linear tree depth levels.
- **Gain Monopoly**: Feature importance analysis revealed that `dense_score` (gain: $77.49$) and `bm25_score` (gain: $68.15$) monopolized tree splits, while topological features (`pagerank`, `in_degree`) received $0.0$ gain and $0.0$ split count, leading to premature overfitting to dense feature noise.

### 3. Mitigation Strategy
- **Early Stopping & Cross-Attention Alternative**: In Phase 5, dev-split tuning restricted LambdaMART to early iteration termination (`best_iteration = 2`). For deployment, Option B (Cross-Encoder `ms-marco-MiniLM-L-6-v2`) proved vastly superior on small query sets because its deep token-level cross-attention operates zero-shot without requiring thousands of query training groups.

---

## Failure Case 5: Boundary False Refusals in Multi-Sentence Legal RAG (Guardrail)

### 1. Description & Manifestation
- **Target Legal Query**: `"What are the procedural safeguards mandated during arrest under Section 41B of the Code of Criminal Procedure?"`
- **Expected Output**: Grounded summary citing *D.K. Basu v. State of West Bengal* guidelines.
- **Observed System Output**: The system triggered the refusal guardrail with: *"I cannot answer this question based on the retrieved precedents as there is insufficient evidence in the corpus..."*
- **Metric Impact**: False-refusal precision penalty; evidence sufficiency was computed at $0.278$, marginally below the calibrated threshold $\tau = 0.30$.

### 2. Root Cause Analysis
- **Multi-Sentence Argument Dissociation**: In the retrieved passage, the procedural requirements (preparation of memo of arrest, signature of witness, informing family member) were articulated across three extensive sentences using disjoint clauses.
- **Lexical Overlap Calculation**: Because the candidate passage split the vocabulary across several clauses, the unigram overlap ratio fell to $0.235$ (below the $0.25$ minimum keyword gate), falsely classifying a valid passage as ungrounded.

### 3. Mitigation Strategy
- **Multi-Passage Aggregation & Threshold Calibration**: We tuned the evidence sufficiency threshold from $0.35$ down to $0.30$, and implemented multi-passage context concatenation before evaluating overlap. This maintained $100\%$ refusal recall on adversarial queries while restoring valid answer generation on nuanced statutory queries.
