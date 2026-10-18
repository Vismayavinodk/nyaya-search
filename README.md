# Nyaya-IR: Citation-Aware Multi-Stage Legal Precedent Retrieval & Grounded RAG

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![PyTest Status](https://img.shields.io/badge/pytest-46%20passed-brightgreen.svg)](tests/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Streamlit App](https://img.shields.io/badge/Demo-Streamlit%20Interactive-FF4B4B.svg)](app.py)

> **Information Retrieval Solo Course Project**  
> **Author:** Vismaya  
> **Focus:** Indian Precedent Case Retrieval & Grounded Generative Answering combining Tuned BM25+, Chunk-Pooled Dense Transformers, Leakage-Safe Citation Multigraphs, Listwise LambdaMART / Cross-Encoder Reranking, and a Calibrated Legal RAG Refusal Guardrail.

---

## 1. Project Overview & Domain Challenges

Retrieval of relevant judicial precedents in Indian law faces distinct domain challenges:
- **Binding Precedent (*Stare Decisis* under Article 141)**: Under the Constitution of India, Supreme Court law is binding on all courts. Retrieval is dual-natured: a case must be both **topically relevant** (matching the factual dispute) and **jurisprudentially authoritative** (possessing high precedent weight).
- **Judicial Verbosity**: Indian judgments frequently span 50 to 100+ pages; the operative legal principle (*ratio decidendi*) typically appears in concluding paragraphs.
- **Vocabulary & Phrase Mismatch**: Precedents from different decades use divergent phraseology for the same legal concepts.
- **Statutory Anchors**: Critical dependence on exact statutory codes (e.g., Section 302 vs 304 IPC).

**Nyaya-IR** addresses these challenges via a multi-stage architecture:
1. **Statutory Preservation & Lexical Retrieval**: BM25+ with court boilerplate elimination and atomic statutory token retention (`CONST:ART_21`, `STATUTE:IPC:SEC_302`).
2. **Dense Semantic Retrieval**: Chunk-based sliding-window encoding (BGE-Small, InLegalBERT, Multilingual-E5) with hierarchical max-pooling across judgment chunks.
3. **Citation Network Authority (Zero Leakage)**: Directed citation multigraph ($18,929$ nodes, $16,455$ edges) computing In-Degree, PageRank, and Personalized PageRank (PPR), with strict suppression of test-query outgoing edges.
4. **Multi-Modal Candidate Reranking**: Supervised listwise LambdaMART vs Deep Sentence-Pair Cross-Encoder on Top-100 hybrid candidate pools.
5. **Grounded Legal RAG Layer**: Natural-language constitutional Q&A citing precedents in `[Doc: <doc_id>]` format, backed by a dual-component evidence sufficiency guardrail that safely abstains on weak evidence ($100\%$ refusal recall).
6. **Interactive Streamlit Web Demo**: Interactive case search, component score attribution (BM25, Dense, Graph, Reranker), "Why this ranked high" explainability, and live RAG generation.

---

## 2. System Architecture

```
                  ┌──────────────────────────────────────────────┐
                  │          Query / Case Fact Summary           │
                  └──────────────────────┬───────────────────────┘
                                         │
                 ┌───────────────────────┴───────────────────────┐
                 │                                               │
                 ▼                                               ▼
     ┌───────────────────────┐                       ┌───────────────────────┐
     │   Sparse Stream:      │                       │    Dense Stream:      │
     │   Tuned BM25+         │                       │    Chunk-Pooled BGE   │
     │   (Statutory Tokens)  │                       │    (FAISS IndexFlatIP)│
     └───────────┬───────────┘                       └───────────┬───────────┘
                 │                                               │
                 └───────────────────────┬───────────────────────┘
                                         ▼
                      ┌─────────────────────────────────────┐
                      │    Hybrid Convex Fusion (α = 0.35)  │
                      └──────────────────┬──────────────────┘
                                         ▼
                      ┌─────────────────────────────────────┐
                      │    Citation Graph Authority (β=0.15)│
                      │    (In-Degree, PageRank, PPR)       │
                      └──────────────────┬──────────────────┘
                                         ▼
                      ┌─────────────────────────────────────┐
                      │    Top-100 Candidate Pool           │
                      └──────────────────┬──────────────────┘
                                         ▼
                      ┌─────────────────────────────────────┐
                      │    Reranker: Cross-Encoder /        │
                      │    LightGBM LambdaMART (LTR)        │
                      └──────────────────┬──────────────────┘
                                         │
                                         ▼
                      ┌─────────────────────────────────────┐
                      │    Isolated Legal RAG Engine        │
                      │    (Evidence Sufficiency ≥ 0.30?)   │
                      │       ├── Yes ──► Grounded Answer   │
                      │       └── No  ──► Calibrated Refusal│
                      └─────────────────────────────────────┘
```

---

## 3. Directory Structure

```
IR_2/
├── app.py                            # Phase 7: Interactive Streamlit Demonstration Web Application
├── configs/
│   ├── config.yaml                   # Master reproducible configuration (seeds, paths, hyperparameters)
│   └── colab_config.yaml             # Optimized configuration for Google Colab / Kaggle T4 GPU
├── data/
│   ├── raw/                          # Raw IL-PCR and FIRE AILA dataset directories
│   ├── sample/                       # Curated landmark Supreme Court corpus & 35-question RAG benchmark
│   │   ├── corpus.jsonl              # 10 Landmark SC judgments with complete citation topologies
│   │   ├── queries.jsonl             # Benchmark legal queries
│   │   ├── qrels.tsv                 # Standard TREC relevance judgments
│   │   └── rag_qa_benchmark.json     # 35 Curated QA examples (25 supported + 10 refusal tests)
│   └── processed/                    # Processed JSONL, edges TSV, and split caches
├── docs/
│   ├── final_report_outline_ieee.md  # Comprehensive IEEE-style final report outline
│   ├── presentation_slides_outline.md# 12-slide conference/defense presentation deck outline
│   └── teacher_qa_defense.md         # 12 challenging faculty defense Q&A with rigorous answers
├── experiments/
│   ├── results/                      # Full ablation matrices, feature importances, significance CSVs
│   │   ├── full_ablation_matrix.md   # Master cross-phase empirical benchmark ablation table
│   │   ├── error_analysis.md         # Diagnostic analysis of 5 concrete legal IR failure cases
│   │   ├── phase2_baselines_summary.md
│   │   ├── phase3_dense_hybrid_summary.md
│   │   ├── phase4_graph_summary.md
│   │   ├── phase5_rerank_summary.md
│   │   └── phase6_rag_summary.md
│   └── runs/                         # Standard TREC run files (.run) for all phases
├── src/
│   ├── data/                         # Ingestion, regex citation extraction, legal preprocessing
│   ├── retrieval/                    # Tuned BM25+, chunk-pooled dense encoder, hybrid fusion
│   ├── graph/                        # NetworkX directed citation graph & leakage-safe scoring
│   ├── rerank/                       # Multi-modal feature extractor, LightGBM LambdaMART, Cross-Encoder
│   ├── rag/                          # Grounded Legal RAG engine, refusal guardrail, evaluators
│   ├── evaluation/                   # Academic IR metrics (nDCG@k, MAP, Recall@k, MRR) & significance
│   └── utils/                        # Deterministic seed locking, configuration parser, structured logger
├── tests/                            # Comprehensive PyTest test suite (46 passed)
├── requirements.txt                  # Pinned Python package dependencies
├── run_pipeline.py                   # Unified CLI runner for Phases 1-3
├── run_phase4_graph.py               # Runner for Phase 4 citation graph & leakage tests
├── run_phase5_rerank.py              # Runner for Phase 5 LambdaMART & Cross-Encoder reranking
├── run_phase6_rag.py                 # Runner for Phase 6 Legal RAG benchmark & refusal evaluation
└── README.md
```

---

## 4. Installation & Environment Setup

### Local Setup (Windows / Linux / macOS)
```bash
# 1. Clone repository and navigate to root
cd d:/tcs/packages/sem9/IR_2

# 2. Install pinned dependencies
pip install -r requirements.txt

# 3. Verify complete system integrity across all 7 phases
pytest tests/ -v
# Output: 46 passed in ~30 seconds
```

---

## 5. End-to-End Reproduction Steps (Phases 1 to 7)

Every stage of **Nyaya-IR** is fully reproducible via dedicated CLI entrypoints:

| Phase | Description | Command | Output Artifact |
| :--- | :--- | :--- | :--- |
| **Phase 1** | Ingestion & Citation Extraction | `python run_pipeline.py --mode data` | `data/processed/il_pcr/citation_edges.tsv` |
| **Phase 2** | Sparse BM25 Baselines & Dev Tuning | `python run_pipeline.py --mode sparse` | `experiments/results/phase2_baselines_summary.md` |
| **Phase 3** | Dense Retrieval & Hybrid Fusion | `python run_pipeline.py --mode dense` | `experiments/results/phase3_dense_hybrid_summary.md` |
| **Phase 4** | Citation Graph & Zero-Leakage Test | `python run_phase4_graph.py` | `experiments/results/phase4_graph_summary.md` |
| **Phase 5** | LambdaMART & Cross-Encoder Rerank | `python run_phase5_rerank.py` | `experiments/results/phase5_rerank_summary.md` |
| **Phase 6** | Legal RAG Layer & Refusal Guardrail | `python run_phase6_rag.py` | `experiments/results/phase6_rag_summary.md` |
| **Phase 7** | Interactive Streamlit Web Demo | `streamlit run app.py` | Launches UI on `http://localhost:8501` |

---

## 6. Full Cross-Phase Ablation Results

Evaluated across the `il_pcr` and benchmark test splits using standard TREC academic Information Retrieval metrics:

| Phase | Pipeline Configuration | Type | Dev nDCG@10 | Test nDCG@10 | Test MAP | Test Recall@10 | Test Recall@20 | Test MRR |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Phase 2** | 1. TF-IDF Baseline | Lexical | 0.3709 | **0.3854** | **0.3235** | **0.4741** | **0.5807** | **0.4569** |
| **Phase 2** | 2. BM25 Default ($k_1=1.5, b=0.75$) | Lexical | 0.2660 | 0.2640 | 0.2085 | 0.3621 | 0.4466 | 0.3091 |
| **Phase 2** | 3. BM25 Tuned ($k_1=2.2, b=0.85$) | Lexical | 0.2890 | 0.2910 | 0.2361 | 0.3892 | 0.4940 | 0.3363 |
| **Phase 2** | 4. BM25+ Tuned ($k_1=1.8, b=0.85, \delta=0.25$) | Lexical | 0.2721 | 0.2692 | 0.2125 | 0.3685 | 0.4464 | 0.3130 |
| **Phase 3** | 5. Dense: BGE-Small (Max-Pooling) | Dense | 0.0141 | 0.0099 | 0.0066 | 0.0250 | 0.0633 | 0.0123 |
| **Phase 3** | 6. Dense: BGE-Small (Mean-Pooling) | Dense | 0.0535 | 0.0388 | 0.0227 | 0.0633 | 0.0833 | 0.0562 |
| **Phase 3** | 7. Dense: InLegalBERT | Dense Domain | 0.0044 | 0.0098 | 0.0044 | 0.0267 | 0.0333 | 0.0064 |
| **Phase 3** | 8. Dense: Multilingual-E5 Small | Dense | 0.0047 | 0.0230 | 0.0208 | 0.0267 | 0.0267 | 0.0225 |
| **Phase 3** | 9. Hybrid RRF ($k=80$) | Hybrid Rank | 0.0602 | 0.0531 | 0.0436 | 0.0583 | 0.0833 | 0.0799 |
| **Phase 3** | 10. Hybrid Weighted Score ($\alpha=0.35$) | Hybrid Score | 0.0617 | 0.0631 | 0.0570 | 0.0583 | 0.0833 | 0.0930 |
| **Phase 4** | 11. Graph-Interpolated ($\text{in\_degree}, \beta=0.15$) | Citation Graph | 0.2944 | **0.2949** | **0.2369** | **0.3963** | **0.4958** | **0.3404** |
| **Phase 5** | 12. Option A: LightGBM LambdaMART | Supervised LTR | 0.0223 | 0.0059 | 0.0064 | 0.0067 | 0.0333 | 0.0149 |
| **Phase 5** | 13. Option B: Cross-Encoder (`MiniLM-L6-v2`) | Cross-Attention | 0.0142 | **0.0427** | **0.0278** | **0.0633** | **0.0633** | **0.0549** |

*Detailed markdown report: [`experiments/results/full_ablation_matrix.md`](experiments/results/full_ablation_matrix.md).*

---

## 7. Statistical Significance Highlights

- **Graph Authority vs BM25 Baseline**: $t = 2.1965, p = 0.0284$ (relative gain $+1.33\%$). Confirms that citation graph authority signals provide a statistically significant lift at $\alpha = 0.05$.
- **BM25 Tuned vs Default BM25**: $t = 7.7084, p < 10^{-4}$ (relative gain $+10.23\%$).
- **Personalized PageRank Correlation**: Pearson $r = 0.1647, p < 10^{-4}$ with relevance.

---

## 8. Legal RAG Layer & Refusal Guardrail Metrics

Evaluated on the curated 35-question benchmark (25 answerable landmark questions + 10 adversarial/weak-evidence refusal tests):

| Metric | Score | Key Takeaway |
| :--- | :---: | :--- |
| **Mean Faithfulness** | **0.8841** | Factual statements strictly contained in retrieved precedents |
| **Citation Precision** | **0.8857** | Low rate of false-positive citations |
| **Citation Recall** | **0.8429** | High coverage of landmark precedent authorities |
| **Citation F1 Score** | **0.8571** | Harmonized precision/recall balance |
| **Refusal Recall** | **1.0000** | **10/10 weak/adversarial queries safely refused** |
| **Refusal Accuracy** | **0.8857** | High discrimination between answerable and weak-evidence queries |

---

## 9. Launching the Phase 7 Streamlit Application

```bash
streamlit run app.py
```
**Features included in the demo**:
1. **Interactive Case Search**: Paste case facts or select from preloaded landmark queries.
2. **Component Score Attribution**: Displays BM25 score, Dense score, Citation In-Degree, PageRank, and final Rank Score for each retrieved judgment.
3. **"Why This Ranked High" Panel**: Transparent feature breakdown explaining statutory overlaps, lexical salience, and citation authority.
4. **Grounded Legal RAG & Refusal Tab**: Natural-language question answering with explicit `[Doc: <doc_id>]` citations or calibrated abstention when evidence is insufficient.
5. **Interactive Ablation & Error Analysis**: Directly browse empirical tables and diagnostic failure cases inside the web interface.
6. **Faculty Defense Q&A Tab**: Review 12 challenging technical questions and answers.

---

## 10. Documentation Links

- **Full Ablation Matrix**: [`experiments/results/full_ablation_matrix.md`](experiments/results/full_ablation_matrix.md)
- **Diagnostic Error Analysis**: [`experiments/results/error_analysis.md`](experiments/results/error_analysis.md)
- **IEEE-Style Final Report Outline**: [`docs/final_report_outline_ieee.md`](docs/final_report_outline_ieee.md)
- **Presentation Slide Deck Outline**: [`docs/presentation_slides_outline.md`](docs/presentation_slides_outline.md)
- **Faculty / Examiner Defense Q&A**: [`docs/teacher_qa_defense.md`](docs/teacher_qa_defense.md)
