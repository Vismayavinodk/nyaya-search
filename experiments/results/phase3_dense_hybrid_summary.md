# Phase 3: Dense Retrieval and Hybrid Fusion Evaluation Report

**Dataset**: `il_pcr` (250 cases, 50 dev queries, 50 test queries)
**Hardware Device**: `auto` | **FAISS Index**: `IndexFlatIP (Cosine)`

## 1. Dense Models & Hybrid Fusion Benchmark Results

|    | Configuration                              | Type           |   Dev nDCG@10 |   Test nDCG@10 |   Test MAP |   Test Recall@10 |   Test Recall@20 |   Test MRR |
|----|--------------------------------------------|----------------|---------------|----------------|------------|------------------|------------------|------------|
|  0 | 0. Tuned BM25 Baseline (k1=2.2, b=0.85)    | Lexical        |        0.0583 |         0.0631 |     0.0550 |           0.0583 |           0.0583 |     0.0900 |
|  1 | Dense: BGE Small (General Retrieval)       | Dense (max)    |        0.0141 |         0.0099 |     0.0066 |           0.0250 |           0.0633 |     0.0123 |
|  2 | Dense: BGE Small (Mean Pooling Ablation)   | Dense (mean)   |        0.0535 |         0.0388 |     0.0227 |           0.0633 |           0.0833 |     0.0562 |
|  3 | Dense: InLegalBERT (Legal Domain Specific) | Dense (max)    |        0.0044 |         0.0098 |     0.0044 |           0.0267 |           0.0333 |     0.0064 |
|  4 | Dense: Multilingual-E5 Small (Semantic)    | Dense (max)    |        0.0047 |         0.0230 |     0.0208 |           0.0267 |           0.0267 |     0.0225 |
|  5 | Hybrid RRF (k=80)                          | Hybrid (Rank)  |        0.0602 |         0.0531 |     0.0436 |           0.0583 |           0.0833 |     0.0799 |
|  6 | Hybrid Weighted Score (alpha=0.35)         | Hybrid (Score) |        0.0617 |         0.0631 |     0.0570 |           0.0583 |           0.0833 |     0.0930 |

## 2. Paired Statistical Significance Testing (nDCG@10)

|    | Comparison                      |   mean_baseline |   mean_treatment |   relative_gain_percent |   t_statistic |   t_pvalue | is_significant_p05   |
|----|---------------------------------|-----------------|------------------|-------------------------|---------------|------------|----------------------|
|  0 | Hybrid Weighted vs BM25 Tuned   |          0.0631 |           0.0631 |                  0.0000 |        0.0000 |     1.0000 | False                |
|  1 | Hybrid Weighted vs Best Dense   |          0.0388 |           0.0631 |                 62.5122 |        0.9996 |     0.3224 | False                |
|  2 | Hybrid RRF vs BM25 Tuned        |          0.0631 |           0.0531 |                -15.8441 |       -1.0000 |     0.3222 | False                |
|  3 | BGE Max-Pooling vs Mean-Pooling |          0.0388 |           0.0099 |                -74.4953 |       -1.6927 |     0.0969 | False                |

## 3. Dev-Split Hybrid Fusion Tuning

- **Optimal Convex Weight**: `alpha = 0.35` (Dense weight = 0.35, BM25 weight = 0.65)
- **Optimal RRF Constant**: `k = 80`

## 4. Run Files & Per-Query Artifacts

- BGE Max-Pooling Run: `experiments/runs/dense_baai_bge-small-en-v1.5_max_test.run`
- InLegalBERT Run: `experiments/runs/dense_law-ai_inlegalbert_max_test.run`
- Multilingual-E5 Run: `experiments/runs/dense_intfloat_multilingual-e5-small_max_test.run`
- Hybrid RRF Run: `experiments/runs/hybrid_rrf_tuned_test.run`
- Hybrid Weighted Run: `experiments/runs/hybrid_weighted_tuned_test.run`
- Per-Query CSVs: `experiments/results/per_query_*.csv`
