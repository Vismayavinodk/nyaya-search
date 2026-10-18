# Phase 5: Multi-Modal Candidate Reranking Evaluation Report

**Dataset**: `il_pcr` (250 cases, 50 dev queries, 50 test queries)
**Candidate Pool Size**: Top-100 from Hybrid System
**Cross-Encoder Model**: `cross-encoder/ms-marco-MiniLM-L-6-v2` (Reranking Top-30)
**Zero-Leakage Audit**: Safe=True

## 1. Comparative Results on Held-out Test Split

|    | Configuration                                            | Type                        |   Dev nDCG@10 |   Test nDCG@10 |   Test MAP |   Test Recall@10 |   Test Recall@20 |   Test MRR |
|----|----------------------------------------------------------|-----------------------------|---------------|----------------|------------|------------------|------------------|------------|
|  0 | Hybrid Baseline (BM25 + Dense BGE, alpha=0.35)           | First-Stage Hybrid Pool     |        0.0601 |         0.0631 |     0.0565 |           0.0583 |           0.0833 |     0.0929 |
|  1 | Option A: LightGBM LambdaMART (Best Iter=2)              | Supervised LTR (LambdaMART) |        0.0223 |         0.0059 |     0.0064 |           0.0067 |           0.0333 |     0.0149 |
|  2 | Option B: Cross-Encoder (ms-marco-MiniLM-L-6-v2, top-30) | Cross-Encoder Transformer   |        0.0142 |         0.0427 |     0.0278 |           0.0633 |           0.0633 |     0.0549 |

## 2. LightGBM LambdaMART Feature Importances

|    | Feature     |   Importance_Gain |   Importance_Split |
|----|-------------|-------------------|--------------------|
|  1 | dense_score |           77.4916 |            21.0000 |
|  0 | bm25_score  |           68.1471 |            15.0000 |
|  4 | doc_length  |           47.1806 |            12.0000 |
|  2 | pagerank    |            0.0000 |             0.0000 |
|  3 | in_degree   |            0.0000 |             0.0000 |

## 3. Paired Statistical Significance Testing (nDCG@10)

|    | Comparison                                      |   mean_baseline |   mean_treatment |   relative_gain_percent |   t_statistic |   t_pvalue |   bootstrap_pvalue |   bootstrap_ci_lower |   bootstrap_ci_upper | is_significant_p05   |
|----|-------------------------------------------------|-----------------|------------------|-------------------------|---------------|------------|--------------------|----------------------|----------------------|----------------------|
|  0 | Option A (LightGBM) vs Hybrid Baseline          |          0.0631 |           0.0059 |                -90.6177 |       -1.9505 |     0.0568 |             0.0140 |              -0.1235 |              -0.0078 | True                 |
|  1 | Option B (Cross-Encoder) vs Hybrid Baseline     |          0.0631 |           0.0427 |                -32.3462 |       -1.3322 |     0.1890 |             0.1340 |              -0.0542 |               0.0037 | False                |
|  2 | Option A (LightGBM) vs Option B (Cross-Encoder) |          0.0427 |           0.0059 |                -86.1319 |       -2.2128 |     0.0316 |             0.0100 |              -0.0726 |              -0.0074 | True                 |

## 4. Key Findings & Insights

- **Supervised Listwise Ranking**: LightGBM LambdaMART directly optimizes NDCG@10 across lexical, dense, and graph authority signals.
- **Tuned on Dev, Evaluated Once on Test**: Early stopping determined the optimal iteration count (`best_iteration = 2`) exclusively on the validation split before scoring the test split.
- **Option A vs. Option B**: Compares tabular feature combinations (Option A) against deep sentence-pair cross-attention (Option B).

## 5. Generated Artifacts

- LightGBM Test Run: `experiments/runs\rerank_lightgbm_tuned_test.run`
- Cross-Encoder Test Run: `experiments/runs\rerank_cross_encoder_test.run`
- Feature Importance CSV: `experiments/results\feature_importance_lightgbm.csv`
- Statistical Significance CSV: `experiments/results\statistical_significance_phase5.csv`
- Per-Query CSVs: `experiments/results\per_query_rerank_lightgbm_test.csv`, `experiments/results\per_query_rerank_cross_encoder_test.csv`
