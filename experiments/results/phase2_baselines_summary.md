# Phase 2: TF-IDF and BM25 Baselines Evaluation Report

**Dataset**: `il_pcr` (3183 cases, 627 dev queries, 627 test queries)

## 1. Experimental Results Comparison

|    | Model                                   |   Dev nDCG@10 |   Dev MAP |   Test nDCG@10 |   Test MAP |   Test Recall@10 |   Test Recall@20 |   Test MRR |
|----|-----------------------------------------|---------------|-----------|----------------|------------|------------------|------------------|------------|
|  0 | 1. TF-IDF Baseline                      |        0.3709 |    0.3078 |         0.3854 |     0.3235 |           0.4741 |           0.5807 |     0.4569 |
|  1 | 2. BM25 Default (k1=1.5, b=0.75)        |        0.2660 |    0.2148 |         0.2640 |     0.2085 |           0.3621 |           0.4466 |     0.3091 |
|  2 | 3. BM25 Tuned (k1=2.2, b=0.85)          |        0.2890 |    0.2364 |         0.2910 |     0.2361 |           0.3892 |           0.4940 |     0.3363 |
|  3 | 4. BM25+ Tuned (k1=1.8, b=0.85, d=0.25) |        0.2721 |    0.2187 |         0.2692 |     0.2125 |           0.3685 |           0.4464 |     0.3130 |

## 2. Paired Statistical Significance Testing (nDCG@10)

|    | Comparison                    |   mean_baseline |   mean_treatment |   relative_gain_percent |   t_statistic |   t_pvalue | is_significant_p05   |
|----|-------------------------------|-----------------|------------------|-------------------------|---------------|------------|----------------------|
|  0 | BM25 Tuned vs TF-IDF Baseline |          0.3854 |           0.2910 |                -24.4936 |       -9.9055 |     0.0000 | True                 |
|  1 | BM25 Tuned vs BM25 Default    |          0.2640 |           0.2910 |                 10.2329 |        7.7084 |     0.0000 | True                 |
|  2 | BM25+ Tuned vs BM25 Tuned     |          0.2910 |           0.2692 |                 -7.4906 |       -6.7085 |     0.0000 | True                 |

## 3. Dev Set Tuning Parameters

- **Optimal BM25Okapi**: `k1=2.2`, `b=0.85`
- **Optimal BM25+**: `k1=1.8`, `b=0.85`, `delta=0.25`

## 4. Run Files & Artifacts

- TF-IDF Test Run: `experiments/runs/tfidf_test.run`
- BM25 Default Test Run: `experiments/runs/bm25_default_test.run`
- BM25 Tuned Test Run: `experiments/runs/bm25_tuned_test.run`
- BM25+ Tuned Test Run: `experiments/runs/bm25_plus_tuned_test.run`
- Per-Query Score Files: `experiments/results/per_query_*.csv`
