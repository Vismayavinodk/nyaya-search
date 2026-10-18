# Phase 4: Citation Graph & Graph Score Interpolation Evaluation Report

**Dataset**: `il_pcr` (3183 cases, 627 dev queries, 627 test queries)
**Graph Dimensions**: 18929 nodes, 16455 citation edges
**Leakage Audit**: Safe=True (0 outgoing edges from test queries)

## 1. Comparative Results across Phases (BM25 vs. Hybrid vs. Graph-Interpolated)

|    | Configuration                                      | Type                |   Dev nDCG@10 |   Test nDCG@10 |   Test MAP |   Test Recall@10 |   Test Recall@20 |   Test MRR |
|----|----------------------------------------------------|---------------------|---------------|----------------|------------|------------------|------------------|------------|
|  0 | Phase 2: Tuned BM25 Baseline                       | Lexical             |        0.2890 |         0.2910 |     0.2361 |           0.3892 |           0.4940 |     0.3363 |
|  1 | Phase 3: Hybrid Weighted Score                     | Hybrid (BM25+Dense) |        0.0617 |         0.0631 |     0.0570 |           0.0583 |           0.0833 |     0.0930 |
|  2 | Phase 4: Graph-Interpolated (in_degree, beta=0.15) | Graph-Augmented     |        0.2944 |         0.2949 |     0.2369 |           0.3963 |           0.4958 |     0.3404 |

## 2. Feature-Relevance Correlations

|    | Feature                     |   Pearson_r |   Pearson_pvalue |   Spearman_rho |   Spearman_pvalue | Statistically_Significant   |
|----|-----------------------------|-------------|------------------|----------------|-------------------|-----------------------------|
|  0 | Personalized PageRank (PPR) |      0.1647 |           0.0000 |         0.1540 |            0.0000 | True                        |
|  1 | Statute Jaccard Overlap     |      0.0936 |           0.0000 |         0.0827 |            0.0000 | True                        |
|  2 | Base Retrieval Score        |      0.0690 |           0.0000 |         0.0546 |            0.0000 | True                        |
|  3 | PageRank                    |      0.0388 |           0.0000 |         0.0287 |            0.0000 | True                        |
|  4 | In-Degree Centrality        |      0.0371 |           0.0000 |         0.0268 |            0.0000 | True                        |
|  5 | HITS Authority              |     -0.0154 |           0.0065 |         0.0128 |            0.0235 | True                        |
|  6 | HITS Hub                    |     -0.0173 |           0.0022 |        -0.0495 |            0.0000 | True                        |

## 3. Paired Statistical Significance Testing (nDCG@10)

|    | Comparison                           |   mean_baseline |   mean_treatment |   relative_gain_percent |   t_statistic |   t_pvalue | is_significant_p05   |
|----|--------------------------------------|-----------------|------------------|-------------------------|---------------|------------|----------------------|
|  0 | Graph-Interpolated vs BM25 Baseline  |          0.2910 |           0.2949 |                  1.3317 |        2.1965 |     0.0284 | True                 |
|  1 | Graph-Interpolated vs Phase 3 Hybrid |          0.0631 |           0.3480 |                451.4527 |        5.4788 |     0.0000 | True                 |

## 4. Dev-Split Graph Score Interpolation Tuning

- **Optimal Graph Signal**: `in_degree`
- **Optimal Interpolation Weight**: `beta = 0.15` (Base weight = 0.85, Graph weight = 0.15)

## 5. Generated Artifacts

- Citation Edges TSV: `data/processed\il_pcr\citation_edges.tsv`
- Interpolated Test Run: `experiments/runs/graph_interpolated_tuned_test.run`
- Feature Correlations CSV: `experiments/results\feature_relevance_correlations.csv`
- Tuning Grid CSV: `experiments/results\graph_interpolation_dev_tuning_results.csv`
- Per-Query Scores CSV: `experiments/results\per_query_graph_interpolated_tuned_test.csv`
