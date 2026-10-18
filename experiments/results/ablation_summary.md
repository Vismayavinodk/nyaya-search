# Ablation Experiment Results

| Configuration                     |   ndcg@5 |   ndcg@10 |      map |   recall@5 |   recall@10 |   recall@20 |   mrr |
|:----------------------------------|---------:|----------:|---------:|-----------:|------------:|------------:|------:|
| 1. BM25 Alone                     | 0.907284 |  0.960886 | 0.766667 |   0.666667 |           1 |           1 |     1 |
| 2. Dense Alone                    | 0.907284 |  0.963104 | 0.777778 |   0.666667 |           1 |           1 |     1 |
| 3. Hybrid (BM25 + Dense RRF)      | 0.907284 |  0.960886 | 0.766667 |   0.666667 |           1 |           1 |     1 |
| 4. Hybrid + Citation Graph        | 0.703167 |  0.703167 | 0.805556 |   1        |           1 |           1 |     1 |
| 5. Full LTR Reranker (LambdaMART) | 0.979019 |  0.979019 | 0.866667 |   1        |           1 |           1 |     1 |

*Evaluation conducted across nDCG@5, nDCG@10, MAP, Recall@k, and MRR.*
