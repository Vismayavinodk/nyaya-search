"""
Evaluation metrics for the Legal RAG Layer.
Measures:
1. Faithfulness (context grounding & hallucination detection)
2. Citation Correctness (Precision, Recall, F1 against gold precedent IDs)
3. Refusal Guardrail Performance (Refusal Precision, Recall, and Accuracy on weak-evidence queries)
"""

from __future__ import annotations
import re
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd

from src.rag.schema import LegalQAExample, RAGResponse
from src.utils.logger import get_logger

logger = get_logger("RAGEvaluator")


class RAGEvaluator:
    """
    Evaluates faithfulness, citation correctness, and refusal behavior for legal RAG systems.
    """

    STOPWORDS = {
        "a", "an", "the", "in", "on", "of", "and", "or", "for", "with", "at", "by", "from",
        "to", "is", "was", "are", "were", "be", "been", "this", "that", "it", "as", "court",
        "supreme", "india", "judgment", "held", "under", "case", "v", "state", "union"
    }

    def _tokenize_content_words(self, text: str) -> set:
        """Extracts significant lexical content words."""
        tokens = re.findall(r"\b[a-zA-Z0-9_\-]{3,}\b", text.lower())
        return {t for t in tokens if t not in self.STOPWORDS}

    def compute_faithfulness(self, response: RAGResponse, retrieved_context: str) -> float:
        """
        Measures context grounding of the generated answer against the retrieved precedents.
        Score ranges from 0.0 (complete hallucination) to 1.0 (fully grounded).
        Valid refusal on weak evidence yields 1.0 (truthful abstention).
        """
        if response.is_refusal:
            return 1.0

        ans_words = self._tokenize_content_words(response.answer)
        if not ans_words:
            return 1.0

        ctx_words = self._tokenize_content_words(retrieved_context)
        if not ctx_words:
            return 0.0

        grounded_words = ans_words.intersection(ctx_words)
        faithfulness = len(grounded_words) / len(ans_words)
        return float(np.clip(faithfulness, 0.0, 1.0))

    def compute_citation_metrics(
        self,
        predicted_docs: List[str],
        gold_docs: List[str],
        is_refusal: bool
    ) -> Tuple[float, float, float]:
        """
        Computes Citation Precision, Recall, and F1.
        """
        # If it was an unanswerable question and the system refused, citation is perfectly correct
        if not gold_docs and is_refusal:
            return 1.0, 1.0, 1.0

        if not gold_docs and not is_refusal:
            # Answered an unanswerable question with citations -> false positive citations
            prec = 0.0 if predicted_docs else 1.0
            return prec, 1.0, prec

        if gold_docs and not predicted_docs:
            return 0.0, 0.0, 0.0

        pred_set = {d.lower().strip() for d in predicted_docs}
        gold_set = {d.lower().strip() for d in gold_docs}

        true_positives = len(pred_set.intersection(gold_set))
        precision = true_positives / len(pred_set) if pred_set else 0.0
        recall = true_positives / len(gold_set) if gold_set else 0.0

        if precision + recall > 1e-9:
            f1 = (2.0 * precision * recall) / (precision + recall)
        else:
            f1 = 0.0

        return float(precision), float(recall), float(f1)

    def evaluate_benchmark(
        self,
        examples: List[LegalQAExample],
        responses: List[RAGResponse]
    ) -> Tuple[Dict[str, float], pd.DataFrame]:
        """
        Evaluates a complete benchmark batch of QA responses.
        """
        records: List[Dict[str, Any]] = []

        ex_map = {ex.question_id: ex for ex in examples}

        for resp in responses:
            if resp.question_id not in ex_map:
                continue
            ex = ex_map[resp.question_id]

            # Reconstruct context text from retrieved passages
            ctx_text = " ".join([p.get("text", "") for p in resp.retrieved_passages])

            # 1. Faithfulness
            faithfulness = self.compute_faithfulness(resp, ctx_text)

            # 2. Citation Metrics
            c_prec, c_rec, c_f1 = self.compute_citation_metrics(
                predicted_docs=resp.cited_doc_ids,
                gold_docs=ex.gold_doc_ids,
                is_refusal=resp.is_refusal
            )

            # 3. Refusal Evaluation
            # Correct refusal: is_answerable == False and is_refusal == True
            # Correct answer: is_answerable == True and is_refusal == False
            correct_behavior = (
                (not ex.is_answerable and resp.is_refusal) or
                (ex.is_answerable and not resp.is_refusal)
            )

            records.append({
                "question_id": ex.question_id,
                "domain": ex.domain,
                "is_answerable": ex.is_answerable,
                "is_refusal": resp.is_refusal,
                "evidence_score": resp.evidence_score,
                "faithfulness": round(faithfulness, 4),
                "citation_precision": round(c_prec, 4),
                "citation_recall": round(c_rec, 4),
                "citation_f1": round(c_f1, 4),
                "correct_behavior": bool(correct_behavior),
                "cited_docs_count": len(resp.cited_doc_ids),
                "gold_docs_count": len(ex.gold_doc_ids)
            })

        df_results = pd.DataFrame(records)
        if df_results.empty:
            return {}, pd.DataFrame()

        # Aggregate Metrics
        total_q = len(df_results)
        unanswerable_mask = ~df_results["is_answerable"]
        answerable_mask = df_results["is_answerable"]

        mean_faithfulness = float(df_results["faithfulness"].mean())
        mean_citation_prec = float(df_results["citation_precision"].mean())
        mean_citation_rec = float(df_results["citation_recall"].mean())
        mean_citation_f1 = float(df_results["citation_f1"].mean())
        overall_behavior_accuracy = float(df_results["correct_behavior"].mean())

        # Refusal Metrics
        actual_refusals = df_results[df_results["is_refusal"]]
        true_refusals = df_results[df_results["is_refusal"] & unanswerable_mask]
        total_unanswerable = int(unanswerable_mask.sum())

        refusal_precision = (len(true_refusals) / len(actual_refusals)) if len(actual_refusals) > 0 else 1.0
        refusal_recall = (len(true_refusals) / total_unanswerable) if total_unanswerable > 0 else 1.0
        refusal_f1 = (2 * refusal_precision * refusal_recall / (refusal_precision + refusal_recall)) if (refusal_precision + refusal_recall) > 0 else 1.0

        summary = {
            "total_questions": total_q,
            "answerable_questions": int(answerable_mask.sum()),
            "unanswerable_questions": total_unanswerable,
            "mean_faithfulness": round(mean_faithfulness, 4),
            "citation_precision": round(mean_citation_prec, 4),
            "citation_recall": round(mean_citation_rec, 4),
            "citation_f1": round(mean_citation_f1, 4),
            "refusal_accuracy": round(overall_behavior_accuracy, 4),
            "refusal_precision": round(refusal_precision, 4),
            "refusal_recall": round(refusal_recall, 4),
            "refusal_f1": round(refusal_f1, 4)
        }

        return summary, df_results
