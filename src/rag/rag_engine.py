"""
Core Legal RAG Engine for Indian Precedent Retrieval and Grounded Answer Generation.
Features:
- Natural-language question ingestion
- Passage/precedent retrieval with relevance confidence scoring
- Guardrail: Refusal mechanism when evidence is weak or ungrounded
- Free open model support (Flan-T5 / SmolLM) with deterministic offline fallback
- Explicit precedent and passage citation tracking [Doc: <doc_id>]
"""

from __future__ import annotations
import os
import re
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

from src.rag.schema import RAGResponse, RAGPassage, LegalQAExample
from src.rag.prompts import format_rag_prompt, extract_cited_doc_ids, REFUSAL_STRING
from src.utils.logger import get_logger

logger = get_logger("LegalRAGEngine")


class LegalRAGEngine:
    """
    RAG generation engine combining passage retrieval, refusal guardrail, and citation-backed synthesis.
    """

    def __init__(
        self,
        model_name: str = "google/flan-t5-small",
        min_evidence_threshold: float = 0.30,
        max_context_passages: int = 3,
        device: Optional[str] = None,
        use_open_model: bool = True
    ):
        self.model_name = model_name
        self.min_evidence_threshold = float(min_evidence_threshold)
        self.max_context_passages = int(max_context_passages)
        self.use_open_model = use_open_model

        if device is None:
            import torch
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.tokenizer = None
        self.generator_model = None

        if self.use_open_model:
            self._init_model()

    def _init_model(self) -> None:
        """Initializes small open-source instruction-following model."""
        try:
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
            logger.info(f"Initializing open-source generation model '{self.model_name}' on device '{self.device}'...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.generator_model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name).to(self.device)
            logger.info("Open generation model loaded successfully.")
        except Exception as e:
            logger.warning(
                f"Could not load generator '{self.model_name}': {e}. "
                "Operating in deterministic grounded extraction mode."
            )
            self.tokenizer = None
            self.generator_model = None

    def assess_evidence_sufficiency(
        self,
        passages: List[Dict[str, Any]],
        question: str
    ) -> Tuple[bool, float]:
        """
        Calculates an evidence confidence score from retrieved passages.
        Returns (is_sufficient, confidence_score).
        """
        if not passages:
            return False, 0.0

        scores = [float(p.get("score", 0.0)) for p in passages]
        max_score = max(scores) if scores else 0.0

        # Term overlap between question and passages
        q_words = set(re.findall(r"\b[a-zA-Z0-9]{4,}\b", question.lower()))
        combined_text = " ".join([p.get("text", "") for p in passages]).lower()
        matched_words = [w for w in q_words if w in combined_text]
        overlap_ratio = len(matched_words) / len(q_words) if q_words else 0.0

        # Combined evidence score
        evidence_score = 0.6 * max_score + 0.4 * overlap_ratio

        is_sufficient = bool(evidence_score >= self.min_evidence_threshold and overlap_ratio >= 0.15)
        return is_sufficient, float(evidence_score)

    def _generate_grounded_answer(
        self,
        prompt: str,
        passages: List[Dict[str, Any]],
        question: str
    ) -> str:
        """
        Generates a grounded short answer from context passages using the open model or fallback synthesis.
        """
        if self.generator_model is not None and self.tokenizer is not None:
            try:
                inputs = self.tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True).to(self.device)
                outputs = self.generator_model.generate(
                    **inputs,
                    max_new_tokens=128,
                    temperature=0.1,
                    num_beams=2,
                    early_stopping=True
                )
                raw_ans = self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
                if raw_ans and len(raw_ans) > 10:
                    # Ensure doc citations are present
                    doc_ids = [p["doc_id"] for p in passages[:2]]
                    cites_found = extract_cited_doc_ids(raw_ans)
                    if not cites_found and doc_ids:
                        raw_ans = f"{raw_ans} [Doc: {doc_ids[0]}]"
                    return raw_ans
            except Exception as e:
                logger.warning(f"Generation inference failed: {e}. Falling back to grounded synthesis.")

        # Deterministic grounded extraction
        lead_passage = passages[0]
        did = lead_passage.get("doc_id", "precedent")
        title = lead_passage.get("title", "")
        text = lead_passage.get("text", "").strip()

        # Extract primary legal sentence from text
        sentences = [s.strip() for s in re.split(r"[.\n]", text) if len(s.strip()) > 25]
        summary_sentence = sentences[0] if sentences else text[:200]

        return f"According to {title}, {summary_sentence}. [Doc: {did}]"

    def answer_question(
        self,
        question_id: str,
        question: str,
        retrieved_passages: List[Dict[str, Any]]
    ) -> RAGResponse:
        """
        Takes a natural-language legal question and retrieved candidate passages,
        evaluates evidence strength, applies refusal guardrail if weak, and produces a cited answer.
        """
        # Truncate context to top-K passages
        context_passages = retrieved_passages[:self.max_context_passages]

        # 1. Guardrail: Assess evidence strength
        is_sufficient, evidence_score = self.assess_evidence_sufficiency(context_passages, question)

        if not is_sufficient:
            # Refusal triggered
            return RAGResponse(
                question_id=question_id,
                question=question,
                answer=REFUSAL_STRING,
                cited_doc_ids=[],
                is_refusal=True,
                evidence_score=round(evidence_score, 4),
                retrieved_passages=context_passages,
                model_name=self.model_name
            )

        # 2. Formulate grounded prompt
        prompt = format_rag_prompt(question, context_passages)

        # 3. Generate answer
        answer = self._generate_grounded_answer(prompt, context_passages, question)

        # 4. Extract citations
        cited_docs = extract_cited_doc_ids(answer)
        if not cited_docs and context_passages:
            # Inherit top retrieved passage if not explicitly cited
            cited_docs = [context_passages[0]["doc_id"]]

        return RAGResponse(
            question_id=question_id,
            question=question,
            answer=answer,
            cited_doc_ids=cited_docs,
            is_refusal=False,
            evidence_score=round(evidence_score, 4),
            retrieved_passages=context_passages,
            model_name=self.model_name
        )

    def answer_batch(
        self,
        qa_examples: List[LegalQAExample],
        retrieved_contexts: Dict[str, List[Dict[str, Any]]]
    ) -> List[RAGResponse]:
        """
        Processes a full batch of legal QA examples.
        """
        responses: List[RAGResponse] = []
        for ex in qa_examples:
            passages = retrieved_contexts.get(ex.question_id, [])
            resp = self.answer_question(
                question_id=ex.question_id,
                question=ex.question,
                retrieved_passages=passages
            )
            responses.append(resp)
        return responses
