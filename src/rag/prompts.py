"""
Prompts and prompt templates for the Legal RAG Layer.
Enforces grounded generation, explicit precedent citations, and calibrated refusal
when retrieved evidence is weak or absent.
"""

from __future__ import annotations
import re
from typing import List, Dict, Tuple, Any

REFUSAL_STRING = (
    "I cannot answer this question based on the retrieved precedents as there is insufficient "
    "or weak evidence in the legal record to support a reliable answer."
)

LEGAL_RAG_SYSTEM_PROMPT = """You are a precise, highly reliable Indian Legal Case Law Assistant.
Answer the user's question using ONLY the provided retrieved precedent passages below.

Strict Rules:
1. Citations: You MUST cite the source judgment using the exact identifier format [Doc: <doc_id>] for every factual statement.
2. Grounding: Do NOT extrapolate beyond the facts stated in the passages.
3. Weak Evidence Refusal: If the retrieved passages DO NOT contain sufficient legal evidence to answer the question, or if the question is unanswerable from the context, you MUST refuse by stating:
   "I cannot answer this question based on the retrieved precedents as there is insufficient or weak evidence in the legal record to support a reliable answer."
4. Conciseness: Keep the answer short, factual, and direct (2-4 sentences)."""


def format_rag_prompt(question: str, passages: List[Dict[str, Any]]) -> str:
    """
    Constructs the grounded QA prompt containing numbered precedents and explicit citation instructions.
    """
    if not passages:
        context_str = "No relevant precedents retrieved."
    else:
        context_blocks = []
        for i, p in enumerate(passages, start=1):
            doc_id = p.get("doc_id", f"doc_{i}")
            title = p.get("title", "Judicial Precedent")
            text = p.get("text", "").strip()
            # truncate passage text for prompt context window
            snippet = text[:600] if len(text) > 600 else text
            context_blocks.append(f"[{doc_id}] ({title}):\n{snippet}")
        context_str = "\n\n".join(context_blocks)

    prompt = (
        f"{LEGAL_RAG_SYSTEM_PROMPT}\n\n"
        f"=== RETRIEVED LEGAL PRECEDENTS ===\n"
        f"{context_str}\n\n"
        f"=== USER LEGAL QUESTION ===\n"
        f"{question}\n\n"
        f"=== GROUNDED SHORT ANSWER (with [Doc: <doc_id>] citations) ===\n"
    )
    return prompt


def extract_cited_doc_ids(text: str) -> List[str]:
    """
    Extracts all cited document IDs from generated response text.
    Matches formats such as [Doc: case_001], [case_001], [Doc ID: case_001].
    """
    patterns = [
        r"\[(?:Doc|Doc\s*ID|Precedent)?[:\s]*([a-zA-Z0-9_\-]+)\]",
        r"\b(case_[a-zA-Z0-9_\-]+)\b"
    ]
    citations = set()
    for pat in patterns:
        matches = re.findall(pat, text, flags=re.IGNORECASE)
        for m in matches:
            clean_m = m.strip()
            if clean_m.lower() not in ("doc", "id", "precedent", "refusal", "case"):
                citations.add(clean_m)
    return sorted(list(citations))
