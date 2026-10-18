"""
Legal text preprocessor tailored to Indian court judgments.
Performs court boilerplate cleaning, Indian legal stopword filtering,
and statutory token normalization for BM25 and dense retrieval.
"""

import re
from typing import List, Set


# Core domain-specific stopwords that add noise to legal case retrieval
INDIAN_LEGAL_STOPWORDS: Set[str] = {
    # Judicial boilerplate & procedural terms
    "appellant", "appellants", "respondent", "respondents", "petitioner", "petitioners",
    "learned", "counsel", "advocate", "senior", "honble", "hon'ble", "justice", "bench",
    "coram", "court", "supreme", "high", "tribunal", "registry", "registrys",
    "appeal", "appeals", "petition", "petitions", "writ", "special", "leave", "slp",
    "impugned", "judgment", "order", "decree", "dated", "annexure", "exhibit",
    "affidavit", "rejoinder", "counter", "written", "statement", "plaint",
    "herein", "therein", "wherein", "hereunder", "aforesaid", "abovementioned",
    "inter", "alia", "suo", "motu", "versus", "vs", "v",
    # Standard English stopwords
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "with",
    "by", "about", "against", "between", "into", "through", "during", "before",
    "after", "above", "below", "from", "up", "down", "in", "out", "over", "under",
    "again", "further", "then", "once", "here", "there", "when", "where", "why",
    "how", "all", "any", "both", "each", "few", "more", "most", "other", "some",
    "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very",
    "can", "will", "just", "don", "should", "now", "is", "are", "was", "were",
    "be", "been", "being", "have", "has", "had", "having", "do", "does", "did",
    "doing", "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you",
    "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself",
    "she", "her", "hers", "herself", "it", "its", "itself", "they", "them",
    "their", "theirs", "themselves", "what", "which", "who", "whom", "this",
    "that", "these", "those", "am", "as", "until", "while", "of"
}


class LegalPreprocessor:
    """
    Standardized preprocessor for legal documents and factual query statements.
    Ensures consistency between indexing and querying stages.
    """

    # Boilerplate patterns at top of judgments
    BOILERPLATE_PATTERNS = [
        re.compile(r"IN\s+THE\s+SUPREME\s+COURT\s+OF\s+INDIA.*?(?=JUDGMENT|ORDER|\n\n)", re.IGNORECASE | re.DOTALL),
        re.compile(r"(CIVIL|CRIMINAL)\s+APPELLATE\s+JURISDICTION.*?(?=JUDGMENT|ORDER|\n\n)", re.IGNORECASE | re.DOTALL),
        re.compile(r"CIVIL\s+APPEAL\s+NO\.\s*\d+\s+OF\s+\d+", re.IGNORECASE),
        re.compile(r"CRIMINAL\s+APPEAL\s+NO\.\s*\d+\s+OF\s+\d+", re.IGNORECASE),
        re.compile(r"SPECIAL\s+LEAVE\s+PETITION\s*\([A-Z]+\)\s*NO\.\s*\d+\s+OF\s+\d+", re.IGNORECASE),
    ]

    def __init__(
        self,
        lower_case: bool = True,
        remove_boilerplate: bool = True,
        remove_stopwords: bool = True,
        min_token_len: int = 2,
        max_token_len: int = 35
    ):
        self.lower_case = lower_case
        self.remove_boilerplate = remove_boilerplate
        self.remove_stopwords = remove_stopwords
        self.min_token_len = min_token_len
        self.max_token_len = max_token_len

    def clean_text(self, text: str) -> str:
        """
        Cleans raw legal text: removes boilerplate, normalizes whitespace,
        preserves statutory alphanumeric terms.
        """
        if not text:
            return ""

        cleaned = text

        if self.remove_boilerplate:
            for pattern in self.BOILERPLATE_PATTERNS:
                cleaned = pattern.sub(" ", cleaned)

        # Standardize quotes, dashes, and extra spaces
        cleaned = re.sub(r"[\u2018\u2019]", "'", cleaned)
        cleaned = re.sub(r"[\u201C\u201D]", '"', cleaned)
        cleaned = re.sub(r"[\u2013\u2014]", "-", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned)

        return cleaned.strip()

    def tokenize(self, text: str) -> List[str]:
        """
        Tokenizes text into a clean list of terms for BM25 indexing.
        Preserves statutory codes like 'sec-302', 'art-21', 'crpc', 'ipc'.
        """
        cleaned = self.clean_text(text)
        if self.lower_case:
            cleaned = cleaned.lower()

        # Extract words and hyphenated/dotted legal tokens (e.g., sec_302, art_21, 302-ipc)
        raw_tokens = re.findall(r"\b[a-zA-Z0-9_\-\:]+\b", cleaned)

        tokens: List[str] = []
        for token in raw_tokens:
            token_clean = token.strip("-_:")
            if not token_clean:
                continue

            if len(token_clean) < self.min_token_len or len(token_clean) > self.max_token_len:
                continue

            if self.remove_stopwords and token_clean in INDIAN_LEGAL_STOPWORDS:
                continue

            # Skip purely numeric strings that are just line numbers or small page counts
            if token_clean.isdigit() and len(token_clean) < 3:
                continue

            tokens.append(token_clean)

        return tokens

    def preprocess_query(self, query_text: str) -> List[str]:
        """Tokenizes and filters a query string."""
        return self.tokenize(query_text)

    def preprocess_document(self, doc_text: str) -> List[str]:
        """Tokenizes and filters a legal judgment document."""
        return self.tokenize(doc_text)

    def head_tail_truncate(
        self,
        text: str,
        max_words: int = 400,
        head_ratio: float = 0.5
    ) -> str:
        """
        Dual-anchor windowing truncation strategy for long judicial texts.
        Preserves the beginning (facts & issues) and the ending (ratio decidendi & holding),
        avoiding information loss from naive head-only or tail-only truncation.

        Args:
            text: Cleaned judgment text.
            max_words: Maximum words in truncated output.
            head_ratio: Fraction of max_words allocated to the head window.

        Returns:
            Truncated string preserving head and tail connected by a continuity marker.
        """
        words = text.split()
        if len(words) <= max_words:
            return text

        head_count = int(max_words * head_ratio)
        tail_count = max_words - head_count

        head_part = " ".join(words[:head_count])
        tail_part = " ".join(words[-tail_count:]) if tail_count > 0 else ""

        return f"{head_part} [LEGAL_SECTION_BREAK] {tail_part}".strip()

    def chunk_text(
        self,
        text: str,
        chunk_size: int = 400,
        chunk_overlap: int = 50
    ) -> List[str]:
        """
        Sliding-window chunking strategy for long judicial texts.
        Ensures transformer subword limits are respected without losing intermediate rationale.

        Args:
            text: Cleaned judgment text.
            chunk_size: Word count per window.
            chunk_overlap: Number of overlapping words between consecutive windows.

        Returns:
            List of overlapping text chunks.
        """
        words = text.split()
        if len(words) <= chunk_size:
            return [text]

        chunks = []
        step = chunk_size - chunk_overlap
        if step <= 0:
            step = chunk_size // 2 or 1

        for i in range(0, len(words), step):
            chunk = " ".join(words[i:i + chunk_size])
            chunks.append(chunk)
            if i + chunk_size >= len(words):
                break
        return chunks
