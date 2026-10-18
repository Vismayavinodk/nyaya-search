"""
Indian Legal Citation Extractor.
Extracts and normalizes judicial citations (AIR, SCC, SCR, SCALE, ILR)
and statutory provisions (IPC, CrPC, CPC, Constitution of India) from raw court texts.
"""

import re
from typing import List, Tuple
from src.data.schema import CitationMatch


class IndianLegalCitationExtractor:
    """
    Robust rule-based parser tailored to Indian legal reporters and statutory references.
    Converts various citation formats into standardized canonical keys for graph linking.
    """

    # Common Indian law reporters
    # 1. AIR (All India Reporter): AIR 1973 SC 1461 or 1973 AIR SC 1461
    REGEX_AIR = re.compile(
        r"(?:AIR\s*(?P<year1>\d{4})\s*(?P<court1>SC|Del|Bom|Cal|Mad|All|Pat|Guj|Kar|Ker|P&H|MP|Raj|AP)\s*(?P<page1>\d+)|"
        r"(?P<year2>\d{4})\s*AIR\s*(?P<court2>SC)?\s*(?P<page2>\d+))",
        re.IGNORECASE
    )

    # 2. SCC (Supreme Court Cases): (1973) 4 SCC 225 or 2017 (10) SCC 1
    REGEX_SCC = re.compile(
        r"(?:\(?\s*(?P<year>\d{4})\s*\)?\s*(?:\(?\s*(?P<vol>\d+)\s*\)?\s*)?SCC\s*(?:\(?\s*(?P<supp>Supp)\s*\)?\s*)?(?P<page>\d+))",
        re.IGNORECASE
    )

    # 3. SCR (Supreme Court Reports): [1973] 1 SCR 1 or (1950) SCR 88
    REGEX_SCR = re.compile(
        r"(?:[\[\(]?\s*(?P<year>\d{4})\s*[\]\)]?\s*(?:\(?\s*(?P<vol>\d+|Supp)\s*\)?\s*)?SCR\s*(?P<page>\d+))",
        re.IGNORECASE
    )

    # 4. SCALE: 1997 (2) SCALE 24
    REGEX_SCALE = re.compile(
        r"(?:(?P<year>\d{4})\s*\(\s*(?P<vol>\d+)\s*\)\s*SCALE\s*(?P<page>\d+))",
        re.IGNORECASE
    )

    # 5. Constitutional Articles: Article 21 of the Constitution of India, Art. 14
    REGEX_CONSTITUTION = re.compile(
        r"(?:(?:Article|Art\.)\s*(?P<art>\d+[A-Z]?)(?:\s*(?:\([a-z0-9]+\))*)?\s*(?:of\s+(?:the\s+)?Constitution(?:\s+of\s+India)?))",
        re.IGNORECASE
    )

    # 6. Specific Statutes & Sections (IPC, CrPC, CPC, Evidence Act)
    REGEX_STATUTE_CODE = re.compile(
        r"(?:(?:Section|Sec\.|s\.)\s*(?P<sec>\d+[A-Z]?)(?:\s*\([a-z0-9]+\))*\s*(?:of\s+(?:the\s+)?)?"
        r"(?P<act>Indian\s+Penal\s+Code|IPC|Code\s+of\s+Criminal\s+Procedure|CrPC|Code\s+of\s+Civil\s+Procedure|CPC|Indian\s+Evidence\s+Act|IEA))",
        re.IGNORECASE
    )

    # Normalization mapping for statutory act names
    STATUTE_CANONICAL = {
        "indian penal code": "IPC",
        "ipc": "IPC",
        "code of criminal procedure": "CrPC",
        "crpc": "CrPC",
        "code of civil procedure": "CPC",
        "cpc": "CPC",
        "indian evidence act": "IEA",
        "iea": "IEA"
    }

    def __init__(self):
        pass

    def extract_all(self, text: str) -> List[CitationMatch]:
        """
        Extracts all case citations, statutory sections, and constitutional articles from text.

        Args:
            text: Raw judgment or query text.

        Returns:
            List of CitationMatch objects with canonical normalized representations.
        """
        if not text:
            return []

        matches: List[CitationMatch] = []
        matches.extend(self._extract_air(text))
        matches.extend(self._extract_scc(text))
        matches.extend(self._extract_scr(text))
        matches.extend(self._extract_scale(text))
        matches.extend(self._extract_constitution(text))
        matches.extend(self._extract_statutes(text))

        # Sort matches by start position in text
        matches.sort(key=lambda m: m.start_char)
        return matches

    def _extract_air(self, text: str) -> List[CitationMatch]:
        results = []
        for match in self.REGEX_AIR.finditer(text):
            raw = match.group(0).strip()
            gd = match.groupdict()
            year = gd.get("year1") or gd.get("year2")
            court = gd.get("court1") or gd.get("court2") or "SC"
            page = gd.get("page1") or gd.get("page2")
            norm = f"AIR:{year}:{court.upper()}:{page}"
            results.append(CitationMatch(
                raw_text=raw,
                normalized_form=norm,
                citation_type="case_precedent",
                reporter="AIR",
                year=int(year) if year else None,
                start_char=match.start(),
                end_char=match.end()
            ))
        return results

    def _extract_scc(self, text: str) -> List[CitationMatch]:
        results = []
        for match in self.REGEX_SCC.finditer(text):
            raw = match.group(0).strip()
            gd = match.groupdict()
            year = gd.get("year")
            vol = gd.get("vol") or "1"
            supp = gd.get("supp")
            page = gd.get("page")
            prefix = "SCC-SUPP" if supp else "SCC"
            norm = f"{prefix}:{year}:{vol}:{page}"
            results.append(CitationMatch(
                raw_text=raw,
                normalized_form=norm,
                citation_type="case_precedent",
                reporter="SCC",
                year=int(year) if year else None,
                start_char=match.start(),
                end_char=match.end()
            ))
        return results

    def _extract_scr(self, text: str) -> List[CitationMatch]:
        results = []
        for match in self.REGEX_SCR.finditer(text):
            raw = match.group(0).strip()
            gd = match.groupdict()
            year = gd.get("year")
            vol = gd.get("vol") or "1"
            page = gd.get("page")
            norm = f"SCR:{year}:{vol}:{page}"
            results.append(CitationMatch(
                raw_text=raw,
                normalized_form=norm,
                citation_type="case_precedent",
                reporter="SCR",
                year=int(year) if year else None,
                start_char=match.start(),
                end_char=match.end()
            ))
        return results

    def _extract_scale(self, text: str) -> List[CitationMatch]:
        results = []
        for match in self.REGEX_SCALE.finditer(text):
            raw = match.group(0).strip()
            gd = match.groupdict()
            year = gd.get("year")
            vol = gd.get("vol") or "1"
            page = gd.get("page")
            norm = f"SCALE:{year}:{vol}:{page}"
            results.append(CitationMatch(
                raw_text=raw,
                normalized_form=norm,
                citation_type="case_precedent",
                reporter="SCALE",
                year=int(year) if year else None,
                start_char=match.start(),
                end_char=match.end()
            ))
        return results

    def _extract_constitution(self, text: str) -> List[CitationMatch]:
        results = []
        for match in self.REGEX_CONSTITUTION.finditer(text):
            raw = match.group(0).strip()
            art = match.group("art").upper()
            norm = f"CONST:ART_{art}"
            results.append(CitationMatch(
                raw_text=raw,
                normalized_form=norm,
                citation_type="constitutional_article",
                reporter=None,
                year=1950,
                start_char=match.start(),
                end_char=match.end()
            ))
        return results

    def _extract_statutes(self, text: str) -> List[CitationMatch]:
        results = []
        for match in self.REGEX_STATUTE_CODE.finditer(text):
            raw = match.group(0).strip()
            sec = match.group("sec").upper()
            raw_act = match.group("act").strip().lower()
            canonical_act = self.STATUTE_CANONICAL.get(raw_act, raw_act.upper())
            norm = f"STATUTE:{canonical_act}:SEC_{sec}"
            results.append(CitationMatch(
                raw_text=raw,
                normalized_form=norm,
                citation_type="statute",
                reporter=None,
                year=None,
                start_char=match.start(),
                end_char=match.end()
            ))
        return results

    def get_citation_keys(self, text: str) -> Tuple[List[str], List[str]]:
        """
        Convenience method returning separate lists of normalized case citations and statutes.

        Returns:
            Tuple of (case_citations, statute_citations)
        """
        all_matches = self.extract_all(text)
        cases = [m.normalized_form for m in all_matches if m.citation_type == "case_precedent"]
        statutes = [m.normalized_form for m in all_matches if m.citation_type in ("statute", "constitutional_article")]
        # Deduplicate while preserving order
        return list(dict.fromkeys(cases)), list(dict.fromkeys(statutes))
