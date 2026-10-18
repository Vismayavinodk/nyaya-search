"""
Data Download, Inspection, and Standardization Pipeline for IL-PCR and FIRE AILA.
Generates:
1. corpus.jsonl (doc_id, text, cleaned_text, length, cited_cases, cited_statutes)
2. queries.jsonl (query_id, text, cleaned_text, length, cited_statutes)
3. qrels.tsv (standard TREC 4-column format: qid 0 did rel)
4. citation_edges.tsv (source_doc_id target_doc_id citation_type)
"""

import os
import io
import json
import zipfile
import urllib.request
import urllib.error
from typing import Dict, List, Tuple, Optional, Set
import pandas as pd
from tqdm import tqdm

from src.data.schema import LegalCase, LegalQuery, Qrel, CitationEdge
from src.data.legal_preprocessor import LegalPreprocessor
from src.data.citation_extractor import IndianLegalCitationExtractor
from src.data.sample_generator import generate_synthetic_legal_dataset
from src.utils.logger import get_logger

logger = get_logger("DownloadAndBuild")


class CorpusBuilder:
    """
    Downloads, inspects, and normalizes legal datasets into standardized TREC/JSONL outputs.
    """

    # Public repository endpoints
    FIRE_AILA_GITHUB_BASE = "https://raw.githubusercontent.com/Law-AI/aila-2019-dataset/master/"
    IL_PCR_HF_URL = "https://huggingface.co/datasets/Exploration-Lab/IL-PCSR/resolve/main/"

    def __init__(
        self,
        preprocessor: Optional[LegalPreprocessor] = None,
        citation_extractor: Optional[IndianLegalCitationExtractor] = None
    ):
        self.preprocessor = preprocessor or LegalPreprocessor()
        self.extractor = citation_extractor or IndianLegalCitationExtractor()

    def download_file(self, url: str, target_path: str, timeout: int = 30) -> bool:
        """Downloads a remote file with progress logging."""
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        try:
            logger.info(f"Downloading from: {url}")
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) LegalIR-Research/1.0"}
            )
            with urllib.request.urlopen(req, timeout=timeout) as response, open(target_path, "wb") as out_file:
                total_length = response.getheader("content-length")
                if total_length:
                    total_length = int(total_length)
                content = response.read()
                out_file.write(content)
            logger.info(f"Successfully saved to {target_path} ({len(content)} bytes)")
            return True
        except Exception as e:
            logger.warning(f"Download failed from {url}: {e}")
            return False

    def download_and_extract_fire_aila(self, raw_dir: str) -> bool:
        """
        Downloads FIRE AILA dataset files from the official repository.
        """
        os.makedirs(raw_dir, exist_ok=True)
        qrels_url = f"{self.FIRE_AILA_GITHUB_BASE}relevance_judgments_priorcases.txt"
        target_qrels = os.path.join(raw_dir, "relevance_judgments_priorcases.txt")

        success = self.download_file(qrels_url, target_qrels)
        if not success:
            logger.warning("Could not download FIRE AILA directly. Will check for existing local raw files.")
        return success

    def download_il_pcr(self, raw_dir: str) -> bool:
        """
        Downloads IL-PCR / IL-PCSR benchmark files.
        """
        os.makedirs(raw_dir, exist_ok=True)
        files = ["corpus.jsonl", "queries.jsonl", "qrels.tsv"]
        all_ok = True
        for f in files:
            url = f"{self.IL_PCR_HF_URL}{f}"
            dest = os.path.join(raw_dir, f)
            if not os.path.exists(dest):
                ok = self.download_file(url, dest)
                if not ok:
                    all_ok = False
        return all_ok

    def process_and_export(
        self,
        cases: List[LegalCase],
        queries: List[LegalQuery],
        qrels: List[Qrel],
        output_dir: str
    ) -> Dict[str, str]:
        """
        Normalizes text, cleans boilerplate, extracts citation edges,
        and exports corpus.jsonl, queries.jsonl, qrels.tsv, and citation_edges.tsv.

        Returns:
            Dictionary of exported file paths.
        """
        os.makedirs(output_dir, exist_ok=True)

        corpus_file = os.path.join(output_dir, "corpus.jsonl")
        queries_file = os.path.join(output_dir, "queries.jsonl")
        qrels_file = os.path.join(output_dir, "qrels.tsv")
        edges_file = os.path.join(output_dir, "citation_edges.tsv")

        logger.info(f"Processing and standardizing {len(cases)} cases...")
        edge_set: Set[Tuple[str, str, str]] = set()

        # 1. Process and export corpus.jsonl
        with open(corpus_file, "w", encoding="utf-8") as f:
            for case in tqdm(cases, desc="Exporting Corpus"):
                raw_text = case.text
                cleaned = self.preprocessor.clean_text(raw_text)
                word_length = len(cleaned.split()) if cleaned else 0

                # Extract citations if not already provided
                case_cites = case.cited_cases
                statute_cites = case.cited_statutes
                if not case_cites and not statute_cites and raw_text:
                    case_cites, statute_cites = self.extractor.get_citation_keys(raw_text)

                # Record citation edges (case -> cited_case)
                for cited in case_cites:
                    edge_set.add((case.doc_id, cited, "case_precedent"))

                record = {
                    "doc_id": case.doc_id,
                    "text": raw_text,
                    "cleaned_text": cleaned,
                    "length": word_length,
                    "title": case.title or case.doc_id,
                    "court": case.court or "Supreme Court of India",
                    "year": case.year,
                    "cited_cases": case_cites,
                    "cited_statutes": statute_cites,
                    "ratio_summary": case.ratio_summary,
                    "metadata": case.metadata
                }
                f.write(json.dumps(record) + "\n")

        # 2. Process and export queries.jsonl
        logger.info(f"Processing {len(queries)} queries...")
        with open(queries_file, "w", encoding="utf-8") as f:
            for q in tqdm(queries, desc="Exporting Queries"):
                raw_text = q.full_query_text
                cleaned = self.preprocessor.clean_text(raw_text)
                word_length = len(cleaned.split()) if cleaned else 0

                statute_cites = q.cited_statutes
                if not statute_cites and raw_text:
                    _, statute_cites = self.extractor.get_citation_keys(raw_text)

                record = {
                    "query_id": q.query_id,
                    "text": raw_text,
                    "cleaned_text": cleaned,
                    "length": word_length,
                    "facts": q.facts,
                    "issues": q.issues,
                    "cited_statutes": statute_cites,
                    "metadata": q.metadata
                }
                f.write(json.dumps(record) + "\n")

        # 3. Export qrels.tsv in standard TREC format
        logger.info(f"Exporting {len(qrels)} qrels to TREC format...")
        with open(qrels_file, "w", encoding="utf-8") as f:
            for qrel in qrels:
                f.write(qrel.to_trec_line())

        # 4. Export citation_edges.tsv
        logger.info(f"Exporting {len(edge_set)} citation edges...")
        with open(edges_file, "w", encoding="utf-8") as f:
            f.write("source_doc_id\ttarget_doc_id\tcitation_type\n")
            for u, v, ctype in sorted(edge_set):
                f.write(f"{u}\t{v}\t{ctype}\n")

        logger.info(f"Dataset successfully built in {output_dir}")
        return {
            "corpus": corpus_file,
            "queries": queries_file,
            "qrels": qrels_file,
            "edges": edges_file
        }

    def build_dataset(
        self,
        dataset_name: str,
        raw_dir: str,
        output_dir: str
    ) -> Dict[str, str]:
        """
        Orchestrates downloading and building of target dataset.
        """
        logger.info(f"Starting dataset build for '{dataset_name}'")

        if dataset_name == "sample":
            cases, queries, qrels = generate_synthetic_legal_dataset()
            return self.process_and_export(cases, queries, qrels, output_dir)

        elif dataset_name == "fire_aila":
            # Attempt download
            self.download_and_extract_fire_aila(raw_dir)
            from src.data.dataset_loaders import DatasetLoader
            loader = DatasetLoader(citation_extractor=self.extractor)
            corpus_dict, queries_dict, qrels = loader.load_fire_aila(raw_dir)
            if not corpus_dict:
                logger.warning("No raw FIRE AILA files found. Falling back to synthetic landmark dataset.")
                cases, queries, qrels = generate_synthetic_legal_dataset()
            else:
                cases = list(corpus_dict.values())
                queries = list(queries_dict.values())
            return self.process_and_export(cases, queries, qrels, output_dir)

        elif dataset_name == "il_pcr":
            self.download_il_pcr(raw_dir)
            from src.data.dataset_loaders import DatasetLoader
            loader = DatasetLoader(citation_extractor=self.extractor)
            corpus_dict, queries_dict, qrels = loader.load_il_pcr(raw_dir)
            if not corpus_dict:
                logger.warning("No raw IL-PCR files found. Falling back to synthetic landmark dataset.")
                cases, queries, qrels = generate_synthetic_legal_dataset()
            else:
                cases = list(corpus_dict.values())
                queries = list(queries_dict.values())
            return self.process_and_export(cases, queries, qrels, output_dir)

        else:
            raise ValueError(f"Unknown dataset_name: '{dataset_name}'. Choose 'sample', 'fire_aila', or 'il_pcr'.")
