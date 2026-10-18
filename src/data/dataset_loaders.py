"""
Dataset loaders and splitters for IL-PCR, FIRE AILA, and standardized JSONL benchmarks.
Provides unified parsing so all downstream retrieval, graph, and reranking modules
are completely decoupled from raw dataset formats.
"""

import os
import glob
import json
import random
from typing import Dict, List, Tuple, Optional
from src.data.schema import LegalCase, LegalQuery, Qrel
from src.data.citation_extractor import IndianLegalCitationExtractor
from src.data.sample_generator import generate_synthetic_legal_dataset, save_dataset_to_disk
from src.utils.logger import get_logger

logger = get_logger("DatasetLoader")


class DatasetLoader:
    """
    Unified dataset ingestion engine.
    Loads and normalizes legal corpora, query sets, and relevance judgments.
    """

    def __init__(self, citation_extractor: Optional[IndianLegalCitationExtractor] = None):
        self.extractor = citation_extractor or IndianLegalCitationExtractor()

    def load_standard_jsonl(
        self,
        corpus_path: str,
        queries_path: str,
        qrels_path: str
    ) -> Tuple[Dict[str, LegalCase], Dict[str, LegalQuery], List[Qrel]]:
        """
        Loads standardized JSONL corpus and queries with TREC qrels.

        Returns:
            Tuple of (corpus_dict, queries_dict, qrels_list)
        """
        corpus: Dict[str, LegalCase] = {}
        with open(corpus_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    data = json.loads(line)
                    case = LegalCase.from_dict(data)
                    corpus[case.doc_id] = case

        queries: Dict[str, LegalQuery] = {}
        with open(queries_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    data = json.loads(line)
                    query = LegalQuery.from_dict(data)
                    queries[query.query_id] = query

        qrels: List[Qrel] = []
        with open(qrels_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    qrels.append(Qrel.from_trec_line(line))

        logger.info(f"Loaded {len(corpus)} cases, {len(queries)} queries, and {len(qrels)} qrels.")
        return corpus, queries, qrels

    def load_or_create_sample(
        self,
        sample_dir: str = "data/sample"
    ) -> Tuple[Dict[str, LegalCase], Dict[str, LegalQuery], List[Qrel]]:
        """Loads sample benchmark from disk or creates it if not yet present."""
        corpus_file = os.path.join(sample_dir, "corpus.jsonl")
        queries_file = os.path.join(sample_dir, "queries.jsonl")
        qrels_file = os.path.join(sample_dir, "qrels.tsv")

        if not (os.path.exists(corpus_file) and os.path.exists(queries_file) and os.path.exists(qrels_file)):
            logger.info(f"Sample data not found at {sample_dir}. Generating synthetic benchmark...")
            cases, queries, qrels = generate_synthetic_legal_dataset()
            save_dataset_to_disk(cases, queries, qrels, sample_dir)

        return self.load_standard_jsonl(corpus_file, queries_file, qrels_file)

    def load_fire_aila(
        self,
        raw_dir: str
    ) -> Tuple[Dict[str, LegalCase], Dict[str, LegalQuery], List[Qrel]]:
        """
        Parses raw FIRE AILA (Artificial Intelligence for Legal Assistance) files.
        Corpus consists of precedent text files (e.g. Object_c*.txt / case_*.txt),
        queries (Query_*.txt or queries.json), and relevance_judgments.txt.
        """
        corpus: Dict[str, LegalCase] = {}
        queries: Dict[str, LegalQuery] = {}
        qrels: List[Qrel] = []

        # 1. Load Precedent Cases
        case_files = glob.glob(os.path.join(raw_dir, "**", "*.txt"), recursive=True)
        for filepath in case_files:
            filename = os.path.basename(filepath)
            if "query" in filename.lower() or "relevance" in filename.lower() or "qrel" in filename.lower():
                continue

            doc_id = os.path.splitext(filename)[0]
            try:
                with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                    text = f.read().strip()
            except Exception as e:
                logger.warning(f"Could not read case file {filepath}: {e}")
                continue

            cases, statutes = self.extractor.get_citation_keys(text)
            corpus[doc_id] = LegalCase(
                doc_id=doc_id,
                title=doc_id,
                text=text,
                cited_cases=cases,
                cited_statutes=statutes
            )

        # 2. Load Queries
        query_pattern = os.path.join(raw_dir, "**", "*query*.txt")
        query_files = glob.glob(query_pattern, recursive=True)
        for filepath in query_files:
            filename = os.path.basename(filepath)
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()

            # Handle Query_doc.txt where queries are formatted as AILA_Q1||fact text
            if "||" in content:
                for line in content.splitlines():
                    line = line.strip()
                    if not line or "||" not in line:
                        continue
                    parts = line.split("||", 1)
                    qid = parts[0].strip()
                    facts = parts[1].strip()
                    cases, statutes = self.extractor.get_citation_keys(facts)
                    queries[qid] = LegalQuery(
                        query_id=qid,
                        text=facts,
                        facts=facts,
                        cited_statutes=statutes
                    )
            else:
                qid = os.path.splitext(filename)[0]
                facts = content.strip()
                cases, statutes = self.extractor.get_citation_keys(facts)
                queries[qid] = LegalQuery(
                    query_id=qid,
                    text=facts,
                    facts=facts,
                    cited_statutes=statutes
                )

        # 3. Load Qrels
        qrel_files = glob.glob(os.path.join(raw_dir, "**", "*relevance*.txt"), recursive=True)
        for qrel_path in qrel_files:
            with open(qrel_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    if len(parts) >= 4:
                        # Standard TREC: qid 0 did rel or qid Q0 did rel
                        qid = parts[0]
                        did = parts[2]
                        rel = int(parts[3]) if parts[3].isdigit() else 1
                        qrels.append(Qrel(query_id=qid, doc_id=did, relevance=rel))
                    elif len(parts) >= 2:
                        qid = parts[0]
                        did = parts[-2] if len(parts) > 2 else parts[1]
                        rel = int(parts[-1]) if parts[-1].isdigit() else 1
                        qrels.append(Qrel(query_id=qid, doc_id=did, relevance=rel))

        if not corpus and not queries:
            raise FileNotFoundError(
                f"No FIRE AILA files found in '{raw_dir}'.\n"
                "Please place FIRE AILA files in data/raw/fire_aila:\n"
                "  - Object_casedocs/ (folder with case text files C1.txt, C2.txt...)\n"
                "  - Query_doc.txt (query fact scenarios)\n"
                "  - relevance_judgments_priorcases.txt (ground truth annotations)\n"
                "Or run on the verified landmark benchmark with: --dataset sample"
            )

        logger.info(f"Loaded FIRE AILA: {len(corpus)} cases, {len(queries)} queries, {len(qrels)} qrels")
        return corpus, queries, qrels

    def load_il_pcr(
        self,
        raw_dir: str,
        processed_dir: Optional[str] = None
    ) -> Tuple[Dict[str, LegalCase], Dict[str, LegalQuery], List[Qrel]]:
        """
        Parses IL-PCR (Indian Legal Precedent Case Retrieval) dataset.
        Supports standard JSONL formats as well as Hugging Face parquet dumps.
        If processed_dir or raw_dir already contains standardized corpus.jsonl, queries.jsonl,
        and qrels.tsv, loads them immediately for fast execution.
        """
        # 1. Fast path: check processed directory cache
        if processed_dir and os.path.exists(processed_dir):
            c_p = os.path.join(processed_dir, "corpus.jsonl")
            q_p = os.path.join(processed_dir, "queries.jsonl")
            r_p = os.path.join(processed_dir, "qrels.tsv")
            if os.path.exists(c_p) and os.path.exists(q_p) and os.path.exists(r_p):
                logger.info(f"Loading pre-standardized IL-PCR dataset from: {processed_dir}")
                return self.load_standard_jsonl(c_p, q_p, r_p)

        # 2. Check for standard JSONL in raw_dir
        corpus_path = os.path.join(raw_dir, "corpus.jsonl")
        queries_path = os.path.join(raw_dir, "queries.jsonl")
        qrels_path = os.path.join(raw_dir, "qrels.tsv")

        if os.path.exists(corpus_path) and os.path.exists(queries_path) and os.path.exists(qrels_path):
            logger.info(f"Loading standardized IL-PCR dataset from: {raw_dir}")
            return self.load_standard_jsonl(corpus_path, queries_path, qrels_path)

        corpus: Dict[str, LegalCase] = {}
        queries: Dict[str, LegalQuery] = {}
        qrels: List[Qrel] = []
        seen_qrels = set()

        # 3. Check for parquet files (Exploration-Lab/IL-PCSR schema, searching recursively)
        parquet_files = glob.glob(os.path.join(raw_dir, "**", "*.parquet"), recursive=True)
        if parquet_files:
            import pandas as pd
            import numpy as np
            logger.info(f"Found {len(parquet_files)} parquet files in {raw_dir}. Loading IL-PCSR parquet...")

            # Load candidate precedents (matches 'precedants', 'precedent_candidates', 'corpus', etc., excluding query files)
            cand_files = [
                f for f in parquet_files
                if ("preced" in f.lower() or "candidate" in f.lower() or "corpus" in f.lower())
                and "quer" not in f.lower()
            ]
            if not cand_files and parquet_files:
                cand_files = [parquet_files[0]]

            for cf in cand_files:
                logger.info(f"Loading precedent candidate judgments from: {cf}")
                df_cand = pd.read_parquet(cf)
                id_col = next((c for c in ["id", "doc_id", "case_id", "candidate_id"] if c in df_cand.columns), df_cand.columns[0])
                text_col = next((c for c in ["text", "document_text", "case_text", "content"] if c in df_cand.columns), df_cand.columns[1])
                title_col = next((c for c in ["case_title", "title", "name"] if c in df_cand.columns), None)
                date_col = next((c for c in ["date", "year"] if c in df_cand.columns), None)
                jur_col = next((c for c in ["jurisdiction", "court"] if c in df_cand.columns), None)
                stat_col = next((c for c in ["relevant_statutes", "statutes"] if c in df_cand.columns), None)
                prec_col = next((c for c in ["relevant_precedent_ids", "precedents", "cited_cases"] if c in df_cand.columns), None)

                for _, row in df_cand.iterrows():
                    did = str(row[id_col])
                    raw_val = row[text_col]
                    if isinstance(raw_val, (list, np.ndarray)):
                        txt = " ".join([str(x) for x in raw_val if x])
                    else:
                        txt = str(raw_val)

                    ttl = str(row[title_col]) if title_col and row[title_col] else did
                    court_val = str(row[jur_col]) if jur_col and row[jur_col] else "Supreme Court of India"
                    year_val = None
                    if date_col and row[date_col]:
                        d_str = str(row[date_col])[:4]
                        if d_str.isdigit():
                            year_val = int(d_str)

                    # Extract / normalize statutes and precedents
                    statutes = []
                    if stat_col and row[stat_col] is not None:
                        s_raw = row[stat_col]
                        if isinstance(s_raw, (list, np.ndarray)):
                            statutes = [str(x) for x in s_raw if x]

                    precedents = []
                    if prec_col and row[prec_col] is not None:
                        p_raw = row[prec_col]
                        if isinstance(p_raw, (list, np.ndarray)):
                            precedents = [str(x) for x in p_raw if x]

                    # Use safety cap for citation regex extraction
                    cases_extracted, stat_extracted = self.extractor.get_citation_keys(txt[:100000])
                    all_cases = list(dict.fromkeys(precedents + cases_extracted))
                    all_statutes = list(dict.fromkeys(statutes + stat_extracted))

                    corpus[did] = LegalCase(
                        doc_id=did,
                        title=ttl,
                        text=txt,
                        court=court_val,
                        year=year_val,
                        cited_cases=all_cases,
                        cited_statutes=all_statutes,
                        metadata={
                            "date": str(row[date_col]) if date_col and row[date_col] else None,
                            "jurisdiction": court_val
                        }
                    )

            # Load query splits (dev_queries, test_queries, train_queries)
            query_files = [f for f in parquet_files if f not in cand_files]
            query_files.sort()
            for qf in query_files:
                split_tag = "dev" if "dev" in qf.lower() else ("test" if "test" in qf.lower() else "train")
                logger.info(f"Loading queries ({split_tag} split) from: {qf}")
                df_q = pd.read_parquet(qf)
                qid_col = next((c for c in ["id", "query_id", "qid"] if c in df_q.columns), df_q.columns[0])
                text_col = next((c for c in ["text", "facts", "query_text", "query"] if c in df_q.columns), df_q.columns[1])
                title_col = next((c for c in ["case_title", "title", "name"] if c in df_q.columns), None)
                rel_col = next((c for c in ["relevant_precedent_ids", "precedents", "relevant_precedents", "relevant_cases"] if c in df_q.columns), None)
                stat_col = next((c for c in ["relevant_statutes", "statutes"] if c in df_q.columns), None)
                role_col = next((c for c in ["rhetorical_roles", "roles"] if c in df_q.columns), None)
                jur_col = next((c for c in ["jurisdiction", "court"] if c in df_q.columns), None)
                date_col = next((c for c in ["date", "year"] if c in df_q.columns), None)

                for _, row in df_q.iterrows():
                    qid = str(row[qid_col])
                    raw_val = row[text_col]
                    if isinstance(raw_val, (list, np.ndarray)):
                        qtxt_full = " ".join([str(x) for x in raw_val if x])
                    else:
                        qtxt_full = str(raw_val)

                    # Separate facts and issues using rhetorical role annotations if present
                    facts_str = qtxt_full
                    issues_str = None
                    if role_col and isinstance(raw_val, (list, np.ndarray)):
                        roles_arr = row[role_col]
                        if isinstance(roles_arr, (list, np.ndarray)) and len(roles_arr) == len(raw_val):
                            fact_parts = [str(t).strip() for t, r in zip(raw_val, roles_arr) if "fact" in str(r).lower() and str(t).strip()]
                            issue_parts = [str(t).strip() for t, r in zip(raw_val, roles_arr) if "issue" in str(r).lower() and str(t).strip()]
                            if fact_parts:
                                facts_str = " ".join(fact_parts)
                            if issue_parts:
                                issues_str = " ".join(issue_parts)

                    # Search text combines facts and issues without downstream reasoning/conclusions
                    query_statement = f"{facts_str} {issues_str}".strip() if issues_str else facts_str

                    statutes = []
                    if stat_col and row[stat_col] is not None:
                        s_raw = row[stat_col]
                        if isinstance(s_raw, (list, np.ndarray)):
                            statutes = [str(x) for x in s_raw if x]

                    _, stat_extracted = self.extractor.get_citation_keys(query_statement[:10000])
                    all_statutes = list(dict.fromkeys(statutes + stat_extracted))

                    queries[qid] = LegalQuery(
                        query_id=qid,
                        text=query_statement,
                        facts=facts_str,
                        issues=issues_str,
                        cited_statutes=all_statutes,
                        metadata={
                            "split": split_tag,
                            "case_title": str(row[title_col]) if title_col and row[title_col] else qid,
                            "jurisdiction": str(row[jur_col]) if jur_col and row[jur_col] else None,
                            "date": str(row[date_col]) if date_col and row[date_col] else None,
                            "full_text": qtxt_full
                        }
                    )

                    if rel_col and row[rel_col] is not None:
                        rel_items = row[rel_col]
                        if isinstance(rel_items, str):
                            try:
                                rel_items = json.loads(rel_items)
                            except Exception:
                                rel_items = [rel_items]
                        if isinstance(rel_items, (list, np.ndarray, set)):
                            for target_did in rel_items:
                                target_str = str(target_did).strip()
                                if target_str:
                                    qrel_key = (qid, target_str)
                                    if qrel_key not in seen_qrels:
                                        seen_qrels.add(qrel_key)
                                        qrels.append(Qrel(query_id=qid, doc_id=target_str, relevance=1))

            if corpus:
                logger.info(f"Loaded IL-PCSR from Parquet: {len(corpus)} cases, {len(queries)} queries, {len(qrels)} qrels")
                return corpus, queries, qrels

        # 2. Fallback to general JSON / JSONL detection
        jsonl_files = glob.glob(os.path.join(raw_dir, "*.jsonl"))
        for jf in jsonl_files:
            base = os.path.basename(jf).lower()
            with open(jf, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    item = json.loads(line)
                    if "corpus" in base or "case" in base:
                        case = LegalCase.from_dict(item)
                        corpus[case.doc_id] = case
                    elif "query" in base:
                        query = LegalQuery.from_dict(item)
                        queries[query.query_id] = query

        if not corpus:
            raise FileNotFoundError(
                f"No IL-PCR dataset files found in '{raw_dir}'.\n"
                "To run on IL-PCR, please place the dataset files in data/raw/il_pcr:\n"
                "  Option A: Download parquet files from HuggingFace Exploration-Lab/IL-PCSR\n"
                "            (precedent_candidates.parquet, dev_queries.parquet, train_queries.parquet)\n"
                "  Option B: Download the dataset zip from the ACL 2023 paper link: https://github.com/Exploration-Lab/IL-PCR\n"
                "            and extract corpus.jsonl, queries.jsonl, and qrels.tsv into data/raw/il_pcr\n\n"
                "To test the complete pipeline immediately with verified landmark SC precedents, run:\n"
                "  python run_pipeline.py --mode all --dataset sample"
            )

        return corpus, queries, qrels

    @staticmethod
    def split_queries(
        queries: Dict[str, LegalQuery],
        qrels: List[Qrel],
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        seed: int = 42
    ) -> Tuple[List[str], List[str], List[str]]:
        # Check if queries carry official benchmark partition tags
        has_official_splits = any(q.metadata.get("split") in ("train", "dev", "test") for q in queries.values())
        if has_official_splits:
            train_ids = [qid for qid, q in queries.items() if q.metadata.get("split") == "train"]
            val_ids = [qid for qid, q in queries.items() if q.metadata.get("split") == "dev"]
            test_ids = [qid for qid, q in queries.items() if q.metadata.get("split") == "test"]
            if train_ids or val_ids or test_ids:
                logger.info(f"Using official benchmark query splits: Train={len(train_ids)}, Dev={len(val_ids)}, Test={len(test_ids)}")
                return train_ids, val_ids, test_ids

        assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-5, "Split ratios must sum to 1.0"
        rng = random.Random(seed)
        query_ids = sorted(list(queries.keys()))
        rng.shuffle(query_ids)

        n_total = len(query_ids)
        n_train = max(1, int(n_total * train_ratio))
        n_val = max(1, int(n_total * val_ratio)) if n_total > 2 else 0

        train_ids = query_ids[:n_train]
        val_ids = query_ids[n_train:n_train + n_val]
        test_ids = query_ids[n_train + n_val:]

        if not test_ids and len(query_ids) >= 2:
            test_ids = [train_ids.pop()]

        return train_ids, val_ids, test_ids
