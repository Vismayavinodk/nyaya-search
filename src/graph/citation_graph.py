"""
Citation Graph Engine for Judicial Precedents.
Constructs directed citation network and computes topological authority features:
In-Degree, PageRank, Personalized PageRank (PPR), Co-Citation, and Bibliographic Coupling.
"""

import networkx as nx
import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Set, Optional, Tuple, Any
from src.data.schema import LegalCase, LegalQuery
from src.utils.logger import get_logger

logger = get_logger("CitationGraphEngine")


class CitationGraphEngine:
    """
    Constructs and analyzes the directed citation network of Indian court judgments.
    In the graph, edge (u, v) denotes that judgment u cites precedent judgment v.
    """

    def __init__(
        self,
        damping_factor: float = 0.85,
        max_iter: int = 100,
        tol: float = 1e-6
    ):
        self.damping_factor = float(damping_factor)
        self.max_iter = int(max_iter)
        self.tol = float(tol)

        self.graph = nx.DiGraph()
        self.doc_statutes: Dict[str, Set[str]] = {}
        self.in_degree_cache: Dict[str, float] = {}
        self.pagerank_cache: Dict[str, float] = {}
        self.hits_authorities_cache: Dict[str, float] = {}
        self.hits_hubs_cache: Dict[str, float] = {}
        self.node2vec_cache: Dict[str, np.ndarray] = {}
        self._ppr_cache: Dict[Tuple[str, ...], Dict[str, float]] = {}

    def build_graph(
        self,
        corpus: Dict[str, LegalCase],
        queries: Optional[Dict[str, LegalQuery]] = None,
        exclude_outgoing_for: Optional[Set[str]] = None
    ) -> None:
        """
        Builds the citation graph from corpus judgments and optional query cases.
        Guarantees zero test data leakage by strictly omitting/removing outgoing citation edges
        from any query/case ID specified in exclude_outgoing_for.

        Args:
            corpus: Dict of legal cases.
            queries: Optional dict of queries (to include query nodes in graph).
            exclude_outgoing_for: Set of doc_ids/query_ids whose outgoing edges MUST be omitted.
        """
        logger.info(f"Building citation graph from {len(corpus)} cases...")
        self.graph.clear()
        self.doc_statutes.clear()
        self._ppr_cache.clear()
        exclude_set = set(exclude_outgoing_for) if exclude_outgoing_for else set()

        # 1. Add all corpus documents as graph nodes
        for doc_id, case in corpus.items():
            self.graph.add_node(doc_id, title=case.title, year=case.year)
            self.doc_statutes[doc_id] = set(case.cited_statutes)

        # 1b. Add query nodes if provided
        if queries:
            for qid, q in queries.items():
                if not self.graph.has_node(qid):
                    self.graph.add_node(qid, title=qid, year=None)
                self.doc_statutes[qid] = set(q.cited_statutes)

        # 2. Add directed citation edges (u -> v where u cites v)
        edge_count = 0
        removed_leakage_edges = 0
        for doc_id, case in corpus.items():
            if doc_id in exclude_set:
                removed_leakage_edges += len(case.cited_cases)
                continue  # Leakage prevention: skip outgoing edges for held-out nodes

            for cited_doc in case.cited_cases:
                if not self.graph.has_node(cited_doc):
                    self.graph.add_node(cited_doc)
                self.graph.add_edge(doc_id, cited_doc)
                edge_count += 1

        if removed_leakage_edges > 0:
            logger.info(f"Leakage Prevention: Suppressed {removed_leakage_edges} outgoing edges from test nodes.")

        logger.info(f"Citation graph constructed: {self.graph.number_of_nodes()} nodes, {edge_count} citation edges.")
        self._compute_global_metrics()

    def remove_outgoing_edges(self, node_ids: List[str]) -> int:
        """
        Explicitly strips all outgoing citation edges for the given list of nodes.
        Recomputes topological metrics.
        """
        removed = 0
        for nid in node_ids:
            if self.graph.has_node(nid):
                out_edges = list(self.graph.out_edges(nid))
                self.graph.remove_edges_from(out_edges)
                removed += len(out_edges)
        if removed > 0:
            logger.info(f"Removed {removed} outgoing edges for {len(node_ids)} nodes to prevent test leakage.")
            self._compute_global_metrics()
        return removed

    def export_citation_edges(self, filepath: str) -> List[Tuple[str, str, str]]:
        """
        Exports the directed citation edge list to a TSV file.
        Format: source_doc_id\ttarget_doc_id\tcitation_type
        """
        import os
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        edges = []
        with open(filepath, "w", encoding="utf-8") as f:
            f.write("source_doc_id\ttarget_doc_id\tcitation_type\n")
            for u, v in self.graph.edges():
                ctype = "case_precedent"
                f.write(f"{u}\t{v}\t{ctype}\n")
                edges.append((u, v, ctype))
        logger.info(f"Exported {len(edges)} citation edges to {filepath}")
        return edges

    def _compute_global_metrics(self) -> None:
        """Precomputes normalized in-degree centrality and global PageRank."""
        n_nodes = self.graph.number_of_nodes()
        if n_nodes <= 1:
            self.in_degree_cache = {n: 0.0 for n in self.graph.nodes()}
            self.pagerank_cache = {n: 1.0 for n in self.graph.nodes()}
            return

        # 1. Normalized In-Degree Centrality
        self.in_degree_cache = nx.in_degree_centrality(self.graph)

        # 2. Global PageRank
        try:
            self.pagerank_cache = nx.pagerank(
                self.graph,
                alpha=self.damping_factor,
                max_iter=self.max_iter,
                tol=self.tol
            )
        except Exception as e:
            logger.warning(f"PageRank convergence failed ({e}), defaulting to uniform distribution.")
            self.pagerank_cache = {n: 1.0 / n_nodes for n in self.graph.nodes()}

        # 3. Kleinberg's HITS (Hubs and Authorities)
        try:
            hubs, authorities = nx.hits(self.graph, max_iter=self.max_iter, tol=self.tol)
            self.hits_hubs_cache = hubs
            self.hits_authorities_cache = authorities
        except Exception as e:
            logger.warning(f"HITS computation fallback ({e})")
            self.hits_hubs_cache = {n: 1.0 / n_nodes for n in self.graph.nodes()}
            self.hits_authorities_cache = {n: 1.0 / n_nodes for n in self.graph.nodes()}

    def get_in_degree(self, doc_id: str) -> float:
        """Returns in-degree centrality of a case."""
        return self.in_degree_cache.get(doc_id, 0.0)

    def get_pagerank(self, doc_id: str) -> float:
        """Returns global PageRank of a case."""
        return self.pagerank_cache.get(doc_id, 0.0)

    def get_hits_authority(self, doc_id: str) -> float:
        """Returns HITS authority score of a case."""
        return self.hits_authorities_cache.get(doc_id, 0.0)

    def get_hits_hub(self, doc_id: str) -> float:
        """Returns HITS hub score of a case."""
        return self.hits_hubs_cache.get(doc_id, 0.0)

    def compute_node2vec_embeddings(
        self,
        dimensions: int = 64,
        walk_length: int = 20,
        num_walks: int = 10,
        window_size: int = 5,
        seed: int = 42
    ) -> Dict[str, np.ndarray]:
        """
        Computes dense structural embeddings of legal cases using random walks and Word2Vec (Skip-gram).
        """
        import random
        from gensim.models import Word2Vec

        random.seed(seed)
        np.random.seed(seed)

        undirected_g = self.graph.to_undirected()
        nodes = [n for n in undirected_g.nodes() if undirected_g.degree(n) > 0]
        if not nodes:
            nodes = list(self.graph.nodes())

        if not nodes:
            return {}

        walks: List[List[str]] = []
        for _ in range(num_walks):
            shuffled_nodes = list(nodes)
            random.shuffle(shuffled_nodes)
            for node in shuffled_nodes:
                walk = [str(node)]
                curr = node
                for _ in range(walk_length - 1):
                    neighbors = list(undirected_g.neighbors(curr))
                    if not neighbors:
                        break
                    curr = random.choice(neighbors)
                    walk.append(str(curr))
                walks.append(walk)

        model = Word2Vec(
            sentences=walks,
            vector_size=dimensions,
            window=window_size,
            min_count=1,
            sg=1,
            workers=1,
            epochs=5,
            seed=seed
        )

        self.node2vec_cache = {
            str(n): np.array(model.wv[str(n)], dtype=np.float32)
            for n in self.graph.nodes() if str(n) in model.wv
        }
        logger.info(f"Computed Node2Vec embeddings ({dimensions}-d) for {len(self.node2vec_cache)} cases.")
        return self.node2vec_cache

    def get_node2vec_embedding(self, doc_id: str) -> Optional[np.ndarray]:
        """Returns node2vec embedding vector for a case, or None if not computed."""
        return self.node2vec_cache.get(doc_id)

    def compute_feature_relevance_correlations(
        self,
        candidate_rankings: Dict[str, List[Tuple[str, float]]],
        qrels: Dict[str, Dict[str, int]],
        queries: Optional[Dict[str, LegalQuery]] = None
    ) -> pd.DataFrame:
        """
        Computes correlation (Point-Biserial / Pearson r and Spearman rho) between
        graph centrality features and ground-truth relevance for retrieved candidates.
        """
        rows = []
        for qid, ranked_list in candidate_rankings.items():
            query_qrels = qrels.get(qid, {})
            q_statutes = queries[qid].cited_statutes if queries and qid in queries else []
            top_seed = [ranked_list[0][0]] if ranked_list else []
            ppr_dict = self.compute_personalized_pagerank(top_seed) if top_seed else self.pagerank_cache

            for doc_id, base_score in ranked_list:
                rel = query_qrels.get(doc_id, 0)
                in_deg = self.get_in_degree(doc_id)
                pr = self.get_pagerank(doc_id)
                auth = self.get_hits_authority(doc_id)
                hub = self.get_hits_hub(doc_id)
                ppr_val = ppr_dict.get(doc_id, 0.0)
                stat_jaccard = self.compute_statute_jaccard(q_statutes, doc_id) if q_statutes else 0.0

                rows.append({
                    "qid": qid,
                    "doc_id": doc_id,
                    "relevance": float(rel),
                    "is_relevant": 1.0 if rel > 0 else 0.0,
                    "in_degree": float(in_deg),
                    "pagerank": float(pr),
                    "hits_authority": float(auth),
                    "hits_hub": float(hub),
                    "personalized_pagerank": float(ppr_val),
                    "statute_jaccard": float(stat_jaccard),
                    "base_retrieval_score": float(base_score)
                })

        df_pairs = pd.DataFrame(rows)
        if df_pairs.empty or df_pairs["relevance"].nunique() <= 1:
            logger.warning("Insufficient variance in relevance labels to compute correlations.")
            return pd.DataFrame()

        feature_cols = [
            ("in_degree", "In-Degree Centrality"),
            ("pagerank", "PageRank"),
            ("hits_authority", "HITS Authority"),
            ("hits_hub", "HITS Hub"),
            ("personalized_pagerank", "Personalized PageRank (PPR)"),
            ("statute_jaccard", "Statute Jaccard Overlap"),
            ("base_retrieval_score", "Base Retrieval Score")
        ]

        summary_records = []
        for col, col_name in feature_cols:
            if col not in df_pairs.columns:
                continue
            x = df_pairs[col].values
            y = df_pairs["relevance"].values

            if np.all(x == x[0]):
                p_r, p_pval = 0.0, 1.0
                s_rho, s_pval = 0.0, 1.0
            else:
                try:
                    p_res = stats.pearsonr(x, y)
                    p_r, p_pval = float(p_res.statistic), float(p_res.pvalue)
                except Exception:
                    p_r, p_pval = 0.0, 1.0

                try:
                    s_res = stats.spearmanr(x, y)
                    s_rho, s_pval = float(s_res.statistic), float(s_res.pvalue)
                except Exception:
                    s_rho, s_pval = 0.0, 1.0

            summary_records.append({
                "Feature": col_name,
                "Pearson_r": round(p_r, 4),
                "Pearson_pvalue": round(p_pval, 5),
                "Spearman_rho": round(s_rho, 4),
                "Spearman_pvalue": round(s_pval, 5),
                "Statistically_Significant": p_pval < 0.05
            })

        df_corr = pd.DataFrame(summary_records).sort_values(by="Spearman_rho", ascending=False).reset_index(drop=True)
        return df_corr

    def compute_personalized_pagerank(
        self,
        seed_nodes: List[str]
    ) -> Dict[str, float]:
        """
        Computes Personalized PageRank (PPR) teleporting to given seed precedent cases.

        Args:
            seed_nodes: List of doc_ids acting as teleport targets.

        Returns:
            Dictionary mapping doc_id to PPR score.
        """
        valid_seeds = [s for s in seed_nodes if self.graph.has_node(s)]
        if not valid_seeds:
            return self.pagerank_cache

        cache_key = tuple(sorted(valid_seeds))
        if hasattr(self, "_ppr_cache") and cache_key in self._ppr_cache:
            return self._ppr_cache[cache_key]

        # Uniform distribution over valid seeds
        personalization = {n: 0.0 for n in self.graph.nodes()}
        weight = 1.0 / len(valid_seeds)
        for s in valid_seeds:
            personalization[s] = weight

        try:
            ppr = nx.pagerank(
                self.graph,
                alpha=self.damping_factor,
                personalization=personalization,
                max_iter=self.max_iter,
                tol=self.tol
            )
            if hasattr(self, "_ppr_cache"):
                self._ppr_cache[cache_key] = ppr
            return ppr
        except Exception as e:
            logger.warning(f"PPR computation failed ({e}), returning global PageRank.")
            return self.pagerank_cache

    def compute_statute_jaccard(self, query_statutes: List[str], doc_id: str) -> float:
        """
        Calculates Jaccard similarity between query statutory provisions and document provisions.
        """
        q_set = set(query_statutes)
        d_set = self.doc_statutes.get(doc_id, set())

        if not q_set or not d_set:
            return 0.0

        intersection = len(q_set.intersection(d_set))
        union = len(q_set.union(d_set))
        return float(intersection / union) if union > 0 else 0.0

    def compute_bibliographic_coupling(self, doc_a: str, doc_b: str) -> int:
        """
        Returns number of shared precedents cited by both doc_a and doc_b:
        |OutNeighbors(doc_a) cap OutNeighbors(doc_b)|
        """
        if not (self.graph.has_node(doc_a) and self.graph.has_node(doc_b)):
            return 0
        out_a = set(self.graph.successors(doc_a))
        out_b = set(self.graph.successors(doc_b))
        return len(out_a.intersection(out_b))

    def compute_co_citation(self, doc_a: str, doc_b: str) -> int:
        """
        Returns number of other decisions that cite both doc_a and doc_b:
        |InNeighbors(doc_a) cap InNeighbors(doc_b)|
        """
        if not (self.graph.has_node(doc_a) and self.graph.has_node(doc_b)):
            return 0
        in_a = set(self.graph.predecessors(doc_a))
        in_b = set(self.graph.predecessors(doc_b))
        return len(in_a.intersection(in_b))
