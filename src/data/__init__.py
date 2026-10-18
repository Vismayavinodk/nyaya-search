"""
Data processing, schema definitions, legal citation extraction, and dataset loaders.
"""

from src.data.schema import LegalCase, LegalQuery, Qrel, CitationMatch, CitationEdge
from src.data.citation_extractor import IndianLegalCitationExtractor
from src.data.legal_preprocessor import LegalPreprocessor
from src.data.dataset_loaders import DatasetLoader
from src.data.sample_generator import generate_synthetic_legal_dataset
from src.data.download_and_build import CorpusBuilder

__all__ = [
    "LegalCase",
    "LegalQuery",
    "Qrel",
    "CitationMatch",
    "CitationEdge",
    "IndianLegalCitationExtractor",
    "LegalPreprocessor",
    "DatasetLoader",
    "generate_synthetic_legal_dataset",
    "CorpusBuilder",
]
