"""
Embeddings Package
"""

from .base_embeddings import BaseEmbeddingScorer
from .cosine_similarity import CosineSimilarityScorer

__all__ = ['BaseEmbeddingScorer', 'CosineSimilarityScorer'] 