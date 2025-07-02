from abc import ABC, abstractmethod

class BaseEmbeddingScorer(ABC):
    """Abstract base class for embedding-based similarity scoring"""
    
    @abstractmethod
    async def compute_similarity(self, text1: str, text2: str) -> float:
        """Compute similarity between two texts"""
        pass 