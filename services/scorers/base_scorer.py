from abc import ABC, abstractmethod
from typing import Tuple

class BaseLLMScorer(ABC):
    """Abstract base class for LLM-based scoring"""
    
    @abstractmethod
    async def score(self, resume_text: str, job_text: str) -> Tuple[float, str]:
        """Score resume against job description and return score with explanation"""
        pass 