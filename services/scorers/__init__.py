"""
Resume Scorer Package
"""

from .base_scorer import BaseLLMScorer
from .chatgpt_scorer import ChatGPTScorer

__all__ = ['BaseLLMScorer', 'ChatGPTScorer'] 