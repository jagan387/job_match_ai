from langchain.embeddings import OpenAIEmbeddings
import numpy as np
from .base_embeddings import BaseEmbeddingScorer
from dotenv import load_dotenv

# Load environment variables (needed for OpenAI API key)
load_dotenv()

class CosineSimilarityScorer(BaseEmbeddingScorer):
    def __init__(self):
        self.embeddings = OpenAIEmbeddings()
    
    async def compute_similarity(self, text1: str, text2: str) -> float:
        """Compute cosine similarity between two texts using embeddings"""
        # Get embeddings
        emb1 = await self.embeddings.aembed_query(text1)
        emb2 = await self.embeddings.aembed_query(text2)
        
        # Convert to numpy arrays
        vec1 = np.array(emb1)
        vec2 = np.array(emb2)
        
        # Compute cosine similarity
        similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        return float(similarity) 