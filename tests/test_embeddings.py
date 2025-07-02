import pytest
import numpy as np
from services.embeddings import CosineSimilarityScorer, BaseEmbeddingScorer
from unittest.mock import AsyncMock, MagicMock, patch

class TestBaseEmbeddingScorer:
    def test_abstract_methods(self):
        # Verify that BaseEmbeddingScorer is properly abstract
        with pytest.raises(TypeError):
            BaseEmbeddingScorer()

class TestCosineSimilarityScorer:
    @pytest.fixture
    def mock_embeddings(self):
        mock = MagicMock()
        mock.aembed_query = AsyncMock()
        return mock

    @pytest.fixture
    def scorer(self, mock_embeddings):
        with patch('services.embeddings.cosine_similarity.OpenAIEmbeddings', return_value=mock_embeddings):
            return CosineSimilarityScorer()

    @pytest.mark.asyncio
    async def test_compute_similarity_identical(self, scorer, mock_embeddings):
        # Test with identical vectors
        mock_embeddings.aembed_query.side_effect = [
            [1.0, 0.5, 0.3],  # First call
            [1.0, 0.5, 0.3]   # Second call
        ]
        
        similarity = await scorer.compute_similarity("text1", "text1")
        assert np.isclose(similarity, 1.0)  # Should be exactly 1.0 for identical vectors
        assert mock_embeddings.aembed_query.call_count == 2

    @pytest.mark.asyncio
    async def test_compute_similarity_orthogonal(self, scorer, mock_embeddings):
        # Test with orthogonal vectors
        mock_embeddings.aembed_query.side_effect = [
            [1.0, 0.0, 0.0],  # First call
            [0.0, 1.0, 0.0]   # Second call
        ]
        
        similarity = await scorer.compute_similarity("text1", "text2")
        assert np.isclose(similarity, 0.0)  # Should be 0.0 for orthogonal vectors
        assert mock_embeddings.aembed_query.call_count == 2

    @pytest.mark.asyncio
    async def test_compute_similarity_opposite(self, scorer, mock_embeddings):
        # Test with opposite vectors
        mock_embeddings.aembed_query.side_effect = [
            [1.0, 0.5, 0.3],    # First call
            [-1.0, -0.5, -0.3]  # Second call
        ]
        
        similarity = await scorer.compute_similarity("text1", "text2")
        assert np.isclose(similarity, -1.0)  # Should be -1.0 for opposite vectors
        assert mock_embeddings.aembed_query.call_count == 2

    @pytest.mark.asyncio
    async def test_compute_similarity_partial(self, scorer, mock_embeddings):
        # Test with partially similar vectors
        mock_embeddings.aembed_query.side_effect = [
            [1.0, 0.0, 0.0],  # First call
            [0.7, 0.7, 0.0]   # Second call
        ]
        
        similarity = await scorer.compute_similarity("text1", "text2")
        assert 0.0 < similarity < 1.0  # Should be between 0 and 1
        assert mock_embeddings.aembed_query.call_count == 2

    def test_initialization(self):
        with patch('services.embeddings.cosine_similarity.OpenAIEmbeddings') as mock_embeddings:
            scorer = CosineSimilarityScorer()
            assert mock_embeddings.called
            assert hasattr(scorer, 'embeddings') 