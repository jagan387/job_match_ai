import pytest
import numpy as np
from services.workflow.nodes import (
    ResumeParserNode, TextEmbeddingNode, SimilarityScoreNode,
    TechnicalSkillsNode, CulturalFitNode, ScoreCombinerNode,
    FeedbackNode, IterationDecisionNode
)
from unittest.mock import AsyncMock, MagicMock, patch

@pytest.fixture
def mock_parser():
    parser = AsyncMock()
    parser.parse_file.return_value = "Parsed content"
    return parser

@pytest.fixture
def mock_embedder():
    embedder = MagicMock()
    embedder.embeddings = AsyncMock()
    embedder.embeddings.aembed_query.return_value = [0.1, 0.2, 0.3]
    return embedder

@pytest.fixture
def mock_scorer():
    scorer = AsyncMock()
    scorer.score.return_value = (85.0, "Test explanation")
    scorer.eval_score.return_value = (92.0, "Evaluation complete")
    return scorer

@pytest.mark.asyncio
async def test_resume_parser_node(mock_parser, mock_resume_file, mock_job_file):
    node = ResumeParserNode(parser=mock_parser)
    state = {
        "resume_file": mock_resume_file,
        "job_file": mock_job_file
    }
    
    result = await node.process(state)
    
    assert "resume_text" in result
    assert "job_desc" in result
    assert mock_parser.parse_file.call_count == 2

@pytest.mark.asyncio
async def test_text_embedding_node(mock_embedder):
    node = TextEmbeddingNode(embedder=mock_embedder)
    state = {
        "resume_text": "Sample resume",
        "job_desc": "Sample job"
    }
    
    result = await node.process(state)
    
    assert "resume_emb" in result
    assert "job_emb" in result
    assert len(result["resume_emb"]) == 3
    assert mock_embedder.embeddings.aembed_query.call_count == 2

@pytest.mark.asyncio
async def test_similarity_score_node():
    node = SimilarityScoreNode()
    state = {
        "resume_emb": [1, 0, 0],
        "job_emb": [1, 0, 0]
    }
    
    result = await node.process(state)
    
    assert "cosine_score" in result
    assert result["cosine_score"] == 1.0  # Perfect similarity for identical vectors

@pytest.mark.asyncio
async def test_technical_skills_node(mock_scorer):
    node = TechnicalSkillsNode(scorer=mock_scorer)
    state = {
        "resume_text": "Sample resume",
        "job_desc": "Sample job"
    }
    
    result = await node.process(state)
    
    assert "skill_score" in result
    assert "skill_explain" in result
    assert result["skill_score"] == 85.0
    assert mock_scorer.score.called

@pytest.mark.asyncio
async def test_cultural_fit_node(mock_scorer):
    node = CulturalFitNode(scorer=mock_scorer)
    state = {
        "resume_text": "Sample resume",
        "job_desc": "Sample job"
    }
    
    result = await node.process(state)
    
    assert "culture_score" in result
    assert "culture_explain" in result
    assert result["culture_score"] == 85.0
    assert mock_scorer.score.called

@pytest.mark.asyncio
async def test_score_combiner_node():
    node = ScoreCombinerNode()
    state = {
        "cosine_score": 0.8,
        "skill_score": 85.0,
        "culture_score": 75.0,
        "skill_explain": "Good skills",
        "culture_explain": "Good culture fit"
    }
    
    result = await node.process(state)
    
    assert "final_score" in result
    assert "final_explanation" in result
    # Test weighted average calculation
    expected_score = (0.3 * 80 + 0.4 * 85 + 0.3 * 75)
    assert abs(result["final_score"] - expected_score) < 0.01

@pytest.mark.asyncio
async def test_feedback_node(mock_scorer, sample_evaluation):
    node = FeedbackNode(scorer=mock_scorer)
    state = {
        "skill_score": 85.0,
        "skill_explain": "Good skills",
        "culture_score": 75.0,
        "culture_explain": "Good culture fit",
        "final_score": 80.0,
        "final_explanation": "Overall good",
        "job_desc": "Sample job",
        "iteration": 1
    }
    
    result = await node.process(state)
    
    assert "feedback_status" in result
    assert "feedback_text" in result
    assert mock_scorer.eval_score.called
    assert result["feedback_status"] in ["No changes needed", "Changes needed"]

def test_iteration_decision_node():
    node = IterationDecisionNode(max_iterations=3)
    
    # Test continuation
    state = {"iteration": 1, "feedback_status": "Changes needed"}
    assert node.decide(state) == "continue"
    
    # Test completion due to feedback
    state = {"iteration": 1, "feedback_status": "No changes needed"}
    assert node.decide(state) == "end"
    
    # Test completion due to max iterations
    state = {"iteration": 3, "feedback_status": "Changes needed"}
    assert node.decide(state) == "end" 