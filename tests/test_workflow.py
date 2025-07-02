import pytest
from services.workflow.graph import ResumeWorkflow
from unittest.mock import AsyncMock, patch
from copy import deepcopy

@pytest.fixture
def mock_nodes():
    """Mock all node processors"""
    
    def create_state_preserving_mock(updates):
        """Create a mock that preserves existing state and applies updates"""
        async def mock_process(state):
            # Create a copy of the current state
            new_state = deepcopy(state)
            # Update with new values
            new_state.update(updates)
            return new_state
        return AsyncMock(side_effect=mock_process)
    
    return {
        "parser": create_state_preserving_mock({
            "resume_text": "parsed resume", 
            "job_desc": "parsed job"
        }),
        "embedder": create_state_preserving_mock({
            "resume_emb": [0.1], 
            "job_emb": [0.2]
        }),
        "similarity": create_state_preserving_mock({
            "cosine_score": 0.8
        }),
        "technical": create_state_preserving_mock({
            "skill_score": 85.0, 
            "skill_explain": "Good skills"
        }),
        "cultural": create_state_preserving_mock({
            "culture_score": 75.0, 
            "culture_explain": "Good culture"
        }),
        "combiner": create_state_preserving_mock({
            "final_score": 80.0, 
            "final_explanation": "Overall good"
        }),
        "feedback": create_state_preserving_mock({
            "feedback_status": "No changes needed", 
            "feedback_text": "Complete"
        })
    }

@pytest.mark.asyncio
async def test_workflow_initialization():
    workflow = ResumeWorkflow(max_iterations=3)
    assert workflow.max_iterations == 3
    assert workflow.graph is not None

@pytest.mark.asyncio
async def test_workflow_execution(mock_nodes, mock_resume_file, mock_job_file):
    with patch('services.workflow.nodes.ResumeParserNode.process', mock_nodes["parser"]), \
         patch('services.workflow.nodes.TextEmbeddingNode.process', mock_nodes["embedder"]), \
         patch('services.workflow.nodes.SimilarityScoreNode.process', mock_nodes["similarity"]), \
         patch('services.workflow.nodes.TechnicalSkillsNode.process', mock_nodes["technical"]), \
         patch('services.workflow.nodes.CulturalFitNode.process', mock_nodes["cultural"]), \
         patch('services.workflow.nodes.ScoreCombinerNode.process', mock_nodes["combiner"]), \
         patch('services.workflow.nodes.FeedbackNode.process', mock_nodes["feedback"]):
        
        workflow = ResumeWorkflow(max_iterations=3)
        result = await workflow.run(mock_resume_file, mock_job_file)
        
        assert isinstance(result, dict)
        assert "final_score" in result
        assert "technical_score" in result
        assert "cultural_score" in result
        assert "embedding_score" in result
        assert "iterations" in result
        
        # Verify scores
        assert result["final_score"] == 80.0
        assert result["technical_score"] == 85.0
        assert result["cultural_score"] == 75.0
        assert abs(result["embedding_score"] - 80.0) < 0.01  # 0.8 * 100

@pytest.mark.asyncio
async def test_workflow_error_handling(mock_nodes, mock_resume_file, mock_job_file):
    error_mock = AsyncMock(side_effect=ValueError("Test error"))
    
    with patch('services.workflow.nodes.ResumeParserNode.process', error_mock), \
         pytest.raises(ValueError) as exc_info:
        workflow = ResumeWorkflow(max_iterations=3)
        await workflow.run(mock_resume_file, mock_job_file)
    
    assert "Test error" in str(exc_info.value)

@pytest.mark.asyncio
async def test_workflow_iteration_limit(mock_nodes, mock_resume_file, mock_job_file):
    # Create a new mock for feedback that always requests changes but preserves state
    async def feedback_mock(state):
        new_state = deepcopy(state)
        new_state.update({
            "feedback_status": "Changes needed",
            "feedback_text": "Need improvements"
        })
        return new_state
    
    mock_nodes["feedback"] = AsyncMock(side_effect=feedback_mock)
    
    with patch('services.workflow.nodes.ResumeParserNode.process', mock_nodes["parser"]), \
         patch('services.workflow.nodes.TextEmbeddingNode.process', mock_nodes["embedder"]), \
         patch('services.workflow.nodes.SimilarityScoreNode.process', mock_nodes["similarity"]), \
         patch('services.workflow.nodes.TechnicalSkillsNode.process', mock_nodes["technical"]), \
         patch('services.workflow.nodes.CulturalFitNode.process', mock_nodes["cultural"]), \
         patch('services.workflow.nodes.ScoreCombinerNode.process', mock_nodes["combiner"]), \
         patch('services.workflow.nodes.FeedbackNode.process', mock_nodes["feedback"]):
        
        workflow = ResumeWorkflow(max_iterations=3)
        result = await workflow.run(mock_resume_file, mock_job_file)
        
        # Should stop at max_iterations even though feedback requests changes
        assert result["iterations"] == 3  # Now we can assert exact value since state is preserved 