import pytest
from services.scorers.chatgpt_scorer import ChatGPTScorer
from unittest.mock import AsyncMock, MagicMock

@pytest.fixture
def mock_llm_response():
    mock_response = MagicMock()
    mock_response.content = """Score: 85.5
Rationale: Strengths: Strong Python experience, microservices architecture knowledge, and team leadership; Gaps: Less Kubernetes experience than desired"""
    return mock_response

@pytest.fixture
def mock_eval_response():
    mock_response = MagicMock()
    mock_response.content = """Score: 92.5
Rationale: Evaluation covers all key requirements and provides clear reasoning for scores"""
    return mock_response

@pytest.fixture
def scorer(monkeypatch):
    scorer = ChatGPTScorer()
    # Mock the ChatOpenAI instance
    mock_llm = AsyncMock()
    monkeypatch.setattr(scorer, "llm", mock_llm)
    return scorer, mock_llm

@pytest.mark.asyncio
async def test_score_method(scorer, sample_resume_text, sample_job_description, mock_llm_response):
    scorer_instance, mock_llm = scorer
    mock_llm.ainvoke.return_value = mock_llm_response
    
    score, rationale = await scorer_instance.score(
        sample_resume_text,
        sample_job_description,
        "technical skills"
    )
    
    assert isinstance(score, float)
    assert score == 85.5
    assert "Strong Python experience" in rationale
    assert mock_llm.ainvoke.called

@pytest.mark.asyncio
async def test_eval_score_method(scorer, sample_evaluation, sample_job_description, mock_eval_response):
    scorer_instance, mock_llm = scorer
    mock_llm.ainvoke.return_value = mock_eval_response
    
    score, rationale = await scorer_instance.eval_score(
        sample_evaluation,
        sample_job_description
    )
    
    assert isinstance(score, float)
    assert score == 92.5
    assert "covers all key requirements" in rationale
    assert mock_llm.ainvoke.called

def test_prompt_templates(scorer):
    scorer_instance, _ = scorer
    
    # Test scoring prompts
    assert "technical recruiter" in scorer_instance.score_system_template.lower()
    assert "evaluate the match" in scorer_instance.score_human_template.lower()
    
    # Test evaluation prompts
    assert "talent acquisition specialist" in scorer_instance.eval_system_template.lower()
    assert "review if this evaluation" in scorer_instance.eval_human_template.lower() 