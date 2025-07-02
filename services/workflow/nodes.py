from typing import Dict, Any
import numpy as np
from ..parsers import PDFResumeParser, BaseResumeParser
from ..embeddings import CosineSimilarityScorer, BaseEmbeddingScorer
from ..scorers import ChatGPTScorer, BaseLLMScorer
from .base import (
    BaseNode, BaseParserNode, BaseEmbeddingNode,
    BaseScoringNode, BaseFeedbackNode, BaseDecisionNode
)
from services.utils import logger

class ResumeParserNode(BaseParserNode):
    """Node for parsing resume and job description"""
    def __init__(self, parser: BaseResumeParser = None):
        self.parser = parser or PDFResumeParser()

    async def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        logger.info("Starting document parsing")
        resume_text = await self.parser.parse_file(state["resume_file"])
        job_text = await self.parser.parse_file(state["job_file"])
        
        logger.debug(f"Parsed resume length: {len(resume_text)} chars")
        logger.debug(f"Parsed job description length: {len(job_text)} chars")
        logger.debug("=== Parsed Resume Text ===\n" + resume_text[:100] + "...(truncated)")
        logger.debug("=== Parsed Job Description ===\n" + job_text[:100] + "...(truncated)")
        
        return {
            **state,
            "resume_text": resume_text,
            "job_desc": job_text
        }

class TextEmbeddingNode(BaseEmbeddingNode):
    """Node for generating text embeddings"""
    def __init__(self, embedder: BaseEmbeddingScorer = None):
        self.embedder = embedder or CosineSimilarityScorer()

    async def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        logger.debug("Generating embeddings for resume and job description")
        resume_emb = await self.embedder.embeddings.aembed_query(state["resume_text"])
        job_emb = await self.embedder.embeddings.aembed_query(state["job_desc"])
        
        logger.debug(f"Generated embeddings - Resume: {len(resume_emb)} dims, Job: {len(job_emb)} dims")
        return {
            **state,
            "resume_emb": resume_emb,
            "job_emb": job_emb
        }

class SimilarityScoreNode(BaseEmbeddingNode):
    """Node for computing similarity scores"""
    async def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        vec1 = np.array(state["resume_emb"])
        vec2 = np.array(state["job_emb"])
        
        similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        logger.info(f"Computed embedding similarity score: {similarity:.4f}")
        return {
            **state,
            "cosine_score": float(similarity)
        }

class TechnicalSkillsNode(BaseScoringNode):
    """Node for evaluating technical skills"""
    def __init__(self, scorer: BaseLLMScorer = None):
        self.scorer = scorer or ChatGPTScorer()

    async def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        logger.info("Evaluating technical skills")
        logger.debug("=== Technical Skills Evaluation Request ===")
        logger.debug(f"Resume Text:\n{state['resume_text'][:100]}...(truncated)")
        logger.debug(f"Job Description:\n{state['job_desc'][:100]}...(truncated)")
        logger.debug("Context: technical skills and experience")
        
        score, explanation = await self.scorer.score(
            state["resume_text"],
            state["job_desc"],
            context="technical skills and experience"
        )
        
        logger.info(f"Technical skills score: {score:.2f}")
        logger.debug(f"Technical skills explanation:\n{explanation}")
        
        return {
            **state,
            "skill_score": score,
            "skill_explain": explanation
        }

class CulturalFitNode(BaseScoringNode):
    """Node for evaluating cultural fit"""
    def __init__(self, scorer: BaseLLMScorer = None):
        self.scorer = scorer or ChatGPTScorer()

    async def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        logger.info("Evaluating cultural fit")
        logger.debug("=== Cultural Fit Evaluation Request ===")
        logger.debug(f"Resume Text:\n{state['resume_text'][:100]}...(truncated)")
        logger.debug(f"Job Description:\n{state['job_desc'][:100]}...(truncated)")
        logger.debug("Context: cultural fit and soft skills")
        
        score, explanation = await self.scorer.score(
            state["resume_text"],
            state["job_desc"],
            context="cultural fit and soft skills"
        )
        
        logger.info(f"Cultural fit score: {score:.2f}")
        logger.debug(f"Cultural fit explanation:\n{explanation}")
        
        return {
            **state,
            "culture_score": score,
            "culture_explain": explanation
        }

class ScoreCombinerNode(BaseNode):
    """Node for combining different scores"""
    def __init__(self, weights: Dict[str, float] = None):
        self.weights = weights or {
            "cosine": 0.3,
            "technical": 0.4,
            "cultural": 0.3
        }

    async def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        cosine_score_normalized = state["cosine_score"] * 100
        
        final_score = (
            self.weights["cosine"] * cosine_score_normalized +
            self.weights["technical"] * state["skill_score"] +
            self.weights["cultural"] * state["culture_score"]
        )
        
        logger.info(f"Combined scores - Embedding: {cosine_score_normalized:.2f}, Technical: {state['skill_score']:.2f}, Cultural: {state['culture_score']:.2f}")
        logger.info(f"Final weighted score: {final_score:.2f}")
        logger.debug(f"Score weights used - Embedding: {self.weights['cosine']}, Technical: {self.weights['technical']}, Cultural: {self.weights['cultural']}")
        
        final_explanation = f"""
Technical Skills Assessment: {state["skill_explain"]}

Cultural Fit Assessment: {state["culture_explain"]}

Embedding Similarity Score: {cosine_score_normalized:.2f}/100 (indicating semantic relevance between resume and job requirements)
"""
        
        return {
            **state,
            "final_score": final_score,
            "final_explanation": final_explanation.strip()
        }

class FeedbackNode(BaseFeedbackNode):
    """Node for providing feedback on evaluation"""
    def __init__(self, scorer: BaseLLMScorer = None):
        self.scorer = scorer or ChatGPTScorer()

    async def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        logger.info(f"Processing feedback for iteration {state['iteration']}")
        
        current_eval = f"""
Technical Score: {state['skill_score']}
Technical Explanation: {state['skill_explain']}
Cultural Score: {state['culture_score']}
Cultural Explanation: {state['culture_explain']}
Overall Score: {state['final_score']}
Overall Explanation: {state['final_explanation']}
"""
        logger.debug("=== Feedback Evaluation Request ===")
        logger.debug(f"Current Evaluation:\n{current_eval}")
        logger.debug(f"Job Description:\n{state['job_desc'][:100]}...(truncated)")
        
        # Use the new eval_score method specifically designed for evaluation completeness
        status, feedback = await self.scorer.eval_score(
            current_eval,
            state["job_desc"]
        )
        
        feedback_status = "No changes needed" if status >= 80 else "Changes needed"
        logger.info(f"Feedback status: {feedback_status} (completeness score: {status})")
        logger.debug(f"Feedback details:\n{feedback}")
        
        return {
            **state,
            "feedback_status": feedback_status,
            "feedback_text": feedback
        }

class IterationDecisionNode(BaseDecisionNode):
    """Node for deciding whether to continue the feedback loop iteration"""
    def __init__(self, max_iterations: int = 3):
        """Initialize with maximum number of feedback loop iterations"""
        self.max_iterations = max_iterations

    def decide(self, state: Dict[str, Any]) -> str:
        """
        Decide whether to continue iterating or end the workflow.
        
        The workflow will end if either:
        1. Maximum iterations reached (returns current best result)
        2. Feedback indicates no changes needed
        
        Args:
            state: Current workflow state
            
        Returns:
            str: Either "continue" to loop back to skills evaluation,
                 or "end" to terminate the workflow
        """
        current_iteration = state.get("iteration", 1)
        
        # First check if feedback indicates no changes needed
        if state.get("feedback_status") == "No changes needed":
            logger.info("Feedback indicates no further improvements needed")
            return "end"
            
        # End the workflow if the maximum number of iterations is reached
        if current_iteration >= self.max_iterations:
            logger.warning(f"Reached iteration limit ({current_iteration}), returning current best result")
            return "end"
            
        # Increment iteration counter
        state["iteration"] = current_iteration + 1
        
        return "continue" 