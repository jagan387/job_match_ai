from typing import Dict, Any, cast
from langgraph.graph import StateGraph, END
from fastapi import UploadFile
from .base import WorkflowState
from .nodes import (
    ResumeParserNode, TextEmbeddingNode, SimilarityScoreNode,
    TechnicalSkillsNode, CulturalFitNode, ScoreCombinerNode,
    FeedbackNode, IterationDecisionNode
)
from services.utils import logger

class ResumeWorkflow:
    def __init__(self, max_iterations: int = 3):
        logger.info(f"Initializing Resume Workflow with max_iterations={max_iterations}")
        # Initialize nodes
        self.parser_node = ResumeParserNode()
        self.embedding_node = TextEmbeddingNode()
        self.similarity_node = SimilarityScoreNode()
        self.technical_node = TechnicalSkillsNode()
        self.cultural_node = CulturalFitNode()
        self.combiner_node = ScoreCombinerNode()
        self.feedback_node = FeedbackNode()
        self.decision_node = IterationDecisionNode(max_iterations=max_iterations)
        self.max_iterations = max_iterations
        
        # Build graph
        self.graph = self._build_graph()
        logger.info("Resume Workflow initialized successfully")

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow"""
        logger.debug("Building workflow graph")
        
        # Create workflow graph with our TypedDict state
        workflow = StateGraph(WorkflowState)

        # Add nodes with proper state typing
        async def parse_wrapper(state: WorkflowState) -> WorkflowState:
            result = await self.parser_node.process(state)
            return cast(WorkflowState, result)

        async def embed_wrapper(state: WorkflowState) -> WorkflowState:
            result = await self.embedding_node.process(state)
            return cast(WorkflowState, result)

        async def similarity_wrapper(state: WorkflowState) -> WorkflowState:
            result = await self.similarity_node.process(state)
            return cast(WorkflowState, result)

        async def skills_wrapper(state: WorkflowState) -> WorkflowState:
            result = await self.technical_node.process(state)
            return cast(WorkflowState, result)

        async def culture_wrapper(state: WorkflowState) -> WorkflowState:
            result = await self.cultural_node.process(state)
            return cast(WorkflowState, result)

        async def combine_wrapper(state: WorkflowState) -> WorkflowState:
            result = await self.combiner_node.process(state)
            return cast(WorkflowState, result)

        async def feedback_wrapper(state: WorkflowState) -> WorkflowState:
            result = await self.feedback_node.process(state)
            return cast(WorkflowState, result)

        # Add nodes with proper state handling
        workflow.add_node("parse", parse_wrapper)
        workflow.add_node("embed", embed_wrapper)
        workflow.add_node("similarity", similarity_wrapper)
        workflow.add_node("skills", skills_wrapper)
        workflow.add_node("culture", culture_wrapper)
        workflow.add_node("combine", combine_wrapper)
        workflow.add_node("feedback", feedback_wrapper)

        # Add edges
        workflow.add_edge("parse", "embed")
        workflow.add_edge("embed", "similarity")
        workflow.add_edge("similarity", "skills")
        workflow.add_edge("skills", "culture")
        workflow.add_edge("culture", "combine")
        workflow.add_edge("combine", "feedback")

        # Add conditional edge for refinement loop
        workflow.add_conditional_edges(
            "feedback",
            self.decision_node.decide,
            {
                "continue": "skills",  # Loop back to skills evaluation
                "end": END
            }
        )

        # Set entry point
        workflow.set_entry_point("parse")
        logger.debug("Workflow graph built successfully")

        return workflow.compile()

    async def run(self, resume_file: UploadFile, job_file: UploadFile) -> Dict[str, Any]:
        """Run the workflow on input files"""
        logger.info(f"Starting workflow execution for resume: {resume_file.filename}")
        
        # Initialize state with proper typing
        initial_state: WorkflowState = {
            "resume_file": resume_file,
            "job_file": job_file,
            "iteration": 1
        }

        # Run the graph with appropriate recursion limit
        try:
            # Set recursion limit to max_iterations * 15 to give plenty of buffer as recursions means total numbers of nodes and not just the loop. So, if in single loop we have 10 nodes, then recursions will be 10, even though our internal iterations will be 1.
            config = {"recursion_limit": self.max_iterations * 15}
            final_state = await self.graph.ainvoke(initial_state, config)
            logger.info(f"Workflow completed successfully after {final_state['iteration']} iterations")
        except Exception as e:
            logger.error(f"Workflow execution failed: {str(e)}")
            raise

        # Return relevant results
        results = {
            "final_score": final_state["final_score"],
            "explanation": final_state["final_explanation"],
            "technical_score": final_state["skill_score"],
            "cultural_score": final_state["culture_score"],
            "embedding_score": final_state["cosine_score"] * 100,
            "iterations": final_state["iteration"]
        }
        
        logger.info(f"Final score: {results['final_score']:.2f}, Total iterations: {results['iterations']}")
        return results 