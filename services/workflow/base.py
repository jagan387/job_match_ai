from abc import ABC, abstractmethod
from typing import Dict, Any, TypedDict, Optional
from fastapi import UploadFile

class WorkflowState(TypedDict, total=False):
    # Input files
    resume_file: UploadFile
    job_file: UploadFile
    
    # Parser output
    resume_text: str
    job_desc: str
    
    # Embedding output
    resume_emb: list
    job_emb: list
    
    # Similarity output
    cosine_score: float
    
    # Skills evaluation output
    skill_score: float
    skill_explain: str
    
    # Cultural fit output
    culture_score: float
    culture_explain: str
    
    # Combined scoring output
    final_score: float
    final_explanation: str
    
    # Workflow control - tracks the overall feedback loop iterations
    iteration: int
    feedback_status: str
    feedback_text: str

class BaseNode(ABC):
    """Base interface for all workflow nodes"""
    @abstractmethod
    async def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Process the state and return updates"""
        pass

class BaseParserNode(BaseNode):
    """Interface for parser nodes"""
    pass

class BaseEmbeddingNode(BaseNode):
    """Interface for embedding-related nodes"""
    pass

class BaseScoringNode(BaseNode):
    """Interface for scoring nodes"""
    pass

class BaseFeedbackNode(BaseNode):
    """Interface for feedback nodes"""
    pass

class BaseDecisionNode(ABC):
    """Interface for nodes that make workflow decisions"""
    @abstractmethod
    def decide(self, state: Dict[str, Any]) -> str:
        """Make a decision based on the state"""
        pass 