from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from typing import Tuple
from .base_scorer import BaseLLMScorer
from dotenv import load_dotenv
from services.utils import logger

# Load environment variables (needed for OpenAI API key)
load_dotenv()

class ChatGPTScorer(BaseLLMScorer):
    def __init__(self):
        self.llm = ChatOpenAI(temperature=0.3)
        
        # System message template for scoring
        self.score_system_template = """You are an experienced technical recruiter with expertise in assessing candidates for engineering/software development roles. You always evaluate profiles realistically with your experience, leaving room for everyone to learn some skills on the job. But you also know that some required skills, experience and leadership exposure are must to even begin with. You avoid generic praise and base your scores only on factual evidence from the resume compared to the job description. Try to make safe deductions from the resume and the job description. For example, resume's don't exactly always mention the total work experience of a person, so you will have to calculate the total work experience and experience specifically managing people by looking at the various jobs held by the candidate and adding them and then compare with what the jd is asking."""
        
        # Human message template for scoring
        self.score_human_template = """Evaluate the match between the following candidate resume and job description. Focus strictly on the {context} and penalize missing or insufficient experience. Consider the following:

- Give higher weight to must-have or mandatory skills, years of experience, and leadership/ownership aspects if mentioned in the job description.
- If the candidate is too junior for the role, reduce the score significantly even if they have partial relevant exposure.
- Do not assume or guess skills unless explicitly stated in the resume.
- In a single line, provide transparent reasoning about both strengths and gaps. Strengths should cover the different requirements from Job description which are matching with the candidats resume, while gaps should cover the requirements which are not matching with the candidats resume. Don't leave this blank, instead say none observed.

Candidate Resume:
{resume}

Job Description:
{job_description}

Please respond in the following format (strictly):

Score: [decimal number between 0-100]
Rationale: [In a single line, mention both strengths and clear gaps in experience, skillset, and seniority for all responses]"""

        # System message template for evaluation completeness
        self.eval_system_template = """You are a highly experienced senior talent acquisition specialist. You're helping me review the cv/resume shortlisting done by a junior in your team. Your goal is to determine if the current cv/resume match score evaluation provides sufficient insights for decision making, NOT to assess if it's a perfect match.

Consider an evaluation complete if it:
1. Covers the key technical, skills, and experience requirements from the job description
2. Provides clear reasoning for the scores given, as per the JD and the candidate's resume
3. Identifies major strengths or gaps in the candidate's resume
4. Has internally consistent scoring
5. Is not giving vague interpretations

Do NOT require:
- Exhaustive analysis of every minor detail
- Perfect alignment with job requirements
- Multiple iterations if main points are covered
- Additional feedback if core assessment is clear"""

        # Human message template for evaluation completeness
        self.eval_human_template = """Review if this evaluation provides sufficient information for decision making. Focus on completeness and consistency, not the actual scores.

Current Evaluation:
{text_to_evaluate}

Job Description (for reference):
{job_description}

Please respond in the following format (strictly):

Score: [decimal number between 0-100, where 90+ means evaluation is complete enough]
Rationale: [Brief assessment of evaluation completeness, NOT the candidate's fit]"""

        # Create message templates for scoring
        score_system_message = SystemMessagePromptTemplate.from_template(self.score_system_template)
        score_human_message = HumanMessagePromptTemplate.from_template(self.score_human_template)
        self.score_prompt = ChatPromptTemplate.from_messages([
            score_system_message,
            score_human_message
        ])

        # Create message templates for evaluation
        eval_system_message = SystemMessagePromptTemplate.from_template(self.eval_system_template)
        eval_human_message = HumanMessagePromptTemplate.from_template(self.eval_human_template)
        self.eval_prompt = ChatPromptTemplate.from_messages([
            eval_system_message,
            eval_human_message
        ])

    async def score(self, resume: str, job_description: str, context: str = "overall match") -> Tuple[float, str]:
        """
        Score text against job description with specific context.
        
        Args:
            resume: The resume text to evaluate
            job_description: The job description to compare against
            context: What aspect to focus on (technical skills, cultural fit, etc.)
        
        Returns:
            Tuple of (score, explanation)
        """
        logger.debug("=== ChatGPT Scoring Request ===")
        logger.debug(f"Context: {context}")
        logger.debug(f"Resume:\n{resume[:100]}...(truncated)")
        logger.debug(f"Job description:\n{job_description[:100]}...(truncated)")
        
        messages = self.score_prompt.format_messages(
            resume=resume,
            job_description=job_description,
            context=context
        )
        
        # Log the exact prompt being sent
        logger.debug("=== ChatGPT Prompt ===")
        for msg in messages:
            logger.debug(f"Role: {msg.type}")
            logger.debug(f"Content:\n{msg.content[:50]}...(truncated)...{msg.content[-50:]}\n")
        
        response = await self.llm.ainvoke(messages)
        response_text = response.content
        
        logger.debug("=== ChatGPT Response ===")
        logger.debug(f"{response_text}\n")
        
        # Parse response
        lines = response_text.split('\n')
        score_line = [line for line in lines if line.startswith('Score:')][0]
        rationale_line = [line for line in lines if line.startswith('Rationale:')][0]
        
        score = float(score_line.split(':')[1].strip())
        # Keep the full rationale including "Strengths:" and "Gaps:" prefixes
        rationale = rationale_line.replace('Rationale:', '').strip()
        
        logger.debug(f"Parsed score: {score}")
        logger.debug(f"Parsed rationale: {rationale}")
        
        return score, rationale
    
    async def eval_score(self, evaluation: str, job_description: str) -> Tuple[float, str]:
        """
        Evaluate if the current assessment is complete enough.
        
        Args:
            evaluation: The current evaluation to review
            job_description: The job description for reference
            
        Returns:
            Tuple of (completeness_score, explanation)
            Score of 90+ indicates the evaluation is complete enough
        """
        logger.debug("=== ChatGPT Evaluation Completeness Request ===")
        logger.debug(f"Current evaluation:\n{evaluation[:100]}...(truncated)")
        
        messages = self.eval_prompt.format_messages(
            text_to_evaluate=evaluation,
            job_description=job_description
        )
        
        # Log the exact prompt being sent
        logger.debug("=== ChatGPT Evaluation Prompt ===")
        for msg in messages:
            logger.debug(f"Role: {msg.type}")
            logger.debug(f"Content:\n{msg.content[:50]}...(truncated)...{msg.content[-50:]}\n")
        
        response = await self.llm.ainvoke(messages)
        response_text = response.content
        
        logger.debug("=== ChatGPT Evaluation Response ===")
        logger.debug(f"{response_text[:50]}...(truncated)...{response_text[-50:]}\n")
        
        # Parse response
        lines = response_text.split('\n')
        score_line = [line for line in lines if line.startswith('Score:')][0]
        rationale_line = [line for line in lines if line.startswith('Rationale:')][0]
        
        score = float(score_line.split(':')[1].strip())
        rationale = rationale_line.split(':')[1].strip()
        
        logger.debug(f"Parsed evaluation score: {score}")
        logger.debug(f"Parsed evaluation rationale: {rationale}")
        
        return score, rationale 