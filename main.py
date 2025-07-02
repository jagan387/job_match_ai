from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict
from services.workflow import ResumeWorkflow
from services.utils import warning_filter

# Initialize FastAPI with warning filtering
with warning_filter:
    app = FastAPI(title="Resume Parser API")
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

@app.post("/score")
async def score_resume(
    resume: UploadFile = File(...),
    job_description: UploadFile = File(...)
) -> Dict:
    """
    Score a resume against a job description.
    Returns both embedding similarity and LLM-based scores.
    """
    with warning_filter:
        workflow = ResumeWorkflow()
        result = await workflow.run(resume, job_description)
        return result

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True) 
