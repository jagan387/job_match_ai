# Job Match AI

A FastAPI-based service that scores resumes against job descriptions using LangChain, OpenAI embeddings, and LLM-based evaluation.

## Features

- PDF and DOCX resume parsing
- Embedding-based similarity scoring
- LLM-based resume evaluation
- Combined scoring system with weighted average
- FastAPI endpoint for easy integration

## Setup

1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file with your OpenAI API key:
```
OPENAI_API_KEY=your_api_key_here
```

## Running the API

Start the FastAPI server:
```bash
uvicorn main:app --reload
```

The API will be available at `http://localhost:8000`

## API Endpoints

### POST /score

Scores a resume against a job description.

**Request:**
- Multipart form data with two files:
  - `resume`: PDF or DOCX file
  - `job_description`: PDF or DOCX file
  - Example: curl -X POST "http://localhost:8000/score" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "resume=@samples/sample_resume.pdf" \
  -F "job_description=@samples/sample_job.pdf"

**Response:**
```json
{
    "final_score": 85.5,
    "embedding_similarity": 0.82,
    "llm_score": 87,
    "explanation": "Detailed explanation of the score..."
}
```

## Implementation Details

- Uses multiple agents for performing various tasks both in sequence and in parallel. Built using langraph's multi-agentic workflow mechanism for orchestration
- Improves the inference by using a feedback loop to enhance the results
- Leverages LangChain's UnstructuredPDFLoader for PDF parsing
- Implements OpenAI embeddings for semantic similarity
- Uses ChatGPT for detailed resume evaluation
- Combines scores using weighted average (30% embedding similarity, 70% LLM score)
