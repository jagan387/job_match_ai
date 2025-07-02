import pytest
from fastapi import UploadFile
from io import BytesIO

@pytest.fixture
def sample_resume_text():
    return """
JOHN DOE
Software Engineer
john@example.com

EXPERIENCE
Senior Software Engineer | Tech Corp (2020-Present)
- Led development of microservices architecture
- Managed team of 5 engineers
- Tech stack: Python, FastAPI, Docker

Software Engineer | StartupCo (2018-2020)
- Full-stack development with React and Django
- Implemented CI/CD pipelines
- Improved test coverage by 40%

SKILLS
Languages: Python, JavaScript, TypeScript
Frameworks: FastAPI, Django, React
Tools: Docker, Kubernetes, AWS
"""

@pytest.fixture
def sample_job_description():
    return """
Senior Software Engineer
Required Experience: 5+ years
Must Have Skills:
- Python development
- Microservices architecture
- Team leadership experience
- CI/CD implementation

Nice to Have:
- React/Frontend experience
- Cloud platform knowledge
- Kubernetes expertise
"""

@pytest.fixture
def mock_resume_file():
    content = b"Sample resume content"
    return UploadFile(
        filename="resume.pdf",
        file=BytesIO(content)
    )

@pytest.fixture
def mock_job_file():
    content = b"Sample job description content"
    return UploadFile(
        filename="job.pdf",
        file=BytesIO(content)
    )

@pytest.fixture
def sample_evaluation():
    return """
Technical Score: 85
Technical Explanation: Strong Python and microservices experience, demonstrated leadership
Cultural Score: 75
Cultural Explanation: Shows collaboration and team management capabilities
Overall Score: 82
Overall Explanation: Good technical background with relevant leadership experience
""" 