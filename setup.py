from setuptools import setup, find_packages

setup(
    name="resume_scorer",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "fastapi",
        "langchain",
        "openai",
        "numpy",
        "langgraph",
        "python-dotenv",
        "python-multipart"
    ],
    extras_require={
        "dev": [
            "pytest",
            "pytest-asyncio",
            "pytest-cov",
            "pytest-mock",
            "pytest-env"
        ]
    }
) 