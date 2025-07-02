from abc import ABC, abstractmethod
from fastapi import UploadFile

class BaseResumeParser(ABC):
    """Abstract base class for resume parsers"""
    
    @abstractmethod
    async def parse_file(self, file: UploadFile) -> str:
        """Parse the uploaded file and return text content"""
        pass 