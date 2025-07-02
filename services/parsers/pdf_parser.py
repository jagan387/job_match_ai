from fastapi import UploadFile
import tempfile
import os
from langchain.document_loaders import UnstructuredPDFLoader
from .base_parser import BaseResumeParser

class PDFResumeParser(BaseResumeParser):
    async def parse_file(self, file: UploadFile) -> str:
        """Parse PDF file and return text content"""
        content = await file.read()
        
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
            tmp.write(content)
            tmp_path = tmp.name

        try:
            # Use UnstructuredPDFLoader for better text extraction
            loader = UnstructuredPDFLoader(tmp_path)
            documents = loader.load()
            text = ' '.join([doc.page_content for doc in documents])
            return text
        finally:
            # Clean up temporary file
            os.unlink(tmp_path) 