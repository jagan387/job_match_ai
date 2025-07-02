"""
Shared utilities for the resume scorer package.
"""

import logging
import sys
import warnings
import os
from typing import Optional
from contextlib import contextmanager
from io import StringIO

class WarningFilter:
    def __init__(self):
        self.stderr = StringIO()
        self._real_stderr = sys.stderr

    def __enter__(self):
        sys.stderr = self.stderr
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stderr = self._real_stderr
        # Only print stderr content that we want to keep
        filtered_content = self.stderr.getvalue()
        if filtered_content:
            lines = filtered_content.splitlines()
            for line in lines:
                if not any(warning in line for warning in [
                    "CropBox missing",
                    "LangChainDeprecationWarning",
                    "langchain_community.chat_models",
                    "warn_deprecated"
                ]) and not line.strip().startswith("warn_deprecated"):
                    print(line, file=self._real_stderr)

# Create a global warning filter
warning_filter = WarningFilter()
sys.stderr = warning_filter.stderr

# Also suppress warnings through the warnings module
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

def setup_workflow_logger(log_level: str = "DEBUG", log_file: Optional[str] = None) -> logging.Logger:
    """
    Configure and return a logger for the resume workflow
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional file path to write logs to
    """
    # Create logger
    logger = logging.getLogger("resume_workflow")
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Prevent propagation to root logger to avoid duplicate logs
    logger.propagate = False
    
    # Create formatters
    console_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Remove any existing handlers
    logger.handlers = []
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    return logger

# Create default logger
logger = setup_workflow_logger() 