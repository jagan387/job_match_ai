import pytest
import sys
import logging
import os
from io import StringIO
from services.utils import WarningFilter, setup_workflow_logger

def test_warning_filter():
    # Create a warning filter and a test output buffer
    test_output = StringIO()
    original_stderr = sys.stderr
    
    try:
        # Replace stderr with our test buffer
        sys.stderr = test_output
        
        # Create and use the warning filter
        with WarningFilter():
            # Write directly to sys.stderr (which is our test buffer)
            print("CropBox missing warning", file=sys.stderr)
            print("LangChainDeprecationWarning: some warning", file=sys.stderr)
            print("Important error message", file=sys.stderr)
            print("warn_deprecated: some warning", file=sys.stderr)
            print("langchain_community.chat_models warning", file=sys.stderr)
        
        # Get the filtered output
        filtered_output = test_output.getvalue()
        
        # The important error message should be in the output
        assert "Important error message" in filtered_output
        # The filtered warnings should not be in the output
        assert "CropBox missing" not in filtered_output
        assert "LangChainDeprecationWarning" not in filtered_output
        assert "warn_deprecated" not in filtered_output
        assert "langchain_community.chat_models" not in filtered_output
    
    finally:
        # Restore original stderr
        sys.stderr = original_stderr

def test_setup_workflow_logger():
    # Test with default settings
    logger = setup_workflow_logger()
    assert logger.name == "resume_workflow"
    assert logger.level == logging.DEBUG
    assert len(logger.handlers) == 1
    assert isinstance(logger.handlers[0], logging.StreamHandler)
    assert logger.handlers[0].stream == sys.stdout
    
    # Test with custom log level
    logger = setup_workflow_logger(log_level="INFO")
    assert logger.level == logging.INFO
    
    # Test with log file
    test_log_file = "test.log"
    try:
        logger = setup_workflow_logger(log_file=test_log_file)
        assert len(logger.handlers) == 2
        assert isinstance(logger.handlers[1], logging.FileHandler)
        assert logger.handlers[1].baseFilename == os.path.abspath(test_log_file)
    finally:
        # Clean up test log file
        if os.path.exists(test_log_file):
            os.remove(test_log_file)

def test_logger_formatting():
    # Test console formatter
    logger = setup_workflow_logger()
    console_handler = logger.handlers[0]
    assert "%(asctime)s - %(levelname)s - %(message)s" in console_handler.formatter._fmt
    assert console_handler.formatter.datefmt == "%H:%M:%S"
    
    # Test file formatter
    test_log_file = "test.log"
    try:
        logger = setup_workflow_logger(log_file=test_log_file)
        file_handler = logger.handlers[1]
        assert "%(name)s" in file_handler.formatter._fmt
    finally:
        if os.path.exists(test_log_file):
            os.remove(test_log_file)

def test_logger_propagation():
    logger = setup_workflow_logger()
    assert not logger.propagate  # Should not propagate to avoid duplicate logs 