"""
Notebook Editor Agent - AI-powered editing of extremely large Jupyter notebooks

This package provides a complete AI agent that can efficiently process and edit
large Jupyter notebooks using the 7 foundational building blocks of AI agents.
"""

from agent import NotebookEditorAgent
from notebook_utils import (
    EditOperation, 
    EditRequest, 
    NotebookChunk, 
    NotebookChunker, 
    NotebookValidator
)
from intelligence import NotebookIntelligence, NotebookAnalysis, EditSuggestion
from memory import NotebookMemory, EditSession
from tools import NotebookTools

__version__ = "1.0.0"
__author__ = "AI Cookbook"

__all__ = [
    "NotebookEditorAgent",
    "EditOperation",
    "EditRequest", 
    "NotebookChunk",
    "NotebookChunker",
    "NotebookValidator",
    "NotebookIntelligence",
    "NotebookAnalysis",
    "EditSuggestion",
    "NotebookMemory",
    "EditSession",
    "NotebookTools"
]