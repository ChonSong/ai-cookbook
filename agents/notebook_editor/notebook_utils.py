"""
AI Agent for Editing Extremely Large Jupyter Notebooks

This module provides an AI-powered agent that can efficiently process and edit
large Jupyter notebooks by using chunking strategies, structured outputs,
and the 7 foundational building blocks of AI agents.

Key Features:
- Memory-efficient processing of large notebooks through chunking
- AI-powered understanding of notebook content and structure
- Intelligent editing capabilities with validation
- Maintains notebook integrity and proper formatting
- Supports various editing operations (cell modification, insertion, deletion)
"""

from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
from enum import Enum
import json
import nbformat
from pathlib import Path


class EditOperation(Enum):
    """Types of editing operations supported by the agent."""
    MODIFY_CELL = "modify_cell"
    INSERT_CELL = "insert_cell"
    DELETE_CELL = "delete_cell"
    MOVE_CELL = "move_cell"
    SPLIT_CELL = "split_cell"
    MERGE_CELLS = "merge_cells"
    ADD_MARKDOWN = "add_markdown"
    UPDATE_METADATA = "update_metadata"


@dataclass
class NotebookChunk:
    """Represents a chunk of a notebook for processing."""
    cells: List[Dict[str, Any]]
    start_index: int
    end_index: int
    metadata: Dict[str, Any]
    
    @property
    def size(self) -> int:
        """Returns the number of cells in this chunk."""
        return len(self.cells)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert chunk to dictionary format."""
        return {
            "cells": self.cells,
            "start_index": self.start_index,
            "end_index": self.end_index,
            "metadata": self.metadata
        }


@dataclass
class EditRequest:
    """Represents an edit request for the notebook."""
    operation: EditOperation
    target_index: Optional[int] = None
    content: Optional[str] = None
    cell_type: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    description: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert edit request to dictionary format."""
        return {
            "operation": self.operation.value,
            "target_index": self.target_index,
            "content": self.content,
            "cell_type": self.cell_type,
            "metadata": self.metadata,
            "description": self.description
        }


class NotebookChunker:
    """Handles chunking of large notebooks for efficient processing."""
    
    def __init__(self, max_cells_per_chunk: int = 50, max_content_length: int = 10000):
        self.max_cells_per_chunk = max_cells_per_chunk
        self.max_content_length = max_content_length
    
    def chunk_notebook(self, notebook: nbformat.NotebookNode) -> List[NotebookChunk]:
        """
        Split a notebook into manageable chunks.
        
        Args:
            notebook: The notebook to chunk
            
        Returns:
            List of NotebookChunk objects
        """
        chunks = []
        cells = notebook.cells
        current_chunk_cells = []
        current_chunk_size = 0
        start_index = 0
        
        for i, cell in enumerate(cells):
            cell_content_length = len(str(cell.get('source', '')))
            
            # Check if adding this cell would exceed limits
            if (len(current_chunk_cells) >= self.max_cells_per_chunk or 
                current_chunk_size + cell_content_length > self.max_content_length):
                
                # Create chunk if we have cells
                if current_chunk_cells:
                    chunk = NotebookChunk(
                        cells=current_chunk_cells.copy(),
                        start_index=start_index,
                        end_index=i - 1,
                        metadata={"chunk_size": len(current_chunk_cells)}
                    )
                    chunks.append(chunk)
                    
                    # Reset for next chunk
                    current_chunk_cells = []
                    current_chunk_size = 0
                    start_index = i
            
            # Add cell to current chunk
            current_chunk_cells.append(cell)
            current_chunk_size += cell_content_length
        
        # Add remaining cells as final chunk
        if current_chunk_cells:
            chunk = NotebookChunk(
                cells=current_chunk_cells,
                start_index=start_index,
                end_index=len(cells) - 1,
                metadata={"chunk_size": len(current_chunk_cells)}
            )
            chunks.append(chunk)
        
        return chunks
    
    def estimate_memory_usage(self, notebook: nbformat.NotebookNode) -> Dict[str, Any]:
        """
        Estimate memory usage and processing requirements for a notebook.
        
        Args:
            notebook: The notebook to analyze
            
        Returns:
            Dictionary with memory and processing estimates
        """
        total_cells = len(notebook.cells)
        total_content_length = sum(len(str(cell.get('source', ''))) for cell in notebook.cells)
        
        # Count different cell types
        cell_types = {}
        for cell in notebook.cells:
            cell_type = cell.get('cell_type', 'unknown')
            cell_types[cell_type] = cell_types.get(cell_type, 0) + 1
        
        # Estimate chunks needed
        estimated_chunks = self.chunk_notebook(notebook)
        
        return {
            "total_cells": total_cells,
            "total_content_length": total_content_length,
            "cell_types": cell_types,
            "estimated_chunks": len(estimated_chunks),
            "recommended_chunk_size": min(self.max_cells_per_chunk, max(10, total_cells // 10)),
            "processing_complexity": "high" if total_cells > 200 else "medium" if total_cells > 50 else "low"
        }


class NotebookValidator:
    """Validates notebook structure and content integrity."""
    
    @staticmethod
    def validate_notebook(notebook: nbformat.NotebookNode) -> Dict[str, Any]:
        """
        Validate notebook structure and return validation results.
        
        Args:
            notebook: The notebook to validate
            
        Returns:
            Dictionary with validation results
        """
        issues = []
        warnings = []
        
        # Check basic structure
        if not hasattr(notebook, 'cells'):
            issues.append("Notebook missing 'cells' attribute")
        
        if not hasattr(notebook, 'metadata'):
            issues.append("Notebook missing 'metadata' attribute")
        
        # Validate cells
        for i, cell in enumerate(notebook.cells):
            if 'cell_type' not in cell:
                issues.append(f"Cell {i} missing 'cell_type'")
            
            if 'source' not in cell:
                issues.append(f"Cell {i} missing 'source'")
            
            # Check for extremely large cells
            if 'source' in cell and len(str(cell['source'])) > 100000:
                warnings.append(f"Cell {i} is very large ({len(str(cell['source']))} characters)")
        
        return {
            "is_valid": len(issues) == 0,
            "issues": issues,
            "warnings": warnings,
            "cell_count": len(notebook.cells) if hasattr(notebook, 'cells') else 0
        }
    
    @staticmethod
    def validate_edit_request(edit_request: EditRequest, notebook_size: int) -> Dict[str, Any]:
        """
        Validate an edit request against notebook constraints.
        
        Args:
            edit_request: The edit request to validate
            notebook_size: Current size of the notebook
            
        Returns:
            Dictionary with validation results
        """
        issues = []
        
        # Validate target index
        if edit_request.target_index is not None:
            if edit_request.target_index < 0 or edit_request.target_index >= notebook_size:
                issues.append(f"Target index {edit_request.target_index} out of range (0-{notebook_size-1})")
        
        # Validate operation-specific requirements
        if edit_request.operation == EditOperation.MODIFY_CELL:
            if edit_request.target_index is None:
                issues.append("MODIFY_CELL operation requires target_index")
            if not edit_request.content:
                issues.append("MODIFY_CELL operation requires content")
        
        elif edit_request.operation == EditOperation.INSERT_CELL:
            if not edit_request.content:
                issues.append("INSERT_CELL operation requires content")
            if not edit_request.cell_type:
                issues.append("INSERT_CELL operation requires cell_type")
        
        elif edit_request.operation == EditOperation.DELETE_CELL:
            if edit_request.target_index is None:
                issues.append("DELETE_CELL operation requires target_index")
        
        return {
            "is_valid": len(issues) == 0,
            "issues": issues
        }