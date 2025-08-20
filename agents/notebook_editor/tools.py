"""
Tools Component for Notebook Editing Agent

This module implements the "Tools" building block - concrete operations
for reading, writing, and manipulating Jupyter notebooks.
"""

from typing import List, Dict, Any, Optional, Tuple
import json
import nbformat
from pathlib import Path
import shutil
from datetime import datetime
import hashlib

from .notebook_utils import EditRequest, EditOperation, NotebookChunk


class NotebookTools:
    """
    Tools for notebook file operations and manipulations.
    
    This implements the "Tools" building block from the 7 foundational
    building blocks of AI agents.
    """
    
    def __init__(self, backup_enabled: bool = True):
        self.backup_enabled = backup_enabled
        self.backup_dir = Path("./notebook_backups")
        if self.backup_enabled:
            self.backup_dir.mkdir(exist_ok=True)
    
    def load_notebook(self, notebook_path: str) -> nbformat.NotebookNode:
        """
        Load a Jupyter notebook from file.
        
        Args:
            notebook_path: Path to the notebook file
            
        Returns:
            Loaded notebook object
            
        Raises:
            FileNotFoundError: If notebook file doesn't exist
            nbformat.ValidationError: If notebook format is invalid
        """
        path = Path(notebook_path)
        if not path.exists():
            raise FileNotFoundError(f"Notebook not found: {notebook_path}")
        
        try:
            with open(path, 'r', encoding='utf-8') as f:
                notebook = nbformat.read(f, as_version=4)
            return notebook
        except Exception as e:
            raise nbformat.ValidationError(f"Failed to load notebook: {e}")
    
    def save_notebook(self, notebook: nbformat.NotebookNode, notebook_path: str, 
                     create_backup: bool = True) -> Dict[str, Any]:
        """
        Save a Jupyter notebook to file.
        
        Args:
            notebook: The notebook object to save
            notebook_path: Path where to save the notebook
            create_backup: Whether to create a backup before saving
            
        Returns:
            Dictionary with save operation results
        """
        path = Path(notebook_path)
        
        # Create backup if enabled and file exists
        backup_path = None
        if create_backup and self.backup_enabled and path.exists():
            backup_path = self._create_backup(notebook_path)
        
        try:
            # Validate notebook before saving
            nbformat.validate(notebook)
            
            # Write to temporary file first
            temp_path = path.with_suffix('.tmp')
            with open(temp_path, 'w', encoding='utf-8') as f:
                nbformat.write(notebook, f)
            
            # Atomic move to final location
            shutil.move(str(temp_path), str(path))
            
            return {
                "success": True,
                "notebook_path": str(path),
                "backup_path": str(backup_path) if backup_path else None,
                "cell_count": len(notebook.cells),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            # Clean up temp file if it exists
            if temp_path.exists():
                temp_path.unlink()
            
            return {
                "success": False,
                "error": str(e),
                "notebook_path": str(path),
                "backup_path": str(backup_path) if backup_path else None
            }
    
    def apply_edit(self, notebook: nbformat.NotebookNode, edit_request: EditRequest) -> Dict[str, Any]:
        """
        Apply an edit operation to a notebook.
        
        Args:
            notebook: The notebook to edit
            edit_request: The edit operation to apply
            
        Returns:
            Dictionary with operation results
        """
        try:
            result = {"success": False, "operation": edit_request.operation.value}
            
            if edit_request.operation == EditOperation.MODIFY_CELL:
                result = self._modify_cell(notebook, edit_request)
            
            elif edit_request.operation == EditOperation.INSERT_CELL:
                result = self._insert_cell(notebook, edit_request)
            
            elif edit_request.operation == EditOperation.DELETE_CELL:
                result = self._delete_cell(notebook, edit_request)
            
            elif edit_request.operation == EditOperation.MOVE_CELL:
                result = self._move_cell(notebook, edit_request)
            
            elif edit_request.operation == EditOperation.SPLIT_CELL:
                result = self._split_cell(notebook, edit_request)
            
            elif edit_request.operation == EditOperation.MERGE_CELLS:
                result = self._merge_cells(notebook, edit_request)
            
            elif edit_request.operation == EditOperation.ADD_MARKDOWN:
                result = self._add_markdown(notebook, edit_request)
            
            elif edit_request.operation == EditOperation.UPDATE_METADATA:
                result = self._update_metadata(notebook, edit_request)
            
            else:
                result = {
                    "success": False,
                    "error": f"Unsupported operation: {edit_request.operation}",
                    "operation": edit_request.operation.value
                }
            
            return result
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "operation": edit_request.operation.value
            }
    
    def extract_code_cells(self, notebook: nbformat.NotebookNode) -> List[Dict[str, Any]]:
        """
        Extract all code cells from a notebook.
        
        Args:
            notebook: The notebook to extract from
            
        Returns:
            List of code cell information
        """
        code_cells = []
        
        for i, cell in enumerate(notebook.cells):
            if cell.cell_type == 'code':
                code_cells.append({
                    "index": i,
                    "source": cell.source,
                    "execution_count": cell.get('execution_count'),
                    "outputs": cell.get('outputs', []),
                    "metadata": cell.get('metadata', {})
                })
        
        return code_cells
    
    def extract_markdown_cells(self, notebook: nbformat.NotebookNode) -> List[Dict[str, Any]]:
        """
        Extract all markdown cells from a notebook.
        
        Args:
            notebook: The notebook to extract from
            
        Returns:
            List of markdown cell information
        """
        markdown_cells = []
        
        for i, cell in enumerate(notebook.cells):
            if cell.cell_type == 'markdown':
                markdown_cells.append({
                    "index": i,
                    "source": cell.source,
                    "metadata": cell.get('metadata', {})
                })
        
        return markdown_cells
    
    def get_notebook_statistics(self, notebook: nbformat.NotebookNode) -> Dict[str, Any]:
        """
        Generate statistics about a notebook.
        
        Args:
            notebook: The notebook to analyze
            
        Returns:
            Dictionary with notebook statistics
        """
        stats = {
            "total_cells": len(notebook.cells),
            "cell_types": {},
            "total_lines": 0,
            "total_characters": 0,
            "has_outputs": False,
            "execution_counts": []
        }
        
        for cell in notebook.cells:
            cell_type = cell.cell_type
            stats["cell_types"][cell_type] = stats["cell_types"].get(cell_type, 0) + 1
            
            # Count lines and characters
            source = cell.get('source', '')
            if isinstance(source, list):
                source = ''.join(source)
            
            stats["total_lines"] += source.count('\n') + 1 if source else 0
            stats["total_characters"] += len(source)
            
            # Check for outputs
            if cell.get('outputs'):
                stats["has_outputs"] = True
            
            # Collect execution counts
            if cell.get('execution_count'):
                stats["execution_counts"].append(cell['execution_count'])
        
        return stats
    
    def _create_backup(self, notebook_path: str) -> Path:
        """Create a backup of the notebook file."""
        path = Path(notebook_path)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_name = f"{path.stem}_{timestamp}.backup{path.suffix}"
        backup_path = self.backup_dir / backup_name
        
        shutil.copy2(path, backup_path)
        return backup_path
    
    def _modify_cell(self, notebook: nbformat.NotebookNode, edit_request: EditRequest) -> Dict[str, Any]:
        """Modify an existing cell."""
        if edit_request.target_index is None or edit_request.target_index >= len(notebook.cells):
            return {"success": False, "error": "Invalid target index"}
        
        cell = notebook.cells[edit_request.target_index]
        old_content = cell.source
        
        cell.source = edit_request.content or ""
        
        # Update metadata if provided
        if edit_request.metadata:
            cell.metadata.update(edit_request.metadata)
        
        return {
            "success": True,
            "operation": "modify_cell",
            "target_index": edit_request.target_index,
            "old_content": old_content,
            "new_content": cell.source
        }
    
    def _insert_cell(self, notebook: nbformat.NotebookNode, edit_request: EditRequest) -> Dict[str, Any]:
        """Insert a new cell."""
        cell_type = edit_request.cell_type or 'code'
        content = edit_request.content or ""
        
        # Create new cell
        if cell_type == 'code':
            new_cell = nbformat.v4.new_code_cell(content)
        elif cell_type == 'markdown':
            new_cell = nbformat.v4.new_markdown_cell(content)
        else:
            new_cell = nbformat.v4.new_raw_cell(content)
        
        # Add metadata if provided
        if edit_request.metadata:
            new_cell.metadata.update(edit_request.metadata)
        
        # Insert at specified position or at end
        insert_index = edit_request.target_index if edit_request.target_index is not None else len(notebook.cells)
        notebook.cells.insert(insert_index, new_cell)
        
        return {
            "success": True,
            "operation": "insert_cell",
            "insert_index": insert_index,
            "cell_type": cell_type,
            "content": content
        }
    
    def _delete_cell(self, notebook: nbformat.NotebookNode, edit_request: EditRequest) -> Dict[str, Any]:
        """Delete a cell."""
        if edit_request.target_index is None or edit_request.target_index >= len(notebook.cells):
            return {"success": False, "error": "Invalid target index"}
        
        deleted_cell = notebook.cells.pop(edit_request.target_index)
        
        return {
            "success": True,
            "operation": "delete_cell",
            "deleted_index": edit_request.target_index,
            "deleted_content": deleted_cell.source,
            "deleted_type": deleted_cell.cell_type
        }
    
    def _move_cell(self, notebook: nbformat.NotebookNode, edit_request: EditRequest) -> Dict[str, Any]:
        """Move a cell to a different position."""
        # This would need additional parameters for destination index
        # For now, return not implemented
        return {
            "success": False,
            "error": "Move cell operation needs additional parameters (destination index)"
        }
    
    def _split_cell(self, notebook: nbformat.NotebookNode, edit_request: EditRequest) -> Dict[str, Any]:
        """Split a cell into multiple cells."""
        if edit_request.target_index is None or edit_request.target_index >= len(notebook.cells):
            return {"success": False, "error": "Invalid target index"}
        
        # This would need additional parameters for split position
        # For now, return not implemented
        return {
            "success": False,
            "error": "Split cell operation needs additional parameters (split positions)"
        }
    
    def _merge_cells(self, notebook: nbformat.NotebookNode, edit_request: EditRequest) -> Dict[str, Any]:
        """Merge multiple cells."""
        # This would need additional parameters for which cells to merge
        # For now, return not implemented
        return {
            "success": False,
            "error": "Merge cells operation needs additional parameters (cell indices to merge)"
        }
    
    def _add_markdown(self, notebook: nbformat.NotebookNode, edit_request: EditRequest) -> Dict[str, Any]:
        """Add a markdown cell with documentation."""
        content = edit_request.content or "# Documentation\n\nAdd your documentation here."
        
        new_cell = nbformat.v4.new_markdown_cell(content)
        
        if edit_request.metadata:
            new_cell.metadata.update(edit_request.metadata)
        
        insert_index = edit_request.target_index if edit_request.target_index is not None else len(notebook.cells)
        notebook.cells.insert(insert_index, new_cell)
        
        return {
            "success": True,
            "operation": "add_markdown",
            "insert_index": insert_index,
            "content": content
        }
    
    def _update_metadata(self, notebook: nbformat.NotebookNode, edit_request: EditRequest) -> Dict[str, Any]:
        """Update notebook or cell metadata."""
        if edit_request.target_index is not None:
            # Update cell metadata
            if edit_request.target_index >= len(notebook.cells):
                return {"success": False, "error": "Invalid target index"}
            
            cell = notebook.cells[edit_request.target_index]
            old_metadata = cell.metadata.copy()
            
            if edit_request.metadata:
                cell.metadata.update(edit_request.metadata)
            
            return {
                "success": True,
                "operation": "update_metadata",
                "target": "cell",
                "target_index": edit_request.target_index,
                "old_metadata": old_metadata,
                "new_metadata": cell.metadata
            }
        else:
            # Update notebook metadata
            old_metadata = notebook.metadata.copy()
            
            if edit_request.metadata:
                notebook.metadata.update(edit_request.metadata)
            
            return {
                "success": True,
                "operation": "update_metadata",
                "target": "notebook",
                "old_metadata": old_metadata,
                "new_metadata": notebook.metadata
            }