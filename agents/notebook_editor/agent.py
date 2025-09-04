"""
Main Notebook Editor Agent

This module brings together all the building blocks to create a complete
AI agent for editing extremely large Jupyter notebooks.
"""

from typing import List, Dict, Any, Optional
import os
from pathlib import Path
import instructor
from openai import OpenAI

from .notebook_utils import NotebookChunker, NotebookValidator, EditRequest, EditOperation
from .intelligence import NotebookIntelligence
from .memory import NotebookMemory
from .tools import NotebookTools


class NotebookEditorAgent:
    """
    Complete AI agent for editing extremely large Jupyter notebooks.
    
    This agent combines all 7 foundational building blocks:
    1. Intelligence - AI understanding and generation
    2. Memory - Context persistence across interactions
    3. Tools - Concrete notebook operations
    4. Validation - Ensuring notebook integrity
    5. Control - Decision making and flow control
    6. Recovery - Error handling and rollback
    7. Feedback - Learning from results
    """
    
    def __init__(self, 
                 openai_api_key: Optional[str] = None,
                 model: str = "gpt-4o",
                 max_cells_per_chunk: int = 50,
                 enable_backups: bool = True,
                 memory_storage_path: Optional[str] = None):
        """
        Initialize the notebook editor agent.
        
        Args:
            openai_api_key: OpenAI API key (uses env var if not provided)
            model: OpenAI model to use
            max_cells_per_chunk: Maximum cells per processing chunk
            enable_backups: Whether to create backups before editing
            memory_storage_path: Path for persistent memory storage
        """
        # Initialize OpenAI client
        api_key = openai_api_key or os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable or pass it directly.")
        
        self.client = OpenAI(api_key=api_key)
        instructor.patch(self.client)
        self.model = model
        
        # Initialize building blocks
        self.chunker = NotebookChunker(max_cells_per_chunk=max_cells_per_chunk)
        self.validator = NotebookValidator()
        self.intelligence = NotebookIntelligence(client=self.client, model=model)
        self.memory = NotebookMemory(storage_path=memory_storage_path)
        self.tools = NotebookTools(backup_enabled=enable_backups)
        
        # Agent state
        self.current_notebook = None
        self.current_notebook_path = None
        self.current_chunks = []
    
    def start_editing_session(self, notebook_path: str, user_goals: List[str] = None) -> Dict[str, Any]:
        """
        Start a new notebook editing session.
        
        Args:
            notebook_path: Path to the notebook to edit
            user_goals: List of user's goals for this session
            
        Returns:
            Session initialization results
        """
        try:
            # Load and validate notebook
            notebook = self.tools.load_notebook(notebook_path)
            validation_result = self.validator.validate_notebook(notebook)
            
            if not validation_result["is_valid"]:
                return {
                    "success": False,
                    "error": "Notebook validation failed",
                    "issues": validation_result["issues"]
                }
            
            # Store notebook state
            self.current_notebook = notebook
            self.current_notebook_path = notebook_path
            
            # Create chunks for processing
            self.current_chunks = self.chunker.chunk_notebook(notebook)
            
            # Start memory session
            session_id = self.memory.start_session(notebook_path, user_goals)
            
            # Get notebook statistics
            stats = self.tools.get_notebook_statistics(notebook)
            
            # Store session context
            self.memory.update_session_context("notebook_stats", stats)
            self.memory.update_session_context("chunk_count", len(self.current_chunks))
            
            return {
                "success": True,
                "session_id": session_id,
                "notebook_path": notebook_path,
                "notebook_stats": stats,
                "chunk_count": len(self.current_chunks),
                "validation_warnings": validation_result.get("warnings", [])
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to start session: {str(e)}"
            }
    
    def analyze_notebook(self, detailed: bool = False) -> Dict[str, Any]:
        """
        Analyze the current notebook using AI.
        
        Args:
            detailed: Whether to perform detailed analysis of each chunk
            
        Returns:
            Analysis results
        """
        if not self.current_notebook:
            return {"success": False, "error": "No active notebook session"}
        
        try:
            results = {
                "success": True,
                "overview": {},
                "chunk_analyses": [],
                "recommendations": []
            }
            
            # Basic statistics
            stats = self.tools.get_notebook_statistics(self.current_notebook)
            memory_estimate = self.chunker.estimate_memory_usage(self.current_notebook)
            
            results["overview"] = {
                "statistics": stats,
                "memory_estimate": memory_estimate,
                "processing_strategy": self._determine_processing_strategy(memory_estimate)
            }
            
            if detailed:
                # Analyze each chunk with AI
                for i, chunk in enumerate(self.current_chunks):
                    # Check cache first
                    chunk_key = f"chunk_{i}_{hash(str(chunk.to_dict()))}"
                    cached_analysis = self.memory.get_cached_analysis(chunk_key)
                    
                    if cached_analysis:
                        analysis = cached_analysis
                    else:
                        # Perform AI analysis
                        analysis = self.intelligence.analyze_chunk(chunk)
                        # Cache the result
                        self.memory.cache_chunk_analysis(chunk_key, analysis.model_dump())
                    
                    results["chunk_analyses"].append({
                        "chunk_index": i,
                        "cells_range": f"{chunk.start_index}-{chunk.end_index}",
                        "analysis": analysis if isinstance(analysis, dict) else analysis.model_dump()
                    })
            
            # Generate overall recommendations
            results["recommendations"] = self._generate_overall_recommendations(results)
            
            # Record in memory
            self.memory.add_conversation(
                "analyze_notebook",
                f"Analyzed notebook with {len(self.current_chunks)} chunks",
                {"analysis_type": "detailed" if detailed else "basic"}
            )
            
            return results
            
        except Exception as e:
            # Check for authentication errors specifically
            if "authentication" in str(e).lower():
                print(f"Authentication error: Please check your OPENAI_API_KEY.")
            return {
                "success": False,
                "error": f"Analysis failed: {str(e)}"
            }
    
    def edit_notebook(self, user_request: str, auto_apply: bool = False) -> Dict[str, Any]:
        """
        Generate and optionally apply edits based on user request.
        
        Args:
            user_request: Natural language description of desired edits
            auto_apply: Whether to automatically apply suggested edits
            
        Returns:
            Edit results
        """
        if not self.current_notebook:
            return {"success": False, "error": "No active notebook session"}
        
        try:
            # Add context from memory
            context = self.memory.get_context_for_llm()
            enhanced_request = f"{context}\n\nUser Request: {user_request}"
            
            all_suggestions = []
            
            # Generate edit suggestions for relevant chunks
            relevant_chunks = self._identify_relevant_chunks(user_request)
            
            for chunk_idx in relevant_chunks:
                chunk = self.current_chunks[chunk_idx]
                suggestions = self.intelligence.suggest_edits(chunk, enhanced_request)
                
                for suggestion in suggestions:
                    # Adjust indices to global notebook indices
                    if suggestion.target_index is not None:
                        suggestion.target_index += chunk.start_index
                    
                    all_suggestions.append({
                        "chunk_index": chunk_idx,
                        "suggestion": suggestion.model_dump()
                    })
            
            # Record conversation
            self.memory.add_conversation(
                user_request,
                f"Generated {len(all_suggestions)} edit suggestions",
                {"auto_apply": auto_apply}
            )
            
            results = {
                "success": True,
                "user_request": user_request,
                "suggestions": all_suggestions,
                "applied_edits": []
            }
            
            # Apply edits if requested
            if auto_apply and all_suggestions:
                results["applied_edits"] = self._apply_suggested_edits(all_suggestions)
            
            return results
            
        except Exception as e:
            if "authentication" in str(e).lower():
                print(f"Authentication error: Please check your OPENAI_API_KEY.")
            return {
                "success": False,
                "error": f"Edit generation failed: {str(e)}"
            }
    
    def apply_edit(self, edit_request: EditRequest) -> Dict[str, Any]:
        """
        Apply a specific edit to the notebook.
        
        Args:
            edit_request: The edit to apply
            
        Returns:
            Edit application results
        """
        if not self.current_notebook:
            return {"success": False, "error": "No active notebook session"}
        
        try:
            # Validate edit request
            validation = self.validator.validate_edit_request(
                edit_request, 
                len(self.current_notebook.cells)
            )
            
            if not validation["is_valid"]:
                return {
                    "success": False,
                    "error": "Invalid edit request",
                    "issues": validation["issues"]
                }
            
            # Apply the edit
            result = self.tools.apply_edit(self.current_notebook, edit_request)
            
            if result["success"]:
                # Update chunks if notebook structure changed
                if edit_request.operation in [EditOperation.INSERT_CELL, EditOperation.DELETE_CELL]:
                    self.current_chunks = self.chunker.chunk_notebook(self.current_notebook)
                
                # Record in memory
                self.memory.record_edit(edit_request, result)
                
                # Update session context
                new_stats = self.tools.get_notebook_statistics(self.current_notebook)
                self.memory.update_session_context("notebook_stats", new_stats)
            
            return result
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Edit application failed: {str(e)}"
            }
    
    def save_notebook(self, output_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Save the current notebook state.
        
        Args:
            output_path: Optional different path to save to
            
        Returns:
            Save operation results
        """
        if not self.current_notebook:
            return {"success": False, "error": "No active notebook session"}
        
        save_path = output_path or self.current_notebook_path
        
        try:
            # Final validation before save
            validation = self.validator.validate_notebook(self.current_notebook)
            
            if not validation["is_valid"]:
                return {
                    "success": False,
                    "error": "Cannot save invalid notebook",
                    "issues": validation["issues"]
                }
            
            # Save notebook
            result = self.tools.save_notebook(self.current_notebook, save_path)
            
            if result["success"]:
                # Update session context
                self.memory.update_session_context("last_save", result["timestamp"])
                
                # Record in memory
                self.memory.add_conversation(
                    f"save_notebook to {save_path}",
                    "Notebook saved successfully",
                    result
                )
            
            return result
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Save failed: {str(e)}"
            }
    
    def end_session(self) -> Dict[str, Any]:
        """
        End the current editing session.
        
        Returns:
            Session summary
        """
        session_summary = self.memory.end_session()
        
        # Clear agent state
        self.current_notebook = None
        self.current_notebook_path = None
        self.current_chunks = []
        
        return session_summary or {"message": "No active session to end"}
    
    def get_session_status(self) -> Dict[str, Any]:
        """
        Get current session status and statistics.
        
        Returns:
            Current session information
        """
        if not self.memory.current_session:
            return {"active_session": False}
        
        return {
            "active_session": True,
            "session_id": self.memory.current_session.session_id,
            "notebook_path": self.current_notebook_path,
            "edit_count": len(self.memory.current_session.edit_history),
            "chunk_count": len(self.current_chunks),
            "user_goals": self.memory.current_session.user_goals,
            "context": self.memory.current_session.context
        }
    
    def _determine_processing_strategy(self, memory_estimate: Dict[str, Any]) -> str:
        """Determine the best processing strategy based on notebook size."""
        complexity = memory_estimate["processing_complexity"]
        cell_count = memory_estimate["total_cells"]
        
        if complexity == "high" or cell_count > 200:
            return "chunked_processing_with_caching"
        elif complexity == "medium" or cell_count > 50:
            return "chunked_processing"
        else:
            return "full_notebook_processing"
    
    def _identify_relevant_chunks(self, user_request: str) -> List[int]:
        """
        Identify which chunks are relevant to the user's request.
        
        This is a simplified implementation - in practice, you might use
        embedding similarity or more sophisticated relevance scoring.
        """
        # For now, return all chunks for comprehensive editing
        # In practice, you'd implement smarter chunk selection
        return list(range(len(self.current_chunks)))
    
    def _apply_suggested_edits(self, suggestions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Apply a list of suggested edits to the notebook."""
        applied_edits = []
        
        # Sort suggestions by priority and apply high priority ones first
        high_priority = [s for s in suggestions if s["suggestion"]["priority"] == "high"]
        
        for suggestion_data in high_priority[:5]:  # Limit to 5 auto-applied edits
            suggestion = suggestion_data["suggestion"]
            
            # Convert suggestion to EditRequest
            edit_request = EditRequest(
                operation=EditOperation(suggestion["operation"]),
                target_index=suggestion.get("target_index"),
                content=suggestion.get("content"),
                cell_type=suggestion.get("cell_type"),
                description=suggestion["reasoning"]
            )
            
            # Apply the edit
            result = self.apply_edit(edit_request)
            applied_edits.append({
                "suggestion": suggestion,
                "result": result
            })
        
        return applied_edits
    
    def _generate_overall_recommendations(self, analysis_results: Dict[str, Any]) -> List[str]:
        """Generate high-level recommendations based on analysis."""
        recommendations = []
        
        stats = analysis_results["overview"]["statistics"]
        
        # Basic recommendations based on statistics
        if stats["total_cells"] > 100:
            recommendations.append("Consider splitting this large notebook into smaller, focused notebooks")
        
        if stats["cell_types"].get("code", 0) > stats["cell_types"].get("markdown", 0) * 3:
            recommendations.append("Add more markdown cells for documentation and explanations")
        
        if not stats["has_outputs"]:
            recommendations.append("Run cells to generate outputs for better readability")
        
        return recommendations