"""
AI Intelligence Component for Notebook Editing

This module implements the "Intelligence" building block - the core AI component
that uses LLMs to understand notebook content and generate edit suggestions.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import json
from openai import OpenAI
from pydantic import BaseModel, Field

from notebook_utils import NotebookChunk, EditRequest, EditOperation


class NotebookAnalysis(BaseModel):
    """Structured output for notebook analysis."""
    summary: str = Field(description="Brief summary of the notebook content")
    main_topics: List[str] = Field(description="Main topics covered in the notebook")
    cell_types_distribution: Dict[str, int] = Field(description="Distribution of cell types")
    complexity_level: str = Field(description="Overall complexity level: low, medium, high")
    suggestions: List[str] = Field(description="Suggestions for improvement")
    potential_issues: List[str] = Field(description="Potential issues or problems identified")


class EditSuggestion(BaseModel):
    """Structured output for edit suggestions."""
    operation: str = Field(description="Type of edit operation")
    target_index: Optional[int] = Field(description="Index of target cell (if applicable)")
    reasoning: str = Field(description="Explanation for why this edit is suggested")
    content: Optional[str] = Field(description="New content for the cell (if applicable)")
    cell_type: Optional[str] = Field(description="Type of cell (code, markdown, raw)")
    priority: str = Field(description="Priority level: low, medium, high")


class NotebookIntelligence:
    """
    Core AI intelligence component for understanding and editing notebooks.
    
    This implements the "Intelligence" building block from the 7 foundational
    building blocks of AI agents.
    """
    
    def __init__(self, client: Optional[OpenAI] = None, model: str = "gpt-4o"):
        self.client = client or OpenAI()
        self.model = model
    
    def analyze_chunk(self, chunk: NotebookChunk) -> NotebookAnalysis:
        """
        Analyze a notebook chunk to understand its content and structure.
        
        Args:
            chunk: The notebook chunk to analyze
            
        Returns:
            NotebookAnalysis with structured insights
        """
        # Prepare chunk content for analysis
        chunk_content = self._format_chunk_for_analysis(chunk)
        
        system_prompt = """
        You are an expert data scientist and notebook analyst. Analyze the provided 
        Jupyter notebook chunk and provide insights about its content, structure, 
        and potential improvements.
        
        Focus on:
        - Understanding the main purpose and topics
        - Identifying the types of cells and their distribution
        - Assessing code quality and documentation
        - Suggesting improvements for clarity and maintainability
        - Identifying potential issues or problems
        """
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Analyze this notebook chunk:\n\n{chunk_content}"}
        ]
        
        response = self.client.beta.chat.completions.parse(
            model=self.model,
            messages=messages,
            response_format=NotebookAnalysis,
        )
        
        return response.choices[0].message.parsed
    
    def suggest_edits(self, chunk: NotebookChunk, user_request: str) -> List[EditSuggestion]:
        """
        Generate edit suggestions for a notebook chunk based on user request.
        
        Args:
            chunk: The notebook chunk to edit
            user_request: Description of what the user wants to achieve
            
        Returns:
            List of EditSuggestion objects
        """
        chunk_content = self._format_chunk_for_analysis(chunk)
        
        system_prompt = """
        You are an expert Jupyter notebook editor. Based on the user's request,
        suggest specific edits to improve the notebook. Each suggestion should
        include the operation type, target location, reasoning, and content.
        
        Available operations:
        - modify_cell: Change existing cell content
        - insert_cell: Add new cell at specific position
        - delete_cell: Remove a cell
        - add_markdown: Add documentation/explanation
        - move_cell: Reorder cells
        - split_cell: Split large cell into smaller ones
        - merge_cells: Combine related cells
        
        Always provide clear reasoning and consider notebook best practices.
        """
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"""
            User request: {user_request}
            
            Notebook chunk to edit:
            {chunk_content}
            
            Please suggest specific edits to fulfill the user's request.
            """}
        ]
        
        # Use function calling to get structured suggestions
        tools = [{
            "type": "function",
            "function": {
                "name": "suggest_edit",
                "description": "Suggest a specific edit to the notebook",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "operation": {
                            "type": "string",
                            "enum": ["modify_cell", "insert_cell", "delete_cell", "add_markdown", "move_cell", "split_cell", "merge_cells"]
                        },
                        "target_index": {
                            "type": "integer",
                            "description": "Index of target cell (0-based, relative to chunk)"
                        },
                        "reasoning": {
                            "type": "string",
                            "description": "Explanation for this edit"
                        },
                        "content": {
                            "type": "string",
                            "description": "New content for the cell"
                        },
                        "cell_type": {
                            "type": "string",
                            "enum": ["code", "markdown", "raw"]
                        },
                        "priority": {
                            "type": "string",
                            "enum": ["low", "medium", "high"]
                        }
                    },
                    "required": ["operation", "reasoning", "priority"]
                }
            }
        }]
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=tools,
            tool_choice="auto"
        )
        
        suggestions = []
        if response.choices[0].message.tool_calls:
            for tool_call in response.choices[0].message.tool_calls:
                if tool_call.function.name == "suggest_edit":
                    args = json.loads(tool_call.function.arguments)
                    suggestion = EditSuggestion(**args)
                    suggestions.append(suggestion)
        
        return suggestions
    
    def explain_code(self, code_content: str, context: str = "") -> str:
        """
        Generate explanation for code content.
        
        Args:
            code_content: The code to explain
            context: Additional context about the notebook
            
        Returns:
            Explanation of the code
        """
        system_prompt = """
        You are an expert programmer and data scientist. Explain the provided code 
        in a clear, educational manner. Focus on what the code does, why it's 
        written this way, and any important concepts it demonstrates.
        """
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"""
            Context: {context}
            
            Code to explain:
            ```python
            {code_content}
            ```
            
            Please provide a clear explanation of this code.
            """}
        ]
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.3
        )
        
        return response.choices[0].message.content
    
    def generate_documentation(self, chunk: NotebookChunk) -> str:
        """
        Generate documentation for a notebook chunk.
        
        Args:
            chunk: The notebook chunk to document
            
        Returns:
            Generated documentation
        """
        chunk_content = self._format_chunk_for_analysis(chunk)
        
        system_prompt = """
        You are a technical documentation expert. Generate clear, comprehensive 
        documentation for the provided notebook chunk. Include:
        - Overview of what this section does
        - Key concepts and methods used
        - Expected inputs and outputs
        - Any important assumptions or limitations
        """
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Generate documentation for this notebook chunk:\n\n{chunk_content}"}
        ]
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.3
        )
        
        return response.choices[0].message.content
    
    def _format_chunk_for_analysis(self, chunk: NotebookChunk) -> str:
        """
        Format a notebook chunk for LLM analysis.
        
        Args:
            chunk: The notebook chunk to format
            
        Returns:
            Formatted string representation
        """
        formatted_cells = []
        
        for i, cell in enumerate(chunk.cells):
            cell_type = cell.get('cell_type', 'unknown')
            source = cell.get('source', '')
            
            # Convert source to string if it's a list
            if isinstance(source, list):
                source = ''.join(source)
            
            formatted_cell = f"""
Cell {chunk.start_index + i} ({cell_type}):
{source}
"""
            formatted_cells.append(formatted_cell)
        
        return f"""
Notebook Chunk (cells {chunk.start_index}-{chunk.end_index}):
{'=' * 50}
{''.join(formatted_cells)}
"""