"""
Memory Component for Notebook Editing Agent

This module implements the "Memory" building block - context persistence
across interactions for maintaining state during notebook editing sessions.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import json
from pathlib import Path

from notebook_utils import EditRequest, NotebookChunk


@dataclass
class EditSession:
    """Represents an editing session with history and context."""
    session_id: str
    notebook_path: str
    start_time: datetime
    edit_history: List[Dict[str, Any]] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)
    user_goals: List[str] = field(default_factory=list)
    
    def add_edit(self, edit_request: EditRequest, result: Dict[str, Any]):
        """Add an edit operation to the session history."""
        self.edit_history.append({
            "timestamp": datetime.now().isoformat(),
            "edit_request": edit_request.to_dict(),
            "result": result
        })
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert session to dictionary format."""
        return {
            "session_id": self.session_id,
            "notebook_path": self.notebook_path,
            "start_time": self.start_time.isoformat(),
            "edit_history": self.edit_history,
            "context": self.context,
            "user_goals": self.user_goals
        }


class NotebookMemory:
    """
    Memory system for maintaining context across notebook editing operations.
    
    This implements the "Memory" building block from the 7 foundational
    building blocks of AI agents.
    """
    
    def __init__(self, storage_path: Optional[str] = None):
        self.storage_path = Path(storage_path) if storage_path else Path("./notebook_memory")
        self.storage_path.mkdir(exist_ok=True)
        
        # In-memory state
        self.current_session: Optional[EditSession] = None
        self.chunk_cache: Dict[str, NotebookChunk] = {}
        self.analysis_cache: Dict[str, Dict[str, Any]] = {}
        self.conversation_history: List[Dict[str, Any]] = []
    
    def start_session(self, notebook_path: str, user_goals: List[str] = None) -> str:
        """
        Start a new editing session.
        
        Args:
            notebook_path: Path to the notebook being edited
            user_goals: List of user's stated goals for this session
            
        Returns:
            Session ID
        """
        session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.current_session = EditSession(
            session_id=session_id,
            notebook_path=notebook_path,
            start_time=datetime.now(),
            user_goals=user_goals or []
        )
        
        # Clear caches for new session
        self.chunk_cache.clear()
        self.analysis_cache.clear()
        self.conversation_history.clear()
        
        # Save session
        self._save_session()
        
        return session_id
    
    def end_session(self) -> Optional[Dict[str, Any]]:
        """
        End the current editing session and return session summary.
        
        Returns:
            Session summary dictionary
        """
        if not self.current_session:
            return None
        
        # Generate session summary
        summary = {
            "session_id": self.current_session.session_id,
            "duration": (datetime.now() - self.current_session.start_time).total_seconds(),
            "total_edits": len(self.current_session.edit_history),
            "goals_achieved": self._assess_goals_achievement(),
            "final_context": self.current_session.context
        }
        
        # Save final session state
        self._save_session()
        
        # Clear current session
        self.current_session = None
        
        return summary
    
    def add_conversation(self, user_input: str, agent_response: str, context: Dict[str, Any] = None):
        """
        Add a conversation exchange to memory.
        
        Args:
            user_input: User's input/request
            agent_response: Agent's response
            context: Additional context information
        """
        conversation_entry = {
            "timestamp": datetime.now().isoformat(),
            "user_input": user_input,
            "agent_response": agent_response,
            "context": context or {}
        }
        
        self.conversation_history.append(conversation_entry)
        
        # Keep only last 20 exchanges to manage memory
        if len(self.conversation_history) > 20:
            self.conversation_history = self.conversation_history[-20:]
    
    def record_edit(self, edit_request: EditRequest, result: Dict[str, Any]):
        """
        Record an edit operation in the session history.
        
        Args:
            edit_request: The edit request that was executed
            result: Result of the edit operation
        """
        if self.current_session:
            self.current_session.add_edit(edit_request, result)
            self._save_session()
    
    def cache_chunk_analysis(self, chunk_key: str, analysis: Dict[str, Any]):
        """
        Cache analysis results for a notebook chunk.
        
        Args:
            chunk_key: Unique identifier for the chunk
            analysis: Analysis results to cache
        """
        self.analysis_cache[chunk_key] = {
            "analysis": analysis,
            "timestamp": datetime.now().isoformat()
        }
    
    def get_cached_analysis(self, chunk_key: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve cached analysis for a chunk.
        
        Args:
            chunk_key: Unique identifier for the chunk
            
        Returns:
            Cached analysis or None if not found
        """
        cached = self.analysis_cache.get(chunk_key)
        if cached:
            # Check if cache is still fresh (within 1 hour)
            cache_time = datetime.fromisoformat(cached["timestamp"])
            if (datetime.now() - cache_time).total_seconds() < 3600:
                return cached["analysis"]
        return None
    
    def get_context_for_llm(self) -> str:
        """
        Generate context string for LLM calls based on current session state.
        
        Returns:
            Formatted context string
        """
        if not self.current_session:
            return "No active session."
        
        context_parts = []
        
        # Session information
        context_parts.append(f"Session: {self.current_session.session_id}")
        context_parts.append(f"Notebook: {self.current_session.notebook_path}")
        
        # User goals
        if self.current_session.user_goals:
            context_parts.append(f"User Goals: {', '.join(self.current_session.user_goals)}")
        
        # Recent edit history (last 5 edits)
        recent_edits = self.current_session.edit_history[-5:]
        if recent_edits:
            context_parts.append("Recent Edits:")
            for edit in recent_edits:
                op = edit["edit_request"]["operation"]
                desc = edit["edit_request"].get("description", "No description")
                context_parts.append(f"- {op}: {desc}")
        
        # Recent conversation (last 3 exchanges)
        recent_conversation = self.conversation_history[-3:]
        if recent_conversation:
            context_parts.append("Recent Conversation:")
            for conv in recent_conversation:
                context_parts.append(f"User: {conv['user_input'][:100]}...")
                context_parts.append(f"Agent: {conv['agent_response'][:100]}...")
        
        return "\n".join(context_parts)
    
    def update_session_context(self, key: str, value: Any):
        """
        Update session context with new information.
        
        Args:
            key: Context key
            value: Context value
        """
        if self.current_session:
            self.current_session.context[key] = value
            self._save_session()
    
    def get_session_context(self, key: str, default: Any = None) -> Any:
        """
        Get value from session context.
        
        Args:
            key: Context key
            default: Default value if key not found
            
        Returns:
            Context value or default
        """
        if self.current_session:
            return self.current_session.context.get(key, default)
        return default
    
    def _save_session(self):
        """Save current session to persistent storage."""
        if self.current_session:
            session_file = self.storage_path / f"{self.current_session.session_id}.json"
            with open(session_file, 'w') as f:
                json.dump(self.current_session.to_dict(), f, indent=2)
    
    def _assess_goals_achievement(self) -> List[str]:
        """
        Assess which user goals have been achieved based on edit history.
        
        Returns:
            List of achieved goals
        """
        # This is a simplified implementation
        # In practice, you might use LLM to assess goal achievement
        achieved_goals = []
        
        if self.current_session and self.current_session.edit_history:
            # Mark goals as achieved if there were successful edits
            # This is a placeholder - real implementation would be more sophisticated
            achieved_goals = self.current_session.user_goals.copy()
        
        return achieved_goals
    
    def load_session(self, session_id: str) -> bool:
        """
        Load a previous session from storage.
        
        Args:
            session_id: ID of the session to load
            
        Returns:
            True if session was loaded successfully
        """
        session_file = self.storage_path / f"{session_id}.json"
        
        if not session_file.exists():
            return False
        
        try:
            with open(session_file, 'r') as f:
                session_data = json.load(f)
            
            self.current_session = EditSession(
                session_id=session_data["session_id"],
                notebook_path=session_data["notebook_path"],
                start_time=datetime.fromisoformat(session_data["start_time"]),
                edit_history=session_data["edit_history"],
                context=session_data["context"],
                user_goals=session_data["user_goals"]
            )
            
            return True
        except Exception as e:
            print(f"Error loading session {session_id}: {e}")
            return False
    
    def list_sessions(self) -> List[Dict[str, Any]]:
        """
        List all available sessions.
        
        Returns:
            List of session summaries
        """
        sessions = []
        
        for session_file in self.storage_path.glob("session_*.json"):
            try:
                with open(session_file, 'r') as f:
                    session_data = json.load(f)
                
                sessions.append({
                    "session_id": session_data["session_id"],
                    "notebook_path": session_data["notebook_path"],
                    "start_time": session_data["start_time"],
                    "edit_count": len(session_data["edit_history"])
                })
            except Exception as e:
                print(f"Error reading session file {session_file}: {e}")
        
        return sorted(sessions, key=lambda x: x["start_time"], reverse=True)