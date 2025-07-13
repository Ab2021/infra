"""
Memory Manager - Central coordinator for the three-tier memory system
Orchestrates Working Memory, Session Memory, and Long-term Knowledge Memory
"""

import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime
import json
import uuid

from .working_memory import WorkingMemory
from .session_memory import SessionMemory
from .long_term_memory import LongTermKnowledgeMemory

class MemoryManager:
    """
    Central memory coordinator that orchestrates the three-tier memory architecture.
    Provides unified interface for all memory operations while managing the
    complex interactions between different memory layers.
    """
    
    def __init__(self, config):
        self.config = config
        
        # Initialize memory tiers
        self.working_memory = WorkingMemory()
        self.session_memory = SessionMemory(
            db_path=getattr(config, 'session_db_path', 'data/session_memory.db')
        )
        self.knowledge_memory = LongTermKnowledgeMemory(
            db_path=getattr(config, 'knowledge_db_path', 'data/knowledge_memory.db'),
            vector_path=getattr(config, 'vector_store_path', 'data/vector_store')
        )
        
        # Memory coordination state
        self.active_sessions = {}
        self.memory_locks = {}
        
    async def initialize_processing_session(self, user_id: str, session_id: str, query: str) -> Dict:
        """
        Initializes a new processing session with complete memory context.
        
        Args:
            user_id: User identifier for personalization
            session_id: Unique session identifier
            query: Initial user query
            
        Returns:
            Complete memory context for session initialization
        """
        
        # Initialize working memory for this session
        working_context = await self.working_memory.initialize_session(
            session_id=session_id,
            user_id=user_id,
            initial_query=query
        )
        
        # Load or create session memory
        session_context = await self.session_memory.get_or_create_session(
            user_id=user_id,
            session_id=session_id
        )
        
        # Retrieve relevant long-term knowledge
        knowledge_context = await self.knowledge_memory.get_relevant_context(
            query=query,
            user_id=user_id,
            session_context=session_context
        )
        
        # Register active session
        self.active_sessions[session_id] = {
            "user_id": user_id,
            "start_time": datetime.now(),
            "query": query,
            "context_loaded": True
        }
        
        return {
            "working_context": working_context,
            "session_context": session_context,
            "knowledge_context": knowledge_context,
            "memory_status": "initialized"
        }
    
    async def get_contextual_memories(self, query: str, user_id: str, context_type: str) -> Dict:
        """
        Retrieves relevant memories from all tiers for enhanced processing.
        
        Args:
            query: Current query being processed
            user_id: User identifier
            context_type: Type of context needed (nlu, schema, sql, etc.)
            
        Returns:
            Relevant memories from all memory tiers
        """
        
        # Retrieve from each memory tier concurrently
        working_memories, session_memories, knowledge_memories = await asyncio.gather(
            self.working_memory.get_relevant_context(query, context_type),
            self.session_memory.get_relevant_memories(user_id, query, context_type),
            self.knowledge_memory.get_relevant_patterns(query, context_type)
        )
        
        return {
            "working_memories": working_memories,
            "session_memories": session_memories,
            "knowledge_memories": knowledge_memories,
            "retrieval_timestamp": datetime.now().isoformat()
        }
    
    async def update_memory_from_processing(self, session_id: str, agent_name: str, 
                                          processing_data: Dict, success: bool = True):
        """
        Updates memory tiers with results from agent processing.
        
        Args:
            session_id: Active session identifier
            agent_name: Name of agent providing the update
            processing_data: Data to store in memory
            success: Whether the processing was successful
        """
        
        timestamp = datetime.now().isoformat()
        
        # Always update working memory
        await self.working_memory.update_context(
            agent_name=agent_name,
            update_data={
                **processing_data,
                "timestamp": timestamp,
                "success": success
            }
        )
        
        # Update session memory if this is meaningful for conversation context
        if self._should_update_session_memory(agent_name, processing_data, success):
            session_info = self.active_sessions.get(session_id, {})
            await self.session_memory.add_processing_result(
                user_id=session_info.get("user_id"),
                session_id=session_id,
                agent_name=agent_name,
                processing_data=processing_data,
                success=success
            )
        
        # Update long-term knowledge if this represents a learning opportunity
        if success and self._should_update_knowledge_memory(agent_name, processing_data):
            await self.knowledge_memory.learn_from_interaction(
                agent_name=agent_name,
                interaction_data={
                    **processing_data,
                    "session_context": self.active_sessions.get(session_id, {}),
                    "timestamp": timestamp
                }
            )
    
    async def finalize_session(self, session_id: str, final_results: Dict, user_feedback: Optional[Dict] = None):
        """
        Finalizes a processing session and extracts learning opportunities.
        
        Args:
            session_id: Session to finalize
            final_results: Complete processing results
            user_feedback: Optional user feedback for learning
        """
        
        if session_id not in self.active_sessions:
            return
        
        session_info = self.active_sessions[session_id]
        
        # Create comprehensive session summary
        session_summary = await self._create_session_summary(session_id, final_results, user_feedback)
        
        # Update session memory with complete interaction
        await self.session_memory.finalize_session_interaction(
            session_id=session_id,
            session_summary=session_summary
        )
        
        # Extract and store learning patterns in long-term memory
        if session_summary.get("success", False):
            await self.knowledge_memory.extract_learning_patterns(session_summary)
        
        # Clean up working memory for this session
        await self.working_memory.cleanup_session(session_id)
        
        # Remove from active sessions
        del self.active_sessions[session_id]
    
    async def get_memory_insights(self, user_id: str) -> Dict:
        """
        Provides insights about memory state and learning progress.
        
        Args:
            user_id: User to get insights for
            
        Returns:
            Memory insights and statistics
        """
        
        # Get insights from each memory tier
        session_insights = await self.session_memory.get_user_insights(user_id)
        knowledge_insights = await self.knowledge_memory.get_learning_insights()
        
        return {
            "session_insights": session_insights,
            "knowledge_insights": knowledge_insights,
            "active_sessions": len(self.active_sessions),
            "memory_health": self._assess_memory_health()
        }
    
    def _should_update_session_memory(self, agent_name: str, data: Dict, success: bool) -> bool:
        """Determines if processing result should be stored in session memory."""
        # Store successful results from key agents
        key_agents = ["nlu_processor", "sql_generator", "visualizer"]
        return success and agent_name in key_agents
    
    def _should_update_knowledge_memory(self, agent_name: str, data: Dict) -> bool:
        """Determines if processing result represents a learning opportunity."""
        # Learn from successful SQL generation and schema analysis
        learning_agents = ["sql_generator", "schema_analyzer"]
        return agent_name in learning_agents and data.get("confidence", 0) > 0.7
    
    async def _create_session_summary(self, session_id: str, results: Dict, feedback: Optional[Dict]) -> Dict:
        """Creates comprehensive summary of session for learning."""
        session_info = self.active_sessions.get(session_id, {})
        working_data = await self.working_memory.get_session_data(session_id)
        
        return {
            "session_id": session_id,
            "user_id": session_info.get("user_id"),
            "original_query": session_info.get("query"),
            "processing_time": (datetime.now() - session_info.get("start_time", datetime.now())).total_seconds(),
            "final_results": results,
            "user_feedback": feedback,
            "working_memory_data": working_data,
            "success": results.get("success", False),
            "timestamp": datetime.now().isoformat()
        }
    
    def _assess_memory_health(self) -> Dict:
        """Assesses overall health and performance of memory system."""
        return {
            "status": "healthy",  # Implement actual health checks
            "working_memory_size": len(self.active_sessions),
            "performance_metrics": {
                "avg_retrieval_time": 0.05,  # Placeholder
                "cache_hit_rate": 0.85  # Placeholder
            }
        }
