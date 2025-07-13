"""
Working Memory - Real-time processing context
Manages immediate processing state and agent coordination
"""

import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime
import json
from collections import defaultdict

class WorkingMemory:
    """
    Manages immediate processing context and real-time agent coordination.
    Acts as the shared workspace where agents collaborate on current queries.
    """
    
    def __init__(self):
        # Current session data - volatile memory cleared after completion
        self.active_sessions = {}
        
        # Agent coordination state
        self.agent_coordination = {}
        
        # Processing artifacts from current sessions
        self.processing_artifacts = defaultdict(dict)
        
        # Real-time communication logs
        self.agent_communications = defaultdict(list)
        
        # Thread safety for concurrent access
        self._locks = {}
    
    async def initialize_session(self, session_id: str, user_id: str, initial_query: str) -> Dict:
        """
        Initializes working memory for a new processing session.
        
        Args:
            session_id: Unique session identifier
            user_id: User identifier
            initial_query: User's query to process
            
        Returns:
            Initial working memory context
        """
        
        # Create session lock for thread safety
        self._locks[session_id] = asyncio.Lock()
        
        async with self._locks[session_id]:
            # Initialize session context
            self.active_sessions[session_id] = {
                "session_id": session_id,
                "user_id": user_id,
                "initial_query": initial_query,
                "start_time": datetime.now(),
                "processing_stage": "initialized",
                "current_agent": None,
                "completed_agents": [],
                "error_count": 0
            }
            
            # Initialize agent coordination for this session
            self.agent_coordination[session_id] = {
                "current_agent": None,
                "agent_queue": [],
                "handoff_context": {},
                "shared_variables": {},
                "processing_status": "ready"
            }
            
            # Initialize processing artifacts storage
            self.processing_artifacts[session_id] = {
                "parsed_entities": [],
                "schema_candidates": [],
                "query_variations": [],
                "validation_checkpoints": [],
                "performance_metrics": {}
            }
            
            return {
                "session_initialized": True,
                "session_id": session_id,
                "timestamp": datetime.now().isoformat()
            }
    
    async def update_context(self, agent_name: str, update_data: Dict, session_id: str = None):
        """
        Updates working memory with new information from an agent.
        
        Args:
            agent_name: Name of agent providing update
            update_data: Data to store
            session_id: Session to update (if None, updates all active sessions)
        """
        
        timestamp = datetime.now().isoformat()
        
        # Determine which sessions to update
        sessions_to_update = [session_id] if session_id else list(self.active_sessions.keys())
        
        for sid in sessions_to_update:
            if sid not in self.active_sessions:
                continue
                
            async with self._locks.get(sid, asyncio.Lock()):
                # Record agent communication
                self.agent_communications[sid].append({
                    "agent": agent_name,
                    "timestamp": timestamp,
                    "data": update_data,
                    "processing_stage": self.active_sessions[sid]["processing_stage"]
                })
                
                # Update processing artifacts based on data type
                self._update_processing_artifacts(sid, agent_name, update_data)
                
                # Update session state
                self.active_sessions[sid]["current_agent"] = agent_name
                if agent_name not in self.active_sessions[sid]["completed_agents"]:
                    self.active_sessions[sid]["completed_agents"].append(agent_name)
                
                # Check if agent dependencies are satisfied for coordination
                await self._check_agent_dependencies(sid)
    
    async def get_relevant_context(self, query: str, context_type: str) -> Dict:
        """
        Retrieves relevant context from working memory for enhanced processing.
        
        Args:
            query: Current query being processed
            context_type: Type of context needed
            
        Returns:
            Relevant working memory context
        """
        
        relevant_context = {
            "active_processing": [],
            "shared_variables": {},
            "recent_communications": []
        }
        
        # Gather context from all active sessions
        for session_id, session_data in self.active_sessions.items():
            # Include relevant processing artifacts
            artifacts = self.processing_artifacts.get(session_id, {})
            if self._is_context_relevant(artifacts, query, context_type):
                relevant_context["active_processing"].append({
                    "session_id": session_id,
                    "artifacts": artifacts,
                    "stage": session_data["processing_stage"]
                })
            
            # Include shared variables that might be useful
            coordination = self.agent_coordination.get(session_id, {})
            shared_vars = coordination.get("shared_variables", {})
            if shared_vars:
                relevant_context["shared_variables"][session_id] = shared_vars
            
            # Include recent agent communications
            recent_comms = self.agent_communications.get(session_id, [])[-3:]  # Last 3 communications
            if recent_comms:
                relevant_context["recent_communications"].extend(recent_comms)
        
        return relevant_context
    
    async def coordinate_agent_handoff(self, current_agent: str, next_agent: str, 
                                     handoff_data: Dict, session_id: str):
        """
        Manages smooth handoff between agents with context preservation.
        
        Args:
            current_agent: Agent completing processing
            next_agent: Agent to receive control
            handoff_data: Context data for handoff
            session_id: Session for handoff
        """
        
        if session_id not in self.active_sessions:
            return
        
        async with self._locks[session_id]:
            coordination = self.agent_coordination[session_id]
            
            # Record handoff
            coordination["handoff_context"][f"{current_agent}_to_{next_agent}"] = {
                "handoff_data": handoff_data,
                "timestamp": datetime.now().isoformat()
            }
            
            # Update current agent
            coordination["current_agent"] = next_agent
            self.active_sessions[session_id]["current_agent"] = next_agent
            
            # Update processing stage if this represents a major transition
            stage_transitions = {
                "nlu_processor": "entity_extraction",
                "schema_analyzer": "schema_analysis", 
                "sql_generator": "query_generation",
                "validator": "validation",
                "visualizer": "visualization"
            }
            
            if next_agent in stage_transitions:
                self.active_sessions[session_id]["processing_stage"] = stage_transitions[next_agent]
    
    def _update_processing_artifacts(self, session_id: str, agent_name: str, update_data: Dict):
        """Updates processing artifacts based on agent and data type."""
        
        artifacts = self.processing_artifacts[session_id]
        
        # Agent-specific artifact updates
        if agent_name == "nlu_processor":
            if "entities" in update_data:
                artifacts["parsed_entities"].extend(update_data["entities"])
        
        elif agent_name == "schema_analyzer":
            if "schema_info" in update_data:
                artifacts["schema_candidates"].append(update_data["schema_info"])
        
        elif agent_name == "sql_generator":
            if "generated_sql" in update_data:
                artifacts["query_variations"].append(update_data["generated_sql"])
        
        elif agent_name == "validator":
            if "validation_result" in update_data:
                artifacts["validation_checkpoints"].append(update_data["validation_result"])
        
        # Performance metrics from any agent
        if "performance_metrics" in update_data:
            artifacts["performance_metrics"][agent_name] = update_data["performance_metrics"]
    
    async def _check_agent_dependencies(self, session_id: str):
        """
        Checks if waiting agents have their dependencies satisfied.
        Enables parallel processing where possible.
        """
        
        coordination = self.agent_coordination[session_id]
        artifacts = self.processing_artifacts[session_id]
        
        # Define agent dependencies
        dependencies = {
            "schema_analyzer": ["parsed_entities"],
            "sql_generator": ["schema_candidates"],
            "validator": ["query_variations"],
            "visualizer": ["validation_checkpoints"]
        }
        
        # Check each agent's dependencies
        for agent, required_artifacts in dependencies.items():
            if agent not in coordination.get("agent_queue", []):
                continue
                
            # Check if all required artifacts are available
            dependencies_satisfied = all(
                artifacts.get(artifact) for artifact in required_artifacts
            )
            
            if dependencies_satisfied:
                # Remove from queue and mark as ready
                coordination["agent_queue"].remove(agent)
                coordination["processing_status"] = f"{agent}_ready"
    
    def _is_context_relevant(self, artifacts: Dict, query: str, context_type: str) -> bool:
        """Determines if processing artifacts are relevant to current context."""
        
        # Simple relevance checking - can be enhanced with ML
        if context_type == "nlu" and artifacts.get("parsed_entities"):
            return True
        elif context_type == "schema" and artifacts.get("schema_candidates"):
            return True
        elif context_type == "sql" and artifacts.get("query_variations"):
            return True
        
        return False
    
    async def get_session_data(self, session_id: str) -> Dict:
        """Retrieves complete session data for memory operations."""
        
        if session_id not in self.active_sessions:
            return {}
        
        return {
            "session_info": self.active_sessions[session_id],
            "coordination": self.agent_coordination.get(session_id, {}),
            "artifacts": self.processing_artifacts.get(session_id, {}),
            "communications": self.agent_communications.get(session_id, [])
        }
    
    async def cleanup_session(self, session_id: str):
        """Cleans up working memory after session completion."""
        
        # Remove all session data
        self.active_sessions.pop(session_id, None)
        self.agent_coordination.pop(session_id, None)
        self.processing_artifacts.pop(session_id, None)
        self.agent_communications.pop(session_id, None)
        
        # Remove session lock
        self._locks.pop(session_id, None)
