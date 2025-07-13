"""
SQL Generation Workflow - Main LangGraph orchestration
Implements the complete workflow for SQL generation with memory integration
"""

import asyncio
import logging
from typing import Dict, List, Optional, Literal
from datetime import datetime

from langgraph.graph import StateGraph, START, END
from langgraph.types import Command

from .state_schema import SQLAgentState, RoutingDecision, create_initial_state
from ..agents.nlu_agent import NLUAgent
from ..agents.schema_intelligence_agent import SchemaIntelligenceAgent
from ..agents.sql_generator_agent import SQLGeneratorAgent
from ..agents.validation_security_agent import ValidationSecurityAgent
from ..agents.visualization_agent import VisualizationAgent

class SQLGenerationWorkflow:
    """
    Main workflow orchestrator that manages the complete SQL generation
    process from natural language input to dashboard creation using LangGraph.
    """
    
    def __init__(self, memory_system, database_connector, llm_provider):
        self.memory_system = memory_system
        self.database_connector = database_connector
        self.llm_provider = llm_provider
        
        # Initialize specialized agents
        self.agents = self._initialize_agents()
        
        # Create the LangGraph workflow
        self.workflow = self._create_workflow_graph()
        
        # Configure logging
        self.logger = logging.getLogger(__name__)
        
    def _initialize_agents(self) -> Dict:
        """Initializes all specialized agents with shared dependencies."""
        return {
            "nlu": NLUAgent(self.memory_system, self.llm_provider),
            "schema": SchemaIntelligenceAgent(self.memory_system, self.database_connector),
            "sql_generator": SQLGeneratorAgent(self.memory_system, self.llm_provider),
            "validator": ValidationSecurityAgent(self.memory_system, self.database_connector),
            "visualizer": VisualizationAgent(self.memory_system, self.llm_provider)
        }
    
    def _create_workflow_graph(self) -> StateGraph:
        """
        Creates the main LangGraph workflow with sophisticated routing logic.
        """
        
        # Initialize state graph with our comprehensive schema
        graph = StateGraph(SQLAgentState)
        
        # Add processing nodes for each major workflow step
        graph.add_node("initialize_session", self._initialize_session_node)
        graph.add_node("memory_manager", self._memory_management_node)
        graph.add_node("nlu_processor", self._nlu_processing_node)
        graph.add_node("schema_analyzer", self._schema_analysis_node)
        graph.add_node("sql_generator", self._sql_generation_node)
        graph.add_node("query_validator", self._validation_node)
        graph.add_node("query_executor", self._execution_node)
        graph.add_node("visualizer", self._visualization_node)
        
        # Add coordination and management nodes
        graph.add_node("supervisor", self._supervisor_node)
        graph.add_node("error_handler", self._error_recovery_node)
        graph.add_node("quality_assessor", self._quality_assessment_node)
        
        # Define workflow entry point
        graph.add_edge(START, "initialize_session")
        graph.add_edge("initialize_session", "memory_manager")
        graph.add_edge("memory_manager", "nlu_processor")
        
        # Main processing flow with intelligent routing
        graph.add_conditional_edges(
            "nlu_processor",
            self._route_after_nlu,
            {
                "schema_analysis": "schema_analyzer",
                "direct_sql": "sql_generator",
                "clarification_needed": "error_handler"
            }
        )
        
        graph.add_conditional_edges(
            "schema_analyzer",
            self._route_after_schema,
            {
                "generate_sql": "sql_generator",
                "need_more_context": "error_handler",
                "schema_error": "error_handler"
            }
        )
        
        graph.add_conditional_edges(
            "sql_generator",
            self._route_after_sql_generation,
            {
                "validate": "query_validator",
                "regenerate": "sql_generator",
                "generation_failed": "error_handler"
            }
        )
        
        graph.add_conditional_edges(
            "query_validator",
            self._route_after_validation,
            {
                "execute": "query_executor",
                "fix_and_regenerate": "sql_generator",
                "validation_failed": "error_handler"
            }
        )
        
        graph.add_conditional_edges(
            "query_executor",
            self._route_after_execution,
            {
                "visualize": "visualizer",
                "execution_failed": "error_handler",
                "complete": "quality_assessor"
            }
        )
        
        # Quality assessment and completion paths
        graph.add_conditional_edges(
            "quality_assessor",
            self._route_after_quality_check,
            {
                "complete": END,
                "improve": "supervisor",
                "regenerate": "sql_generator"
            }
        )
        
        # Error recovery and supervision
        graph.add_edge("error_handler", "supervisor")
        graph.add_edge("visualizer", "quality_assessor")
        
        graph.add_conditional_edges(
            "supervisor",
            self._supervisor_routing_logic,
            {
                "continue_processing": "nlu_processor",
                "retry_generation": "sql_generator",
                "escalate_error": "error_handler",
                "complete": END
            }
        )
        
        return graph
    
    async def execute(self, user_query: str, user_id: str = "anonymous") -> Dict:
        """
        Executes the complete SQL generation workflow.
        
        Args:
            user_query: Natural language query from user
            user_id: User identifier for personalization
            
        Returns:
            Complete workflow results
        """
        
        # Create initial state
        initial_state = create_initial_state(user_query, user_id)
        
        try:
            # Execute the workflow graph
            result = await self.workflow.ainvoke(initial_state)
            
            # Finalize memory with results
            await self.memory_system.finalize_session(
                session_id=result["session_id"],
                final_results=result,
                user_feedback=None  # Can be added later
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Workflow execution failed: {str(e)}")
            return {
                "error": str(e),
                "session_id": initial_state["session_id"],
                "success": False
            }
    
    # === Node Implementations ===
    
    async def _initialize_session_node(self, state: SQLAgentState) -> SQLAgentState:
        """Initializes session with proper memory context setup."""
        
        session_context = await self.memory_system.initialize_processing_session(
            user_id=state["user_id"],
            session_id=state["session_id"],
            query=state["user_query"]
        )
        
        return {
            **state,
            "memory_context": session_context,
            "current_agent": "session_initializer",
            "processing_stage": "initialized",
            "timestamp": datetime.now().isoformat()
        }
    
    async def _memory_management_node(self, state: SQLAgentState) -> SQLAgentState:
        """Loads relevant memory context for processing enhancement."""
        
        contextual_memories = await self.memory_system.get_contextual_memories(
            query=state["user_query"],
            user_id=state["user_id"],
            context_type="general"
        )
        
        return {
            **state,
            "similar_past_queries": contextual_memories.get("session_memories", {}).get("similar_queries", []),
            "learned_patterns": contextual_memories.get("knowledge_memories", {}).get("patterns", {}),
            "user_preferences": contextual_memories.get("session_memories", {}).get("preferences", {}),
            "current_agent": "memory_manager"
        }
    
    async def _nlu_processing_node(self, state: SQLAgentState) -> SQLAgentState:
        """Processes natural language understanding with memory enhancement."""
        
        try:
            nlu_result = await self.agents["nlu"].process_query(
                query=state["user_query"],
                context={
                    "user_id": state["user_id"],
                    "session_id": state["session_id"],
                    "memory_context": state.get("memory_context", {}),
                    "conversation_history": state.get("similar_past_queries", [])
                }
            )
            
            # Update memory with processing results
            await self.memory_system.update_memory_from_processing(
                session_id=state["session_id"],
                agent_name="nlu_processor",
                processing_data=nlu_result,
                success=True
            )
            
            return {
                **state,
                **nlu_result,
                "current_agent": "nlu_processor",
                "completed_agents": [*state.get("completed_agents", []), "nlu_processor"],
                "processing_stage": "nlu_complete"
            }
            
        except Exception as e:
            return {
                **state,
                "error_history": [
                    *state.get("error_history", []),
                    {
                        "agent": "nlu_processor",
                        "error": str(e),
                        "timestamp": datetime.now().isoformat(),
                        "recovery_strategy": "clarification_needed"
                    }
                ],
                "current_agent": "error_handler"
            }
    
    # === Routing Logic ===
    
    def _route_after_nlu(self, state: SQLAgentState) -> RoutingDecision:
        """Determines next step after NLU processing."""
        
        confidence_scores = state.get("confidence_scores", {})
        entities = state.get("entities_extracted", [])
        ambiguities = state.get("ambiguities_detected", [])
        
        # Check confidence levels
        overall_confidence = confidence_scores.get("overall", 0)
        intent_clarity = confidence_scores.get("intent_clarity", 0)
        
        if overall_confidence < 0.6 or len(ambiguities) > 2:
            return "clarification_needed"
        
        if entities and intent_clarity > 0.8:
            return "schema_analysis"
        
        # For simple queries with known patterns
        if self._is_simple_pattern_query(state.get("query_intent", {}), entities):
            return "direct_sql"
        
        return "schema_analysis"
    
    def _route_after_schema(self, state: SQLAgentState) -> RoutingDecision:
        """Routes after schema analysis based on results quality."""
        
        relevant_tables = state.get("relevant_tables", [])
        schema_confidence = state.get("schema_confidence", 0)
        
        if not relevant_tables:
            return "need_more_context"
        
        if schema_confidence < 0.4:
            return "need_more_context"
        
        return "generate_sql"
    
    def _route_after_sql_generation(self, state: SQLAgentState) -> RoutingDecision:
        """Routes after SQL generation based on generation success."""
        
        generated_sql = state.get("generated_sql")
        iteration_count = state.get("iteration_count", 0)
        
        if not generated_sql:
            if iteration_count < 3:
                return "regenerate"
            else:
                return "generation_failed"
        
        return "validate"
    
    def _route_after_validation(self, state: SQLAgentState) -> RoutingDecision:
        """Routes after validation based on validation results."""
        
        validation_results = state.get("validation_results", {})
        is_valid = validation_results.get("is_valid", False)
        
        if not is_valid:
            if state.get("iteration_count", 0) < 2:
                return "fix_and_regenerate"
            else:
                return "validation_failed"
        
        return "execute"
    
    def _route_after_execution(self, state: SQLAgentState) -> RoutingDecision:
        """Routes after query execution based on execution results."""
        
        execution_status = state.get("execution_status")
        query_results = state.get("query_results")
        
        if execution_status != "success" or not query_results:
            return "execution_failed"
        
        # Check if visualization is requested
        query_intent = state.get("query_intent", {})
        output_preference = query_intent.get("output_preference", "table")
        
        if output_preference in ["chart", "dashboard"]:
            return "visualize"
        
        return "complete"
    
    def _route_after_quality_check(self, state: SQLAgentState) -> RoutingDecision:
        """Routes after quality assessment."""
        
        quality_assessment = state.get("quality_assessment", {})
        quality_score = quality_assessment.get("overall_score", 0.5)
        
        if quality_score > 0.8:
            return "complete"
        elif quality_score > 0.6:
            return "improve"
        else:
            return "regenerate"
    
    def _supervisor_routing_logic(self, state: SQLAgentState) -> RoutingDecision:
        """High-level supervisor routing for complex coordination."""
        
        error_history = state.get("error_history", [])
        iteration_count = state.get("iteration_count", 0)
        
        if iteration_count > 5:
            return "escalate_error"
        
        if len(error_history) >= 3:
            recent_errors = error_history[-3:]
            if len(set(error["agent"] for error in recent_errors)) == 1:
                return "retry_generation"
            else:
                return "escalate_error"
        
        if self._is_processing_complete(state):
            return "complete"
        
        return "continue_processing"
    
    # === Utility Methods ===
    
    def _is_simple_pattern_query(self, intent: Dict, entities: List[Dict]) -> bool:
        """Checks if query matches simple patterns that can skip schema analysis."""
        
        if not intent or not entities:
            return False
        
        # Simple aggregation queries
        if (intent.get("primary_action") == "aggregate" and 
            len(entities) <= 3 and 
            any(e.get("type") == "metric" for e in entities)):
            return True
        
        return False
    
    def _is_processing_complete(self, state: SQLAgentState) -> bool:
        """Checks if all required processing is complete."""
        
        required_components = ["generated_sql", "query_results"]
        return all(state.get(component) for component in required_components)
    
    # Additional node implementations would go here...
    # (schema_analysis_node, sql_generation_node, etc.)
