"""
SQL Generation Workflow - Main LangGraph orchestration
Implements the complete workflow for SQL generation with memory integration
"""

import asyncio
import logging
from typing import Dict, List, Optional, Literal
from datetime import datetime

try:
    from langgraph.graph import StateGraph, START, END
    from langgraph.types import Command
except ImportError:
    # Fallback for different LangGraph versions
    from langgraph.graph import StateGraph
    try:
        from langgraph.constants import START, END
    except ImportError:
        START = "__start__"
        END = "__end__"
    
    # Command may not be available in all versions
    try:
        from langgraph.types import Command
    except ImportError:
        Command = None

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
    
    async def _schema_analysis_node(self, state: SQLAgentState) -> SQLAgentState:
        """Performs schema analysis with enhanced data pattern detection."""
        
        try:
            schema_result = await self.agents["schema"].analyze_schema_requirements(
                entities=state.get("entities_extracted", []),
                intent=state.get("query_intent", {}),
                context={
                    "user_id": state["user_id"],
                    "session_id": state["session_id"],
                    "memory_context": state.get("memory_context", {})
                }
            )
            
            # Update memory with schema analysis
            await self.memory_system.update_memory_from_processing(
                session_id=state["session_id"],
                agent_name="schema_analyzer",
                processing_data=schema_result.__dict__,
                success=True
            )
            
            return {
                **state,
                "relevant_tables": schema_result.relevant_tables,
                "table_relationships": schema_result.table_relationships,
                "column_metadata": schema_result.column_metadata,
                "sample_data": schema_result.sample_data,
                "column_analysis": schema_result.column_analysis,
                "data_patterns": schema_result.data_patterns,
                "filtering_suggestions": schema_result.filtering_suggestions,
                "schema_confidence": schema_result.confidence_score,
                "optimization_suggestions": schema_result.optimization_suggestions,
                "current_agent": "schema_analyzer",
                "completed_agents": [*state.get("completed_agents", []), "schema_analyzer"],
                "processing_stage": "schema_complete"
            }
            
        except Exception as e:
            return {
                **state,
                "error_history": [
                    *state.get("error_history", []),
                    {
                        "agent": "schema_analyzer",
                        "error": str(e),
                        "timestamp": datetime.now().isoformat(),
                        "recovery_strategy": "schema_error"
                    }
                ],
                "current_agent": "error_handler"
            }
    
    async def _sql_generation_node(self, state: SQLAgentState) -> SQLAgentState:
        """Generates SQL with visualization metadata."""
        
        try:
            sql_result = await self.agents["sql_generator"].generate_sql(
                intent=state.get("query_intent", {}),
                schema_context={
                    "relevant_tables": state.get("relevant_tables", []),
                    "table_relationships": state.get("table_relationships", {}),
                    "column_metadata": state.get("column_metadata", {}),
                    "filtering_suggestions": state.get("filtering_suggestions", {})
                },
                entities=state.get("entities_extracted", [])
            )
            
            # Update memory with SQL generation
            await self.memory_system.update_memory_from_processing(
                session_id=state["session_id"],
                agent_name="sql_generator",
                processing_data=sql_result,
                success=True
            )
            
            return {
                **state,
                "generated_sql": sql_result.get("generated_sql"),
                "generation_strategy": sql_result.get("generation_strategy"),
                "sql_confidence": sql_result.get("confidence"),
                "sql_alternatives": sql_result.get("alternatives", []),
                "visualization_metadata": sql_result.get("visualization_metadata", {}),
                "current_agent": "sql_generator",
                "completed_agents": [*state.get("completed_agents", []), "sql_generator"],
                "processing_stage": "sql_complete"
            }
            
        except Exception as e:
            return {
                **state,
                "error_history": [
                    *state.get("error_history", []),
                    {
                        "agent": "sql_generator",
                        "error": str(e),
                        "timestamp": datetime.now().isoformat(),
                        "recovery_strategy": "regenerate"
                    }
                ],
                "current_agent": "error_handler"
            }
    
    async def _validation_node(self, state: SQLAgentState) -> SQLAgentState:
        """Validates SQL query for security and performance."""
        
        try:
            validation_result = await self.agents["validator"].validate_sql(
                sql=state.get("generated_sql", ""),
                context={
                    "query_intent": state.get("query_intent", {}),
                    "entities": state.get("entities_extracted", []),
                    "schema_context": {
                        "relevant_tables": state.get("relevant_tables", []),
                        "table_relationships": state.get("table_relationships", {})
                    }
                }
            )
            
            # Update memory with validation results
            await self.memory_system.update_memory_from_processing(
                session_id=state["session_id"],
                agent_name="validator",
                processing_data=validation_result,
                success=validation_result.get("is_valid", False)
            )
            
            return {
                **state,
                "validation_results": validation_result,
                "is_sql_valid": validation_result.get("is_valid", False),
                "security_passed": validation_result.get("security_passed", False),
                "validation_recommendations": validation_result.get("recommendations", []),
                "current_agent": "validator",
                "completed_agents": [*state.get("completed_agents", []), "validator"],
                "processing_stage": "validation_complete"
            }
            
        except Exception as e:
            return {
                **state,
                "error_history": [
                    *state.get("error_history", []),
                    {
                        "agent": "validator",
                        "error": str(e),
                        "timestamp": datetime.now().isoformat(),
                        "recovery_strategy": "validation_failed"
                    }
                ],
                "current_agent": "error_handler"
            }
    
    async def _execution_node(self, state: SQLAgentState) -> SQLAgentState:
        """Executes validated SQL query."""
        
        try:
            sql = state.get("generated_sql", "")
            
            if not sql:
                raise Exception("No SQL query to execute")
            
            # Execute query using database connector
            query_results = await self.database_connector.execute_query(sql)
            
            return {
                **state,
                "query_results": query_results,
                "execution_status": "success",
                "execution_timestamp": datetime.now().isoformat(),
                "current_agent": "executor",
                "completed_agents": [*state.get("completed_agents", []), "executor"],
                "processing_stage": "execution_complete"
            }
            
        except Exception as e:
            return {
                **state,
                "error_history": [
                    *state.get("error_history", []),
                    {
                        "agent": "executor",
                        "error": str(e),
                        "timestamp": datetime.now().isoformat(),
                        "recovery_strategy": "execution_failed"
                    }
                ],
                "execution_status": "failed",
                "current_agent": "error_handler"
            }
    
    async def _visualization_node(self, state: SQLAgentState) -> SQLAgentState:
        """Creates visualization recommendations and code."""
        
        try:
            viz_result = await self.agents["visualizer"].analyze_and_recommend(
                query_results=state.get("query_results", []),
                query_intent=state.get("query_intent", {}),
                schema_context={
                    "relevant_tables": state.get("relevant_tables", []),
                    "column_metadata": state.get("column_metadata", {}),
                    "data_patterns": state.get("data_patterns", {})
                },
                entities=state.get("entities_extracted", [])
            )
            
            # Update memory with visualization results
            await self.memory_system.update_memory_from_processing(
                session_id=state["session_id"],
                agent_name="visualizer",
                processing_data=viz_result.__dict__,
                success=True
            )
            
            return {
                **state,
                "visualization_recommendations": viz_result.recommended_charts,
                "visualization_insights": viz_result.data_insights,
                "dashboard_layout": viz_result.dashboard_layout,
                "interactive_features": viz_result.interactive_features,
                "visualization_confidence": viz_result.confidence_score,
                "current_agent": "visualizer",
                "completed_agents": [*state.get("completed_agents", []), "visualizer"],
                "processing_stage": "visualization_complete"
            }
            
        except Exception as e:
            return {
                **state,
                "error_history": [
                    *state.get("error_history", []),
                    {
                        "agent": "visualizer",
                        "error": str(e),
                        "timestamp": datetime.now().isoformat(),
                        "recovery_strategy": "visualization_failed"
                    }
                ],
                "current_agent": "error_handler"
            }
    
    async def _error_recovery_node(self, state: SQLAgentState) -> SQLAgentState:
        """Handles errors and provides recovery strategies."""
        
        error_history = state.get("error_history", [])
        
        if not error_history:
            return {**state, "current_agent": "supervisor"}
        
        latest_error = error_history[-1]
        recovery_strategy = latest_error.get("recovery_strategy", "unknown")
        
        # Increment iteration count for retry logic
        iteration_count = state.get("iteration_count", 0) + 1
        
        return {
            **state,
            "iteration_count": iteration_count,
            "recovery_strategy": recovery_strategy,
            "error_context": latest_error,
            "current_agent": "supervisor",
            "processing_stage": "error_recovery"
        }
    
    async def _quality_assessment_node(self, state: SQLAgentState) -> SQLAgentState:
        """Assesses overall quality of results."""
        
        # Calculate overall quality score
        quality_factors = {
            "nlu_confidence": state.get("confidence_scores", {}).get("overall", 0),
            "schema_confidence": state.get("schema_confidence", 0),
            "sql_confidence": state.get("sql_confidence", 0),
            "validation_passed": 1.0 if state.get("is_sql_valid", False) else 0.0,
            "execution_success": 1.0 if state.get("execution_status") == "success" else 0.0,
            "visualization_confidence": state.get("visualization_confidence", 0)
        }
        
        overall_score = sum(quality_factors.values()) / len(quality_factors)
        
        quality_assessment = {
            "overall_score": overall_score,
            "quality_factors": quality_factors,
            "recommendations": [],
            "success": overall_score >= 0.7
        }
        
        # Add quality recommendations
        if quality_factors["nlu_confidence"] < 0.6:
            quality_assessment["recommendations"].append("Consider rephrasing query for better understanding")
        
        if quality_factors["schema_confidence"] < 0.6:
            quality_assessment["recommendations"].append("Query may benefit from more specific table/column references")
        
        if not quality_factors["validation_passed"]:
            quality_assessment["recommendations"].append("SQL validation failed - query needs refinement")
        
        return {
            **state,
            "quality_assessment": quality_assessment,
            "overall_success": quality_assessment["success"],
            "current_agent": "quality_assessor",
            "completed_agents": [*state.get("completed_agents", []), "quality_assessor"],
            "processing_stage": "quality_complete"
        }
    
    async def _supervisor_node(self, state: SQLAgentState) -> SQLAgentState:
        """High-level supervision and coordination."""
        
        completed_agents = state.get("completed_agents", [])
        error_history = state.get("error_history", [])
        iteration_count = state.get("iteration_count", 0)
        
        supervision_decision = {
            "action": "continue_processing",
            "rationale": "Normal processing flow",
            "next_agent": None
        }
        
        # Check for completion
        required_agents = ["nlu_processor", "schema_analyzer", "sql_generator", "validator", "executor"]
        if all(agent in completed_agents for agent in required_agents):
            supervision_decision["action"] = "complete"
            supervision_decision["rationale"] = "All required processing completed successfully"
        
        # Check for excessive errors
        elif len(error_history) >= 3:
            supervision_decision["action"] = "escalate_error"
            supervision_decision["rationale"] = "Too many errors encountered"
        
        # Check for excessive iterations
        elif iteration_count > 5:
            supervision_decision["action"] = "escalate_error"
            supervision_decision["rationale"] = "Maximum iterations exceeded"
        
        return {
            **state,
            "supervision_decision": supervision_decision,
            "current_agent": "supervisor",
            "processing_stage": "supervised"
        }
