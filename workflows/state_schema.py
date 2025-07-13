"""
State Schema Definition for SQL Agent System
Defines the comprehensive state structure used throughout the LangGraph workflow
"""

from typing import Dict, List, Optional, Literal, TypedDict, Any
from datetime import datetime

class SQLAgentState(TypedDict):
    """
    Comprehensive state schema that serves as shared memory across all agents.
    This state flows through the entire LangGraph workflow, enabling sophisticated
    agent coordination and context preservation.
    """
    
    # === Core Query Processing ===
    user_query: str
    processed_query: Optional[str]
    query_intent: Optional[Dict[str, Any]]
    entities_extracted: List[Dict[str, str]]
    ambiguities_detected: List[str]
    
    # === Schema and Database Context ===
    relevant_tables: List[Dict[str, Any]]
    table_relationships: Dict[str, List[str]]
    column_metadata: Dict[str, Dict[str, Any]]
    sample_data: Dict[str, List[Dict]]
    schema_confidence: Optional[float]
    
    # === SQL Generation and Validation ===
    generated_sql: Optional[str]
    sql_alternatives: List[str]
    validation_results: Dict[str, Any]
    execution_status: Optional[str]
    query_results: Optional[List[Dict]]
    performance_metrics: Dict[str, float]
    
    # === Visualization and Dashboard ===
    visualization_specs: Optional[Dict[str, Any]]
    chart_recommendations: List[Dict[str, str]]
    dashboard_layout: Optional[Dict[str, Any]]
    generated_charts: List[Dict[str, Any]]
    
    # === Memory and Learning ===
    memory_context: Dict[str, Any]
    similar_past_queries: List[Dict[str, Any]]
    learned_patterns: Dict[str, Any]
    user_preferences: Dict[str, Any]
    
    # === Error Handling and Iteration ===
    error_history: List[Dict[str, str]]
    iteration_count: int
    feedback_messages: List[str]
    success_indicators: Dict[str, bool]
    
    # === Agent Coordination ===
    current_agent: str
    next_agent: Optional[str]
    agent_handoff_context: Dict[str, Any]
    completed_agents: List[str]
    
    # === Session Management ===
    session_id: str
    user_id: str
    timestamp: str
    processing_stage: str
    
    # === Quality and Confidence Scoring ===
    confidence_scores: Dict[str, float]
    quality_assessment: Dict[str, Any]
    optimization_suggestions: List[str]

# Type definitions for specific components
QueryIntent = Dict[str, Any]
EntityExtraction = Dict[str, Any]
TableInfo = Dict[str, Any]
ValidationResult = Dict[str, Any]
VisualizationSpec = Dict[str, Any]

# Routing decision types for type safety
RoutingDecision = Literal[
    "schema_analysis",
    "direct_sql", 
    "clarification_needed",
    "generate_sql",
    "need_more_context",
    "schema_error",
    "validate",
    "regenerate", 
    "generation_failed",
    "execute",
    "fix_and_regenerate",
    "validation_failed",
    "visualize",
    "execution_failed",
    "complete",
    "improve",
    "continue_processing",
    "retry_generation", 
    "escalate_error"
]

# Agent processing results for structured returns
class AgentResult(TypedDict):
    """Standard result structure for agent processing."""
    success: bool
    data: Dict[str, Any]
    confidence: float
    processing_time: float
    next_action: Optional[str]
    error_info: Optional[Dict[str, Any]]

# Memory context structures
class MemoryContext(TypedDict):
    """Structure for memory context integration."""
    working_context: Dict[str, Any]
    session_context: Dict[str, Any]
    knowledge_context: Dict[str, Any]
    retrieval_metadata: Dict[str, Any]

# Error recovery structures
class ErrorInfo(TypedDict):
    """Structure for error information and recovery."""
    error_type: str
    error_message: str
    agent_name: str
    timestamp: str
    recovery_strategy: str
    context: Dict[str, Any]

# Performance tracking structures
class PerformanceMetrics(TypedDict):
    """Structure for performance metrics tracking."""
    processing_time: float
    memory_usage: float
    database_queries: int
    llm_calls: int
    cache_hits: int
    success_rate: float

def create_initial_state(user_query: str, user_id: str = "anonymous") -> SQLAgentState:
    """
    Creates initial state for a new SQL agent processing session.
    
    Args:
        user_query: User's natural language query
        user_id: User identifier for personalization
        
    Returns:
        Initial state dictionary with required fields populated
    """
    
    session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{user_id}"
    
    return SQLAgentState(
        # Core query processing
        user_query=user_query,
        processed_query=None,
        query_intent=None,
        entities_extracted=[],
        ambiguities_detected=[],
        
        # Schema and database context
        relevant_tables=[],
        table_relationships={},
        column_metadata={},
        sample_data={},
        schema_confidence=None,
        
        # SQL generation and validation
        generated_sql=None,
        sql_alternatives=[],
        validation_results={},
        execution_status=None,
        query_results=None,
        performance_metrics={},
        
        # Visualization and dashboard
        visualization_specs=None,
        chart_recommendations=[],
        dashboard_layout=None,
        generated_charts=[],
        
        # Memory and learning
        memory_context={},
        similar_past_queries=[],
        learned_patterns={},
        user_preferences={},
        
        # Error handling and iteration
        error_history=[],
        iteration_count=0,
        feedback_messages=[],
        success_indicators={},
        
        # Agent coordination
        current_agent="session_initializer",
        next_agent=None,
        agent_handoff_context={},
        completed_agents=[],
        
        # Session management
        session_id=session_id,
        user_id=user_id,
        timestamp=datetime.now().isoformat(),
        processing_stage="initialized",
        
        # Quality and confidence scoring
        confidence_scores={},
        quality_assessment={},
        optimization_suggestions=[]
    )

def validate_state_transition(current_state: SQLAgentState, new_state: SQLAgentState) -> bool:
    """
    Validates that a state transition is valid and maintains data integrity.
    
    Args:
        current_state: Current state before transition
        new_state: Proposed new state after transition
        
    Returns:
        True if transition is valid, False otherwise
    """
    
    # Core fields should not be lost
    if not new_state.get("user_query") or not new_state.get("session_id"):
        return False
    
    # Iteration count should only increase
    if new_state.get("iteration_count", 0) < current_state.get("iteration_count", 0):
        return False
    
    # Completed agents list should only grow
    current_agents = set(current_state.get("completed_agents", []))
    new_agents = set(new_state.get("completed_agents", []))
    if not current_agents.issubset(new_agents):
        return False
    
    # Timestamp should be updated
    if new_state.get("timestamp") == current_state.get("timestamp"):
        return False
    
    return True

def extract_processing_summary(state: SQLAgentState) -> Dict[str, Any]:
    """
    Extracts a summary of processing results from the final state.
    
    Args:
        state: Final processing state
        
    Returns:
        Summary of processing results for memory storage
    """
    
    return {
        "session_id": state.get("session_id"),
        "user_query": state.get("user_query"),
        "processing_success": bool(state.get("generated_sql") and state.get("query_results")),
        "agents_involved": state.get("completed_agents", []),
        "processing_time": state.get("performance_metrics", {}).get("total_time", 0),
        "confidence_scores": state.get("confidence_scores", {}),
        "error_count": len(state.get("error_history", [])),
        "iteration_count": state.get("iteration_count", 0),
        "final_sql": state.get("generated_sql"),
        "visualization_created": bool(state.get("visualization_specs")),
        "learning_opportunities": state.get("optimization_suggestions", [])
    }
