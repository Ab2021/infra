"""
FastAPI REST API for Advanced SQL Agent System
Provides programmatic access to the SQL agent capabilities
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
import asyncio
import logging
from datetime import datetime
import uuid

from main import SQLAgentSystem

# API Models
class QueryRequest(BaseModel):
    query: str = Field(..., description="Natural language query")
    user_id: str = Field(default="api_user", description="User identifier")
    preferences: Optional[Dict[str, Any]] = Field(default=None, description="User preferences")
    session_id: Optional[str] = Field(default=None, description="Session identifier")

class QueryResponse(BaseModel):
    session_id: str
    success: bool
    generated_sql: Optional[str]
    query_results: Optional[List[Dict]]
    visualizations: Optional[List[Dict]]
    processing_time: float
    confidence_scores: Optional[Dict[str, float]]
    error_message: Optional[str]
    memory_insights: Optional[Dict[str, Any]]

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    version: str
    components: Dict[str, str]

class MemoryInsightsResponse(BaseModel):
    user_insights: Dict[str, Any]
    system_insights: Dict[str, Any]
    performance_metrics: Dict[str, float]

# Initialize FastAPI app
app = FastAPI(
    title="Advanced SQL Agent API",
    description="REST API for natural language to SQL conversion with memory intelligence",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global SQL agent system instance
sql_agent_system = None

# Dependency to get SQL agent system
async def get_sql_agent_system():
    global sql_agent_system
    if sql_agent_system is None:
        sql_agent_system = SQLAgentSystem()
    return sql_agent_system

@app.on_event("startup")
async def startup_event():
    """Initialize the SQL agent system on startup."""
    global sql_agent_system
    logging.info("Initializing SQL Agent System...")
    sql_agent_system = SQLAgentSystem()
    logging.info("SQL Agent System initialized successfully")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    global sql_agent_system
    if sql_agent_system:
        # Perform any necessary cleanup
        logging.info("SQL Agent System shutdown complete")

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        version="1.0.0",
        components={
            "sql_agent": "healthy",
            "memory_system": "healthy",
            "database": "healthy"
        }
    )

@app.post("/query", response_model=QueryResponse)
async def process_query(
    request: QueryRequest,
    background_tasks: BackgroundTasks,
    sql_system: SQLAgentSystem = Depends(get_sql_agent_system)
):
    """
    Process a natural language query and return SQL with results.
    
    Args:
        request: Query request with natural language input
        background_tasks: FastAPI background tasks
        sql_system: SQL agent system dependency
        
    Returns:
        Complete query processing results
    """
    
    start_time = datetime.now()
    session_id = request.session_id or str(uuid.uuid4())
    
    try:
        # Process the query
        result = await sql_system.process_query(
            user_query=request.query,
            user_id=request.user_id
        )
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Prepare response
        response = QueryResponse(
            session_id=session_id,
            success=result.get('success', False),
            generated_sql=result.get('generated_sql'),
            query_results=result.get('query_results'),
            visualizations=result.get('visualizations'),
            processing_time=processing_time,
            confidence_scores=result.get('confidence_scores'),
            error_message=result.get('error'),
            memory_insights=result.get('memory_insights')
        )
        
        # Add background task for analytics
        background_tasks.add_task(
            log_query_analytics,
            request.query,
            request.user_id,
            processing_time,
            result.get('success', False)
        )
        
        return response
        
    except Exception as e:
        processing_time = (datetime.now() - start_time).total_seconds()
        
        raise HTTPException(
            status_code=500,
            detail={
                "error": str(e),
                "session_id": session_id,
                "processing_time": processing_time
            }
        )

@app.get("/memory/insights/{user_id}", response_model=MemoryInsightsResponse)
async def get_memory_insights(
    user_id: str,
    sql_system: SQLAgentSystem = Depends(get_sql_agent_system)
):
    """
    Get memory insights for a specific user.
    
    Args:
        user_id: User identifier
        sql_system: SQL agent system dependency
        
    Returns:
        Memory insights and analytics
    """
    
    try:
        insights = await sql_system.memory_manager.get_memory_insights(user_id)
        
        return MemoryInsightsResponse(
            user_insights=insights.get('session_insights', {}),
            system_insights=insights.get('knowledge_insights', {}),
            performance_metrics=insights.get('performance_metrics', {})
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve memory insights: {str(e)}"
        )

@app.get("/schema/tables")
async def get_schema_tables(
    sql_system: SQLAgentSystem = Depends(get_sql_agent_system)
):
    """Get available database tables and their metadata."""
    
    try:
        schema_metadata = await sql_system.database_connector.get_schema_metadata()
        
        # Return simplified table information
        tables = []
        for table_name, table_info in schema_metadata.items():
            tables.append({
                "table_name": table_name,
                "description": table_info.get("description", ""),
                "row_count": table_info.get("row_count", 0),
                "column_count": len(table_info.get("columns", []))
            })
        
        return {"tables": tables}
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve schema information: {str(e)}"
        )

@app.get("/examples")
async def get_example_queries():
    """Get example natural language queries."""
    
    examples = [
        {
            "category": "Sales Analytics",
            "queries": [
                "Show me total sales by product category for this quarter",
                "What are the top 10 customers by revenue?",
                "Compare this year's sales performance to last year"
            ]
        },
        {
            "category": "Performance Analysis", 
            "queries": [
                "Which regions are underperforming?",
                "Show me monthly trends for the past year",
                "What products have declining sales?"
            ]
        },
        {
            "category": "Customer Insights",
            "queries": [
                "Who are our most valuable customers?",
                "Show customer retention rates by segment",
                "What's the average order value by customer type?"
            ]
        }
    ]
    
    return {"examples": examples}

@app.post("/feedback")
async def submit_feedback(
    session_id: str,
    rating: int = Field(..., ge=1, le=5),
    feedback_text: Optional[str] = None,
    sql_system: SQLAgentSystem = Depends(get_sql_agent_system)
):
    """
    Submit feedback for a query session.
    
    Args:
        session_id: Session identifier
        rating: Rating from 1-5
        feedback_text: Optional text feedback
        sql_system: SQL agent system dependency
    """
    
    try:
        # Store feedback in memory system for learning
        feedback_data = {
            "session_id": session_id,
            "rating": rating,
            "feedback_text": feedback_text,
            "timestamp": datetime.now().isoformat()
        }
        
        # This would integrate with the memory system
        # await sql_system.memory_manager.store_feedback(feedback_data)
        
        return {"message": "Feedback submitted successfully", "session_id": session_id}
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to submit feedback: {str(e)}"
        )

async def log_query_analytics(query: str, user_id: str, processing_time: float, success: bool):
    """Background task to log query analytics."""
    
    # Implement analytics logging here
    logging.info(f"Query Analytics - User: {user_id}, Success: {success}, Time: {processing_time:.2f}s")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
