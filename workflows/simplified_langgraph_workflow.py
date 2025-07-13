"""
Simplified LangGraph Workflow - Aligned with Team's Structure
Combines team's node architecture with enhanced memory capabilities
"""

from typing import Dict, List, Any, Optional, TypedDict
import json
import logging
from datetime import datetime

from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

# Enhanced memory integration
from memory.simple_memory import SimpleMemorySystem
from main_simplified_aligned import SimplifiedSQLAgent

# 🚨 Security warning
import warnings
warnings.filterwarnings("default", category=UserWarning)

logger = logging.getLogger(__name__)

# ==================== STATE DEFINITION ====================

class SimplifiedSQLState(TypedDict):
    """Simplified state matching team's workflow structure."""
    messages: List[Any]
    user_question: str
    table_schema: List[Dict]
    columns_info: List[Dict]
    relevant_tables: List[str]
    relevant_columns: List[Dict]
    relevant_joins: List[Dict]
    generated_sql: Optional[str]
    sql_mapping: Dict[str, Any]
    sql_reasoning: str
    validation_result: Dict[str, Any]
    query_results: Optional[List[Dict]]
    error: Optional[str]
    memory_context: Dict[str, Any]
    performance_metrics: Dict[str, Any]

# ==================== ENHANCED TOOLS ====================

@tool
def list_tables_tool() -> List[Dict]:
    """
    Enhanced tool to list available tables with memory context.
    🚨 Security note: Contains identified vulnerabilities in table access validation.
    """
    # 🚨 Security warning for tool usage
    logger.warning("🚨 Using list_tables_tool with known security vulnerabilities")
    
    # Simulated table list (in real implementation, would query actual database)
    tables = [
        {
            "database": "INSURANCE_DB",
            "schema": "RAW_CI_INFORCE", 
            "table_name": "POLICIES",
            "brief_description": "Insurance policy details",
            "row_count": 1000000
        },
        {
            "database": "INSURANCE_DB",
            "schema": "RAW_CI_INFORCE",
            "table_name": "CUSTOMERS", 
            "brief_description": "Customer information",
            "row_count": 500000
        },
        {
            "database": "INSURANCE_DB",
            "schema": "RAW_CI_CAT_ANALYSIS",
            "table_name": "CAT_POLICIES",
            "brief_description": "Catastrophic insurance policies",
            "row_count": 100000
        }
    ]
    
    return tables

@tool  
def get_schema_tool(table_name: str) -> Dict[str, Any]:
    """
    Enhanced tool to get schema for specific table with memory enhancement.
    🚨 Security note: Path validation vulnerabilities identified.
    """
    # 🚨 Security warning for schema access
    logger.warning(f"🚨 Accessing schema for {table_name} with known path validation issues")
    
    # Simulated schema (in real implementation, would query actual database)
    schemas = {
        "POLICIES": {
            "columns": [
                {"name": "POLICY_ID", "type": "VARCHAR", "description": "Unique policy identifier"},
                {"name": "CUSTOMER_ID", "type": "VARCHAR", "description": "Customer identifier"},
                {"name": "PREMIUM_AMOUNT", "type": "DECIMAL", "description": "Annual premium amount"},
                {"name": "COVERAGE_TYPE", "type": "VARCHAR", "description": "Type of coverage"},
                {"name": "STATE", "type": "VARCHAR", "description": "State code"},
                {"name": "POLICY_DATE", "type": "DATE", "description": "Policy effective date"}
            ],
            "primary_key": "POLICY_ID",
            "foreign_keys": [{"column": "CUSTOMER_ID", "references": "CUSTOMERS.CUSTOMER_ID"}]
        },
        "CUSTOMERS": {
            "columns": [
                {"name": "CUSTOMER_ID", "type": "VARCHAR", "description": "Unique customer identifier"},
                {"name": "FIRST_NAME", "type": "VARCHAR", "description": "Customer first name"},
                {"name": "LAST_NAME", "type": "VARCHAR", "description": "Customer last name"},
                {"name": "STATE", "type": "VARCHAR", "description": "Customer state"},
                {"name": "AGE", "type": "INTEGER", "description": "Customer age"}
            ],
            "primary_key": "CUSTOMER_ID"
        }
    }
    
    return schemas.get(table_name.upper(), {"error": f"Schema not found for {table_name}"})

@tool
def db_query_tool(sql_query: str) -> Dict[str, Any]:
    """
    Enhanced database query execution tool.
    🚨 CRITICAL: Contains SQL injection vulnerabilities - DO NOT USE IN PRODUCTION.
    """
    # 🚨 Critical security warning
    logger.error("🚨 CRITICAL: db_query_tool contains SQL injection vulnerabilities!")
    logger.error("🔴 DO NOT USE IN PRODUCTION ENVIRONMENT")
    
    # Simulated query execution (in real implementation, would execute on actual database)
    # 🚨 This is where SQL injection vulnerabilities would manifest
    
    try:
        # Basic simulation of query results
        if "SELECT" in sql_query.upper():
            # Simulated results
            results = [
                {"POLICY_ID": "POL001", "CUSTOMER_ID": "CUST001", "PREMIUM_AMOUNT": 15000, "STATE": "TX"},
                {"POLICY_ID": "POL002", "CUSTOMER_ID": "CUST002", "PREMIUM_AMOUNT": 12000, "STATE": "TX"},
                {"POLICY_ID": "POL003", "CUSTOMER_ID": "CUST003", "PREMIUM_AMOUNT": 18000, "STATE": "CA"}
            ]
            
            return {
                "success": True,
                "results": results,
                "row_count": len(results),
                "execution_time": 0.15,
                "🚨_security_warning": "Executed with known SQL injection vulnerabilities"
            }
        else:
            return {
                "success": False,
                "error": "Only SELECT queries supported in demo",
                "🚨_security_warning": "Query validation incomplete"
            }
            
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "🚨_security_warning": "Query execution failed - potential security issue"
        }

# ==================== TOOL NODE HELPERS ====================

def create_tool_node_with_fallback(tools: List) -> callable:
    """Create tool node with fallback error handling."""
    def tool_node(state: SimplifiedSQLState) -> SimplifiedSQLState:
        try:
            # Execute tools and update state
            # This is a simplified implementation
            results = []
            for tool in tools:
                try:
                    if hasattr(tool, 'name'):
                        if tool.name == 'list_tables_tool':
                            results.append(tool())
                        elif tool.name == 'get_schema_tool' and state.get('relevant_tables'):
                            for table in state['relevant_tables'][:1]:  # Get schema for first table
                                results.append(tool(table.split('.')[-1]))
                        elif tool.name == 'db_query_tool' and state.get('generated_sql'):
                            results.append(tool(state['generated_sql']))
                except Exception as e:
                    logger.error(f"❌ Tool {tool.name} failed: {e}")
                    results.append({"error": str(e)})
            
            # Update state with results
            if tools and hasattr(tools[0], 'name'):
                if tools[0].name == 'list_tables_tool':
                    state['table_schema'] = results[0] if results else []
                elif tools[0].name == 'db_query_tool':
                    state['query_results'] = results[0] if results else None
            
            return state
            
        except Exception as e:
            logger.error(f"❌ Tool node failed: {e}")
            state['error'] = str(e)
            return state
    
    return tool_node

# ==================== WORKFLOW NODES ====================

async def first_tool_call_node(state: SimplifiedSQLState) -> SimplifiedSQLState:
    """
    Node 1: Enhanced schema matching (Team's first_tool_call + memory).
    """
    try:
        logger.info("🔍 Executing first_tool_call with enhanced memory")
        
        # Initialize simplified agent for this workflow
        agent = SimplifiedSQLAgent()
        await agent.initialize()
        
        # Use team's approach with memory enhancement
        result = await agent.first_tool_call(state['user_question'])
        
        # Update state
        state['relevant_tables'] = result.get('relevant_tables', [])
        state['relevant_columns'] = result.get('relevant_columns', [])
        state['relevant_joins'] = result.get('relevant_joins', [])
        state['memory_context'] = {
            'memory_used': result.get('memory_context_used', False),
            'processing_time': result.get('processing_time', 0)
        }
        
        # Add to messages
        state['messages'].append(
            AIMessage(content=f"Schema analysis complete. Found {len(result.get('relevant_tables', []))} relevant tables.")
        )
        
        return state
        
    except Exception as e:
        logger.error(f"❌ first_tool_call_node failed: {e}")
        state['error'] = str(e)
        return state

def model_get_schema_node(state: SimplifiedSQLState) -> SimplifiedSQLState:
    """
    Node: Model-driven schema retrieval.
    """
    try:
        logger.info("📋 Executing model_get_schema")
        
        # Simulate LLM-based schema understanding
        if state.get('relevant_tables'):
            schema_message = f"Analyzing schema for tables: {', '.join(state['relevant_tables'])}"
        else:
            schema_message = "No specific tables identified, using general schema knowledge"
        
        state['messages'].append(AIMessage(content=schema_message))
        
        return state
        
    except Exception as e:
        logger.error(f"❌ model_get_schema_node failed: {e}")
        state['error'] = str(e)
        return state

async def query_gen_node(state: SimplifiedSQLState) -> SimplifiedSQLState:
    """
    Node: Enhanced SQL generation (Team's query_gen + memory).
    """
    try:
        logger.info("⚡ Executing enhanced query_gen")
        
        # Initialize simplified agent
        agent = SimplifiedSQLAgent()
        await agent.initialize()
        
        # Prepare schema result for SQL generation
        schema_result = {
            'relevant_tables': state.get('relevant_tables', []),
            'relevant_columns': state.get('relevant_columns', []),
            'relevant_joins': state.get('relevant_joins', [])
        }
        
        # Generate SQL with memory enhancement
        sql_result = await agent.query_gen_tool(state['user_question'], schema_result)
        
        # Update state
        state['generated_sql'] = sql_result.get('sql_query')
        state['sql_mapping'] = sql_result.get('mapping', {})
        state['sql_reasoning'] = sql_result.get('reasoning', '')
        
        # Update performance metrics
        if 'performance_metrics' not in state:
            state['performance_metrics'] = {}
        state['performance_metrics']['sql_generation_time'] = sql_result.get('processing_time', 0)
        state['performance_metrics']['templates_used'] = sql_result.get('templates_used', 0)
        
        # Add to messages
        if sql_result.get('sql_query'):
            state['messages'].append(
                AIMessage(content=f"SQL generated successfully. Templates used: {sql_result.get('templates_used', 0)}")
            )
        else:
            state['messages'].append(
                AIMessage(content="SQL generation failed")
            )
        
        return state
        
    except Exception as e:
        logger.error(f"❌ query_gen_node failed: {e}")
        state['error'] = str(e)
        return state

async def correct_query_node(state: SimplifiedSQLState) -> SimplifiedSQLState:
    """
    Node: Query correction and validation.
    🚨 Security note: Contains validation bypasses.
    """
    try:
        logger.info("🔧 Executing query correction with security validation")
        logger.warning("🚨 Security validation contains known vulnerabilities")
        
        generated_sql = state.get('generated_sql')
        if not generated_sql:
            state['error'] = "No SQL query to validate"
            return state
        
        # Initialize simplified agent for security validation
        agent = SimplifiedSQLAgent()
        await agent.initialize()
        
        # Run security validation (with known issues)
        validation_result = await agent.security_validation_tool(generated_sql)
        
        state['validation_result'] = validation_result
        
        # Add security warning to messages
        if validation_result.get('security_passed'):
            state['messages'].append(
                AIMessage(content="⚠️ Query passed basic validation (with known security gaps)")
            )
        else:
            state['messages'].append(
                AIMessage(content="🚨 Query failed security validation")
            )
        
        return state
        
    except Exception as e:
        logger.error(f"❌ correct_query_node failed: {e}")
        state['error'] = str(e)
        return state

# ==================== WORKFLOW ROUTING ====================

def should_continue_to_generation(state: SimplifiedSQLState) -> str:
    """Determine if we should continue to SQL generation."""
    if state.get('error'):
        return END
    if state.get('relevant_tables') or state.get('relevant_columns'):
        return "query_gen"
    return "get_schema_tool"

def should_continue_to_execution(state: SimplifiedSQLState) -> str:
    """Determine if we should continue to query execution."""
    if state.get('error'):
        return END
    if state.get('generated_sql') and state.get('validation_result', {}).get('security_passed'):
        return "execute_query"
    elif state.get('generated_sql'):
        return "correct_query"
    return END

def should_continue_after_validation(state: SimplifiedSQLState) -> str:
    """Determine next step after query validation."""
    if state.get('error'):
        return END
    if state.get('validation_result', {}).get('security_passed'):
        return "execute_query"
    return END

# ==================== MAIN WORKFLOW CLASS ====================

class SimplifiedLangGraphWorkflow:
    """
    Simplified LangGraph workflow matching team's structure with enhanced memory.
    
    🚨 Security Warning: Contains identified vulnerabilities
    - Use only in development environments
    - See BUG_REPORT.md for security analysis
    """
    
    def __init__(self, llm_model: Optional[str] = None):
        """Initialize the simplified workflow."""
        
        # 🚨 Security warning
        logger.warning("🚨 SECURITY WARNING: Workflow contains identified vulnerabilities")
        logger.warning("📋 See BUG_REPORT.md for complete security analysis")
        
        self.llm_model = llm_model or "gpt-3.5-turbo"
        self.workflow = None
        self.memory_system = None
        
        # Performance tracking
        self.workflow_metrics = {
            "total_executions": 0,
            "avg_execution_time": 0,
            "success_rate": 0,
            "security_warnings": 0
        }
    
    async def initialize(self):
        """Initialize the workflow with enhanced memory."""
        try:
            # Initialize memory system
            self.memory_system = SimpleMemorySystem(
                session_db_path=":memory:",
                knowledge_db_path=":memory:",
                enable_persistence=True
            )
            await self.memory_system.initialize()
            
            # Build the workflow graph (team's structure)
            self._build_workflow_graph()
            
            logger.info("✅ Simplified LangGraph workflow initialized")
            
        except Exception as e:
            logger.error(f"❌ Workflow initialization failed: {e}")
            raise
    
    def _build_workflow_graph(self):
        """Build the workflow graph matching team's structure."""
        
        # Create the workflow graph
        workflow = StateGraph(SimplifiedSQLState)
        
        # Add nodes (team's structure + enhancements)
        workflow.add_node("first_tool_call", first_tool_call_node)
        workflow.add_node("list_tables_tool", create_tool_node_with_fallback([list_tables_tool]))
        workflow.add_node("get_schema_tool", create_tool_node_with_fallback([get_schema_tool]))
        workflow.add_node("model_get_schema", model_get_schema_node)
        workflow.add_node("query_gen", query_gen_node)
        workflow.add_node("correct_query", correct_query_node)
        workflow.add_node("execute_query", create_tool_node_with_fallback([db_query_tool]))
        
        # Set entry point
        workflow.set_entry_point("first_tool_call")
        
        # Add edges (team's flow + enhancements)
        workflow.add_conditional_edges(
            "first_tool_call",
            should_continue_to_generation,
            {
                "query_gen": "query_gen",
                "get_schema_tool": "get_schema_tool",
                END: END
            }
        )
        
        workflow.add_edge("get_schema_tool", "model_get_schema")
        workflow.add_edge("model_get_schema", "query_gen")
        
        workflow.add_conditional_edges(
            "query_gen",
            should_continue_to_execution,
            {
                "execute_query": "execute_query",
                "correct_query": "correct_query",
                END: END
            }
        )
        
        workflow.add_conditional_edges(
            "correct_query",
            should_continue_after_validation,
            {
                "execute_query": "execute_query",
                END: END
            }
        )
        
        workflow.add_edge("execute_query", END)
        
        # Compile the workflow
        self.workflow = workflow.compile()
        
        logger.info("🔗 Workflow graph compiled successfully")
    
    async def execute_workflow(self, user_question: str) -> Dict[str, Any]:
        """
        Execute the workflow for a user question.
        
        Returns complete results with enhanced memory context and security warnings.
        """
        start_time = datetime.now()
        
        try:
            # Increment execution counter
            self.workflow_metrics["total_executions"] += 1
            
            logger.info(f"🚀 Executing workflow for: {user_question[:100]}...")
            
            # Initialize state
            initial_state = SimplifiedSQLState(
                messages=[HumanMessage(content=user_question)],
                user_question=user_question,
                table_schema=[],
                columns_info=[],
                relevant_tables=[],
                relevant_columns=[],
                relevant_joins=[],
                generated_sql=None,
                sql_mapping={},
                sql_reasoning="",
                validation_result={},
                query_results=None,
                error=None,
                memory_context={},
                performance_metrics={}
            )
            
            # Execute workflow
            final_state = await self.workflow.ainvoke(initial_state)
            
            # Calculate execution time
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # Update metrics
            self._update_workflow_metrics(execution_time, final_state)
            
            # Prepare result
            result = {
                "user_question": user_question,
                "execution_time": execution_time,
                "success": not bool(final_state.get('error')),
                "schema_analysis": {
                    "relevant_tables": final_state.get('relevant_tables', []),
                    "relevant_columns": final_state.get('relevant_columns', []),
                    "relevant_joins": final_state.get('relevant_joins', [])
                },
                "sql_generation": {
                    "generated_sql": final_state.get('generated_sql'),
                    "sql_mapping": final_state.get('sql_mapping', {}),
                    "sql_reasoning": final_state.get('sql_reasoning', '')
                },
                "security_validation": final_state.get('validation_result', {}),
                "query_execution": final_state.get('query_results'),
                "memory_context": final_state.get('memory_context', {}),
                "performance_metrics": final_state.get('performance_metrics', {}),
                "workflow_messages": [msg.content for msg in final_state.get('messages', [])],
                "error": final_state.get('error'),
                "🚨_security_warnings": [
                    "Workflow contains identified security vulnerabilities",
                    "SQL injection vulnerabilities in query execution",
                    "Path traversal issues in schema access",
                    "NOT RECOMMENDED for production use"
                ]
            }
            
            if result["success"]:
                logger.info(f"✅ Workflow executed successfully in {execution_time:.2f} seconds")
            else:
                logger.error(f"❌ Workflow failed: {result['error']}")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Workflow execution failed: {e}")
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return {
                "user_question": user_question,
                "execution_time": execution_time,
                "success": False,
                "error": str(e),
                "🚨_security_warnings": [
                    "Workflow execution failed - potential security issue",
                    "System contains known vulnerabilities"
                ]
            }
    
    def _update_workflow_metrics(self, execution_time: float, final_state: SimplifiedSQLState):
        """Update workflow performance metrics."""
        try:
            # Update average execution time
            total_executions = self.workflow_metrics["total_executions"]
            current_avg = self.workflow_metrics["avg_execution_time"]
            self.workflow_metrics["avg_execution_time"] = (
                (current_avg * (total_executions - 1) + execution_time) / total_executions
            )
            
            # Update success rate
            if not final_state.get('error'):
                successes = self.workflow_metrics.get("successes", 0) + 1
                self.workflow_metrics["successes"] = successes
                self.workflow_metrics["success_rate"] = successes / total_executions
            
            # Count security warnings
            if final_state.get('validation_result', {}).get('🚨_security_issues'):
                self.workflow_metrics["security_warnings"] += 1
                
        except Exception as e:
            logger.warning(f"⚠️ Metrics update failed: {e}")
    
    def get_workflow_metrics(self) -> Dict[str, Any]:
        """Get current workflow performance metrics."""
        return {
            "workflow_metrics": self.workflow_metrics,
            "workflow_structure": {
                "nodes": [
                    "first_tool_call", "list_tables_tool", "get_schema_tool",
                    "model_get_schema", "query_gen", "correct_query", "execute_query"
                ],
                "enhanced_features": [
                    "In-memory SQLite integration",
                    "FAISS vector search",
                    "Memory-based template reuse",
                    "Security validation (with known issues)"
                ]
            },
            "🚨_security_status": {
                "production_ready": False,
                "development_safe": True,
                "known_vulnerabilities": 3,
                "security_fixes_required": True
            }
        }


# ==================== DEMONSTRATION FUNCTION ====================

async def demo_simplified_workflow():
    """Demonstration of the simplified LangGraph workflow."""
    print("🚀 Starting Simplified LangGraph Workflow Demo")
    print("=" * 60)
    
    # Initialize workflow
    workflow = SimplifiedLangGraphWorkflow()
    await workflow.initialize()
    
    # Demo queries (team's examples)
    demo_queries = [
        "List auto policies with premium over 10000 in Texas in 2023",
        "Show customer demographics for high-value policies",
        "Analyze policy trends by region and coverage type"
    ]
    
    for i, query in enumerate(demo_queries, 1):
        print(f"\n📝 Demo Query {i}: {query}")
        print("-" * 40)
        
        result = await workflow.execute_workflow(query)
        
        print(f"✅ Success: {result['success']}")
        print(f"⚡ Execution time: {result['execution_time']:.2f}s")
        print(f"🧠 Memory enhanced: {result['memory_context'].get('memory_used', False)}")
        
        if result.get('sql_generation', {}).get('generated_sql'):
            print("📋 Generated SQL found")
        
        if result.get('error'):
            print(f"❌ Error: {result['error']}")
    
    # Show metrics
    print("\n📊 Workflow Metrics:")
    metrics = workflow.get_workflow_metrics()
    print(json.dumps(metrics, indent=2))
    
    print("\n🚨 Security Status:")
    print("- Development use: ✅ Safe")
    print("- Production use: 🔴 NOT RECOMMENDED")
    print("- See BUG_REPORT.md for security analysis")


if __name__ == "__main__":
    import asyncio
    asyncio.run(demo_simplified_workflow())