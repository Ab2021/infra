"""
Simplified Main Application
Direct 3-agent orchestration for NLU-based SQL dashboard generation
"""

import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Core imports
from config.settings import Settings, get_settings
from memory.simple_memory import SimpleMemorySystem
from agents.query_understanding_agent import QueryUnderstandingAgent
from agents.data_profiling_agent import DataProfilingAgent
from agents.sql_visualization_agent import SQLVisualizationAgent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/simple_system.log', mode='a') if Path('logs').exists() else logging.NullHandler()
    ]
)

logger = logging.getLogger(__name__)

class SimpleSQLAgentSystem:
    """
    Simplified SQL Agent System with 3-step processing:
    1. Query Understanding Agent (NLU + Schema Intelligence)
    2. Data Profiling Agent (Data analysis + Conditions)
    3. SQL Visualization Agent (SQL generation + Charts)
    """
    
    def __init__(self, settings: Settings = None):
        """Initialize the simplified system"""
        self.settings = settings or get_settings()
        
        # Core components
        self.memory_system = None
        self.database_connector = None
        self.llm_provider = None
        
        # Agents
        self.query_agent = None
        self.profiling_agent = None
        self.sql_viz_agent = None
        
        # Schema information
        self.schema_info = {}
        
        self.logger = logging.getLogger(__name__)
    
    async def initialize(self):
        """Initialize all system components"""
        try:
            self.logger.info("Initializing Simplified SQL Agent System")
            
            # Validate configuration
            if not self.settings.validate_configuration():
                raise ValueError("Invalid configuration")
            
            # Initialize memory system
            self.memory_system = SimpleMemorySystem(
                session_db_path=self.settings.session_db_path,
                knowledge_db_path=self.settings.knowledge_db_path,
                similarity_threshold=self.settings.vector_similarity_threshold
            )
            
            # Initialize database connector
            self.database_connector = await self._initialize_database_connector()
            
            # Initialize LLM provider
            self.llm_provider = await self._initialize_llm_provider()
            
            # Load schema information
            self.schema_info = await self._load_schema_info()
            
            # Initialize agents
            self.query_agent = QueryUnderstandingAgent(
                llm_provider=self.llm_provider,
                memory_system=self.memory_system,
                schema_info=self.schema_info
            )
            
            self.profiling_agent = DataProfilingAgent(
                database_connector=self.database_connector,
                memory_system=self.memory_system
            )
            
            self.sql_viz_agent = SQLVisualizationAgent(
                llm_provider=self.llm_provider,
                memory_system=self.memory_system,
                database_connector=self.database_connector
            )
            
            # Get system stats
            stats = await self.memory_system.get_memory_stats()
            self.logger.info(f"System initialized successfully. Memory stats: {stats}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize system: {e}")
            raise
    
    async def _initialize_database_connector(self):
        """Initialize Snowflake database connector"""
        try:
            from database.snowflake_connector import SnowflakeConnector
            
            connector = SnowflakeConnector(
                account=self.settings.snowflake_account,
                user=self.settings.snowflake_user,
                password=self.settings.snowflake_password,
                warehouse=self.settings.snowflake_warehouse,
                database=self.settings.snowflake_database,
                schema=self.settings.snowflake_schema,
                role=self.settings.snowflake_role
            )
            
            await connector.initialize()
            return connector
            
        except Exception as e:
            self.logger.error(f"Failed to initialize database connector: {e}")
            raise
    
    async def _initialize_llm_provider(self):
        """Initialize LLM provider"""
        try:
            llm_config = self.settings.get_llm_config()
            
            if llm_config["provider"] == "openai":
                from langchain_openai import ChatOpenAI
                return ChatOpenAI(
                    api_key=llm_config["api_key"],
                    model=llm_config["model"],
                    temperature=llm_config["temperature"],
                    max_tokens=llm_config["max_tokens"]
                )
            else:
                raise ValueError(f"Unsupported LLM provider: {llm_config['provider']}")
                
        except Exception as e:
            self.logger.error(f"Failed to initialize LLM provider: {e}")
            raise
    
    async def _load_schema_info(self) -> Dict:
        """Load schema information from database"""
        try:
            # Get basic schema metadata from database
            schema_metadata = await self.database_connector.get_schema_metadata()
            
            # Add descriptions and enrich with domain knowledge
            enriched_schema = await self._enrich_schema_info(schema_metadata)
            
            return enriched_schema
            
        except Exception as e:
            self.logger.warning(f"Failed to load schema info: {e}")
            return {"tables": {}}
    
    async def _enrich_schema_info(self, schema_metadata: Dict) -> Dict:
        """Enrich schema with descriptions and domain knowledge"""
        
        # For now, return the basic schema
        # In production, you would add table/column descriptions here
        enriched = {
            "tables": schema_metadata,
            "descriptions": {},
            "relationships": {}
        }
        
        # Add common table descriptions based on naming patterns
        for table_name, table_info in schema_metadata.items():
            description = ""
            
            if "fact" in table_name.lower():
                description = "Fact table containing transactional/measurable data"
            elif "dim" in table_name.lower():
                description = "Dimension table containing descriptive attributes"
            elif "sales" in table_name.lower():
                description = "Sales transaction data"
            elif "customer" in table_name.lower():
                description = "Customer information and attributes"
            elif "product" in table_name.lower():
                description = "Product catalog and attributes"
            elif "time" in table_name.lower() or "date" in table_name.lower():
                description = "Time/date dimension for temporal analysis"
            
            enriched["descriptions"][table_name] = description
        
        return enriched
    
    async def process_query(self, user_query: str, user_id: str = "anonymous", 
                          session_id: str = None) -> Dict[str, Any]:
        """
        Process user query through the 3-agent pipeline
        
        Args:
            user_query: Natural language query
            user_id: User identifier
            session_id: Session identifier (created if not provided)
            
        Returns:
            Complete processing results with SQL and visualization
        """
        
        start_time = datetime.now()
        
        try:
            self.logger.info(f"Processing query: {user_query}")
            
            # Step 0: Setup session
            if not session_id:
                session_id = await self.memory_system.create_session(user_id)
            
            session_context = await self.memory_system.get_session_context(session_id)
            
            # Step 1: Query Understanding Agent
            self.logger.info("Step 1: Understanding query and identifying tables/columns")
            understanding_result = await self.query_agent.process(
                user_query=user_query,
                session_context=session_context
            )
            
            if understanding_result.get("error"):
                return self._create_error_response(
                    understanding_result["error"], 
                    "Query understanding failed",
                    user_query, session_id
                )
            
            # Step 2: Data Profiling Agent
            self.logger.info("Step 2: Profiling data and determining conditions")
            profiling_result = await self.profiling_agent.process(
                tables=understanding_result.get("identified_tables", []),
                columns=understanding_result.get("required_columns", []),
                intent=understanding_result.get("query_intent", {})
            )
            
            if profiling_result.get("error"):
                return self._create_error_response(
                    profiling_result["error"],
                    "Data profiling failed", 
                    user_query, session_id
                )
            
            # Step 3: SQL Visualization Agent
            self.logger.info("Step 3: Generating SQL and determining visualization")
            sql_viz_result = await self.sql_viz_agent.process(
                query_intent=understanding_result.get("query_intent", {}),
                column_profiles=profiling_result.get("column_profiles", {}),
                suggested_filters=profiling_result.get("suggested_filters", [])
            )
            
            if sql_viz_result.get("error"):
                return self._create_error_response(
                    sql_viz_result["error"],
                    "SQL generation failed",
                    user_query, session_id
                )
            
            # Combine results
            processing_time = (datetime.now() - start_time).total_seconds()
            
            final_result = {
                "success": True,
                "user_query": user_query,
                "session_id": session_id,
                "processing_time": processing_time,
                
                # Agent results
                "query_understanding": understanding_result,
                "data_profiling": profiling_result,
                "sql_visualization": sql_viz_result,
                
                # Key outputs for dashboard
                "sql_query": sql_viz_result.get("sql_query", ""),
                "chart_config": sql_viz_result.get("chart_config", {}),
                "plotting_code": sql_viz_result.get("plotting_code", ""),
                "validation": sql_viz_result.get("validation", {}),
                
                "timestamp": datetime.now().isoformat()
            }
            
            # Store in conversation history
            await self.memory_system.add_to_conversation(
                session_id=session_id,
                query=user_query,
                intent=understanding_result.get("query_intent", {}),
                tables_used=[t.get("name", t) for t in understanding_result.get("identified_tables", [])],
                chart_type=sql_viz_result.get("chart_config", {}).get("chart_type"),
                success=True
            )
            
            self.logger.info(f"Query processed successfully in {processing_time:.2f} seconds")
            
            return final_result
            
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            self.logger.error(f"Query processing failed after {processing_time:.2f}s: {e}")
            
            return self._create_error_response(
                str(e), "System error during processing", 
                user_query, session_id, processing_time
            )
    
    async def execute_sql_and_visualize(self, sql_query: str, chart_config: Dict, 
                                      plotting_code: str) -> Dict:
        """
        Execute SQL query and prepare visualization data
        
        Args:
            sql_query: Generated SQL query
            chart_config: Chart configuration
            plotting_code: Streamlit plotting code
            
        Returns:
            Execution results with data and visualization
        """
        
        try:
            start_time = datetime.now()
            
            # Execute SQL query
            self.logger.info(f"Executing SQL: {sql_query[:100]}...")
            query_results = await self.database_connector.execute_query(sql_query)
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return {
                "success": True,
                "query_results": query_results,
                "result_count": len(query_results),
                "execution_time": execution_time,
                "chart_config": chart_config,
                "plotting_code": plotting_code,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"SQL execution failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "query_results": [],
                "execution_time": 0
            }
    
    def _create_error_response(self, error_message: str, error_type: str, 
                             user_query: str, session_id: str = None, 
                             processing_time: float = 0) -> Dict:
        """Create standardized error response"""
        
        return {
            "success": False,
            "error": error_message,
            "error_type": error_type,
            "user_query": user_query,
            "session_id": session_id,
            "processing_time": processing_time,
            "timestamp": datetime.now().isoformat(),
            "sql_query": "",
            "chart_config": {},
            "plotting_code": ""
        }
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get current system status and statistics"""
        
        try:
            status = {
                "system_status": "running",
                "configuration": {
                    "memory_backend": self.settings.memory_backend,
                    "session_db_path": self.settings.session_db_path,
                    "knowledge_db_path": self.settings.knowledge_db_path,
                    "llm_provider": self.settings.llm_provider,
                    "llm_model": getattr(self.settings, 'openai_model', 'unknown')
                },
                "schema_info": {
                    "total_tables": len(self.schema_info.get("tables", {})),
                    "tables": list(self.schema_info.get("tables", {}).keys())[:10]  # First 10
                }
            }
            
            if self.memory_system:
                status["memory_stats"] = await self.memory_system.get_memory_stats()
            
            return status
            
        except Exception as e:
            self.logger.error(f"Error getting system status: {e}")
            return {"system_status": "error", "error": str(e)}
    
    async def cleanup(self):
        """Cleanup system resources"""
        try:
            self.logger.info("Cleaning up system resources")
            
            # Cleanup old memory data
            if self.memory_system:
                await self.memory_system.cleanup_old_data(days_old=7)
            
            # Close database connections
            if self.database_connector:
                await self.database_connector.close()
            
            self.logger.info("System cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")


async def main():
    """Main application entry point for testing"""
    try:
        # Create necessary directories
        from config.settings import create_directories
        create_directories()
        
        # Initialize settings
        settings = get_settings()
        
        # Create system instance
        system = SimpleSQLAgentSystem(settings)
        
        # Initialize system
        await system.initialize()
        
        print("\\n🤖 Simplified SQL Agent System")
        print("=" * 50)
        print("Commands:")
        print("  'status' - Show system status")
        print("  'exit' - Quit")
        print("  Or enter natural language SQL queries")
        print("=" * 50)
        
        while True:
            try:
                user_input = input("\\n📝 Query: ").strip()
                
                if user_input.lower() in ['exit', 'quit', 'q']:
                    break
                
                if user_input.lower() == 'status':
                    status = await system.get_system_status()
                    print(f"\\n📊 System Status:")
                    print(f"  Status: {status.get('system_status')}")
                    print(f"  Tables: {status.get('schema_info', {}).get('total_tables', 0)}")
                    print(f"  Memory: {status.get('memory_stats', {})}")
                    continue
                
                if not user_input:
                    continue
                
                # Process query
                print("\\n🔄 Processing...")
                result = await system.process_query(user_input)
                
                # Display results
                if result.get('success'):
                    print(f"\\n✅ Success! ({result.get('processing_time', 0):.2f}s)")
                    print(f"\\n📝 Generated SQL:")
                    print(result.get('sql_query', 'No SQL generated'))
                    
                    chart_config = result.get('chart_config', {})
                    if chart_config:
                        print(f"\\n📊 Chart: {chart_config.get('chart_type', 'unknown')}")
                        print(f"   Title: {chart_config.get('title', 'No title')}")
                    
                    validation = result.get('validation', {})
                    if validation.get('guardrails_passed'):
                        print("✅ All guardrails passed")
                    else:
                        print("⚠️  Some guardrails failed")
                        for warning in validation.get('performance_checks', []):
                            print(f"   - {warning}")
                
                else:
                    print(f"\\n❌ Error: {result.get('error', 'Unknown error')}")
                    print(f"   Type: {result.get('error_type', 'Unknown')}")
                
            except KeyboardInterrupt:
                print("\\n\\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\\n❌ Error: {e}")
        
        # Cleanup
        await system.cleanup()
        
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        print(f"\\n💥 Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\\n\\n👋 System shutdown requested")
    except Exception as e:
        print(f"\\n💥 Failed to start system: {e}")
        sys.exit(1)