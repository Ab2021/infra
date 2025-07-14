"""
SQL Agent System - Main Application
Enterprise-grade multi-agent system for natural language to SQL conversion
with intelligent memory, data profiling, and automated visualization.
"""

import asyncio
import logging
import os
import sys
import pandas as pd
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Core imports
from config.settings import Settings, get_settings
from memory.memory_system import MemorySystem
from agents.query_understanding_agent import QueryUnderstandingAgent
from agents.data_profiling_agent import DataProfilingAgent
from agents.sql_visualization_agent import SQLVisualizationAgent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/system.log', mode='a') if Path('logs').exists() else logging.NullHandler()
    ]
)

logger = logging.getLogger(__name__)

class SQLAgentSystem:
    """
    Production SQL Agent System with 3-agent pipeline:
    1. Query Understanding Agent (NLU + Schema Intelligence)
    2. Data Profiling Agent (Data analysis + Smart filtering)
    3. SQL Visualization Agent (SQL generation + Visualization)
    
    Features:
    - Intelligent memory system with pattern learning
    - Snowflake database integration
    - Column name normalization and validation
    - Template-based SQL generation with guardrails
    - Automated chart recommendations
    """
    
    def __init__(self, settings: Settings = None):
        """Initialize the SQL Agent System"""
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
        self.table_catalog = None
        self.column_catalog = None
        self.all_tables = []
        self.all_columns = []
        
        # Column name mapping for handling spaces/underscores
        self.column_mapping = {}
        self.reverse_mapping = {}
        
        self.logger = logging.getLogger(__name__)
    
    async def initialize(self, schema_file_path: str = None):
        """
        Initialize all system components
        
        Args:
            schema_file_path: Path to Excel schema file (optional)
        """
        try:
            self.logger.info("Initializing SQL Agent System")
            
            # Load schema information first
            if schema_file_path:
                await self._load_schema_from_excel(schema_file_path)
            
            # Validate configuration
            if not self.settings.validate_configuration():
                raise ValueError("Invalid configuration")
            
            # Initialize memory system
            self.memory_system = MemorySystem(
                session_db_path=self.settings.session_db_path,
                knowledge_db_path=self.settings.knowledge_db_path,
                similarity_threshold=self.settings.vector_similarity_threshold
            )
            await self.memory_system.initialize()
            
            # Initialize database connector
            self.database_connector = await self._initialize_database_connector()
            
            # Initialize LLM provider
            self.llm_provider = await self._initialize_llm_provider()
            
            # Load database schema info if not from Excel
            if not schema_file_path:
                self.schema_info = await self._load_database_schema()
            
            # Initialize agents
            await self._initialize_agents()
            
            # Get system stats
            stats = await self.memory_system.get_memory_stats()
            self.logger.info(f"System initialized successfully. Memory stats: {stats}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize system: {e}")
            raise
    
    async def _load_schema_from_excel(self, schema_file_path: str):
        """Load schema information from Excel file"""
        try:
            self.logger.info(f"Loading schema from {schema_file_path}")
            
            # Load table descriptions
            self.table_catalog = pd.read_excel(
                schema_file_path,
                sheet_name='Table_descriptions'
            )[['DATABASE', 'SCHEMA', 'TABLE', 'Brief_Description', 'Detailed_Comments']]
            
            # Load column descriptions
            self.column_catalog = pd.read_excel(
                schema_file_path,
                sheet_name="Table's Column Summaries"
            )[['Table Name', 'Feature Name', 'Data Type', 'Description', 'sample_100_distinct']]
            
            # Convert to list-of-dicts for uniform access
            self.all_tables = [
                {
                    "database": row["DATABASE"],
                    "schema": row["SCHEMA"],
                    "table_name": row["TABLE"],
                    "brief_description": row["Brief_Description"],
                    "detailed_comments": row["Detailed_Comments"]
                }
                for idx, row in self.table_catalog.iterrows()
            ]
            
            self.all_columns = [
                {
                    "table_name": row["Table Name"],
                    "column_name": row["Feature Name"],
                    "data_type": row["Data Type"],
                    "description": row["Description"],
                    "sample_100_distinct": row["sample_100_distinct"]
                }
                for idx, row in self.column_catalog.iterrows()
            ]
            
            # Create column name mappings
            self._create_column_mappings()
            
            # Convert to schema_info format
            self.schema_info = {
                "tables": {
                    f"{table['database']}.{table['schema']}.{table['table_name']}": {
                        "columns": [
                            {
                                "name": col["column_name"],
                                "type": col["data_type"],
                                "description": col["description"]
                            }
                            for col in self.all_columns 
                            if col["table_name"] == table["table_name"]
                        ],
                        "description": table["brief_description"]
                    }
                    for table in self.all_tables
                }
            }
            
            self.logger.info(f"Loaded {len(self.all_tables)} tables and {len(self.all_columns)} columns")
            
        except Exception as e:
            self.logger.error(f"Failed to load schema from Excel: {e}")
            raise
    
    def _create_column_mappings(self):
        """Create mappings between column name variations"""
        self.column_mapping = {}
        self.reverse_mapping = {}
        
        for col in self.all_columns:
            original = str(col["column_name"]).strip()
            normalized = original.replace(' ', '_').replace('-', '_')
            
            # Create bidirectional mappings
            self.column_mapping[normalized.lower()] = original
            self.reverse_mapping[original.lower()] = normalized
            
            # Also map exact matches
            self.column_mapping[original.lower()] = original
            self.reverse_mapping[normalized.lower()] = normalized
    
    def resolve_column_name(self, column_name: str, table_name: str = None) -> str:
        """Resolve column name variations to correct database names"""
        if pd.isna(column_name):
            return None
        
        cleaned = str(column_name).strip()
        
        # Try exact match first
        if table_name:
            for col in self.all_columns:
                if (col["table_name"].lower() == table_name.lower() and 
                    col["column_name"].lower() == cleaned.lower()):
                    return col["column_name"]
        
        # Try mapping lookup
        lower_cleaned = cleaned.lower()
        if lower_cleaned in self.column_mapping:
            return self.column_mapping[lower_cleaned]
        
        # Try variations
        variations = [
            cleaned.replace('_', ' '),
            cleaned.replace(' ', '_'),
            cleaned.replace('-', '_'),
            cleaned.replace('_', '-')
        ]
        
        for variation in variations:
            if variation.lower() in self.column_mapping:
                return self.column_mapping[variation.lower()]
        
        return cleaned
    
    async def _initialize_database_connector(self):
        """Initialize database connector"""
        try:
            # Check if we have Snowflake connection available
            try:
                from src.snowflake_connection import create_snowflake_connection
                
                class SnowflakeWrapper:
                    def __init__(self):
                        self.connection = None
                    
                    async def initialize(self):
                        self.connection = create_snowflake_connection()
                    
                    async def execute_query(self, sql_query: str):
                        if not self.connection:
                            await self.initialize()
                        
                        with self.connection.cursor() as cursor:
                            cursor.execute(sql_query)
                            rows = cursor.fetchall()
                            colnames = [d[0] for d in cursor.description]
                            return [dict(zip(colnames, row)) for row in rows]
                    
                    async def get_schema_metadata(self):
                        # Return schema info from Excel if available
                        if self.schema_info:
                            return {
                                table_name.split('.')[-1]: {
                                    "columns": table_info["columns"]
                                }
                                for table_name, table_info in self.schema_info["tables"].items()
                            }
                        return {}
                    
                    async def close(self):
                        if self.connection:
                            self.connection.close()
                
                wrapper = SnowflakeWrapper()
                await wrapper.initialize()
                return wrapper
                
            except ImportError:
                self.logger.warning("Snowflake connection not available, using mock connector")
                return None
            
        except Exception as e:
            self.logger.error(f"Failed to initialize database connector: {e}")
            return None
    
    async def _initialize_llm_provider(self):
        """Initialize LLM provider"""
        try:
            # Check if GPT API is available
            try:
                from src.gpt_class import GptApi
                return GptApi()
            except ImportError:
                self.logger.warning("GPT API not available, using mock LLM")
                
                class MockLLM:
                    def get_gpt_response_non_streaming(self, payload):
                        class MockResponse:
                            def json(self):
                                return {
                                    'choices': [{
                                        'message': {
                                            'content': '{"relevant_tables": [], "relevant_columns": []}'
                                        }
                                    }]
                                }
                        return MockResponse()
                
                return MockLLM()
                
        except Exception as e:
            self.logger.error(f"Failed to initialize LLM provider: {e}")
            raise
    
    async def _load_database_schema(self) -> Dict:
        """Load schema information from database if not from Excel"""
        try:
            if self.database_connector:
                schema_metadata = await self.database_connector.get_schema_metadata()
                return {"tables": schema_metadata}
            return {"tables": {}}
        except Exception as e:
            self.logger.warning(f"Failed to load database schema: {e}")
            return {"tables": {}}
    
    async def _initialize_agents(self):
        """Initialize all agents"""
        # Agent 1: Query Understanding
        self.query_agent = QueryUnderstandingAgent(
            llm_provider=self.llm_provider,
            memory_system=self.memory_system,
            schema_info=self.schema_info,
            all_tables=self.all_tables,
            all_columns=self.all_columns
        )
        
        # Agent 2: Data Profiling
        self.profiling_agent = DataProfilingAgent(
            database_connector=self.database_connector,
            memory_system=self.memory_system
        )
        
        # Agent 3: SQL Visualization
        self.sql_viz_agent = SQLVisualizationAgent(
            llm_provider=self.llm_provider,
            memory_system=self.memory_system,
            database_connector=self.database_connector
        )
    
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
    
    async def execute_sql_and_get_results(self, sql_query: str, chart_config: Dict = None, 
                                        plotting_code: str = None) -> Dict:
        """
        Execute SQL query and prepare results
        
        Args:
            sql_query: Generated SQL query
            chart_config: Chart configuration (optional)
            plotting_code: Streamlit plotting code (optional)
            
        Returns:
            Execution results with data and metadata
        """
        
        try:
            start_time = datetime.now()
            
            # Execute SQL query
            self.logger.info(f"Executing SQL: {sql_query[:100]}...")
            
            if self.database_connector:
                # Execute on actual database
                query_results = await self.database_connector.execute_query(sql_query)
                df = pd.DataFrame(query_results)
            else:
                # Mock execution for testing
                df = pd.DataFrame({"message": ["No database connection available"]})
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return {
                "success": True,
                "query_results": df.to_dict('records'),
                "dataframe": df,
                "result_count": len(df),
                "execution_time": execution_time,
                "chart_config": chart_config or {},
                "plotting_code": plotting_code or "",
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"SQL execution failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "query_results": [],
                "dataframe": pd.DataFrame(),
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
                    "memory_backend": getattr(self.settings, 'memory_backend', 'sqlite'),
                    "session_db_path": getattr(self.settings, 'session_db_path', ':memory:'),
                    "knowledge_db_path": getattr(self.settings, 'knowledge_db_path', ':memory:'),
                    "llm_provider": getattr(self.settings, 'llm_provider', 'mock'),
                },
                "schema_info": {
                    "total_tables": len(self.all_tables),
                    "total_columns": len(self.all_columns),
                    "tables": [f"{t['database']}.{t['schema']}.{t['table_name']}" for t in self.all_tables[:10]]
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

# Global system instance for easy access
_system_instance = None

async def get_system_instance(schema_file_path: str = None) -> SQLAgentSystem:
    """Get or create global system instance"""
    global _system_instance
    
    if _system_instance is None:
        _system_instance = SQLAgentSystem()
        await _system_instance.initialize(schema_file_path)
    
    return _system_instance

async def main():
    """Main application entry point for testing"""
    try:
        # Create necessary directories
        from config.settings import create_directories
        create_directories()
        
        # Initialize settings
        settings = get_settings()
        
        # Create system instance
        system = SQLAgentSystem(settings)
        
        # Initialize system
        await system.initialize()
        
        print("\n[AI] SQL Agent System")
        print("=" * 50)
        print("System initialized successfully!")
        print("=" * 50)
        
        # Show system status
        status = await system.get_system_status()
        print(f"\n[STATS] System Status:")
        print(f"  Status: {status.get('system_status')}")
        print(f"  Tables: {status.get('schema_info', {}).get('total_tables', 0)}")
        print(f"  Columns: {status.get('schema_info', {}).get('total_columns', 0)}")
        print(f"  Memory: {status.get('memory_stats', {})}")
        
        # Run a demo query to test the system
        demo_query = "Show me sales data for the last quarter"
        print(f"\n[DEMO] Testing with query: '{demo_query}'")
        print("[PROC] Processing...")
        
        result = await system.process_query(demo_query)
        
        # Display results
        if result.get('success'):
            print(f"\n[PASS] Success! ({result.get('processing_time', 0):.2f}s)")
            print(f"\n[NOTE] Generated SQL:")
            print(result.get('sql_query', 'No SQL generated'))
            
            chart_config = result.get('chart_config', {})
            if chart_config:
                print(f"\n[STATS] Chart: {chart_config.get('chart_type', 'unknown')}")
                print(f"   Title: {chart_config.get('title', 'No title')}")
            
            validation = result.get('validation', {})
            if validation.get('guardrails_passed'):
                print("[PASS] All guardrails passed")
            else:
                print("[WARN] Some guardrails failed")
                for warning in validation.get('performance_checks', []):
                    print(f"   - {warning}")
        
        else:
            print(f"\n[FAIL] Error: {result.get('error', 'Unknown error')}")
            print(f"   Type: {result.get('error_type', 'Unknown')}")
        
        print(f"\n[INFO] System test completed. Use the Streamlit UI or FastAPI for interactive usage.")
        print(f"  Streamlit: streamlit run ui/streamlit_app.py")
        print(f"  FastAPI: python api/fastapi_app.py")
        
        # Cleanup
        await system.cleanup()
        
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        print(f"\n[ERROR] Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n[BYE] System shutdown requested")
    except Exception as e:
        print(f"\n[ERROR] Failed to start system: {e}")
        sys.exit(1)