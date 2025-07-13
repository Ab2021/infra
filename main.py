"""
Advanced SQL Agent System - Minimal Configuration
Main application entry point optimized for minimal dependencies and in-memory SQLite usage.
"""

import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from config.settings import Settings, get_settings
from memory.minimal_memory import MinimalMemorySystem, MemoryManager
from workflows.sql_workflow import SQLGenerationWorkflow

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

class AdvancedSQLAgentSystem:
    """
    Main system orchestrator with minimal dependencies.
    Uses in-memory SQLite and optional Redis/embedding support.
    """
    
    def __init__(self, settings: Settings = None):
        """Initialize the SQL agent system."""
        self.settings = settings or get_settings()
        self.memory_system = None
        self.memory_manager = None
        self.workflow = None
        self.database_connector = None
        self.llm_provider = None
        
        self.logger = logging.getLogger(__name__)
        
    async def initialize(self):
        """Initialize all system components."""
        try:
            self.logger.info("Initializing Advanced SQL Agent System (Minimal Configuration)")
            
            # Validate configuration
            if not self.settings.validate_configuration():
                raise ValueError("Invalid configuration")
            
            # Initialize memory system with minimal configuration
            self.memory_system = MinimalMemorySystem(
                session_db_path=self.settings.session_db_path,  # ":memory:" for in-memory
                knowledge_db_path=self.settings.knowledge_db_path,
                enable_vector_search=self.settings.enable_vector_search,
                similarity_threshold=self.settings.vector_similarity_threshold
            )
            
            self.memory_manager = MemoryManager(self.memory_system)
            
            # Initialize database connector
            self.database_connector = await self._initialize_database_connector()
            
            # Initialize LLM provider
            self.llm_provider = await self._initialize_llm_provider()
            
            # Initialize workflow
            self.workflow = SQLGenerationWorkflow(
                memory_system=self.memory_manager,
                database_connector=self.database_connector,
                llm_provider=self.llm_provider
            )
            
            # Get memory stats
            stats = await self.memory_system.get_memory_stats()
            self.logger.info(f"System initialized successfully. Memory stats: {stats}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize system: {e}")
            raise
    
    async def _initialize_database_connector(self):
        """Initialize database connector."""
        try:
            # Import here to avoid dependency issues if not needed
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
        """Initialize LLM provider."""
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
        
    async def process_query(self, user_query: str, user_id: str = "anonymous") -> Dict[str, Any]:
        """
        Process a natural language query and return SQL results.
        
        Args:
            user_query: Natural language query
            user_id: User identifier
            
        Returns:
            Complete processing results
        """
        try:
            self.logger.info(f"Processing query for user {user_id}: {user_query}")
            
            # Execute workflow
            result = await self.workflow.execute(user_query, user_id)
            
            self.logger.info(f"Query processing completed. Success: {result.get('success', False)}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error processing query: {e}")
            return {
                "success": False,
                "error": str(e),
                "user_query": user_query,
                "user_id": user_id
            }
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get current system status and statistics."""
        try:
            status = {
                "system_status": "running",
                "configuration": {
                    "memory_backend": self.settings.memory_backend,
                    "session_db_path": self.settings.session_db_path,
                    "knowledge_db_path": self.settings.knowledge_db_path,
                    "use_redis": self.settings.use_redis,
                    "enable_vector_search": self.settings.enable_vector_search,
                    "llm_provider": self.settings.llm_provider,
                    "llm_model": getattr(self.settings, 'openai_model', 'unknown')
                }
            }
            
            if self.memory_system:
                status["memory_stats"] = await self.memory_system.get_memory_stats()
            
            return status
            
        except Exception as e:
            self.logger.error(f"Error getting system status: {e}")
            return {"system_status": "error", "error": str(e)}
    
    async def cleanup(self):
        """Cleanup system resources."""
        try:
            self.logger.info("Cleaning up system resources")
            
            # Cleanup old sessions
            if self.memory_system:
                await self.memory_system.cleanup_old_sessions()
            
            # Close database connections
            if self.database_connector:
                await self.database_connector.close()
            
            self.logger.info("System cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")

async def main():
    """Main application entry point."""
    try:
        # Create necessary directories
        from config.settings import create_directories
        create_directories()
        
        # Initialize settings
        settings = get_settings()
        
        # Create system instance
        system = AdvancedSQLAgentSystem(settings)
        
        # Initialize system
        await system.initialize()
        
        # Interactive mode
        print("\n🤖 Advanced SQL Agent System - Minimal Configuration")
        print("=" * 60)
        print("Type 'status' to see system status")
        print("Type 'exit' to quit")
        print("Enter natural language queries to generate SQL")
        print("=" * 60)
        
        while True:
            try:
                user_input = input("\n📝 Query: ").strip()
                
                if user_input.lower() in ['exit', 'quit', 'q']:
                    break
                
                if user_input.lower() == 'status':
                    status = await system.get_system_status()
                    print(f"\n📊 System Status: {status}")
                    continue
                
                if not user_input:
                    continue
                
                # Process query
                result = await system.process_query(user_input)
                
                # Display results
                print(f"\n🔄 Processing Results:")
                print(f"Success: {result.get('success', False)}")
                
                if result.get('success'):
                    if result.get('generated_sql'):
                        print(f"Generated SQL:\n{result['generated_sql']}")
                    
                    if result.get('query_results'):
                        print(f"Results: {len(result['query_results'])} rows")
                        # Show first few rows
                        for i, row in enumerate(result['query_results'][:3]):
                            print(f"  Row {i+1}: {row}")
                        if len(result['query_results']) > 3:
                            print(f"  ... and {len(result['query_results'])-3} more rows")
                else:
                    print(f"Error: {result.get('error', 'Unknown error')}")
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
        
        # Cleanup
        await system.cleanup()
        
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        print(f"\n💥 Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n👋 System shutdown requested")
    except Exception as e:
        print(f"\n💥 Failed to start system: {e}")
        sys.exit(1)
