"""
Advanced SQL Agent System
Main application entry point for the memory-driven SQL generation system.
"""

import asyncio
import logging
from typing import Optional
from agents.sql_workflow import SQLGenerationWorkflow
from memory.memory_manager import MemoryManager
from database.snowflake_connector import SnowflakeConnector
from config.settings import Settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/sql_agent.log'),
        logging.StreamHandler()
    ]
)

class SQLAgentSystem:
    """
    Main system orchestrator that coordinates all components of the
    advanced SQL agent system with memory-driven intelligence.
    """
    
    def __init__(self, config: Optional[Settings] = None):
        self.config = config or Settings()
        self.logger = logging.getLogger(__name__)
        
        # Initialize core components
        self.memory_manager = MemoryManager(self.config)
        self.database_connector = SnowflakeConnector(self.config)
        self.workflow = SQLGenerationWorkflow(
            memory_system=self.memory_manager,
            database_connector=self.database_connector,
            llm_provider=self.config.llm_provider
        )
        
    async def process_query(self, user_query: str, user_id: str = "anonymous") -> dict:
        """
        Processes a user query through the complete SQL agent workflow.
        
        Args:
            user_query: Natural language query from user
            user_id: User identifier for personalization and memory
            
        Returns:
            Complete response with SQL, results, and visualizations
        """
        try:
            self.logger.info(f"Processing query for user {user_id}: {user_query}")
            
            # Initialize processing state
            initial_state = {
                "user_query": user_query,
                "user_id": user_id,
                "timestamp": datetime.now().isoformat()
            }
            
            # Execute workflow
            result = await self.workflow.execute(initial_state)
            
            self.logger.info(f"Query processed successfully for user {user_id}")
            return result
            
        except Exception as e:
            self.logger.error(f"Query processing failed: {str(e)}")
            raise

if __name__ == "__main__":
    # Example usage
    import asyncio
    
    async def main():
        system = SQLAgentSystem()
        result = await system.process_query(
            "Show me total sales by product category for this quarter"
        )
        print("Query Result:", result)
    
    asyncio.run(main())
