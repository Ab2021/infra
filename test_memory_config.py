#!/usr/bin/env python3
"""
Test script to verify in-memory SQLite configuration
Tests both simple memory system and session memory with in-memory setup
"""

import asyncio
import os
import sys
import logging

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from memory.simple_memory import SimpleMemorySystem
from memory.session_memory import SessionMemory
from config.settings import Settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_simple_memory_system():
    """Test SimpleMemorySystem with in-memory configuration"""
    logger.info("Testing SimpleMemorySystem with in-memory SQLite...")
    
    try:
        # Initialize with in-memory configuration
        memory_system = SimpleMemorySystem(
            session_db_path=":memory:",
            knowledge_db_path=":memory:",
            similarity_threshold=0.3,
            enable_persistence=False
        )
        
        # Test session creation
        session_id = await memory_system.create_session("test_user")
        logger.info(f"Created session: {session_id}")
        
        # Test adding conversation
        await memory_system.add_to_conversation(
            session_id=session_id,
            query="Show me sales data",
            intent={"type": "analytics", "tables": ["sales"]},
            tables_used=["sales"],
            chart_type="bar_chart",
            success=True
        )
        logger.info("Added conversation entry")
        
        # Test storing successful query
        await memory_system.store_successful_query(
            query="Show me sales data",
            sql="SELECT * FROM sales",
            execution_time=1.2,
            result_count=100,
            tables_used=["sales"],
            chart_type="bar_chart"
        )
        logger.info("Stored successful query")
        
        # Test finding similar queries
        similar_queries = await memory_system.find_similar_queries("sales information")
        logger.info(f"Found {len(similar_queries)} similar queries")
        
        # Get memory stats
        stats = await memory_system.get_memory_stats()
        logger.info(f"Memory stats: {stats}")
        
        logger.info("✅ SimpleMemorySystem test passed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ SimpleMemorySystem test failed: {e}")
        return False

async def test_session_memory():
    """Test SessionMemory with in-memory configuration"""
    logger.info("Testing SessionMemory with in-memory SQLite...")
    
    try:
        # Initialize with in-memory configuration
        session_memory = SessionMemory(
            db_path=":memory:",
            enable_persistence=False
        )
        
        # Test session creation
        session_data = await session_memory.get_or_create_session("test_user", "test_session")
        logger.info(f"Session data: {session_data['session_id']}")
        
        # Test processing result storage
        await session_memory.add_processing_result(
            user_id="test_user",
            session_id="test_session",
            agent_name="test_agent",
            processing_data={"result": "success", "query": "test"},
            success=True
        )
        logger.info("Added processing result")
        
        # Test getting relevant memories
        memories = await session_memory.get_relevant_memories(
            user_id="test_user",
            query="test query",
            context_type="analytics"
        )
        logger.info(f"Retrieved memories: {len(memories.get('similar_queries', []))}")
        
        # Test user insights
        insights = await session_memory.get_user_insights("test_user")
        logger.info(f"User insights: {insights}")
        
        logger.info("✅ SessionMemory test passed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ SessionMemory test failed: {e}")
        return False

def test_settings_configuration():
    """Test settings configuration for in-memory usage"""
    logger.info("Testing Settings configuration...")
    
    try:
        # Test default in-memory configuration
        settings = Settings(
            # Required Snowflake settings (use dummy values for test)
            snowflake_account="test_account",
            snowflake_user="test_user", 
            snowflake_password="test_password",
            snowflake_warehouse="test_warehouse",
            snowflake_database="test_database",
            # Memory configuration
            session_db_path=":memory:",
            knowledge_db_path=":memory:",
            # LLM configuration
            openai_api_key="test_key"
        )
        
        logger.info(f"Session DB path: {settings.session_db_path}")
        logger.info(f"Knowledge DB path: {settings.knowledge_db_path}")
        logger.info(f"Memory backend: {settings.memory_backend}")
        
        # Test configuration validation would require real credentials
        # Just check that the paths are correctly set for in-memory
        assert settings.session_db_path == ":memory:"
        assert settings.knowledge_db_path == ":memory:"
        
        logger.info("✅ Settings configuration test passed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Settings test failed: {e}")
        return False

async def test_persistence_functionality():
    """Test persistence functionality with in-memory primary and file backup"""
    logger.info("Testing persistence functionality...")
    
    try:
        # Test with persistence enabled
        memory_system = SimpleMemorySystem(
            session_db_path=":memory:",
            knowledge_db_path=":memory:",
            enable_persistence=True,
            persistent_session_path="test_data/test_session.db",
            persistent_knowledge_path="test_data/test_knowledge.db"
        )
        
        # Add some data
        session_id = await memory_system.create_session("persist_user")
        await memory_system.add_to_conversation(
            session_id=session_id,
            query="Test persistence query",
            success=True
        )
        
        # Test save to persistent storage
        await memory_system.save_to_persistent_storage()
        logger.info("Saved to persistent storage")
        
        # Cleanup test files
        import shutil
        if os.path.exists("test_data"):
            shutil.rmtree("test_data")
        
        logger.info("✅ Persistence functionality test passed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Persistence test failed: {e}")
        return False

async def main():
    """Run all tests"""
    logger.info("🚀 Starting SQLite in-memory configuration tests...")
    
    results = []
    
    # Test settings configuration
    results.append(test_settings_configuration())
    
    # Test memory systems
    results.append(await test_simple_memory_system())
    results.append(await test_session_memory())
    results.append(await test_persistence_functionality())
    
    # Summary
    passed = sum(results)
    total = len(results)
    
    logger.info(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed! SQLite in-memory configuration is working correctly.")
        return 0
    else:
        logger.error("❌ Some tests failed. Please check the configuration.")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)