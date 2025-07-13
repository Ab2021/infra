"""
Test configuration and fixtures for the SQL agent system
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock
from config.settings import Settings

@pytest.fixture
def test_settings():
    """Test configuration settings."""
    return Settings(
        snowflake_account="test_account",
        snowflake_user="test_user", 
        snowflake_password="test_password",
        snowflake_warehouse="test_warehouse",
        snowflake_database="test_database",
        openai_api_key="test_key",
        debug_mode=True
    )

@pytest.fixture
def mock_database_connector():
    """Mock database connector for testing."""
    connector = Mock()
    connector.get_schema_metadata = AsyncMock(return_value={
        "test_table": {
            "columns": [
                {"name": "id", "type": "INTEGER"},
                {"name": "name", "type": "VARCHAR"}
            ]
        }
    })
    connector.execute_query = AsyncMock(return_value=(
        [{"id": 1, "name": "test"}], 
        {"execution_time": 0.1}
    ))
    return connector

@pytest.fixture
def mock_memory_system():
    """Mock memory system for testing."""
    memory = Mock()
    memory.initialize_processing_session = AsyncMock(return_value={})
    memory.get_contextual_memories = AsyncMock(return_value={})
    memory.update_memory_from_processing = AsyncMock()
    return memory

@pytest.fixture
def mock_llm_provider():
    """Mock LLM provider for testing."""
    llm = Mock()
    llm.ainvoke = AsyncMock(return_value=Mock(
        content='{"intent": {"primary_action": "select"}, "entities": []}'
    ))
    return llm

@pytest.fixture
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()
