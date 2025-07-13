"""
Integration tests for the complete SQL agent system
"""

import pytest
import asyncio
from main import SQLAgentSystem

class TestEndToEndWorkflow:
    """End-to-end integration tests."""
    
    @pytest.mark.asyncio
    async def test_simple_query_workflow(self, test_settings):
        """Test complete workflow for simple query."""
        
        # This would require a test database setup
        pytest.skip("Requires test database setup")
        
        system = SQLAgentSystem(test_settings)
        
        result = await system.process_query(
            "Show me total sales by category",
            "test_user"
        )
        
        assert result["success"] is True
        assert "generated_sql" in result
        assert "query_results" in result
    
    @pytest.mark.asyncio
    async def test_complex_query_workflow(self, test_settings):
        """Test workflow for complex query requiring multiple iterations."""
        
        pytest.skip("Requires test database setup")
        
        system = SQLAgentSystem(test_settings)
        
        result = await system.process_query(
            "Compare this quarter's performance to last year for products launched in 2023",
            "test_user"
        )
        
        assert result["success"] is True
    
    @pytest.mark.asyncio
    async def test_memory_learning(self, test_settings):
        """Test that system learns from successful interactions."""
        
        pytest.skip("Requires test database setup")
        
        system = SQLAgentSystem(test_settings)
        
        # Process same query twice and verify improvement
        result1 = await system.process_query("Show sales by category", "test_user")
        result2 = await system.process_query("Show sales by category", "test_user")
        
        # Second query should be faster due to memory
        assert result2["processing_time"] <= result1["processing_time"]

class TestDatabaseIntegration:
    """Database integration tests."""
    
    @pytest.mark.asyncio
    async def test_snowflake_connection(self, test_settings):
        """Test Snowflake database connection."""
        
        pytest.skip("Requires Snowflake test environment")
        
        from database.snowflake_connector import SnowflakeConnector
        
        connector = SnowflakeConnector(test_settings)
        await connector.connect()
        
        schema = await connector.get_schema_metadata()
        assert schema is not None
        
        await connector.close()
    
    @pytest.mark.asyncio
    async def test_query_execution(self, test_settings):
        """Test actual query execution."""
        
        pytest.skip("Requires Snowflake test environment")
        
        from database.snowflake_connector import SnowflakeConnector
        
        connector = SnowflakeConnector(test_settings)
        await connector.connect()
        
        results, metadata = await connector.execute_query("SELECT 1 as test")
        
        assert len(results) == 1
        assert results[0]["TEST"] == 1
        assert metadata["execution_time"] > 0
        
        await connector.close()

class TestAPIIntegration:
    """API integration tests."""
    
    def test_fastapi_endpoints(self):
        """Test FastAPI endpoints."""
        
        from fastapi.testclient import TestClient
        from api.fastapi_app import app
        
        client = TestClient(app)
        
        # Test health endpoint
        response = client.get("/health")
        assert response.status_code == 200
        
        # Test examples endpoint  
        response = client.get("/examples")
        assert response.status_code == 200
        assert "examples" in response.json()
