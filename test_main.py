"""
Comprehensive Test Suite for SQL Agent System
Tests the complete pipeline from natural language to SQL generation and execution.
"""

import asyncio
import logging
import os
import sys
import pandas as pd
import json
import time
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/test_system.log', mode='w') if Path('logs').exists() else logging.NullHandler()
    ]
)

logger = logging.getLogger(__name__)

# ========================================
# TEST CONFIGURATION
# ========================================

class TestConfig:
    """Test configuration and constants"""
    
    # Schema file path - UPDATE THIS PATH TO YOUR EXCEL FILE
    SCHEMA_FILE = "hackathon_final_schema_file_v1.xlsx"
    
    # Test queries for different scenarios
    TEST_QUERIES = [
        # Basic aggregation
        {
            "query": "Show total premium by state for 2023",
            "expected_action": "aggregation",
            "expected_tables": ["policy", "premium"],
            "description": "Basic aggregation with time filter"
        },
        
        # Trend analysis
        {
            "query": "Show premium trends over time by quarter",
            "expected_action": "trend_analysis", 
            "expected_tables": ["policy", "premium"],
            "description": "Time-based trend analysis"
        },
        
        # Comparison
        {
            "query": "Compare claims by region for different policy types",
            "expected_action": "comparison",
            "expected_tables": ["claims", "policy"],
            "description": "Multi-dimensional comparison"
        },
        
        # Filtering with conditions
        {
            "query": "List auto policies with premium over 10000 in Texas in 2023",
            "expected_action": "filtering",
            "expected_tables": ["policy"],
            "description": "Complex filtering with multiple conditions"
        },
        
        # Ranking query
        {
            "query": "Show top 10 customers by total claims amount",
            "expected_action": "ranking",
            "expected_tables": ["customer", "claims"],
            "description": "Ranking with aggregation"
        }
    ]
    
    # Expected system components
    REQUIRED_COMPONENTS = [
        "memory_system",
        "query_agent", 
        "profiling_agent",
        "sql_viz_agent"
    ]

# ========================================
# UTILITY FUNCTIONS
# ========================================

def create_test_directories():
    """Create necessary directories for testing"""
    directories = ['logs', 'data', 'data/vector_store']
    for dir_path in directories:
        Path(dir_path).mkdir(parents=True, exist_ok=True)

def print_test_header(test_name: str, description: str = ""):
    """Print formatted test header"""
    print(f"\n{'='*60}")
    print(f"🧪 TEST: {test_name}")
    if description:
        print(f"📋 Description: {description}")
    print(f"{'='*60}")

def print_test_result(success: bool, message: str = ""):
    """Print formatted test result"""
    status = "✅ PASSED" if success else "❌ FAILED"
    print(f"\n{status}")
    if message:
        print(f"📝 Details: {message}")

def format_duration(start_time: datetime) -> str:
    """Format duration since start time"""
    duration = (datetime.now() - start_time).total_seconds()
    return f"{duration:.2f}s"

# ========================================
# SCHEMA LOADING TESTS
# ========================================

async def test_schema_loading():
    """Test loading schema from Excel file"""
    print_test_header("Schema Loading", "Test loading and parsing Excel schema file")
    
    start_time = datetime.now()
    
    try:
        # Check if schema file exists
        if not Path(TestConfig.SCHEMA_FILE).exists():
            print_test_result(False, f"Schema file not found: {TestConfig.SCHEMA_FILE}")
            return False
        
        # Load schema using pandas (same as main system)
        logger.info(f"Loading schema from {TestConfig.SCHEMA_FILE}")
        
        # Test table catalog loading
        table_catalog = pd.read_excel(
            TestConfig.SCHEMA_FILE,
            sheet_name='Table_descriptions'
        )[['DATABASE', 'SCHEMA', 'TABLE', 'Brief_Description', 'Detailed_Comments']]
        
        # Test column catalog loading
        column_catalog = pd.read_excel(
            TestConfig.SCHEMA_FILE,
            sheet_name="Table's Column Summaries"
        )[['Table Name', 'Feature Name', 'Data Type', 'Description', 'sample_100_distinct']]
        
        # Validate loaded data
        tables_count = len(table_catalog)
        columns_count = len(column_catalog)
        
        if tables_count == 0 or columns_count == 0:
            print_test_result(False, f"Empty schema data: {tables_count} tables, {columns_count} columns")
            return False
        
        # Check for column name variations (spaces, etc.)
        column_names_with_spaces = column_catalog['Feature Name'].apply(lambda x: ' ' in str(x)).sum()
        
        print(f"📊 Schema Statistics:")
        print(f"   - Tables loaded: {tables_count}")
        print(f"   - Columns loaded: {columns_count}")
        print(f"   - Columns with spaces: {column_names_with_spaces}")
        print(f"   - Load time: {format_duration(start_time)}")
        
        # Display sample data
        print(f"\n📋 Sample Tables:")
        for i, row in table_catalog.head(3).iterrows():
            print(f"   - {row['DATABASE']}.{row['SCHEMA']}.{row['TABLE']}: {row['Brief_Description']}")
        
        print(f"\n📋 Sample Columns:")
        for i, row in column_catalog.head(5).iterrows():
            print(f"   - {row['Table Name']}.{row['Feature Name']} ({row['Data Type']})")
        
        print_test_result(True, f"Successfully loaded {tables_count} tables and {columns_count} columns")
        return True
        
    except Exception as e:
        print_test_result(False, f"Schema loading failed: {e}")
        return False

# ========================================
# SYSTEM INITIALIZATION TESTS
# ========================================

async def test_system_initialization():
    """Test system initialization with all components"""
    print_test_header("System Initialization", "Test initialization of all system components")
    
    start_time = datetime.now()
    
    try:
        # Import main system
        from main import SQLAgentSystem, get_settings
        from config.settings import create_directories
        
        # Create directories
        create_directories()
        
        # Initialize settings
        settings = get_settings()
        
        # Create system instance
        logger.info("Creating SQL Agent System instance")
        system = SQLAgentSystem(settings)
        
        # Initialize with schema file
        logger.info("Initializing system with schema")
        await system.initialize(TestConfig.SCHEMA_FILE)
        
        # Verify all components are initialized
        missing_components = []
        for component in TestConfig.REQUIRED_COMPONENTS:
            if not hasattr(system, component) or getattr(system, component) is None:
                missing_components.append(component)
        
        if missing_components:
            print_test_result(False, f"Missing components: {missing_components}")
            return None, False
        
        # Test system status
        status = await system.get_system_status()
        
        print(f"📊 System Status:")
        print(f"   - Status: {status.get('system_status')}")
        print(f"   - Tables: {status.get('schema_info', {}).get('total_tables', 0)}")
        print(f"   - Columns: {status.get('schema_info', {}).get('total_columns', 0)}")
        print(f"   - Memory stats: {status.get('memory_stats', {})}")
        print(f"   - Init time: {format_duration(start_time)}")
        
        print_test_result(True, f"All {len(TestConfig.REQUIRED_COMPONENTS)} components initialized successfully")
        return system, True
        
    except Exception as e:
        print_test_result(False, f"System initialization failed: {e}")
        return None, False

# ========================================
# AGENT PIPELINE TESTS
# ========================================

async def test_agent_pipeline(system, test_query: Dict):
    """Test the complete agent pipeline for a single query"""
    
    query_text = test_query["query"]
    description = test_query["description"]
    
    print_test_header(f"Agent Pipeline - {query_text}", description)
    
    start_time = datetime.now()
    
    try:
        logger.info(f"Testing query: {query_text}")
        
        # Process query through the pipeline
        result = await system.process_query(query_text)
        
        processing_time = format_duration(start_time)
        
        # Validate result structure
        required_fields = ["success", "user_query", "sql_query", "chart_config"]
        missing_fields = [field for field in required_fields if field not in result]
        
        if missing_fields:
            print_test_result(False, f"Missing result fields: {missing_fields}")
            return False
        
        # Check if processing was successful
        if not result.get("success"):
            print_test_result(False, f"Processing failed: {result.get('error', 'Unknown error')}")
            return False
        
        # Display results
        print(f"📊 Pipeline Results:")
        print(f"   - Processing time: {result.get('processing_time', 0):.2f}s")
        print(f"   - Total time: {processing_time}")
        print(f"   - Confidence: {result.get('query_understanding', {}).get('confidence', 0):.2f}")
        
        print(f"\n🎯 Query Understanding:")
        understanding = result.get('query_understanding', {})
        if understanding.get('query_intent'):
            intent = understanding['query_intent']
            if hasattr(intent, '__dict__'):
                print(f"   - Action: {intent.action}")
                print(f"   - Metrics: {intent.metrics}")
                print(f"   - Dimensions: {intent.dimensions}")
            else:
                print(f"   - Intent: {intent}")
        
        print(f"\n📊 Data Profiling:")
        profiling = result.get('data_profiling', {})
        column_profiles = profiling.get('column_profiles', {})
        print(f"   - Columns profiled: {len(column_profiles)}")
        print(f"   - Quality metrics: {profiling.get('quality_metrics', {})}")
        
        print(f"\n🔍 SQL Generation:")
        sql_query = result.get('sql_query', '')
        chart_config = result.get('chart_config', {})
        print(f"   - SQL length: {len(sql_query)} characters")
        print(f"   - Chart type: {chart_config.get('chart_type', 'none')}")
        print(f"   - Validation: {result.get('validation', {}).get('guardrails_passed', False)}")
        
        # Display generated SQL (truncated)
        if sql_query:
            print(f"\n📝 Generated SQL:")
            print(f"   {sql_query[:200]}{'...' if len(sql_query) > 200 else ''}")
        
        print_test_result(True, f"Pipeline completed successfully in {processing_time}")
        return True
        
    except Exception as e:
        print_test_result(False, f"Pipeline test failed: {e}")
        return False

# ========================================
# SQL EXECUTION TESTS
# ========================================

async def test_sql_execution(system):
    """Test SQL execution capabilities"""
    print_test_header("SQL Execution", "Test SQL query execution and result handling")
    
    start_time = datetime.now()
    
    try:
        # Test simple query execution
        test_queries = [
            "SELECT 1 as test_column",
            "SELECT 'Hello' as message, 123 as number",
        ]
        
        for i, sql_query in enumerate(test_queries, 1):
            logger.info(f"Testing SQL execution {i}: {sql_query}")
            
            # Execute SQL
            execution_result = await system.execute_sql_and_get_results(sql_query)
            
            print(f"\n📋 Test Query {i}: {sql_query}")
            print(f"   - Success: {execution_result.get('success', False)}")
            print(f"   - Rows returned: {execution_result.get('result_count', 0)}")
            print(f"   - Execution time: {execution_result.get('execution_time', 0):.3f}s")
            
            if execution_result.get('success'):
                df = execution_result.get('dataframe')
                if df is not None and not df.empty:
                    print(f"   - Columns: {list(df.columns)}")
                    print(f"   - Sample data: {df.iloc[0].to_dict() if len(df) > 0 else 'No data'}")
        
        print_test_result(True, f"SQL execution tests completed in {format_duration(start_time)}")
        return True
        
    except Exception as e:
        print_test_result(False, f"SQL execution test failed: {e}")
        return False

# ========================================
# MEMORY SYSTEM TESTS
# ========================================

async def test_memory_system(system):
    """Test memory system functionality"""
    print_test_header("Memory System", "Test memory storage and retrieval capabilities")
    
    start_time = datetime.now()
    
    try:
        memory_system = system.memory_system
        
        # Test session creation
        logger.info("Testing session management")
        session_id = await memory_system.create_session("test_user")
        print(f"📋 Session created: {session_id}")
        
        # Test conversation storage
        await memory_system.add_to_conversation(
            session_id=session_id,
            query="Test query for memory",
            intent={"action": "test", "metrics": ["test_metric"]},
            tables_used=["test_table"],
            chart_type="bar_chart",
            success=True
        )
        
        # Test session context retrieval
        context = await memory_system.get_session_context(session_id)
        conversation_count = len(context.get("conversation_history", []))
        print(f"📋 Conversations in session: {conversation_count}")
        
        # Test successful query storage
        await memory_system.store_successful_query(
            query="Test successful query",
            sql="SELECT * FROM test_table",
            execution_time=0.5,
            result_count=100,
            tables_used=["test_table"],
            chart_type="bar_chart"
        )
        
        # Test pattern similarity search
        similar_queries = await memory_system.find_similar_queries("Test query", top_k=3)
        print(f"📋 Similar queries found: {len(similar_queries)}")
        
        # Test memory statistics
        stats = await memory_system.get_memory_stats()
        print(f"\n📊 Memory Statistics:")
        for key, value in stats.items():
            print(f"   - {key}: {value}")
        
        print_test_result(True, f"Memory system tests completed in {format_duration(start_time)}")
        return True
        
    except Exception as e:
        print_test_result(False, f"Memory system test failed: {e}")
        return False

# ========================================
# COLUMN NAME HANDLING TESTS
# ========================================

async def test_column_name_handling(system):
    """Test column name normalization and handling"""
    print_test_header("Column Name Handling", "Test handling of column names with spaces and special characters")
    
    start_time = datetime.now()
    
    try:
        # Test column name resolution
        agent = system.query_agent
        
        test_cases = [
            ("Feature Name", "Feature Name"),  # Should remain the same
            ("Feature_Name", "Feature Name"),  # Should convert to space version
            ("feature name", "Feature Name"),  # Should handle case
            ("Policy Number", "Policy Number"),  # Should handle existing spaces
        ]
        
        print(f"\n📋 Column Name Resolution Tests:")
        for input_name, expected in test_cases:
            resolved = agent.resolve_column_name(input_name)
            quoted = agent.quote_column_name(resolved or input_name)
            
            print(f"   - '{input_name}' → '{resolved}' → '{quoted}'")
        
        # Test column mapping creation
        print(f"\n📊 Column Mapping Statistics:")
        print(f"   - Total mappings: {len(agent.column_mapping)}")
        print(f"   - Available columns: {len(agent.all_columns)}")
        
        # Sample some mappings
        sample_mappings = list(agent.column_mapping.items())[:5]
        print(f"\n📋 Sample Mappings:")
        for key, value in sample_mappings:
            print(f"   - '{key}' → '{value}'")
        
        print_test_result(True, f"Column name handling tests completed in {format_duration(start_time)}")
        return True
        
    except Exception as e:
        print_test_result(False, f"Column name handling test failed: {e}")
        return False

# ========================================
# END-TO-END INTEGRATION TESTS
# ========================================

async def test_end_to_end_integration():
    """Run complete end-to-end integration tests"""
    print_test_header("End-to-End Integration", "Complete system integration test with real queries")
    
    total_start_time = datetime.now()
    
    # Test results tracking
    test_results = {
        "schema_loading": False,
        "system_initialization": False,
        "agent_pipeline": [],
        "sql_execution": False,
        "memory_system": False,
        "column_handling": False
    }
    
    try:
        # 1. Test Schema Loading
        test_results["schema_loading"] = await test_schema_loading()
        
        # 2. Test System Initialization
        system, init_success = await test_system_initialization()
        test_results["system_initialization"] = init_success
        
        if not system:
            print_test_result(False, "Cannot continue without system initialization")
            return test_results
        
        # 3. Test Agent Pipeline with multiple queries
        print(f"\n{'='*60}")
        print(f"🔄 TESTING AGENT PIPELINE WITH {len(TestConfig.TEST_QUERIES)} QUERIES")
        print(f"{'='*60}")
        
        for i, test_query in enumerate(TestConfig.TEST_QUERIES, 1):
            print(f"\n--- Query {i}/{len(TestConfig.TEST_QUERIES)} ---")
            result = await test_agent_pipeline(system, test_query)
            test_results["agent_pipeline"].append({
                "query": test_query["query"],
                "success": result
            })
            
            # Small delay between tests
            await asyncio.sleep(0.5)
        
        # 4. Test SQL Execution
        test_results["sql_execution"] = await test_sql_execution(system)
        
        # 5. Test Memory System
        test_results["memory_system"] = await test_memory_system(system)
        
        # 6. Test Column Name Handling
        test_results["column_handling"] = await test_column_name_handling(system)
        
        # Cleanup
        await system.cleanup()
        
        # Generate final report
        total_time = format_duration(total_start_time)
        
        print(f"\n{'='*60}")
        print(f"📊 FINAL TEST REPORT")
        print(f"{'='*60}")
        
        print(f"⏱️  Total test time: {total_time}")
        
        print(f"\n📋 Component Tests:")
        print(f"   - Schema Loading: {'✅' if test_results['schema_loading'] else '❌'}")
        print(f"   - System Init: {'✅' if test_results['system_initialization'] else '❌'}")
        print(f"   - SQL Execution: {'✅' if test_results['sql_execution'] else '❌'}")
        print(f"   - Memory System: {'✅' if test_results['memory_system'] else '❌'}")
        print(f"   - Column Handling: {'✅' if test_results['column_handling'] else '❌'}")
        
        print(f"\n📋 Query Pipeline Tests:")
        pipeline_success = 0
        for result in test_results["agent_pipeline"]:
            status = "✅" if result["success"] else "❌"
            print(f"   - {status} {result['query']}")
            if result["success"]:
                pipeline_success += 1
        
        print(f"\n📊 Success Rate:")
        total_tests = (5 + len(test_results["agent_pipeline"]))  # 5 component tests + pipeline tests
        total_success = (
            sum([
                test_results["schema_loading"],
                test_results["system_initialization"], 
                test_results["sql_execution"],
                test_results["memory_system"],
                test_results["column_handling"]
            ]) + pipeline_success
        )
        
        success_rate = (total_success / total_tests) * 100
        print(f"   - Overall: {total_success}/{total_tests} ({success_rate:.1f}%)")
        
        if success_rate >= 80:
            print(f"\n🎉 INTEGRATION TESTS PASSED! System is ready for production.")
        else:
            print(f"\n⚠️  INTEGRATION TESTS PARTIALLY FAILED. Review failed components.")
        
        return test_results
        
    except Exception as e:
        print_test_result(False, f"Integration test failed: {e}")
        return test_results

# ========================================
# INTERACTIVE TEST MODE
# ========================================

async def interactive_test_mode():
    """Interactive test mode for manual testing"""
    print_test_header("Interactive Test Mode", "Manual testing with custom queries")
    
    try:
        # Initialize system
        from main import SQLAgentSystem, get_settings
        from config.settings import create_directories
        
        create_directories()
        settings = get_settings()
        system = SQLAgentSystem(settings)
        
        # Check if schema file exists
        if not Path(TestConfig.SCHEMA_FILE).exists():
            print(f"❌ Schema file not found: {TestConfig.SCHEMA_FILE}")
            print(f"📝 Please update TestConfig.SCHEMA_FILE path in test_main.py")
            return
        
        await system.initialize(TestConfig.SCHEMA_FILE)
        
        print(f"\n🤖 SQL Agent System - Interactive Test Mode")
        print(f"{'='*60}")
        print(f"Commands:")
        print(f"  'status' - Show system status")
        print(f"  'test <query>' - Test a specific query")
        print(f"  'examples' - Show example queries")
        print(f"  'exit' - Quit")
        print(f"{'='*60}")
        
        while True:
            try:
                user_input = input(f"\n🧪 Test Command: ").strip()
                
                if user_input.lower() in ['exit', 'quit', 'q']:
                    break
                
                if user_input.lower() == 'status':
                    status = await system.get_system_status()
                    print(f"\n📊 System Status:")
                    for key, value in status.items():
                        print(f"   - {key}: {value}")
                    continue
                
                if user_input.lower() == 'examples':
                    print(f"\n📋 Example Queries:")
                    for i, example in enumerate(TestConfig.TEST_QUERIES, 1):
                        print(f"   {i}. {example['query']}")
                        print(f"      ({example['description']})")
                    continue
                
                if user_input.startswith('test '):
                    query = user_input[5:].strip()
                    if not query:
                        print("❌ Please provide a query after 'test'")
                        continue
                    
                    print(f"\n🔄 Testing query: {query}")
                    start_time = datetime.now()
                    
                    result = await system.process_query(query)
                    processing_time = format_duration(start_time)
                    
                    if result.get('success'):
                        print(f"✅ Success! ({processing_time})")
                        print(f"📝 SQL: {result.get('sql_query', '')[:200]}...")
                        print(f"📊 Chart: {result.get('chart_config', {}).get('chart_type', 'none')}")
                    else:
                        print(f"❌ Failed: {result.get('error', 'Unknown error')}")
                    continue
                
                if not user_input:
                    continue
                
                # Default: treat as query
                print(f"\n🔄 Processing: {user_input}")
                start_time = datetime.now()
                
                result = await system.process_query(user_input)
                processing_time = format_duration(start_time)
                
                if result.get('success'):
                    print(f"✅ Success! ({processing_time})")
                    sql_query = result.get('sql_query', '')
                    if sql_query:
                        print(f"📝 Generated SQL:")
                        print(f"   {sql_query}")
                    
                    chart_config = result.get('chart_config', {})
                    if chart_config:
                        print(f"📊 Chart Config: {chart_config}")
                else:
                    print(f"❌ Error: {result.get('error', 'Unknown error')}")
                
            except KeyboardInterrupt:
                print(f"\n\n👋 Exiting interactive mode...")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
        
        await system.cleanup()
        
    except Exception as e:
        print(f"❌ Interactive mode failed: {e}")

# ========================================
# MAIN TEST RUNNER
# ========================================

async def main():
    """Main test runner"""
    print(f"\n🧪 SQL AGENT SYSTEM - COMPREHENSIVE TEST SUITE")
    print(f"{'='*60}")
    print(f"📅 Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📁 Working Directory: {os.getcwd()}")
    print(f"📋 Schema File: {TestConfig.SCHEMA_FILE}")
    print(f"{'='*60}")
    
    # Create test directories
    create_test_directories()
    
    # Check command line arguments
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
        
        if mode == 'interactive':
            await interactive_test_mode()
            return
        elif mode == 'schema':
            await test_schema_loading()
            return
        elif mode == 'init':
            await test_system_initialization()
            return
    
    # Run full integration tests
    results = await test_end_to_end_integration()
    
    print(f"\n🏁 Test suite completed!")
    print(f"📊 Run 'python test_main.py interactive' for manual testing")

if __name__ == "__main__":
    """
    Test runner with multiple modes:
    
    python test_main.py                 # Run full integration tests
    python test_main.py interactive     # Interactive testing mode
    python test_main.py schema          # Test schema loading only
    python test_main.py init            # Test system initialization only
    """
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print(f"\n\n👋 Tests interrupted by user")
    except Exception as e:
        print(f"\n💥 Test suite failed: {e}")
        sys.exit(1)