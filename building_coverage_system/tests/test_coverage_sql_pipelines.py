"""
Tests for coverage_sql_pipelines module.

This module contains unit tests for SQL extraction and data pulling
functionality from the original Codebase 1.
"""

import pytest
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
from building_coverage_system.coverage_sql_pipelines.src.sql_extract import (
    SQLExtractor,
    create_sql_extractor
)
from building_coverage_system.coverage_sql_pipelines.src.data_pull import (
    DataPuller,
    create_data_puller
)
from building_coverage_system.tests.fixtures.sample_data import (
    create_sample_claims_data,
    create_sample_credentials,
    create_sample_sql_queries
)


class TestSQLExtractor:
    """Test cases for SQLExtractor class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.sample_credentials = create_sample_credentials()
        self.sample_queries = create_sample_sql_queries()
        
        # Mock SQLAlchemy engine creation
        with patch('building_coverage_system.coverage_sql_pipelines.src.sql_extract.sqlalchemy.create_engine') as mock_create_engine:
            self.mock_engine = Mock()
            mock_create_engine.return_value = self.mock_engine
            
            self.extractor = SQLExtractor(
                credentials=self.sample_credentials,
                sql_queries=self.sample_queries
            )
    
    def test_extractor_initialization(self):
        """Test SQL extractor initialization."""
        assert self.extractor.credentials == self.sample_credentials
        assert self.extractor.sql_queries == self.sample_queries
        assert self.extractor.stats['total_extractions'] == 0
    
    @patch('building_coverage_system.coverage_sql_pipelines.src.sql_extract.pd.read_sql_query')
    def test_extract_building_coverage_features(self, mock_read_sql):
        """Test building coverage feature extraction."""
        # Setup mock data
        sample_data = create_sample_claims_data(100)
        mock_read_sql.return_value = sample_data
        
        # Mock the _get_engine method
        with patch.object(self.extractor, '_get_engine', return_value=self.mock_engine):
            result_df = self.extractor.extract_building_coverage_features()
        
        assert isinstance(result_df, pd.DataFrame)
        assert len(result_df) > 0
        assert 'extraction_timestamp' in result_df.columns
        assert 'data_source' in result_df.columns
        
        # Check statistics were updated
        assert self.extractor.stats['total_extractions'] == 1
        assert self.extractor.stats['successful_extractions'] == 1
    
    @patch('building_coverage_system.coverage_sql_pipelines.src.sql_extract.ThreadPoolExecutor')
    @patch('building_coverage_system.coverage_sql_pipelines.src.sql_extract.pd.read_sql_query')
    def test_extract_from_multiple_sources(self, mock_read_sql, mock_executor):
        """Test extraction from multiple data sources."""
        # Setup mock data
        sample_data = create_sample_claims_data(50)
        mock_read_sql.return_value = sample_data
        
        # Setup mock executor
        mock_future = Mock()
        mock_future.result.return_value = sample_data
        
        mock_executor_instance = Mock()
        mock_executor_instance.submit.return_value = mock_future
        mock_executor_instance.__enter__.return_value = mock_executor_instance
        mock_executor_instance.__exit__.return_value = None
        mock_executor.return_value = mock_executor_instance
        
        # Mock as_completed to return the future
        with patch('building_coverage_system.coverage_sql_pipelines.src.sql_extract.as_completed', return_value=[mock_future]), \
             patch.object(self.extractor, '_get_engine', return_value=self.mock_engine), \
             patch.object(self.extractor, '_extract_from_source', return_value=sample_data):
            
            sources = ['primary', 'aip']
            results = self.extractor.extract_from_multiple_sources(sources)
        
        assert isinstance(results, dict)
        assert len(results) == len(sources)
        for source in sources:
            assert source in results
            assert isinstance(results[source], pd.DataFrame)
    
    def test_build_connection_string(self):
        """Test connection string building."""
        # Test primary source
        conn_str = self.extractor._build_connection_string('primary')
        
        assert 'mssql+pyodbc://' in conn_str
        assert self.sample_credentials['server'] in conn_str
        assert self.sample_credentials['database'] in conn_str
        
        # Test with username/password
        assert self.sample_credentials['username'] in conn_str
        assert self.sample_credentials['password'] in conn_str
        
        # Test AIP source
        conn_str_aip = self.extractor._build_connection_string('aip')
        assert self.sample_credentials['aip_server'] in conn_str_aip
        
        # Test unknown source
        with pytest.raises(ValueError):
            self.extractor._build_connection_string('unknown_source')
    
    def test_apply_query_modifications(self):
        """Test query modification functionality."""
        base_query = "SELECT * FROM claims_data"
        
        # Test with filters
        filters = {
            'lob_codes': ['HO', 'CO'],
            'date_range': ('2023-01-01', '2023-12-31'),
            'min_text_length': 100
        }
        
        modified_query = self.extractor._apply_query_modifications(base_query, filters, limit=1000)
        
        assert "LOBCD IN ('HO', 'CO')" in modified_query
        assert "LOSSDT BETWEEN '2023-01-01' AND '2023-12-31'" in modified_query
        assert "LEN(clean_FN_TEXT) >= 100" in modified_query
        assert "TOP 1000" in modified_query
    
    def test_post_process_features(self):
        """Test feature post-processing."""
        # Create test data with various data types
        test_df = pd.DataFrame({
            'CLAIMNO': ['CLM001', 'CLM002', 'CLM001'],  # Duplicate for testing
            'clean_FN_TEXT': ['  Fire damage  ', 'Water damage', ''],
            'LOSSDT': ['2023-01-15', '2023-02-20', '2023-03-25'],
            'RESERVE_TOTAL': ['1000.50', '2000.75', None],
            'PAID_TOTAL': [500, 1000, 1500]
        })
        
        processed_df = self.extractor._post_process_features(test_df)
        
        # Check text cleaning
        assert processed_df['clean_FN_TEXT'].iloc[0] == 'Fire damage'
        
        # Check date conversion
        assert pd.api.types.is_datetime64_any_dtype(processed_df['LOSSDT'])
        
        # Check numeric conversion
        assert pd.api.types.is_numeric_dtype(processed_df['RESERVE_TOTAL'])
        
        # Check duplicate removal
        assert len(processed_df) == 2  # One duplicate should be removed
        
        # Check metadata addition
        assert 'extraction_timestamp' in processed_df.columns
        assert 'data_source' in processed_df.columns
    
    def test_standardize_column_names(self):
        """Test column name standardization."""
        test_df = pd.DataFrame({
            'CLAIM_DESC': ['Fire damage', 'Water damage'],
            'LINE_OF_BUSINESS': ['HO', 'CO'],
            'LOSS_DATE': ['2023-01-01', '2023-02-01'],
            'CLAIM_NUMBER': ['CLM001', 'CLM002']
        })
        
        standardized_df = self.extractor._standardize_column_names(test_df)
        
        # Check that columns were renamed
        assert 'clean_FN_TEXT' in standardized_df.columns
        assert 'LOBCD' in standardized_df.columns
        assert 'LOSSDT' in standardized_df.columns
        assert 'CLAIMNO' in standardized_df.columns
        
        # Check that old names are gone
        assert 'CLAIM_DESC' not in standardized_df.columns
        assert 'LINE_OF_BUSINESS' not in standardized_df.columns
    
    @patch('building_coverage_system.coverage_sql_pipelines.src.sql_extract.pd.read_sql_query')
    def test_test_connections(self, mock_read_sql):
        """Test connection testing functionality."""
        # Mock successful connection
        mock_read_sql.return_value = pd.DataFrame({'test_column': [1]})
        
        with patch.object(self.extractor, '_get_engine', return_value=self.mock_engine):
            results = self.extractor.test_connections()
        
        assert isinstance(results, dict)
        assert 'primary' in results
        
        # Check result structure
        for source, result in results.items():
            assert 'source' in result
            assert 'success' in result
            assert 'response_time' in result
    
    def test_get_statistics(self):
        """Test statistics retrieval."""
        # Update some stats manually for testing
        self.extractor.stats['total_extractions'] = 10
        self.extractor.stats['successful_extractions'] = 8
        self.extractor.stats['total_extraction_time'] = 100.0
        self.extractor.stats['total_rows_extracted'] = 1000
        
        stats = self.extractor.get_statistics()
        
        assert stats['total_extractions'] == 10
        assert stats['success_rate'] == 0.8
        assert stats['avg_extraction_time'] == 10.0
        assert stats['avg_rows_per_extraction'] == 125.0
    
    def test_close_connections(self):
        """Test connection cleanup."""
        # Add a mock engine to test cleanup
        mock_engine = Mock()
        self.extractor.engines['test_source'] = mock_engine
        
        self.extractor.close_connections()
        
        # Verify engine disposal was called
        mock_engine.dispose.assert_called_once()
        
        # Verify engines dict was cleared
        assert len(self.extractor.engines) == 0


class TestDataPuller:
    """Test cases for DataPuller class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.sample_credentials = create_sample_credentials()
        self.data_puller = DataPuller(self.sample_credentials)
    
    def test_data_puller_initialization(self):
        """Test data puller initialization."""
        assert self.data_puller.credentials == self.sample_credentials
        assert self.data_puller.stats['total_queries'] == 0
        assert isinstance(self.data_puller.connection_pools, dict)
    
    @patch('building_coverage_system.coverage_sql_pipelines.src.data_pull.pd.read_sql_query')
    @patch('building_coverage_system.coverage_sql_pipelines.src.data_pull.pyodbc.connect')
    def test_pull_data(self, mock_connect, mock_read_sql):
        """Test basic data pulling functionality."""
        # Setup mocks
        mock_conn = Mock()
        mock_connect.return_value = mock_conn
        
        sample_data = create_sample_claims_data(100)
        mock_read_sql.return_value = sample_data
        
        query = "SELECT * FROM claims WHERE status = 'O'"
        
        result_df = self.data_puller.pull_data(query, source='primary')
        
        assert isinstance(result_df, pd.DataFrame)
        assert len(result_df) == len(sample_data)
        assert self.data_puller.stats['total_queries'] == 1
        assert self.data_puller.stats['successful_queries'] == 1
    
    @patch('building_coverage_system.coverage_sql_pipelines.src.data_pull.pd.read_sql_query')
    @patch('building_coverage_system.coverage_sql_pipelines.src.data_pull.pyodbc.connect')
    def test_pull_data_with_retry(self, mock_connect, mock_read_sql):
        """Test data pulling with retry logic."""
        # Setup mocks for failure then success
        mock_conn = Mock()
        mock_connect.return_value = mock_conn
        
        sample_data = create_sample_claims_data(50)
        
        # First call fails, second succeeds
        mock_read_sql.side_effect = [Exception("Connection error"), sample_data]
        
        with patch.object(self.data_puller, 'pull_data', side_effect=[Exception("Connection error"), sample_data]):
            query = "SELECT * FROM claims"
            
            result_df = self.data_puller.pull_data_with_retry(query, max_retries=2)
        
        assert isinstance(result_df, pd.DataFrame)
        assert len(result_df) == len(sample_data)
        assert self.data_puller.stats['retry_attempts'] > 0
    
    @patch('building_coverage_system.coverage_sql_pipelines.src.data_pull.ThreadPoolExecutor')
    def test_pull_data_parallel(self, mock_executor):
        """Test parallel data pulling."""
        # Setup mock executor
        sample_data1 = create_sample_claims_data(30)
        sample_data2 = create_sample_claims_data(40)
        
        mock_future1 = Mock()
        mock_future1.result.return_value = sample_data1
        mock_future2 = Mock()
        mock_future2.result.return_value = sample_data2
        
        mock_executor_instance = Mock()
        mock_executor_instance.submit.side_effect = [mock_future1, mock_future2]
        mock_executor_instance.__enter__.return_value = mock_executor_instance
        mock_executor_instance.__exit__.return_value = None
        mock_executor.return_value = mock_executor_instance
        
        # Mock as_completed
        with patch('building_coverage_system.coverage_sql_pipelines.src.data_pull.as_completed', 
                  return_value=[mock_future1, mock_future2]):
            
            queries = [
                ("SELECT * FROM table1", "primary"),
                ("SELECT * FROM table2", "aip")
            ]
            
            with patch.object(self.data_puller, 'pull_data_with_retry', side_effect=[sample_data1, sample_data2]):
                results = self.data_puller.pull_data_parallel(queries)
        
        assert len(results) == 2
        assert isinstance(results[0], pd.DataFrame)
        assert isinstance(results[1], pd.DataFrame)
        assert len(results[0]) == len(sample_data1)
        assert len(results[1]) == len(sample_data2)
    
    def test_build_connection_string(self):
        """Test ODBC connection string building."""
        # Test primary source
        conn_str = self.data_puller._build_connection_string('primary')
        
        assert 'DRIVER={ODBC Driver 17 for SQL Server}' in conn_str
        assert f"SERVER={self.sample_credentials['server']}" in conn_str
        assert f"DATABASE={self.sample_credentials['database']}" in conn_str
        assert f"UID={self.sample_credentials['username']}" in conn_str
        assert f"PWD={self.sample_credentials['password']}" in conn_str
        assert 'Encrypt=yes' in conn_str
        
        # Test AIP source
        conn_str_aip = self.data_puller._build_connection_string('aip')
        assert self.sample_credentials['aip_server'] in conn_str_aip
        
        # Test unknown source
        with pytest.raises(ValueError):
            self.data_puller._build_connection_string('unknown_source')
    
    @patch('building_coverage_system.coverage_sql_pipelines.src.data_pull.pyodbc.connect')
    def test_execute_non_query(self, mock_connect):
        """Test non-query execution (INSERT, UPDATE, DELETE)."""
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_cursor.rowcount = 5
        mock_conn.cursor.return_value = mock_cursor
        mock_connect.return_value = mock_conn
        
        query = "UPDATE claims SET status = 'C' WHERE id = 123"
        
        with patch.object(self.data_puller, '_get_connection') as mock_get_conn:
            mock_get_conn.return_value.__enter__.return_value = mock_conn
            mock_get_conn.return_value.__exit__.return_value = None
            
            rows_affected = self.data_puller.execute_non_query(query)
        
        assert rows_affected == 5
        mock_cursor.execute.assert_called_once_with(query)
        mock_conn.commit.assert_called_once()
    
    @patch('building_coverage_system.coverage_sql_pipelines.src.data_pull.pyodbc.connect')
    def test_test_connection(self, mock_connect):
        """Test connection testing functionality."""
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_cursor.fetchone.return_value = ('SQL Server 2019', '2023-01-01 10:00:00')
        mock_conn.cursor.return_value = mock_cursor
        mock_connect.return_value = mock_conn
        
        with patch.object(self.data_puller, '_get_connection') as mock_get_conn:
            mock_get_conn.return_value.__enter__.return_value = mock_conn
            mock_get_conn.return_value.__exit__.return_value = None
            
            result = self.data_puller.test_connection('primary')
        
        assert result['success'] is True
        assert result['source'] == 'primary'
        assert 'response_time' in result
        assert 'server_info' in result
        assert result['server_info']['version'] == 'SQL Server 2019'
    
    @patch('building_coverage_system.coverage_sql_pipelines.src.data_pull.pd.read_sql_query')
    @patch('building_coverage_system.coverage_sql_pipelines.src.data_pull.pyodbc.connect')
    def test_get_table_info(self, mock_connect, mock_read_sql):
        """Test table information retrieval."""
        mock_conn = Mock()
        mock_connect.return_value = mock_conn
        
        # Mock responses for row count and column info queries
        row_count_df = pd.DataFrame({'row_count': [1000]})
        columns_df = pd.DataFrame({
            'COLUMN_NAME': ['ID', 'NAME', 'AMOUNT'],
            'DATA_TYPE': ['int', 'varchar', 'decimal'],
            'IS_NULLABLE': ['NO', 'YES', 'YES'],
            'CHARACTER_MAXIMUM_LENGTH': [None, 50, None]
        })
        
        mock_read_sql.side_effect = [row_count_df, columns_df]
        
        with patch.object(self.data_puller, 'pull_data', side_effect=[row_count_df, columns_df]):
            table_info = self.data_puller.get_table_info('test_table')
        
        assert table_info['table_name'] == 'test_table'
        assert table_info['row_count'] == 1000
        assert table_info['column_count'] == 3
        assert len(table_info['columns']) == 3
    
    @patch('building_coverage_system.coverage_sql_pipelines.src.data_pull.sqlalchemy.create_engine')
    def test_bulk_insert(self, mock_create_engine):
        """Test bulk insert functionality."""
        mock_engine = Mock()
        mock_create_engine.return_value = mock_engine
        
        test_df = create_sample_claims_data(100)
        
        with patch.object(test_df, 'to_sql') as mock_to_sql:
            rows_inserted = self.data_puller.bulk_insert(
                test_df, 
                'test_table', 
                batch_size=50
            )
        
        assert rows_inserted == len(test_df)
        
        # Should call to_sql twice for batch_size=50 with 100 rows
        assert mock_to_sql.call_count == 2
        mock_engine.dispose.assert_called_once()
    
    def test_get_statistics(self):
        """Test statistics retrieval."""
        # Update some stats manually for testing
        self.data_puller.stats['total_queries'] = 20
        self.data_puller.stats['successful_queries'] = 18
        self.data_puller.stats['total_query_time'] = 100.0
        self.data_puller.stats['total_rows_fetched'] = 5000
        
        stats = self.data_puller.get_statistics()
        
        assert stats['total_queries'] == 20
        assert stats['success_rate'] == 0.9
        assert stats['avg_query_time'] == 5.0
        assert stats['avg_rows_per_query'] == 5000 / 18  # Only successful queries
    
    def test_reset_statistics(self):
        """Test statistics reset functionality."""
        # Set some stats
        self.data_puller.stats['total_queries'] = 10
        self.data_puller.stats['successful_queries'] = 8
        
        # Reset
        self.data_puller.reset_statistics()
        
        # Verify reset
        assert self.data_puller.stats['total_queries'] == 0
        assert self.data_puller.stats['successful_queries'] == 0
        assert self.data_puller.stats['total_rows_fetched'] == 0


class TestFactoryFunctions:
    """Test cases for factory functions."""
    
    def test_create_sql_extractor(self):
        """Test SQL extractor factory function."""
        credentials = create_sample_credentials()
        queries = create_sample_sql_queries()
        
        with patch('building_coverage_system.coverage_sql_pipelines.src.sql_extract.sqlalchemy.create_engine'):
            extractor = create_sql_extractor(credentials, queries)
        
        assert isinstance(extractor, SQLExtractor)
        assert extractor.credentials == credentials
        assert extractor.sql_queries == queries
    
    def test_create_data_puller(self):
        """Test data puller factory function."""
        credentials = create_sample_credentials()
        
        puller = create_data_puller(credentials)
        
        assert isinstance(puller, DataPuller)
        assert puller.credentials == credentials


class TestIntegrationScenarios:
    """Integration test scenarios for SQL pipelines."""
    
    @patch('building_coverage_system.coverage_sql_pipelines.src.sql_extract.sqlalchemy.create_engine')
    @patch('building_coverage_system.coverage_sql_pipelines.src.sql_extract.pd.read_sql_query')
    def test_full_extraction_workflow(self, mock_read_sql, mock_create_engine):
        """Test complete extraction workflow."""
        # Setup mocks
        mock_engine = Mock()
        mock_create_engine.return_value = mock_engine
        
        sample_data = create_sample_claims_data(200)
        mock_read_sql.return_value = sample_data
        
        # Create extractor
        credentials = create_sample_credentials()
        queries = create_sample_sql_queries()
        extractor = SQLExtractor(credentials, queries)
        
        # Test extraction with filters
        filters = {
            'lob_codes': ['HO', 'CO'],
            'date_range': ('2023-01-01', '2023-12-31'),
            'min_text_length': 50
        }
        
        with patch.object(extractor, '_get_engine', return_value=mock_engine):
            result_df = extractor.extract_building_coverage_features(
                source_filters=filters,
                limit=1000
            )
        
        # Verify results
        assert isinstance(result_df, pd.DataFrame)
        assert len(result_df) > 0
        
        # Check post-processing was applied
        assert 'extraction_timestamp' in result_df.columns
        assert 'data_source' in result_df.columns
        
        # Verify statistics
        stats = extractor.get_statistics()
        assert stats['total_extractions'] == 1
        assert stats['successful_extractions'] == 1
    
    @patch('building_coverage_system.coverage_sql_pipelines.src.data_pull.pyodbc.connect')
    @patch('building_coverage_system.coverage_sql_pipelines.src.data_pull.pd.read_sql_query')
    def test_data_puller_error_handling(self, mock_read_sql, mock_connect):
        """Test data puller error handling and recovery."""
        credentials = create_sample_credentials()
        puller = DataPuller(credentials)
        
        # Test connection failure followed by success
        mock_conn = Mock()
        mock_connect.side_effect = [Exception("Connection failed"), mock_conn]
        
        sample_data = create_sample_claims_data(50)
        mock_read_sql.return_value = sample_data
        
        query = "SELECT * FROM claims"
        
        # Should succeed on retry
        with patch.object(puller, 'pull_data', side_effect=[Exception("Connection failed"), sample_data]):
            result_df = puller.pull_data_with_retry(query, max_retries=2)
        
        assert isinstance(result_df, pd.DataFrame)
        assert len(result_df) == len(sample_data)
        
        # Check retry statistics
        assert puller.stats['retry_attempts'] > 0
    
    def test_performance_monitoring(self):
        """Test performance monitoring across both classes."""
        credentials = create_sample_credentials()
        queries = create_sample_sql_queries()
        
        # Test extractor performance tracking
        with patch('building_coverage_system.coverage_sql_pipelines.src.sql_extract.sqlalchemy.create_engine'):
            extractor = SQLExtractor(credentials, queries)
        
        # Manually update stats to simulate operations
        extractor._update_stats(100, 5.0, True)
        extractor._update_stats(200, 3.0, True)
        extractor._update_stats(0, 2.0, False)
        
        extractor_stats = extractor.get_statistics()
        assert extractor_stats['total_extractions'] == 3
        assert extractor_stats['successful_extractions'] == 2
        assert extractor_stats['avg_extraction_time'] == (5.0 + 3.0 + 2.0) / 3
        
        # Test data puller performance tracking
        puller = DataPuller(credentials)
        
        puller._update_stats(50, 2.0, True)
        puller._update_stats(75, 4.0, True)
        puller._update_stats(0, 1.0, False)
        
        puller_stats = puller.get_statistics()
        assert puller_stats['total_queries'] == 3
        assert puller_stats['successful_queries'] == 2
        assert puller_stats['success_rate'] == 2/3