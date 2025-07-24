"""
SQL feature extraction for building coverage system.

This module provides the original Codebase 1 SQL extraction functionality
for retrieving and processing claim data from various data sources.
"""

import pandas as pd
import pyodbc
import sqlalchemy
from typing import Dict, Any, List, Optional, Union
import logging
from datetime import datetime, timedelta
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)


class SQLExtractor:
    """
    SQL extraction engine for building coverage data.
    
    This class handles data extraction from multiple SQL databases
    using the original Codebase 1 extraction patterns and logic.
    """
    
    def __init__(
        self,
        credentials: Dict[str, Any],
        sql_queries: Dict[str, str],
        connection_config: Optional[Dict[str, Any]] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the SQL extractor.
        
        Args:
            credentials (Dict[str, Any]): Database credentials
            sql_queries (Dict[str, str]): SQL queries to execute
            connection_config (Optional[Dict[str, Any]]): Connection configuration
            logger (Optional[logging.Logger]): Logger instance
        """
        self.credentials = credentials
        self.sql_queries = sql_queries
        self.connection_config = connection_config or self._get_default_connection_config()
        self.logger = logger or logging.getLogger(__name__)
        
        # Connection pools for different data sources
        self.connections = {}
        self.engines = {}
        
        # Extraction statistics
        self.stats = {
            'total_extractions': 0,
            'successful_extractions': 0,
            'failed_extractions': 0,
            'total_rows_extracted': 0,
            'total_extraction_time': 0.0
        }
        
        self.logger.info("SQLExtractor initialized")
    
    def extract_building_coverage_features(
        self,
        source_filters: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Extract building coverage features from the main data source.
        
        Args:
            source_filters (Optional[Dict[str, Any]]): Additional filters to apply
            limit (Optional[int]): Maximum number of records to extract
            
        Returns:
            pd.DataFrame: Extracted features dataframe
        """
        self.logger.info("Starting building coverage feature extraction")
        start_time = time.time()
        
        try:
            # Get the main feature query
            base_query = self.sql_queries.get('feature_queries', {}).get('main_claims_query', '')
            if not base_query:
                raise ValueError("Main claims query not found in SQL queries")
            
            # Apply filters and limits
            query = self._apply_query_modifications(base_query, source_filters, limit)
            
            # Execute the query
            df = self._execute_query(query, 'primary')
            
            # Post-process the results
            df = self._post_process_features(df)
            
            # Update statistics
            extraction_time = time.time() - start_time
            self._update_stats(len(df), extraction_time, success=True)
            
            self.logger.info(f"Feature extraction completed: {len(df)} records in {extraction_time:.2f} seconds")
            
            return df
            
        except Exception as e:
            extraction_time = time.time() - start_time
            self._update_stats(0, extraction_time, success=False)
            self.logger.error(f"Feature extraction failed: {str(e)}")
            raise
    
    def extract_from_multiple_sources(
        self,
        sources: List[str],
        max_workers: int = 3
    ) -> Dict[str, pd.DataFrame]:
        """
        Extract data from multiple sources in parallel.
        
        Args:
            sources (List[str]): List of data source names
            max_workers (int): Maximum number of parallel workers
            
        Returns:
            Dict[str, pd.DataFrame]: Results from each source
        """
        self.logger.info(f"Starting parallel extraction from {len(sources)} sources")
        
        results = {}
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit extraction tasks
            future_to_source = {
                executor.submit(self._extract_from_source, source): source
                for source in sources
            }
            
            # Collect results
            for future in as_completed(future_to_source):
                source = future_to_source[future]
                
                try:
                    result_df = future.result()
                    results[source] = result_df
                    self.logger.info(f"Successfully extracted {len(result_df)} records from {source}")
                    
                except Exception as e:
                    self.logger.error(f"Failed to extract from {source}: {str(e)}")
                    results[source] = pd.DataFrame()  # Empty dataframe for failed extractions
        
        # Log summary
        total_records = sum(len(df) for df in results.values())
        self.logger.info(f"Parallel extraction completed: {total_records} total records from {len(sources)} sources")
        
        return results
    
    def _extract_from_source(self, source: str) -> pd.DataFrame:
        """
        Extract data from a specific source.
        
        Args:
            source (str): Data source name
            
        Returns:
            pd.DataFrame: Extracted data
        """
        query_key = f'{source}_claims_query'
        query = self.sql_queries.get('feature_queries', {}).get(query_key)
        
        if not query:
            self.logger.warning(f"No query found for source {source}")
            return pd.DataFrame()
        
        try:
            df = self._execute_query(query, source)
            return self._post_process_features(df)
            
        except Exception as e:
            self.logger.error(f"Error extracting from {source}: {str(e)}")
            return pd.DataFrame()
    
    def _execute_query(
        self,
        query: str,
        source: str = 'primary',
        timeout: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Execute a SQL query against the specified data source.
        
        Args:
            query (str): SQL query to execute
            source (str): Data source name
            timeout (Optional[int]): Query timeout in seconds
            
        Returns:
            pd.DataFrame: Query results
        """
        self.logger.debug(f"Executing query against {source} source")
        
        # Get or create connection
        engine = self._get_engine(source)
        
        # Set timeout
        query_timeout = timeout or self.connection_config.get('query_timeout', 300)
        
        try:
            # Execute query with pandas
            df = pd.read_sql_query(
                query,
                engine,
                params=None,
                chunksize=None
            )
            
            self.logger.debug(f"Query executed successfully: {len(df)} rows returned")
            return df
            
        except Exception as e:
            self.logger.error(f"Query execution failed: {str(e)}")
            raise
    
    def _get_engine(self, source: str) -> sqlalchemy.engine.Engine:
        """
        Get or create a SQLAlchemy engine for the specified source.
        
        Args:
            source (str): Data source name
            
        Returns:
            sqlalchemy.engine.Engine: Database engine
        """
        if source not in self.engines:
            connection_string = self._build_connection_string(source)
            
            # Create engine with connection pooling
            self.engines[source] = sqlalchemy.create_engine(
                connection_string,
                pool_size=self.connection_config.get('pool_size', 5),
                max_overflow=self.connection_config.get('max_overflow', 10),
                pool_timeout=self.connection_config.get('pool_timeout', 30),
                pool_recycle=self.connection_config.get('pool_recycle', 3600),
                echo=self.connection_config.get('echo_sql', False)
            )
            
            self.logger.debug(f"Created SQLAlchemy engine for {source}")
        
        return self.engines[source]
    
    def _build_connection_string(self, source: str) -> str:
        """
        Build connection string for the specified source.
        
        Args:
            source (str): Data source name
            
        Returns:
            str: Connection string
        """
        if source == 'primary':
            server = self.credentials['server']
            database = self.credentials['database']
            username = self.credentials.get('username', '')
            password = self.credentials.get('password', '')
            driver = self.credentials.get('driver', 'ODBC Driver 17 for SQL Server')
        elif source == 'aip':
            server = self.credentials.get('aip_server', '')
            database = self.credentials.get('aip_database', '')
            username = self.credentials.get('aip_username', '')
            password = self.credentials.get('aip_password', '')
            driver = self.credentials.get('driver', 'ODBC Driver 17 for SQL Server')
        elif source == 'atlas':
            server = self.credentials.get('atlas_server', '')
            database = self.credentials.get('atlas_database', '')
            username = self.credentials.get('atlas_username', '')
            password = self.credentials.get('atlas_password', '')
            driver = self.credentials.get('driver', 'ODBC Driver 17 for SQL Server')
        else:
            raise ValueError(f"Unknown data source: {source}")
        
        if not server or not database:
            raise ValueError(f"Missing connection information for source: {source}")
        
        # Build SQL Server connection string
        if username and password:
            conn_str = f"mssql+pyodbc://{username}:{password}@{server}/{database}?driver={driver.replace(' ', '+')}"
        else:
            conn_str = f"mssql+pyodbc://@{server}/{database}?driver={driver.replace(' ', '+')}&trusted_connection=yes"
        
        return conn_str
    
    def _apply_query_modifications(
        self,
        base_query: str,
        filters: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None
    ) -> str:
        """
        Apply filters and modifications to a base query.
        
        Args:
            base_query (str): Base SQL query
            filters (Optional[Dict[str, Any]]): Additional filters
            limit (Optional[int]): Row limit
            
        Returns:
            str: Modified query
        """
        query = base_query
        
        # Apply filters
        if filters:
            additional_conditions = []
            
            if 'lob_codes' in filters:
                lob_list = "', '".join(filters['lob_codes'])
                additional_conditions.append(f"LOBCD IN ('{lob_list}')")
            
            if 'date_range' in filters:
                start_date, end_date = filters['date_range']
                additional_conditions.append(f"LOSSDT BETWEEN '{start_date}' AND '{end_date}'")
            
            if 'min_text_length' in filters:
                min_length = filters['min_text_length']
                additional_conditions.append(f"LEN(clean_FN_TEXT) >= {min_length}")
            
            # Add conditions to WHERE clause
            if additional_conditions:
                if 'WHERE' in query.upper():
                    query += " AND " + " AND ".join(additional_conditions)
                else:
                    query += " WHERE " + " AND ".join(additional_conditions)
        
        # Apply limit
        if limit:
            if 'TOP' not in query.upper():
                query = query.replace('SELECT', f'SELECT TOP {limit}', 1)
        
        return query
    
    def _post_process_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Post-process extracted features.
        
        Args:
            df (pd.DataFrame): Raw extracted data
            
        Returns:
            pd.DataFrame: Post-processed data
        """
        if df.empty:
            return df
        
        # Standardize column names
        df = self._standardize_column_names(df)
        
        # Clean text data
        if 'clean_FN_TEXT' in df.columns:
            df['clean_FN_TEXT'] = df['clean_FN_TEXT'].astype(str).str.strip()
            df = df[df['clean_FN_TEXT'] != '']
        
        # Convert date columns
        date_columns = ['LOSSDT', 'REPORTEDDT']
        for col in date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
        
        # Convert numeric columns
        numeric_columns = ['RESERVE_TOTAL', 'PAID_TOTAL', 'INCURRED_TOTAL']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Add extraction metadata
        df['extraction_timestamp'] = datetime.now()
        df['data_source'] = 'sql_extract'
        
        # Remove duplicates
        if 'CLAIMNO' in df.columns:
            df = df.drop_duplicates(subset=['CLAIMNO'], keep='first')
        
        self.logger.debug(f"Post-processing completed: {len(df)} records")
        
        return df
    
    def _standardize_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize column names across different data sources.
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            pd.DataFrame: Dataframe with standardized column names
        """
        # Column name mappings for different sources
        column_mappings = {
            'CLAIM_DESC': 'clean_FN_TEXT',
            'CLAIM_DESCRIPTION': 'clean_FN_TEXT',
            'DESCRIPTION': 'clean_FN_TEXT',
            'LINE_OF_BUSINESS': 'LOBCD',
            'LOB_CODE': 'LOBCD',
            'LOSS_DESCRIPTION': 'LOSSDESC',
            'LOSS_CAUSE': 'LOSSDESC',
            'LOSS_DATE': 'LOSSDT',
            'DATE_OF_LOSS': 'LOSSDT',
            'REPORTED_DATE': 'REPORTEDDT',
            'DATE_REPORTED': 'REPORTEDDT',
            'CLAIM_NUMBER': 'CLAIMNO',
            'CLAIM_ID': 'CLAIMKEY',
            'RESERVE_AMOUNT': 'RESERVE_TOTAL',
            'OUTSTANDING_RESERVE': 'RESERVE_TOTAL',
            'STATUS': 'STATUSCD',
            'CLAIM_STATUS': 'STATUSCD'
        }
        
        # Apply mappings
        df = df.rename(columns=column_mappings)
        
        return df
    
    def _update_stats(self, row_count: int, extraction_time: float, success: bool):
        """
        Update extraction statistics.
        
        Args:
            row_count (int): Number of rows extracted
            extraction_time (float): Time taken for extraction
            success (bool): Whether extraction was successful
        """
        self.stats['total_extractions'] += 1
        self.stats['total_extraction_time'] += extraction_time
        
        if success:
            self.stats['successful_extractions'] += 1
            self.stats['total_rows_extracted'] += row_count
        else:
            self.stats['failed_extractions'] += 1
    
    def _get_default_connection_config(self) -> Dict[str, Any]:
        """
        Get default connection configuration.
        
        Returns:
            Dict[str, Any]: Default configuration
        """
        return {
            'pool_size': 5,
            'max_overflow': 10,
            'pool_timeout': 30,
            'pool_recycle': 3600,
            'query_timeout': 300,
            'echo_sql': False,
            'isolation_level': 'READ_COMMITTED'
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get extraction statistics.
        
        Returns:
            Dict[str, Any]: Current statistics
        """
        stats = self.stats.copy()
        
        if stats['total_extractions'] > 0:
            stats['success_rate'] = stats['successful_extractions'] / stats['total_extractions']
            stats['avg_extraction_time'] = stats['total_extraction_time'] / stats['total_extractions']
        else:
            stats['success_rate'] = 0.0
            stats['avg_extraction_time'] = 0.0
        
        if stats['successful_extractions'] > 0:
            stats['avg_rows_per_extraction'] = stats['total_rows_extracted'] / stats['successful_extractions']
        else:
            stats['avg_rows_per_extraction'] = 0.0
        
        return stats
    
    def test_connections(self) -> Dict[str, Dict[str, Any]]:
        """
        Test connections to all configured data sources.
        
        Returns:
            Dict[str, Dict[str, Any]]: Connection test results
        """
        sources = ['primary']
        
        # Add optional sources if configured
        if self.credentials.get('aip_server'):
            sources.append('aip')
        if self.credentials.get('atlas_server'):
            sources.append('atlas')
        
        results = {}
        
        for source in sources:
            result = {
                'source': source,
                'success': False,
                'response_time': 0.0,
                'error': None
            }
            
            try:
                start_time = time.time()
                
                # Simple test query
                test_query = "SELECT 1 as test_column"
                test_df = self._execute_query(test_query, source)
                
                result['success'] = len(test_df) > 0
                result['response_time'] = time.time() - start_time
                
                self.logger.info(f"Connection test successful for {source}")
                
            except Exception as e:
                result['error'] = str(e)
                self.logger.error(f"Connection test failed for {source}: {str(e)}")
            
            results[source] = result
        
        return results
    
    def close_connections(self):
        """
        Close all database connections.
        """
        for source, engine in self.engines.items():
            try:
                engine.dispose()
                self.logger.debug(f"Closed connection for {source}")
            except Exception as e:
                self.logger.warning(f"Error closing connection for {source}: {str(e)}")
        
        self.engines.clear()
        self.connections.clear()
        
        self.logger.info("All database connections closed")
    
    def __del__(self):
        """
        Cleanup connections when object is destroyed.
        """
        try:
            self.close_connections()
        except:
            pass  # Ignore errors during cleanup


def create_sql_extractor(
    credentials: Dict[str, Any],
    sql_queries: Dict[str, str],
    connection_config: Optional[Dict[str, Any]] = None,
    logger: Optional[logging.Logger] = None
) -> SQLExtractor:
    """
    Factory function to create a SQL extractor instance.
    
    Args:
        credentials (Dict[str, Any]): Database credentials
        sql_queries (Dict[str, str]): SQL queries
        connection_config (Optional[Dict[str, Any]]): Connection configuration
        logger (Optional[logging.Logger]): Logger instance
        
    Returns:
        SQLExtractor: Configured SQL extractor instance
    """
    return SQLExtractor(
        credentials=credentials,
        sql_queries=sql_queries,
        connection_config=connection_config,
        logger=logger
    )