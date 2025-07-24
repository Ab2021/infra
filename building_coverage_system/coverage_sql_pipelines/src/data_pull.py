"""
Database connectivity and data pulling utilities.

This module provides database connection management and data pulling
functionalities for the original Codebase 1 SQL pipeline components.
"""

import pandas as pd
import pyodbc
import sqlalchemy
from typing import Dict, Any, List, Optional, Union, Tuple
import logging
from datetime import datetime, timedelta
import time
import threading
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class DataPuller:
    """
    Database connectivity and data pulling manager.
    
    This class provides robust database connection management and
    data extraction capabilities with connection pooling, retry logic,
    and error handling.
    """
    
    def __init__(
        self,
        credentials: Dict[str, Any],
        connection_config: Optional[Dict[str, Any]] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the data puller.
        
        Args:
            credentials (Dict[str, Any]): Database credentials
            connection_config (Optional[Dict[str, Any]]): Connection configuration
            logger (Optional[logging.Logger]): Logger instance
        """
        self.credentials = credentials
        self.connection_config = connection_config or self._get_default_config()
        self.logger = logger or logging.getLogger(__name__)
        
        # Connection management
        self.connection_pools = {}
        self.active_connections = {}
        self.connection_lock = threading.Lock()
        
        # Statistics
        self.stats = {
            'total_queries': 0,
            'successful_queries': 0,
            'failed_queries': 0,
            'total_rows_fetched': 0,
            'total_query_time': 0.0,
            'connection_errors': 0,
            'retry_attempts': 0
        }
        
        self.logger.info("DataPuller initialized")
    
    def pull_data(
        self,
        query: str,
        source: str = 'primary',
        params: Optional[Dict[str, Any]] = None,
        chunk_size: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Pull data using the specified query.
        
        Args:
            query (str): SQL query to execute
            source (str): Data source name
            params (Optional[Dict[str, Any]]): Query parameters
            chunk_size (Optional[int]): Chunk size for large result sets
            
        Returns:
            pd.DataFrame: Query results
        """
        self.logger.debug(f"Pulling data from {source} with query length {len(query)}")
        start_time = time.time()
        
        try:
            with self._get_connection(source) as conn:
                if chunk_size:
                    df = self._pull_data_chunked(query, conn, params, chunk_size)
                else:
                    df = self._pull_data_direct(query, conn, params)
                
                # Update statistics
                query_time = time.time() - start_time
                self._update_stats(len(df), query_time, success=True)
                
                self.logger.debug(f"Data pull completed: {len(df)} rows in {query_time:.2f} seconds")
                return df
                
        except Exception as e:
            query_time = time.time() - start_time
            self._update_stats(0, query_time, success=False)
            self.logger.error(f"Data pull failed: {str(e)}")
            raise
    
    def pull_data_with_retry(
        self,
        query: str,
        source: str = 'primary',
        params: Optional[Dict[str, Any]] = None,
        max_retries: int = 3,
        retry_delay: float = 2.0
    ) -> pd.DataFrame:
        """
        Pull data with automatic retry on failure.
        
        Args:
            query (str): SQL query to execute
            source (str): Data source name
            params (Optional[Dict[str, Any]]): Query parameters
            max_retries (int): Maximum number of retry attempts
            retry_delay (float): Delay between retries in seconds
            
        Returns:
            pd.DataFrame: Query results
        """
        last_exception = None
        
        for attempt in range(max_retries + 1):
            try:
                if attempt > 0:
                    self.logger.info(f"Retry attempt {attempt} for data pull from {source}")
                    time.sleep(retry_delay * attempt)  # Exponential backoff
                    self.stats['retry_attempts'] += 1
                
                return self.pull_data(query, source, params)
                
            except Exception as e:
                last_exception = e
                self.logger.warning(f"Data pull attempt {attempt + 1} failed: {str(e)}")
                
                if attempt == max_retries:
                    self.logger.error(f"All {max_retries + 1} attempts failed for data pull from {source}")
                    raise last_exception
        
        raise last_exception
    
    def pull_data_parallel(
        self,
        queries: List[Tuple[str, str]],  # (query, source) pairs
        max_workers: int = 3
    ) -> List[pd.DataFrame]:
        """
        Pull data from multiple queries in parallel.
        
        Args:
            queries (List[Tuple[str, str]]): List of (query, source) pairs
            max_workers (int): Maximum number of parallel workers
            
        Returns:
            List[pd.DataFrame]: Results from all queries
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        self.logger.info(f"Starting parallel data pull for {len(queries)} queries")
        
        results = [None] * len(queries)  # Preserve order
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all queries
            future_to_index = {
                executor.submit(self.pull_data_with_retry, query, source): index
                for index, (query, source) in enumerate(queries)
            }
            
            # Collect results
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                
                try:
                    result_df = future.result()
                    results[index] = result_df
                    self.logger.debug(f"Query {index + 1} completed: {len(result_df)} rows")
                    
                except Exception as e:
                    self.logger.error(f"Query {index + 1} failed: {str(e)}")
                    results[index] = pd.DataFrame()  # Empty dataframe for failed queries
        
        successful_queries = sum(1 for df in results if not df.empty)
        self.logger.info(f"Parallel data pull completed: {successful_queries}/{len(queries)} queries successful")
        
        return results
    
    @contextmanager
    def _get_connection(self, source: str):
        """
        Get a database connection with proper cleanup.
        
        Args:
            source (str): Data source name
            
        Yields:
            Connection object
        """
        conn = None
        try:
            conn = self._create_connection(source)
            yield conn
        finally:
            if conn:
                self._close_connection(conn, source)
    
    def _create_connection(self, source: str):
        """
        Create a new database connection.
        
        Args:
            source (str): Data source name
            
        Returns:
            Database connection
        """
        try:
            connection_string = self._build_connection_string(source)
            
            # Create pyodbc connection
            conn = pyodbc.connect(
                connection_string,
                timeout=self.connection_config.get('connection_timeout', 30),
                autocommit=True
            )
            
            # Set connection properties
            conn.timeout = self.connection_config.get('query_timeout', 300)
            
            self.logger.debug(f"Created new connection for {source}")
            return conn
            
        except Exception as e:
            self.stats['connection_errors'] += 1
            self.logger.error(f"Failed to create connection for {source}: {str(e)}")
            raise
    
    def _close_connection(self, conn, source: str):
        """
        Close a database connection.
        
        Args:
            conn: Database connection
            source (str): Data source name
        """
        try:
            if conn and not conn.closed:
                conn.close()
                self.logger.debug(f"Closed connection for {source}")
        except Exception as e:
            self.logger.warning(f"Error closing connection for {source}: {str(e)}")
    
    def _build_connection_string(self, source: str) -> str:
        """
        Build connection string for the specified source.
        
        Args:
            source (str): Data source name
            
        Returns:
            str: ODBC connection string
        """
        if source == 'primary':
            server = self.credentials['server']
            database = self.credentials['database']
            username = self.credentials.get('username')
            password = self.credentials.get('password')
        elif source == 'aip':
            server = self.credentials.get('aip_server')
            database = self.credentials.get('aip_database')
            username = self.credentials.get('aip_username')
            password = self.credentials.get('aip_password')
        elif source == 'atlas':
            server = self.credentials.get('atlas_server')
            database = self.credentials.get('atlas_database')
            username = self.credentials.get('atlas_username')
            password = self.credentials.get('atlas_password')
        else:
            raise ValueError(f"Unknown data source: {source}")
        
        if not server or not database:
            raise ValueError(f"Missing connection information for source: {source}")
        
        # Build ODBC connection string
        driver = self.credentials.get('driver', 'ODBC Driver 17 for SQL Server')
        
        conn_parts = [
            f"DRIVER={{{driver}}}",
            f"SERVER={server}",
            f"DATABASE={database}",
            f"Connection Timeout={self.connection_config.get('connection_timeout', 30)}",
            f"Command Timeout={self.connection_config.get('query_timeout', 300)}"
        ]
        
        # Add authentication
        if username and password:
            conn_parts.extend([f"UID={username}", f"PWD={password}"])
        else:
            conn_parts.append("Trusted_Connection=yes")
        
        # Add additional options
        conn_parts.extend([
            "Encrypt=yes",
            "TrustServerCertificate=no"
        ])
        
        return ";".join(conn_parts)
    
    def _pull_data_direct(
        self,
        query: str,
        conn,
        params: Optional[Dict[str, Any]] = None
    ) -> pd.DataFrame:
        """
        Pull data directly without chunking.
        
        Args:
            query (str): SQL query
            conn: Database connection
            params (Optional[Dict[str, Any]]): Query parameters
            
        Returns:
            pd.DataFrame: Query results
        """
        try:
            df = pd.read_sql_query(query, conn, params=params)
            return df
            
        except Exception as e:
            self.logger.error(f"Direct data pull failed: {str(e)}")
            raise
    
    def _pull_data_chunked(
        self,
        query: str,
        conn,
        params: Optional[Dict[str, Any]] = None,
        chunk_size: int = 10000
    ) -> pd.DataFrame:
        """
        Pull data in chunks for large result sets.
        
        Args:
            query (str): SQL query
            conn: Database connection
            params (Optional[Dict[str, Any]]): Query parameters
            chunk_size (int): Number of rows per chunk
            
        Returns:
            pd.DataFrame: Combined results from all chunks
        """
        try:
            chunks = []
            
            for chunk_df in pd.read_sql_query(
                query,
                conn,
                params=params,
                chunksize=chunk_size
            ):
                chunks.append(chunk_df)
                self.logger.debug(f"Processed chunk with {len(chunk_df)} rows")
            
            if chunks:
                df = pd.concat(chunks, ignore_index=True)
                self.logger.debug(f"Chunked data pull completed: {len(chunks)} chunks, {len(df)} total rows")
                return df
            else:
                return pd.DataFrame()
                
        except Exception as e:
            self.logger.error(f"Chunked data pull failed: {str(e)}")
            raise
    
    def execute_non_query(
        self,
        query: str,
        source: str = 'primary',
        params: Optional[Dict[str, Any]] = None
    ) -> int:
        """
        Execute a non-query SQL statement (INSERT, UPDATE, DELETE).
        
        Args:
            query (str): SQL statement to execute
            source (str): Data source name
            params (Optional[Dict[str, Any]]): Query parameters
            
        Returns:
            int: Number of affected rows
        """
        self.logger.debug(f"Executing non-query against {source}")
        
        try:
            with self._get_connection(source) as conn:
                cursor = conn.cursor()
                
                if params:
                    cursor.execute(query, params)
                else:
                    cursor.execute(query)
                
                affected_rows = cursor.rowcount
                conn.commit()
                
                self.logger.debug(f"Non-query executed: {affected_rows} rows affected")
                return affected_rows
                
        except Exception as e:
            self.logger.error(f"Non-query execution failed: {str(e)}")
            raise
    
    def test_connection(self, source: str = 'primary') -> Dict[str, Any]:
        """
        Test connection to a data source.
        
        Args:
            source (str): Data source name
            
        Returns:
            Dict[str, Any]: Connection test results
        """
        result = {
            'source': source,
            'success': False,
            'response_time': 0.0,
            'error': None,
            'server_info': None
        }
        
        try:
            start_time = time.time()
            
            with self._get_connection(source) as conn:
                # Simple test query
                cursor = conn.cursor()
                cursor.execute("SELECT @@VERSION as server_version, GETDATE() as server_time")
                row = cursor.fetchone()
                
                result['success'] = True
                result['response_time'] = time.time() - start_time
                result['server_info'] = {
                    'version': row[0] if row else 'Unknown',
                    'server_time': row[1] if row else None
                }
                
                self.logger.info(f"Connection test successful for {source} ({result['response_time']:.2f}s)")
                
        except Exception as e:
            result['error'] = str(e)
            self.logger.error(f"Connection test failed for {source}: {str(e)}")
        
        return result
    
    def get_table_info(
        self,
        table_name: str,
        source: str = 'primary'
    ) -> Dict[str, Any]:
        """
        Get information about a database table.
        
        Args:
            table_name (str): Name of the table
            source (str): Data source name
            
        Returns:
            Dict[str, Any]: Table information
        """
        try:
            # Query to get table information
            info_query = f"""
            SELECT 
                COUNT(*) as row_count
            FROM {table_name}
            """
            
            columns_query = f"""
            SELECT 
                COLUMN_NAME,
                DATA_TYPE,
                IS_NULLABLE,
                CHARACTER_MAXIMUM_LENGTH
            FROM INFORMATION_SCHEMA.COLUMNS
            WHERE TABLE_NAME = '{table_name}'
            ORDER BY ORDINAL_POSITION
            """
            
            # Get row count
            row_count_df = self.pull_data(info_query, source)
            row_count = row_count_df.iloc[0]['row_count'] if not row_count_df.empty else 0
            
            # Get column information
            columns_df = self.pull_data(columns_query, source)
            
            table_info = {
                'table_name': table_name,
                'row_count': row_count,
                'column_count': len(columns_df),
                'columns': columns_df.to_dict('records') if not columns_df.empty else []
            }
            
            return table_info
            
        except Exception as e:
            self.logger.error(f"Failed to get table info for {table_name}: {str(e)}")
            raise
    
    def bulk_insert(
        self,
        df: pd.DataFrame,
        table_name: str,
        source: str = 'primary',
        if_exists: str = 'append',
        batch_size: int = 1000
    ) -> int:
        """
        Bulk insert dataframe into database table.
        
        Args:
            df (pd.DataFrame): Data to insert
            table_name (str): Target table name
            source (str): Data source name
            if_exists (str): Action if table exists ('append', 'replace', 'fail')
            batch_size (int): Number of rows per batch
            
        Returns:
            int: Number of rows inserted
        """
        if df.empty:
            self.logger.warning("Attempted to bulk insert empty dataframe")
            return 0
        
        self.logger.info(f"Starting bulk insert of {len(df)} rows to {table_name}")
        
        try:
            # Create SQLAlchemy engine for bulk operations
            connection_string = self._build_sqlalchemy_connection_string(source)
            engine = sqlalchemy.create_engine(connection_string)
            
            # Insert data in batches
            total_inserted = 0
            
            for i in range(0, len(df), batch_size):
                batch_df = df.iloc[i:i+batch_size]
                
                batch_df.to_sql(
                    table_name,
                    engine,
                    if_exists=if_exists if i == 0 else 'append',
                    index=False,
                    method='multi'
                )
                
                total_inserted += len(batch_df)
                self.logger.debug(f"Inserted batch {i//batch_size + 1}: {len(batch_df)} rows")
            
            engine.dispose()
            
            self.logger.info(f"Bulk insert completed: {total_inserted} rows inserted to {table_name}")
            return total_inserted
            
        except Exception as e:
            self.logger.error(f"Bulk insert failed: {str(e)}")
            raise
    
    def _build_sqlalchemy_connection_string(self, source: str) -> str:
        """
        Build SQLAlchemy connection string.
        
        Args:
            source (str): Data source name
            
        Returns:
            str: SQLAlchemy connection string
        """
        if source == 'primary':
            server = self.credentials['server']
            database = self.credentials['database']
            username = self.credentials.get('username')
            password = self.credentials.get('password')
        elif source == 'aip':
            server = self.credentials.get('aip_server')
            database = self.credentials.get('aip_database')
            username = self.credentials.get('aip_username')
            password = self.credentials.get('aip_password')
        elif source == 'atlas':
            server = self.credentials.get('atlas_server')
            database = self.credentials.get('atlas_database')
            username = self.credentials.get('atlas_username')
            password = self.credentials.get('atlas_password')
        else:
            raise ValueError(f"Unknown data source: {source}")
        
        driver = self.credentials.get('driver', 'ODBC Driver 17 for SQL Server')
        
        if username and password:
            conn_str = f"mssql+pyodbc://{username}:{password}@{server}/{database}?driver={driver.replace(' ', '+')}"
        else:
            conn_str = f"mssql+pyodbc://@{server}/{database}?driver={driver.replace(' ', '+')}&trusted_connection=yes"
        
        return conn_str
    
    def _update_stats(self, row_count: int, query_time: float, success: bool):
        """
        Update query statistics.
        
        Args:
            row_count (int): Number of rows processed
            query_time (float): Query execution time
            success (bool): Whether query was successful
        """
        self.stats['total_queries'] += 1
        self.stats['total_query_time'] += query_time
        
        if success:
            self.stats['successful_queries'] += 1
            self.stats['total_rows_fetched'] += row_count
        else:
            self.stats['failed_queries'] += 1
    
    def _get_default_config(self) -> Dict[str, Any]:
        """
        Get default connection configuration.
        
        Returns:
            Dict[str, Any]: Default configuration
        """
        return {
            'connection_timeout': 30,
            'query_timeout': 300,
            'max_retries': 3,
            'retry_delay': 2.0,
            'pool_size': 5,
            'max_connections': 20
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get query execution statistics.
        
        Returns:
            Dict[str, Any]: Current statistics
        """
        stats = self.stats.copy()
        
        if stats['total_queries'] > 0:
            stats['success_rate'] = stats['successful_queries'] / stats['total_queries']
            stats['avg_query_time'] = stats['total_query_time'] / stats['total_queries']
        else:
            stats['success_rate'] = 0.0
            stats['avg_query_time'] = 0.0
        
        if stats['successful_queries'] > 0:
            stats['avg_rows_per_query'] = stats['total_rows_fetched'] / stats['successful_queries']
        else:
            stats['avg_rows_per_query'] = 0.0
        
        return stats
    
    def reset_statistics(self):
        """
        Reset query statistics.
        """
        self.stats = {
            'total_queries': 0,
            'successful_queries': 0,
            'failed_queries': 0,
            'total_rows_fetched': 0,
            'total_query_time': 0.0,
            'connection_errors': 0,
            'retry_attempts': 0
        }
        
        self.logger.info("DataPuller statistics reset")


def create_data_puller(
    credentials: Dict[str, Any],
    connection_config: Optional[Dict[str, Any]] = None,
    logger: Optional[logging.Logger] = None
) -> DataPuller:
    """
    Factory function to create a data puller instance.
    
    Args:
        credentials (Dict[str, Any]): Database credentials
        connection_config (Optional[Dict[str, Any]]): Connection configuration
        logger (Optional[logging.Logger]): Logger instance
        
    Returns:
        DataPuller: Configured data puller instance
    """
    return DataPuller(
        credentials=credentials,
        connection_config=connection_config,
        logger=logger
    )