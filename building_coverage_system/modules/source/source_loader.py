"""
Multi-source data loader for building coverage system.

This module provides the SourceLoader class for loading data from multiple
database sources in parallel, including AIP, Atlas, and Snowflake databases.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
from typing import List, Dict, Optional, Callable, Any
import logging
import time

# Import original components for data access
from coverage_sql_pipelines.src.sql_extract import FeatureExtractor


class SourceLoader:
    """
    Multi-source data loader with parallel execution capabilities.
    
    This class provides functionality to load data from multiple database sources
    in parallel, combining the results and handling errors gracefully. It uses
    the existing FeatureExtractor logic for database connectivity while adding
    parallel processing capabilities similar to Codebase 2's source loading.
    
    Attributes:
        credentials_dict (Dict): Database credentials configuration
        sql_queries (Dict): SQL queries for data extraction
        crypto_spark: Spark cryptography instance
        logger: Logging instance for operations
        SQL_QUERY_CONFIGS (Dict): SQL query configuration parameters
        source_handlers (Dict[str, Callable]): Mapping of source names to handler functions
        default_timeout (int): Default timeout for source operations in seconds
    """
    
    def __init__(self, credentials_dict: Dict, sql_queries: Dict, crypto_spark, 
                 logger, SQL_QUERY_CONFIGS: Dict, default_timeout: int = 300):
        """
        Initialize the multi-source data loader.
        
        Args:
            credentials_dict (Dict): Database credentials configuration
            sql_queries (Dict): SQL queries for data extraction
            crypto_spark: Spark cryptography instance
            logger: Logging instance for operations
            SQL_QUERY_CONFIGS (Dict): SQL query configuration parameters
            default_timeout (int): Default timeout for operations in seconds
        """
        self.credentials_dict = credentials_dict
        self.sql_queries = sql_queries
        self.crypto_spark = crypto_spark
        self.logger = logger if logger else logging.getLogger(__name__)
        self.SQL_QUERY_CONFIGS = SQL_QUERY_CONFIGS
        self.default_timeout = default_timeout
        
        # Initialize source handlers mapping
        self.source_handlers = {
            'aip': self.load_from_aip,
            'atlas': self.load_from_atlas,
            'snowflake': self.load_from_snowflake
        }
        
        # Initialize feature extractor for database operations
        self._init_feature_extractor()
    
    def _init_feature_extractor(self):
        """
        Initialize the feature extractor for database operations.
        
        This method creates a FeatureExtractor instance using the original
        Codebase 1 logic for database connectivity.
        """
        try:
            self.feature_extractor = FeatureExtractor(
                self.credentials_dict,
                self.sql_queries,
                self.crypto_spark,
                self.logger,
                self.SQL_QUERY_CONFIGS
            )
            self.logger.info("Feature extractor initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize feature extractor: {e}")
            self.feature_extractor = None
    
    def load_data_parallel(self, sources: Optional[List[str]] = None, 
                          max_workers: int = 3, timeout: Optional[int] = None) -> pd.DataFrame:
        """
        Load data from multiple sources in parallel.
        
        This method loads data from the specified sources using parallel processing,
        similar to Codebase 2's multi-threading approach. It combines results from
        all sources and handles errors gracefully.
        
        Args:
            sources (Optional[List[str]]): List of source names to load from.
                                         Defaults to ['aip', 'atlas', 'snowflake']
            max_workers (int): Maximum number of parallel workers (default: 3)
            timeout (Optional[int]): Timeout for operations in seconds
            
        Returns:
            pd.DataFrame: Combined data from all sources with source tracking
        """
        if sources is None:
            sources = ['aip', 'atlas', 'snowflake']
        
        if timeout is None:
            timeout = self.default_timeout
        
        self.logger.info(f"Starting parallel data loading from sources: {sources}")
        start_time = time.time()
        
        results = []
        source_stats = {}
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit parallel loading tasks
            future_to_source = {
                executor.submit(self._load_from_source_with_timeout, source, timeout): source 
                for source in sources if source in self.source_handlers
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_source):
                source = future_to_source[future]
                
                try:
                    data, load_time, record_count = future.result()
                    source_stats[source] = {
                        'status': 'success',
                        'records': record_count,
                        'load_time': load_time
                    }
                    
                    if not data.empty:
                        data['data_source'] = source  # Track source of data
                        data['load_timestamp'] = time.time()  # Track when loaded
                        results.append(data)
                        self.logger.info(f"Successfully loaded {record_count} records from {source} in {load_time:.2f}s")
                    else:
                        self.logger.warning(f"No data returned from source: {source}")
                        
                except Exception as e:
                    source_stats[source] = {
                        'status': 'failed',
                        'error': str(e),
                        'records': 0,
                        'load_time': 0
                    }
                    self.logger.error(f"Failed to load from {source}: {e}")
        
        # Combine and process results
        combined_df = self._combine_and_deduplicate_results(results)
        
        total_time = time.time() - start_time
        total_records = len(combined_df)
        
        self.logger.info(f"Parallel loading completed in {total_time:.2f}s. "
                        f"Total records: {total_records}")
        self._log_source_statistics(source_stats)
        
        return combined_df
    
    def _load_from_source_with_timeout(self, source: str, timeout: int) -> tuple:
        """
        Load data from a specific source with timeout handling.
        
        Args:
            source (str): Source name to load from
            timeout (int): Timeout in seconds
            
        Returns:
            tuple: (data, load_time, record_count)
        """
        start_time = time.time()
        
        try:
            data = self.load_from_source(source)
            load_time = time.time() - start_time
            record_count = len(data)
            
            return data, load_time, record_count
            
        except Exception as e:
            load_time = time.time() - start_time
            self.logger.error(f"Error loading from {source} after {load_time:.2f}s: {e}")
            return pd.DataFrame(), load_time, 0
    
    def load_from_source(self, source: str) -> pd.DataFrame:
        """
        Load data from a specific source.
        
        This method routes the loading request to the appropriate handler
        based on the source name.
        
        Args:
            source (str): Name of the source to load from
            
        Returns:
            pd.DataFrame: Data loaded from the specified source
            
        Raises:
            ValueError: If source is not supported
        """
        if source not in self.source_handlers:
            raise ValueError(f"Unsupported source: {source}. "
                           f"Supported sources: {list(self.source_handlers.keys())}")
        
        handler = self.source_handlers[source]
        return handler()
    
    def load_from_aip(self) -> pd.DataFrame:
        """
        Load data from AIP SQL Data Warehouse.
        
        This method uses the original FeatureExtractor logic to load data
        from the AIP database, maintaining backward compatibility.
        
        Returns:
            pd.DataFrame: Data loaded from AIP
        """
        try:
            self.logger.debug("Loading data from AIP SQL Data Warehouse")
            
            if self.feature_extractor is None:
                raise RuntimeError("Feature extractor not initialized")
            
            # Use existing method from FeatureExtractor
            data = self.feature_extractor.get_aip_data()
            
            if data is not None and not data.empty:
                self.logger.debug(f"Loaded {len(data)} records from AIP")
                return data
            else:
                self.logger.warning("No data returned from AIP")
                return pd.DataFrame()
                
        except Exception as e:
            self.logger.error(f"AIP data loading failed: {e}")
            return pd.DataFrame()
    
    def load_from_atlas(self) -> pd.DataFrame:
        """
        Load data from Atlas SQL Data Warehouse.
        
        This method uses the original FeatureExtractor logic to load data
        from the Atlas database.
        
        Returns:
            pd.DataFrame: Data loaded from Atlas
        """
        try:
            self.logger.debug("Loading data from Atlas SQL Data Warehouse")
            
            if self.feature_extractor is None:
                raise RuntimeError("Feature extractor not initialized")
            
            # Use existing method from FeatureExtractor
            data = self.feature_extractor.get_atlas_data()
            
            if data is not None and not data.empty:
                self.logger.debug(f"Loaded {len(data)} records from Atlas")
                return data
            else:
                self.logger.warning("No data returned from Atlas")
                return pd.DataFrame()
                
        except Exception as e:
            self.logger.error(f"Atlas data loading failed: {e}")
            return pd.DataFrame()
    
    def load_from_snowflake(self) -> pd.DataFrame:
        """
        Load data from Snowflake database.
        
        This method uses the original FeatureExtractor logic to load data
        from Snowflake.
        
        Returns:
            pd.DataFrame: Data loaded from Snowflake
        """
        try:
            self.logger.debug("Loading data from Snowflake")
            
            if self.feature_extractor is None:
                raise RuntimeError("Feature extractor not initialized")
            
            # Use existing method from FeatureExtractor
            data = self.feature_extractor.get_snowflake_data()
            
            if data is not None and not data.empty:
                self.logger.debug(f"Loaded {len(data)} records from Snowflake")
                return data
            else:
                self.logger.warning("No data returned from Snowflake")
                return pd.DataFrame()
                
        except Exception as e:
            self.logger.error(f"Snowflake data loading failed: {e}")
            return pd.DataFrame()
    
    def _combine_and_deduplicate_results(self, results: List[pd.DataFrame]) -> pd.DataFrame:
        """
        Combine results from multiple sources and remove duplicates.
        
        This method combines dataframes from different sources and removes
        duplicate records based on CLAIMNO, keeping the first occurrence.
        
        Args:
            results (List[pd.DataFrame]): List of dataframes to combine
            
        Returns:
            pd.DataFrame: Combined and deduplicated dataframe
        """
        if not results:
            self.logger.warning("No data to combine")
            return pd.DataFrame()
        
        try:
            # Combine all dataframes
            combined_df = pd.concat(results, ignore_index=True)
            
            initial_count = len(combined_df)
            
            # Remove duplicates based on CLAIMNO if column exists
            if 'CLAIMNO' in combined_df.columns:
                combined_df = combined_df.drop_duplicates(subset=['CLAIMNO'], keep='first')
                final_count = len(combined_df)
                duplicates_removed = initial_count - final_count
                
                if duplicates_removed > 0:
                    self.logger.info(f"Removed {duplicates_removed} duplicate claims")
            else:
                self.logger.warning("CLAIMNO column not found, skipping deduplication")
                final_count = initial_count
            
            self.logger.info(f"Combined data: {initial_count} → {final_count} records")
            
            return combined_df
            
        except Exception as e:
            self.logger.error(f"Error combining results: {e}")
            return pd.DataFrame()
    
    def _log_source_statistics(self, source_stats: Dict[str, Dict[str, Any]]):
        """
        Log detailed statistics for each source.
        
        Args:
            source_stats (Dict[str, Dict[str, Any]]): Statistics for each source
        """
        self.logger.info("Source loading statistics:")
        
        total_records = 0
        successful_sources = 0
        
        for source, stats in source_stats.items():
            status = stats['status']
            records = stats.get('records', 0)
            load_time = stats.get('load_time', 0)
            
            if status == 'success':
                self.logger.info(f"  {source}: {records} records in {load_time:.2f}s")
                total_records += records
                successful_sources += 1
            else:
                error = stats.get('error', 'Unknown error')
                self.logger.error(f"  {source}: FAILED - {error}")
        
        self.logger.info(f"Summary: {successful_sources}/{len(source_stats)} sources successful, "
                        f"{total_records} total records")
    
    def get_source_capabilities(self) -> Dict[str, Dict[str, Any]]:
        """
        Get capabilities and status of each data source.
        
        Returns:
            Dict[str, Dict[str, Any]]: Source capabilities and status information
        """
        capabilities = {}
        
        for source_name in self.source_handlers.keys():
            capabilities[source_name] = {
                'available': True,
                'handler_function': self.source_handlers[source_name].__name__,
                'description': self._get_source_description(source_name),
                'last_test': None,  # Could be implemented to test connectivity
                'estimated_records': self._get_estimated_record_count(source_name)
            }
        
        return capabilities
    
    def _get_source_description(self, source_name: str) -> str:
        """
        Get description for a data source.
        
        Args:
            source_name (str): Name of the source
            
        Returns:
            str: Description of the source
        """
        descriptions = {
            'aip': 'AIP SQL Data Warehouse - Primary claims data source',
            'atlas': 'Atlas SQL Data Warehouse - Secondary claims data source',
            'snowflake': 'Snowflake Database - File notes and supplementary data'
        }
        return descriptions.get(source_name, f'Unknown source: {source_name}')
    
    def _get_estimated_record_count(self, source_name: str) -> str:
        """
        Get estimated record count for a source.
        
        Args:
            source_name (str): Name of the source
            
        Returns:
            str: Estimated record count description
        """
        # This could be implemented to query actual record counts
        estimates = {
            'aip': '10K-50K records',
            'atlas': '5K-25K records', 
            'snowflake': '1K-10K records'
        }
        return estimates.get(source_name, 'Unknown')
    
    def test_source_connectivity(self, source: str) -> Dict[str, Any]:
        """
        Test connectivity to a specific source.
        
        Args:
            source (str): Name of the source to test
            
        Returns:
            Dict[str, Any]: Test results including status and timing
        """
        test_result = {
            'source': source,
            'status': 'unknown',
            'response_time': 0.0,
            'error': None,
            'timestamp': time.time()
        }
        
        if source not in self.source_handlers:
            test_result['status'] = 'failed'
            test_result['error'] = f'Unsupported source: {source}'
            return test_result
        
        start_time = time.time()
        
        try:
            # Attempt to load a small sample of data
            data = self.load_from_source(source)
            response_time = time.time() - start_time
            
            test_result['response_time'] = response_time
            
            if data is not None and not data.empty:
                test_result['status'] = 'success'
                test_result['sample_records'] = len(data)
            else:
                test_result['status'] = 'success_no_data'
                test_result['sample_records'] = 0
                
        except Exception as e:
            test_result['status'] = 'failed'
            test_result['error'] = str(e)
            test_result['response_time'] = time.time() - start_time
        
        return test_result
    
    def test_all_sources(self) -> Dict[str, Dict[str, Any]]:
        """
        Test connectivity to all configured sources.
        
        Returns:
            Dict[str, Dict[str, Any]]: Test results for all sources
        """
        self.logger.info("Testing connectivity to all sources")
        
        test_results = {}
        for source in self.source_handlers.keys():
            test_results[source] = self.test_source_connectivity(source)
        
        # Log summary
        successful_sources = sum(1 for result in test_results.values() 
                               if result['status'] == 'success')
        total_sources = len(test_results)
        
        self.logger.info(f"Source connectivity test completed: "
                        f"{successful_sources}/{total_sources} sources accessible")
        
        return test_results