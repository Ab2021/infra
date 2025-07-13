"""
Snowflake Database Connector
Handles all interactions with Snowflake database including schema analysis and query execution
"""

import snowflake.connector
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor

class SnowflakeConnector:
    """
    Sophisticated Snowflake database connector that provides schema intelligence,
    optimized query execution, and performance monitoring capabilities.
    """
    
    def __init__(self, config):
        self.config = config
        self.connection = None
        self.cursor = None
        self.logger = logging.getLogger(__name__)
        
        # Connection pooling and performance optimization
        self.executor = ThreadPoolExecutor(max_workers=5)
        self.query_cache = {}
        self.schema_cache = {}
        
        # Performance monitoring
        self.execution_stats = {
            "total_queries": 0,
            "successful_queries": 0,
            "avg_execution_time": 0.0,
            "cache_hits": 0
        }
    
    async def connect(self):
        """Establishes connection to Snowflake with optimal configuration."""
        
        try:
            connection_params = {
                'user': self.config.snowflake_user,
                'password': self.config.snowflake_password,
                'account': self.config.snowflake_account,
                'warehouse': self.config.snowflake_warehouse,
                'database': self.config.snowflake_database,
                'schema': self.config.snowflake_schema,
                'role': self.config.snowflake_role,
                # Performance optimizations
                'client_session_keep_alive': True,
                'autocommit': True,
                'numpy': True  # For better pandas integration
            }
            
            # Use thread pool for blocking connection
            self.connection = await asyncio.get_event_loop().run_in_executor(
                self.executor,
                lambda: snowflake.connector.connect(**connection_params)
            )
            
            self.cursor = self.connection.cursor()
            self.logger.info("Successfully connected to Snowflake")
            
            # Initialize session with optimization settings
            await self._optimize_session()
            
        except Exception as e:
            self.logger.error(f"Failed to connect to Snowflake: {str(e)}")
            raise
    
    async def get_schema_metadata(self, include_samples: bool = False) -> Dict[str, Dict]:
        """
        Retrieves comprehensive schema metadata with optional sample data.
        
        Args:
            include_samples: Whether to include sample data for each table
            
        Returns:
            Complete schema metadata dictionary
        """
        
        cache_key = f"schema_metadata_{include_samples}"
        if cache_key in self.schema_cache:
            return self.schema_cache[cache_key]
        
        try:
            schema_metadata = {}
            
            # Get all tables in the database
            tables_query = """
                SELECT 
                    table_name,
                    table_type,
                    row_count,
                    bytes,
                    comment
                FROM information_schema.tables 
                WHERE table_schema = CURRENT_SCHEMA()
                ORDER BY table_name
            """
            
            tables_df = await self._execute_query_to_dataframe(tables_query)
            
            for _, table_row in tables_df.iterrows():
                table_name = table_row['TABLE_NAME']
                
                # Get column information for this table
                columns_query = f"""
                    SELECT 
                        column_name,
                        data_type,
                        is_nullable,
                        column_default,
                        comment,
                        ordinal_position
                    FROM information_schema.columns 
                    WHERE table_name = '{table_name}' 
                      AND table_schema = CURRENT_SCHEMA()
                    ORDER BY ordinal_position
                """
                
                columns_df = await self._execute_query_to_dataframe(columns_query)
                
                # Process column information
                columns_info = []
                for _, col_row in columns_df.iterrows():
                    column_info = {
                        "name": col_row['COLUMN_NAME'],
                        "type": col_row['DATA_TYPE'],
                        "nullable": col_row['IS_NULLABLE'] == 'YES',
                        "default": col_row['COLUMN_DEFAULT'],
                        "comment": col_row['COMMENT'],
                        "position": col_row['ORDINAL_POSITION']
                    }
                    columns_info.append(column_info)
                
                # Get foreign key relationships
                foreign_keys = await self._get_foreign_key_relationships(table_name)
                
                # Build table metadata
                table_metadata = {
                    "table_name": table_name,
                    "table_type": table_row['TABLE_TYPE'],
                    "row_count": table_row['ROW_COUNT'],
                    "size_bytes": table_row['BYTES'],
                    "description": table_row['COMMENT'],
                    "columns": columns_info,
                    "foreign_keys": foreign_keys,
                    "indexes": await self._get_table_indexes(table_name)
                }
                
                # Add sample data if requested
                if include_samples:
                    table_metadata["sample_data"] = await self.get_sample_data(table_name, 5)
                
                schema_metadata[table_name] = table_metadata
            
            # Cache the results
            self.schema_cache[cache_key] = schema_metadata
            
            self.logger.info(f"Retrieved schema metadata for {len(schema_metadata)} tables")
            return schema_metadata
            
        except Exception as e:
            self.logger.error(f"Failed to get schema metadata: {str(e)}")
            raise
    
    async def get_sample_data(self, table_name: str, sample_size: int = 10) -> List[Dict]:
        """
        Retrieves sample data from a table for context understanding.
        
        Args:
            table_name: Name of table to sample
            sample_size: Number of rows to retrieve
            
        Returns:
            List of sample records as dictionaries
        """
        
        try:
            # Use SAMPLE for better performance on large tables
            sample_query = f"""
                SELECT * 
                FROM {table_name} 
                SAMPLE ({min(sample_size * 10, 1000)} ROWS)
                LIMIT {sample_size}
            """
            
            df = await self._execute_query_to_dataframe(sample_query)
            return df.to_dict('records')
            
        except Exception as e:
            self.logger.warning(f"Failed to get sample data for {table_name}: {str(e)}")
            return []
    
    async def execute_query(self, sql: str, parameters: Optional[Dict] = None) -> Tuple[List[Dict], Dict]:
        """
        Executes SQL query with performance monitoring and error handling.
        
        Args:
            sql: SQL query string
            parameters: Optional parameters for parameterized queries
            
        Returns:
            Tuple of (results as list of dicts, execution metadata)
        """
        
        start_time = datetime.now()
        
        try:
            # Check cache first
            cache_key = f"{sql}_{str(parameters)}" if parameters else sql
            if cache_key in self.query_cache:
                self.execution_stats["cache_hits"] += 1
                return self.query_cache[cache_key], {"cached": True, "execution_time": 0.0}
            
            # Execute query
            if parameters:
                results_df = await self._execute_query_to_dataframe(sql, parameters)
            else:
                results_df = await self._execute_query_to_dataframe(sql)
            
            # Convert to list of dictionaries
            results = results_df.to_dict('records')
            
            # Calculate execution time
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # Update performance statistics
            self._update_execution_stats(execution_time, True)
            
            # Cache results for simple SELECT queries
            if sql.strip().upper().startswith('SELECT') and len(results) < 1000:
                self.query_cache[cache_key] = results
            
            execution_metadata = {
                "execution_time": execution_time,
                "row_count": len(results),
                "cached": False,
                "timestamp": datetime.now().isoformat()
            }
            
            self.logger.info(f"Query executed successfully in {execution_time:.2f}s, {len(results)} rows returned")
            return results, execution_metadata
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            self._update_execution_stats(execution_time, False)
            
            self.logger.error(f"Query execution failed: {str(e)}")
            self.logger.error(f"SQL: {sql}")
            
            raise Exception(f"Query execution failed: {str(e)}")
    
    async def validate_sql(self, sql: str) -> Dict[str, Any]:
        """
        Validates SQL syntax and estimates execution cost.
        
        Args:
            sql: SQL query to validate
            
        Returns:
            Validation results with syntax check and cost estimate
        """
        
        try:
            # Use EXPLAIN to validate syntax and get execution plan
            explain_query = f"EXPLAIN {sql}"
            
            explain_results = await self._execute_query_to_dataframe(explain_query)
            
            # Parse execution plan for cost estimation
            execution_plan = explain_results.to_string()
            
            # Basic cost estimation based on operations
            estimated_cost = self._estimate_query_cost(execution_plan)
            
            # Check for potential performance issues
            performance_warnings = self._analyze_performance_risks(sql, execution_plan)
            
            return {
                "is_valid": True,
                "estimated_cost": estimated_cost,
                "execution_plan": execution_plan,
                "performance_warnings": performance_warnings,
                "validation_timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                "is_valid": False,
                "error_message": str(e),
                "validation_timestamp": datetime.now().isoformat()
            }
    
    async def _execute_query_to_dataframe(self, sql: str, parameters: Optional[Dict] = None) -> pd.DataFrame:
        """Executes query and returns results as pandas DataFrame."""
        
        def execute_blocking():
            if parameters:
                self.cursor.execute(sql, parameters)
            else:
                self.cursor.execute(sql)
            return self.cursor.fetch_pandas_all()
        
        # Execute in thread pool to avoid blocking
        df = await asyncio.get_event_loop().run_in_executor(
            self.executor,
            execute_blocking
        )
        
        return df
    
    async def _get_foreign_key_relationships(self, table_name: str) -> List[Dict]:
        """Gets foreign key relationships for a table."""
        
        try:
            fk_query = f"""
                SELECT 
                    column_name,
                    referenced_table_name,
                    referenced_column_name
                FROM information_schema.referential_constraints rc
                JOIN information_schema.key_column_usage kcu 
                  ON rc.constraint_name = kcu.constraint_name
                WHERE kcu.table_name = '{table_name}'
                  AND kcu.table_schema = CURRENT_SCHEMA()
            """
            
            fk_df = await self._execute_query_to_dataframe(fk_query)
            return fk_df.to_dict('records')
            
        except Exception:
            return []
    
    async def _get_table_indexes(self, table_name: str) -> List[Dict]:
        """Gets index information for a table."""
        
        # Snowflake doesn't have traditional indexes, but we can check for clustering keys
        try:
            cluster_query = f"""
                SHOW TABLES LIKE '{table_name}' IN SCHEMA {self.config.snowflake_schema}
            """
            
            # This is a simplified implementation
            # In practice, you'd parse the clustering key information
            return []
            
        except Exception:
            return []
    
    async def _optimize_session(self):
        """Optimizes the Snowflake session for performance."""
        
        optimization_settings = [
            "ALTER SESSION SET QUERY_TAG = 'SQL_AGENT_SYSTEM'",
            "ALTER SESSION SET STATEMENT_TIMEOUT_IN_SECONDS = 300",
            "ALTER SESSION SET USE_CACHED_RESULT = TRUE"
        ]
        
        for setting in optimization_settings:
            try:
                await self._execute_query_to_dataframe(setting)
            except Exception as e:
                self.logger.warning(f"Failed to apply optimization setting: {setting}, Error: {e}")
    
    def _estimate_query_cost(self, execution_plan: str) -> str:
        """Estimates query execution cost based on execution plan."""
        
        # Simple cost estimation logic
        if "FULL TABLE SCAN" in execution_plan.upper():
            return "HIGH"
        elif "JOIN" in execution_plan.upper() and "SORT" in execution_plan.upper():
            return "MEDIUM"
        else:
            return "LOW"
    
    def _analyze_performance_risks(self, sql: str, execution_plan: str) -> List[str]:
        """Analyzes potential performance risks in the query."""
        
        warnings = []
        sql_upper = sql.upper()
        plan_upper = execution_plan.upper()
        
        if "SELECT *" in sql_upper:
            warnings.append("Using SELECT * may retrieve unnecessary columns")
        
        if "FULL TABLE SCAN" in plan_upper:
            warnings.append("Query requires full table scan - consider adding filters")
        
        if sql_upper.count("JOIN") > 3:
            warnings.append("Complex query with multiple joins - consider optimization")
        
        if "ORDER BY" in sql_upper and "LIMIT" not in sql_upper:
            warnings.append("ORDER BY without LIMIT may be expensive on large datasets")
        
        return warnings
    
    def _update_execution_stats(self, execution_time: float, success: bool):
        """Updates internal execution statistics."""
        
        self.execution_stats["total_queries"] += 1
        
        if success:
            self.execution_stats["successful_queries"] += 1
            
            # Update rolling average
            current_avg = self.execution_stats["avg_execution_time"]
            total_successful = self.execution_stats["successful_queries"]
            
            new_avg = ((current_avg * (total_successful - 1)) + execution_time) / total_successful
            self.execution_stats["avg_execution_time"] = new_avg
    
    async def close(self):
        """Closes database connection and cleans up resources."""
        
        if self.cursor:
            self.cursor.close()
        
        if self.connection:
            self.connection.close()
        
        self.executor.shutdown(wait=True)
        
        self.logger.info("Snowflake connection closed")
    
    def get_performance_stats(self) -> Dict:
        """Returns current performance statistics."""
        
        return {
            **self.execution_stats,
            "cache_size": len(self.query_cache),
            "success_rate": (
                self.execution_stats["successful_queries"] / 
                max(self.execution_stats["total_queries"], 1)
            ) * 100
        }
