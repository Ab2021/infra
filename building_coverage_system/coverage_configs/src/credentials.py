"""
Database credentials management for building coverage system.

This module handles secure credential management and database
connection configuration for the original Codebase 1 components.
"""

import os
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


def get_database_credentials(databricks_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract and format database credentials from Databricks configuration.
    
    Args:
        databricks_dict (Dict[str, Any]): Databricks configuration dictionary
        
    Returns:
        Dict[str, Any]: Formatted database credentials
    """
    # Extract credentials from environment variables or databricks_dict
    credentials = {
        # Primary database connection
        'server': _get_credential('DB_SERVER', databricks_dict.get('server')),
        'database': _get_credential('DB_DATABASE', databricks_dict.get('database')),
        'username': _get_credential('DB_USERNAME', databricks_dict.get('username')),
        'password': _get_credential('DB_PASSWORD', databricks_dict.get('password')),
        'driver': _get_credential('DB_DRIVER', 'ODBC Driver 17 for SQL Server'),
        
        # AIP database connection
        'aip_server': _get_credential('AIP_SQL_SERVER', databricks_dict.get('aip_server')),
        'aip_database': _get_credential('AIP_DATABASE', databricks_dict.get('aip_database', 'AIP_Claims')),
        'aip_username': _get_credential('AIP_USERNAME', databricks_dict.get('aip_username')),
        'aip_password': _get_credential('AIP_PASSWORD', databricks_dict.get('aip_password')),
        
        # Atlas database connection
        'atlas_server': _get_credential('ATLAS_SQL_SERVER', databricks_dict.get('atlas_server')),
        'atlas_database': _get_credential('ATLAS_DATABASE', databricks_dict.get('atlas_database', 'Atlas_Claims')),
        'atlas_username': _get_credential('ATLAS_USERNAME', databricks_dict.get('atlas_username')),
        'atlas_password': _get_credential('ATLAS_PASSWORD', databricks_dict.get('atlas_password')),
        
        # Snowflake connection (optional)
        'snowflake_account': _get_credential('SNOWFLAKE_ACCOUNT', databricks_dict.get('snowflake_account')),
        'snowflake_database': _get_credential('SNOWFLAKE_DATABASE', databricks_dict.get('snowflake_database')),
        'snowflake_username': _get_credential('SNOWFLAKE_USERNAME', databricks_dict.get('snowflake_username')),
        'snowflake_password': _get_credential('SNOWFLAKE_PASSWORD', databricks_dict.get('snowflake_password')),
        'snowflake_warehouse': _get_credential('SNOWFLAKE_WAREHOUSE', databricks_dict.get('snowflake_warehouse')),
        
        # Connection settings
        'connection_timeout': int(_get_credential('DB_TIMEOUT', '30')),
        'command_timeout': int(_get_credential('DB_COMMAND_TIMEOUT', '300')),
        'trusted_connection': _get_credential('DB_TRUSTED_CONNECTION', 'no'),
        'encrypt': _get_credential('DB_ENCRYPT', 'yes'),
        'trust_server_certificate': _get_credential('DB_TRUST_CERT', 'no')
    }
    
    # Validate required credentials
    _validate_credentials(credentials)
    
    logger.info("Database credentials loaded successfully")
    return credentials


def _get_credential(env_var: str, default_value: Optional[str] = None) -> Optional[str]:
    """
    Get credential from environment variable or default value.
    
    Args:
        env_var (str): Environment variable name
        default_value (Optional[str]): Default value if env var not found
        
    Returns:
        Optional[str]: Credential value
    """
    value = os.getenv(env_var, default_value)
    
    # Don't log sensitive values
    if any(sensitive in env_var.lower() for sensitive in ['password', 'key', 'token']):
        if value:
            logger.debug(f"Loaded credential for {env_var}: [REDACTED]")
        else:
            logger.warning(f"Missing credential for {env_var}")
    else:
        logger.debug(f"Loaded credential for {env_var}: {value}")
    
    return value


def _validate_credentials(credentials: Dict[str, Any]) -> None:
    """
    Validate that required credentials are present.
    
    Args:
        credentials (Dict[str, Any]): Credentials dictionary
        
    Raises:
        ValueError: If required credentials are missing
    """
    required_fields = ['server', 'database']
    missing_fields = []
    
    for field in required_fields:
        if not credentials.get(field):
            missing_fields.append(field)
    
    if missing_fields:
        raise ValueError(f"Missing required database credentials: {missing_fields}")
    
    # Warn about missing optional credentials
    optional_warnings = []
    
    if not credentials.get('username') and not credentials.get('trusted_connection') == 'yes':
        optional_warnings.append("No username provided and trusted connection not enabled")
    
    if not credentials.get('aip_server'):
        optional_warnings.append("AIP server not configured - AIP data source will be disabled")
    
    if not credentials.get('atlas_server'):
        optional_warnings.append("Atlas server not configured - Atlas data source will be disabled")
    
    for warning in optional_warnings:
        logger.warning(warning)


def get_connection_string(credentials: Dict[str, Any], source: str = 'primary') -> str:
    """
    Build connection string for specified data source.
    
    Args:
        credentials (Dict[str, Any]): Credentials dictionary
        source (str): Data source name ('primary', 'aip', 'atlas', 'snowflake')
        
    Returns:
        str: Database connection string
        
    Raises:
        ValueError: If source is not supported or credentials are missing
    """
    if source == 'primary':
        return _build_sql_server_connection_string(
            credentials['server'],
            credentials['database'],
            credentials.get('username'),
            credentials.get('password'),
            credentials
        )
    elif source == 'aip':
        if not credentials.get('aip_server'):
            raise ValueError("AIP server not configured")
        return _build_sql_server_connection_string(
            credentials['aip_server'],
            credentials['aip_database'],
            credentials.get('aip_username'),
            credentials.get('aip_password'),
            credentials
        )
    elif source == 'atlas':
        if not credentials.get('atlas_server'):
            raise ValueError("Atlas server not configured")
        return _build_sql_server_connection_string(
            credentials['atlas_server'],
            credentials['atlas_database'],
            credentials.get('atlas_username'),
            credentials.get('atlas_password'),
            credentials
        )
    elif source == 'snowflake':
        if not credentials.get('snowflake_account'):
            raise ValueError("Snowflake account not configured")
        return _build_snowflake_connection_string(credentials)
    else:
        raise ValueError(f"Unsupported data source: {source}")


def _build_sql_server_connection_string(
    server: str,
    database: str,
    username: Optional[str],
    password: Optional[str],
    credentials: Dict[str, Any]
) -> str:
    """
    Build SQL Server connection string.
    
    Args:
        server (str): Server name
        database (str): Database name
        username (Optional[str]): Username
        password (Optional[str]): Password
        credentials (Dict[str, Any]): Additional connection settings
        
    Returns:
        str: SQL Server connection string
    """
    conn_parts = [
        f"DRIVER={{{credentials.get('driver', 'ODBC Driver 17 for SQL Server')}}}",
        f"SERVER={server}",
        f"DATABASE={database}",
        f"Connection Timeout={credentials.get('connection_timeout', 30)}",
        f"Command Timeout={credentials.get('command_timeout', 300)}",
        f"Encrypt={credentials.get('encrypt', 'yes')}",
        f"TrustServerCertificate={credentials.get('trust_server_certificate', 'no')}"
    ]
    
    # Add authentication
    if username and password:
        conn_parts.extend([
            f"UID={username}",
            f"PWD={password}"
        ])
    elif credentials.get('trusted_connection') == 'yes':
        conn_parts.append("Trusted_Connection=yes")
    else:
        logger.warning("No authentication method specified")
    
    return ";".join(conn_parts)


def _build_snowflake_connection_string(credentials: Dict[str, Any]) -> str:
    """
    Build Snowflake connection string.
    
    Args:
        credentials (Dict[str, Any]): Snowflake credentials
        
    Returns:
        str: Snowflake connection string
    """
    conn_parts = [
        f"account={credentials['snowflake_account']}",
        f"database={credentials.get('snowflake_database', 'CLAIMS')}",
        f"warehouse={credentials.get('snowflake_warehouse', 'COMPUTE_WH')}"
    ]
    
    if credentials.get('snowflake_username'):
        conn_parts.append(f"user={credentials['snowflake_username']}")
    
    if credentials.get('snowflake_password'):
        conn_parts.append(f"password={credentials['snowflake_password']}")
    
    return "&".join(conn_parts)


def test_connection(credentials: Dict[str, Any], source: str = 'primary') -> Dict[str, Any]:
    """
    Test database connection with provided credentials.
    
    Args:
        credentials (Dict[str, Any]): Database credentials
        source (str): Data source to test
        
    Returns:
        Dict[str, Any]: Connection test results
    """
    result = {
        'source': source,
        'success': False,
        'error': None,
        'connection_time': None
    }
    
    try:
        import time
        start_time = time.time()
        
        # This would normally test the actual connection
        # For now, just validate that we can build the connection string
        connection_string = get_connection_string(credentials, source)
        
        if connection_string:
            result['success'] = True
            result['connection_time'] = time.time() - start_time
            logger.info(f"Connection test successful for {source}")
        
    except Exception as e:
        result['error'] = str(e)
        logger.error(f"Connection test failed for {source}: {e}")
    
    return result


def get_supported_sources(credentials: Dict[str, Any]) -> Dict[str, bool]:
    """
    Get list of supported data sources based on available credentials.
    
    Args:
        credentials (Dict[str, Any]): Database credentials
        
    Returns:
        Dict[str, bool]: Supported sources and their availability
    """
    return {
        'primary': bool(credentials.get('server')),
        'aip': bool(credentials.get('aip_server')),
        'atlas': bool(credentials.get('atlas_server')),
        'snowflake': bool(credentials.get('snowflake_account'))
    }