"""
SQL queries for building coverage system.

This module contains all SQL queries used by the original Codebase 1
components, organized by functionality and data source.
"""

from typing import Dict, Any


def get_sql_queries() -> Dict[str, Any]:
    """
    Get all SQL queries organized by category.
    
    Returns:
        Dict[str, Any]: Dictionary containing all SQL queries
    """
    return {
        'feature_queries': get_feature_queries(),
        'validation_queries': get_validation_queries(),
        'lookup_queries': get_lookup_queries(),
        'reporting_queries': get_reporting_queries()
    }


def get_feature_queries() -> Dict[str, str]:
    """
    Get feature extraction queries.
    
    Returns:
        Dict[str, str]: Feature extraction SQL queries
    """
    return {
        'main_claims_query': '''
            SELECT DISTINCT
                c.CLAIMNO,
                c.CLAIMKEY,
                c.clean_FN_TEXT,
                c.LOBCD,
                c.LOSSDESC,
                c.LOSSDT,
                c.REPORTEDDT,
                c.STATUSCD,
                c.RESERVE_TOTAL,
                c.PAID_TOTAL,
                c.INCURRED_TOTAL,
                p.POLICY_TYPE,
                p.COVERAGE_LIMITS
            FROM Claims c WITH (NOLOCK)
            LEFT JOIN Policies p WITH (NOLOCK) ON c.POLICYNO = p.POLICYNO
            WHERE c.LOBCD IN ('15', '17')
                AND c.clean_FN_TEXT IS NOT NULL
                AND LEN(c.clean_FN_TEXT) >= 100
                AND c.STATUSCD IN ('O', 'C', 'R')  -- Open, Closed, Reopened
                AND c.LOSSDT >= DATEADD(YEAR, -6, GETDATE())
            ORDER BY c.LOSSDT DESC
        ''',
        
        'aip_claims_query': '''
            SELECT 
                CLAIMNO,
                CLAIMKEY,
                CLAIM_DESC as clean_FN_TEXT,
                LINE_OF_BUSINESS as LOBCD,
                LOSS_DESCRIPTION as LOSSDESC,
                LOSS_DATE as LOSSDT,
                REPORTED_DATE as REPORTEDDT,
                CLAIM_STATUS as STATUSCD,
                RESERVE_AMOUNT as RESERVE_TOTAL
            FROM AIP_CLAIMS WITH (NOLOCK)
            WHERE LINE_OF_BUSINESS IN ('PROPERTY', 'CASUALTY')
                AND CLAIM_DESC IS NOT NULL
                AND LEN(CLAIM_DESC) >= 100
                AND LOSS_DATE >= DATEADD(YEAR, -6, GETDATE())
                AND CLAIM_STATUS = 'ACTIVE'
        ''',
        
        'atlas_claims_query': '''
            SELECT 
                CLAIM_NUMBER as CLAIMNO,
                CLAIM_ID as CLAIMKEY,
                DESCRIPTION as clean_FN_TEXT,
                LOB_CODE as LOBCD,
                LOSS_CAUSE as LOSSDESC,
                DATE_OF_LOSS as LOSSDT,
                DATE_REPORTED as REPORTEDDT,
                STATUS as STATUSCD,
                OUTSTANDING_RESERVE as RESERVE_TOTAL
            FROM ATLAS_CLAIMS WITH (NOLOCK)
            WHERE LOB_CODE IN ('15', '17', '18')
                AND DESCRIPTION IS NOT NULL
                AND LEN(DESCRIPTION) >= 100
                AND DATE_OF_LOSS >= DATEADD(YEAR, -6, GETDATE())
        ''',
        
        'building_specific_query': '''
            SELECT 
                c.CLAIMNO,
                c.CLAIMKEY,
                c.clean_FN_TEXT,
                c.LOBCD,
                c.LOSSDESC,
                c.LOSSDT,
                c.REPORTEDDT,
                bc.BUILDING_TYPE,
                bc.CONSTRUCTION_TYPE,
                bc.YEAR_BUILT,
                bc.SQUARE_FOOTAGE,
                bc.REPLACEMENT_COST
            FROM Claims c WITH (NOLOCK)
            INNER JOIN Building_Coverage bc WITH (NOLOCK) ON c.CLAIMNO = bc.CLAIMNO
            WHERE c.LOBCD IN ('15', '17')
                AND c.clean_FN_TEXT IS NOT NULL
                AND (
                    c.clean_FN_TEXT LIKE '%building%' OR
                    c.clean_FN_TEXT LIKE '%structure%' OR
                    c.clean_FN_TEXT LIKE '%foundation%' OR
                    c.clean_FN_TEXT LIKE '%roof%' OR
                    c.clean_FN_TEXT LIKE '%wall%'
                )
        ''',
        
        'incremental_query': '''
            SELECT 
                c.CLAIMNO,
                c.CLAIMKEY,
                c.clean_FN_TEXT,
                c.LOBCD,
                c.LOSSDESC,
                c.LOSSDT,
                c.REPORTEDDT,
                c.LAST_MODIFIED_DATE
            FROM Claims c WITH (NOLOCK)
            WHERE c.LOBCD IN ('15', '17')
                AND c.clean_FN_TEXT IS NOT NULL
                AND c.LAST_MODIFIED_DATE > ?
            ORDER BY c.LAST_MODIFIED_DATE
        '''
    }


def get_validation_queries() -> Dict[str, str]:
    """
    Get data validation queries.
    
    Returns:
        Dict[str, str]: Validation SQL queries
    """
    return {
        'data_quality_check': '''
            SELECT 
                COUNT(*) as total_claims,
                COUNT(CASE WHEN clean_FN_TEXT IS NOT NULL THEN 1 END) as claims_with_text,
                COUNT(CASE WHEN LEN(clean_FN_TEXT) >= 100 THEN 1 END) as claims_sufficient_text,
                AVG(LEN(clean_FN_TEXT)) as avg_text_length,
                COUNT(DISTINCT LOBCD) as unique_lob_codes,
                COUNT(CASE WHEN LOSSDT IS NOT NULL THEN 1 END) as claims_with_loss_date
            FROM Claims WITH (NOLOCK)
            WHERE LOBCD IN ('15', '17')
                AND LOSSDT >= DATEADD(YEAR, -1, GETDATE())
        ''',
        
        'duplicate_check': '''
            SELECT 
                CLAIMNO,
                COUNT(*) as duplicate_count
            FROM Claims WITH (NOLOCK)
            WHERE LOBCD IN ('15', '17')
            GROUP BY CLAIMNO
            HAVING COUNT(*) > 1
        ''',
        
        'text_quality_check': '''
            SELECT 
                CLAIMNO,
                LEN(clean_FN_TEXT) as text_length,
                CASE 
                    WHEN clean_FN_TEXT LIKE '%test%' THEN 'TEST_DATA'
                    WHEN clean_FN_TEXT LIKE '%lorem ipsum%' THEN 'PLACEHOLDER'
                    WHEN LEN(clean_FN_TEXT) < 10 THEN 'TOO_SHORT'
                    WHEN LEN(clean_FN_TEXT) > 10000 THEN 'TOO_LONG'
                    ELSE 'OK'
                END as quality_flag
            FROM Claims WITH (NOLOCK)
            WHERE LOBCD IN ('15', '17')
                AND clean_FN_TEXT IS NOT NULL
        ''',
        
        'date_validation': '''
            SELECT 
                CLAIMNO,
                LOSSDT,
                REPORTEDDT,
                DATEDIFF(day, LOSSDT, REPORTEDDT) as reporting_lag_days,
                CASE 
                    WHEN LOSSDT > GETDATE() THEN 'FUTURE_LOSS_DATE'
                    WHEN REPORTEDDT < LOSSDT THEN 'REPORT_BEFORE_LOSS'
                    WHEN DATEDIFF(day, LOSSDT, REPORTEDDT) > 365 THEN 'LONG_REPORTING_LAG'
                    WHEN DATEDIFF(day, LOSSDT, REPORTEDDT) < 0 THEN 'NEGATIVE_REPORTING_LAG'
                    ELSE 'OK'
                END as date_quality_flag
            FROM Claims WITH (NOLOCK)
            WHERE LOBCD IN ('15', '17')
                AND LOSSDT IS NOT NULL
                AND REPORTEDDT IS NOT NULL
        '''
    }


def get_lookup_queries() -> Dict[str, str]:
    """
    Get lookup and reference queries.
    
    Returns:
        Dict[str, str]: Lookup SQL queries
    """
    return {
        'lob_codes': '''
            SELECT DISTINCT 
                LOBCD,
                LOB_DESCRIPTION,
                COVERAGE_TYPE,
                IS_BUILDING_COVERAGE
            FROM LOB_Codes WITH (NOLOCK)
            WHERE LOBCD IN ('15', '17', '18', '19')
            ORDER BY LOBCD
        ''',
        
        'coverage_types': '''
            SELECT DISTINCT
                COVERAGE_TYPE,
                COVERAGE_DESCRIPTION,
                REQUIRES_BUILDING_ASSESSMENT
            FROM Coverage_Types WITH (NOLOCK)
            WHERE IS_ACTIVE = 1
        ''',
        
        'building_keywords': '''
            SELECT 
                KEYWORD,
                CATEGORY,
                WEIGHT,
                IS_POSITIVE_INDICATOR
            FROM Building_Keywords WITH (NOLOCK)
            WHERE IS_ACTIVE = 1
            ORDER BY WEIGHT DESC
        ''',
        
        'loss_causes': '''
            SELECT DISTINCT
                LOSS_CAUSE_CODE,
                LOSS_CAUSE_DESCRIPTION,
                TYPICALLY_INVOLVES_BUILDING
            FROM Loss_Causes WITH (NOLOCK)
            WHERE IS_ACTIVE = 1
        '''
    }


def get_reporting_queries() -> Dict[str, str]:
    """
    Get reporting and analytics queries.
    
    Returns:
        Dict[str, str]: Reporting SQL queries
    """
    return {
        'processing_summary': '''
            SELECT 
                CONVERT(date, PROCESSED_DATE) as processing_date,
                COUNT(*) as claims_processed,
                COUNT(CASE WHEN PREDICTION = 'BUILDING COVERAGE' THEN 1 END) as building_coverage_predictions,
                COUNT(CASE WHEN CONFIDENCE >= 0.8 THEN 1 END) as high_confidence_predictions,
                AVG(CONFIDENCE) as avg_confidence,
                AVG(PROCESSING_TIME_SECONDS) as avg_processing_time
            FROM Building_Coverage_Predictions WITH (NOLOCK)
            WHERE PROCESSED_DATE >= DATEADD(day, -30, GETDATE())
            GROUP BY CONVERT(date, PROCESSED_DATE)
            ORDER BY processing_date DESC
        ''',
        
        'accuracy_metrics': '''
            SELECT 
                bp.PREDICTION,
                bp.CONFIDENCE,
                CASE 
                    WHEN bp.PREDICTION = 'BUILDING COVERAGE' AND bc.CLAIMNO IS NOT NULL THEN 'TRUE_POSITIVE'
                    WHEN bp.PREDICTION = 'BUILDING COVERAGE' AND bc.CLAIMNO IS NULL THEN 'FALSE_POSITIVE'
                    WHEN bp.PREDICTION = 'NO BUILDING COVERAGE' AND bc.CLAIMNO IS NOT NULL THEN 'FALSE_NEGATIVE'
                    WHEN bp.PREDICTION = 'NO BUILDING COVERAGE' AND bc.CLAIMNO IS NULL THEN 'TRUE_NEGATIVE'
                    ELSE 'UNKNOWN'
                END as prediction_accuracy
            FROM Building_Coverage_Predictions bp WITH (NOLOCK)
            LEFT JOIN Building_Coverage bc WITH (NOLOCK) ON bp.CLAIMNO = bc.CLAIMNO
            WHERE bp.PROCESSED_DATE >= DATEADD(day, -7, GETDATE())
        ''',
        
        'performance_metrics': '''
            SELECT 
                COUNT(*) as total_claims,
                AVG(PROCESSING_TIME_SECONDS) as avg_processing_time,
                MAX(PROCESSING_TIME_SECONDS) as max_processing_time,
                MIN(PROCESSING_TIME_SECONDS) as min_processing_time,
                COUNT(CASE WHEN ERROR_MESSAGE IS NOT NULL THEN 1 END) as error_count,
                COUNT(CASE WHEN CONFIDENCE >= 0.8 THEN 1 END) as high_confidence_count
            FROM Building_Coverage_Predictions WITH (NOLOCK)
            WHERE PROCESSED_DATE >= DATEADD(hour, -24, GETDATE())
        '''
    }


def get_query_by_name(query_name: str) -> str:
    """
    Get a specific query by name.
    
    Args:
        query_name (str): Name of the query to retrieve
        
    Returns:
        str: SQL query string
        
    Raises:
        KeyError: If query name is not found
    """
    all_queries = get_sql_queries()
    
    # Search through all query categories
    for category, queries in all_queries.items():
        if query_name in queries:
            return queries[query_name]
    
    # If not found in categories, check if it's a direct match
    for category, queries in all_queries.items():
        for name, query in queries.items():
            if name == query_name:
                return query
    
    raise KeyError(f"Query '{query_name}' not found")


def get_parameterized_query(query_name: str, parameters: Dict[str, Any]) -> str:
    """
    Get a parameterized query with values substituted.
    
    Args:
        query_name (str): Name of the query
        parameters (Dict[str, Any]): Parameters to substitute
        
    Returns:
        str: Query with parameters substituted
    """
    query = get_query_by_name(query_name)
    
    # Simple parameter substitution (in production, use proper SQL parameters)
    for param_name, param_value in parameters.items():
        placeholder = f"{{{param_name}}}"
        if placeholder in query:
            if isinstance(param_value, str):
                query = query.replace(placeholder, f"'{param_value}'")
            else:
                query = query.replace(placeholder, str(param_value))
    
    return query


def validate_query_syntax(query: str) -> Dict[str, Any]:
    """
    Basic validation of SQL query syntax.
    
    Args:
        query (str): SQL query to validate
        
    Returns:
        Dict[str, Any]: Validation results
    """
    result = {
        'is_valid': True,
        'errors': [],
        'warnings': []
    }
    
    # Basic syntax checks
    query_upper = query.upper().strip()
    
    # Check for required keywords
    if not query_upper.startswith('SELECT'):
        result['errors'].append('Query must start with SELECT')
        result['is_valid'] = False
    
    # Check for potential security issues
    dangerous_keywords = ['DELETE', 'DROP', 'TRUNCATE', 'ALTER', 'CREATE']
    for keyword in dangerous_keywords:
        if keyword in query_upper:
            result['errors'].append(f'Dangerous keyword detected: {keyword}')
            result['is_valid'] = False
    
    # Check for balanced parentheses
    if query.count('(') != query.count(')'):
        result['errors'].append('Unbalanced parentheses')
        result['is_valid'] = False
    
    # Warnings
    if 'WITH (NOLOCK)' not in query_upper:
        result['warnings'].append('Consider using WITH (NOLOCK) for read queries')
    
    if 'ORDER BY' not in query_upper and 'GROUP BY' not in query_upper:
        result['warnings'].append('Consider adding ORDER BY for consistent results')
    
    return result