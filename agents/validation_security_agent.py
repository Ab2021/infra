"""
Validation and Security Agent - Quality guardian
Ensures SQL safety, correctness, and performance optimization
"""

import re
import sqlparse
from typing import Dict, List, Optional, Any
from datetime import datetime

class ValidationSecurityAgent:
    """
    Comprehensive validation and security agent that ensures
    SQL queries are safe, syntactically correct, and optimized.
    """
    
    def __init__(self, memory_system, database_connector):
        self.memory_system = memory_system
        self.database_connector = database_connector
        self.agent_name = "validator"
        
        # Security patterns to detect and prevent
        self.security_patterns = [
            r";\s*(DROP|DELETE|TRUNCATE|ALTER)\s+",
            r"UNION\s+SELECT.*--",
            r"1\s*=\s*1",
            r"OR\s+1\s*=\s*1",
            r"--\s*$",
            r"/\*.*\*/"
        ]
    
    async def validate_sql(self, sql: str, context: Dict) -> Dict[str, Any]:
        """
        Comprehensive SQL validation including syntax, security, and performance.
        
        Args:
            sql: SQL query to validate
            context: Validation context with intent and schema info
            
        Returns:
            Validation results with recommendations
        """
        
        validation_result = {
            "is_valid": True,
            "security_passed": True,
            "syntax_valid": True,
            "performance_warnings": [],
            "security_issues": [],
            "syntax_errors": [],
            "recommendations": [],
            "confidence_score": 1.0
        }
        
        try:
            # 1. Security validation
            security_result = self._validate_security(sql)
            validation_result.update(security_result)
            
            # 2. Syntax validation
            syntax_result = self._validate_syntax(sql)
            validation_result.update(syntax_result)
            
            # 3. Performance analysis
            performance_result = await self._analyze_performance(sql, context)
            validation_result.update(performance_result)
            
            # 4. Business logic validation
            business_result = self._validate_business_logic(sql, context)
            validation_result.update(business_result)
            
            # 5. Calculate overall validity
            validation_result["is_valid"] = (
                validation_result["security_passed"] and
                validation_result["syntax_valid"] and
                len(validation_result["security_issues"]) == 0
            )
            
            # 6. Generate recommendations
            validation_result["recommendations"] = self._generate_recommendations(validation_result)
            
            # 7. Update memory with validation results
            await self._update_memory(sql, validation_result)
            
            return validation_result
            
        except Exception as e:
            return {
                "is_valid": False,
                "error": str(e),
                "agent": self.agent_name,
                "recovery_strategy": "validation_failed"
            }
    
    def _validate_security(self, sql: str) -> Dict[str, Any]:
        """Validates SQL for security vulnerabilities."""
        
        security_issues = []
        sql_upper = sql.upper()
        
        # Check for dangerous patterns
        for pattern in self.security_patterns:
            if re.search(pattern, sql_upper, re.IGNORECASE):
                security_issues.append(f"Potential SQL injection pattern detected: {pattern}")
        
        # Check for unauthorized operations
        forbidden_operations = ["DROP", "DELETE", "TRUNCATE", "ALTER", "CREATE", "INSERT", "UPDATE"]
        for operation in forbidden_operations:
            if re.search(f"\\b{operation}\\b", sql_upper):
                security_issues.append(f"Unauthorized operation detected: {operation}")
        
        # Check for comment-based attacks
        if re.search(r"--", sql) or re.search(r"/\*.*\*/", sql):
            security_issues.append("SQL comments detected - potential security risk")
        
        return {
            "security_passed": len(security_issues) == 0,
            "security_issues": security_issues
        }
    
    def _validate_syntax(self, sql: str) -> Dict[str, Any]:
        """Validates SQL syntax using sqlparse."""
        
        syntax_errors = []
        
        try:
            # Parse the SQL
            parsed = sqlparse.parse(sql)
            
            if not parsed:
                syntax_errors.append("Unable to parse SQL - invalid syntax")
                return {"syntax_valid": False, "syntax_errors": syntax_errors}
            
            # Basic syntax checks
            statement = parsed[0]
            
            # Check for balanced parentheses
            if not self._check_balanced_parentheses(sql):
                syntax_errors.append("Unbalanced parentheses")
            
            # Check for valid statement type
            if not self._is_valid_statement_type(statement):
                syntax_errors.append("Invalid or unsupported statement type")
            
            # Check for proper quoting
            if not self._check_proper_quoting(sql):
                syntax_errors.append("Improper string quoting")
            
        except Exception as e:
            syntax_errors.append(f"Syntax parsing error: {str(e)}")
        
        return {
            "syntax_valid": len(syntax_errors) == 0,
            "syntax_errors": syntax_errors
        }
    
    async def _analyze_performance(self, sql: str, context: Dict) -> Dict[str, Any]:
        """Analyzes query for performance issues."""
        
        performance_warnings = []
        
        try:
            # Use database connector to get execution plan
            validation_result = await self.database_connector.validate_sql(sql)
            
            execution_plan = validation_result.get("execution_plan", "")
            estimated_cost = validation_result.get("estimated_cost", "UNKNOWN")
            
            # Analyze execution plan for performance issues
            if "FULL TABLE SCAN" in execution_plan.upper():
                performance_warnings.append("Query requires full table scan - consider adding indexes or filters")
            
            if estimated_cost == "HIGH":
                performance_warnings.append("Query has high estimated execution cost")
            
            # Check for common performance anti-patterns
            sql_upper = sql.upper()
            
            if "SELECT *" in sql_upper:
                performance_warnings.append("Using SELECT * - consider specifying only needed columns")
            
            if sql_upper.count("JOIN") > 5:
                performance_warnings.append("Complex query with many joins - consider optimization")
            
            if "ORDER BY" in sql_upper and "LIMIT" not in sql_upper:
                performance_warnings.append("ORDER BY without LIMIT can be expensive on large datasets")
            
            if "GROUP BY" in sql_upper and "HAVING" not in sql_upper and "WHERE" not in sql_upper:
                performance_warnings.append("GROUP BY without filtering - consider adding WHERE clause")
            
        except Exception as e:
            performance_warnings.append(f"Performance analysis failed: {str(e)}")
        
        return {"performance_warnings": performance_warnings}
    
    def _validate_business_logic(self, sql: str, context: Dict) -> Dict[str, Any]:
        """Validates business logic and query intent alignment."""
        
        recommendations = []
        
        intent = context.get("query_intent", {})
        primary_action = intent.get("primary_action", "")
        
        # Check alignment between intent and SQL
        sql_upper = sql.upper()
        
        if primary_action == "aggregate":
            if not any(agg in sql_upper for agg in ["SUM", "COUNT", "AVG", "MAX", "MIN"]):
                recommendations.append("Query intent suggests aggregation but no aggregate functions found")
        
        if primary_action == "filter":
            if "WHERE" not in sql_upper:
                recommendations.append("Query intent suggests filtering but no WHERE clause found")
        
        if primary_action == "sort":
            if "ORDER BY" not in sql_upper:
                recommendations.append("Query intent suggests sorting but no ORDER BY clause found")
        
        # Check for temporal queries
        temporal_scope = intent.get("temporal_scope")
        if temporal_scope and not any(date_func in sql_upper for date_func in ["DATE", "YEAR", "MONTH", "DAY"]):
            recommendations.append("Query has temporal scope but no date functions found")
        
        return {"business_logic_recommendations": recommendations}
    
    def _generate_recommendations(self, validation_result: Dict) -> List[str]:
        """Generates improvement recommendations based on validation results."""
        
        recommendations = []
        
        # Security recommendations
        if validation_result.get("security_issues"):
            recommendations.append("Review and fix security issues before execution")
        
        # Performance recommendations
        performance_warnings = validation_result.get("performance_warnings", [])
        if performance_warnings:
            recommendations.extend([f"Performance: {warning}" for warning in performance_warnings])
        
        # Syntax recommendations
        syntax_errors = validation_result.get("syntax_errors", [])
        if syntax_errors:
            recommendations.extend([f"Syntax: {error}" for error in syntax_errors])
        
        # Business logic recommendations
        business_recs = validation_result.get("business_logic_recommendations", [])
        recommendations.extend(business_recs)
        
        return recommendations
    
    def _check_balanced_parentheses(self, sql: str) -> bool:
        """Checks for balanced parentheses in SQL."""
        
        count = 0
        for char in sql:
            if char == '(':
                count += 1
            elif char == ')':
                count -= 1
                if count < 0:
                    return False
        
        return count == 0
    
    def _is_valid_statement_type(self, statement) -> bool:
        """Checks if the statement type is valid (SELECT only for our use case)."""
        
        # For our SQL agent, we only allow SELECT statements
        first_token = str(statement.tokens[0]).strip().upper()
        return first_token == "SELECT"
    
    def _check_proper_quoting(self, sql: str) -> bool:
        """Checks for proper string quoting."""
        
        # Basic check for unmatched quotes
        single_quotes = sql.count("'") 
        double_quotes = sql.count('"')
        
        # Should have even number of quotes (assuming no escaped quotes for simplicity)
        return single_quotes % 2 == 0 and double_quotes % 2 == 0
    
    async def _update_memory(self, sql: str, validation_result: Dict):
        """Updates memory with validation results."""
        
        await self.memory_system.working_memory.update_context(
            agent_name=self.agent_name,
            update_data={
                "validated_sql": sql,
                "validation_result": validation_result,
                "timestamp": datetime.now().isoformat()
            }
        )
