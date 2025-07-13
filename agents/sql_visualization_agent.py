"""
SQL Visualization Agent - Simplified
Combines SQL generation, validation, and visualization into one focused agent
"""

import json
import logging
import re
from typing import Dict, List, Optional, Any
from datetime import datetime
from dataclasses import dataclass

@dataclass
class SQLResult:
    """Result from SQL generation"""
    sql_query: str
    query_type: str  # SELECT, INSERT, UPDATE, DELETE
    estimated_rows: int
    complexity_score: float
    performance_warnings: List[str]
    guardrails_passed: bool

@dataclass
class ChartConfig:
    """Chart configuration for visualization"""
    chart_type: str  # bar_chart, line_chart, scatter_plot, pie_chart, table
    title: str
    x_axis: str
    y_axis: Optional[str] = None
    color_column: Optional[str] = None
    aggregation: Optional[str] = None
    chart_config: Dict[str, Any] = None

class SQLVisualizationAgent:
    """
    Agent 3: Generates SQL queries with guardrails and determines visualization
    Combines SQL generation, validation, and chart recommendations
    """
    
    def __init__(self, llm_provider, memory_system, database_connector):
        self.llm_provider = llm_provider
        self.memory_system = memory_system
        self.database_connector = database_connector
        self.logger = logging.getLogger(__name__)
        
        # SQL Templates for common patterns
        self.sql_templates = {
            "aggregation": {
                "basic": "SELECT {dimensions}, {aggregation}({metric}) as {metric_alias} FROM {tables} {joins} {filters} GROUP BY {dimensions} ORDER BY {order_by}",
                "with_time": "SELECT {time_dimension}, {dimensions}, {aggregation}({metric}) as {metric_alias} FROM {tables} {joins} {filters} GROUP BY {time_dimension}, {dimensions} ORDER BY {time_dimension}, {order_by}"
            },
            "trend_analysis": "SELECT {time_dimension}, {aggregation}({metric}) as {metric_alias} FROM {tables} {joins} {filters} GROUP BY {time_dimension} ORDER BY {time_dimension}",
            "comparison": "SELECT {dimensions}, {aggregation}({metric}) as {metric_alias} FROM {tables} {joins} {filters} GROUP BY {dimensions} ORDER BY {metric_alias} DESC",
            "ranking": "SELECT {dimensions}, {aggregation}({metric}) as {metric_alias} FROM {tables} {joins} {filters} GROUP BY {dimensions} ORDER BY {metric_alias} {rank_direction} LIMIT {limit}",
            "filtering": "SELECT {columns} FROM {tables} {joins} {filters} ORDER BY {order_by} LIMIT {limit}"
        }
        
        # Chart type mappings
        self.chart_mappings = {
            "trend_analysis": "line_chart",
            "comparison": "bar_chart", 
            "ranking": "bar_chart",
            "aggregation": "bar_chart",
            "filtering": "table"
        }
        
        # Guardrails configuration
        self.guardrails = {
            "max_result_rows": 10000,
            "forbidden_keywords": ["DELETE", "DROP", "TRUNCATE", "ALTER", "CREATE"],
            "required_filters_for_large_tables": ["date", "time", "created_at"],
            "max_joins": 5,
            "timeout_seconds": 300
        }
    
    async def process(self, query_intent: Dict, column_profiles: Dict, 
                     suggested_filters: List[Dict]) -> Dict:
        """
        Main processing method for SQL generation and visualization
        
        Args:
            query_intent: Structured intent from Agent 1
            column_profiles: Column profiles from Agent 2
            suggested_filters: Filter suggestions from Agent 2
            
        Returns:
            Dict with SQL query, chart config, and plotting code
        """
        try:
            self.logger.info(f"Generating SQL for intent: {query_intent.get('action', 'unknown')}")
            
            # Step 1: Generate SQL query
            sql_result = await self._generate_sql(query_intent, column_profiles, suggested_filters)
            
            # Step 2: Apply guardrails and validation
            validation_result = await self._apply_guardrails(sql_result, column_profiles)
            
            # Step 3: Determine appropriate visualization
            chart_config = await self._determine_visualization(
                sql_result, query_intent, column_profiles
            )
            
            # Step 4: Generate plotting code
            plotting_code = self._generate_plotting_code(chart_config, sql_result)
            
            # Step 5: Store successful patterns
            if validation_result["guardrails_passed"]:
                await self._store_successful_pattern(query_intent, sql_result, chart_config)
            
            return {
                "sql_query": sql_result.sql_query,
                "sql_metadata": {
                    "query_type": sql_result.query_type,
                    "estimated_rows": sql_result.estimated_rows,
                    "complexity_score": sql_result.complexity_score,
                    "performance_warnings": sql_result.performance_warnings
                },
                "validation": validation_result,
                "chart_config": chart_config.__dict__,
                "plotting_code": plotting_code,
                "generation_timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error generating SQL and visualization: {e}")
            return {
                "error": str(e),
                "sql_query": "",
                "chart_config": {},
                "plotting_code": ""
            }
    
    async def _generate_sql(self, intent: Dict, column_profiles: Dict, 
                          filters: List[Dict]) -> SQLResult:
        """Generate SQL query based on intent and profiles"""
        
        try:
            # Determine query pattern
            action = intent.get("action", "aggregation")
            
            # Get similar successful queries from memory
            similar_queries = await self._get_similar_sql_patterns(intent)
            
            if similar_queries:
                # Adapt existing successful pattern
                sql_query = await self._adapt_existing_pattern(similar_queries[0], intent, column_profiles, filters)
            else:
                # Generate from template
                sql_query = self._generate_from_template(action, intent, column_profiles, filters)
            
            # Analyze generated SQL
            query_analysis = self._analyze_sql_query(sql_query)
            
            return SQLResult(
                sql_query=sql_query,
                query_type=query_analysis["type"],
                estimated_rows=query_analysis["estimated_rows"],
                complexity_score=query_analysis["complexity"],
                performance_warnings=query_analysis["warnings"],
                guardrails_passed=False  # Will be set by validation
            )
            
        except Exception as e:
            self.logger.error(f"SQL generation failed: {e}")
            raise e
    
    def _generate_from_template(self, action: str, intent: Dict, 
                              column_profiles: Dict, filters: List[Dict]) -> str:
        """Generate SQL from predefined templates"""
        
        # Get appropriate template
        template = self.sql_templates.get(action, self.sql_templates["aggregation"]["basic"])
        if isinstance(template, dict):
            # Choose sub-template based on intent
            if intent.get("time_scope"):
                template = template.get("with_time", template.get("basic", list(template.values())[0]))
            else:
                template = template.get("basic", list(template.values())[0])
        
        # Extract components from intent and profiles
        components = self._extract_sql_components(intent, column_profiles, filters)
        
        # Fill template
        try:
            sql_query = template.format(**components)
            
            # Clean up the query
            sql_query = self._clean_sql_query(sql_query)
            
            return sql_query
            
        except KeyError as e:
            self.logger.warning(f"Template formatting failed for key {e}, using fallback")
            return self._generate_fallback_sql(intent, column_profiles, filters)
    
    def _extract_sql_components(self, intent: Dict, column_profiles: Dict, 
                              filters: List[Dict]) -> Dict:
        """Extract SQL components from intent and profiles"""
        
        components = {
            "dimensions": "",
            "metric": "",
            "metric_alias": "",
            "aggregation": "SUM",
            "tables": "",
            "joins": "",
            "filters": "",
            "order_by": "",
            "limit": "1000",
            "time_dimension": "",
            "columns": "*",
            "rank_direction": "DESC"
        }
        
        # Extract metrics and dimensions
        metrics = intent.get("metrics", [])
        dimensions = intent.get("dimensions", [])
        
        # Find actual column names from profiles
        metric_columns = []
        dimension_columns = []
        all_tables = set()
        
        for profile_key, profile in column_profiles.items():
            table_name = profile["table_name"]
            column_name = profile["column_name"]
            all_tables.add(table_name)
            
            # Match metrics
            for metric in metrics:
                if metric.lower() in column_name.lower() or column_name.lower() in metric.lower():
                    metric_columns.append(f"{table_name}.{column_name}")
            
            # Match dimensions
            for dimension in dimensions:
                if dimension.lower() in column_name.lower() or column_name.lower() in dimension.lower():
                    dimension_columns.append(f"{table_name}.{column_name}")
            
            # Check for time columns
            if profile.get("data_type") == "date" or "date" in column_name.lower():
                if intent.get("time_scope"):
                    components["time_dimension"] = f"{table_name}.{column_name}"
        
        # Set components
        components["metric"] = metric_columns[0] if metric_columns else "COUNT(*)"
        components["metric_alias"] = metrics[0] if metrics else "total"
        components["dimensions"] = ", ".join(dimension_columns) if dimension_columns else "1"
        components["tables"] = ", ".join(sorted(all_tables))
        components["order_by"] = dimension_columns[0] if dimension_columns else components["metric_alias"]
        
        # Determine aggregation
        action = intent.get("action", "")
        if "count" in action.lower() or "count" in str(metrics).lower():
            components["aggregation"] = "COUNT"
        elif "average" in action.lower() or "avg" in str(metrics).lower():
            components["aggregation"] = "AVG"
        elif "max" in action.lower() or "maximum" in str(metrics).lower():
            components["aggregation"] = "MAX"
        elif "min" in action.lower() or "minimum" in str(metrics).lower():
            components["aggregation"] = "MIN"
        else:
            components["aggregation"] = "SUM"
        
        # Build filters
        filter_conditions = []
        for filter_item in filters:
            if filter_item.get("confidence", 0) > 0.6:  # Only high-confidence filters
                condition = filter_item.get("condition", "")
                if condition:
                    filter_conditions.append(condition)
        
        if filter_conditions:
            components["filters"] = "WHERE " + " AND ".join(filter_conditions)
        
        # Build joins (simplified)
        tables_list = list(all_tables)
        if len(tables_list) > 1:
            joins = self._generate_simple_joins(tables_list, column_profiles)
            components["joins"] = joins
        
        return components
    
    def _generate_simple_joins(self, tables: List[str], column_profiles: Dict) -> str:
        """Generate simple JOIN clauses for multiple tables"""
        
        if len(tables) < 2:
            return ""
        
        joins = []
        
        # Find common columns for joins
        columns_by_table = {}
        for profile_key, profile in column_profiles.items():
            table = profile["table_name"]
            column = profile["column_name"]
            
            if table not in columns_by_table:
                columns_by_table[table] = []
            columns_by_table[table].append(column)
        
        # Generate joins between consecutive tables
        for i in range(len(tables) - 1):
            table1 = tables[i]
            table2 = tables[i + 1]
            
            # Look for common columns
            table1_columns = set(columns_by_table.get(table1, []))
            table2_columns = set(columns_by_table.get(table2, []))
            common_columns = table1_columns & table2_columns
            
            if common_columns:
                join_column = list(common_columns)[0]
                joins.append(f"JOIN {table2} ON {table1}.{join_column} = {table2}.{join_column}")
            else:
                # Look for ID patterns
                id_column = self._find_id_column(table1, table2, columns_by_table)
                if id_column:
                    joins.append(f"JOIN {table2} ON {id_column}")
        
        return " ".join(joins) if joins else ""
    
    def _find_id_column(self, table1: str, table2: str, columns_by_table: Dict) -> Optional[str]:
        """Find ID column for joining two tables"""
        
        table1_columns = columns_by_table.get(table1, [])
        table2_columns = columns_by_table.get(table2, [])
        
        # Common ID patterns
        patterns = [
            ("id", "id"),
            (f"{table1}_id", "id"),
            ("id", f"{table1}_id"),
            (f"{table2}_id", "id"),
            ("id", f"{table2}_id")
        ]
        
        for pattern1, pattern2 in patterns:
            if pattern1 in table1_columns and pattern2 in table2_columns:
                return f"{table1}.{pattern1} = {table2}.{pattern2}"
        
        return None
    
    def _clean_sql_query(self, sql_query: str) -> str:
        """Clean and format SQL query"""
        
        # Remove extra whitespace
        sql_query = re.sub(r'\s+', ' ', sql_query.strip())
        
        # Remove empty clauses
        sql_query = re.sub(r'\bWHERE\s*GROUP\b', 'GROUP', sql_query)
        sql_query = re.sub(r'\bWHERE\s*ORDER\b', 'ORDER', sql_query)
        sql_query = re.sub(r'\bWHERE\s*$', '', sql_query)
        
        # Ensure proper semicolon
        if not sql_query.endswith(';'):
            sql_query += ';'
        
        return sql_query
    
    def _generate_fallback_sql(self, intent: Dict, column_profiles: Dict, 
                             filters: List[Dict]) -> str:
        """Generate simple fallback SQL when template fails"""
        
        # Get first table and some columns
        tables = set()
        columns = []
        
        for profile_key, profile in column_profiles.items():
            tables.add(profile["table_name"])
            columns.append(f"{profile['table_name']}.{profile['column_name']}")
        
        if not tables:
            return "SELECT 1 as fallback_query;"
        
        table_name = list(tables)[0]
        column_list = ", ".join(columns[:5])  # First 5 columns
        
        return f"SELECT {column_list} FROM {table_name} LIMIT 100;"
    
    def _analyze_sql_query(self, sql_query: str) -> Dict:
        """Analyze SQL query for complexity and performance"""
        
        analysis = {
            "type": "SELECT",
            "estimated_rows": 1000,
            "complexity": 0.5,
            "warnings": []
        }
        
        sql_upper = sql_query.upper()
        
        # Determine query type
        if sql_upper.startswith("SELECT"):
            analysis["type"] = "SELECT"
        elif sql_upper.startswith("INSERT"):
            analysis["type"] = "INSERT"
        elif sql_upper.startswith("UPDATE"):
            analysis["type"] = "UPDATE"
        elif sql_upper.startswith("DELETE"):
            analysis["type"] = "DELETE"
        
        # Calculate complexity
        complexity_factors = 0
        
        # Count JOINs
        join_count = sql_upper.count("JOIN")
        complexity_factors += join_count * 0.2
        
        # Count subqueries
        subquery_count = sql_upper.count("SELECT") - 1  # Minus main SELECT
        complexity_factors += subquery_count * 0.3
        
        # Count aggregations
        agg_functions = ["SUM", "COUNT", "AVG", "MAX", "MIN", "GROUP BY"]
        agg_count = sum(1 for func in agg_functions if func in sql_upper)
        complexity_factors += agg_count * 0.1
        
        analysis["complexity"] = min(1.0, complexity_factors)
        
        # Generate warnings
        if join_count > 3:
            analysis["warnings"].append("Query has many JOINs, may be slow")
        
        if "LIMIT" not in sql_upper:
            analysis["warnings"].append("No LIMIT clause, result set may be large")
        
        if subquery_count > 2:
            analysis["warnings"].append("Complex nested queries detected")
        
        # Estimate rows based on LIMIT clause
        limit_match = re.search(r'LIMIT\s+(\d+)', sql_upper)
        if limit_match:
            analysis["estimated_rows"] = int(limit_match.group(1))
        
        return analysis
    
    async def _apply_guardrails(self, sql_result: SQLResult, column_profiles: Dict) -> Dict:
        """Apply security and performance guardrails"""
        
        validation = {
            "guardrails_passed": True,
            "security_checks": [],
            "performance_checks": [],
            "recommendations": []
        }
        
        sql_upper = sql_result.sql_query.upper()
        
        # Security checks
        for forbidden in self.guardrails["forbidden_keywords"]:
            if forbidden in sql_upper:
                validation["guardrails_passed"] = False
                validation["security_checks"].append(f"Forbidden keyword detected: {forbidden}")
        
        # Check for SQL injection patterns
        injection_patterns = ["'", "--", "/*", "*/", ";", "UNION", "OR 1=1"]
        for pattern in injection_patterns:
            if pattern in sql_result.sql_query and pattern != ";" or sql_result.sql_query.count(";") > 1:
                validation["security_checks"].append(f"Potential injection pattern: {pattern}")
        
        # Performance checks
        if sql_result.estimated_rows > self.guardrails["max_result_rows"]:
            validation["performance_checks"].append(f"Result set too large: {sql_result.estimated_rows} rows")
            validation["recommendations"].append("Add more specific filters or LIMIT clause")
        
        if sql_result.complexity_score > 0.8:
            validation["performance_checks"].append("Query complexity is high")
            validation["recommendations"].append("Consider simplifying the query")
        
        # Check for required filters on large tables
        large_tables = self._identify_large_tables(column_profiles)
        for table in large_tables:
            if table in sql_result.sql_query and not self._has_date_filter(sql_result.sql_query):
                validation["performance_checks"].append(f"Large table {table} missing date filter")
                validation["recommendations"].append(f"Add date filter for table {table}")
        
        # Overall validation
        if validation["security_checks"] or len(validation["performance_checks"]) > 2:
            validation["guardrails_passed"] = False
        
        sql_result.guardrails_passed = validation["guardrails_passed"]
        
        return validation
    
    def _identify_large_tables(self, column_profiles: Dict) -> List[str]:
        """Identify tables that are likely to be large"""
        
        large_tables = []
        
        for profile_key, profile in column_profiles.items():
            table_name = profile["table_name"]
            total_count = profile.get("total_count", 0)
            
            # Consider table large if it has many rows
            if total_count > 100000:  # 100K rows
                large_tables.append(table_name)
            
            # Also check table name patterns
            large_table_patterns = ["fact", "transaction", "event", "log"]
            if any(pattern in table_name.lower() for pattern in large_table_patterns):
                large_tables.append(table_name)
        
        return list(set(large_tables))
    
    def _has_date_filter(self, sql_query: str) -> bool:
        """Check if query has date-based filtering"""
        
        date_patterns = [
            r"date\s*>=",
            r"date\s*>",
            r"date\s*BETWEEN",
            r"YEAR\s*\(",
            r"MONTH\s*\(",
            r"QUARTER\s*\(",
            r"created_at\s*>=",
            r"timestamp\s*>="
        ]
        
        for pattern in date_patterns:
            if re.search(pattern, sql_query, re.IGNORECASE):
                return True
        
        return False
    
    async def _determine_visualization(self, sql_result: SQLResult, intent: Dict, 
                                     column_profiles: Dict) -> ChartConfig:
        """Determine appropriate visualization for the query results"""
        
        action = intent.get("action", "aggregation")
        dimensions = intent.get("dimensions", [])
        metrics = intent.get("metrics", [])
        
        # Default chart type based on action
        chart_type = self.chart_mappings.get(action, "bar_chart")
        
        # Analyze SQL to refine chart choice
        sql_analysis = self._analyze_sql_for_visualization(sql_result.sql_query)
        
        # Determine chart components
        x_axis = self._determine_x_axis(sql_analysis, dimensions, column_profiles)
        y_axis = self._determine_y_axis(sql_analysis, metrics, column_profiles)
        title = self._generate_chart_title(intent, x_axis, y_axis)
        
        # Refine chart type based on data characteristics
        chart_type = self._refine_chart_type(chart_type, sql_analysis, intent)
        
        # Create chart configuration
        chart_config = ChartConfig(
            chart_type=chart_type,
            title=title,
            x_axis=x_axis,
            y_axis=y_axis,
            chart_config=self._get_chart_specific_config(chart_type, sql_analysis)
        )
        
        return chart_config
    
    def _analyze_sql_for_visualization(self, sql_query: str) -> Dict:
        """Analyze SQL query to determine visualization characteristics"""
        
        analysis = {
            "has_aggregation": False,
            "has_groupby": False,
            "has_time_dimension": False,
            "has_multiple_metrics": False,
            "estimated_categories": 10
        }
        
        sql_upper = sql_query.upper()
        
        # Check for aggregation functions
        agg_functions = ["SUM", "COUNT", "AVG", "MAX", "MIN"]
        analysis["has_aggregation"] = any(func in sql_upper for func in agg_functions)
        
        # Check for GROUP BY
        analysis["has_groupby"] = "GROUP BY" in sql_upper
        
        # Check for time dimensions
        time_functions = ["YEAR", "MONTH", "QUARTER", "DATE"]
        analysis["has_time_dimension"] = any(func in sql_upper for func in time_functions)
        
        # Estimate number of categories
        if "LIMIT" in sql_upper:
            limit_match = re.search(r'LIMIT\s+(\d+)', sql_upper)
            if limit_match:
                analysis["estimated_categories"] = min(int(limit_match.group(1)), 20)
        
        return analysis
    
    def _determine_x_axis(self, sql_analysis: Dict, dimensions: List[str], 
                         column_profiles: Dict) -> str:
        """Determine X-axis for the chart"""
        
        # Look for dimension columns in the query
        for profile_key, profile in column_profiles.items():
            column_name = profile["column_name"]
            
            # Check if this column appears in dimensions
            for dimension in dimensions:
                if dimension.lower() in column_name.lower():
                    return column_name
            
            # Check for categorical or date columns
            if profile.get("data_type") in ["categorical", "date"]:
                return column_name
        
        # Fallback to first dimension or generic name
        return dimensions[0] if dimensions else "category"
    
    def _determine_y_axis(self, sql_analysis: Dict, metrics: List[str], 
                         column_profiles: Dict) -> Optional[str]:
        """Determine Y-axis for the chart"""
        
        # Look for metric columns
        for profile_key, profile in column_profiles.items():
            column_name = profile["column_name"]
            
            # Check if this column appears in metrics
            for metric in metrics:
                if metric.lower() in column_name.lower():
                    return column_name
            
            # Check for numeric columns
            if profile.get("data_type") in ["integer", "float"]:
                return column_name
        
        # If aggregation is present, return the metric alias
        if sql_analysis.get("has_aggregation"):
            return metrics[0] if metrics else "value"
        
        return None
    
    def _generate_chart_title(self, intent: Dict, x_axis: str, y_axis: Optional[str]) -> str:
        """Generate appropriate chart title"""
        
        action = intent.get("action", "")
        
        if action == "trend_analysis":
            return f"{y_axis or 'Value'} Trends Over Time"
        elif action == "comparison":
            return f"{y_axis or 'Value'} by {x_axis}"
        elif action == "ranking":
            return f"Top {x_axis} by {y_axis or 'Value'}"
        else:
            return f"{y_axis or 'Data'} by {x_axis}"
    
    def _refine_chart_type(self, initial_type: str, sql_analysis: Dict, intent: Dict) -> str:
        """Refine chart type based on SQL analysis"""
        
        # If time dimension is present, prefer line chart for trends
        if sql_analysis.get("has_time_dimension") and intent.get("action") == "trend_analysis":
            return "line_chart"
        
        # If no aggregation, prefer table view
        if not sql_analysis.get("has_aggregation") and not sql_analysis.get("has_groupby"):
            return "table"
        
        # If many categories, prefer horizontal bar chart
        if sql_analysis.get("estimated_categories", 0) > 10:
            return "table"  # Too many for chart
        
        # If only one metric and few categories, pie chart could work
        if sql_analysis.get("estimated_categories", 0) <= 5 and initial_type == "bar_chart":
            # Check if it makes sense for pie chart
            if intent.get("action") in ["comparison", "aggregation"]:
                return "pie_chart"
        
        return initial_type
    
    def _get_chart_specific_config(self, chart_type: str, sql_analysis: Dict) -> Dict:
        """Get chart-specific configuration"""
        
        config = {}
        
        if chart_type == "bar_chart":
            config = {
                "orientation": "vertical",
                "show_values": True,
                "color_scheme": "viridis"
            }
        elif chart_type == "line_chart":
            config = {
                "show_points": True,
                "smooth": False,
                "color_scheme": "blues"
            }
        elif chart_type == "pie_chart":
            config = {
                "show_percentages": True,
                "explode_largest": False
            }
        elif chart_type == "table":
            config = {
                "sortable": True,
                "searchable": True,
                "page_size": 20
            }
        
        return config
    
    def _generate_plotting_code(self, chart_config: ChartConfig, sql_result: SQLResult) -> str:
        """Generate Streamlit plotting code"""
        
        chart_type = chart_config.chart_type
        
        if chart_type == "bar_chart":
            return f"""
# Bar Chart
if not df.empty:
    if '{chart_config.x_axis}' in df.columns:
        chart_data = df.set_index('{chart_config.x_axis}')
        st.bar_chart(chart_data['{chart_config.y_axis or 'value'}'] if '{chart_config.y_axis or 'value'}' in chart_data.columns else chart_data)
    else:
        st.dataframe(df)
    
    st.caption("📊 {chart_config.title}")
"""
        
        elif chart_type == "line_chart":
            return f"""
# Line Chart
if not df.empty:
    if '{chart_config.x_axis}' in df.columns:
        chart_data = df.set_index('{chart_config.x_axis}')
        st.line_chart(chart_data['{chart_config.y_axis or 'value'}'] if '{chart_config.y_axis or 'value'}' in chart_data.columns else chart_data)
    else:
        st.dataframe(df)
    
    st.caption("📈 {chart_config.title}")
"""
        
        elif chart_type == "pie_chart":
            return f"""
# Pie Chart using Plotly
import plotly.express as px

if not df.empty and len(df) <= 10:
    fig = px.pie(df, 
                values='{chart_config.y_axis or df.columns[-1]}', 
                names='{chart_config.x_axis or df.columns[0]}',
                title='{chart_config.title}')
    st.plotly_chart(fig, use_container_width=True)
else:
    st.dataframe(df)
    st.caption("🥧 Data shown as table (too many categories for pie chart)")
"""
        
        else:  # table or fallback
            return f"""
# Data Table
if not df.empty:
    st.dataframe(df, use_container_width=True)
    st.caption(f"📋 {chart_config.title} - {{len(df)}} rows")
    
    # Add download button
    csv = df.to_csv(index=False)
    st.download_button(
        label="📥 Download CSV",
        data=csv,
        file_name=f"query_results_{{datetime.now().strftime('%Y%m%d_%H%M%S')}}.csv",
        mime="text/csv"
    )
else:
    st.warning("No data returned from query")
"""
    
    async def _get_similar_sql_patterns(self, intent: Dict) -> List[Dict]:
        """Get similar SQL patterns from memory"""
        
        try:
            # Create a query description for similarity search
            query_description = f"{intent.get('action', '')} {' '.join(intent.get('metrics', []))} {' '.join(intent.get('dimensions', []))}"
            
            similar_patterns = await self.memory_system.find_similar_queries(
                query=query_description,
                top_k=3
            )
            
            return similar_patterns
            
        except Exception as e:
            self.logger.warning(f"Failed to get similar patterns: {e}")
            return []
    
    async def _adapt_existing_pattern(self, pattern: Dict, intent: Dict, 
                                    column_profiles: Dict, filters: List[Dict]) -> str:
        """Adapt an existing successful SQL pattern to current intent"""
        
        base_sql = pattern.get("generated_sql", "")
        
        if not base_sql:
            return self._generate_from_template("aggregation", intent, column_profiles, filters)
        
        # Simple adaptation: replace table/column names
        adapted_sql = base_sql
        
        # Replace table names
        for profile_key, profile in column_profiles.items():
            table_name = profile["table_name"]
            column_name = profile["column_name"]
            
            # Look for similar column patterns in the base SQL
            base_sql_upper = base_sql.upper()
            if any(metric.upper() in base_sql_upper for metric in intent.get("metrics", [])):
                # Replace with actual column
                for metric in intent.get("metrics", []):
                    if metric.upper() in base_sql_upper:
                        adapted_sql = re.sub(
                            rf'\b{metric}\b', 
                            f"{table_name}.{column_name}", 
                            adapted_sql, 
                            flags=re.IGNORECASE
                        )
        
        return adapted_sql
    
    async def _store_successful_pattern(self, intent: Dict, sql_result: SQLResult, 
                                      chart_config: ChartConfig):
        """Store successful SQL and visualization patterns"""
        
        try:
            if sql_result.guardrails_passed and sql_result.complexity_score < 0.8:
                await self.memory_system.store_successful_query(
                    query=f"{intent.get('action', '')} query",
                    sql=sql_result.sql_query,
                    execution_time=0,  # Will be updated after actual execution
                    result_count=sql_result.estimated_rows,
                    tables_used=intent.get("dimensions", []) + intent.get("metrics", []),
                    metadata={
                        "intent": intent,
                        "chart_type": chart_config.chart_type,
                        "complexity": sql_result.complexity_score
                    }
                )
                
        except Exception as e:
            self.logger.warning(f"Failed to store successful pattern: {e}")