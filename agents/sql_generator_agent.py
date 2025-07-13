"""
SQL Generator Agent - Query crafting specialist
Generates optimized SQL queries using templates and learned patterns
"""

from typing import Dict, List, Optional, Tuple
from datetime import datetime
import json
import re

class SQLGeneratorAgent:
    """
    Sophisticated SQL generation agent that creates optimized queries
    using template-based generation and learned patterns.
    """
    
    def __init__(self, memory_system, llm_provider):
        self.memory_system = memory_system
        self.llm_provider = llm_provider
        self.agent_name = "sql_generator"
    
    async def generate_sql(self, intent: Dict, schema_context: Dict, entities: List[Dict]) -> Dict:
        """
        Generates SQL query based on intent, schema context, and entities.
        
        Args:
            intent: Structured query intent
            schema_context: Schema analysis results
            entities: Extracted entities
            
        Returns:
            Generated SQL with metadata
        """
        
        try:
            # Check for existing patterns in memory
            pattern_match = await self._find_matching_pattern(intent, entities)
            
            if pattern_match:
                # Use template-based generation
                generated_sql = await self._generate_from_template(
                    pattern_match, schema_context, entities
                )
                generation_strategy = "template_based"
                confidence = pattern_match.get("success_rate", 0.8)
            else:
                # Generate from scratch using LLM
                generated_sql = await self._generate_from_scratch(
                    intent, schema_context, entities
                )
                generation_strategy = "novel_generation"
                confidence = 0.7
            
            # Optimize the generated SQL
            optimized_sql = self._optimize_sql(generated_sql, schema_context)
            
            # Update memory with generation results
            await self._update_memory(intent, entities, optimized_sql, confidence)
            
            # Generate visualization metadata
            visualization_metadata = self._generate_visualization_metadata(
                optimized_sql, schema_context, intent
            )
            
            return {
                "generated_sql": optimized_sql,
                "generation_strategy": generation_strategy,
                "confidence": confidence,
                "alternatives": await self._generate_alternatives(optimized_sql),
                "optimization_applied": True,
                "visualization_metadata": visualization_metadata
            }
            
        except Exception as e:
            return {
                "error": str(e),
                "agent": self.agent_name,
                "recovery_strategy": "regenerate"
            }
    
    async def _find_matching_pattern(self, intent: Dict, entities: List[Dict]) -> Optional[Dict]:
        """Finds matching query patterns in memory."""
        
        # This would query the memory system for similar patterns
        return await self.memory_system.knowledge_memory.find_similar_patterns(
            intent=intent,
            entities=entities
        )
    
    async def _generate_from_template(self, pattern: Dict, schema_context: Dict, entities: List[Dict]) -> str:
        """Generates SQL using a template pattern."""
        
        template = pattern.get("sql_template", "")
        
        # Replace template variables with actual values
        variables = self._extract_template_variables(template)
        replacements = self._build_replacements(variables, schema_context, entities)
        
        sql = template
        for var, value in replacements.items():
            sql = sql.replace(f"{{{var}}}", value)
        
        return sql
    
    async def _generate_from_scratch(self, intent: Dict, schema_context: Dict, entities: List[Dict]) -> str:
        """Generates SQL from scratch using LLM."""
        
        prompt = self._create_sql_generation_prompt(intent, schema_context, entities)
        
        response = await self.llm_provider.ainvoke([{"role": "user", "content": prompt}])
        
        # Extract SQL from response
        sql = self._extract_sql_from_response(response.content)
        
        return sql
    
    def _create_sql_generation_prompt(self, intent: Dict, schema_context: Dict, entities: List[Dict]) -> str:
        """Creates prompt for SQL generation."""
        
        prompt = f"""
        Generate a SQL query based on the following requirements:
        
        Query Intent:
        {json.dumps(intent, indent=2)}
        
        Available Tables and Columns:
        """
        
        for table in schema_context.get("relevant_tables", []):
            table_name = table.get("table_name")
            columns = [col.get("name") for col in table.get("table_info", {}).get("columns", [])]
            prompt += f"\n{table_name}: {', '.join(columns)}"
        
        prompt += f"""
        
        Extracted Entities:
        {json.dumps(entities, indent=2)}
        
        Requirements:
        - Generate optimized SQL for Snowflake
        - Use appropriate JOINs based on relationships
        - Include proper filtering and aggregation
        - Follow SQL best practices
        - Return only the SQL query, no explanation
        """
        
        return prompt
    
    def _extract_sql_from_response(self, response: str) -> str:
        """Extracts SQL from LLM response."""
        
        # Remove markdown formatting
        if "```sql" in response:
            sql = response.split("```sql")[1].split("```")[0]
        elif "```" in response:
            sql = response.split("```")[1].split("```")[0]
        else:
            sql = response
        
        return sql.strip()
    
    def _optimize_sql(self, sql: str, schema_context: Dict) -> str:
        """Applies optimization techniques to the generated SQL."""
        
        # Basic optimization rules
        optimized = sql
        
        # Add table aliases for readability
        optimized = self._add_table_aliases(optimized)
        
        # Optimize WHERE clause ordering
        optimized = self._optimize_where_clause(optimized)
        
        # Add appropriate indexes hints if beneficial
        optimized = self._add_index_hints(optimized, schema_context)
        
        return optimized
    
    def _add_table_aliases(self, sql: str) -> str:
        """Adds table aliases for better readability."""
        # Simple implementation - would be more sophisticated in practice
        return sql
    
    def _optimize_where_clause(self, sql: str) -> str:
        """Optimizes WHERE clause for better performance."""
        # Simple implementation - would be more sophisticated in practice  
        return sql
    
    def _add_index_hints(self, sql: str, schema_context: Dict) -> str:
        """Adds index hints where beneficial."""
        # Simple implementation - would be more sophisticated in practice
        return sql
    
    async def _generate_alternatives(self, sql: str) -> List[str]:
        """Generates alternative SQL formulations."""
        
        # This would generate alternative approaches to the same query
        alternatives = []
        
        # Example: Different JOIN approaches, subquery vs CTE, etc.
        
        return alternatives
    
    def _extract_template_variables(self, template: str) -> List[str]:
        """Extracts variables from SQL template."""
        
        import re
        return re.findall(r'\{(\w+)\}', template)
    
    def _build_replacements(self, variables: List[str], schema_context: Dict, entities: List[Dict]) -> Dict[str, str]:
        """Builds replacement values for template variables."""
        
        replacements = {}
        
        # Map variables to actual values based on context
        for var in variables:
            if var == "table_name":
                tables = schema_context.get("relevant_tables", [])
                if tables:
                    replacements[var] = tables[0].get("table_name", "")
            elif var == "date_column":
                # Find date columns in schema
                for table in schema_context.get("relevant_tables", []):
                    for col in table.get("table_info", {}).get("columns", []):
                        if "date" in col.get("type", "").lower():
                            replacements[var] = col.get("name", "")
                            break
        
        return replacements
    
    def _generate_visualization_metadata(self, sql: str, schema_context: Dict, intent: Dict) -> Dict:
        """Generates visualization metadata to guide chart creation."""
        
        metadata = {
            "recommended_charts": [],
            "x_axis_suggestions": [],
            "y_axis_suggestions": [],
            "chart_priorities": {},
            "data_characteristics": {},
            "interactive_features": []
        }
        
        # Analyze SQL to determine data structure
        sql_upper = sql.upper()
        
        # Extract SELECT columns to understand output structure
        select_columns = self._extract_select_columns(sql)
        aggregate_functions = self._detect_aggregate_functions(sql)
        has_group_by = "GROUP BY" in sql_upper
        has_order_by = "ORDER BY" in sql_upper
        has_date_functions = any(func in sql_upper for func in ["DATE", "YEAR", "MONTH", "DAY", "DATEPART"])
        
        # Determine data characteristics
        metadata["data_characteristics"] = {
            "has_aggregations": bool(aggregate_functions),
            "has_grouping": has_group_by,
            "has_sorting": has_order_by,
            "has_temporal_data": has_date_functions,
            "column_count": len(select_columns),
            "likely_numeric_columns": self._identify_numeric_columns(select_columns, aggregate_functions),
            "likely_categorical_columns": self._identify_categorical_columns(select_columns, has_group_by)
        }
        
        # Generate chart recommendations based on query structure
        recommendations = self._recommend_charts_from_sql_structure(
            select_columns, aggregate_functions, has_group_by, has_date_functions, intent
        )
        metadata["recommended_charts"] = recommendations
        
        # Suggest axes
        if select_columns:
            if has_group_by and aggregate_functions:
                # GROUP BY column(s) for X-axis, aggregated values for Y-axis
                metadata["x_axis_suggestions"] = select_columns[:2]  # First few columns likely categorical
                metadata["y_axis_suggestions"] = [col for col in select_columns if any(agg in col.upper() for agg in aggregate_functions)]
            elif has_date_functions:
                # Date/time column for X-axis
                date_columns = [col for col in select_columns if any(date_word in col.lower() for date_word in ["date", "time", "year", "month"])]
                metadata["x_axis_suggestions"] = date_columns
                metadata["y_axis_suggestions"] = [col for col in select_columns if col not in date_columns]
            else:
                # First column X-axis, others Y-axis
                metadata["x_axis_suggestions"] = select_columns[:1]
                metadata["y_axis_suggestions"] = select_columns[1:]
        
        # Chart priorities based on query intent
        if intent.get("output_preference") == "chart":
            metadata["chart_priorities"]["user_requested"] = 1.0
        
        if intent.get("primary_action") == "aggregate":
            metadata["chart_priorities"]["bar_chart"] = 0.9
            metadata["chart_priorities"]["pie_chart"] = 0.7
        
        if intent.get("temporal_scope"):
            metadata["chart_priorities"]["line_chart"] = 0.9
            metadata["chart_priorities"]["area_chart"] = 0.7
        
        # Interactive features suggestions
        if len(select_columns) > 2:
            metadata["interactive_features"].append("filter_controls")
        
        if has_date_functions:
            metadata["interactive_features"].append("date_range_selector")
        
        if metadata["data_characteristics"]["likely_categorical_columns"]:
            metadata["interactive_features"].append("category_filter")
        
        return metadata
    
    def _extract_select_columns(self, sql: str) -> List[str]:
        """Extracts column names from SELECT clause."""
        try:
            # Simple extraction - could be enhanced with proper SQL parsing
            sql_upper = sql.upper()
            select_start = sql_upper.find("SELECT") + 6
            from_start = sql_upper.find("FROM")
            
            if select_start > 5 and from_start > select_start:
                select_clause = sql[select_start:from_start].strip()
                
                # Remove common SQL keywords and clean up
                select_clause = select_clause.replace("\n", " ").replace("\t", " ")
                columns = [col.strip() for col in select_clause.split(",")]
                
                # Clean column names (remove aliases, functions, etc.)
                cleaned_columns = []
                for col in columns:
                    # Extract actual column name (handle aliases)
                    if " AS " in col.upper():
                        col = col.split(" AS ")[0].strip()
                    elif " " in col and not any(func in col.upper() for func in ["SUM", "COUNT", "AVG", "MAX", "MIN"]):
                        col = col.split(" ")[-1].strip()
                    
                    # Remove table prefixes
                    if "." in col:
                        col = col.split(".")[-1]
                    
                    cleaned_columns.append(col)
                
                return cleaned_columns[:10]  # Limit to prevent issues
        except:
            pass
        
        return []
    
    def _detect_aggregate_functions(self, sql: str) -> List[str]:
        """Detects aggregate functions used in the query."""
        sql_upper = sql.upper()
        aggregate_functions = []
        
        common_aggregates = ["SUM", "COUNT", "AVG", "MAX", "MIN", "STDDEV", "VARIANCE"]
        
        for agg in common_aggregates:
            if f"{agg}(" in sql_upper:
                aggregate_functions.append(agg)
        
        return aggregate_functions
    
    def _identify_numeric_columns(self, columns: List[str], aggregates: List[str]) -> List[str]:
        """Identifies likely numeric columns based on names and aggregations."""
        numeric_indicators = ["amount", "price", "cost", "total", "sum", "count", "avg", "value", "revenue", "sales"]
        
        numeric_columns = []
        
        for col in columns:
            col_lower = col.lower()
            
            # If column has aggregation, likely numeric
            if any(agg.lower() in col_lower for agg in aggregates):
                numeric_columns.append(col)
            # If column name suggests numeric data
            elif any(indicator in col_lower for indicator in numeric_indicators):
                numeric_columns.append(col)
        
        return numeric_columns
    
    def _identify_categorical_columns(self, columns: List[str], has_group_by: bool) -> List[str]:
        """Identifies likely categorical columns."""
        categorical_indicators = ["name", "category", "type", "status", "region", "country", "city", "department"]
        
        categorical_columns = []
        
        for col in columns:
            col_lower = col.lower()
            
            # If query has GROUP BY, first columns likely categorical
            if has_group_by and len(categorical_columns) < 2:
                categorical_columns.append(col)
            # If column name suggests categorical data
            elif any(indicator in col_lower for indicator in categorical_indicators):
                categorical_columns.append(col)
        
        return categorical_columns
    
    def _recommend_charts_from_sql_structure(self, columns: List[str], aggregates: List[str], 
                                           has_group_by: bool, has_dates: bool, 
                                           intent: Dict) -> List[Dict]:
        """Recommends chart types based on SQL structure analysis."""
        recommendations = []
        
        # Bar chart for aggregated categorical data
        if has_group_by and aggregates:
            recommendations.append({
                "chart_type": "bar",
                "priority": 0.9,
                "rationale": "Aggregated data with grouping ideal for bar charts",
                "x_axis": columns[0] if columns else None,
                "y_axis": columns[1] if len(columns) > 1 else None
            })
        
        # Line chart for time series data
        if has_dates and len(columns) >= 2:
            recommendations.append({
                "chart_type": "line",
                "priority": 0.85,
                "rationale": "Temporal data perfect for trend visualization",
                "x_axis": columns[0],
                "y_axis": columns[1]
            })
        
        # Pie chart for categorical composition
        if has_group_by and len(columns) == 2:
            recommendations.append({
                "chart_type": "pie",
                "priority": 0.7,
                "rationale": "Two-column grouped data suitable for composition chart",
                "x_axis": columns[0],
                "y_axis": columns[1]
            })
        
        # Scatter plot for correlation analysis
        if len(columns) >= 2 and not has_group_by:
            recommendations.append({
                "chart_type": "scatter",
                "priority": 0.6,
                "rationale": "Multiple numeric columns good for correlation analysis",
                "x_axis": columns[0],
                "y_axis": columns[1]
            })
        
        # Table view as fallback
        recommendations.append({
            "chart_type": "table",
            "priority": 0.5,
            "rationale": "Tabular display suitable for any data structure",
            "x_axis": None,
            "y_axis": None
        })
        
        return sorted(recommendations, key=lambda x: x["priority"], reverse=True)
    
    async def _update_memory(self, intent: Dict, entities: List[Dict], sql: str, confidence: float):
        """Updates memory with generation results."""
        
        await self.memory_system.working_memory.update_context(
            agent_name=self.agent_name,
            update_data={
                "generated_sql": sql,
                "intent_processed": intent,
                "entities_used": entities,
                "confidence": confidence,
                "timestamp": datetime.now().isoformat()
            }
        )
