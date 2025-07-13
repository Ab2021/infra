"""
Data Profiling Agent - Simplified
Profiles table data and determines appropriate filtering conditions
"""

import json
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
from collections import Counter
import asyncio

@dataclass
class ColumnProfile:
    """Profile information for a single column"""
    column_name: str
    table_name: str
    data_type: str
    unique_count: int
    total_count: int
    null_count: int
    sample_values: List[Any]
    value_distribution: Dict[str, int]
    min_value: Any = None
    max_value: Any = None
    avg_value: Any = None
    quality_score: float = 1.0

class DataProfilingAgent:
    """
    Agent 2: Profiles data in identified tables and suggests filtering conditions
    Connects to Snowflake to understand actual data patterns
    """
    
    def __init__(self, database_connector, memory_system):
        self.database_connector = database_connector
        self.memory_system = memory_system
        self.logger = logging.getLogger(__name__)
        
        # Configuration
        self.max_sample_size = 1000
        self.max_unique_values = 50
        self.profile_cache = {}  # Cache profiles to avoid repeated queries
    
    async def process(self, tables: List[Dict], columns: List[Dict], intent: Dict) -> Dict:
        """
        Main processing method for data profiling
        
        Args:
            tables: List of identified tables from Agent 1
            columns: List of required columns from Agent 1
            intent: Query intent from Agent 1
            
        Returns:
            Dict with column profiles and suggested filters
        """
        try:
            self.logger.info(f"Profiling {len(tables)} tables and {len(columns)} columns")
            
            # Step 1: Profile all required columns
            column_profiles = await self._profile_columns(tables, columns)
            
            # Step 2: Analyze data relationships
            relationships = await self._analyze_relationships(tables, column_profiles)
            
            # Step 3: Generate filter suggestions based on intent and data
            filter_suggestions = await self._generate_filter_suggestions(
                column_profiles, intent, relationships
            )
            
            # Step 4: Calculate data quality metrics
            quality_metrics = self._calculate_quality_metrics(column_profiles)
            
            # Step 5: Store profiling insights in memory
            await self._store_profiling_insights(column_profiles, filter_suggestions)
            
            return {
                "column_profiles": {
                    f"{p.table_name}.{p.column_name}": p.__dict__ 
                    for p in column_profiles
                },
                "relationships": relationships,
                "suggested_filters": filter_suggestions,
                "quality_metrics": quality_metrics,
                "profiling_timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error profiling data: {e}")
            return {
                "error": str(e),
                "column_profiles": {},
                "suggested_filters": []
            }
    
    async def _profile_columns(self, tables: List[Dict], columns: List[Dict]) -> List[ColumnProfile]:
        """Profile each required column by sampling data"""
        
        profiles = []
        
        # Group columns by table for efficient querying
        columns_by_table = {}
        for column in columns:
            table_name = column["table"]
            if table_name not in columns_by_table:
                columns_by_table[table_name] = []
            columns_by_table[table_name].append(column)
        
        # Profile each table's columns
        for table_name, table_columns in columns_by_table.items():
            table_profiles = await self._profile_table_columns(table_name, table_columns)
            profiles.extend(table_profiles)
        
        return profiles
    
    async def _profile_table_columns(self, table_name: str, columns: List[Dict]) -> List[ColumnProfile]:
        """Profile all columns for a specific table"""
        
        profiles = []
        
        try:
            # Build profiling SQL for all columns at once
            column_names = [col["name"] for col in columns]
            profiling_sql = self._build_profiling_sql(table_name, column_names)
            
            # Execute profiling query
            profile_results = await self.database_connector.execute_query(profiling_sql)
            
            # Process results for each column
            for column in columns:
                column_name = column["name"]
                
                # Extract profile data for this column
                profile = await self._extract_column_profile(
                    table_name, column_name, profile_results
                )
                
                if profile:
                    profiles.append(profile)
                
        except Exception as e:
            self.logger.warning(f"Failed to profile table {table_name}: {e}")
            
            # Create basic profiles for columns even if detailed profiling fails
            for column in columns:
                basic_profile = ColumnProfile(
                    column_name=column["name"],
                    table_name=table_name,
                    data_type="unknown",
                    unique_count=0,
                    total_count=0,
                    null_count=0,
                    sample_values=[],
                    value_distribution={},
                    quality_score=0.5
                )
                profiles.append(basic_profile)
        
        return profiles
    
    def _build_profiling_sql(self, table_name: str, column_names: List[str]) -> str:
        """Build comprehensive profiling SQL for multiple columns"""
        
        # Basic statistics for each column
        column_stats = []
        for col in column_names:
            column_stats.extend([
                f"COUNT(DISTINCT {col}) as {col}_unique_count",
                f"COUNT({col}) as {col}_non_null_count",
                f"COUNT(*) - COUNT({col}) as {col}_null_count"
            ])
            
            # Add type-specific statistics
            column_stats.extend([
                f"CASE WHEN TYPEOF({col}) IN ('NUMBER', 'FLOAT') THEN MIN({col})::VARCHAR ELSE NULL END as {col}_min",
                f"CASE WHEN TYPEOF({col}) IN ('NUMBER', 'FLOAT') THEN MAX({col})::VARCHAR ELSE NULL END as {col}_max",
                f"CASE WHEN TYPEOF({col}) IN ('NUMBER', 'FLOAT') THEN AVG({col})::VARCHAR ELSE NULL END as {col}_avg"
            ])
        
        # Sample values query (separate)
        sample_sql = f"""
        WITH sample_data AS (
            SELECT {', '.join(column_names)}
            FROM {table_name}
            SAMPLE (1000 ROWS)
        ),
        stats AS (
            SELECT 
                {', '.join(column_stats)}
            FROM {table_name}
        )
        SELECT 
            'stats' as query_type,
            {', '.join([f"s.{stat}" for stat in column_stats])}
        FROM stats s
        
        UNION ALL
        
        SELECT 
            'sample' as query_type,
            {', '.join([f"sd.{col}::VARCHAR" for col in column_names])},
            {', '.join(['NULL'] * (len(column_stats) - len(column_names)))}
        FROM sample_data sd
        LIMIT 100
        """
        
        return sample_sql
    
    async def _extract_column_profile(self, table_name: str, column_name: str, 
                                    profile_results: List[Dict]) -> Optional[ColumnProfile]:
        """Extract profile information for a specific column from query results"""
        
        try:
            # Separate stats and sample data
            stats_row = None
            sample_values = []
            
            for row in profile_results:
                if row.get("query_type") == "stats":
                    stats_row = row
                elif row.get("query_type") == "sample":
                    value = row.get(column_name)
                    if value is not None:
                        sample_values.append(value)
            
            if not stats_row:
                return None
            
            # Extract statistics
            unique_count = stats_row.get(f"{column_name}_unique_count", 0)
            non_null_count = stats_row.get(f"{column_name}_non_null_count", 0)
            null_count = stats_row.get(f"{column_name}_null_count", 0)
            total_count = non_null_count + null_count
            
            min_value = stats_row.get(f"{column_name}_min")
            max_value = stats_row.get(f"{column_name}_max")
            avg_value = stats_row.get(f"{column_name}_avg")
            
            # Analyze sample values
            value_distribution = {}
            if sample_values:
                # Count occurrences of each value
                value_counts = Counter(sample_values)
                
                # Keep only top values if too many unique values
                if len(value_counts) > self.max_unique_values:
                    top_values = value_counts.most_common(self.max_unique_values)
                    value_distribution = dict(top_values)
                else:
                    value_distribution = dict(value_counts)
            
            # Infer data type from sample values
            data_type = self._infer_data_type(sample_values)
            
            # Calculate quality score
            quality_score = self._calculate_column_quality(
                total_count, null_count, unique_count, sample_values
            )
            
            return ColumnProfile(
                column_name=column_name,
                table_name=table_name,
                data_type=data_type,
                unique_count=unique_count,
                total_count=total_count,
                null_count=null_count,
                sample_values=sample_values[:20],  # Keep first 20 samples
                value_distribution=value_distribution,
                min_value=min_value,
                max_value=max_value,
                avg_value=avg_value,
                quality_score=quality_score
            )
            
        except Exception as e:
            self.logger.warning(f"Failed to extract profile for {column_name}: {e}")
            return None
    
    def _infer_data_type(self, sample_values: List[Any]) -> str:
        """Infer data type from sample values"""
        
        if not sample_values:
            return "unknown"
        
        # Remove None values for analysis
        non_null_values = [v for v in sample_values if v is not None]
        
        if not non_null_values:
            return "null"
        
        # Check for numeric types
        try:
            numeric_values = [float(v) for v in non_null_values]
            
            # Check if all are integers
            if all(float(v).is_integer() for v in non_null_values):
                return "integer"
            else:
                return "float"
        except (ValueError, TypeError):
            pass
        
        # Check for date types
        date_indicators = ["date", "time", "timestamp"]
        sample_str = str(non_null_values[0]).lower()
        if any(indicator in sample_str for indicator in date_indicators):
            return "date"
        
        # Check for boolean
        unique_values = set(str(v).lower() for v in non_null_values)
        if unique_values.issubset({"true", "false", "1", "0", "yes", "no"}):
            return "boolean"
        
        # Check for categorical (limited unique values)
        if len(set(non_null_values)) <= 20:
            return "categorical"
        
        return "text"
    
    def _calculate_column_quality(self, total_count: int, null_count: int, 
                                unique_count: int, sample_values: List[Any]) -> float:
        """Calculate data quality score for a column"""
        
        if total_count == 0:
            return 0.0
        
        quality_score = 1.0
        
        # Penalize high null percentage
        null_percentage = null_count / total_count
        quality_score -= null_percentage * 0.5
        
        # Reward appropriate uniqueness
        uniqueness_ratio = unique_count / total_count if total_count > 0 else 0
        
        if 0.1 <= uniqueness_ratio <= 0.9:  # Good uniqueness range
            quality_score += 0.1
        elif uniqueness_ratio < 0.01:  # Too few unique values
            quality_score -= 0.2
        
        # Check for data consistency in samples
        if sample_values:
            consistent_types = len(set(type(v).__name__ for v in sample_values if v is not None)) == 1
            if consistent_types:
                quality_score += 0.1
            else:
                quality_score -= 0.2
        
        return max(0.0, min(1.0, quality_score))
    
    async def _analyze_relationships(self, tables: List[Dict], profiles: List[ColumnProfile]) -> Dict:
        """Analyze relationships between tables and columns"""
        
        relationships = {
            "potential_joins": [],
            "common_columns": [],
            "data_consistency": {}
        }
        
        try:
            # Group profiles by table
            profiles_by_table = {}
            for profile in profiles:
                table = profile.table_name
                if table not in profiles_by_table:
                    profiles_by_table[table] = []
                profiles_by_table[table].append(profile)
            
            # Find potential join columns
            if len(profiles_by_table) > 1:
                table_names = list(profiles_by_table.keys())
                
                for i, table1 in enumerate(table_names):
                    for table2 in table_names[i+1:]:
                        joins = self._find_potential_joins(
                            profiles_by_table[table1],
                            profiles_by_table[table2]
                        )
                        relationships["potential_joins"].extend(joins)
            
            # Find common column patterns
            all_column_names = [p.column_name for p in profiles]
            column_counts = Counter(all_column_names)
            
            for column_name, count in column_counts.items():
                if count > 1:  # Column appears in multiple tables
                    relationships["common_columns"].append({
                        "column_name": column_name,
                        "appears_in_tables": count,
                        "tables": [p.table_name for p in profiles if p.column_name == column_name]
                    })
            
        except Exception as e:
            self.logger.warning(f"Failed to analyze relationships: {e}")
        
        return relationships
    
    def _find_potential_joins(self, table1_profiles: List[ColumnProfile], 
                            table2_profiles: List[ColumnProfile]) -> List[Dict]:
        """Find potential join conditions between two tables"""
        
        potential_joins = []
        
        for profile1 in table1_profiles:
            for profile2 in table2_profiles:
                
                # Check for same column name (strong indicator)
                if profile1.column_name == profile2.column_name:
                    potential_joins.append({
                        "type": "same_name_join",
                        "table1": profile1.table_name,
                        "column1": profile1.column_name,
                        "table2": profile2.table_name,
                        "column2": profile2.column_name,
                        "confidence": 0.8
                    })
                
                # Check for ID-like patterns
                elif self._is_id_pattern_match(profile1, profile2):
                    potential_joins.append({
                        "type": "id_pattern_join",
                        "table1": profile1.table_name,
                        "column1": profile1.column_name,
                        "table2": profile2.table_name,
                        "column2": profile2.column_name,
                        "confidence": 0.6
                    })
        
        return potential_joins
    
    def _is_id_pattern_match(self, profile1: ColumnProfile, profile2: ColumnProfile) -> bool:
        """Check if two columns follow ID matching patterns"""
        
        # Common ID patterns
        id_patterns = [
            (f"{profile1.table_name}_id", profile1.column_name),
            (f"{profile2.table_name}_id", profile2.column_name),
            ("id", profile1.column_name),
            ("id", profile2.column_name)
        ]
        
        for pattern, column in id_patterns:
            if pattern == column.lower():
                return True
        
        return False
    
    async def _generate_filter_suggestions(self, profiles: List[ColumnProfile], 
                                         intent: Dict, relationships: Dict) -> List[Dict]:
        """Generate intelligent filter suggestions based on data analysis and intent"""
        
        suggestions = []
        
        try:
            # Process intent filters
            intent_filters = intent.get("filters", {})
            time_scope = intent.get("time_scope")
            
            for profile in profiles:
                column_suggestions = self._generate_column_filters(profile, intent_filters, time_scope)
                suggestions.extend(column_suggestions)
            
            # Add relationship-based filters
            relationship_filters = self._generate_relationship_filters(relationships, intent)
            suggestions.extend(relationship_filters)
            
            # Sort by priority and confidence
            suggestions.sort(key=lambda x: (x.get("priority", 0), x.get("confidence", 0)), reverse=True)
            
        except Exception as e:
            self.logger.warning(f"Failed to generate filter suggestions: {e}")
        
        return suggestions[:10]  # Return top 10 suggestions
    
    def _generate_column_filters(self, profile: ColumnProfile, intent_filters: Dict, 
                               time_scope: str) -> List[Dict]:
        """Generate filter suggestions for a specific column"""
        
        suggestions = []
        
        # Time-based filters
        if time_scope and profile.data_type == "date":
            time_filter = self._create_time_filter(profile, time_scope)
            if time_filter:
                suggestions.append(time_filter)
        
        # Categorical filters
        if profile.data_type == "categorical" and profile.value_distribution:
            categorical_filters = self._create_categorical_filters(profile)
            suggestions.extend(categorical_filters)
        
        # Numeric range filters
        if profile.data_type in ["integer", "float"] and profile.min_value and profile.max_value:
            numeric_filter = self._create_numeric_filter(profile)
            if numeric_filter:
                suggestions.append(numeric_filter)
        
        # Quality-based filters
        if profile.quality_score < 0.8:
            quality_filter = self._create_quality_filter(profile)
            if quality_filter:
                suggestions.append(quality_filter)
        
        # Intent-based filters
        for filter_key, filter_value in intent_filters.items():
            if filter_key.lower() in profile.column_name.lower():
                intent_filter = self._create_intent_filter(profile, filter_key, filter_value)
                if intent_filter:
                    suggestions.append(intent_filter)
        
        return suggestions
    
    def _create_time_filter(self, profile: ColumnProfile, time_scope: str) -> Optional[Dict]:
        """Create time-based filter suggestion"""
        
        time_conditions = {
            "2024": f"YEAR({profile.column_name}) = 2024",
            "2023": f"YEAR({profile.column_name}) = 2023",
            "Q1": f"QUARTER({profile.column_name}) = 1",
            "Q2": f"QUARTER({profile.column_name}) = 2",
            "Q3": f"QUARTER({profile.column_name}) = 3",
            "Q4": f"QUARTER({profile.column_name}) = 4",
            "last year": f"{profile.column_name} >= DATEADD(year, -1, CURRENT_DATE())",
            "this month": f"{profile.column_name} >= DATEADD(month, -1, CURRENT_DATE())"
        }
        
        for scope, condition in time_conditions.items():
            if scope.lower() in time_scope.lower():
                return {
                    "type": "time_filter",
                    "column": f"{profile.table_name}.{profile.column_name}",
                    "condition": condition,
                    "description": f"Filter {profile.column_name} for {time_scope}",
                    "confidence": 0.9,
                    "priority": 10
                }
        
        return None
    
    def _create_categorical_filters(self, profile: ColumnProfile) -> List[Dict]:
        """Create categorical filter suggestions"""
        
        suggestions = []
        
        # Suggest filters for top categories
        for value, count in list(profile.value_distribution.items())[:5]:
            if count > 1:  # Only suggest if value appears multiple times
                suggestions.append({
                    "type": "categorical_filter",
                    "column": f"{profile.table_name}.{profile.column_name}",
                    "condition": f"{profile.column_name} = '{value}'",
                    "description": f"Filter {profile.column_name} = {value} ({count} records)",
                    "confidence": 0.7,
                    "priority": 5
                })
        
        return suggestions
    
    def _create_numeric_filter(self, profile: ColumnProfile) -> Optional[Dict]:
        """Create numeric range filter suggestion"""
        
        try:
            min_val = float(profile.min_value)
            max_val = float(profile.max_value)
            avg_val = float(profile.avg_value) if profile.avg_value else (min_val + max_val) / 2
            
            # Suggest filtering above average
            return {
                "type": "numeric_filter",
                "column": f"{profile.table_name}.{profile.column_name}",
                "condition": f"{profile.column_name} >= {avg_val}",
                "description": f"Filter {profile.column_name} above average ({avg_val:.2f})",
                "confidence": 0.6,
                "priority": 3
            }
            
        except (ValueError, TypeError):
            return None
    
    def _create_quality_filter(self, profile: ColumnProfile) -> Optional[Dict]:
        """Create data quality filter suggestion"""
        
        if profile.null_count > 0:
            return {
                "type": "quality_filter",
                "column": f"{profile.table_name}.{profile.column_name}",
                "condition": f"{profile.column_name} IS NOT NULL",
                "description": f"Exclude NULL values from {profile.column_name}",
                "confidence": 0.8,
                "priority": 8
            }
        
        return None
    
    def _create_intent_filter(self, profile: ColumnProfile, filter_key: str, filter_value: any) -> Optional[Dict]:
        """Create filter based on user intent"""
        
        return {
            "type": "intent_filter",
            "column": f"{profile.table_name}.{profile.column_name}",
            "condition": f"{profile.column_name} = '{filter_value}'",
            "description": f"User requested filter: {filter_key} = {filter_value}",
            "confidence": 0.95,
            "priority": 15
        }
    
    def _generate_relationship_filters(self, relationships: Dict, intent: Dict) -> List[Dict]:
        """Generate filters based on table relationships"""
        
        suggestions = []
        
        # Suggest JOIN conditions
        for join in relationships.get("potential_joins", []):
            if join.get("confidence", 0) > 0.7:
                suggestions.append({
                    "type": "join_condition",
                    "condition": f"{join['table1']}.{join['column1']} = {join['table2']}.{join['column2']}",
                    "description": f"Join {join['table1']} and {join['table2']}",
                    "confidence": join.get("confidence", 0.5),
                    "priority": 12
                })
        
        return suggestions
    
    def _calculate_quality_metrics(self, profiles: List[ColumnProfile]) -> Dict:
        """Calculate overall data quality metrics"""
        
        if not profiles:
            return {"overall_quality": 0.0}
        
        quality_scores = [p.quality_score for p in profiles]
        
        return {
            "overall_quality": sum(quality_scores) / len(quality_scores),
            "high_quality_columns": len([p for p in profiles if p.quality_score > 0.8]),
            "low_quality_columns": len([p for p in profiles if p.quality_score < 0.5]),
            "total_columns": len(profiles),
            "columns_with_nulls": len([p for p in profiles if p.null_count > 0]),
            "highly_unique_columns": len([p for p in profiles if p.unique_count / max(p.total_count, 1) > 0.8])
        }
    
    async def _store_profiling_insights(self, profiles: List[ColumnProfile], 
                                      suggestions: List[Dict]):
        """Store profiling insights in memory for future use"""
        
        try:
            # Store column profiles
            for profile in profiles:
                await self.memory_system.store_schema_insight(
                    table_name=profile.table_name,
                    column_name=profile.column_name,
                    insight_type="column_profile",
                    insight_data={
                        "data_type": profile.data_type,
                        "quality_score": profile.quality_score,
                        "unique_count": profile.unique_count,
                        "sample_values": profile.sample_values[:5]  # Store top 5 samples
                    },
                    confidence_score=profile.quality_score
                )
            
            # Store successful filter patterns
            high_confidence_filters = [s for s in suggestions if s.get("confidence", 0) > 0.8]
            if high_confidence_filters:
                await self.memory_system.store_query_pattern(
                    pattern_type="data_profiling",
                    pattern_description="High-confidence filter suggestions",
                    example_queries=[],
                    metadata={"filters": high_confidence_filters}
                )
                
        except Exception as e:
            self.logger.warning(f"Failed to store profiling insights: {e}")