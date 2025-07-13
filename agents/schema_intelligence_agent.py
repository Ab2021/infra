"""
Schema Intelligence Agent
Database architecture expert that understands table relationships and optimizations.
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import asyncio
from datetime import datetime

@dataclass
class TableRelevance:
    """Represents a table's relevance to a query with explanation."""
    table_name: str
    relevance_score: float
    table_info: Dict
    match_reasons: List[str]
    performance_notes: List[str] = None

@dataclass
class ColumnAnalysis:
    """Detailed column analysis with data patterns."""
    column_name: str
    data_type: str
    unique_values: List[str]
    value_patterns: Dict[str, any]
    suggested_filters: List[Dict]
    data_quality_score: float

@dataclass
class SchemaAnalysisResult:
    """Complete schema analysis results."""
    relevant_tables: List[TableRelevance]
    table_relationships: Dict[str, List[str]]
    column_metadata: Dict[str, Dict]
    sample_data: Dict[str, List[Dict]]
    column_analysis: Dict[str, List[ColumnAnalysis]]
    data_patterns: Dict[str, any]
    filtering_suggestions: Dict[str, List[Dict]]
    confidence_score: float
    optimization_suggestions: List[str]

class SchemaIntelligenceAgent:
    """
    Sophisticated schema analysis agent that combines real-time database
    inspection with accumulated schema knowledge from memory.
    """
    
    def __init__(self, memory_system, database_connector):
        self.memory_system = memory_system
        self.database_connector = database_connector
        self.agent_name = "schema_analyzer"
        
    async def analyze_schema_requirements(self, entities: List[Dict], intent: Dict, context: Dict) -> SchemaAnalysisResult:
        """
        Performs comprehensive database schema analysis to identify optimal tables and relationships.
        
        This method serves as the primary entry point for schema intelligence, combining 
        real-time database inspection with accumulated knowledge from memory to identify
        the most relevant tables, analyze their relationships, and provide intelligent
        filtering suggestions based on actual data patterns.
        
        Args:
            entities (List[Dict]): Extracted entities from NLU processing.
                Each entity contains:
                - type (str): Entity category ("table", "column", "value", "date", "metric")
                - value (str): Extracted text value
                - confidence (float): Extraction confidence score (0.0-1.0)
                - position (tuple, optional): Text position information
                
                Example: [
                    {"type": "table", "value": "sales", "confidence": 0.9},
                    {"type": "column", "value": "revenue", "confidence": 0.8},
                    {"type": "date", "value": "2023", "confidence": 0.95}
                ]
            
            intent (Dict): Structured query intent from NLU analysis.
                Contains:
                - primary_action (str): Main operation ("select", "aggregate", "filter", "join")
                - data_focus (str): Description of target data
                - output_preference (str): Desired output format ("table", "chart", "dashboard")
                - temporal_scope (str, optional): Time period specification
                - analysis_requirements (List[str]): Required analysis types
                
                Example: {
                    "primary_action": "aggregate",
                    "data_focus": "sales by region",
                    "output_preference": "chart",
                    "temporal_scope": "last quarter"
                }
            
            context (Dict): Session and memory context for enhanced analysis.
                Includes:
                - user_id (str): User identifier for personalization
                - session_id (str): Current session identifier
                - memory_context (Dict): Relevant memories from previous queries
                - conversation_history (List): Recent query context
        
        Returns:
            SchemaAnalysisResult: Comprehensive analysis results containing:
                - relevant_tables (List[TableRelevance]): Scored and ranked relevant tables
                - table_relationships (Dict[str, List[str]]): Foreign key relationships
                - column_metadata (Dict[str, Dict]): Detailed column information
                - sample_data (Dict[str, List[Dict]]): Representative data samples
                - column_analysis (Dict[str, List[ColumnAnalysis]]): Deep column analysis
                - data_patterns (Dict[str, Any]): Cross-table pattern detection
                - filtering_suggestions (Dict[str, List[Dict]]): Intelligent filter options
                - confidence_score (float): Overall analysis confidence (0.0-1.0)
                - optimization_suggestions (List[str]): Performance recommendations
        
        Raises:
            Exception: If schema analysis fails due to database connectivity issues,
                      invalid entity extraction, or memory system errors.
        
        Example:
            >>> entities = [{"type": "table", "value": "customers", "confidence": 0.9}]
            >>> intent = {"primary_action": "select", "data_focus": "customer data"}
            >>> context = {"user_id": "user123", "session_id": "sess456"}
            >>> result = await agent.analyze_schema_requirements(entities, intent, context)
            >>> print(result.relevant_tables[0].table_name)  # "customers"
            >>> print(result.confidence_score)  # 0.85
        
        Note:
            - Leverages memory system to improve table relevance over time
            - Performs real-time data sampling for pattern analysis
            - Generates intelligent filtering suggestions based on actual column values
            - Optimizes for both accuracy and query performance
            - Supports multi-table relationship detection and join recommendations
        """
        
        try:
            # Retrieve schema insights from long-term memory
            schema_insights = await self._get_schema_insights(entities, intent)
            
            # Get current database schema metadata
            current_schema = await self.database_connector.get_schema_metadata()
            
            # Enhance schema with memory insights
            enhanced_schema = self._enhance_schema_with_memory(current_schema, schema_insights)
            
            # Find relevant tables based on entities and intent
            relevant_tables = await self._identify_relevant_tables(entities, intent, enhanced_schema)
            
            # Analyze table relationships for query construction
            relationships = self._analyze_table_relationships(relevant_tables, enhanced_schema)
            
            # Extract detailed column metadata
            column_metadata = await self._get_column_metadata(relevant_tables)
            
            # Get representative sample data for context
            sample_data = await self._get_sample_data(relevant_tables)
            
            # Perform deep column analysis with pattern detection
            column_analysis = await self._analyze_columns_deeply(relevant_tables, sample_data, entities, intent)
            
            # Detect data patterns across tables
            data_patterns = self._detect_cross_table_patterns(sample_data, relevant_tables)
            
            # Generate intelligent filtering suggestions
            filtering_suggestions = self._generate_filtering_suggestions(column_analysis, entities, intent)
            
            # Calculate confidence score
            confidence = self._calculate_schema_confidence(relevant_tables, entities)
            
            # Generate optimization suggestions
            optimizations = self._generate_optimization_suggestions(relevant_tables, intent)
            
            # Update memory with new schema insights
            await self._update_schema_memory(entities, relevant_tables, intent, relationships)
            
            result = SchemaAnalysisResult(
                relevant_tables=relevant_tables,
                table_relationships=relationships,
                column_metadata=column_metadata,
                sample_data=sample_data,
                column_analysis=column_analysis,
                data_patterns=data_patterns,
                filtering_suggestions=filtering_suggestions,
                confidence_score=confidence,
                optimization_suggestions=optimizations
            )
            
            return result
            
        except Exception as e:
            raise Exception(f"Schema analysis failed: {str(e)}")
    
    async def _identify_relevant_tables(self, entities: List[Dict], intent: Dict, schema: Dict) -> List[TableRelevance]:
        """
        Uses sophisticated matching to identify most relevant database tables.
        """
        relevant_tables = []
        
        for table_name, table_info in schema.items():
            relevance_score = 0.0
            match_reasons = []
            
            # Entity-based relevance scoring
            for entity in entities:
                entity_value = entity["value"].lower()
                entity_type = entity.get("type", "")
                
                # Direct table name matching
                if entity_value in table_name.lower():
                    relevance_score += 0.8
                    match_reasons.append(f"Table name matches entity '{entity_value}'")
                
                # Column name matching
                for column in table_info.get("columns", []):
                    column_name = column.get("name", "").lower()
                    if entity_value in column_name:
                        relevance_score += 0.4
                        match_reasons.append(f"Column '{column_name}' matches entity '{entity_value}'")
                
                # Business description matching
                description = table_info.get("description", "").lower()
                if entity_value in description:
                    relevance_score += 0.3
                    match_reasons.append(f"Table description contains '{entity_value}'")
            
            # Intent-based relevance
            data_focus = intent.get("data_focus", "").lower()
            if data_focus and any(keyword in table_info.get("description", "").lower() 
                                for keyword in data_focus.split()):
                relevance_score += 0.5
                match_reasons.append("Table aligns with query intent")
            
            # Memory-based relevance from successful past queries
            memory_relevance = await self._get_memory_relevance_score(table_name, entities, intent)
            if memory_relevance > 0:
                relevance_score += memory_relevance
                match_reasons.append("Table used successfully in similar past queries")
            
            # Performance considerations
            performance_notes = self._get_performance_notes(table_name, table_info, intent)
            
            if relevance_score > 0.3:  # Relevance threshold
                relevant_tables.append(TableRelevance(
                    table_name=table_name,
                    relevance_score=relevance_score,
                    table_info=table_info,
                    match_reasons=match_reasons,
                    performance_notes=performance_notes
                ))
        
        # Sort by relevance score and return top candidates
        return sorted(relevant_tables, key=lambda x: x.relevance_score, reverse=True)[:10]
    
    def _analyze_table_relationships(self, relevant_tables: List[TableRelevance], schema: Dict) -> Dict[str, List[str]]:
        """
        Analyzes how relevant tables can be joined based on foreign key relationships.
        """
        relationships = {}
        
        for table in relevant_tables:
            table_name = table.table_name
            table_info = table.table_info
            relationships[table_name] = []
            
            # Find foreign key relationships
            for column in table_info.get("columns", []):
                if column.get("is_foreign_key"):
                    referenced_table = column.get("references_table")
                    if referenced_table and any(t.table_name == referenced_table for t in relevant_tables):
                        relationship = f"{table_name}.{column['name']} -> {referenced_table}.{column.get('references_column', 'id')}"
                        relationships[table_name].append(relationship)
        
        return relationships
    
    async def _get_sample_data(self, relevant_tables: List[TableRelevance], sample_size: int = 5) -> Dict[str, List[Dict]]:
        """
        Retrieves sample data from relevant tables for better context understanding.
        """
        sample_data = {}
        
        for table in relevant_tables[:3]:  # Limit to top 3 tables for performance
            try:
                samples = await self.database_connector.get_sample_data(table.table_name, sample_size)
                sample_data[table.table_name] = samples
            except Exception as e:
                sample_data[table.table_name] = []
                
        return sample_data
    
    def _calculate_schema_confidence(self, relevant_tables: List[TableRelevance], entities: List[Dict]) -> float:
        """
        Calculates confidence score for schema analysis based on table relevance and entity coverage.
        """
        if not relevant_tables:
            return 0.0
        
        # Base confidence from top table relevance
        top_table_score = relevant_tables[0].relevance_score if relevant_tables else 0
        
        # Entity coverage score
        covered_entities = 0
        for entity in entities:
            if any(entity["value"].lower() in table.table_name.lower() or
                   any(entity["value"].lower() in col.get("name", "").lower() 
                       for col in table.table_info.get("columns", []))
                   for table in relevant_tables):
                covered_entities += 1
        
        entity_coverage = covered_entities / len(entities) if entities else 0
        
        # Combine scores
        confidence = (top_table_score * 0.6) + (entity_coverage * 0.4)
        return min(confidence, 1.0)
    
    async def _get_schema_insights(self, entities: List[Dict], intent: Dict) -> Dict:
        """Retrieves relevant schema insights from long-term memory."""
        return await self.memory_system.knowledge_memory.get_schema_insights(
            entities=entities,
            intent=intent
        )
    
    async def _get_memory_relevance_score(self, table_name: str, entities: List[Dict], intent: Dict) -> float:
        """Gets relevance score based on successful past usage patterns."""
        # This would query the memory system for past successful queries
        # that used this table with similar entities and intent
        return 0.0  # Placeholder - implement based on memory system
    
    def _get_performance_notes(self, table_name: str, table_info: Dict, intent: Dict) -> List[str]:
        """Generates performance considerations for table usage."""
        notes = []
        
        # Large table warnings
        if table_info.get("row_count", 0) > 1000000:
            notes.append("Large table - consider using filters and limits")
        
        # Temporal query optimizations
        if intent.get("temporal_scope") and any("date" in col.get("name", "").lower() 
                                               for col in table_info.get("columns", [])):
            notes.append("Consider date range filters for better performance")
        
        return notes
    
    async def _analyze_columns_deeply(self, relevant_tables: List[TableRelevance], 
                                     sample_data: Dict[str, List[Dict]], entities: List[Dict], 
                                     intent: Dict) -> Dict[str, List[ColumnAnalysis]]:
        """Performs deep analysis of columns including data patterns and filtering suggestions."""
        
        column_analysis = {}
        
        for table in relevant_tables:
            table_name = table.table_name
            table_samples = sample_data.get(table_name, [])
            column_analysis[table_name] = []
            
            if not table_samples:
                continue
                
            # Analyze each column in the table
            for column_info in table.table_info.get("columns", []):
                column_name = column_info.get("name", "")
                
                # Extract values for this column from samples
                column_values = [row.get(column_name) for row in table_samples if row.get(column_name) is not None]
                
                if not column_values:
                    continue
                
                # Analyze column patterns
                analysis = await self._analyze_single_column(column_name, column_values, column_info, entities, intent)
                column_analysis[table_name].append(analysis)
        
        return column_analysis
    
    async def _analyze_single_column(self, column_name: str, values: List, 
                                   column_info: Dict, entities: List[Dict], 
                                   intent: Dict) -> ColumnAnalysis:
        """Analyzes a single column's data patterns and characteristics."""
        
        # Get unique values (limited to prevent memory issues)
        unique_values = list(set(str(v) for v in values[:100]))[:20]
        
        # Detect value patterns
        value_patterns = self._detect_value_patterns(values, column_info)
        
        # Generate suggested filters based on entities and intent
        suggested_filters = self._generate_column_filters(column_name, unique_values, values, entities, intent)
        
        # Calculate data quality score
        data_quality_score = self._calculate_data_quality_score(values, column_info)
        
        return ColumnAnalysis(
            column_name=column_name,
            data_type=column_info.get("type", "unknown"),
            unique_values=unique_values,
            value_patterns=value_patterns,
            suggested_filters=suggested_filters,
            data_quality_score=data_quality_score
        )
    
    def _detect_value_patterns(self, values: List, column_info: Dict) -> Dict[str, any]:
        """Detects patterns in column values."""
        
        patterns = {
            "pattern_type": "unknown",
            "statistics": {},
            "categories": [],
            "date_patterns": [],
            "numeric_patterns": {}
        }
        
        # Determine data type and patterns
        if all(isinstance(v, (int, float)) for v in values if v is not None):
            patterns["pattern_type"] = "numeric"
            numeric_values = [v for v in values if v is not None]
            if numeric_values:
                patterns["numeric_patterns"] = {
                    "min": min(numeric_values),
                    "max": max(numeric_values),
                    "range": max(numeric_values) - min(numeric_values),
                    "has_negatives": any(v < 0 for v in numeric_values),
                    "has_decimals": any(isinstance(v, float) and v % 1 != 0 for v in numeric_values)
                }
        
        elif all(isinstance(v, str) for v in values if v is not None):
            patterns["pattern_type"] = "categorical"
            string_values = [v for v in values if v is not None]
            
            # Check for date patterns
            date_keywords = ["date", "time", "created", "updated", "modified"]
            if any(keyword in column_info.get("name", "").lower() for keyword in date_keywords):
                patterns["date_patterns"] = ["potential_date_column"]
            
            # Categorize string patterns
            if len(set(string_values)) < len(string_values) * 0.5:  # Many duplicates
                patterns["categories"] = list(set(string_values))[:10]
            
            # Check for common patterns
            if all(len(v) == len(string_values[0]) for v in string_values):
                patterns["categories"].append("fixed_length_strings")
            
            if all(v.isupper() for v in string_values):
                patterns["categories"].append("uppercase_strings")
            elif all(v.islower() for v in string_values):
                patterns["categories"].append("lowercase_strings")
        
        return patterns
    
    def _generate_column_filters(self, column_name: str, unique_values: List[str], 
                               all_values: List, entities: List[Dict], intent: Dict) -> List[Dict]:
        """Generates intelligent filter suggestions for a column."""
        
        filters = []
        
        # Entity-based filters
        for entity in entities:
            entity_value = str(entity.get("value", "")).lower()
            entity_type = entity.get("type", "")
            
            # Check if entity matches any unique values
            matching_values = [v for v in unique_values if entity_value in v.lower()]
            if matching_values:
                filters.append({
                    "type": "exact_match",
                    "column": column_name,
                    "values": matching_values,
                    "rationale": f"Entity '{entity['value']}' matches column values",
                    "sql_condition": f"{column_name} IN ({', '.join(repr(str(v)) for v in matching_values)})"
                })
        
        # Intent-based filters
        temporal_scope = intent.get("temporal_scope")
        if temporal_scope and any(keyword in column_name.lower() for keyword in ["date", "time", "created"]):
            filters.append({
                "type": "date_range",
                "column": column_name,
                "values": [temporal_scope],
                "rationale": f"Temporal scope '{temporal_scope}' applicable to date column",
                "sql_condition": f"{column_name} >= DATEADD(day, -30, CURRENT_DATE)"  # Example
            })
        
        # Data-driven filters
        if len(unique_values) <= 10 and len(unique_values) > 1:
            # Good candidate for categorical filtering
            filters.append({
                "type": "categorical_options",
                "column": column_name,
                "values": unique_values,
                "rationale": "Limited categories suitable for filtering",
                "sql_condition": f"{column_name} = '<VALUE>'"
            })
        
        # Numeric range filters
        if all(str(v).replace('.', '').replace('-', '').isdigit() for v in unique_values[:5]):
            try:
                numeric_values = [float(v) for v in unique_values if str(v).replace('.', '').replace('-', '').isdigit()]
                if numeric_values:
                    min_val, max_val = min(numeric_values), max(numeric_values)
                    filters.append({
                        "type": "numeric_range",
                        "column": column_name,
                        "values": [min_val, max_val],
                        "rationale": f"Numeric column with range {min_val} to {max_val}",
                        "sql_condition": f"{column_name} BETWEEN {min_val} AND {max_val}"
                    })
            except ValueError:
                pass
        
        return filters
    
    def _calculate_data_quality_score(self, values: List, column_info: Dict) -> float:
        """Calculates data quality score for a column."""
        
        if not values:
            return 0.0
        
        score = 1.0
        
        # Penalize for null values
        null_count = sum(1 for v in values if v is None or v == "")
        null_ratio = null_count / len(values)
        score -= null_ratio * 0.3
        
        # Reward for consistent data types
        type_consistency = len(set(type(v).__name__ for v in values if v is not None)) == 1
        if type_consistency:
            score += 0.1
        
        # Reward for reasonable unique value ratio
        unique_ratio = len(set(str(v) for v in values)) / len(values)
        if 0.1 <= unique_ratio <= 0.9:  # Not all unique, not all same
            score += 0.1
        
        return min(max(score, 0.0), 1.0)
    
    def _detect_cross_table_patterns(self, sample_data: Dict[str, List[Dict]], 
                                   relevant_tables: List[TableRelevance]) -> Dict[str, any]:
        """Detects patterns across multiple tables."""
        
        patterns = {
            "common_columns": [],
            "potential_joins": [],
            "data_distribution": {},
            "cross_table_insights": []
        }
        
        if len(relevant_tables) < 2:
            return patterns
        
        # Find common column names across tables
        all_columns = {}
        for table in relevant_tables:
            table_name = table.table_name
            columns = [col.get("name", "") for col in table.table_info.get("columns", [])]
            all_columns[table_name] = columns
        
        # Identify common columns
        common_cols = set(all_columns[list(all_columns.keys())[0]])
        for table_cols in all_columns.values():
            common_cols = common_cols.intersection(set(table_cols))
        
        patterns["common_columns"] = list(common_cols)
        
        # Suggest potential joins based on common columns
        for col in common_cols:
            if any(keyword in col.lower() for keyword in ["id", "key", "code"]):
                patterns["potential_joins"].append({
                    "column": col,
                    "tables": list(all_columns.keys()),
                    "join_type": "inner_join",
                    "rationale": f"Common identifier column '{col}' found across tables"
                })
        
        # Analyze data distribution across tables
        for table_name, samples in sample_data.items():
            if samples:
                patterns["data_distribution"][table_name] = {
                    "sample_count": len(samples),
                    "column_count": len(samples[0].keys()) if samples else 0,
                    "has_data": len(samples) > 0
                }
        
        return patterns
    
    def _generate_filtering_suggestions(self, column_analysis: Dict[str, List[ColumnAnalysis]], 
                                      entities: List[Dict], intent: Dict) -> Dict[str, List[Dict]]:
        """Generates intelligent filtering suggestions based on column analysis."""
        
        suggestions = {}
        
        for table_name, analyses in column_analysis.items():
            suggestions[table_name] = []
            
            for analysis in analyses:
                # Add high-quality filters from column analysis
                quality_threshold = 0.7
                if analysis.data_quality_score >= quality_threshold:
                    for filter_suggestion in analysis.suggested_filters:
                        if filter_suggestion["type"] in ["exact_match", "categorical_options"]:
                            suggestions[table_name].append({
                                "priority": "high",
                                "filter": filter_suggestion,
                                "column_quality": analysis.data_quality_score
                            })
                
                # Add pattern-based suggestions
                if analysis.value_patterns.get("pattern_type") == "categorical":
                    categories = analysis.value_patterns.get("categories", [])
                    if len(categories) <= 5:
                        suggestions[table_name].append({
                            "priority": "medium",
                            "filter": {
                                "type": "category_selection",
                                "column": analysis.column_name,
                                "options": categories,
                                "rationale": "Small number of categories ideal for filtering"
                            },
                            "column_quality": analysis.data_quality_score
                        })
        
        return suggestions

    async def _get_column_metadata(self, relevant_tables: List[TableRelevance]) -> Dict[str, Dict]:
        """Enhanced column metadata extraction."""
        metadata = {}
        
        for table in relevant_tables:
            table_name = table.table_name
            metadata[table_name] = {}
            
            for column in table.table_info.get("columns", []):
                column_name = column.get("name", "")
                metadata[table_name][column_name] = {
                    "type": column.get("type", "unknown"),
                    "nullable": column.get("nullable", True),
                    "is_primary_key": column.get("is_primary_key", False),
                    "is_foreign_key": column.get("is_foreign_key", False),
                    "max_length": column.get("max_length"),
                    "description": column.get("description", "")
                }
        
        return metadata
    
    async def _update_schema_memory(self, entities: List[Dict], relevant_tables: List[TableRelevance], 
                                  intent: Dict, relationships: Dict):
        """Updates memory with new schema insights from this analysis."""
        await self.memory_system.working_memory.update_context(
            agent_name=self.agent_name,
            update_data={
                "schema_analysis_timestamp": datetime.now().isoformat(),
                "entities_processed": entities,
                "tables_identified": [table.table_name for table in relevant_tables],
                "relationships_found": relationships,
                "intent_processed": intent
            }
        )
