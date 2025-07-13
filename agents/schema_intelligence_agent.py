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
class SchemaAnalysisResult:
    """Complete schema analysis results."""
    relevant_tables: List[TableRelevance]
    table_relationships: Dict[str, List[str]]
    column_metadata: Dict[str, Dict]
    sample_data: Dict[str, List[Dict]]
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
        Analyzes database schema to identify relevant tables and relationships.
        
        Args:
            entities: Extracted entities from NLU processing
            intent: Structured query intent
            context: Session and memory context
            
        Returns:
            Complete schema analysis with recommendations
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
