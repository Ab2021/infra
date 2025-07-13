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
            
            return {
                "generated_sql": optimized_sql,
                "generation_strategy": generation_strategy,
                "confidence": confidence,
                "alternatives": await self._generate_alternatives(optimized_sql),
                "optimization_applied": True
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
