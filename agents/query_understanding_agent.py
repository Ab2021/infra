"""
Query Understanding Agent - Simplified
Combines NLU + Schema Intelligence into one focused agent
"""

import json
import logging
from typing import Dict, List, Optional
from datetime import datetime
from dataclasses import dataclass

@dataclass
class QueryIntent:
    """Simplified query intent structure"""
    action: str  # trend_analysis, comparison, aggregation, filtering
    metrics: List[str]  # columns to measure (sales_amount, count, etc.)
    dimensions: List[str]  # columns to group by (region, date, category)
    filters: Dict[str, any]  # filter conditions
    time_scope: Optional[str] = None  # temporal filters
    output_preference: str = "chart"  # chart, table, dashboard

class QueryUnderstandingAgent:
    """
    Agent 1: Understands user queries and identifies relevant tables/columns
    Combines natural language understanding with schema intelligence
    """
    
    def __init__(self, llm_provider, memory_system, schema_info: Dict):
        self.llm_provider = llm_provider
        self.memory_system = memory_system
        self.schema_info = schema_info
        self.logger = logging.getLogger(__name__)
        
        # Common query patterns
        self.action_keywords = {
            "trend_analysis": ["trend", "over time", "growth", "change", "evolution"],
            "comparison": ["compare", "vs", "versus", "difference", "against"],
            "aggregation": ["total", "sum", "count", "average", "max", "min"],
            "filtering": ["where", "filter", "show only", "exclude", "include"],
            "ranking": ["top", "bottom", "highest", "lowest", "best", "worst"]
        }
        
        # Common metric keywords
        self.metric_keywords = {
            "sales": ["sales", "revenue", "income", "earnings"],
            "count": ["count", "number", "quantity", "amount"],
            "profit": ["profit", "margin", "earnings"],
            "cost": ["cost", "expense", "spend", "budget"]
        }
        
        # Common dimension keywords
        self.dimension_keywords = {
            "time": ["date", "time", "month", "quarter", "year", "day"],
            "location": ["region", "country", "state", "city", "location"],
            "category": ["category", "type", "class", "segment", "group"],
            "product": ["product", "item", "sku", "brand"]
        }
    
    async def process(self, user_query: str, session_context: Dict = None) -> Dict:
        """
        Main processing method for query understanding
        
        Args:
            user_query: Natural language query from user
            session_context: Previous conversation context
            
        Returns:
            Dict with identified tables, columns, and intent
        """
        try:
            self.logger.info(f"Processing query: {user_query}")
            
            # Step 1: Extract basic intent using keywords
            basic_intent = self._extract_basic_intent(user_query)
            
            # Step 2: Get enhanced understanding using LLM
            enhanced_intent = await self._get_llm_understanding(user_query, basic_intent, session_context)
            
            # Step 3: Map to actual tables and columns
            table_mapping = self._map_to_schema(enhanced_intent)
            
            # Step 4: Validate and refine mapping
            validated_mapping = self._validate_mapping(table_mapping)
            
            # Store successful pattern in memory
            await self._store_successful_pattern(user_query, validated_mapping)
            
            return {
                "user_query": user_query,
                "query_intent": validated_mapping["intent"],
                "identified_tables": validated_mapping["tables"],
                "required_columns": validated_mapping["columns"],
                "confidence": validated_mapping["confidence"],
                "processing_time": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error processing query: {e}")
            return {
                "error": str(e),
                "user_query": user_query,
                "confidence": 0.0
            }
    
    def _extract_basic_intent(self, query: str) -> Dict:
        """Extract basic intent using keyword matching"""
        query_lower = query.lower()
        
        # Identify primary action
        primary_action = "aggregation"  # default
        for action, keywords in self.action_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                primary_action = action
                break
        
        # Identify potential metrics
        potential_metrics = []
        for metric, keywords in self.metric_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                potential_metrics.append(metric)
        
        # Identify potential dimensions
        potential_dimensions = []
        for dimension, keywords in self.dimension_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                potential_dimensions.append(dimension)
        
        # Extract time scope
        time_scope = None
        time_indicators = ["2023", "2024", "Q1", "Q2", "Q3", "Q4", "last year", "this month"]
        for indicator in time_indicators:
            if indicator.lower() in query_lower:
                time_scope = indicator
                break
        
        return {
            "primary_action": primary_action,
            "potential_metrics": potential_metrics,
            "potential_dimensions": potential_dimensions,
            "time_scope": time_scope,
            "confidence": 0.6  # Basic keyword matching confidence
        }
    
    async def _get_llm_understanding(self, query: str, basic_intent: Dict, session_context: Dict = None) -> Dict:
        """Use LLM to enhance understanding with context"""
        
        # Build context-aware prompt
        prompt = f"""
        Analyze this SQL query request and extract structured information.
        
        User Query: "{query}"
        
        Available Schema Information:
        {self._format_schema_for_prompt()}
        
        Basic Intent Detected: {json.dumps(basic_intent, indent=2)}
        """
        
        if session_context and session_context.get("conversation_history"):
            prompt += f"""
            
            Previous Conversation Context:
            {self._format_conversation_context(session_context)}
            """
        
        prompt += """
        
        Provide a JSON response with this structure:
        {
            "intent": {
                "action": "trend_analysis|comparison|aggregation|filtering|ranking",
                "metrics": ["list of metrics to calculate"],
                "dimensions": ["list of dimensions to group by"],
                "filters": {"column": "value"},
                "time_scope": "temporal filter if any",
                "output_preference": "chart|table|dashboard"
            },
            "table_requirements": [
                {
                    "table_name": "identified_table",
                    "reason": "why this table is needed",
                    "confidence": 0.0-1.0
                }
            ],
            "column_requirements": [
                {
                    "column_name": "identified_column",
                    "table": "parent_table",
                    "usage": "metric|dimension|filter",
                    "confidence": 0.0-1.0
                }
            ],
            "confidence": 0.0-1.0
        }
        """
        
        try:
            from langchain_core.messages import HumanMessage
            response = await self.llm_provider.ainvoke([HumanMessage(content=prompt)])
            
            # Parse LLM response
            llm_result = self._parse_llm_response(response.content)
            
            # Merge with basic intent
            enhanced_intent = self._merge_intent(basic_intent, llm_result)
            
            return enhanced_intent
            
        except Exception as e:
            self.logger.warning(f"LLM processing failed, using basic intent: {e}")
            return basic_intent
    
    def _map_to_schema(self, intent: Dict) -> Dict:
        """Map intent to actual schema tables and columns"""
        
        table_mapping = {
            "tables": [],
            "columns": [],
            "intent": None,
            "confidence": 0.0
        }
        
        try:
            # Extract requirements from intent
            if "table_requirements" in intent:
                for table_req in intent["table_requirements"]:
                    table_name = table_req["table_name"]
                    if table_name in self.schema_info.get("tables", {}):
                        table_mapping["tables"].append({
                            "name": table_name,
                            "confidence": table_req.get("confidence", 0.5),
                            "reason": table_req.get("reason", "")
                        })
            
            if "column_requirements" in intent:
                for col_req in intent["column_requirements"]:
                    column_name = col_req["column_name"]
                    table_name = col_req.get("table", "")
                    
                    # Validate column exists in table
                    if self._validate_column_in_table(column_name, table_name):
                        table_mapping["columns"].append({
                            "name": column_name,
                            "table": table_name,
                            "usage": col_req.get("usage", "unknown"),
                            "confidence": col_req.get("confidence", 0.5)
                        })
            
            # Create structured intent
            if "intent" in intent:
                table_mapping["intent"] = QueryIntent(
                    action=intent["intent"].get("action", "aggregation"),
                    metrics=intent["intent"].get("metrics", []),
                    dimensions=intent["intent"].get("dimensions", []),
                    filters=intent["intent"].get("filters", {}),
                    time_scope=intent["intent"].get("time_scope"),
                    output_preference=intent["intent"].get("output_preference", "chart")
                )
            
            # Calculate overall confidence
            table_mapping["confidence"] = self._calculate_mapping_confidence(table_mapping)
            
        except Exception as e:
            self.logger.error(f"Error mapping to schema: {e}")
            table_mapping["confidence"] = 0.0
        
        return table_mapping
    
    def _validate_mapping(self, mapping: Dict) -> Dict:
        """Validate and refine the table/column mapping"""
        
        validated = mapping.copy()
        
        # Ensure we have at least one table
        if not validated["tables"]:
            validated["tables"] = self._suggest_default_tables(validated.get("intent"))
        
        # Ensure we have required columns for the intent
        if validated.get("intent"):
            missing_columns = self._identify_missing_columns(validated)
            validated["columns"].extend(missing_columns)
        
        # Remove duplicate tables/columns
        validated["tables"] = self._deduplicate_tables(validated["tables"])
        validated["columns"] = self._deduplicate_columns(validated["columns"])
        
        # Recalculate confidence after validation
        validated["confidence"] = self._calculate_mapping_confidence(validated)
        
        return validated
    
    def _format_schema_for_prompt(self) -> str:
        """Format schema information for LLM prompt"""
        
        schema_summary = []
        
        for table_name, table_info in self.schema_info.get("tables", {}).items():
            columns = table_info.get("columns", [])
            description = table_info.get("description", "")
            
            column_list = [f"  - {col['name']} ({col.get('type', 'unknown')})" 
                          for col in columns[:10]]  # Limit to first 10 columns
            
            schema_summary.append(f"""
Table: {table_name}
Description: {description}
Key Columns:
{chr(10).join(column_list)}
""")
        
        return "\n".join(schema_summary[:5])  # Limit to 5 tables for prompt size
    
    def _format_conversation_context(self, session_context: Dict) -> str:
        """Format conversation history for context"""
        
        history = session_context.get("conversation_history", [])
        
        if not history:
            return "No previous conversation"
        
        formatted = []
        for item in history[-3:]:  # Last 3 interactions
            query = item.get("query", "")
            response = item.get("response", {})
            tables = response.get("identified_tables", [])
            
            formatted.append(f"Previous Query: {query}")
            if tables:
                formatted.append(f"Tables Used: {', '.join([t.get('name', t) for t in tables])}")
        
        return "\n".join(formatted)
    
    def _parse_llm_response(self, response_text: str) -> Dict:
        """Parse LLM JSON response safely"""
        
        try:
            # Clean response text
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0]
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0]
            
            return json.loads(response_text.strip())
            
        except json.JSONDecodeError as e:
            self.logger.warning(f"Failed to parse LLM response: {e}")
            return {"confidence": 0.3}  # Low confidence fallback
    
    def _merge_intent(self, basic_intent: Dict, llm_intent: Dict) -> Dict:
        """Merge basic keyword intent with LLM enhanced intent"""
        
        merged = llm_intent.copy()
        
        # Use basic intent as fallback
        if "intent" not in merged or not merged["intent"]:
            merged["intent"] = {
                "action": basic_intent.get("primary_action", "aggregation"),
                "metrics": basic_intent.get("potential_metrics", []),
                "dimensions": basic_intent.get("potential_dimensions", []),
                "filters": {},
                "time_scope": basic_intent.get("time_scope"),
                "output_preference": "chart"
            }
        
        # Boost confidence if both methods agree
        if (basic_intent.get("primary_action") == 
            merged.get("intent", {}).get("action")):
            merged["confidence"] = min(1.0, merged.get("confidence", 0.5) + 0.2)
        
        return merged
    
    def _validate_column_in_table(self, column_name: str, table_name: str) -> bool:
        """Check if column exists in the specified table"""
        
        if not table_name or table_name not in self.schema_info.get("tables", {}):
            return False
        
        table_columns = self.schema_info["tables"][table_name].get("columns", [])
        column_names = [col.get("name", "") for col in table_columns]
        
        return column_name in column_names
    
    def _calculate_mapping_confidence(self, mapping: Dict) -> float:
        """Calculate overall confidence score for the mapping"""
        
        if not mapping.get("tables") or not mapping.get("columns"):
            return 0.1
        
        # Base confidence on table and column confidence scores
        table_confidences = [t.get("confidence", 0.5) for t in mapping["tables"]]
        column_confidences = [c.get("confidence", 0.5) for c in mapping["columns"]]
        
        avg_table_conf = sum(table_confidences) / len(table_confidences)
        avg_column_conf = sum(column_confidences) / len(column_confidences)
        
        # Weight table confidence more heavily
        overall_confidence = (avg_table_conf * 0.6) + (avg_column_conf * 0.4)
        
        return min(1.0, overall_confidence)
    
    def _suggest_default_tables(self, intent: QueryIntent = None) -> List[Dict]:
        """Suggest default tables when none are identified"""
        
        # Simple heuristic: suggest most commonly used tables
        common_tables = []
        
        for table_name, table_info in self.schema_info.get("tables", {}).items():
            description = table_info.get("description", "").lower()
            
            # Look for fact tables (usually contain metrics)
            if any(keyword in description for keyword in ["fact", "sales", "transaction", "event"]):
                common_tables.append({
                    "name": table_name,
                    "confidence": 0.4,
                    "reason": "Common fact table"
                })
        
        return common_tables[:2]  # Return top 2 suggestions
    
    def _identify_missing_columns(self, mapping: Dict) -> List[Dict]:
        """Identify columns that might be missing for the intent"""
        
        missing = []
        intent = mapping.get("intent")
        
        if not intent:
            return missing
        
        # Check if we have columns for metrics
        existing_columns = [c["name"] for c in mapping.get("columns", [])]
        
        for metric in intent.metrics:
            if metric not in existing_columns:
                # Try to find matching column in schema
                matching_column = self._find_matching_column(metric, mapping["tables"])
                if matching_column:
                    missing.append(matching_column)
        
        return missing
    
    def _find_matching_column(self, metric: str, tables: List[Dict]) -> Optional[Dict]:
        """Find a column that matches the metric requirement"""
        
        for table in tables:
            table_name = table["name"]
            table_info = self.schema_info.get("tables", {}).get(table_name, {})
            
            for column in table_info.get("columns", []):
                column_name = column.get("name", "").lower()
                
                # Simple matching logic
                if metric.lower() in column_name or column_name in metric.lower():
                    return {
                        "name": column["name"],
                        "table": table_name,
                        "usage": "metric",
                        "confidence": 0.6
                    }
        
        return None
    
    def _deduplicate_tables(self, tables: List[Dict]) -> List[Dict]:
        """Remove duplicate tables"""
        seen = set()
        deduplicated = []
        
        for table in tables:
            table_name = table["name"]
            if table_name not in seen:
                seen.add(table_name)
                deduplicated.append(table)
        
        return deduplicated
    
    def _deduplicate_columns(self, columns: List[Dict]) -> List[Dict]:
        """Remove duplicate columns"""
        seen = set()
        deduplicated = []
        
        for column in columns:
            column_key = f"{column['table']}.{column['name']}"
            if column_key not in seen:
                seen.add(column_key)
                deduplicated.append(column)
        
        return deduplicated
    
    async def _store_successful_pattern(self, query: str, mapping: Dict):
        """Store successful query pattern in memory for learning"""
        
        if mapping.get("confidence", 0) > 0.7:  # Only store high-confidence patterns
            try:
                await self.memory_system.store_query_pattern(
                    pattern_type="query_understanding",
                    pattern_description=f"Query: {query[:100]}...",
                    example_queries=[query],
                    metadata={
                        "tables": [t["name"] for t in mapping.get("tables", [])],
                        "columns": [c["name"] for c in mapping.get("columns", [])],
                        "intent": mapping.get("intent").__dict__ if mapping.get("intent") else {}
                    }
                )
            except Exception as e:
                self.logger.warning(f"Failed to store pattern: {e}")