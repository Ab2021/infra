"""
Query Understanding Agent
Transforms natural language queries into structured intent with schema mapping.
Combines NLU capabilities with intelligent schema analysis.
"""

import json
import logging
import re
from typing import Dict, List, Optional, Any
from datetime import datetime
from dataclasses import dataclass

@dataclass
class QueryIntent:
    """Structured query intent representation"""
    action: str  # trend_analysis, comparison, aggregation, filtering, ranking
    metrics: List[str]  # columns to measure (sales_amount, count, etc.)
    dimensions: List[str]  # columns to group by (region, date, category)
    filters: Dict[str, Any]  # filter conditions
    time_scope: Optional[str] = None  # temporal filters
    output_preference: str = "chart"  # chart, table, dashboard

class QueryUnderstandingAgent:
    """
    Agent 1: Query Understanding and Schema Intelligence
    
    Responsibilities:
    - Parse natural language queries into structured intent
    - Identify relevant tables and columns from schema
    - Map user requirements to database entities
    - Provide confidence scores for mappings
    - Learn from successful patterns
    """
    
    def __init__(self, llm_provider=None, memory_system=None, schema_info: Dict = None,
                 all_tables: List[Dict] = None, all_columns: List[Dict] = None):
        """
        Initialize Query Understanding Agent
        
        Args:
            llm_provider: LLM service for enhanced understanding
            memory_system: Memory system for pattern storage
            schema_info: Database schema information
            all_tables: List of available tables
            all_columns: List of available columns
        """
        self.llm_provider = llm_provider
        self.memory_system = memory_system
        self.schema_info = schema_info or {}
        self.all_tables = all_tables or []
        self.all_columns = all_columns or []
        self.logger = logging.getLogger(__name__)
        
        # Common query patterns for keyword matching
        self.action_keywords = {
            "trend_analysis": ["trend", "over time", "growth", "change", "evolution", "timeline"],
            "comparison": ["compare", "vs", "versus", "difference", "against", "between"],
            "aggregation": ["total", "sum", "count", "average", "max", "min", "aggregate"],
            "filtering": ["where", "filter", "show only", "exclude", "include", "with"],
            "ranking": ["top", "bottom", "highest", "lowest", "best", "worst", "rank"]
        }
        
        # Common metric keywords
        self.metric_keywords = {
            "sales": ["sales", "revenue", "income", "earnings", "amount"],
            "count": ["count", "number", "quantity", "total"],
            "profit": ["profit", "margin", "earnings", "net"],
            "cost": ["cost", "expense", "spend", "budget"],
            "premium": ["premium", "payment", "fee"],
            "claims": ["claims", "loss", "payout"]
        }
        
        # Common dimension keywords
        self.dimension_keywords = {
            "time": ["date", "time", "month", "quarter", "year", "day", "period"],
            "location": ["region", "state", "country", "city", "location", "geography"],
            "category": ["category", "type", "class", "segment", "group", "kind"],
            "product": ["product", "policy", "coverage", "plan", "line"],
            "customer": ["customer", "client", "member", "policyholder"]
        }
        
        # Column name normalization utilities
        self.column_mapping = {}
        self.reverse_mapping = {}
        self._create_column_mappings()
    
    def _create_column_mappings(self):
        """Create mappings for column name variations"""
        for col in self.all_columns:
            if col.get("column_name"):
                original = str(col["column_name"]).strip()
                normalized = original.replace(' ', '_').replace('-', '_')
                
                self.column_mapping[normalized.lower()] = original
                self.reverse_mapping[original.lower()] = normalized
                self.column_mapping[original.lower()] = original
                self.reverse_mapping[normalized.lower()] = normalized
    
    def resolve_column_name(self, column_name: str, table_name: str = None) -> str:
        """Resolve column name variations to correct database names"""
        if not column_name or str(column_name).strip() == '':
            return None
        
        cleaned = str(column_name).strip()
        
        # Try exact match first
        if table_name:
            table_short = table_name.split('.')[-1] if '.' in table_name else table_name
            for col in self.all_columns:
                if (col["table_name"].lower() == table_short.lower() and 
                    col["column_name"].lower() == cleaned.lower()):
                    return col["column_name"]
        
        # Try mapping lookup
        lower_cleaned = cleaned.lower()
        if lower_cleaned in self.column_mapping:
            return self.column_mapping[lower_cleaned]
        
        # Try variations
        variations = [
            cleaned.replace('_', ' '),
            cleaned.replace(' ', '_'),
            cleaned.replace('-', '_'),
            cleaned.replace('_', '-')
        ]
        
        for variation in variations:
            if variation.lower() in self.column_mapping:
                return self.column_mapping[variation.lower()]
        
        return cleaned
    
    def quote_column_name(self, column_name: str) -> str:
        """Quote column names that contain spaces or special characters"""
        if not column_name:
            return column_name
        
        if (' ' in column_name or 
            '-' in column_name or 
            any(char in column_name for char in ['(', ')', '.', ',', ';']) or
            column_name.upper() in ['ORDER', 'GROUP', 'SELECT', 'FROM', 'WHERE', 'JOIN']):
            return f'"{column_name}"'
        
        return column_name
    
    async def process(self, user_query: str, session_context: Dict = None) -> Dict:
        """
        Main processing method for query understanding
        
        Args:
            user_query: Natural language query from user
            session_context: Previous conversation context
            
        Returns:
            Dict with identified tables, columns, and structured intent
        """
        try:
            self.logger.info(f"Processing query: {user_query}")
            
            # Step 1: Extract basic intent using keywords
            basic_intent = self._extract_basic_intent(user_query)
            
            # Step 2: Get enhanced understanding using LLM if available
            if self.llm_provider:
                enhanced_intent = await self._get_llm_understanding(user_query, basic_intent, session_context)
            else:
                enhanced_intent = basic_intent
            
            # Step 3: Map to actual tables and columns
            table_mapping = self._map_to_schema(enhanced_intent, user_query)
            
            # Step 4: Validate and refine mapping
            validated_mapping = self._validate_mapping(table_mapping)
            
            # Step 5: Store successful pattern in memory
            if self.memory_system and validated_mapping.get("confidence", 0) > 0.7:
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
        time_indicators = ["2023", "2024", "2025", "Q1", "Q2", "Q3", "Q4", "last year", "this month", "this year"]
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
        
        try:
            # Build context-aware prompt
            tables_context = "\n".join([
                f"{tbl['database']}.{tbl['schema']}.{tbl['table_name']}: {tbl['brief_description']}"
                for tbl in self.all_tables
            ])
            
            columns_context = "\n".join([
                f"{col['table_name']}.{col['column_name']}: {col['description']}"
                for col in self.all_columns[:50]  # Limit for prompt size
            ])
            
            prompt = f"""
            You are an expert in analyzing natural language queries for SQL generation in the insurance domain.
            
            User Query: "{query}"
            
            Available Tables:
            {tables_context}
            
            Available Columns (first 50):
            {columns_context}
            
            Basic Intent Detected: {json.dumps(basic_intent, indent=2)}
            
            IMPORTANT COLUMN NAMING RULES:
            - Use EXACT column names as shown in the columns context
            - Column names may contain SPACES (e.g., "Feature Name", "Policy Number")
            - Do NOT convert spaces to underscores
            - If a column name has spaces, use it exactly as shown
            """
            
            if session_context and session_context.get("conversation_history"):
                prompt += f"""
                
                Previous Conversation Context:
                {self._format_conversation_context(session_context)}
                """
            
            prompt += """
            
            Analyze the query and provide a JSON response with this structure:
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
                        "column_name": "exact_column_name_from_context",
                        "table": "parent_table",
                        "usage": "metric|dimension|filter",
                        "confidence": 0.0-1.0
                    }
                ],
                "confidence": 0.0-1.0
            }
            
            Use EXACT column names as shown in the context. Output only JSON.
            """
            
            # Call LLM through the provider
            if hasattr(self.llm_provider, 'get_gpt_response_non_streaming'):
                payload = {
                    "username": "QUERY_UNDERSTANDING_AGENT",
                    "session_id": "1",
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.1,
                    "max_tokens": 1024
                }
                
                response = self.llm_provider.get_gpt_response_non_streaming(payload)
                content = response.json()['choices'][0]['message']['content']
                
                # Parse LLM response
                llm_result = self._parse_llm_response(content)
                
                # Merge with basic intent
                enhanced_intent = self._merge_intent(basic_intent, llm_result)
                
                return enhanced_intent
            else:
                return basic_intent
                
        except Exception as e:
            self.logger.warning(f"LLM processing failed, using basic intent: {e}")
            return basic_intent
    
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
        
        merged = llm_intent.copy() if llm_intent else {}
        
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
    
    def _map_to_schema(self, intent: Dict, user_query: str) -> Dict:
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
                    # Find matching table
                    for table in self.all_tables:
                        full_name = f"{table['database']}.{table['schema']}.{table['table_name']}"
                        if (table_name.lower() in full_name.lower() or
                            table['table_name'].lower() == table_name.lower()):
                            table_mapping["tables"].append({
                                "name": full_name,
                                "confidence": table_req.get("confidence", 0.5),
                                "reason": table_req.get("reason", "")
                            })
                            break
            
            if "column_requirements" in intent:
                for col_req in intent["column_requirements"]:
                    column_name = col_req["column_name"]
                    table_name = col_req.get("table", "")
                    
                    # Resolve column name
                    resolved_name = self.resolve_column_name(column_name, table_name)
                    
                    if resolved_name:
                        table_mapping["columns"].append({
                            "name": resolved_name,
                            "table": table_name,
                            "usage": col_req.get("usage", "unknown"),
                            "confidence": col_req.get("confidence", 0.5)
                        })
            
            # Fallback: use keyword matching
            if not table_mapping["tables"] or not table_mapping["columns"]:
                keyword_mapping = self._keyword_based_mapping(user_query, intent)
                if not table_mapping["tables"]:
                    table_mapping["tables"] = keyword_mapping["tables"]
                if not table_mapping["columns"]:
                    table_mapping["columns"] = keyword_mapping["columns"]
            
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
    
    def _keyword_based_mapping(self, user_query: str, intent: Dict) -> Dict:
        """Fallback keyword-based mapping when LLM fails"""
        
        query_lower = user_query.lower()
        keywords = set(query_lower.replace(",", " ").replace("_", " ").split())
        
        matched_tables = []
        matched_columns = []
        
        # Find tables
        for table in self.all_tables:
            table_text = (table['table_name'] + " " + 
                         (table['brief_description'] or "")).lower()
            if any(k in table_text for k in keywords):
                full_name = f"{table['database']}.{table['schema']}.{table['table_name']}"
                matched_tables.append({
                    "name": full_name,
                    "confidence": 0.6,
                    "reason": "Keyword match"
                })
        
        # Find columns
        for col in self.all_columns:
            col_text = (col['column_name'] + " " + 
                       (col['description'] or "")).lower()
            if any(k in col_text for k in keywords):
                matched_columns.append({
                    "name": col['column_name'],
                    "table": col['table_name'],
                    "usage": "unknown",
                    "confidence": 0.6
                })
        
        return {"tables": matched_tables[:5], "columns": matched_columns[:10]}
    
    def _validate_mapping(self, mapping: Dict) -> Dict:
        """Validate and refine the table/column mapping"""
        
        validated = mapping.copy()
        
        # Ensure we have at least one table
        if not validated["tables"]:
            validated["tables"] = self._suggest_default_tables()
        
        # Ensure we have required columns for the intent
        if validated.get("intent"):
            missing_columns = self._identify_missing_columns(validated)
            validated["columns"].extend(missing_columns)
        
        # Remove duplicates
        validated["tables"] = self._deduplicate_tables(validated["tables"])
        validated["columns"] = self._deduplicate_columns(validated["columns"])
        
        # Recalculate confidence after validation
        validated["confidence"] = self._calculate_mapping_confidence(validated)
        
        return validated
    
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
    
    def _suggest_default_tables(self) -> List[Dict]:
        """Suggest default tables when none are identified"""
        
        common_tables = []
        
        for table in self.all_tables:
            description = (table.get('brief_description') or "").lower()
            table_name = table['table_name'].lower()
            
            # Look for common patterns
            if any(keyword in description or keyword in table_name 
                   for keyword in ["policy", "claim", "customer", "premium", "coverage"]):
                full_name = f"{table['database']}.{table['schema']}.{table['table_name']}"
                common_tables.append({
                    "name": full_name,
                    "confidence": 0.4,
                    "reason": "Common insurance table"
                })
        
        return common_tables[:2]  # Return top 2 suggestions
    
    def _identify_missing_columns(self, mapping: Dict) -> List[Dict]:
        """Identify columns that might be missing for the intent"""
        
        missing = []
        intent = mapping.get("intent")
        
        if not intent:
            return missing
        
        existing_columns = [c["name"] for c in mapping.get("columns", [])]
        
        # Check if we have columns for metrics
        for metric in intent.metrics:
            if metric not in existing_columns:
                matching_column = self._find_matching_column(metric, mapping["tables"])
                if matching_column:
                    missing.append(matching_column)
        
        return missing
    
    def _find_matching_column(self, metric: str, tables: List[Dict]) -> Optional[Dict]:
        """Find a column that matches the metric requirement"""
        
        for table in tables:
            table_name = table["name"].split('.')[-1]
            
            for column in self.all_columns:
                if column["table_name"] == table_name:
                    column_name = column["column_name"].lower()
                    
                    # Simple matching logic
                    if metric.lower() in column_name or column_name in metric.lower():
                        return {
                            "name": column["column_name"],
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
    
    def _format_conversation_context(self, session_context: Dict) -> str:
        """Format conversation history for context"""
        
        history = session_context.get("conversation_history", [])
        
        if not history:
            return "No previous conversation"
        
        formatted = []
        for item in history[-3:]:  # Last 3 interactions
            query = item.get("query", "")
            tables = item.get("tables_used", [])
            
            formatted.append(f"Previous Query: {query}")
            if tables:
                formatted.append(f"Tables Used: {', '.join(tables)}")
        
        return "\n".join(formatted)
    
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