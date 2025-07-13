"""
Natural Language Understanding Agent
Transforms human language into structured, machine-understandable components.
"""

import json
import uuid
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass
from langchain_core.messages import HumanMessage

@dataclass
class EntityExtraction:
    """Represents an extracted entity with metadata."""
    type: str
    value: str
    confidence: float
    position: tuple = None

@dataclass
class QueryIntent:
    """Structured representation of user query intent."""
    primary_action: str
    data_focus: str
    output_preference: str
    temporal_scope: Optional[str] = None
    analysis_requirements: List[str] = None

class NLUAgent:
    """
    Sophisticated natural language understanding agent that leverages
    memory context to improve understanding accuracy over time.
    """
    
    def __init__(self, memory_system, llm_provider):
        self.memory_system = memory_system
        self.llm_provider = llm_provider
        self.agent_name = "nlu_processor"
        
    async def process_query(self, query: str, context: Dict) -> Dict:
        """
        Processes natural language input to extract intent, entities, and requirements.
        
        Args:
            query: User's natural language query
            context: Session and user context from memory
            
        Returns:
            Structured analysis with intent, entities, and confidence scores
        """
        
        # Retrieve relevant memory context for enhanced understanding
        memory_context = await self._get_memory_context(query, context)
        
        # Create enhanced prompt with memory integration
        prompt = self._create_nlu_prompt(query, memory_context, context)
        
        try:
            # Process with LLM to extract structured information
            response = await self.llm_provider.ainvoke([HumanMessage(content=prompt)])
            parsed_response = self._parse_nlu_response(response.content)
            
            # Extract components
            query_intent = self._extract_intent(parsed_response)
            entities = self._extract_entities(parsed_response)
            ambiguities = parsed_response.get("ambiguities", [])
            confidence_scores = parsed_response.get("confidence", {})
            
            # Update working memory with findings
            await self._update_memory(query, query_intent, entities, confidence_scores)
            
            return {
                "query_intent": query_intent.__dict__,
                "entities_extracted": [e.__dict__ for e in entities],
                "ambiguities_detected": ambiguities,
                "confidence_scores": confidence_scores,
                "processing_metadata": {
                    "agent": self.agent_name,
                    "timestamp": datetime.now().isoformat(),
                    "memory_context_used": bool(memory_context)
                }
            }
            
        except Exception as e:
            return {
                "error": str(e),
                "agent": self.agent_name,
                "recovery_strategy": "clarification_needed"
            }
    
    def _create_nlu_prompt(self, query: str, memory_context: Dict, session_context: Dict) -> str:
        """Creates memory-enhanced prompt for better understanding."""
        
        prompt = f"""
        You are an expert at understanding natural language queries about data analysis.
        Analyze the following query and extract structured information.
        
        User Query: "{query}"
        """
        
        if memory_context.get("similar_queries"):
            prompt += f"""
        
        Similar past queries:
        {self._format_similar_queries(memory_context["similar_queries"])}
        """
        
        if session_context.get("conversation_history"):
            prompt += f"""
        
        Recent conversation context:
        {self._format_conversation_history(session_context["conversation_history"])}
        """
        
        prompt += """
        
        Provide a JSON response with this structure:
        {
            "intent": {
                "primary_action": "select|aggregate|filter|join|sort|compare|analyze",
                "data_focus": "description of what data user wants",
                "output_preference": "table|chart|dashboard|summary",
                "temporal_scope": "time period if specified",
                "analysis_requirements": ["list of analysis types needed"]
            },
            "entities": [
                {
                    "type": "table|column|value|date|metric|dimension",
                    "value": "extracted entity text",
                    "confidence": 0.0-1.0
                }
            ],
            "ambiguities": ["descriptions of unclear aspects"],
            "confidence": {
                "overall": 0.0-1.0,
                "intent_clarity": 0.0-1.0,
                "entity_extraction": 0.0-1.0
            }
        }
        """
        
        return prompt
    
    async def _get_memory_context(self, query: str, context: Dict) -> Dict:
        """Retrieves relevant memory context to enhance understanding."""
        return await self.memory_system.get_contextual_memories(
            query=query,
            user_id=context.get("user_id"),
            context_type="nlu"
        )
    
    def _extract_intent(self, parsed_response: Dict) -> QueryIntent:
        """Extracts structured intent from parsed response."""
        intent_data = parsed_response.get("intent", {})
        return QueryIntent(
            primary_action=intent_data.get("primary_action", "select"),
            data_focus=intent_data.get("data_focus", ""),
            output_preference=intent_data.get("output_preference", "table"),
            temporal_scope=intent_data.get("temporal_scope"),
            analysis_requirements=intent_data.get("analysis_requirements", [])
        )
    
    def _extract_entities(self, parsed_response: Dict) -> List[EntityExtraction]:
        """Extracts entity objects from parsed response."""
        entities_data = parsed_response.get("entities", [])
        return [
            EntityExtraction(
                type=entity.get("type", "unknown"),
                value=entity.get("value", ""),
                confidence=entity.get("confidence", 0.0)
            )
            for entity in entities_data
        ]
    
    def _parse_nlu_response(self, response_text: str) -> Dict:
        """Safely parses LLM response as JSON."""
        try:
            # Handle potential markdown formatting
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0]
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0]
            
            return json.loads(response_text.strip())
        except json.JSONDecodeError:
            return {"error": "Failed to parse LLM response as JSON"}
    
    async def _update_memory(self, query: str, intent: QueryIntent, entities: List[EntityExtraction], confidence: Dict):
        """Updates memory with processing results for future learning."""
        await self.memory_system.working_memory.update_context(
            agent_name=self.agent_name,
            update_data={
                "processed_query": query,
                "extracted_intent": intent.__dict__,
                "extracted_entities": [e.__dict__ for e in entities],
                "confidence_scores": confidence,
                "processing_timestamp": datetime.now().isoformat()
            }
        )
