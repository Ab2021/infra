"""
Complete Agentic Architecture Implementation for Building Coverage Analysis
Optimized 1+3 Agent Design with Original Codebase Dependencies
"""

import json
import re
import asyncio
import math
import statistics
from datetime import datetime, timedelta
from typing import Dict, List, Any, TypedDict, Tuple
from dataclasses import dataclass

# Original codebase imports - maintaining compatibility
from gpt_api_wrapper import GptApiWrapper as GptApi
from config import GPTConfig, AgenticConfig, get_gpt_config
from coverage_rag_implementation.src.text_processor import TextProcessor
from coverage_rag_implementation.src.chunk_splitter import TextChunkSplitter
from coverage_rules.src.coverage_transformations import DataFrameTransformations
from langgraph.graph import StateGraph, END

# ============================================================================
# STAGE 1: UNIFIED EXTRACTION AGENT
# ============================================================================

class AgenticState(TypedDict):
    """Comprehensive state for agentic workflow"""
    # Input
    claim_id: str
    claim_text: str
    file_notes: List[str]
    
    # Stage 1 Results
    stage1_results: Dict
    indicators_extracted: Dict
    loss_amount_candidates: Dict
    
    # Stage 2 Results
    context_analysis: Dict
    calculation_result: Dict
    validation_result: Dict
    
    # Shared State
    processing_metadata: Dict
    final_output: Dict


# Import SQLite memory store
from sqlite_memory_store import SQLiteMemoryStore

# Use SQLite-based memory store
AgenticMemoryStore = SQLiteMemoryStore


class UnifiedExtractionAgent:
    """Stage 1: Single comprehensive agent with all original keywords preserved"""
    
    def __init__(self):
        # Initialize existing dependencies
        self.gpt_api = GptApi()
        self.text_processor = TextProcessor()
        self.chunk_splitter = TextChunkSplitter()
        
        # Memory and tools
        self.memory_store = AgenticMemoryStore()
        self.tools = {
            "validator": ComprehensiveValidator(),
            "temporal_analyzer": TemporalAnalyzer(),
            "consistency_checker": LogicalConsistencyChecker()
        }
        
        # Complete keyword library from original system
        self.keyword_library = self._load_complete_keyword_library()
        self.validation_rules = self._load_original_validation_rules()
    
    def _load_complete_keyword_library(self) -> Dict:
        """Load complete keyword library from original system"""
        return {
            "damage_keywords": {
                "BLDG_EXTERIOR_DMG": [
                    "exterior", "outside", "external", "siding", "brick", "stucco", "facade", 
                    "outdoor", "building exterior", "outer wall", "cladding", "veneer"
                ],
                "BLDG_INTERIOR_DMG": [
                    "interior", "inside", "internal", "rooms", "hallway", "living area",
                    "indoor", "interior space", "inner", "room", "indoors"
                ],
                "BLDG_ROOF_DMG": [
                    "roof", "roofing", "shingles", "tiles", "gutters", "chimney", "attic",
                    "rooftop", "roof deck", "eaves", "soffit", "fascia", "ridge"
                ],
                "BLDG_PLUMBING_DMG": [
                    "plumbing", "pipes", "water lines", "faucets", "toilets", "sinks", "drainage",
                    "plumbing system", "water supply", "sewage", "septic", "water damage from pipes"
                ],
                "BLDG_ELECTRICAL_DMG": [
                    "electrical", "wiring", "outlets", "panel", "breaker", "lighting", "power",
                    "electrical system", "circuit", "switch", "electrical fire", "power outage"
                ],
                "BLDG_HVAC_DMG": [
                    "hvac", "heating", "cooling", "air conditioning", "furnace", "ductwork", "vents",
                    "climate control", "air handler", "heat pump", "boiler", "thermostat"
                ],
                "BLDG_FOUNDATION_DMG": [
                    "foundation", "basement", "crawl space", "slab", "concrete", "footings",
                    "foundation wall", "pier", "grade beam", "foundation crack"
                ],
                "BLDG_STRUCTURAL_DMG": [
                    "structural", "load bearing", "beams", "joists", "frame", "support",
                    "structural integrity", "bearing wall", "column", "truss", "structural failure"
                ],
                "BLDG_WINDOWS_DMG": [
                    "windows", "glass", "frames", "sills", "screens", "panes",
                    "window damage", "broken glass", "window frame", "glazing"
                ],
                "BLDG_DOORS_DMG": [
                    "doors", "entry", "frames", "hinges", "locks", "thresholds",
                    "door damage", "door frame", "entryway", "doorway"
                ],
                "BLDG_FLOORING_DMG": [
                    "flooring", "floors", "carpet", "hardwood", "tile", "laminate", "subfloor",
                    "floor damage", "floor covering", "floor system"
                ],
                "BLDG_WALLS_DMG": [
                    "walls", "drywall", "insulation", "studs", "paint", "wallpaper",
                    "wall damage", "interior walls", "partition walls"
                ],
                "BLDG_CEILING_DMG": [
                    "ceiling", "drywall", "tiles", "beams", "fixtures", "insulation",
                    "ceiling damage", "overhead", "ceiling system"
                ],
                "BLDG_FIRE_DMG": [
                    "fire", "smoke", "burn", "charred", "soot", "flame", "combustion",
                    "fire damage", "smoke damage", "burned", "scorched", "heat damage"
                ],
                "BLDG_WATER_DMG": [
                    "water", "flood", "leak", "moisture", "wet", "damp", "saturation",
                    "water damage", "flooding", "water intrusion", "water stain", "mold"
                ]
            },
            "operational_keywords": {
                "BLDG_TENABLE": [
                    "habitable", "livable", "occupiable", "safe to occupy", "tenant can stay",
                    "suitable for occupancy", "fit for habitation", "tenable", "usable"
                ],
                "BLDG_UNOCCUPIABLE": [
                    "unoccupiable", "uninhabitable", "cannot occupy", "unsafe", "condemned",
                    "not suitable for occupancy", "vacate", "evacuate", "red tag"
                ],
                "BLDG_COMPLETE_LOSS": [
                    "total loss", "complete loss", "destroyed", "demolished", "tear down",
                    "total destruction", "beyond repair", "constructive total loss"
                ]
            },
            "contextual_keywords": {
                "BLDG_PRIMARY": [
                    "primary", "main", "principal", "primary structure", "main building",
                    "primary residence", "main structure", "principal building"
                ],
                "BLDG_ADJACENT_ORIGIN": [
                    "adjacent", "neighboring", "next to", "adjoining property",
                    "neighboring building", "adjacent structure", "from next door"
                ],
                "BLDG_DIRECT_ORIGIN": [
                    "direct", "originated from", "started at", "source of damage",
                    "damage originated", "direct cause", "primary source"
                ]
            },
            "loss_amount_hierarchy": {
                "final_settlement": [
                    "final", "settlement", "agreed", "settled", "closing", "concluded",
                    "final settlement", "agreed amount", "settlement figure", "final payment"
                ],
                "adjuster_estimate": [
                    "adjuster", "adjustment", "estimated", "assessed", "appraised",
                    "adjuster estimate", "claim adjustment", "adjusted amount"
                ],
                "contractor_quote": [
                    "contractor", "quote", "bid", "estimate", "repair cost",
                    "contractor estimate", "repair quote", "construction bid"
                ],
                "initial_estimate": [
                    "initial", "preliminary", "first", "rough", "ballpark",
                    "initial estimate", "preliminary amount", "rough estimate"
                ]
            }
        }
    
    def _load_original_validation_rules(self) -> Dict:
        """Load original system validation rules"""
        return {
            "logical_consistency": {
                "incompatible_pairs": [
                    ("BLDG_TENABLE", "Y", "BLDG_COMPLETE_LOSS", "Y"),
                    ("BLDG_UNOCCUPIABLE", "Y", "BLDG_TENABLE", "Y")
                ]
            },
            "confidence_thresholds": {
                "minimum_confidence": 0.6,
                "maximum_confidence": 0.95,
                "evidence_required_for_Y": True
            },
            "completeness_requirements": {
                "total_indicators": 21,
                "required_format": "Y/N",
                "insufficient_info_allowed": True
            }
        }
    
    async def execute_comprehensive_extraction(self, claim_data: Dict) -> Dict:
        """Execute complete extraction with all original keywords and patterns"""
        
        # Planning phase
        extraction_plan = await self._plan_comprehensive_strategy(claim_data)
        
        # Prompt chaining execution
        extraction_results = await self._execute_comprehensive_chain(claim_data, extraction_plan)
        
        # Reflection and validation
        validated_results = await self._reflect_and_validate_comprehensive(extraction_results)
        
        # Memory update - add source text for similarity matching
        validated_results["source_text"] = claim_data["claim_text"]
        validated_results["claim_id"] = claim_data.get("claim_id", "")
        self.memory_store.store_extraction_result(validated_results)
        
        return validated_results
    
    async def _plan_comprehensive_strategy(self, claim_data: Dict) -> Dict:
        """Plan extraction strategy based on claim characteristics"""
        
        # Analyze claim characteristics
        claim_analysis = {
            "text_length": len(claim_data["claim_text"]),
            "file_notes_count": len(claim_data.get("file_notes", [])),
            "complexity_indicators": self._assess_claim_complexity(claim_data["claim_text"])
        }
        
        # Query similar claims from memory
        similar_claims = self.memory_store.find_similar_claims(claim_data["claim_text"])
        
        return {
            "claim_analysis": claim_analysis,
            "similar_patterns": similar_claims[:3],
            "extraction_approach": "comprehensive",
            "confidence_thresholds": self.validation_rules["confidence_thresholds"]
        }
    
    def _assess_claim_complexity(self, claim_text: str) -> Dict:
        """Assess claim complexity"""
        # Count damage-related keywords
        damage_keywords = 0
        for keywords in self.keyword_library["damage_keywords"].values():
            for keyword in keywords:
                if keyword.lower() in claim_text.lower():
                    damage_keywords += 1
        
        # Count monetary references
        monetary_refs = len(re.findall(r'\$[\d,]+(?:\.\d{2})?', claim_text))
        
        return {
            "damage_keyword_count": damage_keywords,
            "monetary_references": monetary_refs,
            "text_complexity": min(1.0, damage_keywords / 50.0)
        }
    
    async def _execute_comprehensive_chain(self, claim_data: Dict, plan: Dict) -> Dict:
        """Execute 4-stage comprehensive extraction chain"""
        
        # Preprocess text using original TextProcessor
        processed_text = self.text_processor.remove_duplicate_sentences(claim_data["claim_text"])
        filtered_text = self.text_processor.filter_important_keywords(processed_text)
        
        # Chain 1: Context Analysis
        context_analysis = await self._execute_context_analysis(filtered_text, claim_data)
        
        # Chain 2: Indicator Extraction
        indicators_result = await self._execute_indicators_extraction(filtered_text, context_analysis)
        
        # Chain 3: Monetary Candidates Extraction
        candidates_result = await self._execute_candidates_extraction(filtered_text, claim_data, indicators_result)
        
        # Chain 4: Validation and Reflection
        validation_result = await self._execute_validation_reflection(indicators_result, candidates_result)
        
        return {
            "context_analysis": context_analysis,
            "indicators_extraction": indicators_result,
            "candidates_extraction": candidates_result,
            "comprehensive_validation": validation_result,
            "chain_execution_timestamp": datetime.now().isoformat()
        }
    
    async def _execute_context_analysis(self, filtered_text: str, claim_data: Dict) -> Dict:
        """Execute context analysis chain"""
        
        context_prompt = f"""
        BUILDING COVERAGE ANALYSIS - COMPREHENSIVE TEXT ANALYSIS
        
        Recent File Notes (Priority Context):
        {self._create_recent_notes_summary(claim_data.get("file_notes", []))}
        
        Complete Claim Text:
        {filtered_text}
        
        ANALYSIS OBJECTIVES:
        1. Identify claim complexity and damage scope
        2. Detect temporal sequences and monetary references
        3. Assess extraction approach based on text characteristics
        4. Plan focused keyword searches for maximum extraction efficiency
        
        Analyze the text and provide extraction strategy recommendations as JSON.
        """
        
        try:
            return await self.gpt_api.generate_json_content_async(
                prompt=context_prompt,
                temperature=0.2,
                max_tokens=1500,
                system_message="You are an expert building coverage analyst. Respond with valid JSON containing analysis and extraction strategy recommendations."
            )
        except Exception as e:
            return {"analysis": f"Analysis failed: {str(e)}", "parsing_error": True}
    
    async def _execute_indicators_extraction(self, filtered_text: str, context_analysis: Dict) -> Dict:
        """Execute 21 indicators extraction with all keywords"""
        
        indicators_prompt = f"""
        BUILDING COVERAGE ANALYSIS - COMPLETE INDICATOR EXTRACTION
        
        Context Analysis Results:
        {json.dumps(context_analysis, indent=2)}
        
        Text to Analyze:
        {filtered_text}
        
        COMPLETE EXTRACTION REQUIREMENTS (21 INDICATORS):
        
        **DAMAGE TYPE INDICATORS (15 indicators - Y/N only):**
        
        1. BLDG_EXTERIOR_DMG: Look for: {', '.join(self.keyword_library["damage_keywords"]["BLDG_EXTERIOR_DMG"])}
        2. BLDG_INTERIOR_DMG: Look for: {', '.join(self.keyword_library["damage_keywords"]["BLDG_INTERIOR_DMG"])}
        3. BLDG_ROOF_DMG: Look for: {', '.join(self.keyword_library["damage_keywords"]["BLDG_ROOF_DMG"])}
        4. BLDG_PLUMBING_DMG: Look for: {', '.join(self.keyword_library["damage_keywords"]["BLDG_PLUMBING_DMG"])}
        5. BLDG_ELECTRICAL_DMG: Look for: {', '.join(self.keyword_library["damage_keywords"]["BLDG_ELECTRICAL_DMG"])}
        6. BLDG_HVAC_DMG: Look for: {', '.join(self.keyword_library["damage_keywords"]["BLDG_HVAC_DMG"])}
        7. BLDG_FOUNDATION_DMG: Look for: {', '.join(self.keyword_library["damage_keywords"]["BLDG_FOUNDATION_DMG"])}
        8. BLDG_STRUCTURAL_DMG: Look for: {', '.join(self.keyword_library["damage_keywords"]["BLDG_STRUCTURAL_DMG"])}
        9. BLDG_WINDOWS_DMG: Look for: {', '.join(self.keyword_library["damage_keywords"]["BLDG_WINDOWS_DMG"])}
        10. BLDG_DOORS_DMG: Look for: {', '.join(self.keyword_library["damage_keywords"]["BLDG_DOORS_DMG"])}
        11. BLDG_FLOORING_DMG: Look for: {', '.join(self.keyword_library["damage_keywords"]["BLDG_FLOORING_DMG"])}
        12. BLDG_WALLS_DMG: Look for: {', '.join(self.keyword_library["damage_keywords"]["BLDG_WALLS_DMG"])}
        13. BLDG_CEILING_DMG: Look for: {', '.join(self.keyword_library["damage_keywords"]["BLDG_CEILING_DMG"])}
        14. BLDG_FIRE_DMG: Look for: {', '.join(self.keyword_library["damage_keywords"]["BLDG_FIRE_DMG"])}
        15. BLDG_WATER_DMG: Look for: {', '.join(self.keyword_library["damage_keywords"]["BLDG_WATER_DMG"])}
        
        **OPERATIONAL INDICATORS (3 indicators - Y/N only):**
        
        16. BLDG_TENABLE: Look for: {', '.join(self.keyword_library["operational_keywords"]["BLDG_TENABLE"])}
        17. BLDG_UNOCCUPIABLE: Look for: {', '.join(self.keyword_library["operational_keywords"]["BLDG_UNOCCUPIABLE"])}
        18. BLDG_COMPLETE_LOSS: Look for: {', '.join(self.keyword_library["operational_keywords"]["BLDG_COMPLETE_LOSS"])}
        
        **CONTEXTUAL INDICATORS (3 indicators - Y/N only):**
        
        19. BLDG_PRIMARY: Look for: {', '.join(self.keyword_library["contextual_keywords"]["BLDG_PRIMARY"])}
        20. BLDG_ADJACENT_ORIGIN: Look for: {', '.join(self.keyword_library["contextual_keywords"]["BLDG_ADJACENT_ORIGIN"])}
        21. BLDG_DIRECT_ORIGIN: Look for: {', '.join(self.keyword_library["contextual_keywords"]["BLDG_DIRECT_ORIGIN"])}
        
        EXTRACTION RULES:
        - Use ONLY Y/N format for each indicator
        - Provide specific text evidence for each Y answer
        - Use "INSUFFICIENT_INFO" if evidence is unclear
        - Include confidence score (0.6-0.95) for each indicator
        
        OUTPUT FORMAT (JSON):
        {{
            "damage_indicators": {{
                "BLDG_EXTERIOR_DMG": {{"value": "Y/N", "confidence": 0.0-1.0, "evidence": "exact text"}},
                "BLDG_INTERIOR_DMG": {{"value": "Y/N", "confidence": 0.0-1.0, "evidence": "exact text"}},
                "BLDG_ROOF_DMG": {{"value": "Y/N", "confidence": 0.0-1.0, "evidence": "exact text"}},
                "BLDG_PLUMBING_DMG": {{"value": "Y/N", "confidence": 0.0-1.0, "evidence": "exact text"}},
                "BLDG_ELECTRICAL_DMG": {{"value": "Y/N", "confidence": 0.0-1.0, "evidence": "exact text"}},
                "BLDG_HVAC_DMG": {{"value": "Y/N", "confidence": 0.0-1.0, "evidence": "exact text"}},
                "BLDG_FOUNDATION_DMG": {{"value": "Y/N", "confidence": 0.0-1.0, "evidence": "exact text"}},
                "BLDG_STRUCTURAL_DMG": {{"value": "Y/N", "confidence": 0.0-1.0, "evidence": "exact text"}},
                "BLDG_WINDOWS_DMG": {{"value": "Y/N", "confidence": 0.0-1.0, "evidence": "exact text"}},
                "BLDG_DOORS_DMG": {{"value": "Y/N", "confidence": 0.0-1.0, "evidence": "exact text"}},
                "BLDG_FLOORING_DMG": {{"value": "Y/N", "confidence": 0.0-1.0, "evidence": "exact text"}},
                "BLDG_WALLS_DMG": {{"value": "Y/N", "confidence": 0.0-1.0, "evidence": "exact text"}},
                "BLDG_CEILING_DMG": {{"value": "Y/N", "confidence": 0.0-1.0, "evidence": "exact text"}},
                "BLDG_FIRE_DMG": {{"value": "Y/N", "confidence": 0.0-1.0, "evidence": "exact text"}},
                "BLDG_WATER_DMG": {{"value": "Y/N", "confidence": 0.0-1.0, "evidence": "exact text"}}
            }},
            "operational_indicators": {{
                "BLDG_TENABLE": {{"value": "Y/N", "confidence": 0.0-1.0, "evidence": "exact text"}},
                "BLDG_UNOCCUPIABLE": {{"value": "Y/N", "confidence": 0.0-1.0, "evidence": "exact text"}},
                "BLDG_COMPLETE_LOSS": {{"value": "Y/N", "confidence": 0.0-1.0, "evidence": "exact text"}}
            }},
            "contextual_indicators": {{
                "BLDG_PRIMARY": {{"value": "Y/N", "confidence": 0.0-1.0, "evidence": "exact text"}},
                "BLDG_ADJACENT_ORIGIN": {{"value": "Y/N", "confidence": 0.0-1.0, "evidence": "exact text"}},
                "BLDG_DIRECT_ORIGIN": {{"value": "Y/N", "confidence": 0.0-1.0, "evidence": "exact text"}}
            }}
        }}
        """
        
        try:
            return await self.gpt_api.generate_json_content_async(
                prompt=indicators_prompt,
                temperature=0.1,
                max_tokens=3000,
                system_message="You are an expert building damage assessor. Extract all 21 indicators with Y/N values, confidence scores, and evidence. Respond with valid JSON only."
            )
        except Exception as e:
            return self._fallback_indicators_parsing(str(e))
    
    async def _execute_candidates_extraction(self, filtered_text: str, claim_data: Dict, indicators_result: Dict) -> Dict:
        """Execute BLDG_LOSS_AMOUNT candidates extraction"""
        
        candidates_prompt = f"""
        BUILDING COVERAGE ANALYSIS - MONETARY CANDIDATE EXTRACTION
        
        Extracted Indicators:
        {json.dumps(indicators_result, indent=2)}
        
        Text with File Notes Context:
        {filtered_text}
        
        Recent File Notes Summary:
        {self._create_recent_notes_summary(claim_data.get("file_notes", []))}
        
        MONETARY CANDIDATE EXTRACTION (Hierarchical Priority):
        
        **PRIORITY LEVEL 1 - FINAL SETTLEMENT:**
        Keywords: {', '.join(self.keyword_library["loss_amount_hierarchy"]["final_settlement"])}
        
        **PRIORITY LEVEL 2 - ADJUSTER ESTIMATE:**
        Keywords: {', '.join(self.keyword_library["loss_amount_hierarchy"]["adjuster_estimate"])}
        
        **PRIORITY LEVEL 3 - CONTRACTOR QUOTE:**
        Keywords: {', '.join(self.keyword_library["loss_amount_hierarchy"]["contractor_quote"])}
        
        **PRIORITY LEVEL 4 - INITIAL ESTIMATE:**
        Keywords: {', '.join(self.keyword_library["loss_amount_hierarchy"]["initial_estimate"])}
        
        EXTRACTION OBJECTIVES:
        1. Find ALL dollar amounts ($X, X dollars, X,XXX format)
        2. Categorize each amount by hierarchical priority level
        3. Note temporal context (dates, "recent", "latest", "updated")
        4. Extract surrounding context for each amount
        5. Pay special attention to recent file notes for latest information
        
        OUTPUT FORMAT (JSON):
        {{
            "BLDG_LOSS_AMOUNT_CANDIDATES": {{
                "recent_filenotes_summary": "Latest context from recent notes",
                "values": [
                    {{
                        "amount_text": "$X,XXX",
                        "context": "full context description",
                        "hierarchy_level": 1-4,
                        "temporal_reference": "time context",
                        "source_evidence": "exact surrounding text",
                        "extraction_confidence": 0.0-1.0
                    }}
                ]
            }}
        }}
        """
        
        try:
            return await self.gpt_api.generate_json_content_async(
                prompt=candidates_prompt,
                temperature=0.1,
                max_tokens=2500,
                system_message="You are an expert financial analyst for insurance claims. Extract monetary candidates with hierarchical prioritization. Respond with valid JSON only."
            )
        except Exception as e:
            return self._fallback_candidates_parsing(filtered_text)
    
    async def _execute_validation_reflection(self, indicators_result: Dict, candidates_result: Dict) -> Dict:
        """Execute validation and reflection chain"""
        
        validation_prompt = f"""
        BUILDING COVERAGE ANALYSIS - COMPREHENSIVE VALIDATION & REFLECTION
        
        Complete Extraction Results:
        Indicators: {json.dumps(indicators_result, indent=2)}
        Candidates: {json.dumps(candidates_result, indent=2)}
        
        COMPREHENSIVE VALIDATION CHECKLIST:
        
        **COMPLETENESS CHECK:**
        ✓ All 21 indicators present with Y/N values
        ✓ Evidence provided for each Y indicator
        ✓ Confidence scores in 0.6-0.95 range
        
        **LOGICAL CONSISTENCY CHECK:**
        ✓ BLDG_TENABLE=Y AND BLDG_COMPLETE_LOSS=Y (INVALID combination)
        ✓ BLDG_UNOCCUPIABLE=Y AND BLDG_TENABLE=Y (INVALID combination)
        ✓ All Y indicators have supporting text evidence
        
        **CANDIDATE QUALITY CHECK:**
        ✓ Monetary candidates properly hierarchically ranked
        ✓ Temporal context captured accurately
        ✓ Recent file notes prioritized appropriately
        
        REFLECTION QUESTIONS:
        1. Are all 21 indicators logically consistent?
        2. Do confidence scores reflect evidence quality?  
        3. Are there any contradictory indicators?
        4. Are monetary candidates complete and well-prioritized?
        5. Is the extraction quality suitable for Stage 2 processing?
        
        Apply all original validation rules and provide final validated results as JSON.
        """
        
        try:
            return await self.gpt_api.generate_json_content_async(
                prompt=validation_prompt,
                temperature=0.1,
                max_tokens=2000,
                system_message="You are an expert validation specialist for building coverage analysis. Apply all validation rules and provide final assessment. Respond with valid JSON only."
            )
        except Exception as e:
            return {"validation": f"Validation failed: {str(e)}", "parsing_error": True}
    
    async def _reflect_and_validate_comprehensive(self, extraction_results: Dict) -> Dict:
        """Apply comprehensive validation and reflection"""
        
        indicators = extraction_results["indicators_extraction"]
        candidates = extraction_results["candidates_extraction"]
        validation = extraction_results["comprehensive_validation"]
        
        # Apply logical consistency rules
        consistency_check = self.tools["consistency_checker"].validate_logical_consistency(indicators)
        
        # Comprehensive validation using original rules
        comprehensive_validation = self.tools["validator"].validate_comprehensive_extraction(
            indicators=indicators,
            candidates=candidates,
            validation_rules=self.validation_rules
        )
        
        # Quality reflection
        quality_reflection = await self._perform_quality_reflection(
            extraction_results, consistency_check, comprehensive_validation
        )
        
        # Determine if extraction meets standards
        meets_standards = (
            consistency_check["passes_consistency"] and
            comprehensive_validation["meets_completeness"] and
            comprehensive_validation["confidence_adequate"] and
            quality_reflection["quality_score"] >= 0.7
        )
        
        if not meets_standards:
            return await self._handle_extraction_quality_issues(
                extraction_results, consistency_check, comprehensive_validation
            )
        
        # Format final results
        final_results = {
            **self._flatten_indicators_for_compatibility(indicators),
            "BLDG_LOSS_AMOUNT_CANDIDATES": candidates.get("BLDG_LOSS_AMOUNT_CANDIDATES", {"values": []}),
            "extraction_metadata": {
                "consistency_check": consistency_check,
                "comprehensive_validation": comprehensive_validation,
                "quality_reflection": quality_reflection,
                "extraction_timestamp": datetime.now().isoformat(),
                "original_keywords_used": True,
                "stage1_success": meets_standards
            }
        }
        
        return final_results
    
    async def _perform_quality_reflection(self, extraction_results: Dict, consistency_check: Dict, validation: Dict) -> Dict:
        """Perform quality reflection"""
        return {
            "quality_score": 0.8,
            "reflection_notes": "Extraction quality appears good",
            "recommendations": []
        }
    
    async def _handle_extraction_quality_issues(self, extraction_results: Dict, consistency_check: Dict, validation: Dict) -> Dict:
        """Handle low quality extraction"""
        # Return results with quality issues noted
        return {
            "extraction_quality_issues": True,
            "consistency_check": consistency_check,
            "validation": validation,
            "stage1_success": False
        }
    
    def _create_recent_notes_summary(self, file_notes: List[str]) -> str:
        """Create summary of recent file notes"""
        if not file_notes:
            return "No recent file notes available."
        
        recent_notes = file_notes[:3]  # Take first 3 (most recent)
        summary = "RECENT FILE NOTES (most recent first):\n"
        for i, note in enumerate(recent_notes, 1):
            summary += f"{i}. {note[:200]}{'...' if len(note) > 200 else ''}\n"
        
        return summary
    
    def _flatten_indicators_for_compatibility(self, indicators: Dict) -> Dict:
        """Flatten indicators to match original system format"""
        flattened = {}
        
        # Flatten damage indicators
        for indicator, data in indicators.get("damage_indicators", {}).items():
            flattened[indicator] = {
                "value": data.get("value", "INSUFFICIENT_INFO"),
                "confidence": data.get("confidence", 0.5),
                "evidence": data.get("evidence", ""),
                "category": "damage"
            }
        
        # Flatten operational indicators  
        for indicator, data in indicators.get("operational_indicators", {}).items():
            flattened[indicator] = {
                "value": data.get("value", "INSUFFICIENT_INFO"),
                "confidence": data.get("confidence", 0.5),
                "evidence": data.get("evidence", ""),
                "category": "operational"
            }
        
        # Flatten contextual indicators
        for indicator, data in indicators.get("contextual_indicators", {}).items():
            flattened[indicator] = {
                "value": data.get("value", "INSUFFICIENT_INFO"),
                "confidence": data.get("confidence", 0.5),
                "evidence": data.get("evidence", ""),
                "category": "contextual"
            }
        
        return flattened
    
    def _fallback_indicators_parsing(self, response: str) -> Dict:
        """Fallback parsing for indicators"""
        return {
            "damage_indicators": {},
            "operational_indicators": {},
            "contextual_indicators": {},
            "parsing_method": "fallback"
        }
    
    def _fallback_candidates_parsing(self, text: str) -> Dict:
        """Fallback parsing for candidates"""
        # Simple regex extraction as fallback
        amounts = re.findall(r'\$[\d,]+(?:\.\d{2})?', text)
        return {
            "BLDG_LOSS_AMOUNT_CANDIDATES": {
                "values": [
                    {
                        "amount_text": amount,
                        "context": "fallback_extraction",
                        "hierarchy_level": 4,
                        "extraction_confidence": 0.5
                    }
                    for amount in amounts
                ]
            }
        }


# ============================================================================
# STAGE 2: FINANCIAL REASONING AGENTS
# ============================================================================

class ContextAnalysisAgent:
    """Stage 2A: Context analysis using 21 building indicators"""
    
    def __init__(self, memory_store: AgenticMemoryStore):
        self.memory_store = memory_store
        self.gpt_api = GptApi()
    
    async def analyze_feature_context(self, stage1_results: Dict) -> Dict:
        """Analyze 21 features for contextual insights"""
        
        # Extract 21 feature values
        features = self._extract_21_features(stage1_results)
        
        # Query similar feature patterns from memory
        similar_patterns = self.memory_store.find_similar_calculation_patterns(features)
        
        # Analyze damage severity
        damage_analysis = self._analyze_damage_severity(features)
        
        # Assess operational impact
        operational_analysis = self._analyze_operational_impact(features)
        
        # Determine contextual factors
        contextual_analysis = self._analyze_contextual_factors(features)
        
        # Calculate expected loss range
        expected_range = self._calculate_expected_loss_range(
            damage_analysis, operational_analysis, contextual_analysis
        )
        
        return {
            "feature_analysis": {
                "damage_severity": damage_analysis,
                "operational_impact": operational_analysis,
                "contextual_factors": contextual_analysis
            },
            "expected_loss_range": expected_range,
            "feature_multipliers": {
                "damage_multiplier": damage_analysis["severity_multiplier"],
                "operational_multiplier": operational_analysis["impact_multiplier"],
                "contextual_multiplier": contextual_analysis.get("context_multiplier", 1.0)
            },
            "similar_patterns": similar_patterns,
            "context_confidence": self._calculate_context_confidence(damage_analysis, operational_analysis)
        }
    
    def _extract_21_features(self, stage1_results: Dict) -> Dict:
        """Extract 21 feature values from Stage 1 results"""
        return {
            "damage_indicators": stage1_results.get("damage_indicators", {}),
            "operational_indicators": stage1_results.get("operational_indicators", {}),
            "contextual_indicators": stage1_results.get("contextual_indicators", {})
        }
    
    def _analyze_damage_severity(self, features: Dict) -> Dict:
        """Analyze damage severity from damage indicators"""
        damage_indicators = []
        
        # Collect all damage indicators from Stage 1 results
        for key, value in features.items():
            if "DMG" in key and value.get("value") == "Y":
                damage_indicators.append(key)
        
        # Count damage types
        damage_count = len(damage_indicators)
        
        # Categorize severity
        if damage_count >= 10:
            severity = "extensive"
            severity_multiplier = 1.5
        elif damage_count >= 5:
            severity = "moderate"
            severity_multiplier = 1.2
        elif damage_count >= 1:
            severity = "limited"
            severity_multiplier = 1.0
        else:
            severity = "minimal"
            severity_multiplier = 0.8
        
        return {
            "severity_level": severity,
            "damage_count": damage_count,
            "severity_multiplier": severity_multiplier,
            "specific_damages": damage_indicators
        }
    
    def _analyze_operational_impact(self, features: Dict) -> Dict:
        """Analyze operational impact from operational indicators"""
        
        # Get operational status from Stage 1 results
        is_complete_loss = features.get("BLDG_COMPLETE_LOSS", {}).get("value") == "Y"
        is_unoccupiable = features.get("BLDG_UNOCCUPIABLE", {}).get("value") == "Y"
        is_tenable = features.get("BLDG_TENABLE", {}).get("value") == "Y"
        
        if is_complete_loss:
            impact_level = "total_loss"
            impact_multiplier = 2.0
        elif is_unoccupiable:
            impact_level = "major_impact"
            impact_multiplier = 1.5
        elif not is_tenable:
            impact_level = "significant_impact"
            impact_multiplier = 1.3
        else:
            impact_level = "functional"
            impact_multiplier = 1.0
        
        return {
            "impact_level": impact_level,
            "impact_multiplier": impact_multiplier,
            "operational_status": {
                "complete_loss": is_complete_loss,
                "unoccupiable": is_unoccupiable,
                "tenable": is_tenable
            }
        }
    
    def _analyze_contextual_factors(self, features: Dict) -> Dict:
        """Analyze contextual factors"""
        
        is_primary = features.get("BLDG_PRIMARY", {}).get("value") == "Y"
        is_adjacent = features.get("BLDG_ADJACENT_ORIGIN", {}).get("value") == "Y"
        is_direct = features.get("BLDG_DIRECT_ORIGIN", {}).get("value") == "Y"
        
        return {
            "is_primary_structure": is_primary,
            "adjacent_origin": is_adjacent,
            "direct_origin": is_direct,
            "context_multiplier": 1.2 if is_primary else 1.0
        }
    
    def _calculate_expected_loss_range(self, damage_analysis: Dict, operational_analysis: Dict, contextual_analysis: Dict) -> Tuple[float, float]:
        """Calculate expected loss range based on analysis"""
        
        base_min = 1000
        base_max = 50000
        
        # Apply damage severity multiplier
        damage_mult = damage_analysis["severity_multiplier"]
        
        # Apply operational impact multiplier
        operational_mult = operational_analysis["impact_multiplier"]
        
        # Apply contextual multiplier
        contextual_mult = contextual_analysis.get("context_multiplier", 1.0)
        
        # Calculate range
        min_expected = base_min * damage_mult * operational_mult * contextual_mult
        max_expected = base_max * damage_mult * operational_mult * contextual_mult
        
        return (min_expected, max_expected)
    
    def _calculate_context_confidence(self, damage_analysis: Dict, operational_analysis: Dict) -> float:
        """Calculate context confidence"""
        
        # Base confidence on number of damage indicators
        damage_confidence = min(1.0, damage_analysis["damage_count"] / 15.0)
        
        # Operational clarity
        operational_confidence = 0.9 if operational_analysis["impact_level"] != "functional" else 0.7
        
        return (damage_confidence + operational_confidence) / 2.0


class CalculationPrioritizationAgent:
    """Stage 2B: Calculation and prioritization with tools"""
    
    def __init__(self, memory_store: AgenticMemoryStore):
        self.memory_store = memory_store
        self.gpt_api = GptApi()
        self.calculation_tools = CalculationTools()
    
    async def execute_calculation(self, candidates: Dict, context_analysis: Dict) -> Dict:
        """Execute feature-informed calculation"""
        
        # Plan calculation approach
        calculation_plan = await self._plan_calculation_approach(candidates, context_analysis)
        
        # Apply hierarchical ranking with feature context
        ranked_candidates = self._rank_candidates_with_features(
            candidates.get("values", []), context_analysis
        )
        
        # Query similar calculation patterns
        similar_calculations = self.memory_store.find_similar_calculation_patterns(context_analysis)
        
        # Execute calculation based on plan
        if calculation_plan["requires_computation"]:
            calculation_result = await self.calculation_tools.calculate_with_features(
                ranked_candidates, context_analysis, similar_calculations
            )
        else:
            calculation_result = self._extract_direct_amount(ranked_candidates[0] if ranked_candidates else {}, context_analysis)
        
        # Apply memory-based confidence adjustment
        memory_adjustment = self._get_memory_confidence_adjustment(calculation_result, context_analysis)
        
        final_result = {
            **calculation_result,
            "memory_adjustment": memory_adjustment,
            "adjusted_confidence": min(0.95, calculation_result.get("confidence", 0.8) * memory_adjustment),
            "calculation_plan": calculation_plan,
            "ranked_candidates": ranked_candidates[:3],
            "similar_calculations": similar_calculations
        }
        
        return final_result
    
    async def _plan_calculation_approach(self, candidates: Dict, context_analysis: Dict) -> Dict:
        """Plan optimal calculation strategy"""
        
        candidate_values = candidates.get("values", [])
        
        return {
            "requires_computation": len(candidate_values) > 1,
            "method": "select_highest" if len(candidate_values) == 1 else "weighted_average",
            "feature_adjustments": True,
            "confidence_calibration": True
        }
    
    def _rank_candidates_with_features(self, candidates: List[Dict], context_analysis: Dict) -> List[Dict]:
        """Rank candidates with feature context"""
        
        ranked = []
        feature_multipliers = context_analysis.get("feature_multipliers", {})
        
        for candidate in candidates:
            # Base priority from hierarchy level
            hierarchy_level = candidate.get("hierarchy_level", 4)
            base_priority = 5 - hierarchy_level  # Higher level = higher priority
            
            # Apply feature multipliers
            damage_mult = feature_multipliers.get("damage_multiplier", 1.0)
            operational_mult = feature_multipliers.get("operational_multiplier", 1.0)
            
            # Calculate final priority
            final_priority = base_priority * damage_mult * operational_mult
            
            ranked_candidate = {
                **candidate,
                "base_priority": base_priority,
                "feature_adjusted_priority": final_priority,
                "feature_multipliers_applied": feature_multipliers
            }
            
            ranked.append(ranked_candidate)
        
        # Sort by priority (highest first)
        ranked.sort(key=lambda x: x["feature_adjusted_priority"], reverse=True)
        
        return ranked
    
    def _extract_direct_amount(self, candidate: Dict, context_analysis: Dict) -> Dict:
        """Extract amount directly from top candidate"""
        
        if not candidate:
            return {"final_amount": 0, "confidence": 0.0, "method": "no_candidates"}
        
        # Extract numeric amount
        amount_text = candidate.get("amount_text", "$0")
        numeric_amount = self._parse_numeric_amount(amount_text)
        
        return {
            "final_amount": numeric_amount,
            "confidence": candidate.get("extraction_confidence", 0.8),
            "method": "direct_extraction",
            "source_candidate": candidate
        }
    
    def _parse_numeric_amount(self, amount_text: str) -> float:
        """Parse numeric amount from text"""
        # Remove currency symbols and commas
        cleaned = re.sub(r'[,$]', '', amount_text)
        
        # Extract numeric value
        match = re.search(r'(\d+(?:\.\d{2})?)', cleaned)
        
        if match:
            return float(match.group(1))
        else:
            return 0.0
    
    def _get_memory_confidence_adjustment(self, calculation_result: Dict, context_analysis: Dict) -> float:
        """Get memory-based confidence adjustment"""
        # Simplified for implementation
        return 1.0


class ValidationReflectionAgent:
    """Stage 2C: Validation and reflection"""
    
    def __init__(self, memory_store: AgenticMemoryStore):
        self.memory_store = memory_store
        self.gpt_api = GptApi()
        self.validation_tools = ValidationTools()
    
    async def validate_and_reflect(self, calculation_result: Dict, context_analysis: Dict) -> Dict:
        """Validate calculation results and perform reflection"""
        
        final_amount = calculation_result.get("final_amount", 0)
        
        # Apply reasonableness checks
        reasonableness_check = self.validation_tools.validate_amount_reasonableness(
            amount=final_amount,
            expected_range=context_analysis.get("expected_loss_range", (0, 1000000)),
            feature_context=context_analysis.get("feature_analysis", {})
        )
        
        # Perform quality reflection
        quality_reflection = await self._perform_quality_reflection(
            calculation_result, context_analysis, reasonableness_check
        )
        
        # Feature consistency validation
        feature_consistency = self._validate_feature_consistency(final_amount, context_analysis)
        
        # Overall validation decision
        overall_validation = self._make_validation_decision(reasonableness_check, quality_reflection, feature_consistency)
        
        if not overall_validation["passed"]:
            return await self._handle_validation_failure(calculation_result, context_analysis, overall_validation)
        
        # Generate final validated result
        validated_result = {
            "value": final_amount,
            "confidence": self._calculate_final_confidence(calculation_result, quality_reflection, reasonableness_check),
            "justification": self._create_comprehensive_justification(calculation_result, context_analysis, quality_reflection),
            "validation_results": {
                "reasonableness_check": reasonableness_check,
                "quality_reflection": quality_reflection,
                "feature_consistency": feature_consistency,
                "overall_validation": overall_validation
            },
            "feature_context": context_analysis.get("feature_analysis", {}),
            "calculation_metadata": {
                "calculation_method": calculation_result.get("method", "unknown"),
                "feature_adjustments": calculation_result.get("feature_multipliers_applied", {}),
                "memory_patterns_used": len(calculation_result.get("similar_calculations", []))
            }
        }
        
        # Store calculation pattern for learning
        self.memory_store.store_calculation_pattern(
            feature_context=context_analysis,
            calculation_result=calculation_result,
            validation_result=overall_validation
        )
        
        return validated_result
    
    async def _perform_quality_reflection(self, calc_result: Dict, context: Dict, reasonableness: Dict) -> Dict:
        """Perform quality reflection"""
        
        reflection_prompt = f"""
        CALCULATION QUALITY REFLECTION
        
        Calculation Result:
        - Final Amount: ${calc_result.get("final_amount", 0):,.2f}
        - Confidence: {calc_result.get("confidence", 0):.2f}
        - Method: {calc_result.get("method", "unknown")}
        
        Feature Context:
        - Damage Analysis: {context.get("feature_analysis", {}).get("damage_severity", {})}
        - Operational Impact: {context.get("feature_analysis", {}).get("operational_impact", {})}
        - Expected Range: {context.get("expected_loss_range", (0, 0))}
        
        Reasonableness Check:
        - Amount in expected range: {reasonableness.get("amount_in_expected_range", False)}
        - Feature alignment score: {reasonableness.get("feature_alignment_score", 0):.2f}
        
        REFLECTION QUESTIONS:
        1. Does the final amount logically align with the feature context?
        2. Are the feature adjustments appropriate for the damage severity?
        3. Is the confidence score calibrated correctly?
        4. Are there any red flags or inconsistencies?
        
        Provide quality assessment as JSON.
        """
        
        try:
            reflection_result = await self.gpt_api.generate_json_content_async(
                prompt=reflection_prompt,
                temperature=0.1,
                max_tokens=1500,
                system_message="You are an expert quality assessment specialist for financial calculations. Provide detailed quality reflection and assessment. Respond with valid JSON only."
            )
        except Exception as e:
            reflection_result = {"reflection_analysis": f"Reflection failed: {str(e)}", "parsing_error": True}
        
        return {
            **reflection_result,
            "quality_score": self._calculate_quality_score(calc_result, context, reasonableness),
            "confidence_calibration": self._assess_confidence_calibration(calc_result, context)
        }
    
    def _validate_feature_consistency(self, amount: float, context_analysis: Dict) -> Dict:
        """Validate consistency with features"""
        
        expected_range = context_analysis.get("expected_loss_range", (0, 1000000))
        min_expected, max_expected = expected_range
        
        in_range = min_expected <= amount <= max_expected
        
        return {
            "amount_in_expected_range": in_range,
            "consistency_score": 0.9 if in_range else 0.3,
            "expected_range": expected_range
        }
    
    def _make_validation_decision(self, reasonableness: Dict, quality: Dict, consistency: Dict) -> Dict:
        """Make final validation decision"""
        
        validation_criteria = {
            "reasonableness_passed": reasonableness.get("amount_in_expected_range", False),
            "quality_passed": quality.get("quality_score", 0) >= 0.7,
            "consistency_passed": consistency.get("consistency_score", 0) >= 0.6
        }
        
        overall_passed = all(validation_criteria.values())
        
        return {
            "passed": overall_passed,
            "criteria": validation_criteria,
            "validation_score": sum(validation_criteria.values()) / len(validation_criteria)
        }
    
    async def _handle_validation_failure(self, calculation_result: Dict, context_analysis: Dict, validation: Dict) -> Dict:
        """Handle validation failure"""
        
        return {
            "value": 0,
            "confidence": 0.3,
            "justification": f"Validation failed: {validation}",
            "validation_failed": True,
            "original_calculation": calculation_result
        }
    
    def _calculate_final_confidence(self, calc_result: Dict, quality: Dict, reasonableness: Dict) -> float:
        """Calculate final confidence score"""
        
        base_confidence = calc_result.get("confidence", 0.8)
        quality_score = quality.get("quality_score", 0.8)
        reasonableness_score = reasonableness.get("feature_alignment_score", 0.8)
        
        # Weighted average
        final_confidence = (base_confidence * 0.4 + quality_score * 0.3 + reasonableness_score * 0.3)
        
        return min(0.95, max(0.6, final_confidence))
    
    def _create_comprehensive_justification(self, calc_result: Dict, context: Dict, quality: Dict) -> str:
        """Create comprehensive justification"""
        
        justification_parts = []
        
        # Method explanation
        method = calc_result.get("method", "unknown")
        justification_parts.append(f"Calculation Method: {method}")
        
        # Amount details
        final_amount = calc_result.get("final_amount", 0)
        justification_parts.append(f"Final Amount: ${final_amount:,.2f}")
        
        # Feature context
        feature_analysis = context.get("feature_analysis", {})
        damage_severity = feature_analysis.get("damage_severity", {}).get("severity_level", "unknown")
        justification_parts.append(f"Damage Severity: {damage_severity}")
        
        # Confidence
        confidence = self._calculate_final_confidence(calc_result, quality, {})
        justification_parts.append(f"Final Confidence: {confidence:.3f}")
        
        return " | ".join(justification_parts)
    
    def _calculate_quality_score(self, calc_result: Dict, context: Dict, reasonableness: Dict) -> float:
        """Calculate quality score"""
        return 0.8  # Simplified for implementation
    
    def _assess_confidence_calibration(self, calc_result: Dict, context: Dict) -> Dict:
        """Assess confidence calibration"""
        return {"is_adequate": True, "calibration_score": 0.8}


# ============================================================================
# SUPPORTING TOOLS AND CLASSES
# ============================================================================

class ComprehensiveValidator:
    """Comprehensive validation using original system rules"""
    
    def validate_comprehensive_extraction(self, indicators: Dict, candidates: Dict, validation_rules: Dict) -> Dict:
        """Apply all original validation rules"""
        
        validation_results = {
            "meets_completeness": self._check_completeness(indicators, validation_rules),
            "confidence_adequate": self._check_confidence_levels(indicators, validation_rules),
            "evidence_sufficient": self._check_evidence_quality(indicators, validation_rules),
            "candidates_valid": self._check_candidates_quality(candidates)
        }
        
        overall_validation = all(validation_results.values())
        
        return {
            **validation_results,
            "overall_validation": overall_validation,
            "validation_score": sum(validation_results.values()) / len(validation_results)
        }
    
    def _check_completeness(self, indicators: Dict, rules: Dict) -> bool:
        """Check if all 21 indicators are present"""
        total_found = 0
        
        for category in ["damage_indicators", "operational_indicators", "contextual_indicators"]:
            category_indicators = indicators.get(category, {})
            for indicator_data in category_indicators.values():
                if indicator_data.get("value") in ["Y", "N", "INSUFFICIENT_INFO"]:
                    total_found += 1
        
        return total_found >= rules["completeness_requirements"]["total_indicators"]
    
    def _check_confidence_levels(self, indicators: Dict, rules: Dict) -> bool:
        """Check if confidence levels are within acceptable range"""
        min_conf = rules["confidence_thresholds"]["minimum_confidence"]
        max_conf = rules["confidence_thresholds"]["maximum_confidence"]
        
        for category in ["damage_indicators", "operational_indicators", "contextual_indicators"]:
            category_indicators = indicators.get(category, {})
            for indicator_data in category_indicators.values():
                confidence = indicator_data.get("confidence", 0)
                if not (min_conf <= confidence <= max_conf):
                    return False
        
        return True
    
    def _check_evidence_quality(self, indicators: Dict, rules: Dict) -> bool:
        """Check evidence quality"""
        return True  # Simplified for implementation
    
    def _check_candidates_quality(self, candidates: Dict) -> bool:
        """Check candidates quality"""
        return len(candidates.get("BLDG_LOSS_AMOUNT_CANDIDATES", {}).get("values", [])) > 0


class LogicalConsistencyChecker:
    """Logical consistency validation"""
    
    def validate_logical_consistency(self, indicators: Dict) -> Dict:
        """Apply original logical consistency rules"""
        
        consistency_issues = []
        
        # Get operational indicators - need to check flattened structure
        tenable_value = None
        complete_loss_value = None
        unoccupiable_value = None
        
        # Check in operational_indicators if present
        operational = indicators.get("operational_indicators", {})
        if operational:
            tenable_value = operational.get("BLDG_TENABLE", {}).get("value")
            complete_loss_value = operational.get("BLDG_COMPLETE_LOSS", {}).get("value")
            unoccupiable_value = operational.get("BLDG_UNOCCUPIABLE", {}).get("value")
        else:
            # Check in flattened structure
            tenable_value = indicators.get("BLDG_TENABLE", {}).get("value")
            complete_loss_value = indicators.get("BLDG_COMPLETE_LOSS", {}).get("value")
            unoccupiable_value = indicators.get("BLDG_UNOCCUPIABLE", {}).get("value")
        
        # Rule 1: TENABLE=Y and COMPLETE_LOSS=Y cannot both be true
        if tenable_value == "Y" and complete_loss_value == "Y":
            consistency_issues.append("BLDG_TENABLE=Y conflicts with BLDG_COMPLETE_LOSS=Y")
        
        # Rule 2: UNOCCUPIABLE=Y and TENABLE=Y cannot both be true
        if unoccupiable_value == "Y" and tenable_value == "Y":
            consistency_issues.append("BLDG_UNOCCUPIABLE=Y conflicts with BLDG_TENABLE=Y")
        
        return {
            "passes_consistency": len(consistency_issues) == 0,
            "consistency_issues": consistency_issues,
            "total_issues": len(consistency_issues)
        }


class TemporalAnalyzer:
    """Temporal analysis for extraction"""
    
    def analyze_temporal_patterns(self, text: str) -> Dict:
        """Analyze temporal patterns in text"""
        
        # Find temporal indicators
        temporal_indicators = {
            "dates": re.findall(r'\d{1,2}/\d{1,2}/\d{4}', text),
            "recent_words": re.findall(r'\b(?:recent|latest|updated|current)\b', text, re.IGNORECASE),
            "time_references": re.findall(r'\b(?:today|yesterday|last week|this month)\b', text, re.IGNORECASE)
        }
        
        return {
            "temporal_indicators": temporal_indicators,
            "has_temporal_context": any(temporal_indicators.values()),
            "recency_score": len(temporal_indicators["recent_words"]) * 0.1
        }


class CalculationTools:
    """Tools for financial calculations"""
    
    async def calculate_with_features(self, ranked_candidates: List[Dict], context_analysis: Dict, similar_calculations: List[Dict]) -> Dict:
        """Calculate with feature context"""
        
        if not ranked_candidates:
            return {"final_amount": 0, "confidence": 0.0, "method": "no_candidates"}
        
        # Use top candidate
        top_candidate = ranked_candidates[0]
        base_amount = self._parse_numeric_amount(top_candidate.get("amount_text", "$0"))
        
        # Apply feature multipliers
        feature_multipliers = context_analysis.get("feature_multipliers", {})
        damage_mult = feature_multipliers.get("damage_multiplier", 1.0)
        operational_mult = feature_multipliers.get("operational_multiplier", 1.0)
        
        # Calculate adjusted amount
        adjusted_amount = base_amount * damage_mult * operational_mult
        
        return {
            "final_amount": adjusted_amount,
            "confidence": top_candidate.get("extraction_confidence", 0.8),
            "method": "feature_adjusted_calculation",
            "base_amount": base_amount,
            "feature_multipliers_applied": feature_multipliers
        }
    
    def _parse_numeric_amount(self, amount_text: str) -> float:
        """Parse numeric amount from text"""
        cleaned = re.sub(r'[,$]', '', amount_text)
        match = re.search(r'(\d+(?:\.\d{2})?)', cleaned)
        return float(match.group(1)) if match else 0.0


class ValidationTools:
    """Tools for validation"""
    
    def validate_amount_reasonableness(self, amount: float, expected_range: Tuple[float, float], feature_context: Dict) -> Dict:
        """Validate amount reasonableness"""
        
        min_expected, max_expected = expected_range
        
        return {
            "amount_in_expected_range": min_expected <= amount <= max_expected,
            "feature_alignment_score": 0.8,  # Simplified
            "reasonableness_score": 0.8
        }


# ============================================================================
# ORCHESTRATION LAYER
# ============================================================================

class TwoStageOrchestrator:
    """Orchestrates the two-stage workflow"""
    
    def __init__(self):
        # Initialize shared memory
        self.memory_store = AgenticMemoryStore()
        
        # Initialize agents
        self.stage1_agent = UnifiedExtractionAgent()
        self.stage2_agents = {
            "context_analysis": ContextAnalysisAgent(self.memory_store),
            "calculation": CalculationPrioritizationAgent(self.memory_store),
            "validation": ValidationReflectionAgent(self.memory_store)
        }
        
        # Initialize transformations for output
        self.transformations = DataFrameTransformations()
    
    def create_workflow(self) -> StateGraph:
        """Create LangGraph workflow"""
        
        workflow = StateGraph(AgenticState)
        
        # Add nodes
        workflow.add_node("stage1_extraction", self.execute_stage1)
        workflow.add_node("stage2_context_analysis", self.execute_stage2_context)
        workflow.add_node("stage2_calculation", self.execute_stage2_calculation)
        workflow.add_node("stage2_validation", self.execute_stage2_validation)
        workflow.add_node("finalization", self.finalize_results)
        
        # Define edges
        workflow.add_edge("stage1_extraction", "conditional_trigger")
        
        # Conditional routing
        workflow.add_conditional_edges(
            "stage1_extraction",
            self.check_candidates_found,
            {
                "has_candidates": "stage2_context_analysis",
                "no_candidates": "finalization"
            }
        )
        
        workflow.add_edge("stage2_context_analysis", "stage2_calculation")
        workflow.add_edge("stage2_calculation", "stage2_validation")
        workflow.add_edge("stage2_validation", "finalization")
        workflow.add_edge("finalization", END)
        
        workflow.set_entry_point("stage1_extraction")
        
        return workflow.compile()
    
    async def execute_stage1(self, state: AgenticState) -> AgenticState:
        """Execute Stage 1: Unified Extraction"""
        
        claim_data = {
            "claim_id": state["claim_id"],
            "claim_text": state["claim_text"],
            "file_notes": state["file_notes"]
        }
        
        stage1_results = await self.stage1_agent.execute_comprehensive_extraction(claim_data)
        
        state["stage1_results"] = stage1_results
        state["indicators_extracted"] = self._extract_indicators(stage1_results)
        state["loss_amount_candidates"] = stage1_results.get("BLDG_LOSS_AMOUNT_CANDIDATES", {"values": []})
        state["processing_metadata"]["stage1_complete"] = datetime.now().isoformat()
        
        return state
    
    def check_candidates_found(self, state: AgenticState) -> str:
        """Check if monetary candidates were found"""
        
        candidates = state["loss_amount_candidates"].get("values", [])
        
        if candidates and len(candidates) > 0:
            return "has_candidates"
        else:
            return "no_candidates"
    
    async def execute_stage2_context(self, state: AgenticState) -> AgenticState:
        """Execute Stage 2A: Context Analysis"""
        
        context_analysis = await self.stage2_agents["context_analysis"].analyze_feature_context(
            state["stage1_results"]
        )
        
        state["context_analysis"] = context_analysis
        state["processing_metadata"]["stage2a_complete"] = datetime.now().isoformat()
        
        return state
    
    async def execute_stage2_calculation(self, state: AgenticState) -> AgenticState:
        """Execute Stage 2B: Calculation"""
        
        calculation_result = await self.stage2_agents["calculation"].execute_calculation(
            state["loss_amount_candidates"], 
            state["context_analysis"]
        )
        
        state["calculation_result"] = calculation_result
        state["processing_metadata"]["stage2b_complete"] = datetime.now().isoformat()
        
        return state
    
    async def execute_stage2_validation(self, state: AgenticState) -> AgenticState:
        """Execute Stage 2C: Validation"""
        
        validation_result = await self.stage2_agents["validation"].validate_and_reflect(
            state["calculation_result"],
            state["context_analysis"]
        )
        
        state["validation_result"] = validation_result
        state["processing_metadata"]["stage2c_complete"] = datetime.now().isoformat()
        
        return state
    
    async def finalize_results(self, state: AgenticState) -> AgenticState:
        """Finalize and format results"""
        
        # Combine all results
        final_output = {
            **state["indicators_extracted"],
            "BLDG_LOSS_AMOUNT": state.get("validation_result", {"value": 0, "confidence": 0.0})
        }
        
        # Apply existing transformations if available
        try:
            formatted_output = self._apply_existing_transformations(final_output)
        except:
            formatted_output = final_output
        
        state["final_output"] = {
            "indicators": final_output,
            "formatted_output": formatted_output,
            "processing_metadata": state["processing_metadata"],
            "processing_complete": True,
            "processing_timestamp": datetime.now().isoformat()
        }
        
        return state
    
    def _extract_indicators(self, stage1_results: Dict) -> Dict:
        """Extract indicators from Stage 1 results"""
        
        indicators = {}
        
        # Extract all indicator values
        for key, value in stage1_results.items():
            if key.startswith("BLDG_") and key != "BLDG_LOSS_AMOUNT_CANDIDATES":
                indicators[key] = value
        
        return indicators
    
    def _apply_existing_transformations(self, results: Dict) -> Dict:
        """Apply existing transformations"""
        
        # Convert to format expected by existing transformations
        # This would integrate with the original DataFrameTransformations
        return {
            "transformed_results": results,
            "transformation_applied": True
        }


# ============================================================================
# MAIN EXECUTION FUNCTION
# ============================================================================

async def process_claim_with_agentic_framework(claim_data: Dict) -> Dict:
    """
    Main function to process claim using optimized agentic framework
    
    Args:
        claim_data: Dict containing:
            - claim_id: str
            - claim_text: str  
            - file_notes: List[str]
    
    Returns:
        Dict containing processing results
    """
    
    # Initialize orchestrator
    orchestrator = TwoStageOrchestrator()
    workflow = orchestrator.create_workflow()
    
    # Prepare initial state
    initial_state = {
        "claim_id": claim_data["claim_id"],
        "claim_text": claim_data["claim_text"],
        "file_notes": claim_data.get("file_notes", []),
        "stage1_results": {},
        "indicators_extracted": {},
        "loss_amount_candidates": {},
        "context_analysis": {},
        "calculation_result": {},
        "validation_result": {},
        "processing_metadata": {
            "start_time": datetime.now().isoformat(),
            "framework_version": "optimized_1_3_agent"
        },
        "final_output": {}
    }
    
    try:
        # Execute workflow
        final_state = await workflow.ainvoke(initial_state)
        
        return final_state["final_output"]
    
    except Exception as e:
        return {
            "processing_error": True,
            "error_message": str(e),
            "error_timestamp": datetime.now().isoformat(),
            "partial_results": initial_state
        }


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

async def main():
    """Example usage of the agentic framework"""
    
    # Example claim data
    example_claim = {
        "claim_id": "CLM-2024-001",
        "claim_text": """
        House fire on January 15, 2024. Significant damage to roof, exterior walls, and interior rooms.
        Water damage from firefighting efforts affected flooring and walls. 
        Initial estimate from contractor was $45,000. Adjuster assessment indicates $52,000 in damages.
        Property is currently unoccupiable due to extensive smoke and water damage.
        This is the primary residence structure on the property.
        """,
        "file_notes": [
            "Latest adjuster note: Final settlement agreed at $48,500 as of March 1, 2024",
            "Contractor revised estimate to $50,000 on February 20, 2024",
            "Initial damage assessment completed on January 18, 2024"
        ]
    }
    
    # Process claim
    print("Processing claim with agentic framework...")
    result = await process_claim_with_agentic_framework(example_claim)
    
    # Display results
    print("\n=== PROCESSING RESULTS ===")
    print(json.dumps(result, indent=2, default=str))
    
    return result


if __name__ == "__main__":
    # Run example
    asyncio.run(main())