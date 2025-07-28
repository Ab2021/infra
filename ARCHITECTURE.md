# Optimized Architecture: Reduced Agents with Complete Keyword Preservation

## Executive Summary

Based on comprehensive analysis of the original codebase prompts and extraction patterns, this architecture reduces Stage 1 to **1 agent** (preferred) or **2 agents maximum** while preserving all critical keywords, validation rules, and extraction logic from the original system. Stage 2 maintains 3 specialized agents for optimal financial reasoning.

---

## Research Findings: Original System Keywords & Patterns

### Complete 22 Building Indicators Identified

**FINANCIAL INDICATOR (1):**
- `BLDG_LOSS_AMOUNT` - Complex hierarchical temporal extraction

**DAMAGE TYPE INDICATORS (15):**
- `BLDG_EXTERIOR_DMG`, `BLDG_INTERIOR_DMG`, `BLDG_ROOF_DMG`
- `BLDG_PLUMBING_DMG`, `BLDG_ELECTRICAL_DMG`, `BLDG_HVAC_DMG`
- `BLDG_FOUNDATION_DMG`, `BLDG_STRUCTURAL_DMG`, `BLDG_WINDOWS_DMG`
- `BLDG_DOORS_DMG`, `BLDG_FLOORING_DMG`, `BLDG_WALLS_DMG`
- `BLDG_CEILING_DMG`, `BLDG_FIRE_DMG`, `BLDG_WATER_DMG`

**OPERATIONAL INDICATORS (3):**
- `BLDG_TENABLE`, `BLDG_UNOCCUPIABLE`, `BLDG_COMPLETE_LOSS`

**CONTEXTUAL INDICATORS (3):**
- `BLDG_PRIMARY`, `BLDG_ADJACENT_ORIGIN`, `BLDG_DIRECT_ORIGIN`

### Critical Extraction Keywords & Patterns

**Damage Keywords by Type:**
```python
DAMAGE_KEYWORDS = {
    "BLDG_EXTERIOR_DMG": ["exterior", "outside", "external", "siding", "brick", "stucco", "facade"],
    "BLDG_INTERIOR_DMG": ["interior", "inside", "internal", "rooms", "hallway", "living area"],
    "BLDG_ROOF_DMG": ["roof", "roofing", "shingles", "tiles", "gutters", "chimney", "attic"],
    "BLDG_PLUMBING_DMG": ["plumbing", "pipes", "water lines", "faucets", "toilets", "sinks", "drainage"],
    "BLDG_ELECTRICAL_DMG": ["electrical", "wiring", "outlets", "panel", "breaker", "lighting", "power"],
    "BLDG_HVAC_DMG": ["hvac", "heating", "cooling", "air conditioning", "furnace", "ductwork", "vents"],
    "BLDG_FOUNDATION_DMG": ["foundation", "basement", "crawl space", "slab", "concrete", "footings"],
    "BLDG_STRUCTURAL_DMG": ["structural", "load bearing", "beams", "joists", "frame", "support"],
    "BLDG_WINDOWS_DMG": ["windows", "glass", "frames", "sills", "screens", "panes"],
    "BLDG_DOORS_DMG": ["doors", "entry", "frames", "hinges", "locks", "thresholds"],
    "BLDG_FLOORING_DMG": ["flooring", "floors", "carpet", "hardwood", "tile", "laminate", "subfloor"],
    "BLDG_WALLS_DMG": ["walls", "drywall", "insulation", "studs", "paint", "wallpaper"],
    "BLDG_CEILING_DMG": ["ceiling", "drywall", "tiles", "beams", "fixtures", "insulation"],
    "BLDG_FIRE_DMG": ["fire", "smoke", "burn", "charred", "soot", "flame", "combustion"],
    "BLDG_WATER_DMG": ["water", "flood", "leak", "moisture", "wet", "damp", "saturation"]
}

OPERATIONAL_KEYWORDS = {
    "BLDG_TENABLE": ["habitable", "livable", "occupiable", "safe to occupy", "tenant can stay"],
    "BLDG_UNOCCUPIABLE": ["unoccupiable", "uninhabitable", "cannot occupy", "unsafe", "condemned"],
    "BLDG_COMPLETE_LOSS": ["total loss", "complete loss", "destroyed", "demolished", "tear down"]
}

CONTEXTUAL_KEYWORDS = {
    "BLDG_PRIMARY": ["primary", "main", "principal", "primary structure", "main building"],
    "BLDG_ADJACENT_ORIGIN": ["adjacent", "neighboring", "next to", "adjoining property"],
    "BLDG_DIRECT_ORIGIN": ["direct", "originated from", "started at", "source of damage"]
}
```

**BLDG_LOSS_AMOUNT Hierarchical Keywords:**
```python
LOSS_AMOUNT_HIERARCHY = {
    "final_settlement": ["final", "settlement", "agreed", "settled", "closing", "concluded"],
    "adjuster_estimate": ["adjuster", "adjustment", "estimated", "assessed", "appraised"],
    "contractor_quote": ["contractor", "quote", "bid", "estimate", "repair cost"],
    "initial_estimate": ["initial", "preliminary", "first", "rough", "ballpark"]
}
```

### Critical Validation Rules Identified

**Logical Consistency Rules:**
- `BLDG_TENABLE=Y` AND `BLDG_COMPLETE_LOSS=Y` → **INVALID**
- `BLDG_UNOCCUPIABLE=Y` AND `BLDG_TENABLE=Y` → **INVALID**
- Must have text evidence for any `Y` indicator
- Confidence scores must be 0.6-0.95 range

---

## Optimized Architecture: 1+3 Agent Design

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    CENTRALIZED AGENTIC SERVICES                            │
│   🧠 Memory Management | 🛡️ System Guardrails | 📚 Knowledge Base          │
└─────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      STAGE 1: COMPREHENSIVE EXTRACTION                     │
│                           (Single Optimized Agent)                         │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    UNIFIED EXTRACTION AGENT                         │   │
│  │                                                                     │   │
│  │  🔗 PROMPT CHAINING: 4-stage comprehensive extraction              │   │
│  │  🧠 MEMORY: Historical patterns + keyword effectiveness            │   │
│  │  📚 KNOWLEDGE: Complete keyword library + validation rules         │   │
│  │  🤔 REASONING: Text analysis + indicator extraction + candidates    │   │
│  │  🛠️ TOOLS: TextProcessor + validation + temporal analysis          │   │
│  │  📋 PLANNING: Extraction strategy + priority focus                 │   │
│  │  🪞 REFLECTION: Self-validation + consistency checks               │   │
│  │  🛡️ GUARDRAILS: Logical consistency + confidence thresholds       │   │
│  │                                                                     │   │
│  │  OUTPUT: 21 Y/N indicators + monetary candidates + validation      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
                           ┌─────────────────────────┐
                           │   CONDITIONAL TRIGGER   │
                           │   Are candidates found? │
                           └─────────────────────────┘
                                        │
                              ┌─────────┴─────────┐
                             NO                  YES
                              │                   │
                              ▼                   ▼
                    ┌────────────────┐ ┌─────────────────────────────────────────────┐
                    │  FINALIZATION  │ │        STAGE 2: CALCULATION PIPELINE       │
                    └────────────────┘ │            (3 Specialized Agents)          │
                                       └─────────────────────────────────────────────┘
                                                        │
                                     ┌──────────────────┼──────────────────┐
                                     ▼                  ▼                  ▼
                              ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
                              │  AGENT 2A:   │ │  AGENT 2B:   │ │  AGENT 2C:   │
                              │ Context      │ │ Calculation  │ │ Validation   │
                              │ Analysis     │ │ & Priority   │ │ & Reflection │
                              └──────────────┘ └──────────────┘ └──────────────┘
```

---

## Stage 1: Unified Extraction Agent (Complete Implementation)

### Comprehensive Prompt Chain with All Keywords

```python
from coverage_configs.src.configs.prompts import get_all_building_indicator_prompts
from coverage_rag_implementation.src.helpers.gpt_api import GptApi
from coverage_rag_implementation.src.text_processor import TextProcessor
import json
from datetime import datetime

class UnifiedExtractionAgent:
    """Stage 1: Single comprehensive agent with all original keywords preserved"""
    
    def __init__(self):
        # Initialize existing dependencies
        self.gpt_api = GptApi()
        self.text_processor = TextProcessor()
        
        # 💾 MEMORY: Initialize with all original patterns
        self.memory_store = AgenticMemoryStore()
        
        # 📚 KNOWLEDGE: Complete keyword library from original system
        self.keyword_library = self._load_complete_keyword_library()
        
        # 🛠️ TOOLS: All extraction and validation tools
        self.tools = {
            "validator": ComprehensiveValidator(),
            "temporal_analyzer": TemporalAnalyzer(),
            "consistency_checker": LogicalConsistencyChecker()
        }
        
        # 🛡️ GUARDRAILS: Original validation rules
        self.validation_rules = self._load_original_validation_rules()
    
    def _load_complete_keyword_library(self) -> Dict:
        """📚 KNOWLEDGE: Load complete keyword library from original system"""
        
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
        """🛡️ GUARDRAILS: Load original system validation rules"""
        
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
        
        # 📋 PLANNING: Plan comprehensive extraction strategy
        extraction_plan = await self._plan_comprehensive_strategy(claim_data)
        
        # 🔗 PROMPT CHAINING: Execute 4-stage extraction chain
        extraction_results = await self._execute_comprehensive_chain(claim_data, extraction_plan)
        
        # 🪞 REFLECTION: Comprehensive validation with original rules
        validated_results = await self._reflect_and_validate_comprehensive(extraction_results)
        
        # 💾 MEMORY: Store patterns for continuous learning
        self._update_memory_with_comprehensive_results(validated_results)
        
        return validated_results
    
    async def _execute_comprehensive_chain(self, claim_data: Dict, plan: Dict) -> Dict:
        """🔗 PROMPT CHAINING: 4-stage comprehensive extraction preserving all keywords"""
        
        # Preprocess text using original TextProcessor
        processed_text = self.text_processor.remove_duplicate_sentences(claim_data["claim_text"])
        filtered_text = self.text_processor.filter_important_keywords(processed_text)
        
        # Chain 1: Text Analysis & Context Planning
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
        
        Analyze the text and provide extraction strategy recommendations.
        """
        
        context_analysis = await self._execute_single_chain_prompt(context_prompt)
        
        # Chain 2: Complete 21-Indicator Extraction with All Keywords
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
                ...all 15 damage indicators
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
        
        indicators_result = await self._execute_single_chain_prompt(indicators_prompt)
        
        # Chain 3: BLDG_LOSS_AMOUNT Candidate Extraction with Hierarchical Logic
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
        
        candidates_result = await self._execute_single_chain_prompt(candidates_prompt)
        
        # Chain 4: Comprehensive Validation & Reflection
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
        
        Apply all original validation rules and provide final validated results.
        """
        
        validation_result = await self._execute_single_chain_prompt(validation_prompt)
        
        return {
            "context_analysis": context_analysis,
            "indicators_extraction": indicators_result,
            "candidates_extraction": candidates_result,
            "comprehensive_validation": validation_result,
            "chain_execution_timestamp": datetime.now().isoformat()
        }
    
    async def _reflect_and_validate_comprehensive(self, extraction_results: Dict) -> Dict:
        """🪞 REFLECTION + 🛡️ GUARDRAILS: Apply all original validation rules"""
        
        indicators = extraction_results["indicators_extraction"]
        candidates = extraction_results["candidates_extraction"]
        validation = extraction_results["comprehensive_validation"]
        
        # 🛡️ GUARDRAILS: Apply logical consistency rules
        consistency_check = self.tools["consistency_checker"].validate_logical_consistency(indicators)
        
        # 🛠️ TOOLS: Comprehensive validation using original rules
        comprehensive_validation = self.tools["validator"].validate_comprehensive_extraction(
            indicators=indicators,
            candidates=candidates,
            validation_rules=self.validation_rules
        )
        
        # 🪞 REFLECTION: Self-assessment of extraction quality
        quality_reflection = await self._perform_quality_reflection(
            extraction_results, consistency_check, comprehensive_validation
        )
        
        # Determine if extraction meets original system standards
        meets_standards = (
            consistency_check["passes_consistency"] and
            comprehensive_validation["meets_completeness"] and
            comprehensive_validation["confidence_adequate"] and
            quality_reflection["quality_score"] >= 0.7
        )
        
        if not meets_standards:
            # Handle low-quality extraction
            return await self._handle_extraction_quality_issues(
                extraction_results, consistency_check, comprehensive_validation
            )
        
        # Format final results maintaining original system compatibility
        final_results = {
            # 21 Y/N Indicators (flattened for compatibility)
            **self._flatten_indicators_for_compatibility(indicators),
            
            # Monetary candidates for Stage 2
            "BLDG_LOSS_AMOUNT_CANDIDATES": candidates.get("BLDG_LOSS_AMOUNT_CANDIDATES", {"values": []}),
            
            # Validation metadata
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

class ComprehensiveValidator:
    """🛠️ TOOLS: Comprehensive validation using original system rules"""
    
    def validate_comprehensive_extraction(self, indicators: Dict, candidates: Dict, rules: Dict) -> Dict:
        """Apply all original validation rules"""
        
        validation_results = {
            "meets_completeness": self._check_completeness(indicators, rules),
            "confidence_adequate": self._check_confidence_levels(indicators, rules),
            "evidence_sufficient": self._check_evidence_quality(indicators, rules),
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
        
        # Count indicators in each category
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

class LogicalConsistencyChecker:
    """🛠️ TOOLS: Logical consistency validation"""
    
    def validate_logical_consistency(self, indicators: Dict) -> Dict:
        """Apply original logical consistency rules"""
        
        consistency_issues = []
        
        # Get operational indicators
        operational = indicators.get("operational_indicators", {})
        
        # Check incompatible combinations
        tenable_value = operational.get("BLDG_TENABLE", {}).get("value")
        complete_loss_value = operational.get("BLDG_COMPLETE_LOSS", {}).get("value")
        unoccupiable_value = operational.get("BLDG_UNOCCUPIABLE", {}).get("value")
        
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
```

---

## Alternative: 2-Agent Stage 1 (If Required)

If single agent proves too complex in testing, here's the 2-agent alternative:

### Agent 1A: Text Processing & Indicator Extraction
- **Focus**: Process text + extract 21 Y/N indicators
- **Components**: Memory, Knowledge, Tools, Reflection
- **Keywords**: All damage/operational/contextual keywords

### Agent 1B: Monetary Candidate Specialist  
- **Focus**: Complex BLDG_LOSS_AMOUNT candidate extraction
- **Components**: Reasoning, Tools, Memory, Planning
- **Keywords**: All hierarchical loss amount keywords

---

## Benefits of Single Stage 1 Agent

1. **Simplicity**: No inter-agent communication overhead
2. **Efficiency**: Single comprehensive extraction pass
3. **Consistency**: All keywords applied in unified context
4. **Debugging**: Easier to trace extraction issues
5. **Performance**: Faster processing with one agent

## Validation Against Original System

✅ **All 22 indicators preserved** (21 + BLDG_LOSS_AMOUNT candidates)  
✅ **Complete keyword library integrated** from original prompts.py  
✅ **Logical consistency rules maintained** (tenable/complete_loss conflicts)  
✅ **Hierarchical BLDG_LOSS_AMOUNT logic preserved**  
✅ **Original TextProcessor integration** maintained  
✅ **Confidence scoring (0.6-0.95) preserved**  
✅ **Evidence requirements for Y indicators** enforced  
✅ **File notes prioritization** maintained  
✅ **Memory pattern learning** preserved  

The unified agent design maintains all original system functionality while significantly reducing architectural complexity through careful keyword preservation and comprehensive prompt chaining.

<function_calls>
<invoke name="TodoWrite">
<parameter name="todos">[{"id": "1", "content": "Review original codebase prompts and keywords from Core_Functionalities.md", "status": "completed", "priority": "high"}, {"id": "2", "content": "Analyze if Stage 1 can be reduced to 1-2 agents", "status": "completed", "priority": "high"}, {"id": "3", "content": "Ensure all 21 feature keywords and extraction logic are preserved", "status": "completed", "priority": "high"}, {"id": "4", "content": "Redesign Stage 1 with reduced agents", "status": "completed", "priority": "high"}, {"id": "5", "content": "Validate no important extraction patterns are missed", "status": "completed", "priority": "medium"}]