"""
Extraction Core for Agentic Building Coverage Analysis
Key RAGProcessor extraction logic for 22 building indicators
"""

import asyncio
import json
import re
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime


class ExtractionCore:
    """Core extraction logic for building coverage indicators"""
    
    def __init__(self, gpt_wrapper):
        self.gpt_wrapper = gpt_wrapper
        
        # 22 Building indicators for extraction
        self.building_indicators = [
            "BLDG_FIRE_DMG", "BLDG_WATER_DMG", "BLDG_WIND_DMG", "BLDG_HAIL_DMG",
            "BLDG_LIGHTNING_DMG", "BLDG_VANDALISM_DMG", "BLDG_THEFT_DMG", 
            "BLDG_ROOF_DMG", "BLDG_WALLS_DMG", "BLDG_FLOORING_DMG",
            "BLDG_CEILING_DMG", "BLDG_WINDOWS_DMG", "BLDG_DOORS_DMG",
            "BLDG_ELECTRICAL_DMG", "BLDG_PLUMBING_DMG", "BLDG_INTERIOR_DMG",
            "BLDG_TENABLE", "BLDG_PRIMARY_STRUCTURE", "BLDG_OCCUPANCY_TYPE",
            "BLDG_SQUARE_FOOTAGE", "BLDG_YEAR_BUILT", "BLDG_CONSTRUCTION_TYPE"
        ]
        
        # Damage indicators (15)
        self.damage_indicators = [
            "BLDG_FIRE_DMG", "BLDG_WATER_DMG", "BLDG_WIND_DMG", "BLDG_HAIL_DMG",
            "BLDG_LIGHTNING_DMG", "BLDG_VANDALISM_DMG", "BLDG_THEFT_DMG",
            "BLDG_ROOF_DMG", "BLDG_WALLS_DMG", "BLDG_FLOORING_DMG",
            "BLDG_CEILING_DMG", "BLDG_WINDOWS_DMG", "BLDG_DOORS_DMG",
            "BLDG_ELECTRICAL_DMG", "BLDG_PLUMBING_DMG"
        ]
        
        # Operational indicators (3)
        self.operational_indicators = [
            "BLDG_INTERIOR_DMG", "BLDG_TENABLE", "BLDG_PRIMARY_STRUCTURE"
        ]
        
        # Contextual indicators (4) 
        self.contextual_indicators = [
            "BLDG_OCCUPANCY_TYPE", "BLDG_SQUARE_FOOTAGE", "BLDG_YEAR_BUILT", 
            "BLDG_CONSTRUCTION_TYPE"
        ]
        
        # Complete keyword library from original codebase
        self.indicator_keywords = {
            "BLDG_FIRE_DMG": ["fire", "burn", "burning", "burned", "burnt", "smoke", "heat", "flame", "flames", "combustion", "ignition", "ignited", "blaze", "inferno", "scorch", "char", "charred", "soot", "ash", "smolder"],
            "BLDG_WATER_DMG": ["water", "flood", "flooding", "flooded", "leak", "leaking", "leaked", "burst", "pipe", "pipes", "sprinkler", "moisture", "wet", "damp", "soggy", "saturated", "drip", "dripping", "seepage", "overflow"],
            "BLDG_WIND_DMG": ["wind", "hurricane", "tornado", "gust", "gusts", "storm", "blown", "blew", "windstorm", "typhoon", "cyclone", "debris", "uplift", "pressure"],
            "BLDG_HAIL_DMG": ["hail", "hailstone", "hailstones", "ice", "pellet", "pellets", "hailstorm", "dent", "dents", "dented"],
            "BLDG_LIGHTNING_DMG": ["lightning", "strike", "struck", "bolt", "electrical storm", "thunder", "surge", "power surge"],
            "BLDG_VANDALISM_DMG": ["vandalism", "vandal", "graffiti", "malicious", "intentional", "deliberate", "mischief", "defaced", "destroyed", "damaged intentionally"],
            "BLDG_THEFT_DMG": ["theft", "burglary", "break-in", "stolen", "missing", "burglar", "robbed", "robbery", "larceny", "pilfered"],
            "BLDG_ROOF_DMG": ["roof", "roofing", "shingle", "shingles", "tile", "tiles", "membrane", "gutter", "gutters", "eave", "eaves", "ridge", "peak", "soffit", "fascia", "flashing", "chimney", "skylight"],
            "BLDG_WALLS_DMG": ["wall", "walls", "siding", "exterior wall", "interior wall", "drywall", "sheetrock", "plaster", "stud", "studs", "framing", "brick", "masonry", "stucco", "panel", "paneling"],
            "BLDG_FLOORING_DMG": ["floor", "floors", "flooring", "carpet", "carpeting", "hardwood", "tile", "tiles", "vinyl", "linoleum", "laminate", "subflooring", "subfloor", "concrete slab", "basement floor"],
            "BLDG_CEILING_DMG": ["ceiling", "ceilings", "overhead", "drop ceiling", "suspended ceiling", "ceiling tile", "ceiling tiles", "acoustic tile", "plaster ceiling", "drywall ceiling"],
            "BLDG_WINDOWS_DMG": ["window", "windows", "glass", "pane", "panes", "frame", "frames", "sash", "sill", "glazing", "broken glass", "shattered", "cracked glass"],
            "BLDG_DOORS_DMG": ["door", "doors", "entrance", "exit", "doorway", "doorways", "portal", "entry", "doorframe", "door frame", "threshold", "jamb"],
            "BLDG_ELECTRICAL_DMG": ["electrical", "electric", "wiring", "wire", "wires", "outlet", "outlets", "panel", "panels", "circuit", "circuits", "breaker", "fuse", "switch", "switches", "lighting", "fixture", "fixtures"],
            "BLDG_PLUMBING_DMG": ["plumbing", "pipe", "pipes", "plumbing system", "fixture", "fixtures", "faucet", "faucets", "toilet", "toilets", "sink", "sinks", "drain", "drains", "water line", "supply line"],
            "BLDG_INTERIOR_DMG": ["interior", "inside", "internal", "room", "rooms", "space", "spaces", "indoor", "indoors", "living area", "office space", "hallway", "corridor"],
            "BLDG_TENABLE": ["habitable", "livable", "occupiable", "tenable", "usable", "inhabitable", "residential", "safe", "suitable", "fit for occupancy"],
            "BLDG_PRIMARY_STRUCTURE": ["main", "primary", "principal", "original", "main building", "primary structure", "principal building", "primary residence", "main structure"],
            "BLDG_OCCUPANCY_TYPE": ["commercial", "residential", "office", "retail", "warehouse", "industrial", "manufacturing", "apartment", "condominium", "single family", "multi-family"],
            "BLDG_SQUARE_FOOTAGE": ["square feet", "sq ft", "sqft", "square foot", "size", "area", "footage", "dimensions", "floor area"],
            "BLDG_YEAR_BUILT": ["built", "constructed", "construction", "year", "age", "old", "vintage", "erected", "completed"],
            "BLDG_CONSTRUCTION_TYPE": ["construction", "frame", "masonry", "steel", "concrete", "brick", "wood frame", "steel frame", "concrete block", "stone"]
        }
    
    async def extract_all_indicators(self, claim_text: str, claim_id: str = "") -> Dict[str, Any]:
        """Extract all 22 building indicators from claim text"""
        
        # Stage 1: Basic extraction
        stage1_results = await self._extract_damage_indicators(claim_text)
        
        # Stage 2: Operational assessment  
        stage2_results = await self._extract_operational_indicators(claim_text)
        
        # Stage 3: Contextual extraction
        stage3_results = await self._extract_contextual_indicators(claim_text)
        
        # Stage 4: Validation and consolidation
        final_results = await self._validate_and_consolidate(
            claim_text, stage1_results, stage2_results, stage3_results
        )
        
        # Add metadata
        final_results.update({
            "claim_id": claim_id,
            "source_text": claim_text,
            "extraction_timestamp": datetime.now().isoformat(),
            "total_indicators_found": len([k for k, v in final_results.items() 
                                         if k.startswith("BLDG_") and isinstance(v, dict)])
        })
        
        return final_results
    
    async def _extract_damage_indicators(self, claim_text: str) -> Dict[str, Any]:
        """Extract damage indicators (Stage 1) with complete keyword guidance"""
        
        prompt = f"""
        BUILDING COVERAGE ANALYSIS - DAMAGE INDICATORS EXTRACTION
        
        CLAIM TEXT: {claim_text}
        
        Extract the following 15 damage indicators using the provided keyword guidance. For each indicator:
        - value: "Y" if damage is present, "N" if not mentioned/absent  
        - confidence: 0.6-0.95 confidence score based on evidence strength
        - evidence: specific text snippet supporting the decision
        
        DAMAGE INDICATORS WITH KEYWORD GUIDANCE:
        
        1. BLDG_FIRE_DMG - Look for: {', '.join(self.indicator_keywords["BLDG_FIRE_DMG"])}
        2. BLDG_WATER_DMG - Look for: {', '.join(self.indicator_keywords["BLDG_WATER_DMG"])}
        3. BLDG_WIND_DMG - Look for: {', '.join(self.indicator_keywords["BLDG_WIND_DMG"])}
        4. BLDG_HAIL_DMG - Look for: {', '.join(self.indicator_keywords["BLDG_HAIL_DMG"])}
        5. BLDG_LIGHTNING_DMG - Look for: {', '.join(self.indicator_keywords["BLDG_LIGHTNING_DMG"])}
        6. BLDG_VANDALISM_DMG - Look for: {', '.join(self.indicator_keywords["BLDG_VANDALISM_DMG"])}
        7. BLDG_THEFT_DMG - Look for: {', '.join(self.indicator_keywords["BLDG_THEFT_DMG"])}
        8. BLDG_ROOF_DMG - Look for: {', '.join(self.indicator_keywords["BLDG_ROOF_DMG"])}
        9. BLDG_WALLS_DMG - Look for: {', '.join(self.indicator_keywords["BLDG_WALLS_DMG"])}
        10. BLDG_FLOORING_DMG - Look for: {', '.join(self.indicator_keywords["BLDG_FLOORING_DMG"])}
        11. BLDG_CEILING_DMG - Look for: {', '.join(self.indicator_keywords["BLDG_CEILING_DMG"])}
        12. BLDG_WINDOWS_DMG - Look for: {', '.join(self.indicator_keywords["BLDG_WINDOWS_DMG"])}
        13. BLDG_DOORS_DMG - Look for: {', '.join(self.indicator_keywords["BLDG_DOORS_DMG"])}
        14. BLDG_ELECTRICAL_DMG - Look for: {', '.join(self.indicator_keywords["BLDG_ELECTRICAL_DMG"])}
        15. BLDG_PLUMBING_DMG - Look for: {', '.join(self.indicator_keywords["BLDG_PLUMBING_DMG"])}
        
        EXTRACTION RULES:
        - Use ONLY Y/N format for each indicator
        - Provide specific text evidence for each Y answer
        - Use "INSUFFICIENT_INFO" if evidence is unclear
        - Include confidence score (0.6-0.95) for each indicator
        - Match keywords exactly as provided above
        
        Return JSON format only with all 15 indicators.
        """
        
        try:
            response = await self.gpt_wrapper.generate_json_content_async(prompt)
            return self._parse_extraction_response(response, self.damage_indicators)
        except Exception as e:
            print(f"Error extracting damage indicators: {e}")
            return self._create_fallback_results(self.damage_indicators)
    
    async def _extract_operational_indicators(self, claim_text: str) -> Dict[str, Any]:
        """Extract operational indicators (Stage 2) with keyword guidance"""
        
        prompt = f"""
        BUILDING COVERAGE ANALYSIS - OPERATIONAL INDICATORS EXTRACTION
        
        CLAIM TEXT: {claim_text}
        
        Extract the following 3 operational indicators using keyword guidance:
        
        **OPERATIONAL INDICATORS (Y/N only):**
        
        1. BLDG_INTERIOR_DMG - Look for: {', '.join(self.indicator_keywords["BLDG_INTERIOR_DMG"])}
           - "Y" if interior damage is mentioned, "N" otherwise
        
        2. BLDG_TENABLE - Look for: {', '.join(self.indicator_keywords["BLDG_TENABLE"])}
           - "Y" if building is habitable/usable, "N" if not habitable/usable
           - Look for mentions of evacuation, uninhabitable, unusable, closed
        
        3. BLDG_PRIMARY_STRUCTURE - Look for: {', '.join(self.indicator_keywords["BLDG_PRIMARY_STRUCTURE"])}
           - "Y" if main building, "N" if secondary structure/outbuilding
        
        EXTRACTION RULES:
        - Use ONLY Y/N format for each indicator
        - Provide specific text evidence for each answer
        - Include confidence score (0.6-0.95) for each indicator
        - Match keywords exactly as provided above
        
        For each indicator provide:
        - value: "Y" or "N"
        - confidence: 0.6-0.95 confidence score  
        - evidence: supporting text snippet
        
        Return JSON format only with all 3 operational indicators.
        """
        
        try:
            response = await self.gpt_wrapper.generate_json_content_async(prompt)
            return self._parse_extraction_response(response, self.operational_indicators)
        except Exception as e:
            print(f"Error extracting operational indicators: {e}")
            return self._create_fallback_results(self.operational_indicators)
    
    async def _extract_contextual_indicators(self, claim_text: str) -> Dict[str, Any]:
        """Extract contextual indicators (Stage 3) with keyword guidance"""
        
        prompt = f"""
        BUILDING COVERAGE ANALYSIS - CONTEXTUAL INDICATORS EXTRACTION
        
        CLAIM TEXT: {claim_text}
        
        Extract the following 4 contextual indicators using keyword guidance:
        
        **CONTEXTUAL INDICATORS:**
        
        1. BLDG_OCCUPANCY_TYPE - Look for: {', '.join(self.indicator_keywords["BLDG_OCCUPANCY_TYPE"])}
           - Extract: "commercial", "residential", "mixed", "industrial", etc.
           - If not mentioned, use "unknown"
        
        2. BLDG_SQUARE_FOOTAGE - Look for: {', '.join(self.indicator_keywords["BLDG_SQUARE_FOOTAGE"])}
           - Extract numeric value if mentioned (e.g., "2000" for 2000 sq ft)
           - If not mentioned, use "unknown"
        
        3. BLDG_YEAR_BUILT - Look for: {', '.join(self.indicator_keywords["BLDG_YEAR_BUILT"])}
           - Extract 4-digit year if mentioned
           - If not mentioned, use "unknown"
        
        4. BLDG_CONSTRUCTION_TYPE - Look for: {', '.join(self.indicator_keywords["BLDG_CONSTRUCTION_TYPE"])}
           - Extract: "frame", "masonry", "steel", "concrete", etc.
           - If not mentioned, use "unknown"
        
        EXTRACTION RULES:
        - Match keywords exactly as provided above
        - Provide specific text evidence for each extraction
        - Include confidence score (0.6-0.95) for each indicator
        - Use "unknown" only when no relevant information is found
        
        For each indicator provide:
        - value: extracted value or "unknown"
        - confidence: 0.6-0.95 confidence score
        - evidence: supporting text snippet
        
        Return JSON format only with all 4 contextual indicators.
        """
        
        try:
            response = await self.gpt_wrapper.generate_json_content_async(prompt)
            return self._parse_extraction_response(response, self.contextual_indicators)
        except Exception as e:
            print(f"Error extracting contextual indicators: {e}")
            return self._create_fallback_results(self.contextual_indicators)
    
    async def _validate_and_consolidate(self, claim_text: str, stage1: Dict, 
                                      stage2: Dict, stage3: Dict) -> Dict[str, Any]:
        """Validate and consolidate all extraction results (Stage 4) with comprehensive validation"""
        
        # Combine all results
        consolidated = {}
        consolidated.update(stage1)
        consolidated.update(stage2)
        consolidated.update(stage3)
        
        # Apply comprehensive validation with reflection
        validated_results = await self._comprehensive_validation_with_reflection(
            claim_text, consolidated
        )
        
        return validated_results
    
    async def _comprehensive_validation_with_reflection(self, claim_text: str, results: Dict) -> Dict[str, Any]:
        """Comprehensive validation with reflection questions from original codebase"""
        
        validation_prompt = f"""
        BUILDING COVERAGE ANALYSIS - COMPREHENSIVE VALIDATION & REFLECTION
        
        Original Claim Text:
        {claim_text}
        
        Extracted Results to Validate:
        {json.dumps(results, indent=2)}
        
        COMPREHENSIVE VALIDATION CHECKLIST:
        
        **LOGICAL CONSISTENCY CHECKS:**
        ✓ BLDG_TENABLE=N AND extensive damage indicators (VALID combination)
        ✓ BLDG_TENABLE=Y AND BLDG_COMPLETE_LOSS=Y (INVALID combination)
        ✓ Multiple damage types should increase interior damage likelihood
        ✓ All Y indicators have supporting text evidence
        ✓ Confidence scores reflect evidence quality (0.6-0.95 range)
        
        **COMPLETENESS VERIFICATION:**
        ✓ All 22 indicators processed (15 damage + 3 operational + 4 contextual)
        ✓ No indicators marked as "INSUFFICIENT_INFO" without justification
        ✓ Evidence text directly supports each Y/N decision
        ✓ Confidence scores are appropriate for evidence strength
        
        **QUALITY REFLECTION QUESTIONS:**
        1. Are all indicators logically consistent with each other?
        2. Do confidence scores accurately reflect evidence quality?  
        3. Are there any contradictory indicators that need resolution?
        4. Is the extraction quality suitable for financial analysis?
        5. Are contextual indicators properly extracted from available information?
        
        **VALIDATION RULES:**
        - No indicator should have confidence below 0.6
        - Evidence must directly support the Y/N decision
        - Logical inconsistencies must be resolved
        - "Unknown" values require justification for contextual indicators
        
        Apply all validation rules, resolve any inconsistencies, and provide final validated results.
        Include a validation_summary with:
        - total_indicators_validated: count
        - logical_consistency_passed: boolean
        - average_confidence_score: float
        - quality_assessment: "excellent"/"good"/"needs_improvement"
        - validation_notes: list of any issues found and resolved
        
        Return JSON format with validated results and validation_summary.
        """
        
        try:
            response = await self.gpt_wrapper.generate_json_content_async(validation_prompt)
            
            # Parse and apply validation results
            if isinstance(response, str):
                validation_data = json.loads(response)
            else:
                validation_data = response
            
            # Apply logical consistency rules from original codebase
            final_results = self._apply_original_consistency_rules(validation_data)
            
            # Validate confidence thresholds
            final_results = self._validate_confidence_thresholds(final_results)
            
            return final_results
            
        except Exception as e:
            print(f"Comprehensive validation failed: {e}")
            # Apply basic consistency rules as fallback
            consolidated = self._apply_consistency_rules(results)
            return self._validate_confidence_thresholds(consolidated)
    
    def _parse_extraction_response(self, response: str, expected_indicators: List[str]) -> Dict[str, Any]:
        """Parse GPT response into structured format"""
        try:
            if isinstance(response, str):
                data = json.loads(response)
            else:
                data = response
            
            parsed_results = {}
            
            for indicator in expected_indicators:
                if indicator in data:
                    result = data[indicator]
                    if isinstance(result, dict):
                        parsed_results[indicator] = {
                            "value": result.get("value", "N"),
                            "confidence": float(result.get("confidence", 0.5)),
                            "evidence": result.get("evidence", "")
                        }
                    else:
                        # Handle simple value format
                        parsed_results[indicator] = {
                            "value": str(result),
                            "confidence": 0.7,
                            "evidence": f"Extracted value: {result}"
                        }
            
            return parsed_results
            
        except Exception as e:
            print(f"Error parsing extraction response: {e}")
            return self._create_fallback_results(expected_indicators)
    
    def _create_fallback_results(self, indicators: List[str]) -> Dict[str, Any]:
        """Create fallback results when extraction fails"""
        fallback_results = {}
        
        for indicator in indicators:
            fallback_results[indicator] = {
                "value": "N" if indicator in self.damage_indicators else "unknown",
                "confidence": 0.3,
                "evidence": "Extraction failed - fallback result"
            }
        
        return fallback_results
    
    def _apply_original_consistency_rules(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Apply logical consistency rules from original codebase"""
        
        # Extract main results from validation response
        if "validation_summary" in results:
            main_results = {k: v for k, v in results.items() if k != "validation_summary"}
        else:
            main_results = results
        
        # Rule 1: BLDG_TENABLE=Y AND extensive damage is inconsistent
        damage_count = sum(
            1 for indicator in self.damage_indicators 
            if main_results.get(indicator, {}).get("value") == "Y"
        )
        
        if damage_count >= 5 and main_results.get("BLDG_TENABLE", {}).get("value") == "Y":
            if main_results.get("BLDG_TENABLE", {}).get("confidence", 0) < 0.8:
                main_results["BLDG_TENABLE"]["value"] = "N"
                main_results["BLDG_TENABLE"]["confidence"] = 0.7
                main_results["BLDG_TENABLE"]["evidence"] += " (consistency rule: extensive damage implies not tenable)"
        
        # Rule 2: Major damage types imply interior damage
        major_damage_types = ["BLDG_FIRE_DMG", "BLDG_WATER_DMG", "BLDG_WIND_DMG"]
        has_major_damage = any(
            main_results.get(damage, {}).get("value") == "Y" for damage in major_damage_types
        )
        
        if has_major_damage and main_results.get("BLDG_INTERIOR_DMG", {}).get("value") == "N":
            if main_results.get("BLDG_INTERIOR_DMG", {}).get("confidence", 0) < 0.8:
                main_results["BLDG_INTERIOR_DMG"]["value"] = "Y"
                main_results["BLDG_INTERIOR_DMG"]["confidence"] = 0.7
                main_results["BLDG_INTERIOR_DMG"]["evidence"] += " (consistency rule: major damage implies interior impact)"
        
        # Rule 3: Fire damage typically affects multiple building systems
        if main_results.get("BLDG_FIRE_DMG", {}).get("value") == "Y":
            fire_related_indicators = ["BLDG_ELECTRICAL_DMG", "BLDG_INTERIOR_DMG", "BLDG_CEILING_DMG"]
            for indicator in fire_related_indicators:
                if main_results.get(indicator, {}).get("value") == "N":
                    if main_results.get(indicator, {}).get("confidence", 0) < 0.7:
                        main_results[indicator]["value"] = "Y"
                        main_results[indicator]["confidence"] = 0.65
                        main_results[indicator]["evidence"] += " (consistency rule: fire typically damages multiple systems)"
        
        # Rule 4: Water damage typically affects flooring and walls
        if main_results.get("BLDG_WATER_DMG", {}).get("value") == "Y":
            water_related_indicators = ["BLDG_FLOORING_DMG", "BLDG_WALLS_DMG"]
            for indicator in water_related_indicators:
                if main_results.get(indicator, {}).get("value") == "N":
                    if main_results.get(indicator, {}).get("confidence", 0) < 0.7:
                        main_results[indicator]["value"] = "Y"
                        main_results[indicator]["confidence"] = 0.65
                        main_results[indicator]["evidence"] += " (consistency rule: water damage typically affects floors/walls)"
        
        return main_results
    
    def _apply_consistency_rules(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Apply basic consistency rules (fallback method)"""
        return self._apply_original_consistency_rules(results)
    
    def _validate_confidence_thresholds(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and adjust confidence scores"""
        
        for indicator, result in results.items():
            if isinstance(result, dict) and "confidence" in result:
                # Ensure confidence is within valid range
                confidence = result["confidence"]
                if confidence < 0.0:
                    result["confidence"] = 0.0
                elif confidence > 1.0:
                    result["confidence"] = 1.0
                
                # Reduce confidence for fallback results
                if "fallback" in result.get("evidence", "").lower():
                    result["confidence"] = min(result["confidence"], 0.4)
        
        return results
    
    def get_extraction_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary of extraction results"""
        
        damage_found = []
        operational_issues = []
        contextual_info = {}
        
        for indicator, result in results.items():
            if not isinstance(result, dict):
                continue
                
            value = result.get("value", "N")
            confidence = result.get("confidence", 0)
            
            if indicator in self.damage_indicators and value == "Y" and confidence >= 0.6:
                damage_found.append(indicator)
            elif indicator in self.operational_indicators and value == "Y":
                operational_issues.append(indicator)
            elif indicator in self.contextual_indicators and value != "unknown":
                contextual_info[indicator] = value
        
        return {
            "damage_types_found": len(damage_found),
            "damage_indicators": damage_found,
            "operational_issues": operational_issues,
            "contextual_information": contextual_info,
            "high_confidence_results": len([
                r for r in results.values() 
                if isinstance(r, dict) and r.get("confidence", 0) >= 0.8
            ]),
            "extraction_completeness": len([
                r for r in results.values() 
                if isinstance(r, dict) and r.get("value") not in ["N", "unknown"]
            ]) / 22.0
        }