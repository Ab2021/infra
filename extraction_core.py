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
        
        # Keyword mappings for each indicator
        self.indicator_keywords = {
            "BLDG_FIRE_DMG": ["fire", "burn", "smoke", "heat", "flame", "combustion", "ignition"],
            "BLDG_WATER_DMG": ["water", "flood", "leak", "burst", "pipe", "sprinkler", "moisture"],
            "BLDG_WIND_DMG": ["wind", "hurricane", "tornado", "gust", "storm", "blown"],
            "BLDG_HAIL_DMG": ["hail", "hailstone", "ice", "pellet"],
            "BLDG_LIGHTNING_DMG": ["lightning", "strike", "bolt", "electrical storm"],
            "BLDG_VANDALISM_DMG": ["vandalism", "graffiti", "malicious", "intentional"],
            "BLDG_THEFT_DMG": ["theft", "burglary", "break-in", "stolen", "missing"],
            "BLDG_ROOF_DMG": ["roof", "shingle", "tile", "membrane", "gutter", "eave"],
            "BLDG_WALLS_DMG": ["wall", "siding", "exterior", "interior wall", "drywall"],
            "BLDG_FLOORING_DMG": ["floor", "carpet", "hardwood", "tile", "flooring", "subflooring"],
            "BLDG_CEILING_DMG": ["ceiling", "overhead", "drop ceiling", "suspended"],
            "BLDG_WINDOWS_DMG": ["window", "glass", "pane", "frame", "sash"],
            "BLDG_DOORS_DMG": ["door", "entrance", "exit", "doorway", "portal"],
            "BLDG_ELECTRICAL_DMG": ["electrical", "wiring", "outlet", "panel", "circuit"],
            "BLDG_PLUMBING_DMG": ["plumbing", "pipe", "fixture", "faucet", "toilet"],
            "BLDG_INTERIOR_DMG": ["interior", "inside", "internal", "room", "space"],
            "BLDG_TENABLE": ["habitable", "livable", "occupiable", "tenable", "usable"],
            "BLDG_PRIMARY_STRUCTURE": ["main", "primary", "principal", "original", "main building"],
            "BLDG_OCCUPANCY_TYPE": ["commercial", "residential", "office", "retail", "warehouse"],
            "BLDG_SQUARE_FOOTAGE": ["square feet", "sq ft", "sqft", "size", "area"],
            "BLDG_YEAR_BUILT": ["built", "constructed", "year", "age", "old"],
            "BLDG_CONSTRUCTION_TYPE": ["construction", "frame", "masonry", "steel", "concrete"]
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
        """Extract damage indicators (Stage 1)"""
        
        prompt = f"""
        You are an expert insurance adjuster analyzing building damage claims.
        
        CLAIM TEXT: {claim_text}
        
        Extract the following 15 damage indicators. For each indicator, provide:
        - value: "Y" if damage is present, "N" if not mentioned/absent
        - confidence: 0.0-1.0 confidence score
        - evidence: specific text snippet supporting the decision
        
        DAMAGE INDICATORS:
        1. BLDG_FIRE_DMG - Fire, burn, smoke, or heat damage
        2. BLDG_WATER_DMG - Water damage from any source
        3. BLDG_WIND_DMG - Wind or storm damage
        4. BLDG_HAIL_DMG - Hail damage
        5. BLDG_LIGHTNING_DMG - Lightning strike damage
        6. BLDG_VANDALISM_DMG - Vandalism or malicious damage
        7. BLDG_THEFT_DMG - Theft or burglary damage
        8. BLDG_ROOF_DMG - Roof structural damage
        9. BLDG_WALLS_DMG - Wall damage (interior or exterior)
        10. BLDG_FLOORING_DMG - Floor or flooring damage
        11. BLDG_CEILING_DMG - Ceiling damage
        12. BLDG_WINDOWS_DMG - Window damage
        13. BLDG_DOORS_DMG - Door damage
        14. BLDG_ELECTRICAL_DMG - Electrical system damage
        15. BLDG_PLUMBING_DMG - Plumbing system damage
        
        Return JSON format only.
        """
        
        try:
            response = await self.gpt_wrapper.generate_json_content_async(prompt)
            return self._parse_extraction_response(response, self.damage_indicators)
        except Exception as e:
            print(f"Error extracting damage indicators: {e}")
            return self._create_fallback_results(self.damage_indicators)
    
    async def _extract_operational_indicators(self, claim_text: str) -> Dict[str, Any]:
        """Extract operational indicators (Stage 2)"""
        
        prompt = f"""
        You are an expert insurance adjuster analyzing building operational status.
        
        CLAIM TEXT: {claim_text}
        
        Extract the following 3 operational indicators:
        
        1. BLDG_INTERIOR_DMG - Interior space damage affecting usability
           - "Y" if interior damage is mentioned, "N" otherwise
        
        2. BLDG_TENABLE - Building habitability/usability
           - "Y" if building is habitable/usable, "N" if not habitable/usable
           - Look for mentions of evacuation, uninhabitable, unusable, closed
        
        3. BLDG_PRIMARY_STRUCTURE - Is this the main/primary structure
           - "Y" if main building, "N" if secondary structure/outbuilding
        
        For each indicator provide:
        - value: "Y" or "N"
        - confidence: 0.0-1.0 confidence score  
        - evidence: supporting text snippet
        
        Return JSON format only.
        """
        
        try:
            response = await self.gpt_wrapper.generate_json_content_async(prompt)
            return self._parse_extraction_response(response, self.operational_indicators)
        except Exception as e:
            print(f"Error extracting operational indicators: {e}")
            return self._create_fallback_results(self.operational_indicators)
    
    async def _extract_contextual_indicators(self, claim_text: str) -> Dict[str, Any]:
        """Extract contextual indicators (Stage 3)"""
        
        prompt = f"""
        You are an expert insurance adjuster analyzing building context and characteristics.
        
        CLAIM TEXT: {claim_text}
        
        Extract the following 4 contextual indicators:
        
        1. BLDG_OCCUPANCY_TYPE - Building use type
           - Extract: "commercial", "residential", "mixed", "industrial", etc.
           - If not mentioned, use "unknown"
        
        2. BLDG_SQUARE_FOOTAGE - Building size
           - Extract numeric value if mentioned (e.g., "2000" for 2000 sq ft)
           - If not mentioned, use "unknown"
        
        3. BLDG_YEAR_BUILT - Construction year
           - Extract 4-digit year if mentioned
           - If not mentioned, use "unknown"
        
        4. BLDG_CONSTRUCTION_TYPE - Construction material/method
           - Extract: "frame", "masonry", "steel", "concrete", etc.
           - If not mentioned, use "unknown"
        
        For each indicator provide:
        - value: extracted value or "unknown"
        - confidence: 0.0-1.0 confidence score
        - evidence: supporting text snippet
        
        Return JSON format only.
        """
        
        try:
            response = await self.gpt_wrapper.generate_json_content_async(prompt)
            return self._parse_extraction_response(response, self.contextual_indicators)
        except Exception as e:
            print(f"Error extracting contextual indicators: {e}")
            return self._create_fallback_results(self.contextual_indicators)
    
    async def _validate_and_consolidate(self, claim_text: str, stage1: Dict, 
                                      stage2: Dict, stage3: Dict) -> Dict[str, Any]:
        """Validate and consolidate all extraction results (Stage 4)"""
        
        # Combine all results
        consolidated = {}
        consolidated.update(stage1)
        consolidated.update(stage2)
        consolidated.update(stage3)
        
        # Apply logical consistency rules
        consolidated = self._apply_consistency_rules(consolidated)
        
        # Validate confidence thresholds
        consolidated = self._validate_confidence_thresholds(consolidated)
        
        return consolidated
    
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
    
    def _apply_consistency_rules(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Apply logical consistency rules to extraction results"""
        
        # Rule 1: If major damage types are present, interior damage is likely
        major_damage_types = ["BLDG_FIRE_DMG", "BLDG_WATER_DMG", "BLDG_WIND_DMG"]
        has_major_damage = any(
            results.get(damage, {}).get("value") == "Y" for damage in major_damage_types
        )
        
        if has_major_damage and results.get("BLDG_INTERIOR_DMG", {}).get("value") == "N":
            if results.get("BLDG_INTERIOR_DMG", {}).get("confidence", 0) < 0.8:
                results["BLDG_INTERIOR_DMG"]["value"] = "Y"
                results["BLDG_INTERIOR_DMG"]["confidence"] = 0.7
                results["BLDG_INTERIOR_DMG"]["evidence"] += " (inferred from major damage)"
        
        # Rule 2: If extensive damage, likely not tenable
        damage_count = sum(
            1 for indicator in self.damage_indicators 
            if results.get(indicator, {}).get("value") == "Y"
        )
        
        if damage_count >= 5 and results.get("BLDG_TENABLE", {}).get("value") == "Y":
            if results.get("BLDG_TENABLE", {}).get("confidence", 0) < 0.8:
                results["BLDG_TENABLE"]["value"] = "N"
                results["BLDG_TENABLE"]["confidence"] = 0.7
                results["BLDG_TENABLE"]["evidence"] += " (inferred from extensive damage)"
        
        return results
    
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