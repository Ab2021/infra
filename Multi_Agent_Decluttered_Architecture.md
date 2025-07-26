# Multi-Agent Decluttered Architecture - Research & Design

## Executive Summary

Based on comprehensive research, the current two-stage architecture can be optimally decomposed into **6 specialized agents** (3 per stage) to reduce clutter while maintaining full agentic intelligence. This decomposition distributes the 9 agentic components strategically across specialized agents, improving maintainability, debugging, and performance while preserving the core functionality.

---

## Research Findings & Rationale

### Current Architecture Analysis

**Current State:**
- **Stage 1**: Single agent handling all 21 indicators + candidates extraction (9 agentic components)
- **Stage 2**: Single agent handling BLDG_LOSS_AMOUNT calculation (9 agentic components)

**Identified Issues:**
- Each agent is managing too many responsibilities
- Complex debugging due to mixed concerns
- Difficult to optimize individual components
- Hard to isolate failures or performance bottlenecks

### Decomposition Benefits

1. **Reduced Cognitive Load**: Each agent handles 3-4 agentic components vs. all 9
2. **Specialized Expertise**: Focused domain knowledge per agent
3. **Improved Debugging**: Isolated testing and troubleshooting
4. **Enhanced Scalability**: Independent optimization opportunities
5. **Better Maintainability**: Modular updates and improvements

---

## Recommended Architecture: 6-Agent Decomposition

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    CENTRALIZED AGENTIC SERVICES                            │
│   🧠 Memory Management | 🛡️ System Guardrails | 📚 Knowledge Base          │
└─────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          STAGE 1: EXTRACTION PIPELINE                      │
│                              (3 Specialized Agents)                        │
└─────────────────────────────────────────────────────────────────────────────┘
                    │                    │                    │
                    ▼                    ▼                    ▼
        ┌──────────────────┐ ┌──────────────────┐ ┌──────────────────┐
        │   AGENT 1A:      │ │   AGENT 1B:      │ │   AGENT 1C:      │
        │ Text Analysis &  │ │ Feature          │ │ Candidate        │
        │ Context          │ │ Extraction       │ │ Identification   │
        │                  │ │                  │ │                  │
        │ 🧠 Memory        │ │ 🔗 Prompt Chain  │ │ 🧠 Memory        │
        │ 📚 Knowledge     │ │ 🛠️ Tool Use      │ │ 🤔 Reasoning     │
        │ 📋 Planning      │ │ 🪞 Reflection    │ │ 🛠️ Tools         │
        │ 🤔 Reasoning     │ │ 🛡️ Guardrails    │ │ 🪞 Reflection    │
        └──────────────────┘ └──────────────────┘ └──────────────────┘
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
                              │              │ │              │ │              │
                              │ 🤔 Reasoning │ │ 🛠️ Tool Use  │ │ 🪞 Reflection│
                              │ 🧠 Memory    │ │ 📋 Planning  │ │ 🛡️ Guardrails│
                              │ 📚 Knowledge │ │ 🧠 Memory    │ │ 🧠 Memory    │
                              └──────────────┘ └──────────────┘ └──────────────┘
                                                        │
                                                        ▼
                                              ┌──────────────────┐
                                              │  FINAL SYNTHESIS │
                                              │  & INTEGRATION   │
                                              └──────────────────┘
```

---

## STAGE 1: Extraction Pipeline (3 Agents)

### Agent 1A: Text Analysis & Context Agent

**Primary Responsibilities:**
- Preprocess and analyze claim text structure
- Extract contextual information and claim characteristics
- Query memory for similar historical claims
- Plan optimal extraction strategy

**Agentic Components:**
- 🧠 **Memory**: Historical claim pattern matching
- 📚 **Knowledge Retrieval**: Text processing patterns and domain knowledge
- 📋 **Planning**: Strategic analysis planning based on claim type
- 🤔 **Reasoning**: Context understanding and claim characterization

**Implementation:**
```python
class TextAnalysisContextAgent:
    """Agent 1A: Specialized text analysis and context extraction"""
    
    def __init__(self, shared_memory, knowledge_base):
        self.shared_memory = shared_memory
        self.knowledge_base = knowledge_base
        self.text_processor = TextProcessor()  # Existing dependency
        
    async def analyze_and_contextualize(self, claim_data: Dict) -> Dict:
        """📋 PLANNING + 🤔 REASONING: Analyze claim for optimal extraction strategy"""
        
        # 🧠 MEMORY: Query similar claims
        similar_claims = self.shared_memory.find_similar_claims(claim_data["claim_text"])
        
        # 📚 KNOWLEDGE RETRIEVAL: Get text processing patterns
        processing_patterns = self.knowledge_base.get_text_patterns()
        
        # 🤔 REASONING: Analyze claim characteristics
        claim_analysis = {
            "text_complexity": self._assess_text_complexity(claim_data["claim_text"]),
            "damage_indicators_present": self._detect_damage_keywords(claim_data["claim_text"]),
            "temporal_references": self._identify_temporal_markers(claim_data["claim_text"]),
            "monetary_mentions": self._count_monetary_references(claim_data["claim_text"])
        }
        
        # 📋 PLANNING: Create extraction strategy
        extraction_strategy = self._plan_extraction_strategy(claim_analysis, similar_claims)
        
        return {
            "processed_text": self.text_processor.clean_and_structure(claim_data["claim_text"]),
            "claim_characteristics": claim_analysis,
            "similar_patterns": similar_claims[:3],
            "extraction_strategy": extraction_strategy,
            "context_confidence": self._calculate_context_confidence(claim_analysis)
        }
    
    def _plan_extraction_strategy(self, analysis: Dict, similar_claims: List) -> Dict:
        """📋 PLANNING: Create targeted extraction plan"""
        
        strategy = {
            "complexity_level": analysis["text_complexity"],
            "focus_areas": [],
            "extraction_approach": "standard",
            "confidence_thresholds": {}
        }
        
        # Adjust strategy based on analysis
        if analysis["damage_indicators_present"] > 10:
            strategy["focus_areas"].append("comprehensive_damage_extraction")
            strategy["extraction_approach"] = "detailed"
        
        if analysis["temporal_references"] > 5:
            strategy["focus_areas"].append("temporal_sequence_analysis")
        
        return strategy
```

### Agent 1B: Feature Extraction Agent

**Primary Responsibilities:**
- Extract all 21 building indicators using specialized prompts
- Apply indicator-specific validation and confidence scoring
- Cross-validate related indicators for consistency

**Agentic Components:**
- 🔗 **Prompt Chaining**: Progressive indicator extraction refinement
- 🛠️ **Tool Use**: Specialized extraction and validation tools
- 🪞 **Reflection**: Self-validation of extraction quality
- 🛡️ **Guardrails**: Confidence thresholds and consistency checks

**Implementation:**
```python
class FeatureExtractionAgent:
    """Agent 1B: Specialized 21-indicator extraction"""
    
    def __init__(self, shared_memory, tools):
        self.shared_memory = shared_memory
        self.tools = tools
        self.gpt_api = GptApi()  # Existing dependency
        
    async def extract_21_indicators(self, context_data: Dict) -> Dict:
        """🔗 PROMPT CHAINING: Progressive indicator extraction"""
        
        extraction_strategy = context_data["extraction_strategy"]
        processed_text = context_data["processed_text"]
        
        # Chain 1: Damage Indicators (15 indicators)
        damage_prompt = self._create_damage_extraction_prompt(processed_text, extraction_strategy)
        damage_results = await self._execute_extraction_prompt(damage_prompt)
        
        # Chain 2: Operational Indicators (3 indicators)
        operational_prompt = self._create_operational_extraction_prompt(processed_text, damage_results)
        operational_results = await self._execute_extraction_prompt(operational_prompt)
        
        # Chain 3: Contextual Indicators (3 indicators)
        contextual_prompt = self._create_contextual_extraction_prompt(processed_text, operational_results)
        contextual_results = await self._execute_extraction_prompt(contextual_prompt)
        
        # 🪞 REFLECTION: Self-validate extractions
        all_results = {
            "damage_indicators": damage_results,
            "operational_indicators": operational_results,
            "contextual_indicators": contextual_results
        }
        
        validation_results = await self._reflect_and_validate(all_results)
        
        # 🛡️ GUARDRAILS: Apply consistency checks
        consistency_check = self._apply_consistency_guardrails(all_results)
        
        return {
            **all_results,
            "validation_results": validation_results,
            "consistency_check": consistency_check,
            "extraction_confidence": self._calculate_overall_confidence(all_results)
        }
    
    def _create_damage_extraction_prompt(self, text: str, strategy: Dict) -> str:
        """🔗 PROMPT CHAINING: Create damage-specific extraction prompt"""
        
        base_prompt = f"""
        BUILDING DAMAGE INDICATORS EXTRACTION
        
        Text to analyze: {text}
        
        Extraction Strategy: {strategy["extraction_approach"]}
        Focus Areas: {strategy["focus_areas"]}
        
        Extract the following 15 damage indicators (Y/N only):
        1. BLDG_EXTERIOR_DMG: Any exterior building damage mentioned
        2. BLDG_INTERIOR_DMG: Any interior building damage mentioned
        3. BLDG_ROOF_DMG: Any roof damage mentioned
        4. BLDG_PLUMBING_DMG: Any plumbing system damage mentioned
        5. BLDG_ELECTRICAL_DMG: Any electrical system damage mentioned
        6. BLDG_HVAC_DMG: Any HVAC system damage mentioned
        7. BLDG_FOUNDATION_DMG: Any foundation damage mentioned
        8. BLDG_STRUCTURAL_DMG: Any structural damage mentioned
        9. BLDG_WINDOWS_DMG: Any window damage mentioned
        10. BLDG_DOORS_DMG: Any door damage mentioned
        11. BLDG_FLOORING_DMG: Any flooring damage mentioned
        12. BLDG_WALLS_DMG: Any wall damage mentioned
        13. BLDG_CEILING_DMG: Any ceiling damage mentioned
        14. BLDG_FIRE_DMG: Any fire-related damage mentioned
        15. BLDG_WATER_DMG: Any water-related damage mentioned
        
        For each indicator, provide:
        - value: Y/N
        - confidence: 0.0-1.0
        - source: text evidence
        
        Output as JSON.
        """
        
        return base_prompt
    
    async def _reflect_and_validate(self, results: Dict) -> Dict:
        """🪞 REFLECTION: Self-validate extraction quality"""
        
        reflection_prompt = f"""
        EXTRACTION QUALITY REFLECTION
        
        Extracted Results:
        {json.dumps(results, indent=2)}
        
        REFLECTION QUESTIONS:
        1. Are all indicators properly filled with Y/N values?
        2. Do the confidence scores reflect the evidence quality?
        3. Are there any logical inconsistencies between indicators?
        4. Is the evidence strong enough to support Y indicators?
        5. Are there any missed indicators that should be reconsidered?
        
        Provide reflection analysis and quality assessment.
        """
        
        reflection_response = await self.gpt_api.generate_content(
            prompt=reflection_prompt,
            temperature=0.1
        )
        
        return {
            "reflection_analysis": reflection_response,
            "quality_score": self._calculate_quality_score(results),
            "recommendations": self._generate_improvement_recommendations(results)
        }
```

### Agent 1C: Candidate Identification Agent

**Primary Responsibilities:**
- Extract all monetary candidates for BLDG_LOSS_AMOUNT
- Apply temporal analysis and hierarchical prioritization
- Prepare structured candidates for Stage 2 processing

**Agentic Components:**
- 🧠 **Memory**: Temporal extraction patterns and historical success rates
- 🤔 **Reasoning**: Hierarchical analysis and temporal relationships
- 🛠️ **Tools**: Monetary extraction and temporal analysis tools
- 🪞 **Reflection**: Candidate quality assessment and validation

**Implementation:**
```python
class CandidateIdentificationAgent:
    """Agent 1C: Specialized monetary candidate extraction"""
    
    def __init__(self, shared_memory, tools):
        self.shared_memory = shared_memory
        self.tools = tools
        self.gpt_api = GptApi()
        
    async def identify_monetary_candidates(self, context_data: Dict, feature_data: Dict) -> Dict:
        """🤔 REASONING + 🧠 MEMORY: Extract and analyze monetary candidates"""
        
        processed_text = context_data["processed_text"]
        file_notes = context_data.get("file_notes", [])
        
        # 🧠 MEMORY: Query successful temporal extraction patterns
        temporal_patterns = self.shared_memory.get_temporal_extraction_patterns()
        
        # 🛠️ TOOLS: Extract monetary amounts with context
        monetary_extractions = self.tools["monetary_extractor"].extract_amounts_with_context(
            text=processed_text,
            file_notes=file_notes,
            patterns=temporal_patterns
        )
        
        # 🤔 REASONING: Analyze temporal relationships
        temporal_analysis = await self._analyze_temporal_relationships(monetary_extractions, file_notes)
        
        # 🤔 REASONING: Apply hierarchical prioritization
        prioritized_candidates = self._apply_hierarchical_prioritization(
            monetary_extractions, 
            temporal_analysis,
            feature_data
        )
        
        # 🪞 REFLECTION: Assess candidate quality
        quality_assessment = await self._reflect_on_candidate_quality(prioritized_candidates)
        
        return {
            "BLDG_LOSS_AMOUNT_CANDIDATES": {
                "recent_filenotes_summary": self._create_recent_notes_summary(file_notes),
                "values": prioritized_candidates,
                "temporal_analysis": temporal_analysis,
                "quality_assessment": quality_assessment
            },
            "candidate_extraction_confidence": quality_assessment["overall_confidence"]
        }
    
    async def _analyze_temporal_relationships(self, extractions: List[Dict], file_notes: List[str]) -> Dict:
        """🤔 REASONING: Analyze temporal sequence and relationships"""
        
        temporal_prompt = f"""
        TEMPORAL RELATIONSHIP ANALYSIS
        
        Extracted Monetary Amounts:
        {json.dumps(extractions, indent=2)}
        
        Recent File Notes Context:
        {self._create_recent_notes_summary(file_notes)}
        
        TEMPORAL ANALYSIS REQUIRED:
        1. Identify the chronological sequence of amounts
        2. Determine temporal relationships (progression, updates, corrections)
        3. Assess recency priority based on file note timestamps
        4. Identify any temporal inconsistencies or contradictions
        
        Focus on understanding the evolution of loss amounts over time.
        """
        
        response = await self.gpt_api.generate_content(prompt=temporal_prompt, temperature=0.1)
        
        return {
            "temporal_sequence": self._parse_temporal_sequence(response),
            "recency_scores": self._calculate_recency_scores(extractions, file_notes),
            "temporal_consistency": self._assess_temporal_consistency(extractions)
        }
    
    def _apply_hierarchical_prioritization(self, extractions: List[Dict], temporal_analysis: Dict, feature_data: Dict) -> List[Dict]:
        """🤔 REASONING: Apply hierarchical rules with feature context"""
        
        prioritized = []
        
        for extraction in extractions:
            # Base priority from hierarchical rules
            base_priority = self._determine_base_priority(extraction["context"])
            
            # Temporal adjustment
            temporal_score = temporal_analysis["recency_scores"].get(extraction["amount_text"], 1.0)
            
            # Feature-based context adjustment
            feature_adjustment = self._calculate_feature_adjustment(extraction, feature_data)
            
            # Memory-based confidence
            memory_confidence = self.shared_memory.get_candidate_confidence(extraction)
            
            prioritized_candidate = {
                **extraction,
                "base_priority": base_priority,
                "temporal_score": temporal_score,
                "feature_adjustment": feature_adjustment,
                "memory_confidence": memory_confidence,
                "final_priority": base_priority * temporal_score * feature_adjustment * memory_confidence,
                "prioritization_reasoning": self._create_prioritization_reasoning(
                    base_priority, temporal_score, feature_adjustment, memory_confidence
                )
            }
            
            prioritized.append(prioritized_candidate)
        
        # Sort by final priority (highest first)
        return sorted(prioritized, key=lambda x: x["final_priority"], reverse=True)
```

---

## STAGE 2: Calculation Pipeline (3 Agents)

### Agent 2A: Context Analysis Agent

**Primary Responsibilities:**
- Analyze 21 features for contextual insights and damage assessment
- Calculate damage severity and operational impact multipliers
- Generate feature-based expectations for loss amounts

**Agentic Components:**
- 🤔 **Reasoning**: Deep feature analysis and correlation detection
- 🧠 **Memory**: Feature pattern recognition and historical correlations
- 📚 **Knowledge Retrieval**: Domain knowledge about feature relationships

**Implementation:**
```python
class ContextAnalysisAgent:
    """Agent 2A: Specialized 21-feature context analysis"""
    
    def __init__(self, shared_memory, knowledge_base):
        self.shared_memory = shared_memory
        self.knowledge_base = knowledge_base
        
    async def analyze_feature_context(self, stage1_results: Dict) -> Dict:
        """🤔 REASONING: Comprehensive 21-feature analysis"""
        
        # Extract 21 feature values
        features = self._extract_21_features(stage1_results)
        
        # 🧠 MEMORY: Query similar feature patterns
        similar_patterns = self.shared_memory.find_similar_feature_patterns(features)
        
        # 🤔 REASONING: Analyze damage severity
        damage_analysis = self._analyze_damage_severity(features)
        
        # 🤔 REASONING: Assess operational impact
        operational_analysis = self._analyze_operational_impact(features)
        
        # 🤔 REASONING: Determine contextual factors
        contextual_analysis = self._analyze_contextual_factors(features)
        
        # 📚 KNOWLEDGE RETRIEVAL: Get feature correlation rules
        correlation_rules = self.knowledge_base.get_feature_correlations()
        
        # 🤔 REASONING: Calculate expected loss range
        expected_range = self._calculate_expected_loss_range(
            damage_analysis, operational_analysis, contextual_analysis, similar_patterns
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
                "contextual_multiplier": contextual_analysis["context_multiplier"]
            },
            "similar_patterns": similar_patterns,
            "correlation_insights": self._apply_correlation_rules(features, correlation_rules),
            "context_confidence": self._calculate_context_confidence(damage_analysis, operational_analysis)
        }
    
    def _analyze_damage_severity(self, features: Dict) -> Dict:
        """🤔 REASONING: Analyze damage severity from 15 damage indicators"""
        
        damage_indicators = features["damage_indicators"]
        
        # Count positive damage indicators
        damage_count = sum(1 for indicator in damage_indicators.values() 
                         if indicator.get("value") == "Y")
        
        # Assess damage types and severity
        critical_damages = ["BLDG_STRUCTURAL_DMG", "BLDG_FOUNDATION_DMG", "BLDG_FIRE_DMG"]
        critical_count = sum(1 for indicator in critical_damages 
                           if damage_indicators.get(indicator, {}).get("value") == "Y")
        
        moderate_damages = ["BLDG_ROOF_DMG", "BLDG_ELECTRICAL_DMG", "BLDG_PLUMBING_DMG", "BLDG_HVAC_DMG"]
        moderate_count = sum(1 for indicator in moderate_damages 
                           if damage_indicators.get(indicator, {}).get("value") == "Y")
        
        # Calculate severity level and multiplier
        if critical_count >= 2:
            severity = "critical"
            multiplier = 2.0
        elif critical_count >= 1 or damage_count >= 10:
            severity = "extensive"
            multiplier = 1.7
        elif moderate_count >= 3 or damage_count >= 6:
            severity = "moderate"
            multiplier = 1.3
        elif damage_count >= 2:
            severity = "limited"
            multiplier = 1.1
        else:
            severity = "minimal"
            multiplier = 0.9
        
        return {
            "severity_level": severity,
            "total_damage_count": damage_count,
            "critical_damage_count": critical_count,
            "moderate_damage_count": moderate_count,
            "severity_multiplier": multiplier,
            "damage_breakdown": {
                "structural": damage_indicators.get("BLDG_STRUCTURAL_DMG", {}).get("value") == "Y",
                "exterior": damage_indicators.get("BLDG_EXTERIOR_DMG", {}).get("value") == "Y",
                "interior": damage_indicators.get("BLDG_INTERIOR_DMG", {}).get("value") == "Y",
                "systems": moderate_count > 0
            }
        }
```

### Agent 2B: Calculation & Prioritization Agent

**Primary Responsibilities:**
- Apply hierarchical ranking to monetary candidates
- Execute feature-informed calculations using context from Agent 2A
- Generate preliminary loss amounts with confidence scoring

**Agentic Components:**
- 🛠️ **Tool Use**: Advanced calculation and ranking tools
- 📋 **Planning**: Multi-step calculation strategy planning
- 🧠 **Memory**: Historical calculation patterns and success rates

**Implementation:**
```python
class CalculationPrioritizationAgent:
    """Agent 2B: Specialized calculation and prioritization"""
    
    def __init__(self, shared_memory, calculation_tools):
        self.shared_memory = shared_memory
        self.calculation_tools = calculation_tools
        self.gpt_api = GptApi()
        
    async def execute_calculation(self, candidates: Dict, context_analysis: Dict) -> Dict:
        """📋 PLANNING + 🛠️ TOOL USE: Execute feature-informed calculation"""
        
        # 📋 PLANNING: Plan calculation approach
        calculation_plan = await self._plan_calculation_approach(candidates, context_analysis)
        
        # 🛠️ TOOL USE: Apply hierarchical ranking with feature context
        ranked_candidates = self.calculation_tools["ranker"].rank_candidates_with_features(
            candidates=candidates["values"],
            feature_multipliers=context_analysis["feature_multipliers"],
            expected_range=context_analysis["expected_loss_range"]
        )
        
        # 🧠 MEMORY: Query similar calculation patterns
        similar_calculations = self.shared_memory.find_similar_calculations(
            candidates, context_analysis["feature_analysis"]
        )
        
        # 🛠️ TOOL USE: Execute calculation based on plan
        if calculation_plan["requires_computation"]:
            calculation_result = await self.calculation_tools["calculator"].calculate_with_features(
                ranked_candidates=ranked_candidates,
                feature_context=context_analysis,
                memory_patterns=similar_calculations,
                calculation_plan=calculation_plan
            )
        else:
            # Direct extraction from top candidate
            calculation_result = self._extract_direct_amount(ranked_candidates[0], context_analysis)
        
        # 🧠 MEMORY: Apply memory-based confidence adjustment
        memory_adjustment = self.shared_memory.get_calculation_confidence_adjustment(
            calculation_result, context_analysis
        )
        
        final_result = {
            **calculation_result,
            "memory_adjustment": memory_adjustment,
            "adjusted_confidence": min(0.95, calculation_result["confidence"] * memory_adjustment),
            "calculation_plan": calculation_plan,
            "ranked_candidates": ranked_candidates[:3],  # Top 3 for reference
            "similar_calculations": similar_calculations
        }
        
        return final_result
    
    async def _plan_calculation_approach(self, candidates: Dict, context_analysis: Dict) -> Dict:
        """📋 PLANNING: Plan optimal calculation strategy"""
        
        planning_prompt = f"""
        CALCULATION STRATEGY PLANNING
        
        Candidate Summary:
        - Number of candidates: {len(candidates.get("values", []))}
        - Top candidate: {candidates["values"][0] if candidates.get("values") else "None"}
        
        Feature Context:
        - Damage severity: {context_analysis["feature_analysis"]["damage_severity"]["severity_level"]}
        - Operational impact: {context_analysis["feature_analysis"]["operational_impact"]["impact_level"]}
        - Expected range: {context_analysis["expected_loss_range"]}
        
        PLANNING DECISIONS REQUIRED:
        1. Is direct extraction sufficient or is calculation needed?
        2. What feature adjustments should be applied?
        3. How should memory patterns influence the approach?
        4. What confidence calibration is appropriate?
        
        Plan the optimal calculation strategy.
        """
        
        planning_response = await self.gpt_api.generate_content(
            prompt=planning_prompt,
            temperature=0.2
        )
        
        return {
            "requires_computation": self._assess_computation_need(candidates, context_analysis),
            "feature_adjustment_strategy": self._plan_feature_adjustments(context_analysis),
            "confidence_calibration_approach": self._plan_confidence_calibration(candidates, context_analysis),
            "memory_integration_plan": self._plan_memory_integration(context_analysis),
            "planning_reasoning": planning_response
        }
```

### Agent 2C: Validation & Reflection Agent

**Primary Responsibilities:**
- Validate calculation results against feature context and reasonableness
- Apply comprehensive guardrails and safety checks
- Generate final confidence scores and quality assessments

**Agentic Components:**
- 🪞 **Reflection**: Comprehensive result validation and quality assessment
- 🛡️ **Guardrails**: Safety checks, thresholds, and reasonableness validation
- 🧠 **Memory**: Validation pattern storage and learning

**Implementation:**
```python
class ValidationReflectionAgent:
    """Agent 2C: Specialized validation and reflection"""
    
    def __init__(self, shared_memory, validation_tools, guardrails):
        self.shared_memory = shared_memory
        self.validation_tools = validation_tools
        self.guardrails = guardrails
        self.gpt_api = GptApi()
        
    async def validate_and_reflect(self, calculation_result: Dict, context_analysis: Dict) -> Dict:
        """🪞 REFLECTION + 🛡️ GUARDRAILS: Comprehensive validation"""
        
        final_amount = calculation_result.get("final_amount", 0)
        
        # 🛡️ GUARDRAILS: Apply reasonableness checks
        reasonableness_check = self.validation_tools["reasonableness_checker"].validate_amount(
            amount=final_amount,
            expected_range=context_analysis["expected_loss_range"],
            feature_context=context_analysis["feature_analysis"],
            similar_patterns=context_analysis["similar_patterns"]
        )
        
        # 🪞 REFLECTION: Deep quality assessment
        quality_reflection = await self._perform_quality_reflection(
            calculation_result, context_analysis, reasonableness_check
        )
        
        # 🛡️ GUARDRAILS: Feature consistency validation
        feature_consistency = self._validate_feature_consistency(
            final_amount, context_analysis["feature_analysis"]
        )
        
        # 🧠 MEMORY: Historical validation patterns
        historical_validation = self.shared_memory.get_historical_validation_patterns(
            context_analysis["feature_analysis"]
        )
        
        # 🛡️ GUARDRAILS: Overall validation decision
        overall_validation = self._make_validation_decision(
            reasonableness_check, quality_reflection, feature_consistency
        )
        
        if not overall_validation["passed"]:
            # Handle validation failure
            return await self._handle_validation_failure(
                calculation_result, context_analysis, overall_validation
            )
        
        # Generate final validated result
        validated_result = {
            "value": final_amount,
            "confidence": self._calculate_final_confidence(
                calculation_result, quality_reflection, reasonableness_check
            ),
            "justification": self._create_comprehensive_justification(
                calculation_result, context_analysis, quality_reflection
            ),
            "validation_results": {
                "reasonableness_check": reasonableness_check,
                "quality_reflection": quality_reflection,
                "feature_consistency": feature_consistency,
                "overall_validation": overall_validation
            },
            "feature_context": context_analysis["feature_analysis"],
            "calculation_metadata": {
                "calculation_method": calculation_result.get("calculation_method"),
                "feature_adjustments": calculation_result.get("feature_adjustments"),
                "memory_patterns_used": len(calculation_result.get("similar_calculations", []))
            }
        }
        
        # 🧠 MEMORY: Store validation results for learning
        self.shared_memory.store_validation_result(validated_result, context_analysis)
        
        return validated_result
    
    async def _perform_quality_reflection(self, calc_result: Dict, context: Dict, reasonableness: Dict) -> Dict:
        """🪞 REFLECTION: Deep quality assessment"""
        
        reflection_prompt = f"""
        CALCULATION QUALITY REFLECTION
        
        Calculation Result:
        - Final Amount: ${calc_result.get("final_amount", 0):,.2f}
        - Confidence: {calc_result.get("confidence", 0):.2f}
        - Method: {calc_result.get("calculation_method", "unknown")}
        
        Feature Context:
        - Damage Severity: {context["feature_analysis"]["damage_severity"]["severity_level"]}
        - Operational Impact: {context["feature_analysis"]["operational_impact"]["impact_level"]}
        - Expected Range: ${context["expected_loss_range"][0]:,.2f} - ${context["expected_loss_range"][1]:,.2f}
        
        Reasonableness Check:
        - Amount in expected range: {reasonableness["amount_in_expected_range"]}
        - Feature alignment score: {reasonableness["feature_alignment_score"]:.2f}
        - Historical consistency: {reasonableness["memory_consistency_score"]:.2f}
        
        REFLECTION QUESTIONS:
        1. Does the final amount logically align with the 21-feature context?
        2. Are the feature adjustments appropriate for the damage severity?
        3. Is the confidence score calibrated correctly?
        4. Are there any red flags or inconsistencies?
        5. How does this compare to similar historical cases?
        6. What would cause us to question this result?
        
        Provide comprehensive quality assessment and recommendations.
        """
        
        reflection_response = await self.gpt_api.generate_content(
            prompt=reflection_prompt,
            temperature=0.1
        )
        
        return {
            "reflection_analysis": reflection_response,
            "quality_score": self._calculate_quality_score(calc_result, context, reasonableness),
            "confidence_calibration": self._assess_confidence_calibration(calc_result, context),
            "consistency_assessment": self._assess_result_consistency(calc_result, context),
            "improvement_recommendations": self._generate_improvement_recommendations(calc_result, context)
        }
    
    def _make_validation_decision(self, reasonableness: Dict, quality: Dict, consistency: Dict) -> Dict:
        """🛡️ GUARDRAILS: Make final validation decision"""
        
        validation_criteria = {
            "reasonableness_passed": reasonableness["overall_reasonableness"] >= 0.7,
            "quality_passed": quality["quality_score"] >= 0.7,
            "consistency_passed": consistency["consistency_score"] >= 0.6,
            "confidence_adequate": quality["confidence_calibration"]["is_adequate"]
        }
        
        overall_passed = all(validation_criteria.values())
        
        return {
            "passed": overall_passed,
            "criteria": validation_criteria,
            "validation_score": sum(validation_criteria.values()) / len(validation_criteria),
            "validation_timestamp": datetime.now().isoformat()
        }
```

---

## Inter-Agent Communication & Orchestration

### LangGraph Orchestration

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict

class MultiAgentState(TypedDict):
    # Input
    claim_id: str
    claim_text: str
    file_notes: List[str]
    
    # Stage 1 Agent Results
    text_context: Dict      # Agent 1A output
    feature_extractions: Dict  # Agent 1B output
    monetary_candidates: Dict  # Agent 1C output
    
    # Stage 2 Agent Results
    context_analysis: Dict     # Agent 2A output
    calculation_result: Dict   # Agent 2B output
    validation_result: Dict    # Agent 2C output
    
    # Shared State
    shared_memory: AgenticMemoryStore
    processing_metadata: Dict
    final_output: Dict

class MultiAgentOrchestrator:
    """Orchestrates 6-agent communication and workflow"""
    
    def __init__(self):
        # Initialize shared services
        self.shared_memory = AgenticMemoryStore()
        self.shared_knowledge = KnowledgeBase()
        self.shared_tools = ToolRegistry()
        self.system_guardrails = SystemGuardrails()
        
        # Initialize agents
        self.agents = {
            "1A": TextAnalysisContextAgent(self.shared_memory, self.shared_knowledge),
            "1B": FeatureExtractionAgent(self.shared_memory, self.shared_tools),
            "1C": CandidateIdentificationAgent(self.shared_memory, self.shared_tools),
            "2A": ContextAnalysisAgent(self.shared_memory, self.shared_knowledge),
            "2B": CalculationPrioritizationAgent(self.shared_memory, self.shared_tools),
            "2C": ValidationReflectionAgent(self.shared_memory, self.shared_tools, self.system_guardrails)
        }
    
    def create_multi_agent_workflow(self) -> StateGraph:
        """Create 6-agent orchestrated workflow"""
        
        workflow = StateGraph(MultiAgentState)
        
        # Stage 1 agents (sequential)
        workflow.add_node("agent_1A", self.execute_agent_1A)
        workflow.add_node("agent_1B", self.execute_agent_1B)
        workflow.add_node("agent_1C", self.execute_agent_1C)
        
        # Conditional trigger
        workflow.add_node("evaluate_candidates", self.evaluate_candidates)
        
        # Stage 2 agents (sequential)
        workflow.add_node("agent_2A", self.execute_agent_2A)
        workflow.add_node("agent_2B", self.execute_agent_2B)
        workflow.add_node("agent_2C", self.execute_agent_2C)
        
        # Final integration
        workflow.add_node("final_integration", self.final_integration)
        
        # Define workflow
        workflow.add_edge("agent_1A", "agent_1B")
        workflow.add_edge("agent_1B", "agent_1C")
        workflow.add_edge("agent_1C", "evaluate_candidates")
        
        workflow.add_conditional_edges(
            "evaluate_candidates",
            self.check_candidates_found,
            {
                "proceed_stage2": "agent_2A",
                "finalize": "final_integration"
            }
        )
        
        workflow.add_edge("agent_2A", "agent_2B")
        workflow.add_edge("agent_2B", "agent_2C")
        workflow.add_edge("agent_2C", "final_integration")
        workflow.add_edge("final_integration", END)
        
        workflow.set_entry_point("agent_1A")
        
        return workflow.compile()
    
    async def execute_agent_1A(self, state: MultiAgentState) -> MultiAgentState:
        """Execute Text Analysis & Context Agent"""
        
        result = await self.agents["1A"].analyze_and_contextualize({
            "claim_id": state["claim_id"],
            "claim_text": state["claim_text"],
            "file_notes": state["file_notes"]
        })
        
        state["text_context"] = result
        state["processing_metadata"]["agent_1A_complete"] = datetime.now().isoformat()
        
        return state
    
    async def execute_agent_1B(self, state: MultiAgentState) -> MultiAgentState:
        """Execute Feature Extraction Agent"""
        
        result = await self.agents["1B"].extract_21_indicators(state["text_context"])
        
        state["feature_extractions"] = result
        state["processing_metadata"]["agent_1B_complete"] = datetime.now().isoformat()
        
        return state
    
    async def execute_agent_1C(self, state: MultiAgentState) -> MultiAgentState:
        """Execute Candidate Identification Agent"""
        
        result = await self.agents["1C"].identify_monetary_candidates(
            state["text_context"], 
            state["feature_extractions"]
        )
        
        state["monetary_candidates"] = result
        state["processing_metadata"]["agent_1C_complete"] = datetime.now().isoformat()
        
        return state
    
    def check_candidates_found(self, state: MultiAgentState) -> str:
        """Check if monetary candidates were found"""
        
        candidates = state["monetary_candidates"].get("BLDG_LOSS_AMOUNT_CANDIDATES", {}).get("values", [])
        
        if candidates and len(candidates) > 0:
            return "proceed_stage2"
        else:
            return "finalize"

# Main execution function
async def process_claim_multi_agent(claim_data: Dict) -> Dict:
    """Process claim using 6-agent architecture"""
    
    orchestrator = MultiAgentOrchestrator()
    workflow = orchestrator.create_multi_agent_workflow()
    
    initial_state = {
        "claim_id": claim_data["claim_id"],
        "claim_text": claim_data["claim_text"],
        "file_notes": claim_data.get("file_notes", []),
        "text_context": {},
        "feature_extractions": {},
        "monetary_candidates": {},
        "context_analysis": {},
        "calculation_result": {},
        "validation_result": {},
        "shared_memory": orchestrator.shared_memory,
        "processing_metadata": {"start_time": datetime.now().isoformat()},
        "final_output": {}
    }
    
    final_state = await workflow.ainvoke(initial_state)
    
    return final_state["final_output"]
```

---

## Questions & Recommendations for Implementation

### Critical Questions:

1. **Performance Trade-offs**: 
   - Are you willing to accept potential 10-15% increase in processing time for improved maintainability and debugging?
   - Should we implement parallel processing for Stage 1 agents (1A, 1B potentially parallel)?

2. **Memory Architecture**:
   - Do you prefer centralized memory store or distributed memory with synchronization?
   - How should we handle memory conflicts when multiple agents update simultaneously?

3. **Error Recovery**:
   - If Agent 1B fails, should Agent 1C continue with available data or wait for retry?
   - What's the acceptable failure rate before escalating to manual review?

4. **Implementation Approach**:
   - Should we implement incremental migration (Stage 1 decomposition first, then Stage 2)?
   - Do you want A/B testing capability to compare 2-agent vs. 6-agent performance?

### Recommendations:

1. **Start with Stage 1 Decomposition**: Implement agents 1A, 1B, 1C first while keeping Stage 2 as single agent
2. **Implement Comprehensive Monitoring**: Add detailed logging and performance metrics for each agent
3. **Create Fallback Mechanisms**: If any agent fails, graceful degradation to previous architecture
4. **Performance Benchmarking**: Establish baseline metrics before implementing multi-agent architecture

The 6-agent decomposition provides significant benefits in maintainability and specialization while preserving all agentic intelligence capabilities. The key success factor is robust inter-agent communication and comprehensive error handling.

<function_calls>
<invoke name="TodoWrite">
<parameter name="todos">[{"id": "1", "content": "Research optimal agent breakdown for Stage 1 and Stage 2", "status": "completed", "priority": "high"}, {"id": "2", "content": "Analyze agentic component distribution across multiple agents", "status": "completed", "priority": "high"}, {"id": "3", "content": "Design decluttered multi-agent architecture", "status": "completed", "priority": "high"}, {"id": "4", "content": "Validate agent communication and coordination", "status": "completed", "priority": "medium"}, {"id": "5", "content": "Document questions and recommendations for user", "status": "completed", "priority": "medium"}]