# Agentic AI Framework Architecture for Building Coverage Analysis

## Executive Summary

This framework transforms the existing Section 2 (coverage_rag_implementation) functionality into an intelligent agentic system using LangGraph StateGraph. The design focuses on time-based extraction patterns for BLDG_LOSS_AMOUNT with detailed memory management for loss amounts as separate line items, leveraging existing dependencies without introducing new imports.

---

## Core Architecture Overview

### Framework Components
```
┌─────────────────────────────────────────────────────────────┐
│                    LangGraph StateGraph                     │
├─────────────────────────────────────────────────────────────┤
│  Agent Orchestrator                                         │
│  ├── Memory Manager Agent                                   │
│  ├── Text Processing Agent                                  │
│  ├── Temporal Extraction Agent                              │
│  ├── BLDG Loss Analysis Agent                              │
│  ├── Validation Agent                                       │
│  └── Output Synthesis Agent                                 │
└─────────────────────────────────────────────────────────────┘
```

### State Management Structure
```python
# Core State Schema (using existing Pydantic patterns from base_class.py)
class AgenticState(TypedDict):
    claim_id: str
    raw_text: str
    processed_chunks: List[Dict]
    temporal_context: Dict
    memory_store: Dict
    loss_line_items: List[Dict]
    building_indicators: Dict
    validation_results: Dict
    confidence_scores: Dict
    processing_stage: str
```

---

## Agent Node Designs

### 1. Memory Manager Agent
**Purpose**: Manages persistent memory for temporal context and loss amount tracking

**Existing Dependencies Used**:
- `text_processor.py` for text deduplication
- `rag_params.py` for memory configuration parameters

**Memory Structure for BLDG_LOSS_AMOUNT**:
```python
class LossAmountMemory:
    def __init__(self):
        self.temporal_extractions = {
            "historical_amounts": [],  # Previous loss amounts with timestamps
            "current_session": {},     # Current processing session data
            "context_patterns": {},    # Learned extraction patterns
            "validation_history": []   # Previous validation results
        }
        
    def store_loss_line_item(self, loss_item):
        """Store individual loss amounts as separate line items"""
        line_item = {
            "amount": loss_item.get("amount"),
            "timestamp": loss_item.get("timestamp"),
            "source_text": loss_item.get("source_text"),
            "extraction_confidence": loss_item.get("confidence"),
            "validation_status": loss_item.get("validation"),
            "temporal_context": {
                "relative_time": loss_item.get("relative_time"),
                "sequence_order": loss_item.get("sequence"),
                "hierarchical_level": loss_item.get("hierarchy_level")
            }
        }
        self.temporal_extractions["current_session"]["loss_items"].append(line_item)
        
    def get_temporal_context(self, claim_text):
        """Retrieve relevant temporal context for current extraction"""
        # Use existing prompts.py hierarchical logic for time-based extraction
        return {
            "similar_patterns": self._find_similar_patterns(claim_text),
            "historical_confidence": self._calculate_historical_confidence(),
            "temporal_indicators": self._extract_temporal_indicators(claim_text)
        }
```

### 2. Text Processing Agent
**Purpose**: Processes claim text using existing text processing capabilities

**Existing Dependencies Used**:
- `text_processor.py` → `TextProcessor` class
- `chunk_splitter.py` → `TextChunkSplitter` class

**Implementation**:
```python
def process_claim_text(state: AgenticState) -> AgenticState:
    """
    Processes raw claim text using existing TextProcessor
    Leverages existing deduplication and keyword filtering
    """
    processor = TextProcessor()  # From existing coverage_rag_implementation
    
    # Use existing functionality
    cleaned_text = processor.remove_duplicate_sentences(state["raw_text"])
    filtered_text = processor.filter_important_keywords(cleaned_text)
    
    # Create chunks using existing splitter
    splitter = TextChunkSplitter()
    chunks = splitter.split_text_by_sentences(filtered_text)
    
    state["processed_chunks"] = [
        {
            "text": chunk,
            "chunk_id": idx,
            "processing_timestamp": datetime.now().isoformat(),
            "keywords_found": processor.extract_keywords(chunk)
        }
        for idx, chunk in enumerate(chunks)
    ]
    
    return state
```

### 3. Temporal Extraction Agent
**Purpose**: Implements time-based extraction patterns for BLDG_LOSS_AMOUNT

**Existing Dependencies Used**:
- `prompts.py` → Existing BLDG_LOSS_AMOUNT hierarchical logic
- `gpt_api.py` → `GptApi` class for LLM calls

**Time-Based Extraction Logic**:
```python
def extract_temporal_loss_amounts(state: AgenticState) -> AgenticState:
    """
    Implements time-based extraction using existing prompt hierarchical logic
    from prompts.py for BLDG_LOSS_AMOUNT
    """
    memory_manager = state["memory_store"]["loss_memory"]
    temporal_context = memory_manager.get_temporal_context(state["raw_text"])
    
    # Use existing GPT API with enhanced temporal prompts
    gpt_api = GptApi()  # From existing coverage_rag_implementation
    
    # Enhanced prompt incorporating existing hierarchical logic
    temporal_prompt = f"""
    {get_existing_bldg_loss_prompt()}  # From prompts.py
    
    Additional Context from Memory:
    - Similar historical patterns: {temporal_context['similar_patterns']}
    - Previous extraction confidence: {temporal_context['historical_confidence']}
    - Temporal indicators found: {temporal_context['temporal_indicators']}
    
    Focus on time-based extraction with hierarchy:
    1. Explicit dollar amounts with dates
    2. Relative time references (before/after dates)
    3. Sequential loss reporting patterns
    4. Temporal relationship validation
    
    Extract each loss amount as separate line item with:
    - Exact amount
    - Temporal reference
    - Hierarchical priority level
    - Source text location
    """
    
    # Process each chunk with temporal awareness
    temporal_extractions = []
    for chunk in state["processed_chunks"]:
        response = gpt_api.generate_content(
            prompt=temporal_prompt + f"\n\nText to analyze: {chunk['text']}",
            temperature=0.1  # Low temperature for consistent extraction
        )
        
        # Parse temporal extraction results
        extraction_result = parse_temporal_response(response)
        if extraction_result["loss_amounts"]:
            for loss_item in extraction_result["loss_amounts"]:
                memory_manager.store_loss_line_item({
                    **loss_item,
                    "chunk_id": chunk["chunk_id"],
                    "extraction_timestamp": datetime.now().isoformat()
                })
                temporal_extractions.append(loss_item)
    
    state["temporal_context"]["extractions"] = temporal_extractions
    return state
```

### 4. BLDG Loss Analysis Agent
**Purpose**: Analyzes extracted loss amounts using existing RAG processor logic

**Existing Dependencies Used**:
- `rag_processor.py` → `RAGProcessor` class
- `prompts.py` → Existing 22 building indicator prompts

**Implementation**:
```python
def analyze_building_loss(state: AgenticState) -> AgenticState:
    """
    Analyzes building loss using existing RAGProcessor functionality
    Enhanced with temporal context and memory
    """
    rag_processor = RAGProcessor()  # From existing coverage_rag_implementation
    
    # Get memory context for enhanced analysis
    loss_memory = state["memory_store"]["loss_memory"]
    temporal_extractions = state["temporal_context"]["extractions"]
    
    # Use existing get_summary_and_loss_desc_b_code method with enhancements
    analysis_result = rag_processor.get_summary_and_loss_desc_b_code(
        state["raw_text"],
        temporal_context=temporal_extractions,
        memory_context=loss_memory.temporal_extractions
    )
    
    # Process each loss amount as separate line item
    loss_line_items = []
    for extraction in temporal_extractions:
        line_item = {
            "loss_amount": extraction["amount"],
            "confidence_score": extraction["confidence"],
            "temporal_reference": extraction["temporal_context"],
            "source_validation": validate_against_existing_rules(extraction),
            "hierarchical_priority": extraction["hierarchy_level"],
            "related_indicators": correlate_with_22_indicators(extraction, analysis_result)
        }
        loss_line_items.append(line_item)
    
    state["loss_line_items"] = loss_line_items
    state["building_indicators"] = analysis_result
    
    return state
```

### 5. Validation Agent
**Purpose**: Validates extractions using existing validation patterns

**Existing Dependencies Used**:
- `base_class.py` → Pydantic validation patterns
- `bldg_rules.py` → Existing rule definitions

**Implementation**:
```python
def validate_extractions(state: AgenticState) -> AgenticState:
    """
    Validates temporal extractions using existing validation patterns
    """
    validation_results = {}
    
    # Validate each loss line item
    for idx, line_item in enumerate(state["loss_line_items"]):
        validation = {
            "amount_validity": validate_amount_format(line_item["loss_amount"]),
            "temporal_consistency": validate_temporal_logic(line_item["temporal_reference"]),
            "hierarchy_compliance": validate_hierarchy_rules(line_item["hierarchical_priority"]),
            "cross_reference_check": validate_against_memory(line_item, state["memory_store"]),
            "confidence_threshold": line_item["confidence_score"] >= 0.7
        }
        validation_results[f"line_item_{idx}"] = validation
    
    # Overall validation using existing BLDG rules
    overall_validation = apply_existing_bldg_rules(state["building_indicators"])
    validation_results["overall"] = overall_validation
    
    state["validation_results"] = validation_results
    return state
```

### 6. Output Synthesis Agent
**Purpose**: Synthesizes final output using existing transformation patterns

**Existing Dependencies Used**:
- `coverage_transformations.py` → `DataFrameTransformations` class

**Implementation**:
```python
def synthesize_output(state: AgenticState) -> AgenticState:
    """
    Synthesizes final output using existing transformation patterns
    Enhanced with temporal and memory context
    """
    transformer = DataFrameTransformations()  # From existing coverage_rules
    
    # Create enhanced output structure
    output_data = {
        "claim_id": state["claim_id"],
        "processed_loss_amounts": state["loss_line_items"],
        "building_indicators": state["building_indicators"],
        "temporal_analysis": state["temporal_context"],
        "validation_summary": state["validation_results"],
        "memory_insights": state["memory_store"]["loss_memory"].get_insights(),
        "confidence_metrics": calculate_overall_confidence(state)
    }
    
    # Use existing transformation methods
    formatted_output = transformer.select_and_rename_bldg_predictions_for_db(
        pd.DataFrame([output_data])
    )
    
    state["final_output"] = formatted_output
    return state
```

---

## LangGraph StateGraph Implementation

### Graph Definition
```python
from langgraph.graph import StateGraph, END

def create_building_coverage_agent():
    """
    Creates the agentic workflow using LangGraph StateGraph
    """
    # Initialize workflow
    workflow = StateGraph(AgenticState)
    
    # Add agent nodes
    workflow.add_node("memory_manager", initialize_memory)
    workflow.add_node("text_processor", process_claim_text)
    workflow.add_node("temporal_extractor", extract_temporal_loss_amounts)
    workflow.add_node("loss_analyzer", analyze_building_loss)
    workflow.add_node("validator", validate_extractions)
    workflow.add_node("synthesizer", synthesize_output)
    
    # Define workflow edges
    workflow.add_edge("memory_manager", "text_processor")
    workflow.add_edge("text_processor", "temporal_extractor")
    workflow.add_edge("temporal_extractor", "loss_analyzer")
    workflow.add_edge("loss_analyzer", "validator")
    workflow.add_edge("validator", "synthesizer")
    workflow.add_edge("synthesizer", END)
    
    # Set entry point
    workflow.set_entry_point("memory_manager")
    
    return workflow.compile()
```

### Conditional Flow Logic
```python
def add_conditional_logic(workflow):
    """
    Adds conditional routing based on validation results and confidence scores
    """
    def should_reprocess(state: AgenticState) -> str:
        """Route based on validation results"""
        validation = state["validation_results"]
        overall_confidence = calculate_overall_confidence(state)
        
        if overall_confidence < 0.6:
            return "temporal_extractor"  # Retry extraction
        elif any(not v["confidence_threshold"] for v in validation.values()):
            return "loss_analyzer"  # Reanalyze with different parameters
        else:
            return "synthesizer"  # Proceed to output
    
    workflow.add_conditional_edges(
        "validator",
        should_reprocess,
        {
            "temporal_extractor": "temporal_extractor",
            "loss_analyzer": "loss_analyzer", 
            "synthesizer": "synthesizer"
        }
    )
```

---

## Memory Management for All 22 Building Indicators

### Enhanced Memory Structure
```python
class ComprehensiveMemory:
    def __init__(self):
        self.indicator_memory = {
            # Financial Indicators
            "BLDG_LOSS_AMOUNT": LossAmountMemory(),
            
            # Damage Type Indicators  
            "BLDG_EXTERIOR_DMG": DamageTypeMemory("exterior"),
            "BLDG_INTERIOR_DMG": DamageTypeMemory("interior"),
            "BLDG_ROOF_DMG": DamageTypeMemory("roof"),
            "BLDG_PLUMBING_DMG": DamageTypeMemory("plumbing"),
            "BLDG_ELECTRICAL_DMG": DamageTypeMemory("electrical"),
            
            # Operational Indicators
            "BLDG_TENABLE": OperationalMemory("tenable"),
            "BLDG_UNOCCUPIABLE": OperationalMemory("unoccupiable"),
            "BLDG_COMPLETE_LOSS": OperationalMemory("complete_loss"),
            
            # Contextual Indicators
            "BLDG_PRIMARY": ContextualMemory("primary"),
            "BLDG_ADJACENT_ORIGIN": ContextualMemory("adjacent"),
            "BLDG_DIRECT_ORIGIN": ContextualMemory("direct")
            
            # ... (all 22 indicators with specialized memory classes)
        }
        
        self.cross_indicator_patterns = {}  # Patterns across multiple indicators
        self.temporal_correlations = {}     # Time-based correlations
        self.confidence_trends = {}         # Confidence tracking over time

class DamageTypeMemory:
    """Specialized memory for damage type indicators"""
    def __init__(self, damage_type):
        self.damage_type = damage_type
        self.pattern_history = []
        self.keyword_associations = {}
        self.severity_mappings = {}
        
class OperationalMemory:
    """Specialized memory for operational status indicators"""
    def __init__(self, operational_type):
        self.operational_type = operational_type
        self.status_transitions = []  # Track status changes over time
        self.condition_patterns = {}
        self.decision_factors = {}

class ContextualMemory:
    """Specialized memory for contextual indicators"""
    def __init__(self, context_type):
        self.context_type = context_type
        self.relationship_patterns = []
        self.proximity_indicators = {}
        self.causation_chains = {}
```

---

## Execution Flow

### 1. Initialization Phase
```python
async def process_claim(claim_data):
    """Main entry point for processing a claim"""
    
    # Initialize agentic workflow
    agent = create_building_coverage_agent()
    
    # Prepare initial state
    initial_state = {
        "claim_id": claim_data["claim_id"],
        "raw_text": claim_data["claim_text"],
        "processed_chunks": [],
        "temporal_context": {},
        "memory_store": {"loss_memory": LossAmountMemory()},
        "loss_line_items": [],
        "building_indicators": {},
        "validation_results": {},
        "confidence_scores": {},
        "processing_stage": "initialized"
    }
    
    # Execute workflow
    final_state = await agent.ainvoke(initial_state)
    
    return final_state
```

### 2. Memory Persistence
```python
class MemoryPersistence:
    """Handles memory persistence across processing sessions"""
    
    def save_session_memory(self, state: AgenticState):
        """Save memory insights for future sessions"""
        memory_snapshot = {
            "timestamp": datetime.now().isoformat(),
            "claim_id": state["claim_id"],
            "loss_patterns": state["memory_store"]["loss_memory"].get_patterns(),
            "validation_insights": state["validation_results"],
            "confidence_metrics": state["confidence_scores"]
        }
        # Persist to existing database infrastructure
        
    def load_relevant_memory(self, claim_context):
        """Load relevant historical memory for current claim"""
        # Query historical patterns based on claim characteristics
        return historical_memory_context
```

---

## Integration with Existing System

### Using Existing RAG Parameters
```python
# Leverage existing rag_params.py configuration
def configure_agent_parameters():
    """Configure agent using existing RAG parameters"""
    from coverage_configs.src.configs.rag_params import rag_params
    
    agent_config = {
        "gpt_temperature": rag_params["temperature"],
        "max_tokens": rag_params["max_tokens"],
        "chunk_size": rag_params["chunk_size"],
        "overlap_ratio": rag_params["overlap_ratio"],
        "model_path": rag_params["model_path"]
    }
    
    return agent_config
```

### Existing Prompt Integration
```python
# Use existing prompts from prompts.py with enhancements
def enhance_existing_prompts_with_temporal_context():
    """Enhance existing prompts with temporal and memory context"""
    from coverage_configs.src.configs.prompts import get_bldg_loss_prompt
    
    base_prompt = get_bldg_loss_prompt()
    
    enhanced_prompt = f"""
    {base_prompt}
    
    TEMPORAL ENHANCEMENT:
    - Consider time-based relationships in loss reporting
    - Identify sequential patterns in damage descriptions
    - Validate temporal consistency across related extractions
    
    MEMORY INTEGRATION:
    - Reference similar historical patterns from memory
    - Apply learned extraction confidence factors
    - Utilize cross-claim validation insights
    """
    
    return enhanced_prompt
```

---

## Benefits of This Agentic Architecture

### 1. **Time-Based Intelligence**
- Captures temporal relationships in loss amount reporting
- Maintains sequential context across processing steps
- Learns from historical temporal patterns

### 2. **Detailed Memory Management**
- Stores loss amounts as individual line items with full context
- Tracks confidence evolution over time
- Maintains specialized memory for each of the 22 building indicators

### 3. **Existing Dependency Utilization**
- Reuses all existing classes: TextProcessor, RAGProcessor, GptApi
- Leverages existing prompt engineering from prompts.py
- Maintains compatibility with current data pipeline

### 4. **Enhanced Accuracy**
- Memory-driven pattern recognition improves extraction accuracy
- Temporal validation reduces false positives
- Cross-indicator correlation provides context validation

### 5. **Scalable Architecture**
- LangGraph StateGraph enables complex workflow management
- Modular agent design allows independent enhancement
- Conditional routing optimizes processing efficiency

This architecture transforms the existing building coverage system into an intelligent, memory-aware agentic framework while preserving all existing functionality and dependencies.