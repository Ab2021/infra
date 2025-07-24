"""
GPT prompt templates for building coverage analysis.

This module contains all prompt templates used by the RAG system
for building coverage determination and claim analysis.
"""

from typing import Dict, Any, List, Optional


def get_gpt_prompts() -> Dict[str, Any]:
    """
    Get all GPT prompt templates organized by use case.
    
    Returns:
        Dict[str, Any]: Dictionary containing all prompt templates
    """
    return {
        'building_coverage_analysis': get_building_coverage_prompt(),
        'detailed_analysis': get_detailed_analysis_prompt(),
        'confidence_assessment': get_confidence_assessment_prompt(),
        'summary_generation': get_summary_generation_prompt(),
        'validation_prompts': get_validation_prompts()
    }


def get_building_coverage_prompt() -> str:
    """
    Get the main building coverage analysis prompt.
    
    Returns:
        str: Building coverage analysis prompt template
    """
    return '''
You are an expert insurance claim analyst specializing in building coverage determination. Your task is to analyze insurance claim descriptions and determine whether they involve building coverage.

**ANALYSIS CRITERIA:**

**BUILDING COVERAGE INDICATORS (Positive):**
- Structural damage to buildings, foundations, walls, roofs, floors, ceilings
- Building materials and construction elements (wood, concrete, steel, drywall)
- Permanent fixtures and architectural features (windows, doors, built-in cabinets)
- Building systems (HVAC, plumbing, electrical when part of structure)
- Structural repairs, reconstruction, or rebuilding
- Foundation issues, settling, or structural integrity problems
- Building envelope damage (exterior walls, roof systems)
- Load-bearing structural components

**NON-BUILDING COVERAGE (Negative):**
- Personal property and contents (furniture, equipment, inventory)
- Vehicles, machinery, or mobile equipment
- Landscaping, fencing, or outdoor improvements
- Business interruption without physical building damage
- Liability claims without property damage
- Pure financial or consequential losses

**AMBIGUOUS CASES:**
- Attached structures (garages, decks) - generally BUILDING COVERAGE
- Improvements and betterments - depends on permanence
- Tenant improvements - may or may not be building coverage

**ANALYSIS INSTRUCTIONS:**
1. Read the claim description carefully
2. Identify key damage indicators and their relationship to building structure
3. Consider the primary cause and nature of damage
4. Determine if structural building components are involved
5. Assess confidence based on clarity and specificity of description

**OUTPUT FORMAT:**
Provide your analysis in this exact format:

DETERMINATION: [BUILDING COVERAGE / NO BUILDING COVERAGE]
CONFIDENCE: [0.0-1.0]
SUMMARY: [2-3 sentence explanation of your reasoning]
KEY FACTORS: [List of specific evidence supporting your decision]
STRUCTURAL COMPONENTS: [Building components mentioned, if any]

**CLAIM DESCRIPTION TO ANALYZE:**
{claim_text}

Provide your analysis now:
'''


def get_detailed_analysis_prompt() -> str:
    """
    Get detailed analysis prompt for complex claims.
    
    Returns:
        str: Detailed analysis prompt template
    """
    return '''
You are conducting a detailed building coverage analysis for a complex insurance claim. This claim requires careful examination of multiple factors.

**DETAILED ANALYSIS REQUIREMENTS:**

1. **DAMAGE ASSESSMENT:**
   - Identify all types of damage mentioned
   - Categorize damage by building system/component
   - Assess severity and extent of structural impact

2. **CAUSATION ANALYSIS:**
   - Determine primary cause of loss
   - Identify contributing factors
   - Assess whether cause typically affects building structure

3. **COVERAGE IMPLICATIONS:**
   - Evaluate which damages require building coverage
   - Identify any non-building components
   - Consider coverage boundaries and exclusions

4. **REPAIR REQUIREMENTS:**
   - Assess likely repair/replacement needs
   - Consider structural vs. cosmetic repairs
   - Evaluate impact on building integrity

**CLAIM DETAILS:**
Claim Text: {claim_text}
LOB Code: {lob_code}
Loss Date: {loss_date}
Additional Context: {additional_context}

**PROVIDE DETAILED ANALYSIS:**

DAMAGE BREAKDOWN:
[List and categorize all damage types]

STRUCTURAL IMPACT:
[Assess impact on building structure]

COVERAGE DETERMINATION:
[Primary coverage type needed]

CONFIDENCE LEVEL:
[0.0-1.0 with justification]

RECOMMENDATIONS:
[Any additional investigation needed]
'''


def get_confidence_assessment_prompt() -> str:
    """
    Get confidence assessment prompt for prediction validation.
    
    Returns:
        str: Confidence assessment prompt template
    """
    return '''
You are reviewing a building coverage determination to assess confidence and identify potential issues.

**CONFIDENCE ASSESSMENT CRITERIA:**

**HIGH CONFIDENCE (0.8-1.0):**
- Clear, specific description of building damage
- Multiple structural components mentioned
- Unambiguous building-related terminology
- Detailed damage description with context

**MEDIUM CONFIDENCE (0.5-0.79):**
- Some building indicators present
- Limited detail or ambiguous descriptions
- Mixed signals requiring interpretation
- Standard insurance terminology used

**LOW CONFIDENCE (0.0-0.49):**
- Vague or unclear descriptions
- Insufficient information for determination
- Conflicting or ambiguous indicators
- Non-standard or unclear terminology

**REVIEW THE FOLLOWING:**
Original Claim: {claim_text}
Initial Determination: {initial_determination}
Initial Confidence: {initial_confidence}
Reasoning: {initial_reasoning}

**CONFIDENCE VALIDATION:**

CONFIDENCE ASSESSMENT: [0.0-1.0]
JUSTIFICATION: [Why this confidence level is appropriate]
POTENTIAL ISSUES: [Any concerns or ambiguities]
ADDITIONAL INFO NEEDED: [What would improve confidence]
FINAL RECOMMENDATION: [Confirm, adjust, or flag for review]
'''


def get_summary_generation_prompt() -> str:
    """
    Get summary generation prompt for claim processing results.
    
    Returns:
        str: Summary generation prompt template
    """
    return '''
Generate a concise, professional summary of the building coverage analysis for business use.

**SUMMARY REQUIREMENTS:**
- Professional insurance industry language
- Clear, actionable conclusion
- Key supporting evidence
- Appropriate technical detail level
- Suitable for claims processing workflow

**ANALYSIS RESULTS:**
Claim Number: {claim_number}
Determination: {determination}
Confidence: {confidence}
Key Factors: {key_factors}
Structural Components: {structural_components}

**GENERATE PROFESSIONAL SUMMARY:**

CLAIM SUMMARY:
[2-3 sentence professional summary]

COVERAGE RECOMMENDATION:
[Clear recommendation with confidence level]

SUPPORTING EVIDENCE:
[Key factors in business language]

NEXT STEPS:
[Recommended actions based on confidence level]
'''


def get_validation_prompts() -> Dict[str, str]:
    """
    Get validation prompts for quality assurance.
    
    Returns:
        Dict[str, str]: Validation prompt templates
    """
    return {
        'consistency_check': '''
Review these building coverage determinations for consistency:

Claim 1: {claim_1_text}
Determination 1: {determination_1}

Claim 2: {claim_2_text}
Determination 2: {determination_2}

Are these determinations consistent given similar damage patterns? Explain any differences.

CONSISTENCY ASSESSMENT: [Consistent / Inconsistent]
EXPLANATION: [Reasoning for assessment]
RECOMMENDATIONS: [Any adjustments needed]
''',

        'accuracy_validation': '''
Validate this building coverage determination against known outcome:

Claim Description: {claim_text}
Predicted: {predicted_coverage}
Confidence: {predicted_confidence}
Actual Outcome: {actual_coverage}

Analyze the accuracy and identify any learning opportunities.

ACCURACY: [Correct / Incorrect]
ANALYSIS: [Why prediction was right/wrong]
LEARNING: [How to improve similar predictions]
''',

        'edge_case_analysis': '''
This claim has been flagged as a potential edge case for building coverage:

Claim: {claim_text}
Flag Reason: {flag_reason}
Initial Analysis: {initial_analysis}

Provide specialized analysis for this edge case.

EDGE CASE ASSESSMENT:
[Detailed analysis of unusual aspects]

RECOMMENDED APPROACH:
[How to handle this specific situation]

PRECEDENT CONSIDERATIONS:
[Impact on future similar cases]
'''
    }


def format_prompt(template: str, **kwargs) -> str:
    """
    Format a prompt template with provided parameters.
    
    Args:
        template (str): Prompt template with placeholders
        **kwargs: Parameters to substitute in template
        
    Returns:
        str: Formatted prompt
    """
    try:
        return template.format(**kwargs)
    except KeyError as e:
        missing_key = str(e).strip("'")
        raise ValueError(f"Missing required parameter for prompt: {missing_key}")


def get_prompt_by_name(prompt_name: str) -> str:
    """
    Get a specific prompt by name.
    
    Args:
        prompt_name (str): Name of the prompt to retrieve
    
    Returns:
        str: Prompt template
    
    Raises:
        KeyError: If prompt name is not found
    """
    all_prompts = get_gpt_prompts()
    
    # Check main prompts
    if prompt_name in all_prompts:
        return all_prompts[prompt_name]
    
    # Check validation prompts
    if prompt_name in all_prompts.get('validation_prompts', {}):
        return all_prompts['validation_prompts'][prompt_name]
    
    raise KeyError(f"Prompt '{prompt_name}' not found")


def create_custom_prompt(
    base_template: str,
    additional_instructions: str,
    custom_examples: Optional[List[str]] = None
) -> str:
    """
    Create a custom prompt by extending a base template.
    
    Args:
        base_template (str): Base prompt template
        additional_instructions (str): Additional instructions to add
        custom_examples (Optional[List[str]]): Custom examples to include
        
    Returns:
        str: Enhanced prompt template
    """
    enhanced_prompt = base_template
    
    # Add custom instructions
    enhanced_prompt += f"\n\n**ADDITIONAL INSTRUCTIONS:**\n{additional_instructions}"
    
    # Add custom examples if provided
    if custom_examples:
        enhanced_prompt += "\n\n**EXAMPLES:**\n"
        for i, example in enumerate(custom_examples, 1):
            enhanced_prompt += f"{i}. {example}\n"
    
    return enhanced_prompt


def validate_prompt_parameters(prompt_template: str) -> List[str]:
    """
    Extract required parameters from a prompt template.
    
    Args:
        prompt_template (str): Prompt template to analyze
        
    Returns:
        List[str]: List of required parameter names
    """
    import re
    
    # Find all {parameter} patterns
    parameters = re.findall(r'\{([^}]+)\}', prompt_template)
    
    return list(set(parameters))  # Remove duplicates


def get_prompt_metadata() -> Dict[str, Dict[str, Any]]:
    """
    Get metadata about all available prompts.
    
    Returns:
        Dict[str, Dict[str, Any]]: Prompt metadata
    """
    prompts = get_gpt_prompts()
    metadata = {}
    
    for prompt_name, prompt_template in prompts.items():
        if isinstance(prompt_template, dict):
            # Handle nested prompts (like validation_prompts)
            for sub_name, sub_template in prompt_template.items():
                full_name = f"{prompt_name}.{sub_name}"
                metadata[full_name] = {
                    'parameters': validate_prompt_parameters(sub_template),
                    'length': len(sub_template),
                    'category': prompt_name
                }
        else:
            metadata[prompt_name] = {
                'parameters': validate_prompt_parameters(prompt_template),
                'length': len(prompt_template),
                'category': 'main'
            }
    
    return metadata