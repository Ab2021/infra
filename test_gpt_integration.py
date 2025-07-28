"""
Test script for GPT-4o-mini integration
Validates the API wrapper and basic functionality
"""

import asyncio
import os
from gpt_api_wrapper import GptApiWrapper


async def test_basic_functionality():
    """Test basic GPT-4o-mini functionality"""
    
    print("Testing GPT-4o-mini API Wrapper...")
    
    # Initialize API wrapper
    api = GptApiWrapper()
    
    # Test 1: Basic text generation
    print("\n1. Testing basic text generation...")
    try:
        response = api.generate_content(
            prompt="What is building coverage analysis?",
            temperature=0.1,
            max_tokens=100
        )
        print(f"✅ Basic generation: {response[:100]}...")
    except Exception as e:
        print(f"❌ Basic generation failed: {e}")
    
    # Test 2: JSON generation
    print("\n2. Testing JSON generation...")
    try:
        json_response = api.generate_json_content(
            prompt="Create a JSON object with building damage types as keys and boolean values",
            temperature=0.1,
            max_tokens=200
        )
        print(f"✅ JSON generation: {json_response}")
    except Exception as e:
        print(f"❌ JSON generation failed: {e}")
    
    # Test 3: Async functionality
    print("\n3. Testing async functionality...")
    try:
        async_response = await api.generate_content_async(
            prompt="List 3 types of building damage",
            temperature=0.1,
            max_tokens=100
        )
        print(f"✅ Async generation: {async_response[:100]}...")
    except Exception as e:
        print(f"❌ Async generation failed: {e}")
    
    # Test 4: Async JSON functionality
    print("\n4. Testing async JSON functionality...")
    try:
        async_json_response = await api.generate_json_content_async(
            prompt="Create a JSON object with 'damage_type' and 'severity' keys for roof damage",
            temperature=0.1,
            max_tokens=150
        )
        print(f"✅ Async JSON generation: {async_json_response}")
    except Exception as e:
        print(f"❌ Async JSON generation failed: {e}")


async def test_building_coverage_scenario():
    """Test with building coverage specific scenario"""
    
    print("\n\n=== Testing Building Coverage Scenario ===")
    
    api = GptApiWrapper()
    
    # Sample claim text
    sample_claim = """
    House fire on January 15, 2024. Significant damage to roof, exterior walls, and interior rooms.
    Water damage from firefighting efforts affected flooring and walls. 
    Initial estimate from contractor was $45,000. Adjuster assessment indicates $52,000 in damages.
    Property is currently unoccupiable due to extensive smoke and water damage.
    This is the primary residence structure on the property.
    """
    
    # Test indicator extraction prompt
    indicators_prompt = f"""
    BUILDING COVERAGE ANALYSIS - INDICATOR EXTRACTION TEST

    Text to Analyze:
    {sample_claim}

    Extract the following indicators with Y/N values:
    1. BLDG_ROOF_DMG: Look for roof damage mentions
    2. BLDG_EXTERIOR_DMG: Look for exterior damage mentions  
    3. BLDG_INTERIOR_DMG: Look for interior damage mentions
    4. BLDG_WATER_DMG: Look for water damage mentions
    5. BLDG_FIRE_DMG: Look for fire damage mentions
    6. BLDG_UNOCCUPIABLE: Look for occupancy status mentions

    OUTPUT FORMAT (JSON):
    {{
        "indicators": {{
            "BLDG_ROOF_DMG": {{"value": "Y/N", "confidence": 0.0-1.0, "evidence": "exact text"}},
            "BLDG_EXTERIOR_DMG": {{"value": "Y/N", "confidence": 0.0-1.0, "evidence": "exact text"}},
            "BLDG_INTERIOR_DMG": {{"value": "Y/N", "confidence": 0.0-1.0, "evidence": "exact text"}},
            "BLDG_WATER_DMG": {{"value": "Y/N", "confidence": 0.0-1.0, "evidence": "exact text"}},
            "BLDG_FIRE_DMG": {{"value": "Y/N", "confidence": 0.0-1.0, "evidence": "exact text"}},
            "BLDG_UNOCCUPIABLE": {{"value": "Y/N", "confidence": 0.0-1.0, "evidence": "exact text"}}
        }}
    }}
    """
    
    try:
        result = await api.generate_json_content_async(
            prompt=indicators_prompt,
            temperature=0.1,
            max_tokens=1000,
            system_message="You are an expert building damage assessor. Extract indicators with Y/N values, confidence scores, and evidence. Respond with valid JSON only."
        )
        
        print("✅ Building coverage extraction successful:")
        print(f"Result: {result}")
        
        # Validate structure
        if "indicators" in result:
            indicators = result["indicators"]
            for indicator_name, indicator_data in indicators.items():
                if "value" in indicator_data and "confidence" in indicator_data:
                    print(f"  ✅ {indicator_name}: {indicator_data['value']} (conf: {indicator_data['confidence']})")
                else:
                    print(f"  ❌ {indicator_name}: Missing required fields")
        else:
            print("❌ Missing 'indicators' key in response")
            
    except Exception as e:
        print(f"❌ Building coverage extraction failed: {e}")


def main():
    """Main test function"""
    
    # Check if API key is set
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY environment variable not set")
        print("Please set your OpenAI API key: export OPENAI_API_KEY='your-key-here'")
        return
    
    print("🚀 Starting GPT-4o-mini Integration Tests...")
    
    # Run tests
    asyncio.run(test_basic_functionality())
    asyncio.run(test_building_coverage_scenario())
    
    print("\n✅ All tests completed!")


if __name__ == "__main__":
    main()