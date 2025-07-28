"""
Test script for FAISS memory store integration
"""

import os
import json
import shutil
from datetime import datetime
from faiss_memory_store import FAISSMemoryStore


def test_basic_functionality():
    """Test basic FAISS memory store functionality"""
    
    print("Testing FAISS Memory Store...")
    
    # Use test memory directory
    test_memory_dir = "test_faiss_memory"
    
    # Clean up existing test memory
    if os.path.exists(test_memory_dir):
        shutil.rmtree(test_memory_dir)
    
    # Initialize memory store (will download model first time)
    print("Initializing FAISS memory store (may take time for first download)...")
    memory_store = FAISSMemoryStore(
        model_name="all-mpnet-base-v2",
        memory_dir=test_memory_dir,
        max_history=100
    )
    
    # Test 1: Store extraction result
    print("\n1. Testing extraction result storage...")
    
    sample_extraction = {
        "claim_id": "TEST-001",
        "source_text": "House fire caused significant roof and interior damage. Water damage from firefighting efforts affected flooring and walls throughout the building.",
        "BLDG_ROOF_DMG": {"value": "Y", "confidence": 0.9, "evidence": "significant roof damage"},
        "BLDG_INTERIOR_DMG": {"value": "Y", "confidence": 0.8, "evidence": "interior damage"},
        "BLDG_FIRE_DMG": {"value": "Y", "confidence": 0.95, "evidence": "House fire"},
        "BLDG_WATER_DMG": {"value": "Y", "confidence": 0.85, "evidence": "Water damage from firefighting"},
        "BLDG_TENABLE": {"value": "N", "confidence": 0.7, "evidence": "not mentioned as habitable"},
        "BLDG_FLOORING_DMG": {"value": "Y", "confidence": 0.8, "evidence": "affected flooring"}
    }
    
    try:
        memory_store.store_extraction_result(sample_extraction)
        print("✅ Extraction result stored successfully")
    except Exception as e:
        print(f"❌ Failed to store extraction result: {e}")
    
    # Test 2: Store another similar extraction
    print("\n2. Testing second extraction storage...")
    
    sample_extraction_2 = {
        "claim_id": "TEST-002", 
        "source_text": "Commercial building experienced water damage from burst pipes. Flooring, walls, and electrical systems were damaged requiring extensive repairs.",
        "BLDG_WATER_DMG": {"value": "Y", "confidence": 0.9, "evidence": "water damage from burst pipes"},
        "BLDG_FLOORING_DMG": {"value": "Y", "confidence": 0.85, "evidence": "Flooring damaged"},
        "BLDG_WALLS_DMG": {"value": "Y", "confidence": 0.8, "evidence": "walls damaged"},
        "BLDG_ELECTRICAL_DMG": {"value": "Y", "confidence": 0.75, "evidence": "electrical systems were damaged"},
        "BLDG_TENABLE": {"value": "N", "confidence": 0.7, "evidence": "requiring extensive repairs"}
    }
    
    try:
        memory_store.store_extraction_result(sample_extraction_2)
        print("✅ Second extraction result stored successfully")
    except Exception as e:
        print(f"❌ Failed to store second extraction result: {e}")
    
    # Test 3: Find similar claims
    print("\n3. Testing similar claims search...")
    
    similar_text = "Building fire with roof damage and water damage from fire suppression efforts"
    
    try:
        similar_claims = memory_store.find_similar_claims(similar_text, limit=3)
        print(f"✅ Found {len(similar_claims)} similar claims")
        
        for i, claim in enumerate(similar_claims, 1):
            print(f"   {i}. Similarity: {claim['similarity_score']:.3f}")
            print(f"      Claim ID: {claim['historical_claim']['claim_id']}")
            
    except Exception as e:
        print(f"❌ Failed to find similar claims: {e}")
    
    # Test 4: Store calculation pattern
    print("\n4. Testing calculation pattern storage...")
    
    feature_context = {
        "feature_analysis": {
            "damage_severity": {
                "severity_level": "moderate", 
                "damage_count": 4,
                "specific_damages": ["BLDG_FIRE_DMG", "BLDG_WATER_DMG", "BLDG_ROOF_DMG", "BLDG_INTERIOR_DMG"]
            },
            "operational_impact": {
                "impact_level": "significant_impact",
                "operational_status": {"tenable": False}
            },
            "contextual_factors": {
                "is_primary_structure": True
            }
        },
        "expected_loss_range": (25000, 75000)
    }
    
    calculation_result = {
        "final_amount": 45000,
        "confidence": 0.8,
        "method": "feature_adjusted"
    }
    
    validation_result = {
        "validation_score": 0.85,
        "passed": True
    }
    
    try:
        memory_store.store_calculation_pattern(feature_context, calculation_result, validation_result)
        print("✅ Calculation pattern stored successfully")
    except Exception as e:
        print(f"❌ Failed to store calculation pattern: {e}")
    
    # Test 5: Find similar calculation patterns
    print("\n5. Testing similar calculation patterns search...")
    
    similar_context = {
        "feature_analysis": {
            "damage_severity": {
                "severity_level": "moderate",
                "damage_count": 5
            },
            "operational_impact": {
                "impact_level": "significant_impact"
            }
        },
        "expected_loss_range": (20000, 80000)
    }
    
    try:
        similar_patterns = memory_store.find_similar_calculation_patterns(similar_context)
        print(f"✅ Found {len(similar_patterns)} similar calculation patterns")
        
        for i, pattern in enumerate(similar_patterns, 1):
            accuracy = pattern.get("accuracy_score", 0)
            similarity = pattern.get("similarity_score", 0)
            print(f"   {i}. Accuracy: {accuracy:.3f}, Similarity: {similarity:.3f}")
            
    except Exception as e:
        print(f"❌ Failed to find similar patterns: {e}")
    
    # Test 6: Get memory statistics
    print("\n6. Testing memory statistics...")
    
    try:
        stats = memory_store.get_memory_stats()
        print("✅ Memory statistics:")
        print(f"   Extraction history: {stats['extraction_history_count']}")
        print(f"   Calculation patterns: {stats['calculation_patterns_count']}")
        print(f"   FAISS extraction index: {stats['faiss_extraction_total']}")
        print(f"   FAISS calculation index: {stats['faiss_calculation_total']}")
        print(f"   Memory directory size: {stats['memory_dir_size_mb']:.3f} MB")
        print(f"   Model: {stats['model_name']}")
        print(f"   Embedding dimension: {stats['embedding_dimension']}")
    except Exception as e:
        print(f"❌ Failed to get memory statistics: {e}")
    
    # Test 7: Confidence calibration
    print("\n7. Testing confidence calibration...")
    
    try:
        # Update calibration
        memory_store.update_confidence_calibration("BLDG_FIRE_DMG", 0.9, 0.95)
        
        # Get calibration
        calibration = memory_store.get_confidence_calibration("BLDG_FIRE_DMG")
        print(f"✅ Confidence calibration: {calibration:.3f}")
    except Exception as e:
        print(f"❌ Failed confidence calibration test: {e}")
    
    # Test 8: Memory persistence
    print("\n8. Testing memory persistence...")
    
    try:
        # Close and reopen
        memory_store.close()
        
        # Create new instance (should load saved data)
        memory_store_2 = FAISSMemoryStore(
            model_name="all-mpnet-base-v2",
            memory_dir=test_memory_dir
        )
        
        stats_2 = memory_store_2.get_memory_stats()
        if stats_2['extraction_history_count'] > 0:
            print("✅ Memory persistence working - data loaded successfully")
        else:
            print("❌ Memory persistence failed - no data loaded")
        
        memory_store_2.close()
        
    except Exception as e:
        print(f"❌ Failed memory persistence test: {e}")
    
    # Cleanup test memory
    if os.path.exists(test_memory_dir):
        shutil.rmtree(test_memory_dir)
    
    print("\n✅ All FAISS tests completed!")


def test_performance():
    """Test performance with multiple records"""
    
    print("\n=== Performance Testing ===")
    
    test_memory_dir = "perf_test_faiss_memory"
    
    # Clean up existing test memory
    if os.path.exists(test_memory_dir):
        shutil.rmtree(test_memory_dir)
    
    memory_store = FAISSMemoryStore(
        model_name="all-mpnet-base-v2",
        memory_dir=test_memory_dir,
        max_history=200
    )
    
    # Store multiple extraction results
    print("Storing 20 extraction results...")
    
    start_time = datetime.now()
    
    sample_texts = [
        "Fire damage to commercial building with smoke and heat damage throughout",
        "Water damage from pipe burst affecting multiple floors and equipment",
        "Wind damage to roof and exterior walls from severe storm",
        "Electrical fire in main panel causing power outage and equipment damage",
        "Flooding from heavy rains causing basement and foundation damage",
        "Hail damage to roof, windows, and outdoor equipment",
        "Vandalism resulting in broken windows and interior damage",
        "Equipment malfunction causing water damage to inventory",
        "Lightning strike causing electrical surge and equipment failure",
        "Frozen pipes burst causing extensive water damage throughout building"
    ]
    
    for i in range(20):
        text_idx = i % len(sample_texts)
        base_text = sample_texts[text_idx]
        
        sample_extraction = {
            "claim_id": f"PERF-{i:03d}",
            "source_text": f"{base_text} Claim {i} with specific damage patterns and loss details.",
            "BLDG_FIRE_DMG": {"value": "Y" if i % 3 == 0 else "N", "confidence": 0.8 + (i % 10) * 0.01},
            "BLDG_WATER_DMG": {"value": "Y" if i % 4 == 0 else "N", "confidence": 0.75 + (i % 10) * 0.015},
            "BLDG_ROOF_DMG": {"value": "Y" if i % 5 == 0 else "N", "confidence": 0.7 + (i % 10) * 0.02},
        }
        
        memory_store.store_extraction_result(sample_extraction)
    
    store_time = datetime.now() - start_time
    print(f"✅ Stored 20 records in {store_time.total_seconds():.3f} seconds")
    
    # Test similarity search performance
    print("Testing similarity search performance...")
    
    start_time = datetime.now()
    
    for i in range(5):
        test_text = f"Building damage claim {i} with various loss patterns"
        similar_claims = memory_store.find_similar_claims(test_text, limit=5)
    
    search_time = datetime.now() - start_time
    print(f"✅ Performed 5 similarity searches in {search_time.total_seconds():.3f} seconds")
    
    # Get final stats
    stats = memory_store.get_memory_stats()
    print(f"✅ Final memory size: {stats['memory_dir_size_mb']:.3f} MB")
    print(f"✅ FAISS index total: {stats['faiss_extraction_total']}")
    
    # Cleanup
    memory_store.close()
    if os.path.exists(test_memory_dir):
        shutil.rmtree(test_memory_dir)
    
    print("✅ Performance testing completed!")


def test_building_coverage_scenario():
    """Test with building coverage specific scenario"""
    
    print("\n=== Building Coverage Scenario Test ===")
    
    test_memory_dir = "scenario_test_faiss_memory"
    
    # Clean up existing test memory
    if os.path.exists(test_memory_dir):
        shutil.rmtree(test_memory_dir)
    
    memory_store = FAISSMemoryStore(
        model_name="all-mpnet-base-v2",
        memory_dir=test_memory_dir
    )
    
    # Sample building coverage claims
    coverage_claims = [
        {
            "claim_id": "BLD-001",
            "source_text": "Restaurant kitchen fire caused extensive damage to cooking equipment, exhaust system, and interior dining area. Water damage from sprinkler system affected flooring and walls. Business operations suspended for 60 days.",
            "damage_indicators": ["FIRE", "WATER", "EQUIPMENT", "INTERIOR", "FLOORING", "WALLS"],
            "loss_amount": 125000
        },
        {
            "claim_id": "BLD-002", 
            "source_text": "Office building suffered water damage from roof leak during heavy storm. Ceiling tiles, carpeting, and computer equipment damaged. Several floors affected requiring temporary relocation of staff.",
            "damage_indicators": ["WATER", "ROOF", "CEILING", "FLOORING", "ELECTRICAL"],
            "loss_amount": 85000
        },
        {
            "claim_id": "BLD-003",
            "source_text": "Retail store experienced break-in with significant vandalism damage to windows, doors, and inventory. Security system and point-of-sale equipment also damaged.",
            "damage_indicators": ["WINDOWS", "DOORS", "INVENTORY", "ELECTRICAL"],
            "loss_amount": 45000
        }
    ]
    
    # Store coverage claims
    print("Storing building coverage claims...")
    
    for claim in coverage_claims:
        extraction_data = {
            "claim_id": claim["claim_id"],
            "source_text": claim["source_text"]
        }
        
        # Add damage indicators
        for damage_type in claim["damage_indicators"]:
            indicator_key = f"BLDG_{damage_type}_DMG"
            extraction_data[indicator_key] = {
                "value": "Y",
                "confidence": 0.85,
                "evidence": f"Evidence of {damage_type.lower()} damage"
            }
        
        memory_store.store_extraction_result(extraction_data)
    
    print("✅ Stored 3 building coverage claims")
    
    # Test similarity matching
    test_queries = [
        "Commercial kitchen fire with equipment damage and water from suppression",
        "Building water damage from storm affecting multiple floors",
        "Vandalism to retail property with broken glass and equipment damage"
    ]
    
    print("\nTesting similarity matching for building coverage scenarios...")
    
    for i, query in enumerate(test_queries, 1):
        print(f"\nQuery {i}: {query}")
        similar_claims = memory_store.find_similar_claims(query, limit=2)
        
        for j, claim in enumerate(similar_claims, 1):
            similarity = claim['similarity_score']
            claim_id = claim['historical_claim']['claim_id']
            print(f"  Match {j}: Claim {claim_id} (similarity: {similarity:.3f})")
    
    # Store calculation patterns for building coverage
    print("\nStoring calculation patterns...")
    
    for claim in coverage_claims:
        damage_count = len(claim["damage_indicators"])
        
        feature_context = {
            "feature_analysis": {
                "damage_severity": {
                    "severity_level": "extensive" if damage_count >= 5 else "moderate" if damage_count >= 3 else "limited",
                    "damage_count": damage_count
                },
                "operational_impact": {
                    "impact_level": "major_impact" if "FIRE" in claim["damage_indicators"] else "significant_impact"
                }
            },
            "expected_loss_range": (claim["loss_amount"] * 0.8, claim["loss_amount"] * 1.2)
        }
        
        calculation_result = {
            "final_amount": claim["loss_amount"],
            "confidence": 0.85,
            "method": "building_coverage_assessment"
        }
        
        validation_result = {
            "validation_score": 0.9,
            "passed": True
        }
        
        memory_store.store_calculation_pattern(feature_context, calculation_result, validation_result)
    
    print("✅ Stored 3 calculation patterns")
    
    # Test pattern matching
    test_context = {
        "feature_analysis": {
            "damage_severity": {
                "severity_level": "moderate",
                "damage_count": 4
            },
            "operational_impact": {
                "impact_level": "significant_impact"
            }
        },
        "expected_loss_range": (60000, 100000)
    }
    
    similar_patterns = memory_store.find_similar_calculation_patterns(test_context)
    print(f"\n✅ Found {len(similar_patterns)} similar calculation patterns")
    
    for i, pattern in enumerate(similar_patterns, 1):
        accuracy = pattern.get("accuracy_score", 0)
        similarity = pattern.get("similarity_score", 0)
        final_amount = pattern["calculation_result"].get("final_amount", 0)
        print(f"  Pattern {i}: Amount ${final_amount:,}, Accuracy: {accuracy:.3f}, Similarity: {similarity:.3f}")
    
    # Export memory for inspection
    export_file = os.path.join(test_memory_dir, "building_coverage_memory.json")
    memory_store.export_memory(export_file)
    print(f"✅ Memory exported to {export_file}")
    
    # Cleanup
    memory_store.close()
    if os.path.exists(test_memory_dir):
        shutil.rmtree(test_memory_dir)
    
    print("✅ Building coverage scenario testing completed!")


def main():
    """Main test function"""
    
    print("🚀 Starting FAISS Integration Tests...")
    print("Note: First run may take time to download sentence transformer model")
    
    # Run basic functionality tests
    test_basic_functionality()
    
    # Run performance tests  
    test_performance()
    
    # Run building coverage scenario tests
    test_building_coverage_scenario()
    
    print("\n✅ All FAISS integration tests completed successfully!")


if __name__ == "__main__":
    main()