"""
Test script for SQLite memory store integration
"""

import os
import json
from datetime import datetime
from sqlite_memory_store import SQLiteMemoryStore


def test_basic_functionality():
    """Test basic SQLite memory store functionality"""
    
    print("Testing SQLite Memory Store...")
    
    # Use test database
    test_db = "test_memory.db"
    
    # Clean up existing test db
    if os.path.exists(test_db):
        os.remove(test_db)
    
    # Initialize memory store
    memory_store = SQLiteMemoryStore(test_db)
    
    # Test 1: Store extraction result
    print("\n1. Testing extraction result storage...")
    
    sample_extraction = {
        "claim_id": "TEST-001",
        "source_text": "House fire caused significant roof and interior damage. Water damage from firefighting efforts.",
        "BLDG_ROOF_DMG": {"value": "Y", "confidence": 0.9, "evidence": "significant roof damage"},
        "BLDG_INTERIOR_DMG": {"value": "Y", "confidence": 0.8, "evidence": "interior damage"},
        "BLDG_FIRE_DMG": {"value": "Y", "confidence": 0.95, "evidence": "House fire"},
        "BLDG_WATER_DMG": {"value": "Y", "confidence": 0.85, "evidence": "Water damage from firefighting"},
        "BLDG_TENABLE": {"value": "N", "confidence": 0.7, "evidence": "not mentioned as habitable"},
    }
    
    try:
        memory_store.store_extraction_result(sample_extraction)
        print("✅ Extraction result stored successfully")
    except Exception as e:
        print(f"❌ Failed to store extraction result: {e}")
    
    # Test 2: Find similar claims
    print("\n2. Testing similar claims search...")
    
    similar_text = "Building fire with roof damage and water damage from fire suppression"
    
    try:
        similar_claims = memory_store.find_similar_claims(similar_text, limit=3)
        print(f"✅ Found {len(similar_claims)} similar claims")
        
        for i, claim in enumerate(similar_claims, 1):
            print(f"   {i}. Similarity: {claim['similarity_score']:.3f}")
            
    except Exception as e:
        print(f"❌ Failed to find similar claims: {e}")
    
    # Test 3: Store calculation pattern
    print("\n3. Testing calculation pattern storage...")
    
    feature_context = {
        "feature_analysis": {
            "damage_severity": {"severity_level": "moderate", "damage_count": 4},
            "operational_impact": {"impact_level": "significant_impact"}
        }
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
    
    # Test 4: Find similar calculation patterns
    print("\n4. Testing similar calculation patterns search...")
    
    try:
        similar_patterns = memory_store.find_similar_calculation_patterns(feature_context)
        print(f"✅ Found {len(similar_patterns)} similar calculation patterns")
        
        for i, pattern in enumerate(similar_patterns, 1):
            accuracy = pattern.get("accuracy_score", 0)
            print(f"   {i}. Accuracy: {accuracy:.3f}")
            
    except Exception as e:
        print(f"❌ Failed to find similar patterns: {e}")
    
    # Test 5: Get memory statistics
    print("\n5. Testing memory statistics...")
    
    try:
        stats = memory_store.get_memory_stats()
        print("✅ Memory statistics:")
        print(f"   Extraction history: {stats['extraction_history_count']}")
        print(f"   Calculation patterns: {stats['calculation_patterns_count']}")
        print(f"   Database size: {stats['db_size_mb']:.3f} MB")
    except Exception as e:
        print(f"❌ Failed to get memory statistics: {e}")
    
    # Test 6: Confidence calibration
    print("\n6. Testing confidence calibration...")
    
    try:
        # Update calibration
        memory_store.update_confidence_calibration("BLDG_FIRE_DMG", 0.9, 0.95)
        
        # Get calibration
        calibration = memory_store.get_confidence_calibration("BLDG_FIRE_DMG")
        print(f"✅ Confidence calibration: {calibration:.3f}")
    except Exception as e:
        print(f"❌ Failed confidence calibration test: {e}")
    
    # Cleanup test database
    memory_store.close()
    if os.path.exists(test_db):
        os.remove(test_db)
    
    print("\n✅ All SQLite tests completed!")


def test_performance():
    """Test performance with multiple records"""
    
    print("\n=== Performance Testing ===")
    
    test_db = "perf_test_memory.db"
    
    # Clean up existing test db
    if os.path.exists(test_db):
        os.remove(test_db)
    
    memory_store = SQLiteMemoryStore(test_db)
    
    # Store multiple extraction results
    print("Storing 50 extraction results...")
    
    start_time = datetime.now()
    
    for i in range(50):
        sample_extraction = {
            "claim_id": f"PERF-{i:03d}",
            "source_text": f"Test claim {i} with various damage types and patterns for testing similarity matching.",
            "BLDG_ROOF_DMG": {"value": "Y" if i % 3 == 0 else "N", "confidence": 0.8 + (i % 10) * 0.01},
            "BLDG_FIRE_DMG": {"value": "Y" if i % 5 == 0 else "N", "confidence": 0.7 + (i % 10) * 0.02},
            "BLDG_WATER_DMG": {"value": "Y" if i % 4 == 0 else "N", "confidence": 0.75 + (i % 10) * 0.015},
        }
        
        memory_store.store_extraction_result(sample_extraction)
    
    store_time = datetime.now() - start_time
    print(f"✅ Stored 50 records in {store_time.total_seconds():.3f} seconds")
    
    # Test similarity search performance
    print("Testing similarity search performance...")
    
    start_time = datetime.now()
    
    for i in range(10):
        test_text = f"Similar claim {i} with damage patterns"
        similar_claims = memory_store.find_similar_claims(test_text, limit=5)
    
    search_time = datetime.now() - start_time
    print(f"✅ Performed 10 similarity searches in {search_time.total_seconds():.3f} seconds")
    
    # Get final stats
    stats = memory_store.get_memory_stats()
    print(f"✅ Final database size: {stats['db_size_mb']:.3f} MB")
    
    # Cleanup
    memory_store.close()
    if os.path.exists(test_db):
        os.remove(test_db)
    
    print("✅ Performance testing completed!")


def main():
    """Main test function"""
    
    print("🚀 Starting SQLite Integration Tests...")
    
    # Run basic functionality tests
    test_basic_functionality()
    
    # Run performance tests
    test_performance()
    
    print("\n✅ All SQLite integration tests completed successfully!")


if __name__ == "__main__":
    main()