"""
SQLite-based Memory Store for Agentic Building Coverage Analysis
Replaces Redis and diskcache with lightweight SQLite database
"""

import sqlite3
import json
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import threading
from contextlib import contextmanager
import os


class SQLiteMemoryStore:
    """SQLite-based memory management for agentic operations"""
    
    def __init__(self, db_path: str = "agentic_memory.db", max_history: int = 500):
        """Initialize SQLite memory store"""
        self.db_path = db_path
        self.max_history = max_history
        self._lock = threading.Lock()
        self._init_database()
    
    def _init_database(self):
        """Initialize database tables"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Extraction history table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS extraction_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    claim_id TEXT,
                    claim_text_hash TEXT,
                    original_text TEXT,
                    extraction_results TEXT,
                    success_metrics TEXT,
                    timestamp TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Successful patterns table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS successful_patterns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    pattern_type TEXT,
                    pattern_data TEXT,
                    success_count INTEGER DEFAULT 1,
                    confidence_score REAL,
                    last_used DATETIME DEFAULT CURRENT_TIMESTAMP,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Calculation patterns table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS calculation_patterns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    feature_context TEXT,
                    calculation_result TEXT,
                    validation_result TEXT,
                    accuracy_score REAL,
                    timestamp TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Confidence calibration table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS confidence_calibration (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    indicator_type TEXT,
                    predicted_confidence REAL,
                    actual_accuracy REAL,
                    adjustment_factor REAL,
                    sample_count INTEGER DEFAULT 1,
                    last_updated DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Similarity index table for fast lookups
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS similarity_index (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    text_hash TEXT UNIQUE,
                    keywords TEXT,
                    damage_indicators TEXT,
                    operational_status TEXT,
                    similarity_vector TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create indexes for performance
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_extraction_timestamp ON extraction_history(timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_extraction_hash ON extraction_history(claim_text_hash)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_patterns_type ON successful_patterns(pattern_type)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_similarity_hash ON similarity_index(text_hash)")
            
            conn.commit()
    
    @contextmanager
    def _get_connection(self):
        """Get database connection with proper handling"""
        conn = sqlite3.connect(self.db_path, timeout=10.0)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()
    
    def _calculate_text_hash(self, text: str) -> str:
        """Calculate hash for text similarity"""
        return hashlib.md5(text.lower().encode()).hexdigest()
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate text similarity using word overlap"""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0
    
    def find_similar_claims(self, claim_text: str, limit: int = 5) -> List[Dict]:
        """Find similar historical claims"""
        with self._lock:
            similar_claims = []
            
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                # Get recent extraction history
                cursor.execute("""
                    SELECT original_text, extraction_results, success_metrics, timestamp
                    FROM extraction_history 
                    ORDER BY created_at DESC 
                    LIMIT 100
                """)
                
                rows = cursor.fetchall()
                
                for row in rows:
                    historical_text = row['original_text'] or ""
                    similarity_score = self._calculate_text_similarity(claim_text, historical_text)
                    
                    if similarity_score > 0.7:
                        try:
                            extraction_results = json.loads(row['extraction_results'] or '{}')
                            success_metrics = json.loads(row['success_metrics'] or '{}')
                        except json.JSONDecodeError:
                            continue
                        
                        similar_claims.append({
                            "historical_claim": {
                                "original_text": historical_text,
                                "extraction_results": extraction_results,
                                "timestamp": row['timestamp']
                            },
                            "similarity_score": similarity_score,
                            "success_indicators": success_metrics
                        })
                
                # Sort by similarity and return top results
                similar_claims.sort(key=lambda x: x["similarity_score"], reverse=True)
                return similar_claims[:limit]
    
    def store_extraction_result(self, extraction_data: Dict):
        """Store successful extraction for learning"""
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                # Prepare data
                claim_text = extraction_data.get("source_text", "")
                text_hash = self._calculate_text_hash(claim_text)
                success_metrics = self._calculate_success_metrics(extraction_data)
                
                # Insert extraction result
                cursor.execute("""
                    INSERT INTO extraction_history 
                    (claim_id, claim_text_hash, original_text, extraction_results, success_metrics, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    extraction_data.get("claim_id", ""),
                    text_hash,
                    claim_text,
                    json.dumps(extraction_data),
                    json.dumps(success_metrics),
                    datetime.now().isoformat()
                ))
                
                # Update similarity index
                self._update_similarity_index(cursor, text_hash, claim_text, extraction_data)
                
                # Clean old records if needed
                self._cleanup_old_records(cursor)
                
                conn.commit()
    
    def _update_similarity_index(self, cursor, text_hash: str, claim_text: str, extraction_data: Dict):
        """Update similarity index for fast lookups"""
        
        # Extract keywords and indicators
        keywords = self._extract_keywords(claim_text)
        damage_indicators = self._extract_damage_indicators(extraction_data)
        operational_status = self._extract_operational_status(extraction_data)
        
        # Create similarity vector (simplified)
        similarity_vector = {
            "keyword_count": len(keywords),
            "damage_count": len(damage_indicators),
            "has_operational": len(operational_status) > 0
        }
        
        cursor.execute("""
            INSERT OR REPLACE INTO similarity_index 
            (text_hash, keywords, damage_indicators, operational_status, similarity_vector)
            VALUES (?, ?, ?, ?, ?)
        """, (
            text_hash,
            json.dumps(keywords),
            json.dumps(damage_indicators),
            json.dumps(operational_status),
            json.dumps(similarity_vector)
        ))
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract relevant keywords from text"""
        damage_keywords = [
            "fire", "water", "roof", "exterior", "interior", "electrical", 
            "plumbing", "structural", "foundation", "damage", "destroyed"
        ]
        
        found_keywords = []
        text_lower = text.lower()
        for keyword in damage_keywords:
            if keyword in text_lower:
                found_keywords.append(keyword)
        
        return found_keywords
    
    def _extract_damage_indicators(self, extraction_data: Dict) -> List[str]:
        """Extract damage indicators that are 'Y'"""
        damage_indicators = []
        
        for key, value in extraction_data.items():
            if key.startswith("BLDG_") and key.endswith("_DMG"):
                if isinstance(value, dict) and value.get("value") == "Y":
                    damage_indicators.append(key)
        
        return damage_indicators
    
    def _extract_operational_status(self, extraction_data: Dict) -> List[str]:
        """Extract operational status indicators"""
        operational_keys = ["BLDG_TENABLE", "BLDG_UNOCCUPIABLE", "BLDG_COMPLETE_LOSS"]
        operational_status = []
        
        for key in operational_keys:
            if key in extraction_data:
                value = extraction_data[key]
                if isinstance(value, dict) and value.get("value") == "Y":
                    operational_status.append(key)
        
        return operational_status
    
    def _cleanup_old_records(self, cursor):
        """Clean up old records to maintain performance"""
        
        # Keep only the most recent records
        cursor.execute("""
            DELETE FROM extraction_history 
            WHERE id NOT IN (
                SELECT id FROM extraction_history 
                ORDER BY created_at DESC 
                LIMIT ?
            )
        """, (self.max_history,))
        
        # Clean old calculation patterns (keep last 200)
        cursor.execute("""
            DELETE FROM calculation_patterns 
            WHERE id NOT IN (
                SELECT id FROM calculation_patterns 
                ORDER BY created_at DESC 
                LIMIT 200
            )
        """)
        
        # Clean old similarity index (keep last 300)
        cursor.execute("""
            DELETE FROM similarity_index 
            WHERE id NOT IN (
                SELECT id FROM similarity_index 
                ORDER BY created_at DESC 
                LIMIT 300
            )
        """)
    
    def find_similar_calculation_patterns(self, feature_context: Dict) -> List[Dict]:
        """Find similar calculation patterns"""
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT feature_context, calculation_result, validation_result, accuracy_score
                    FROM calculation_patterns 
                    ORDER BY created_at DESC 
                    LIMIT 50
                """)
                
                patterns = []
                rows = cursor.fetchall()
                
                for row in rows:
                    try:
                        stored_context = json.loads(row['feature_context'] or '{}')
                        if self._patterns_match(stored_context, feature_context):
                            patterns.append({
                                "feature_context": stored_context,
                                "calculation_result": json.loads(row['calculation_result'] or '{}'),
                                "validation_result": json.loads(row['validation_result'] or '{}'),
                                "accuracy_score": row['accuracy_score']
                            })
                    except json.JSONDecodeError:
                        continue
                
                return patterns[:5]  # Return top 5 matches
    
    def store_calculation_pattern(self, feature_context: Dict, calculation_result: Dict, validation_result: Dict):
        """Store calculation pattern for learning"""
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                accuracy_score = validation_result.get("validation_score", 0.8)
                
                cursor.execute("""
                    INSERT INTO calculation_patterns 
                    (feature_context, calculation_result, validation_result, accuracy_score, timestamp)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    json.dumps(feature_context),
                    json.dumps(calculation_result),
                    json.dumps(validation_result),
                    accuracy_score,
                    datetime.now().isoformat()
                ))
                
                conn.commit()
    
    def _patterns_match(self, pattern1: Dict, pattern2: Dict) -> bool:
        """Check if patterns match (simplified similarity)"""
        # Check damage severity similarity
        damage1 = pattern1.get("feature_analysis", {}).get("damage_severity", {})
        damage2 = pattern2.get("feature_analysis", {}).get("damage_severity", {})
        
        if damage1.get("severity_level") == damage2.get("severity_level"):
            return True
        
        # Check operational impact similarity
        op1 = pattern1.get("feature_analysis", {}).get("operational_impact", {})
        op2 = pattern2.get("feature_analysis", {}).get("operational_impact", {})
        
        if op1.get("impact_level") == op2.get("impact_level"):
            return True
        
        return False
    
    def _calculate_success_metrics(self, extraction_data: Dict) -> Dict:
        """Calculate success metrics for extraction"""
        total_indicators = 0
        confident_indicators = 0
        
        for key, value in extraction_data.items():
            if key.startswith("BLDG_") and isinstance(value, dict):
                total_indicators += 1
                confidence = value.get("confidence", 0)
                if confidence >= 0.7:
                    confident_indicators += 1
        
        return {
            "confidence_avg": confident_indicators / max(total_indicators, 1),
            "validation_passed": True,
            "completeness": min(1.0, total_indicators / 21.0)
        }
    
    def get_confidence_calibration(self, indicator_type: str) -> float:
        """Get confidence calibration factor for indicator type"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT adjustment_factor 
                FROM confidence_calibration 
                WHERE indicator_type = ?
                ORDER BY last_updated DESC 
                LIMIT 1
            """, (indicator_type,))
            
            row = cursor.fetchone()
            return row['adjustment_factor'] if row else 1.0
    
    def update_confidence_calibration(self, indicator_type: str, predicted: float, actual: float):
        """Update confidence calibration based on actual results"""
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                adjustment_factor = actual / max(predicted, 0.1)
                
                cursor.execute("""
                    INSERT OR REPLACE INTO confidence_calibration 
                    (indicator_type, predicted_confidence, actual_accuracy, adjustment_factor)
                    VALUES (?, ?, ?, ?)
                """, (indicator_type, predicted, actual, adjustment_factor))
                
                conn.commit()
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory store statistics"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Count records in each table
            stats = {}
            
            cursor.execute("SELECT COUNT(*) as count FROM extraction_history")
            stats["extraction_history_count"] = cursor.fetchone()["count"]
            
            cursor.execute("SELECT COUNT(*) as count FROM calculation_patterns")
            stats["calculation_patterns_count"] = cursor.fetchone()["count"]
            
            cursor.execute("SELECT COUNT(*) as count FROM similarity_index")
            stats["similarity_index_count"] = cursor.fetchone()["count"]
            
            cursor.execute("SELECT COUNT(*) as count FROM confidence_calibration")
            stats["confidence_calibration_count"] = cursor.fetchone()["count"]
            
            # Database file size
            if os.path.exists(self.db_path):
                stats["db_size_mb"] = os.path.getsize(self.db_path) / (1024 * 1024)
            else:
                stats["db_size_mb"] = 0
            
            return stats
    
    def cleanup_database(self):
        """Perform database cleanup and optimization"""
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                # Clean old records
                self._cleanup_old_records(cursor)
                
                # Vacuum database to reclaim space
                cursor.execute("VACUUM")
                
                conn.commit()
    
    def close(self):
        """Close database connections and cleanup"""
        # SQLite connections are closed automatically with context managers
        pass