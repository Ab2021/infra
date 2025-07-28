"""
FAISS-based Memory Store for Agentic Building Coverage Analysis
Uses vector similarity matching for efficient claim pattern storage and retrieval
"""

import numpy as np
import faiss
import json
import pickle
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import threading
from sentence_transformers import SentenceTransformer
import hashlib
import re


class FAISSMemoryStore:
    """FAISS-based memory management for agentic operations"""
    
    def __init__(self, 
                 model_name: str = "all-mpnet-base-v2", 
                 max_history: int = 500,
                 similarity_threshold: float = 0.7,
                 memory_dir: str = "faiss_memory"):
        """Initialize FAISS memory store"""
        
        self.model_name = model_name
        self.max_history = max_history
        self.similarity_threshold = similarity_threshold
        self.memory_dir = memory_dir
        self._lock = threading.Lock()
        
        # Create memory directory
        os.makedirs(memory_dir, exist_ok=True)
        
        # Initialize sentence transformer
        print(f"Loading sentence transformer model: {model_name}")
        self.sbert_model = SentenceTransformer(model_name)
        self.embedding_dim = self.sbert_model.get_sentence_embedding_dimension()
        
        # Initialize FAISS indexes
        self.extraction_index = faiss.IndexIDMap(faiss.IndexFlatIP(self.embedding_dim))
        self.calculation_index = faiss.IndexIDMap(faiss.IndexFlatIP(self.embedding_dim))
        
        # Memory storage
        self.extraction_history = []
        self.calculation_patterns = []
        self.confidence_calibration = {}
        self.next_id = 0
        
        # Load existing memory if available
        self._load_memory()
    
    def _get_embedding(self, text: str) -> np.ndarray:
        """Get normalized embedding for text"""
        embedding = self.sbert_model.encode([text], show_progress_bar=False, convert_to_numpy=True)
        # Normalize for cosine similarity with IndexFlatIP
        embedding = embedding / np.linalg.norm(embedding, axis=1, keepdims=True)
        return embedding.astype(np.float32)
    
    def _get_text_hash(self, text: str) -> str:
        """Generate hash for text"""
        return hashlib.md5(text.lower().encode()).hexdigest()
    
    def find_similar_claims(self, claim_text: str, limit: int = 5) -> List[Dict]:
        """Find similar historical claims using FAISS"""
        with self._lock:
            if self.extraction_index.ntotal == 0:
                return []
            
            # Get query embedding
            query_embedding = self._get_embedding(claim_text)
            
            # Search for similar claims
            k = min(limit * 2, self.extraction_index.ntotal)  # Get more candidates
            scores, indices = self.extraction_index.search(query_embedding, k)
            
            similar_claims = []
            for score, idx in zip(scores[0], indices[0]):
                if idx == -1:  # FAISS returns -1 for invalid indices
                    continue
                
                if score >= self.similarity_threshold:
                    historical_claim = self.extraction_history[idx]
                    similar_claims.append({
                        "historical_claim": historical_claim,
                        "similarity_score": float(score),
                        "success_indicators": historical_claim.get("success_metrics", {})
                    })
            
            # Sort by similarity and return top results
            similar_claims.sort(key=lambda x: x["similarity_score"], reverse=True)
            return similar_claims[:limit]
    
    def store_extraction_result(self, extraction_data: Dict):
        """Store successful extraction for learning"""
        with self._lock:
            # Prepare extraction record
            claim_text = extraction_data.get("source_text", "")
            if not claim_text:
                return
            
            success_metrics = self._calculate_success_metrics(extraction_data)
            
            extraction_record = {
                "id": self.next_id,
                "timestamp": datetime.now().isoformat(),
                "extraction_results": extraction_data,
                "success_metrics": success_metrics,
                "original_text": claim_text,
                "claim_id": extraction_data.get("claim_id", "")
            }
            
            # Get embedding
            text_embedding = self._get_embedding(claim_text)
            
            # Add to FAISS index
            self.extraction_index.add_with_ids(
                text_embedding, 
                np.array([self.next_id], dtype=np.int64)
            )
            
            # Store in memory
            self.extraction_history.append(extraction_record)
            self.next_id += 1
            
            # Cleanup if needed
            if len(self.extraction_history) > self.max_history:
                self._cleanup_extraction_history()
            
            # Save to disk
            self._save_memory()
    
    def find_similar_calculation_patterns(self, feature_context: Dict) -> List[Dict]:
        """Find similar calculation patterns using FAISS"""
        with self._lock:
            if self.calculation_index.ntotal == 0:
                return []
            
            # Create search text from feature context
            search_text = self._feature_context_to_text(feature_context)
            
            # Get query embedding
            query_embedding = self._get_embedding(search_text)
            
            # Search for similar patterns
            k = min(10, self.calculation_index.ntotal)
            scores, indices = self.calculation_index.search(query_embedding, k)
            
            similar_patterns = []
            for score, idx in zip(scores[0], indices[0]):
                if idx == -1 or score < 0.6:  # Lower threshold for patterns
                    continue
                
                pattern = self.calculation_patterns[idx]
                similar_patterns.append({
                    "feature_context": pattern["feature_context"],
                    "calculation_result": pattern["calculation_result"], 
                    "validation_result": pattern["validation_result"],
                    "accuracy_score": pattern.get("accuracy_score", 0.8),
                    "similarity_score": float(score)
                })
            
            # Sort by similarity
            similar_patterns.sort(key=lambda x: x["similarity_score"], reverse=True)
            return similar_patterns[:5]
    
    def store_calculation_pattern(self, feature_context: Dict, calculation_result: Dict, validation_result: Dict):
        """Store calculation pattern for learning"""
        with self._lock:
            # Create search text from feature context
            context_text = self._feature_context_to_text(feature_context)
            
            accuracy_score = validation_result.get("validation_score", 0.8)
            
            pattern_record = {
                "id": len(self.calculation_patterns),
                "timestamp": datetime.now().isoformat(),
                "feature_context": feature_context,
                "calculation_result": calculation_result,
                "validation_result": validation_result,
                "accuracy_score": accuracy_score,
                "context_text": context_text
            }
            
            # Get embedding
            context_embedding = self._get_embedding(context_text)
            
            # Add to FAISS index
            pattern_id = len(self.calculation_patterns)
            self.calculation_index.add_with_ids(
                context_embedding,
                np.array([pattern_id], dtype=np.int64)
            )
            
            # Store pattern
            self.calculation_patterns.append(pattern_record)
            
            # Cleanup if needed
            if len(self.calculation_patterns) > 200:
                self._cleanup_calculation_patterns()
            
            # Save to disk
            self._save_memory()
    
    def _feature_context_to_text(self, feature_context: Dict) -> str:
        """Convert feature context to searchable text"""
        text_parts = []
        
        # Extract damage severity info
        feature_analysis = feature_context.get("feature_analysis", {})
        damage_severity = feature_analysis.get("damage_severity", {})
        if damage_severity:
            severity_level = damage_severity.get("severity_level", "")
            damage_count = damage_severity.get("damage_count", 0)
            text_parts.append(f"damage severity {severity_level} count {damage_count}")
        
        # Extract operational impact
        operational_impact = feature_analysis.get("operational_impact", {})
        if operational_impact:
            impact_level = operational_impact.get("impact_level", "")
            text_parts.append(f"operational impact {impact_level}")
        
        # Extract contextual factors
        contextual_factors = feature_analysis.get("contextual_factors", {})
        if contextual_factors:
            is_primary = contextual_factors.get("is_primary_structure", False)
            if is_primary:
                text_parts.append("primary structure")
        
        # Expected loss range
        expected_range = feature_context.get("expected_loss_range", (0, 0))
        if expected_range and expected_range[1] > 0:
            text_parts.append(f"expected loss range {expected_range[0]} to {expected_range[1]}")
        
        return " ".join(text_parts) if text_parts else "unknown context"
    
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
            "completeness": min(1.0, total_indicators / 21.0),
            "total_indicators": total_indicators
        }
    
    def _cleanup_extraction_history(self):
        """Clean up old extraction records"""
        # Keep most recent records
        keep_count = int(self.max_history * 0.8)  # Keep 80% of max
        
        # Rebuild index with recent records
        self.extraction_index = faiss.IndexIDMap(faiss.IndexFlatIP(self.embedding_dim))
        new_history = self.extraction_history[-keep_count:]
        
        for i, record in enumerate(new_history):
            text_embedding = self._get_embedding(record["original_text"])
            self.extraction_index.add_with_ids(
                text_embedding, 
                np.array([i], dtype=np.int64)
            )
            record["id"] = i
        
        self.extraction_history = new_history
        self.next_id = len(new_history)
    
    def _cleanup_calculation_patterns(self):
        """Clean up old calculation patterns"""
        # Keep most recent patterns
        keep_count = 150
        
        # Rebuild index with recent patterns
        self.calculation_index = faiss.IndexIDMap(faiss.IndexFlatIP(self.embedding_dim))
        new_patterns = self.calculation_patterns[-keep_count:]
        
        for i, pattern in enumerate(new_patterns):
            context_embedding = self._get_embedding(pattern["context_text"])
            self.calculation_index.add_with_ids(
                context_embedding,
                np.array([i], dtype=np.int64)
            )
            pattern["id"] = i
        
        self.calculation_patterns = new_patterns
    
    def get_confidence_calibration(self, indicator_type: str) -> float:
        """Get confidence calibration factor for indicator type"""
        return self.confidence_calibration.get(indicator_type, 1.0)
    
    def update_confidence_calibration(self, indicator_type: str, predicted: float, actual: float):
        """Update confidence calibration based on actual results"""
        with self._lock:
            adjustment_factor = actual / max(predicted, 0.1)
            self.confidence_calibration[indicator_type] = adjustment_factor
            self._save_memory()
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory store statistics"""
        return {
            "extraction_history_count": len(self.extraction_history),
            "calculation_patterns_count": len(self.calculation_patterns),
            "confidence_calibration_count": len(self.confidence_calibration),
            "faiss_extraction_total": self.extraction_index.ntotal,
            "faiss_calculation_total": self.calculation_index.ntotal,
            "embedding_dimension": self.embedding_dim,
            "model_name": self.model_name,
            "similarity_threshold": self.similarity_threshold,
            "memory_dir_size_mb": self._get_directory_size() / (1024 * 1024)
        }
    
    def _get_directory_size(self) -> int:
        """Get total size of memory directory"""
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(self.memory_dir):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                if os.path.exists(filepath):
                    total_size += os.path.getsize(filepath)
        return total_size
    
    def _save_memory(self):
        """Save memory to disk"""
        try:
            # Save extraction history
            extraction_file = os.path.join(self.memory_dir, "extraction_history.pkl")
            with open(extraction_file, 'wb') as f:
                pickle.dump({
                    "history": self.extraction_history,
                    "next_id": self.next_id
                }, f)
            
            # Save calculation patterns
            patterns_file = os.path.join(self.memory_dir, "calculation_patterns.pkl")
            with open(patterns_file, 'wb') as f:
                pickle.dump(self.calculation_patterns, f)
            
            # Save confidence calibration
            calibration_file = os.path.join(self.memory_dir, "confidence_calibration.pkl")
            with open(calibration_file, 'wb') as f:
                pickle.dump(self.confidence_calibration, f)
            
            # Save FAISS indexes
            extraction_index_file = os.path.join(self.memory_dir, "extraction_index.faiss")
            faiss.write_index(self.extraction_index, extraction_index_file)
            
            calculation_index_file = os.path.join(self.memory_dir, "calculation_index.faiss")
            faiss.write_index(self.calculation_index, calculation_index_file)
            
        except Exception as e:
            print(f"Error saving memory: {e}")
    
    def _load_memory(self):
        """Load memory from disk"""
        try:
            # Load extraction history
            extraction_file = os.path.join(self.memory_dir, "extraction_history.pkl")
            if os.path.exists(extraction_file):
                with open(extraction_file, 'rb') as f:
                    data = pickle.load(f)
                    self.extraction_history = data.get("history", [])
                    self.next_id = data.get("next_id", 0)
            
            # Load calculation patterns
            patterns_file = os.path.join(self.memory_dir, "calculation_patterns.pkl")
            if os.path.exists(patterns_file):
                with open(patterns_file, 'rb') as f:
                    self.calculation_patterns = pickle.load(f)
            
            # Load confidence calibration
            calibration_file = os.path.join(self.memory_dir, "confidence_calibration.pkl")
            if os.path.exists(calibration_file):
                with open(calibration_file, 'rb') as f:
                    self.confidence_calibration = pickle.load(f)
            
            # Load FAISS indexes
            extraction_index_file = os.path.join(self.memory_dir, "extraction_index.faiss")
            if os.path.exists(extraction_index_file):
                self.extraction_index = faiss.read_index(extraction_index_file)
            
            calculation_index_file = os.path.join(self.memory_dir, "calculation_index.faiss")
            if os.path.exists(calculation_index_file):
                self.calculation_index = faiss.read_index(calculation_index_file)
            
            print(f"Loaded memory: {len(self.extraction_history)} extractions, {len(self.calculation_patterns)} patterns")
            
        except Exception as e:
            print(f"Error loading memory: {e}")
    
    def cleanup_memory(self):
        """Perform memory cleanup and optimization"""
        with self._lock:
            print("Performing memory cleanup...")
            
            # Cleanup old records
            if len(self.extraction_history) > self.max_history:
                self._cleanup_extraction_history()
            
            if len(self.calculation_patterns) > 200:
                self._cleanup_calculation_patterns()
            
            # Save optimized memory
            self._save_memory()
            
            print("Memory cleanup completed")
    
    def reset_memory(self):
        """Reset all memory (use with caution)"""
        with self._lock:
            # Clear in-memory data
            self.extraction_history = []
            self.calculation_patterns = []
            self.confidence_calibration = {}
            self.next_id = 0
            
            # Reset FAISS indexes
            self.extraction_index = faiss.IndexIDMap(faiss.IndexFlatIP(self.embedding_dim))
            self.calculation_index = faiss.IndexIDMap(faiss.IndexFlatIP(self.embedding_dim))
            
            # Remove saved files
            import shutil
            if os.path.exists(self.memory_dir):
                shutil.rmtree(self.memory_dir)
            os.makedirs(self.memory_dir, exist_ok=True)
            
            print("Memory has been reset")
    
    def export_memory(self, output_file: str):
        """Export memory to JSON file"""
        export_data = {
            "extraction_history": self.extraction_history,
            "calculation_patterns": self.calculation_patterns,
            "confidence_calibration": self.confidence_calibration,
            "stats": self.get_memory_stats(),
            "export_timestamp": datetime.now().isoformat()
        }
        
        with open(output_file, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        print(f"Memory exported to {output_file}")
    
    def close(self):
        """Close and save memory"""
        self._save_memory()
        print("FAISS memory store closed and saved")


# Backward compatibility
AgenticMemoryStore = FAISSMemoryStore