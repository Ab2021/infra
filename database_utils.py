"""
Database utilities for SQLite memory store management
"""

import sqlite3
import json
from datetime import datetime
from sqlite_memory_store import SQLiteMemoryStore


def print_database_stats(db_path: str = "agentic_memory.db"):
    """Print database statistics"""
    
    memory_store = SQLiteMemoryStore(db_path)
    stats = memory_store.get_memory_stats()
    
    print("=== SQLite Memory Store Statistics ===")
    print(f"Database file: {db_path}")
    print(f"Database size: {stats['db_size_mb']:.2f} MB")
    print()
    print("Record counts:")
    print(f"  Extraction history: {stats['extraction_history_count']}")
    print(f"  Calculation patterns: {stats['calculation_patterns_count']}")
    print(f"  Similarity index: {stats['similarity_index_count']}")
    print(f"  Confidence calibration: {stats['confidence_calibration_count']}")
    print()


def cleanup_database(db_path: str = "agentic_memory.db"):
    """Clean up old database records"""
    
    print("Cleaning up database...")
    memory_store = SQLiteMemoryStore(db_path)
    memory_store.cleanup_database()
    print("Database cleanup completed.")
    
    # Print updated stats
    print_database_stats(db_path)


def export_extraction_history(db_path: str = "agentic_memory.db", output_file: str = "extraction_history.json"):
    """Export extraction history to JSON file"""
    
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT claim_id, original_text, extraction_results, success_metrics, timestamp
            FROM extraction_history 
            ORDER BY created_at DESC
        """)
        
        records = []
        for row in cursor.fetchall():
            try:
                record = {
                    "claim_id": row["claim_id"],
                    "original_text": row["original_text"],
                    "extraction_results": json.loads(row["extraction_results"] or '{}'),
                    "success_metrics": json.loads(row["success_metrics"] or '{}'),
                    "timestamp": row["timestamp"]
                }
                records.append(record)
            except json.JSONDecodeError:
                continue
        
        with open(output_file, 'w') as f:
            json.dump(records, f, indent=2, default=str)
        
        print(f"Exported {len(records)} extraction records to {output_file}")


def view_recent_extractions(db_path: str = "agentic_memory.db", limit: int = 5):
    """View recent extraction results"""
    
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT claim_id, original_text, success_metrics, timestamp
            FROM extraction_history 
            ORDER BY created_at DESC 
            LIMIT ?
        """, (limit,))
        
        print(f"=== Recent {limit} Extractions ===")
        for i, row in enumerate(cursor.fetchall(), 1):
            try:
                success_metrics = json.loads(row["success_metrics"] or '{}')
                print(f"\n{i}. Claim ID: {row['claim_id']}")
                print(f"   Timestamp: {row['timestamp']}")
                print(f"   Text: {row['original_text'][:100]}...")
                print(f"   Confidence: {success_metrics.get('confidence_avg', 'N/A'):.3f}")
                print(f"   Completeness: {success_metrics.get('completeness', 'N/A'):.3f}")
            except json.JSONDecodeError:
                print(f"{i}. Invalid record data")


def analyze_pattern_performance(db_path: str = "agentic_memory.db"):
    """Analyze calculation pattern performance"""
    
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT feature_context, accuracy_score, timestamp
            FROM calculation_patterns 
            ORDER BY created_at DESC 
            LIMIT 20
        """)
        
        print("=== Calculation Pattern Performance ===")
        total_accuracy = 0
        count = 0
        
        for row in cursor.fetchall():
            try:
                feature_context = json.loads(row["feature_context"] or '{}')
                accuracy = row["accuracy_score"]
                
                damage_info = feature_context.get("feature_analysis", {}).get("damage_severity", {})
                severity = damage_info.get("severity_level", "unknown")
                
                print(f"Severity: {severity:10} | Accuracy: {accuracy:.3f} | Date: {row['timestamp'][:10]}")
                
                total_accuracy += accuracy
                count += 1
            except json.JSONDecodeError:
                continue
        
        if count > 0:
            avg_accuracy = total_accuracy / count
            print(f"\nAverage accuracy: {avg_accuracy:.3f} ({count} patterns)")
        else:
            print("\nNo calculation patterns found.")


def reset_database(db_path: str = "agentic_memory.db"):
    """Reset database (delete all records)"""
    
    confirm = input(f"Are you sure you want to reset database '{db_path}'? (yes/no): ")
    if confirm.lower() != 'yes':
        print("Reset cancelled.")
        return
    
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        
        # Clear all tables
        cursor.execute("DELETE FROM extraction_history")
        cursor.execute("DELETE FROM calculation_patterns")
        cursor.execute("DELETE FROM similarity_index")
        cursor.execute("DELETE FROM confidence_calibration")
        cursor.execute("DELETE FROM successful_patterns")
        
        # Reset auto-increment counters
        cursor.execute("DELETE FROM sqlite_sequence")
        
        conn.commit()
    
    print(f"Database '{db_path}' has been reset.")
    print_database_stats(db_path)


def main():
    """Main utility function"""
    
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python database_utils.py <command>")
        print("\nCommands:")
        print("  stats          - Show database statistics")
        print("  cleanup        - Clean up old records")
        print("  export         - Export extraction history to JSON")
        print("  recent [N]     - View N recent extractions (default: 5)")
        print("  patterns       - Analyze pattern performance")
        print("  reset          - Reset database (WARNING: deletes all data)")
        return
    
    command = sys.argv[1].lower()
    
    if command == "stats":
        print_database_stats()
    
    elif command == "cleanup":
        cleanup_database()
    
    elif command == "export":
        output_file = sys.argv[2] if len(sys.argv) > 2 else "extraction_history.json"
        export_extraction_history(output_file=output_file)
    
    elif command == "recent":
        limit = int(sys.argv[2]) if len(sys.argv) > 2 else 5
        view_recent_extractions(limit=limit)
    
    elif command == "patterns":
        analyze_pattern_performance()
    
    elif command == "reset":
        reset_database()
    
    else:
        print(f"Unknown command: {command}")


if __name__ == "__main__":
    main()