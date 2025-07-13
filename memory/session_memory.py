"""
Session Memory - SQLite-based conversation history and user preferences
Manages user session data and conversation context using SQLite
"""

import sqlite3
import asyncio
import aiosqlite
import json
from typing import Dict, List, Optional, Any
from datetime import datetime
import os
import logging
from pathlib import Path

class SessionMemory:
    """
    Manages session-level memory using SQLite for persistence.
    Stores conversation history, user preferences, and session context.
    """
    
    def __init__(self, db_path: str = ":memory:", enable_persistence: bool = False, persistent_path: str = "data/session_memory.db"):
        self.db_path = db_path
        self.enable_persistence = enable_persistence
        self.persistent_path = persistent_path
        
        # Security: Resolve and validate persistent path to prevent directory traversal
        if self.enable_persistence and self.persistent_path != ":memory:":
            self.persistent_path = str(Path(persistent_path).resolve())
            if not self.persistent_path.startswith(str(Path.cwd())):
                raise ValueError("Database path must be within current working directory")
            
            # Ensure directory exists with secure permissions
            db_dir = Path(os.path.dirname(self.persistent_path))
            db_dir.mkdir(parents=True, exist_ok=True, mode=0o750)
        
        self.logger = logging.getLogger(__name__)
        
        # Initialize database
        self._initialize_database()
        
        # Connection pool for thread safety
        self._connection_lock = asyncio.Lock()
    
    def _get_secure_connection(self):
        """Returns a secure SQLite connection with proper settings."""
        conn = sqlite3.connect(self.db_path, timeout=30.0, check_same_thread=False)
        # Security: Enable foreign key constraints and set secure pragmas
        conn.execute("PRAGMA foreign_keys = ON")
        
        # Configure for in-memory vs persistent storage
        if self.db_path == ":memory:":
            conn.execute("PRAGMA journal_mode = MEMORY")
            conn.execute("PRAGMA synchronous = OFF")
        else:
            conn.execute("PRAGMA journal_mode = WAL")
            conn.execute("PRAGMA synchronous = NORMAL")
        
        conn.execute("PRAGMA temp_store = MEMORY")
        conn.execute("PRAGMA cache_size = 10000")
        conn.execute("PRAGMA mmap_size = 268435456")  # 256MB
        return conn
    
    def _initialize_database(self):
        """Creates SQLite tables for session memory."""
        
        with sqlite3.connect(self.db_path, timeout=30.0, check_same_thread=False) as conn:
            # Security: Enable foreign key constraints and set secure pragmas
            conn.execute("PRAGMA foreign_keys = ON")
            
            # Configure for in-memory vs persistent storage
            if self.db_path == ":memory:":
                conn.execute("PRAGMA journal_mode = MEMORY")
                conn.execute("PRAGMA synchronous = OFF")
            else:
                conn.execute("PRAGMA journal_mode = WAL")
                conn.execute("PRAGMA synchronous = NORMAL")
            
            conn.execute("PRAGMA temp_store = MEMORY")
            conn.execute("PRAGMA mmap_size = 268435456")  # 256MB
            conn.execute("PRAGMA cache_size = 10000")
            cursor = conn.cursor()
            
            # User sessions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS user_sessions (
                    session_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    status TEXT DEFAULT 'active',
                    metadata TEXT
                )
            """)
            
            # Conversation history table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS conversation_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    query TEXT NOT NULL,
                    response TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    success BOOLEAN DEFAULT TRUE,
                    metadata TEXT,
                    FOREIGN KEY (session_id) REFERENCES user_sessions(session_id)
                )
            """)
            
            # User preferences table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS user_preferences (
                    user_id TEXT PRIMARY KEY,
                    preferences TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Processing results table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS processing_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    agent_name TEXT NOT NULL,
                    processing_data TEXT NOT NULL,
                    success BOOLEAN DEFAULT TRUE,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (session_id) REFERENCES user_sessions(session_id)
                )
            """)
            
            # Create indexes for performance
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_sessions_user_id ON user_sessions(user_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_history_session_id ON conversation_history(session_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_history_user_id ON conversation_history(user_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_results_session_id ON processing_results(session_id)")
            
            conn.commit()
            
        # Load from persistent storage if enabled
        if self.enable_persistence and os.path.exists(self.persistent_path):
            asyncio.run(self._load_from_persistent())
    
    async def _load_from_persistent(self):
        \"\"\"Load data from persistent storage to in-memory database\"\"\"
        if not self.enable_persistence or self.db_path != \":memory:\":
            return
            
        try:
            async with aiosqlite.connect(self.persistent_path) as source_db:
                async with aiosqlite.connect(self.db_path) as target_db:
                    # Copy data from persistent to in-memory
                    cursor = await source_db.execute(\"SELECT name FROM sqlite_master WHERE type='table'\")
                    tables = await cursor.fetchall()
                    
                    for (table_name,) in tables:
                        cursor = await source_db.execute(f\"SELECT * FROM {table_name}\")
                        rows = await cursor.fetchall()
                        
                        if rows:
                            cursor = await source_db.execute(f\"PRAGMA table_info({table_name})\")
                            columns = await cursor.fetchall()
                            column_names = [col[1] for col in columns]
                            placeholders = ','.join(['?' for _ in column_names])
                            
                            await target_db.executemany(
                                f\"INSERT OR REPLACE INTO {table_name} VALUES ({placeholders})\", 
                                rows
                            )
                    
                    await target_db.commit()
        except Exception as e:
            self.logger.warning(f\"Failed to load from persistent storage: {e}\")
    
    async def save_to_persistent(self):
        \"\"\"Save in-memory data to persistent storage\"\"\"
        if not self.enable_persistence or self.db_path != \":memory:\":
            return
            
        try:
            async with aiosqlite.connect(self.db_path) as source_db:
                async with aiosqlite.connect(self.persistent_path) as target_db:
                    # Ensure target has correct schema
                    await self._create_tables_async(target_db)
                    
                    # Copy data from in-memory to persistent
                    cursor = await source_db.execute(\"SELECT name FROM sqlite_master WHERE type='table'\")
                    tables = await cursor.fetchall()
                    
                    for (table_name,) in tables:
                        cursor = await source_db.execute(f\"SELECT * FROM {table_name}\")
                        rows = await cursor.fetchall()
                        
                        if rows:
                            cursor = await source_db.execute(f\"PRAGMA table_info({table_name})\")
                            columns = await cursor.fetchall()
                            column_names = [col[1] for col in columns]
                            placeholders = ','.join(['?' for _ in column_names])
                            
                            await target_db.executemany(
                                f\"INSERT OR REPLACE INTO {table_name} VALUES ({placeholders})\", 
                                rows
                            )
                    
                    await target_db.commit()
        except Exception as e:
            self.logger.error(f\"Failed to save to persistent storage: {e}\")
    
    async def _create_tables_async(self, db):
        \"\"\"Create tables using async connection\"\"\"
        await db.execute(\"PRAGMA foreign_keys = ON\")
        await db.execute(\"PRAGMA journal_mode = WAL\")
        await db.execute(\"PRAGMA synchronous = NORMAL\")
        await db.execute(\"PRAGMA temp_store = MEMORY\")
        
        # Create all tables
        await db.execute(\"\"\"
            CREATE TABLE IF NOT EXISTS user_sessions (
                session_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                status TEXT DEFAULT 'active',
                metadata TEXT
            )
        \"\"\")
        
        await db.execute(\"\"\"
            CREATE TABLE IF NOT EXISTS conversation_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                query TEXT NOT NULL,
                response TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                success BOOLEAN DEFAULT TRUE,
                metadata TEXT,
                FOREIGN KEY (session_id) REFERENCES user_sessions(session_id)
            )
        \"\"\")
        
        await db.execute(\"\"\"
            CREATE TABLE IF NOT EXISTS user_preferences (
                user_id TEXT PRIMARY KEY,
                preferences TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        \"\"\")
        
        await db.execute(\"\"\"
            CREATE TABLE IF NOT EXISTS processing_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                agent_name TEXT NOT NULL,
                processing_data TEXT NOT NULL,
                success BOOLEAN DEFAULT TRUE,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (session_id) REFERENCES user_sessions(session_id)
            )
        \"\"\")
        
        # Create indexes
        await db.execute(\"CREATE INDEX IF NOT EXISTS idx_sessions_user_id ON user_sessions(user_id)\")
        await db.execute(\"CREATE INDEX IF NOT EXISTS idx_history_session_id ON conversation_history(session_id)\")
        await db.execute(\"CREATE INDEX IF NOT EXISTS idx_history_user_id ON conversation_history(user_id)\")
        await db.execute(\"CREATE INDEX IF NOT EXISTS idx_results_session_id ON processing_results(session_id)\")
    
    async def get_or_create_session(self, user_id: str, session_id: str) -> Dict:
        """
        Retrieves existing session or creates new one.
        
        Args:
            user_id: User identifier
            session_id: Session identifier
            
        Returns:
            Session context data
        """
        
        # Security: Input validation
        if not user_id or not isinstance(user_id, str) or len(user_id) > 255:
            raise ValueError("Invalid user_id: must be a non-empty string under 256 characters")
        if not session_id or not isinstance(session_id, str) or len(session_id) > 255:
            raise ValueError("Invalid session_id: must be a non-empty string under 256 characters")
        
        async with self._connection_lock:
            with self._get_secure_connection() as conn:
                cursor = conn.cursor()
                
                # Check if session exists
                cursor.execute("""
                    SELECT session_id, metadata, created_at 
                    FROM user_sessions 
                    WHERE session_id = ? AND user_id = ?
                """, (session_id, user_id))
                
                result = cursor.fetchone()
                
                if result:
                    # Session exists, load context
                    metadata = json.loads(result[1]) if result[1] else {}
                    
                    # Get recent conversation history
                    cursor.execute("""
                        SELECT query, response, timestamp, success, metadata
                        FROM conversation_history 
                        WHERE session_id = ? 
                        ORDER BY timestamp DESC 
                        LIMIT 10
                    """, (session_id,))
                    
                    history = []
                    for row in cursor.fetchall():
                        history.append({
                            "query": row[0],
                            "response": row[1],
                            "timestamp": row[2],
                            "success": bool(row[3]),
                            "metadata": json.loads(row[4]) if row[4] else {}
                        })
                    
                    return {
                        "session_id": session_id,
                        "user_id": user_id,
                        "created_at": result[2],
                        "metadata": metadata,
                        "conversation_history": list(reversed(history)),
                        "context_loaded": True
                    }
                else:
                    # Create new session
                    cursor.execute("""
                        INSERT INTO user_sessions (session_id, user_id, metadata)
                        VALUES (?, ?, ?)
                    """, (session_id, user_id, json.dumps({"created": True})))
                    
                    conn.commit()
                    
                    return {
                        "session_id": session_id,
                        "user_id": user_id,
                        "created_at": datetime.now().isoformat(),
                        "metadata": {"created": True},
                        "conversation_history": [],
                        "context_loaded": True
                    }
    
    async def get_relevant_memories(self, user_id: str, query: str, context_type: str) -> Dict:
        """
        Retrieves relevant memories for query processing enhancement.
        
        Args:
            user_id: User identifier
            query: Current query
            context_type: Type of context needed
            
        Returns:
            Relevant memories and patterns
        """
        
        async with self._connection_lock:
            with self._get_secure_connection() as conn:
                cursor = conn.cursor()
                
                # Get similar past queries using simple text matching
                cursor.execute("""
                    SELECT query, response, timestamp, metadata
                    FROM conversation_history 
                    WHERE user_id = ? AND success = 1
                    ORDER BY timestamp DESC 
                    LIMIT 20
                """, (user_id,))
                
                past_queries = []
                for row in cursor.fetchall():
                    # Simple similarity check (can be enhanced with embeddings)
                    if self._is_query_similar(query, row[0]):
                        past_queries.append({
                            "query": row[0],
                            "response": row[1],
                            "timestamp": row[2],
                            "metadata": json.loads(row[3]) if row[3] else {}
                        })
                
                # Get user preferences
                cursor.execute("""
                    SELECT preferences FROM user_preferences WHERE user_id = ?
                """, (user_id,))
                
                preferences_row = cursor.fetchone()
                preferences = json.loads(preferences_row[0]) if preferences_row else {}
                
                return {
                    "similar_queries": past_queries[:5],  # Top 5 similar queries
                    "preferences": preferences,
                    "total_interactions": len(past_queries),
                    "context_type": context_type
                }
    
    async def add_processing_result(self, user_id: str, session_id: str, 
                                  agent_name: str, processing_data: Dict, success: bool = True):
        """
        Stores processing result from an agent.
        
        Args:
            user_id: User identifier
            session_id: Session identifier
            agent_name: Name of processing agent
            processing_data: Data to store
            success: Whether processing was successful
        """
        
        async with self._connection_lock:
            with self._get_secure_connection() as conn:
                cursor = conn.cursor()
                
                # Store processing result
                cursor.execute("""
                    INSERT INTO processing_results 
                    (session_id, agent_name, processing_data, success)
                    VALUES (?, ?, ?, ?)
                """, (session_id, agent_name, json.dumps(processing_data), success))
                
                # Update session timestamp
                cursor.execute("""
                    UPDATE user_sessions 
                    SET updated_at = CURRENT_TIMESTAMP 
                    WHERE session_id = ?
                """, (session_id,))
                
                conn.commit()
    
    async def finalize_session_interaction(self, session_id: str, session_summary: Dict):
        """
        Finalizes session with complete interaction summary.
        
        Args:
            session_id: Session to finalize
            session_summary: Complete session summary
        """
        
        async with self._connection_lock:
            with self._get_secure_connection() as conn:
                cursor = conn.cursor()
                
                # Add to conversation history
                if session_summary.get("original_query") and session_summary.get("final_results"):
                    cursor.execute("""
                        INSERT INTO conversation_history 
                        (session_id, user_id, query, response, success, metadata)
                        VALUES (?, ?, ?, ?, ?, ?)
                    """, (
                        session_id,
                        session_summary.get("user_id"),
                        session_summary.get("original_query"),
                        json.dumps(session_summary.get("final_results")),
                        session_summary.get("success", False),
                        json.dumps({
                            "processing_time": session_summary.get("processing_time"),
                            "user_feedback": session_summary.get("user_feedback")
                        })
                    ))
                
                # Update session status
                cursor.execute("""
                    UPDATE user_sessions 
                    SET status = 'completed', updated_at = CURRENT_TIMESTAMP,
                        metadata = json_set(COALESCE(metadata, '{}'), '$.completed', 'true')
                    WHERE session_id = ?
                """, (session_id,))
                
                conn.commit()
    
    async def get_user_insights(self, user_id: str) -> Dict:
        """
        Provides insights about user's session history and patterns.
        
        Args:
            user_id: User to analyze
            
        Returns:
            User insights and statistics
        """
        
        async with self._connection_lock:
            with self._get_secure_connection() as conn:
                cursor = conn.cursor()
                
                # Basic statistics
                cursor.execute("""
                    SELECT COUNT(*), COUNT(CASE WHEN success = 1 THEN 1 END)
                    FROM conversation_history 
                    WHERE user_id = ?
                """, (user_id,))
                
                total_queries, successful_queries = cursor.fetchone()
                
                # Recent activity
                cursor.execute("""
                    SELECT DATE(timestamp) as date, COUNT(*) as count
                    FROM conversation_history 
                    WHERE user_id = ? AND datetime(timestamp) > datetime('now', '-30 days')
                    GROUP BY DATE(timestamp)
                    ORDER BY date DESC
                """, (user_id,))
                
                recent_activity = [{"date": row[0], "count": row[1]} for row in cursor.fetchall()]
                
                # Most common query patterns (simple keyword analysis)
                cursor.execute("""
                    SELECT query FROM conversation_history 
                    WHERE user_id = ? AND success = 1
                    ORDER BY timestamp DESC 
                    LIMIT 50
                """, (user_id,))
                
                queries = [row[0] for row in cursor.fetchall()]
                common_patterns = self._extract_query_patterns(queries)
                
                return {
                    "total_queries": total_queries or 0,
                    "successful_queries": successful_queries or 0,
                    "success_rate": (successful_queries / total_queries) if total_queries > 0 else 0,
                    "recent_activity": recent_activity,
                    "common_patterns": common_patterns
                }
    
    def _is_query_similar(self, query1: str, query2: str) -> bool:
        """Simple similarity check for queries."""
        # Basic word overlap similarity
        words1 = set(query1.lower().split())
        words2 = set(query2.lower().split())
        
        if len(words1) == 0 or len(words2) == 0:
            return False
        
        overlap = len(words1.intersection(words2))
        return overlap / min(len(words1), len(words2)) > 0.3
    
    def _extract_query_patterns(self, queries: List[str]) -> Dict:
        """Extracts common patterns from user queries."""
        # Simple keyword frequency analysis
        word_freq = {}
        
        for query in queries:
            words = query.lower().split()
            for word in words:
                if len(word) > 3:  # Filter short words
                    word_freq[word] = word_freq.get(word, 0) + 1
        
        # Sort by frequency
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        
        return {
            "common_keywords": sorted_words[:10],
            "total_unique_words": len(word_freq),
            "query_count": len(queries)
        }