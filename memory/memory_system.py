"""
Memory System - Advanced memory management for SQL Agent System
Combines session and knowledge memory with intelligent pattern learning
"""

import asyncio
import json
import logging
import uuid
import aiosqlite
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
import os
import sqlite3

class MemorySystem:
    """
    Production memory system with advanced functionality:
    - Session management for conversation context
    - Knowledge storage for pattern learning
    - Similarity search for query reuse
    - High-performance in-memory operations
    - Optional persistence for data durability
    """
    
    def __init__(self, 
                 session_db_path: str = ":memory:",
                 knowledge_db_path: str = ":memory:",
                 similarity_threshold: float = 0.3,
                 enable_persistence: bool = False,
                 persistent_session_path: str = "data/session_memory.db",
                 persistent_knowledge_path: str = "data/knowledge_memory.db"):
        """
        Initialize memory system
        
        Args:
            session_db_path: SQLite path for session memory (":memory:" for in-memory)
            knowledge_db_path: SQLite path for persistent knowledge
            similarity_threshold: Threshold for text-based similarity matching
            enable_persistence: Enable data persistence to disk
            persistent_session_path: Path for persistent session storage
            persistent_knowledge_path: Path for persistent knowledge storage
        """
        self.session_db_path = session_db_path
        self.knowledge_db_path = knowledge_db_path
        self.similarity_threshold = similarity_threshold
        self.enable_persistence = enable_persistence
        self.persistent_session_path = persistent_session_path
        self.persistent_knowledge_path = persistent_knowledge_path
        
        self.logger = logging.getLogger(__name__)
        
        # Initialize databases (defer async initialization)
        self._initialized = False
        
        # Session context cache
        self.session_contexts = {}
        
        # In-memory backup for persistence
        self._session_backup = {}
        self._knowledge_backup = {}
    
    async def initialize(self):
        """Initialize the memory system asynchronously"""
        if self._initialized:
            return
            
        try:
            # Create knowledge directory if using file path
            if self.knowledge_db_path != ":memory:":
                Path(os.path.dirname(self.knowledge_db_path)).mkdir(parents=True, exist_ok=True)
            
            # Initialize databases
            await self._create_tables()
            self._initialized = True
            self.logger.info("Memory system initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to setup databases: {e}")
            raise
    
    async def _create_tables(self):
        """Create necessary tables in both databases"""
        
        # Session memory tables (conversation context) - use async for in-memory compatibility
        async with aiosqlite.connect(self.session_db_path) as session_db:
            await session_db.execute("PRAGMA foreign_keys = ON")
            if self.session_db_path == ":memory:":
                await session_db.execute("PRAGMA journal_mode = MEMORY")
                await session_db.execute("PRAGMA synchronous = OFF")
            else:
                await session_db.execute("PRAGMA journal_mode = WAL")
                await session_db.execute("PRAGMA synchronous = NORMAL")
            await session_db.execute("PRAGMA temp_store = MEMORY")
            await session_db.execute("PRAGMA cache_size = 10000")
            
            await session_db.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    last_activity TEXT NOT NULL,
                    context_data TEXT
                )
            """)
            
            await session_db.execute("""
                CREATE TABLE IF NOT EXISTS conversation_history (
                    id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    query TEXT NOT NULL,
                    intent TEXT,
                    tables_used TEXT,
                    chart_type TEXT,
                    success INTEGER DEFAULT 1,
                    timestamp TEXT NOT NULL,
                    FOREIGN KEY (session_id) REFERENCES sessions (id)
                )
            """)
            
            await session_db.commit()
        
        # Knowledge memory tables (learning and patterns) - use async for consistency
        async with aiosqlite.connect(self.knowledge_db_path) as knowledge_db:
            await knowledge_db.execute("PRAGMA foreign_keys = ON")
            if self.knowledge_db_path == ":memory:":
                await knowledge_db.execute("PRAGMA journal_mode = MEMORY")
                await knowledge_db.execute("PRAGMA synchronous = OFF")
            else:
                await knowledge_db.execute("PRAGMA journal_mode = WAL")
                await knowledge_db.execute("PRAGMA synchronous = NORMAL")
            await knowledge_db.execute("PRAGMA temp_store = MEMORY")
            await knowledge_db.execute("PRAGMA cache_size = 10000")
            
            await knowledge_db.execute("""
                CREATE TABLE IF NOT EXISTS successful_queries (
                    id TEXT PRIMARY KEY,
                    query_pattern TEXT NOT NULL,
                    sql_query TEXT NOT NULL,
                    tables_used TEXT,
                    chart_type TEXT,
                    execution_time REAL DEFAULT 0.0,
                    result_count INTEGER DEFAULT 0,
                    success_score REAL DEFAULT 1.0,
                    usage_count INTEGER DEFAULT 1,
                    created_at TEXT NOT NULL,
                    last_used TEXT NOT NULL,
                    metadata TEXT
                )
            """)
            
            await knowledge_db.execute("""
                CREATE TABLE IF NOT EXISTS query_patterns (
                    id TEXT PRIMARY KEY,
                    pattern_type TEXT NOT NULL,
                    pattern_description TEXT,
                    template_sql TEXT,
                    example_queries TEXT,
                    success_rate REAL DEFAULT 1.0,
                    usage_count INTEGER DEFAULT 1,
                    created_at TEXT NOT NULL,
                    last_used TEXT NOT NULL,
                    metadata TEXT
                )
            """)
            
            await knowledge_db.execute("""
                CREATE TABLE IF NOT EXISTS schema_insights (
                    id TEXT PRIMARY KEY,
                    table_name TEXT NOT NULL,
                    column_name TEXT,
                    insight_type TEXT NOT NULL,
                    insight_data TEXT,
                    confidence_score REAL DEFAULT 1.0,
                    created_at TEXT NOT NULL,
                    last_updated TEXT NOT NULL
                )
            """)
            
            # Create indexes for better performance
            await knowledge_db.execute("CREATE INDEX IF NOT EXISTS idx_query_pattern ON successful_queries(query_pattern)")
            await knowledge_db.execute("CREATE INDEX IF NOT EXISTS idx_last_used ON successful_queries(last_used)")
            await knowledge_db.execute("CREATE INDEX IF NOT EXISTS idx_pattern_type ON query_patterns(pattern_type)")
            await knowledge_db.execute("CREATE INDEX IF NOT EXISTS idx_table_name ON schema_insights(table_name)")
            
            await knowledge_db.commit()
    
    # =========================
    # Session Management
    # =========================
    
    async def create_session(self, user_id: str, session_id: str = None) -> str:
        """Create or update a user session"""
        if not session_id:
            session_id = str(uuid.uuid4())
        
        timestamp = datetime.now().isoformat()
        
        async with aiosqlite.connect(self.session_db_path) as db:
            await db.execute("""
                INSERT OR REPLACE INTO sessions (id, user_id, created_at, last_activity, context_data)
                VALUES (?, ?, ?, ?, ?)
            """, (session_id, user_id, timestamp, timestamp, "{}"))
            await db.commit()
        
        # Initialize session context cache
        self.session_contexts[session_id] = {
            "user_id": user_id,
            "created_at": timestamp,
            "conversation_history": []
        }
        
        return session_id
    
    async def get_session_context(self, session_id: str) -> Dict:
        """Get session context including conversation history"""
        
        # Check cache first
        if session_id in self.session_contexts:
            context = self.session_contexts[session_id].copy()
        else:
            context = {"conversation_history": []}
        
        # Get recent conversation history from database
        async with aiosqlite.connect(self.session_db_path) as db:
            cursor = await db.execute("""
                SELECT query, intent, tables_used, chart_type, success, timestamp
                FROM conversation_history 
                WHERE session_id = ? 
                ORDER BY timestamp DESC 
                LIMIT 10
            """, (session_id,))
            
            rows = await cursor.fetchall()
            
            history = []
            for row in rows:
                history.append({
                    "query": row[0],
                    "intent": json.loads(row[1]) if row[1] else {},
                    "tables_used": json.loads(row[2]) if row[2] else [],
                    "chart_type": row[3],
                    "success": bool(row[4]),
                    "timestamp": row[5]
                })
            
            context["conversation_history"] = list(reversed(history))  # Chronological order
        
        return context
    
    async def add_to_conversation(self, session_id: str, query: str, intent: Dict = None, 
                                tables_used: List[str] = None, chart_type: str = None, 
                                success: bool = True):
        """Add interaction to conversation history"""
        
        interaction_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()
        
        async with aiosqlite.connect(self.session_db_path) as db:
            await db.execute("""
                INSERT INTO conversation_history 
                (id, session_id, query, intent, tables_used, chart_type, success, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (interaction_id, session_id, query, 
                  json.dumps(intent) if intent else None,
                  json.dumps(tables_used) if tables_used else None,
                  chart_type, int(success), timestamp))
            
            # Update session last activity
            await db.execute("""
                UPDATE sessions SET last_activity = ? WHERE id = ?
            """, (timestamp, session_id))
            
            await db.commit()
        
        # Update cache
        if session_id in self.session_contexts:
            self.session_contexts[session_id]["conversation_history"].append({
                "query": query,
                "intent": intent or {},
                "tables_used": tables_used or [],
                "chart_type": chart_type,
                "success": success,
                "timestamp": timestamp
            })
            
            # Keep only recent history in cache
            if len(self.session_contexts[session_id]["conversation_history"]) > 10:
                self.session_contexts[session_id]["conversation_history"] = \
                    self.session_contexts[session_id]["conversation_history"][-10:]
    
    # =========================
    # Knowledge Management
    # =========================
    
    async def store_successful_query(self, query: str, sql: str, execution_time: float = 0.0,
                                   result_count: int = 0, tables_used: List[str] = None,
                                   chart_type: str = None, metadata: Dict = None):
        """Store successful query pattern for learning"""
        
        query_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()
        
        # Create query pattern for similarity matching
        query_pattern = self._extract_query_pattern(query)
        
        async with aiosqlite.connect(self.knowledge_db_path) as db:
            await db.execute("""
                INSERT INTO successful_queries 
                (id, query_pattern, sql_query, tables_used, chart_type, execution_time, 
                 result_count, created_at, last_used, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (query_id, query_pattern, sql, 
                  json.dumps(tables_used) if tables_used else None,
                  chart_type, execution_time, result_count, timestamp, timestamp,
                  json.dumps(metadata) if metadata else None))
            await db.commit()
        
        return query_id
    
    async def find_similar_queries(self, query: str, top_k: int = 5) -> List[Dict]:
        """Find similar queries using text-based similarity"""
        
        query_pattern = self._extract_query_pattern(query)
        query_words = set(query_pattern.lower().split())
        
        async with aiosqlite.connect(self.knowledge_db_path) as db:
            cursor = await db.execute("""
                SELECT query_pattern, sql_query, tables_used, chart_type, 
                       execution_time, result_count, success_score, usage_count, metadata
                FROM successful_queries 
                ORDER BY last_used DESC 
                LIMIT 50
            """)
            
            rows = await cursor.fetchall()
            
            similar_queries = []
            for row in rows:
                stored_pattern = row[0].lower()
                stored_words = set(stored_pattern.split())
                
                # Calculate Jaccard similarity
                intersection = len(query_words & stored_words)
                union = len(query_words | stored_words)
                similarity = intersection / union if union > 0 else 0
                
                if similarity >= self.similarity_threshold:
                    similar_queries.append({
                        "query_pattern": row[0],
                        "generated_sql": row[1],
                        "tables_used": json.loads(row[2]) if row[2] else [],
                        "chart_type": row[3],
                        "execution_time": row[4],
                        "result_count": row[5],
                        "success_score": row[6],
                        "usage_count": row[7],
                        "similarity_score": similarity,
                        "metadata": json.loads(row[8]) if row[8] else {}
                    })
            
            # Sort by similarity score and return top k
            similar_queries.sort(key=lambda x: x["similarity_score"], reverse=True)
            return similar_queries[:top_k]
    
    async def store_query_pattern(self, pattern_type: str, pattern_description: str,
                                example_queries: List[str], template_sql: str = None,
                                metadata: Dict = None):
        """Store reusable query pattern"""
        
        pattern_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()
        
        async with aiosqlite.connect(self.knowledge_db_path) as db:
            await db.execute("""
                INSERT INTO query_patterns 
                (id, pattern_type, pattern_description, template_sql, example_queries, 
                 created_at, last_used, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (pattern_id, pattern_type, pattern_description, template_sql,
                  json.dumps(example_queries), timestamp, timestamp,
                  json.dumps(metadata) if metadata else None))
            await db.commit()
        
        return pattern_id
    
    async def store_schema_insight(self, table_name: str, insight_type: str,
                                 insight_data: Dict, column_name: str = None,
                                 confidence_score: float = 1.0):
        """Store schema insights for future reference"""
        
        insight_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()
        
        async with aiosqlite.connect(self.knowledge_db_path) as db:
            await db.execute("""
                INSERT OR REPLACE INTO schema_insights 
                (id, table_name, column_name, insight_type, insight_data, 
                 confidence_score, created_at, last_updated)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (insight_id, table_name, column_name, insight_type,
                  json.dumps(insight_data), confidence_score, timestamp, timestamp))
            await db.commit()
        
        return insight_id
    
    # =========================
    # Memory Utilities
    # =========================
    
    def _extract_query_pattern(self, query: str) -> str:
        """Extract searchable pattern from natural language query"""
        
        # Convert to lowercase and remove common words
        common_words = {
            "show", "me", "get", "find", "tell", "what", "is", "are", "the", "a", "an",
            "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by"
        }
        
        words = query.lower().split()
        meaningful_words = [word for word in words if word not in common_words and len(word) > 2]
        
        return " ".join(meaningful_words)
    
    async def cleanup_old_data(self, days_old: int = 30):
        """Clean up old session data and low-usage patterns"""
        
        cutoff_date = (datetime.now() - timedelta(days=days_old)).isoformat()
        
        # Clean up old sessions
        async with aiosqlite.connect(self.session_db_path) as session_db:
            await session_db.execute("""
                DELETE FROM conversation_history 
                WHERE session_id IN (
                    SELECT id FROM sessions WHERE last_activity < ?
                )
            """, (cutoff_date,))
            
            await session_db.execute("""
                DELETE FROM sessions WHERE last_activity < ?
            """, (cutoff_date,))
            
            await session_db.commit()
        
        # Clean up low-performing patterns
        async with aiosqlite.connect(self.knowledge_db_path) as knowledge_db:
            await knowledge_db.execute("""
                DELETE FROM successful_queries 
                WHERE last_used < ? AND usage_count < 3 AND success_score < 0.5
            """, (cutoff_date,))
            
            await knowledge_db.commit()
        
        # Clear session cache
        self.session_contexts.clear()
    
    async def get_memory_stats(self) -> Dict:
        """Get memory system statistics"""
        
        stats = {}
        
        # Session stats
        try:
            async with aiosqlite.connect(self.session_db_path) as session_db:
                cursor = await session_db.execute("SELECT COUNT(*) FROM sessions")
                stats["active_sessions"] = (await cursor.fetchone())[0]
                
                cursor = await session_db.execute("SELECT COUNT(*) FROM conversation_history")
                stats["total_conversations"] = (await cursor.fetchone())[0]
        except:
            stats["active_sessions"] = 0
            stats["total_conversations"] = 0
        
        # Knowledge stats
        try:
            async with aiosqlite.connect(self.knowledge_db_path) as knowledge_db:
                cursor = await knowledge_db.execute("SELECT COUNT(*) FROM successful_queries")
                stats["successful_queries"] = (await cursor.fetchone())[0]
                
                cursor = await knowledge_db.execute("SELECT COUNT(*) FROM query_patterns")
                stats["stored_patterns"] = (await cursor.fetchone())[0]
                
                cursor = await knowledge_db.execute("SELECT COUNT(*) FROM schema_insights")
                stats["schema_insights"] = (await cursor.fetchone())[0]
                
                cursor = await knowledge_db.execute("""
                    SELECT AVG(success_score) FROM successful_queries
                """)
                avg_success = await cursor.fetchone()
                stats["average_success_rate"] = round(avg_success[0] or 0.0, 2)
        except:
            stats["successful_queries"] = 0
            stats["stored_patterns"] = 0
            stats["schema_insights"] = 0
            stats["average_success_rate"] = 0.0
        
        stats["similarity_threshold"] = self.similarity_threshold
        stats["session_cache_size"] = len(self.session_contexts)
        
        return stats